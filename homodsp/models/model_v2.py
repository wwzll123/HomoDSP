import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from deeppbs.nn import MLP

from torch_geometric.nn import CGConv, DynamicEdgeConv
from torch_cluster import radius, radius_graph
from deeppbs.nn import ProtEncoder, BiNet
from .template_geometry import (TemplateGeometryEncoder, DynamicTemplateGeometryEncoder,
                                ResidualSequenceCNN)
from deeppbs.template_schema import DNA_INDICES
from .query_geometry import RadialQueryProteinEncoder

class CNN(nn.Module):
    def __init__(self, dna_channels, hidden_size=8, condition="full"): 
        super(CNN, self).__init__()
        self.condition = condition
        self.conv1 = nn.Conv1d(dna_channels, hidden_size, kernel_size=3, padding='same') 
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size = 3, padding='same') 
        self.fc = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.act = nn.ReLU() 

    def forward(self, x):
        x = self.conv1(x.T)
        x = self.act(x)  
        x = self.conv2(x)  
        x = self.act(x)   
        x = self.fc(x) 
        
        return x.T  

class TemplateEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=8):
        super(TemplateEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.proj = MLP([in_channels, hidden_channels, hidden_channels], dropout=0.0)

    def forward(self, template_x, template_mask=None, template_scores=None, length=None, device=None):
        if device is None:
            device = template_x.device if template_x is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if template_x is None:
            return torch.zeros((length, self.hidden_channels), dtype=torch.float32, device=device)

        if template_x.dim() == 2:
            template_x = template_x.unsqueeze(0)
        template_x = template_x.to(device)
        k, l, _ = template_x.shape

        if template_mask is None:
            template_mask = torch.ones((k, l), dtype=torch.bool, device=device)
        else:
            template_mask = template_mask.to(device)
            if template_mask.dim() == 1:
                template_mask = template_mask.unsqueeze(0)

        if template_scores is None:
            template_scores = torch.zeros((k,), dtype=torch.float32, device=device)
        else:
            template_scores = template_scores.to(device).view(-1)

        h = self.proj(template_x.reshape(k * l, -1)).reshape(k, l, self.hidden_channels)
        score_weight = torch.softmax(template_scores[:k], dim=0).view(k, 1, 1)
        mask_weight = template_mask[:, :, None].float()
        weights = score_weight * mask_weight
        denom = weights.sum(dim=0).clamp_min(1e-6)
        out = (h * weights).sum(dim=0) / denom

        if length is not None and out.shape[0] != length:
            if out.shape[0] > length:
                out = out[:length]
            else:
                pad = torch.zeros((length - out.shape[0], self.hidden_channels), dtype=out.dtype, device=device)
                out = torch.cat((out, pad), dim=0)
        return out

class TemplateInterfaceEncoder(nn.Module):
    def __init__(self, node_channels, hidden_channels=8):
        super(TemplateInterfaceEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.edge_nn = MLP([node_channels, hidden_channels, hidden_channels], dropout=0.0)
        self.node_attn = MLP([hidden_channels, hidden_channels, 1], dropout=0.0)
        self.out_nn = MLP([hidden_channels * 3, hidden_channels, hidden_channels], dropout=0.0)

    def forward(self, template_node_x, template_node_mask=None, template_scores=None, length=None, device=None):
        if device is None:
            device = template_node_x.device if template_node_x is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if template_node_x is None:
            return torch.zeros((length, self.hidden_channels), dtype=torch.float32, device=device)

        template_node_x = template_node_x.to(device)
        if template_node_x.dim() == 3:
            template_node_x = template_node_x.unsqueeze(0)
        k, l, m, _ = template_node_x.shape

        if template_node_mask is None:
            template_node_mask = torch.ones((k, l, m), dtype=torch.bool, device=device)
        else:
            template_node_mask = template_node_mask.to(device)
            if template_node_mask.dim() == 2:
                template_node_mask = template_node_mask.unsqueeze(0)

        if template_scores is None:
            template_scores = torch.zeros((k,), dtype=torch.float32, device=device)
        else:
            template_scores = template_scores.to(device).view(-1)

        node_h = self.edge_nn(template_node_x.reshape(k * l * m, -1)).reshape(k, l, m, self.hidden_channels)
        node_w = template_node_mask[:, :, :, None].float()
        node_denom = node_w.sum(dim=2).clamp_min(1.0)
        mean_h = (node_h * node_w).sum(dim=2) / node_denom
        max_h = node_h.masked_fill(~template_node_mask[:, :, :, None], -1e4).max(dim=2).values
        max_h = torch.where(template_node_mask.any(dim=2, keepdim=True), max_h, torch.zeros_like(max_h))
        attn_logits = self.node_attn(node_h.reshape(k * l * m, -1)).reshape(k, l, m)
        attn_logits = attn_logits.masked_fill(~template_node_mask, -1e4)
        attn_w = torch.softmax(attn_logits, dim=2)[:, :, :, None] * node_w
        attn_denom = attn_w.sum(dim=2).clamp_min(1e-6)
        attn_h = (node_h * attn_w).sum(dim=2) / attn_denom
        per_template = self.out_nn(torch.cat((mean_h, max_h, attn_h), dim=-1).reshape(k * l, -1)).reshape(k, l, self.hidden_channels)

        score_weight = torch.softmax(template_scores[:k], dim=0).view(k, 1, 1)
        pos_mask = template_node_mask.any(dim=2)[:, :, None].float()
        weights = score_weight * pos_mask
        denom = weights.sum(dim=0).clamp_min(1e-6)
        out = (per_template * weights).sum(dim=0) / denom

        if length is not None and out.shape[0] != length:
            if out.shape[0] > length:
                out = out[:length]
            else:
                pad = torch.zeros((length - out.shape[0], self.hidden_channels), dtype=out.dtype, device=device)
                out = torch.cat((out, pad), dim=0)
        return out

class Model(nn.Module):

    def __init__(self, prot_channels, dna_channels, out_channels=4, condition="full", readout="all",**kwargs):
        super(Model, self).__init__()
        
        self.condition = condition
        assert self.condition in ["prot","prot_ag","prot_shape_ag","shape_ag","shape","prot_shape","ag"]

        if self.condition in ["prot_ag","shape_ag","prot_shape_ag","ag"]:
            self.fn_channels = 6
        else:
            self.fn_channels = 0
        
        self.name = "Model"
        self.hidden_size = int(kwargs.get("fusion_hidden", 8))
        self.fusion_depth = int(kwargs.get("fusion_depth", 0))
        self.dna_channels = dna_channels
        self.binet_reduce_channels = 32
        self.dna_embed_dim = int(kwargs.get("query_dna_hidden", 10))
        self.dropout = 0.0
        self.prot_embed_dim = int(kwargs.get("query_prot_hidden", 10))
        self.conv = None #for visualizing conv output
        self.use_templates = kwargs.get("use_templates", False)
        self.template_fusion = kwargs.get("template_fusion", "concat")
        if self.template_fusion not in {"concat", "add"}:
            raise ValueError("template_fusion must be concat or add")
        self.template_hidden = kwargs.get("template_hidden", 8)
        self.template_channels = kwargs.get("template_channels", dna_channels)
        self.template_node_channels = kwargs.get("template_node_channels", 92)
        self.template_encoder_type = kwargs.get("template_encoder_type", "feature")
        if self.template_encoder_type not in {"feature", "interface", "geometry", "dynamic_geometry"}:
            raise ValueError("Unknown template_encoder_type: " + str(self.template_encoder_type))
        if self.template_encoder_type in {"geometry", "dynamic_geometry"} and (not self.use_templates or condition == "shape"):
            raise ValueError("Geometry template encoder requires an active template and query interaction branch")
        self.use_uq_head = kwargs.get("use_uq_head", False)

        self.embed = MLP([11 + self.fn_channels, self.dna_embed_dim, self.dna_embed_dim],
                dropout=self.dropout)
        
        self.query_encoder_type = kwargs.get("query_encoder_type", "legacy")
        if self.query_encoder_type == "radial":
            self.prot_encoder = RadialQueryProteinEncoder(
                prot_channels, self.prot_embed_dim,
                radius=kwargs.get("query_graph_radius", 4.0),
                rbf_bins=kwargs.get("query_rbf_bins", 16),
                max_neighbors=kwargs.get("query_max_neighbors", 32))
        elif self.query_encoder_type == "legacy":
            self.prot_encoder = ProtEncoder(prot_channels, self.prot_embed_dim, condition=self.condition)
        else:
            raise ValueError("Unknown query encoder: " + self.query_encoder_type)

        self.binet = BiNet(self.prot_embed_dim, self.dna_embed_dim, condition=self.condition,
        conv="PPFConv", readout=readout) # +dna_channels
        
        self.reduce_nn = MLP([11*self.dna_embed_dim, 2*self.binet_reduce_channels,
            self.binet_reduce_channels], dropout=self.dropout)

        template_extra_channels = self.template_hidden if self.use_templates and self.template_fusion == "concat" else 0
        query_channels = self.binet_reduce_channels + (
            self.dna_channels if self.condition in ["prot_shape", "prot_shape_ag", "shape_ag"] else 0)
        if self.use_templates:
            if self.template_encoder_type == "dynamic_geometry":
                self.template_encoder = DynamicTemplateGeometryEncoder(
                    self.template_node_channels, query_channels, self.template_hidden,
                    layers=kwargs.get("template_layers", 3),
                    include_dna_identity=kwargs.get("template_include_dna_identity", True),
                    dropout=kwargs.get("template_dropout", 0.1),
                    radius=kwargs.get("template_graph_radius", 8.0),
                    neighbors=kwargs.get("template_graph_neighbors", 8),
                    rbf_bins=kwargs.get("template_rbf_bins", 8),
                    chunk_size=kwargs.get("template_graph_chunk_size", 64),
                    score_prior=kwargs.get("template_score_prior", 0.0),
                    coord_step=kwargs.get("template_coord_step", 0.25),
                    edge_hidden=kwargs.get("template_edge_hidden", 32))
            elif self.template_encoder_type == "geometry":
                self.template_encoder = TemplateGeometryEncoder(
                    self.template_node_channels, query_channels, self.template_hidden,
                    layers=kwargs.get("template_layers", 3),
                    include_dna_identity=kwargs.get("template_include_dna_identity", True),
                    dropout=kwargs.get("template_dropout", 0.1),
                    radius=kwargs.get("template_graph_radius", 8.0),
                    neighbors=kwargs.get("template_graph_neighbors", 8),
                    rbf_bins=kwargs.get("template_rbf_bins", 8),
                    chunk_size=kwargs.get("template_graph_chunk_size", 64),
                    score_prior=kwargs.get("template_score_prior", 0.0))
            elif self.template_encoder_type == "interface":
                self.template_encoder = TemplateInterfaceEncoder(self.template_node_channels, self.template_hidden)
            else:
                self.template_encoder = TemplateEncoder(self.template_channels, self.template_hidden)

        if self.use_templates and self.template_fusion == "add":
            # No bias: absent template embeddings must give exactly zero correction.
            self.template_projection = nn.Linear(self.template_hidden, query_channels, bias=False)

        if self.fusion_depth > 0:
            self.cnn = ResidualSequenceCNN(
                query_channels + template_extra_channels, self.hidden_size,
                depth=self.fusion_depth, dropout=kwargs.get("fusion_dropout", 0.1))
        elif self.condition in ["prot_ag", "prot", "ag"]:
            self.cnn = CNN(self.binet_reduce_channels + template_extra_channels, hidden_size=self.hidden_size,
                    condition=self.condition)
        elif self.condition in ["prot_shape","prot_shape_ag","shape_ag"]:
            self.cnn = CNN(self.binet_reduce_channels +  self.dna_channels + template_extra_channels,
                    hidden_size=self.hidden_size, condition=self.condition)
        #elif self.condition == "shape":
        self.shapecnn = CNN(self.dna_channels, hidden_size=self.hidden_size, condition=self.condition)
        
        self.mlp = MLP([self.hidden_size, self.hidden_size, self.hidden_size, 4],
                dropout=self.dropout)
        if self.use_uq_head:
            self.local_uq_head = MLP([self.hidden_size, self.hidden_size, 1], dropout=self.dropout)
            self.global_uq_feature_dim = 2 * self.hidden_size + 12
            self.global_uq_head = nn.Sequential(
                nn.Linear(self.global_uq_feature_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 2)
            )
       
        #self.groove_cnn = CNN(self.dna_embed_dim, hidden_size=4, condition=self.condition,
        #            kernel_size=7)
        #self.bbone_cnn = CNN(self.dna_embed_dim, hidden_size=4, condition=self.condition,
        #        kernel_size=4)


        #self.bbone_shape_cnn = CNN(self.dna_channels + 4, hidden_size=4, condition=self.condition,
        #        kernel_size=3, padding='same')


        #self.bbone_shape_cnn2 = CNN(4, hidden_size=4, condition=self.condition,
        #        kernel_size=3, padding='same')

        self.global_temp = nn.Parameter(torch.randn(1))

        self.outnn = MLP([8,4,4])

    def strandForward(self, e_prot, v_dna, x_dna, x_dna_point, 
            x_prot, v_prot, prot_vecs, dna_vecs, template_x=None, template_mask=None, template_scores=None,
            template_node_x=None, template_node_mask=None):
        e_prot = e_prot.T
        v_dna = v_dna.view(-1,3)
        dna_vecs = dna_vecs.view(-1,3)
        x_dna_copy = x_dna
        
        if self.condition != "shape":
            x_dna_point_embed = self.embed(x_dna_point)

            x_prot = self.prot_encoder(x_prot, v_prot, e_prot)
            x_binet, conv = self.binet(x_dna_point_embed, v_dna, x_prot, v_prot, prot_vecs,
                    dna_vecs, add_target_features=True, atom_to_mask=None) 
            
            #x_binet = x_binet.reshape(x_binet.shape[0], -1)
            
            x_binet = x_binet.reshape(x_binet.shape[0], -1)
            x_binet = self.reduce_nn(x_binet)

            #x_groove = x_binet[:,[2,3,4,5,6,7,8],:]
            #x_bbone = x_binet[:,[0,1,9,10],:]
            
            #x_groove = self.groove_cnn(x_groove)
            #x_binet = self.reduce_nn(x_binet)#.reshape(-1, 11)
            
            #x_bbone = self.bbone_cnn(x_bbone)
            if self.condition in ["shape_ag", "prot_shape","prot_shape_ag"]:
                #x_bbone = torch.hstack((x_bbone, x_dna))
                x_binet = torch.hstack((x_binet, x_dna)) 
                #x_binet = torch.hstack((x_binet, torch.zeros_like(x_dna)))

            if self.use_templates:
                if self.template_encoder_type in {"geometry", "dynamic_geometry"}:
                    template_embed = self.template_encoder(
                        template_node_x, query_features=x_binet,
                        template_node_mask=template_node_mask,
                        template_scores=template_scores)
                elif self.template_encoder_type == "interface":
                    if template_node_x is not None:
                        template_embed = self.template_encoder(
                            template_node_x,
                            template_node_mask=template_node_mask,
                            template_scores=template_scores,
                            length=x_binet.shape[0],
                            device=x_binet.device
                        )
                    else:
                        template_embed = torch.zeros(
                            (x_binet.shape[0], self.template_hidden),
                            dtype=x_binet.dtype,
                            device=x_binet.device
                        )
                else:
                    template_embed = self.template_encoder(
                        template_x,
                        template_mask=template_mask,
                        template_scores=template_scores,
                        length=x_binet.shape[0],
                        device=x_binet.device
                    )
                if self.template_fusion == "add":
                    x_binet = x_binet + self.template_projection(template_embed)
                else:
                    x_binet = torch.hstack((x_binet, template_embed))
            
            #x_bbone = x_bbone.unsqueeze(0)
            #x_bbone = self.bbone_shape_cnn(x_bbone).T
            #x_bbone = self.bbone_shape_cnn2(x_bbone.unsqueeze(0)).T
            
            
            #x_dnacnn = torch.hstack((x_groove, x_bbone))
            #x_dna = self.outnn(x_dnacnn)

            x_dnacnn = self.cnn(x_binet)
            x_dna = self.mlp(x_dnacnn)
        else:
            x_dnacnn = self.shapecnn(x_dna)
        
            x_dna = self.mlp(x_dnacnn)
        
        #try:
        #    return x_dna*rel_temp[:,None] #torch.cat((x_dna, torch.flip(x_dna, dims=[0,1])), dim=0)/torch.sigmoid(self.global_temp)
        #except:
        return x_dna, conv, x_dnacnn

    def forward(self, data, return_features=False):
        if self.template_encoder_type in {"geometry", "dynamic_geometry"}:
            schema = getattr(data, "template_interface92_schema", None)
            if schema is None or not bool(schema.all()):
                raise ValueError("Geometry encoder requires verified interface92 feature names in the NPZ")
        x_dna_point = data.x_dna_point[:,:(11 + self.fn_channels)]
        template_x = getattr(data, "template_x_dna", None)
        template_mask = getattr(data, "template_mask", None)
        template_scores = getattr(data, "template_scores", None)
        template_node_x = getattr(data, "template_node_x", None)
        template_node_mask = getattr(data, "template_node_mask", None)
        out1, conv, feat1 = self.strandForward(data.e_prot, data.v_dna, data.x_dna, x_dna_point, data.x_prot,
                data.v_prot, data.prot_vecs, data.dna_vecs, template_x, template_mask, template_scores,
                template_node_x, template_node_mask)
        
        self.conv = conv

        x_dna_point = x_dna_point.reshape(-1, 11, 11 + self.fn_channels)
        
        switch = torch.LongTensor([10,9,5,4,3,2,8,7,6,1,0]).to(data.v_dna.device)
        x_dna_point_rc = torch.index_select(x_dna_point, 1, switch)
        v_dna_rc = torch.index_select(data.v_dna, 1, switch)
        dna_vecs_rc = torch.index_select(data.dna_vecs, 1, switch)

        x_dna_point_rc[:,:,:11] = x_dna_point[:,:,:11]
        x_dna_point_rc = torch.flip(x_dna_point_rc, [0]).reshape(-1,11 + self.fn_channels)
        v_dna_rc = torch.flip(v_dna_rc, [0])
        dna_vecs_rc = torch.flip(dna_vecs_rc, [0])
        
        shape_transform = torch.LongTensor([-1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1,
            1]).to(data.v_dna.device)

        x_dna_rc = torch.flip(data.x_dna, [0])*shape_transform[None, :]
        
        tmp = x_dna_rc[0,6:12].clone().detach()
        x_dna_rc[:-1,6:12] = x_dna_rc[1:,6:12]
        x_dna_rc[-1,6:12] = tmp

        template_x_rc = None
        template_mask_rc = None
        if template_x is not None:
            template_x_rc = torch.flip(template_x, [1])
        if template_mask is not None:
            template_mask_rc = torch.flip(template_mask, [1])
        template_node_x_rc = None
        template_node_mask_rc = None
        if template_node_x is not None:
            template_node_x_rc = torch.flip(template_node_x, [1])
            if self.template_encoder_type in {"geometry", "dynamic_geometry"}:
                # Preserve frame geometry; explicit base identity follows strand orientation.
                template_node_x_rc = template_node_x_rc.clone()
                template_node_x_rc[..., DNA_INDICES] = template_node_x_rc[..., [DNA_INDICES[i] for i in [3, 2, 1, 0, 4]]]
        if template_node_mask is not None:
            template_node_mask_rc = torch.flip(template_node_mask, [1])

        out2, _, feat2  = self.strandForward(data.e_prot, v_dna_rc, x_dna_rc, x_dna_point_rc, data.x_prot,
                data.v_prot, data.prot_vecs, dna_vecs_rc, template_x_rc, template_mask_rc, template_scores,
                template_node_x_rc, template_node_mask_rc)

        #out = (out1 + out2.flip([0,1]))/2
        logits = torch.cat((out1, out2), dim=0) / torch.sigmoid(self.global_temp)
        if return_features:
            return logits, torch.cat((feat1, feat2), dim=0)
        if not self.use_uq_head:
            return logits

        local_log_var = torch.cat(
            (self.local_uq_head(feat1), self.local_uq_head(feat2)),
            dim=0
        ).squeeze(-1)
        pooled = torch.cat((feat1.mean(dim=0), feat2.mean(dim=0)), dim=0)
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1) / np.log(4.0)
        max_prob = probs.max(dim=1).values
        local_mae = torch.exp(torch.clamp(local_log_var, min=-8.0, max=8.0))

        def _summary(x):
            return torch.stack((
                x.mean(),
                x.std(unbiased=False),
                x.min(),
                x.max()
            ))

        global_features = torch.cat((
            pooled,
            _summary(local_mae),
            _summary(entropy),
            _summary(max_prob)
        ), dim=0)
        global_quality = self.global_uq_head(global_features.unsqueeze(0)).squeeze(0)
        return {
            "logits": logits,
            "local_log_var": local_log_var,
            "global_quality": global_quality
        }
        
        #return torch.cat((out, out.flip([0,1])),
        #        dim=0)/torch.sigmoid(self.global_temp)
