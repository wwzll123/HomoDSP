import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from deeppbs.nn import MLP

from torch_geometric.nn import CGConv, DynamicEdgeConv
from torch_cluster import radius, radius_graph
from deeppbs.nn import ProtEncoder, BiNet

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
        self.hidden_size = 8
        self.dna_channels = dna_channels
        self.binet_reduce_channels = 32
        self.dna_embed_dim = 10
        self.dropout = 0.0
        self.prot_embed_dim = 10
        self.conv = None #for visualizing conv output
        self.use_templates = kwargs.get("use_templates", False)
        self.template_hidden = kwargs.get("template_hidden", 8)
        self.template_channels = kwargs.get("template_channels", dna_channels)
        self.template_node_channels = kwargs.get("template_node_channels", 92)
        self.template_encoder_type = kwargs.get("template_encoder_type", "feature")
        self.use_uq_head = kwargs.get("use_uq_head", False)
        self.uq_global_output_dim = kwargs.get("uq_global_output_dim", 2)
        self.uq_global_hidden = kwargs.get("uq_global_hidden", self.hidden_size)
        self.uq_global_depth = kwargs.get("uq_global_depth", 1)
        self.uq_global_dropout = kwargs.get("uq_global_dropout", 0.0)

        self.embed = MLP([11 + self.fn_channels, self.dna_embed_dim, self.dna_embed_dim],
                dropout=self.dropout)
        
        self.prot_encoder = ProtEncoder(prot_channels, self.prot_embed_dim, condition=self.condition)

        self.binet = BiNet(self.prot_embed_dim, self.dna_embed_dim, condition=self.condition,
        conv="PPFConv", readout=readout) # +dna_channels
        
        self.reduce_nn = MLP([11*self.dna_embed_dim, 2*self.binet_reduce_channels,
            self.binet_reduce_channels], dropout=self.dropout)

        template_extra_channels = self.template_hidden if self.use_templates else 0
        if self.use_templates:
            if self.template_encoder_type == "interface":
                self.template_encoder = TemplateInterfaceEncoder(self.template_node_channels, self.template_hidden)
            else:
                self.template_encoder = TemplateEncoder(self.template_channels, self.template_hidden)

        if self.condition in ["prot_ag", "prot", "ag"]:
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
            global_layers = []
            in_dim = self.global_uq_feature_dim
            for _ in range(self.uq_global_depth):
                global_layers.append(nn.Linear(in_dim, self.uq_global_hidden))
                global_layers.append(nn.ReLU())
                if self.uq_global_dropout > 0:
                    global_layers.append(nn.Dropout(self.uq_global_dropout))
                in_dim = self.uq_global_hidden
            global_layers.append(nn.Linear(in_dim, self.uq_global_output_dim))
            self.global_uq_head = nn.Sequential(*global_layers)
       
        #self.groove_cnn = CNN(self.dna_embed_dim, hidden_size=4, condition=self.condition,
        #            kernel_size=7)
        #self.bbone_cnn = CNN(self.dna_embed_dim, hidden_size=4, condition=self.condition,
        #        kernel_size=4)


        #self.bbone_shape_cnn = CNN(self.dna_channels + 4, hidden_size=4, condition=self.condition,
        #        kernel_size=3, padding='same')


        #self.bbone_shape_cnn2 = CNN(4, hidden_size=4, condition=self.condition,
        #        kernel_size=3, padding='same')

        self.global_temp = nn.Parameter(torch.randn(1))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
                if self.template_encoder_type == "interface":
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

    def forward(self, data):
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
        
        switch = torch.LongTensor([10,9,5,4,3,2,8,7,6,1,0]).to(self.device)
        x_dna_point_rc = torch.index_select(x_dna_point, 1, switch)
        v_dna_rc = torch.index_select(data.v_dna, 1, switch)
        dna_vecs_rc = torch.index_select(data.dna_vecs, 1, switch)

        x_dna_point_rc[:,:,:11] = x_dna_point[:,:,:11]
        x_dna_point_rc = torch.flip(x_dna_point_rc, [0]).reshape(-1,11 + self.fn_channels)
        v_dna_rc = torch.flip(v_dna_rc, [0])
        dna_vecs_rc = torch.flip(dna_vecs_rc, [0])
        
        shape_transform = torch.LongTensor([-1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1,
            1]).to(self.device)

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
        if template_node_mask is not None:
            template_node_mask_rc = torch.flip(template_node_mask, [1])

        out2, _, feat2  = self.strandForward(data.e_prot, v_dna_rc, x_dna_rc, x_dna_point_rc, data.x_prot,
                data.v_prot, data.prot_vecs, dna_vecs_rc, template_x_rc, template_mask_rc, template_scores,
                template_node_x_rc, template_node_mask_rc)

        #out = (out1 + out2.flip([0,1]))/2
        logits = torch.cat((out1, out2), dim=0) / torch.sigmoid(self.global_temp)
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
