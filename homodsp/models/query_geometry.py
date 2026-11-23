"""Query protein encoder with explicit radial edge geometry.

Keep the legacy two bond + two spatial CGConv stages. Only width and
distance-conditioned spatial messages change; observed coordinates stay fixed.
"""
import torch
from torch import nn
from torch_cluster import radius_graph
from torch_geometric.nn import CGConv
from deeppbs.nn import MLP


class RadialQueryProteinEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, radius=4.0,
                 rbf_bins=16, max_neighbors=32):
        super().__init__()
        if radius <= 0 or rbf_bins < 2 or max_neighbors < 1:
            raise ValueError('Invalid query radial graph configuration')
        self.radius = float(radius)
        self.max_neighbors = int(max_neighbors)
        self.register_buffer('rbf_centers', torch.linspace(0, radius, rbf_bins))
        self.rbf_width = float(radius / (rbf_bins - 1))
        self.embed = MLP([in_channels, hidden_channels, hidden_channels])
        self.bond_conv1 = CGConv(hidden_channels, aggr='mean')
        self.bond_conv2 = CGConv(hidden_channels, aggr='mean')
        self.radi_conv1 = CGConv(hidden_channels, dim=rbf_bins, aggr='mean')
        self.radi_conv2 = CGConv(hidden_channels, dim=rbf_bins, aggr='mean')
        self.act = nn.ReLU()

    def forward(self, x_prot, v_prot, edge_index):
        h = self.embed(x_prot)
        h = self.act(self.bond_conv1(h, edge_index))
        h = self.act(self.bond_conv2(h, edge_index))
        edges = radius_graph(v_prot, r=self.radius,
                             max_num_neighbors=self.max_neighbors)
        distances = (v_prot[edges[0]] - v_prot[edges[1]]).norm(dim=-1)
        rbf = torch.exp(-((distances[:, None] - self.rbf_centers) / self.rbf_width) ** 2)
        h = self.act(self.radi_conv1(h, edges, rbf))
        return self.act(self.radi_conv2(h, edges, rbf))
