"""Geometric interface message passing with query-conditioned template fusion.

Each local graph belongs to one template and one query DNA position. Its nodes
are residue--DNA contact descriptors. Positions are the closest residue atom
in the archived DNA-local frame, NOT residue CA coordinates. Graphs never mix
different DNA-local coordinate frames. The dynamic encoder updates latent coordinates within each graph.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from deeppbs.template_schema import (
    DNA_INDICES, FEATURE_INDEX, POSITION_INDICES, SIDECHAIN_INDICES,
)


ARCH_CONFIG_KEYS = (
    'template_layers', 'template_dropout', 'template_include_dna_identity',
    'template_graph_radius', 'template_graph_neighbors', 'template_rbf_bins',
    'template_graph_chunk_size', 'template_score_prior', 'template_coord_step',
    'template_edge_hidden', 'fusion_hidden',
    'fusion_depth', 'fusion_dropout', 'template_fusion',
    'query_encoder_type', 'query_prot_hidden', 'query_dna_hidden',
    'query_graph_radius', 'query_rbf_bins', 'query_max_neighbors',
)


def architecture_kwargs(config):
    """One source of configuration forwarding for train, eval and prediction."""
    return {key: config[key] for key in ARCH_CONFIG_KEYS if key in config}


def masked_softmax(logits, mask, dim):
    # Finite zero output when a graph or DNA position has no valid nodes/templates.
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked, dim=dim) * mask.to(logits.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def mlp(in_channels, hidden_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.SiLU(),
                         nn.Linear(hidden_channels, out_channels))


class GeometryMessageBlock(nn.Module):
    def __init__(self, hidden, edge_channels, dropout):
        super().__init__()
        self.edge_message = mlp(2 * hidden + edge_channels, hidden, hidden)
        self.node_update = mlp(3 * hidden, hidden, hidden)
        self.node_norm = nn.LayerNorm(hidden)
        self.pool_key = nn.Linear(hidden, hidden, bias=False)
        self.pool_query = nn.Linear(hidden, hidden, bias=False)
        self.hub_update = mlp(2 * hidden, hidden, hidden)
        self.hub_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, hub, mask, edges, geometry):
        batch, count, hidden = nodes.shape
        receiver, sender = edges
        flat = nodes.reshape(-1, hidden)
        message = self.edge_message(torch.cat((flat[receiver], flat[sender], geometry), -1))
        aggregate = torch.zeros_like(flat).index_add(0, receiver, message)
        degree = flat.new_zeros((flat.shape[0], 1)).index_add(
            0, receiver, flat.new_ones((receiver.numel(), 1)))
        aggregate = (aggregate / degree.clamp_min(1)).reshape(batch, count, hidden)
        update = self.node_update(torch.cat((nodes, aggregate, hub[:, None].expand_as(nodes)), -1))
        nodes = self.node_norm(nodes + self.dropout(update))
        nodes = nodes.masked_fill(~mask[..., None], 0)
        score = (self.pool_key(nodes) * self.pool_query(hub)[:, None]).sum(-1) / math.sqrt(hidden)
        weights = masked_softmax(score, mask, dim=1)
        pooled = (nodes * weights[..., None]).sum(1)
        hub = self.hub_norm(hub + self.dropout(self.hub_update(torch.cat((hub, pooled), -1))))
        hub = hub.masked_fill(~mask.any(1)[:, None], 0)
        return nodes, hub


class TemplateGeometryEncoder(nn.Module):
    def __init__(self, node_channels, query_channels, hidden_channels=32, layers=3,
                 include_dna_identity=True, dropout=0.1, radius=8.0, neighbors=8,
                 rbf_bins=8, chunk_size=64, score_prior=0.0):
        super().__init__()
        if node_channels != 92:
            raise ValueError('Geometry encoder requires the named interface92 feature schema')
        if not isinstance(include_dna_identity, bool):
            raise ValueError('template_include_dna_identity must be a JSON boolean')
        if min(hidden_channels, layers, neighbors, rbf_bins, chunk_size) < 1 or radius <= 0:
            raise ValueError('Geometry encoder dimensions, radius and depth must be positive')
        self.hidden_channels = hidden_channels
        self.include_dna_identity = include_dna_identity
        self.radius = float(radius)
        self.neighbors = int(neighbors)
        self.chunk_size = int(chunk_size)
        self.score_prior = float(score_prior)
        self.register_buffer('rbf_centers', torch.linspace(0, radius, rbf_bins))
        self.rbf_width = radius / max(1, rbf_bins - 1)
        kept = [i for i in range(node_channels) if include_dna_identity or i not in DNA_INDICES]
        # Different input widths also make incompatible DNA-on/off checkpoints fail strict reload.
        self.register_buffer('input_indices', torch.tensor(kept, dtype=torch.long))
        self.node_input = nn.Sequential(nn.Linear(len(kept), hidden_channels), nn.LayerNorm(hidden_channels), nn.SiLU())
        self.query_input = nn.Sequential(nn.Linear(query_channels, hidden_channels), nn.LayerNorm(hidden_channels), nn.SiLU())
        self.blocks = nn.ModuleList([
            GeometryMessageBlock(hidden_channels, rbf_bins + 8, dropout) for _ in range(layers)
        ])
        self.template_compatibility = mlp(3 * hidden_channels, hidden_channels, 1)
        self.output = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.LayerNorm(hidden_channels))

    def graph(self, features, mask):
        """Directed capped-radius graph; distances/directions never cross local frames."""
        batch, count, _ = features.shape
        pos = features[..., POSITION_INDICES] * 10.0  # archived units -> Angstrom
        # No learned geometry/coordinates: edge geometry is fixed preprocessing input.
        delta = pos[:, None, :, :] - pos[:, :, None, :]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        valid = mask[:, :, None] & mask[:, None, :]
        valid &= ~torch.eye(count, dtype=torch.bool, device=mask.device)[None]
        valid &= distance <= self.radius
        nearest_distance, nearest = distance.masked_fill(~valid, float('inf')).topk(
            min(self.neighbors, count), dim=-1, largest=False)
        graph_id, target, slot = torch.where(torch.isfinite(nearest_distance))
        source = nearest[graph_id, target, slot]
        d = distance[graph_id, target, source]
        unit = F.normalize(delta[graph_id, target, source], dim=-1, eps=1e-8)
        radial = F.normalize(pos, dim=-1, eps=1e-8)
        side = features[..., SIDECHAIN_INDICES]
        side = F.normalize(side, dim=-1, eps=1e-8)
        has_side = features[..., FEATURE_INDEX['has_sidechain_vector']]
        ri, rj = radial[graph_id, target], radial[graph_id, source]
        si, sj = side[graph_id, target], side[graph_id, source]
        rbf = torch.exp(-((d[:, None] - self.rbf_centers) / self.rbf_width).square())
        dot = lambda a, b: (a * b).sum(-1)
        angles = torch.stack((dot(ri, rj), dot(si, sj), dot(unit, si), dot(unit, sj),
                              dot(unit, ri), dot(unit, rj), has_side[graph_id, target],
                              has_side[graph_id, source]), dim=-1)
        return (graph_id * count + target, graph_id * count + source), torch.cat((rbf, angles), -1)

    def encode_chunk(self, features, mask, query):
        # Ignore arbitrary values in padding, including non-finite padding values.
        features = features.masked_fill(~mask[..., None], 0)
        nodes = self.node_input(features.index_select(-1, self.input_indices))
        nodes = nodes.masked_fill(~mask[..., None], 0)
        hub = nodes.sum(1) / mask.sum(1, keepdim=True).clamp_min(1) + query
        hub = hub.masked_fill(~mask.any(1)[:, None], 0)
        edges, geometry = self.graph(features, mask)
        for block in self.blocks:
            nodes, hub = block(nodes, hub, mask, edges, geometry)
        return hub

    def forward(self, template_node_x, query_features, template_node_mask=None,
                template_scores=None, return_weights=False):
        length = query_features.shape[0]
        if template_node_x is None:
            zeros = query_features.new_zeros((length, self.hidden_channels))
            return (zeros, zeros.new_zeros((0, length))) if return_weights else zeros
        x = template_node_x.to(query_features.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4 or x.shape[1] != length or x.shape[-1] != 92:
            raise ValueError('Expected interface features [K, query_length, M, 92]; no silent resize')
        k, length, count, _ = x.shape
        if template_node_mask is None:
            raise ValueError('Geometry encoder requires an explicit node mask')
        mask = template_node_mask.to(device=x.device, dtype=torch.bool)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape != x.shape[:-1]:
            raise ValueError('Template node mask shape does not match node features')
        if k == 0 or count == 0:
            zeros = query_features.new_zeros((length, self.hidden_channels))
            return (zeros, zeros.new_zeros((k, length))) if return_weights else zeros
        query = self.query_input(query_features)
        q = query[None].expand(k, -1, -1).reshape(k * length, -1)
        flat_x, flat_mask = x.reshape(k * length, count, -1), mask.reshape(k * length, count)
        chunks = [self.encode_chunk(flat_x[i:i+self.chunk_size], flat_mask[i:i+self.chunk_size], q[i:i+self.chunk_size])
                  for i in range(0, k * length, self.chunk_size)]
        encoded = torch.cat(chunks, dim=0).reshape(k, length, self.hidden_channels)
        q = query[None].expand_as(encoded)
        score = self.template_compatibility(torch.cat((encoded, q, encoded * q), -1)).squeeze(-1)
        if self.score_prior:
            if template_scores is None or template_scores.numel() != k:
                raise ValueError('Retrieval score prior requires one score per template')
            score = score + self.score_prior * template_scores.to(score).reshape(k, 1)
        valid = mask.any(-1)
        weight = masked_softmax(score, valid, dim=0)
        out = self.output((encoded * weight[..., None]).sum(0))
        out = out.masked_fill(~valid.any(0)[:, None], 0)
        return (out, weight) if return_weights else out


class DynamicGeometryMessageBlock(nn.Module):
    """Edge-state and coordinate updates on a fixed local interface graph.

    Graph connectivity is fixed from the observed structure. Coordinates are
    refined by bounded, relative-vector updates; edge states are recurrently
    updated from endpoint embeddings and the current geometric descriptors.
    """
    def __init__(self, hidden, edge_hidden, geom_channels, dropout, coord_step):
        super().__init__()
        self.edge_update = mlp(2 * hidden + edge_hidden + geom_channels, edge_hidden, edge_hidden)
        self.edge_norm = nn.LayerNorm(edge_hidden)
        self.message = mlp(hidden + edge_hidden, hidden, hidden)
        self.node_update = mlp(3 * hidden, hidden, hidden)
        self.node_norm = nn.LayerNorm(hidden)
        self.pool_key = nn.Linear(hidden, hidden, bias=False)
        self.pool_query = nn.Linear(hidden, hidden, bias=False)
        self.hub_update = mlp(2 * hidden, hidden, hidden)
        self.hub_norm = nn.LayerNorm(hidden)
        self.coord_coeff = mlp(edge_hidden, edge_hidden, 1)
        # Start as the fixed-coordinate model; training learns bounded updates.
        nn.init.zeros_(self.coord_coeff[-1].weight)
        nn.init.zeros_(self.coord_coeff[-1].bias)
        self.coord_step = float(coord_step)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, hub, pos, mask, edges, edge_state, geom_fn, side, has_side):
        batch, count, hidden = nodes.shape
        receiver, sender = edges
        flat_nodes = nodes.reshape(-1, hidden)
        flat_pos = pos.reshape(-1, 3)
        # Update positions from equivariant relative vectors before the edge
        # and node updates, so every block's coordinate head affects its output.
        coeff = torch.tanh(self.coord_coeff(edge_state)) * self.coord_step
        degree = flat_nodes.new_zeros((flat_nodes.shape[0], 1)).index_add(
            0, receiver, flat_nodes.new_ones((receiver.numel(), 1)))
        displacement = torch.zeros_like(flat_pos).index_add(
            0, receiver, (flat_pos[sender] - flat_pos[receiver]) * coeff)
        displacement = displacement / degree.clamp_min(1)
        flat_pos = flat_pos + displacement
        pos = flat_pos.reshape(batch, count, 3)
        pos = pos.masked_fill(~mask[..., None], 0)
        geom = geom_fn(flat_pos, receiver, sender, side, has_side)
        updated_edge = self.edge_norm(edge_state + self.dropout(self.edge_update(torch.cat((
            flat_nodes[receiver], flat_nodes[sender], edge_state, geom), dim=-1))))
        msg = self.message(torch.cat((flat_nodes[sender], updated_edge), dim=-1))
        aggregate = torch.zeros_like(flat_nodes).index_add(0, receiver, msg)
        aggregate = (aggregate / degree.clamp_min(1)).reshape(batch, count, hidden)
        update = self.node_update(torch.cat((nodes, aggregate, hub[:, None].expand_as(nodes)), -1))
        nodes = self.node_norm(nodes + self.dropout(update))
        nodes = nodes.masked_fill(~mask[..., None], 0)

        score = (self.pool_key(nodes) * self.pool_query(hub)[:, None]).sum(-1) / math.sqrt(hidden)
        weights = masked_softmax(score, mask, dim=1)
        pooled = (nodes * weights[..., None]).sum(1)
        hub = self.hub_norm(hub + self.dropout(self.hub_update(torch.cat((hub, pooled), -1))))
        hub = hub.masked_fill(~mask.any(1)[:, None], 0)
        return nodes, hub, pos, updated_edge


class DynamicTemplateGeometryEncoder(TemplateGeometryEncoder):
    """Three-way interface encoder with simultaneous coordinate and edge updates."""
    def __init__(self, node_channels, query_channels, hidden_channels=32, layers=3,
                 include_dna_identity=True, dropout=0.1, radius=8.0, neighbors=8,
                 rbf_bins=8, chunk_size=64, score_prior=0.0,
                 coord_step=0.25, edge_hidden=32):
        super().__init__(node_channels, query_channels, hidden_channels, layers,
                         include_dna_identity, dropout, radius, neighbors,
                         rbf_bins, chunk_size, score_prior)
        if coord_step <= 0 or edge_hidden < 1:
            raise ValueError('Coordinate step and edge hidden width must be positive')
        geom_channels = rbf_bins + 8
        self.edge_hidden = int(edge_hidden)
        self.coord_step = float(coord_step)
        self.blocks = nn.ModuleList([
            DynamicGeometryMessageBlock(hidden_channels, self.edge_hidden,
                                        geom_channels, dropout, coord_step)
            for _ in range(layers)
        ])
        self.edge_input = nn.Sequential(nn.Linear(geom_channels, edge_hidden),
                                        nn.LayerNorm(edge_hidden), nn.SiLU())

    def dynamic_geometry(self, pos, receiver, sender, side, has_side):
        delta = pos[receiver] - pos[sender]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        unit = F.normalize(delta, dim=-1, eps=1e-8)
        radial = F.normalize(pos, dim=-1, eps=1e-8)
        side = F.normalize(side, dim=-1, eps=1e-8)
        ri, rj = radial[receiver], radial[sender]
        si, sj = side[receiver], side[sender]
        hs_i, hs_j = has_side[receiver], has_side[sender]
        # Invariant pair geometry in each DNA-local frame.
        angles = torch.stack(((ri * rj).sum(-1), (si * sj).sum(-1),
                              (unit * si).sum(-1), (unit * sj).sum(-1),
                              (unit * ri).sum(-1), (unit * rj).sum(-1),
                              hs_i, hs_j), dim=-1)
        rbf = torch.exp(-((distance - self.rbf_centers) / self.rbf_width).square())
        return torch.cat((rbf, angles), dim=-1)

    def encode_chunk(self, features, mask, query):
        features = features.masked_fill(~mask[..., None], 0)
        nodes = self.node_input(features.index_select(-1, self.input_indices))
        nodes = nodes.masked_fill(~mask[..., None], 0)
        hub = nodes.sum(1) / mask.sum(1, keepdim=True).clamp_min(1) + query
        hub = hub.masked_fill(~mask.any(1)[:, None], 0)
        pos = features[..., POSITION_INDICES] * 10.0
        edges, initial_geom = self.graph(features, mask)
        edge_state = self.edge_input(initial_geom)
        side = features[..., SIDECHAIN_INDICES].reshape(-1, 3)
        has_side = features[..., FEATURE_INDEX['has_sidechain_vector']].reshape(-1)
        for block in self.blocks:
            nodes, hub, pos, edge_state = block(nodes, hub, pos, mask, edges,
                                                edge_state, self.dynamic_geometry,
                                                side, has_side)
        return hub


class ResidualSequenceCNN(nn.Module):
    """Wider DNA-position readout, avoiding a second eight-channel bottleneck."""
    def __init__(self, in_channels, hidden_channels=32, depth=3, dropout=0.1):
        super().__init__()
        if depth < 1 or hidden_channels < 1:
            raise ValueError('Fusion depth and width must be positive')
        self.input = nn.Linear(in_channels, hidden_channels)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(depth)])
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1), nn.SiLU(),
            nn.Dropout(dropout), nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        ) for _ in range(depth)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        h = self.input(features)
        for block, norm in zip(self.blocks, self.norms):
            update = block(norm(h).T.unsqueeze(0)).squeeze(0).T
            h = h + self.dropout(update)
        return h
