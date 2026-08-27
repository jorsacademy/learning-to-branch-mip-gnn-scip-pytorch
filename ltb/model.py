from __future__ import annotations
import torch
from torch import nn


class BranchingBipartiteGNN(nn.Module):
    """
    Dense pure-PyTorch variable-constraint bipartite GNN.

    Intended for fixed-size research benchmarks. Candidate masking is applied
    outside the network so non-branchable variables never receive policy mass.
    """

    def __init__(
        self,
        constraint_dim=3,
        variable_dim=7,
        edge_dim=2,
        hidden_dim=64,
        layers=3,
    ):
        super().__init__()
        self.c_embed = nn.Linear(constraint_dim, hidden_dim)
        self.v_embed = nn.Linear(variable_dim, hidden_dim)
        self.e_embed = nn.Linear(edge_dim, hidden_dim)

        self.edge_mlps = nn.ModuleList()
        self.c_mlps = nn.ModuleList()
        self.v_mlps = nn.ModuleList()
        self.c_norm = nn.ModuleList()
        self.v_norm = nn.ModuleList()

        for _ in range(layers):
            self.edge_mlps.append(nn.Sequential(
                nn.Linear(hidden_dim*3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ))
            self.c_mlps.append(nn.Linear(hidden_dim*2, hidden_dim))
            self.v_mlps.append(nn.Linear(hidden_dim*2, hidden_dim))
            self.c_norm.append(nn.LayerNorm(hidden_dim))
            self.v_norm.append(nn.LayerNorm(hidden_dim))

        self.variable_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, constraint_features, variable_features, edge_features):
        c = self.c_embed(constraint_features)
        v = self.v_embed(variable_features)
        e0 = self.e_embed(edge_features)

        for emlp, cmlp, vmlp, cnorm, vnorm in zip(
            self.edge_mlps, self.c_mlps, self.v_mlps, self.c_norm, self.v_norm
        ):
            B, C, H = c.shape
            V = v.shape[1]
            ce = c[:, :, None, :].expand(B, C, V, H)
            ve = v[:, None, :, :].expand(B, C, V, H)
            edge_msg = emlp(torch.cat([ce, ve, e0], dim=-1))

            adjacency = edge_features[..., 0:1]
            edge_msg = edge_msg * adjacency
            cden = adjacency.sum(dim=2).clamp_min(1.0)
            vden = adjacency.sum(dim=1).clamp_min(1.0)
            cagg = edge_msg.sum(dim=2) / cden
            vagg = edge_msg.sum(dim=1) / vden

            c = cnorm(c + cmlp(torch.cat([c, cagg], dim=-1)))
            v = vnorm(v + vmlp(torch.cat([v, vagg], dim=-1)))

        return self.variable_scorer(v).squeeze(-1)


def masked_branching_loss(logits, candidate_mask, expert_best):
    masked = logits.masked_fill(~candidate_mask, float("-inf"))
    return nn.functional.cross_entropy(masked, expert_best)


def choose_candidate_index(logits, candidate_mask):
    masked = logits.masked_fill(~candidate_mask, float("-inf"))
    return int(torch.argmax(masked).item())
