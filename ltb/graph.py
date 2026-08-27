from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from .problem import SetCoverInstance


@dataclass(frozen=True)
class BranchState:
    constraint_features: np.ndarray  # [C,Fc]
    variable_features: np.ndarray    # [V,Fv]
    edge_features: np.ndarray        # [C,V,Fe]
    candidate_mask: np.ndarray       # [V] bool
    expert_scores: np.ndarray | None = None
    expert_best: int | None = None


def static_graph_features(instance: SetCoverInstance):
    A = instance.incidence.astype(np.float32)
    row_degree = A.sum(axis=1)
    col_degree = A.sum(axis=0)
    costs = instance.costs.astype(np.float32)
    demands = instance.demands.astype(np.float32)

    constraint_features = np.column_stack([
        demands / np.maximum(row_degree, 1.0),
        row_degree / max(instance.n_variables, 1),
        demands / max(float(demands.max()), 1.0),
    ]).astype(np.float32)

    variable_static = np.column_stack([
        (costs - costs.mean()) / (costs.std() + 1e-6),
        col_degree / max(instance.n_constraints, 1),
        costs / (col_degree + 1.0),
    ]).astype(np.float32)

    edge_features = np.stack([
        A,
        A * (instance.costs[None, :] / (instance.costs.mean() + 1e-6)),
    ], axis=-1).astype(np.float32)

    return constraint_features, variable_static, edge_features


def extract_scip_branch_state(
    scip,
    instance: SetCoverInstance,
    branch_cands,
    branch_cand_sols,
    branch_cand_fracs,
    *,
    expert_scores=None,
):
    cfeat, vstatic, efeat = static_graph_features(instance)
    V = instance.n_variables
    vdyn = np.zeros((V, 4), dtype=np.float32)
    mask = np.zeros(V, dtype=bool)

    name_to_idx = {f"x_{j}": j for j in range(V)}
    for var, sol, frac in zip(branch_cands, branch_cand_sols, branch_cand_fracs):
        name = var.name
        if name not in name_to_idx:
            continue
        j = name_to_idx[name]
        mask[j] = True
        vdyn[j, 0] = float(sol)
        vdyn[j, 1] = float(frac)
        try:
            vdyn[j, 2] = float(scip.getVarRedcost(var))
        except Exception:
            vdyn[j, 2] = 0.0
        vdyn[j, 3] = 1.0

    variable_features = np.concatenate([vstatic, vdyn], axis=1)

    scores = None
    best = None
    if expert_scores is not None:
        scores = np.full(V, -np.inf, dtype=np.float32)
        for var, score in zip(branch_cands, expert_scores):
            j = name_to_idx.get(var.name)
            if j is not None:
                scores[j] = float(score)
        finite = np.flatnonzero(np.isfinite(scores))
        if len(finite):
            best = int(finite[np.argmax(scores[finite])])

    return BranchState(cfeat, variable_features, efeat, mask, scores, best)


def tensorize_states(states):
    return (
        torch.tensor(np.stack([s.constraint_features for s in states]), dtype=torch.float32),
        torch.tensor(np.stack([s.variable_features for s in states]), dtype=torch.float32),
        torch.tensor(np.stack([s.edge_features for s in states]), dtype=torch.float32),
        torch.tensor(np.stack([s.candidate_mask for s in states]), dtype=torch.bool),
        torch.tensor([int(s.expert_best) for s in states], dtype=torch.long),
    )
