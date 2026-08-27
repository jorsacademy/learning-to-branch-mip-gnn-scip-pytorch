from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .problem import build_scip_model, generate_set_cover_instance
from .branching import LearnedBranchRule, StrongBranchRule


@dataclass(frozen=True)
class SolveMetrics:
    policy: str
    seed: int
    status: str
    objective: float
    nodes: int
    solve_seconds: float
    gap: float


def solve_instance(instance, *, policy, network=None, seed=0, time_limit=15.0, node_limit=5000):
    scip, _ = build_scip_model(instance)
    scip.setRealParam("limits/time", float(time_limit))
    scip.setLongintParam("limits/nodes", int(node_limit))
    scip.setIntParam("randomization/randomseedshift", int(seed % 2_000_000_000))

    if policy == "learned":
        if network is None:
            raise ValueError("learned policy requires network")
        rule = LearnedBranchRule.make(instance, network)
        scip.includeBranchrule(
            rule, "learned_gnn_branching", "GNN branching rule",
            priority=10_000_000, maxdepth=-1, maxbounddist=1.0,
        )
    elif policy == "strong":
        rule = StrongBranchRule.make()
        scip.includeBranchrule(
            rule, "strong_branching_reference", "strong branching reference",
            priority=10_000_000, maxdepth=-1, maxbounddist=1.0,
        )
    elif policy == "pseudocost":
        scip.setIntParam("branching/pscost/priority", 10_000_000)
    elif policy == "default":
        pass
    else:
        raise ValueError("unknown policy")

    scip.optimize()
    status = str(scip.getStatus())
    bestsol = scip.getBestSol()
    obj = float(scip.getSolObjVal(bestsol)) if bestsol is not None else float("nan")
    return SolveMetrics(
        policy=policy,
        seed=seed,
        status=status,
        objective=obj,
        nodes=int(scip.getNNodes()),
        solve_seconds=float(scip.getSolvingTime()),
        gap=float(scip.getGap()),
    )


def benchmark_policies(
    network,
    *,
    seeds,
    n_constraints=28,
    n_variables=64,
    time_limit=15.0,
):
    rows = []
    for seed in seeds:
        instance = generate_set_cover_instance(
            seed=int(seed),
            n_constraints=n_constraints,
            n_variables=n_variables,
        )
        for policy in ("default", "pseudocost", "strong", "learned"):
            rows.append(solve_instance(
                instance,
                policy=policy,
                network=network,
                seed=int(seed),
                time_limit=time_limit,
            ))
    return tuple(rows)
