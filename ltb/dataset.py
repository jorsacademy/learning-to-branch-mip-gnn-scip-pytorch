from __future__ import annotations
from pathlib import Path
import numpy as np
from .expert import StrongBranchCollector
from .problem import (
    build_scip_model,
    configure_branching_research_mode,
    generate_set_cover_instance,
)


def collect_expert_dataset(
    *,
    n_instances=12,
    seed=42,
    n_constraints=28,
    n_variables=64,
    samples_per_instance=12,
    node_limit=200,
):
    all_states = []
    for k in range(n_instances):
        instance = generate_set_cover_instance(
            seed=seed + 1009*k,
            n_constraints=n_constraints,
            n_variables=n_variables,
        )
        scip, _ = build_scip_model(instance)
        configure_branching_research_mode(scip)
        scip.setLongintParam("limits/nodes", int(node_limit))

        storage = []
        rule = StrongBranchCollector.make(
            instance,
            storage,
            max_samples=samples_per_instance,
        )
        scip.includeBranchrule(
            rule,
            "collect_strong_branch",
            "strong-branching expert data collector",
            priority=10_000_000,
            maxdepth=-1,
            maxbounddist=1.0,
        )
        scip.optimize()
        all_states.extend(storage)
    return all_states


def save_states_npz(states, path):
    path = Path(path)
    np.savez_compressed(
        path,
        constraint_features=np.stack([s.constraint_features for s in states]),
        variable_features=np.stack([s.variable_features for s in states]),
        edge_features=np.stack([s.edge_features for s in states]),
        candidate_mask=np.stack([s.candidate_mask for s in states]),
        expert_best=np.asarray([s.expert_best for s in states], dtype=np.int64),
    )


def load_states_npz(path):
    from .graph import BranchState
    d = np.load(path)
    states = []
    for i in range(len(d["expert_best"])):
        states.append(BranchState(
            d["constraint_features"][i],
            d["variable_features"][i],
            d["edge_features"][i],
            d["candidate_mask"][i],
            None,
            int(d["expert_best"][i]),
        ))
    return states
