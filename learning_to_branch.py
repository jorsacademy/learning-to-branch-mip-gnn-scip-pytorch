from __future__ import annotations
import argparse
import numpy as np
import torch

from ltb.problem import generate_set_cover_instance
from ltb.graph import static_graph_features, BranchState
from ltb.model import BranchingBipartiteGNN, masked_branching_loss


def self_test():
    instance = generate_set_cover_instance(seed=7, n_constraints=8, n_variables=16)
    c, vstatic, e = static_graph_features(instance)
    assert c.shape == (8, 3)
    assert vstatic.shape == (16, 3)
    assert e.shape == (8, 16, 2)

    # Create a synthetic branch state for pure-PyTorch validation.
    v = np.concatenate([vstatic, np.zeros((16,4), dtype=np.float32)], axis=1)
    mask = np.zeros(16, dtype=bool)
    mask[[1,4,9]] = True
    state = BranchState(c, v, e, mask, None, 4)

    model = BranchingBipartiteGNN(hidden_dim=24, layers=2)
    logits = model(
        torch.tensor(c)[None],
        torch.tensor(v)[None],
        torch.tensor(e)[None],
    )
    loss = masked_branching_loss(
        logits,
        torch.tensor(mask)[None],
        torch.tensor([4]),
    )
    loss.backward()
    assert logits.shape == (1,16)
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())
    print("Learning-to-Branch pure graph/model self-test: OK")


def integration_smoke(args):
    from ltb.dataset import collect_expert_dataset
    from ltb.train import train_branching_gnn
    from ltb.benchmark import benchmark_policies

    states = collect_expert_dataset(
        n_instances=args.train_instances + args.validation_instances,
        seed=args.seed,
        n_constraints=args.constraints,
        n_variables=args.variables,
        samples_per_instance=args.samples_per_instance,
        node_limit=args.collection_node_limit,
    )
    if len(states) < 4:
        raise RuntimeError(f"too few expert states collected: {len(states)}")

    cut = max(1, int(0.8*len(states)))
    cut = min(cut, len(states)-1)
    result = train_branching_gnn(
        states[:cut],
        states[cut:],
        seed=args.seed,
        epochs=args.epochs,
        batch_size=min(args.batch_size, cut),
        hidden_dim=args.hidden_dim,
        layers=args.layers,
    )
    print(f"collected expert states             : {len(states)}")
    print(f"best expert imitation accuracy      : {result.best_validation_accuracy:.3f}")

    seeds = [args.seed + 200_000 + k for k in range(args.test_instances)]
    rows = benchmark_policies(
        result.model,
        seeds=seeds,
        n_constraints=args.constraints,
        n_variables=args.variables,
        time_limit=args.time_limit,
    )

    print("="*100)
    print("LEARNING TO BRANCH — SCIP B&B BENCHMARK")
    print("="*100)
    for policy in ("default","pseudocost","strong","learned"):
        group = [r for r in rows if r.policy == policy]
        solved = [r for r in group if "optimal" in r.status.lower()]
        print(
            f"{policy:<12} solved={len(solved)}/{len(group)} "
            f"nodes={np.mean([r.nodes for r in group]):8.2f} "
            f"time={np.mean([r.solve_seconds for r in group]):7.3f}s "
            f"gap={np.mean([r.gap for r in group]):9.5f}"
        )


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--integration-smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--constraints", type=int, default=24)
    p.add_argument("--variables", type=int, default=52)
    p.add_argument("--train-instances", type=int, default=6)
    p.add_argument("--validation-instances", type=int, default=2)
    p.add_argument("--test-instances", type=int, default=4)
    p.add_argument("--samples-per-instance", type=int, default=8)
    p.add_argument("--collection-node-limit", type=int, default=120)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=48)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--time-limit", type=float, default=8.0)
    return p.parse_args()


if __name__ == "__main__":
    args=parse_args()
    if args.self_test:
        self_test()
    elif args.integration_smoke:
        integration_smoke(args)
    else:
        raise SystemExit("choose --self-test or --integration-smoke")
