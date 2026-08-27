from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SetCoverInstance:
    incidence: np.ndarray   # [constraint, variable], binary
    costs: np.ndarray       # [variable]
    demands: np.ndarray     # [constraint], positive integer

    def __post_init__(self):
        A = np.asarray(self.incidence, dtype=np.int8)
        c = np.asarray(self.costs, dtype=float)
        b = np.asarray(self.demands, dtype=float)
        if A.ndim != 2 or c.shape != (A.shape[1],) or b.shape != (A.shape[0],):
            raise ValueError("invalid set-cover shapes")
        if np.any((A != 0) & (A != 1)):
            raise ValueError("incidence must be binary")
        if np.any(c <= 0) or np.any(b <= 0):
            raise ValueError("costs/demands must be positive")
        if np.any(A.sum(axis=1) < b):
            raise ValueError("every covering row must have enough incident variables")

    @property
    def n_constraints(self): return int(self.incidence.shape[0])

    @property
    def n_variables(self): return int(self.incidence.shape[1])


def generate_set_cover_instance(
    *,
    seed: int,
    n_constraints: int = 28,
    n_variables: int = 64,
    density: float = 0.18,
    max_demand: int = 2,
) -> SetCoverInstance:
    """
    Random binary set-cover family with row coverage repairs.

    The generator does not plant an optimum. It only guarantees feasibility by
    ensuring every row has at least max_demand incident columns.
    """
    if not (0.02 <= density <= 0.8):
        raise ValueError("density outside supported range")
    rng = np.random.default_rng(seed)
    A = (rng.random((n_constraints, n_variables)) < density).astype(np.int8)

    required = max(1, int(max_demand))
    for i in range(n_constraints):
        current = int(A[i].sum())
        if current < required:
            candidates = np.flatnonzero(A[i] == 0)
            chosen = rng.choice(candidates, size=required-current, replace=False)
            A[i, chosen] = 1

    degree = A.sum(axis=0)
    # Encourage nontrivial trade-offs: cost partly correlates with coverage but
    # contains independent noise and nonlinear variation.
    base = rng.uniform(8.0, 35.0, size=n_variables)
    costs = base + 1.8 * np.sqrt(degree + 1.0) + 3.5 * np.sin(np.arange(n_variables)*0.37)
    costs += rng.normal(0.0, 2.5, size=n_variables)
    costs = np.clip(costs, 1.0, None)

    demands = rng.integers(1, max_demand + 1, size=n_constraints, dtype=np.int32)
    demands = np.minimum(demands, A.sum(axis=1)).astype(np.int32)
    return SetCoverInstance(A, costs.astype(np.float64), demands)


def build_scip_model(instance: SetCoverInstance, *, hide_output: bool = True):
    try:
        from pyscipopt import Model, quicksum
    except ImportError as exc:
        raise RuntimeError("PySCIPOpt is required for SCIP integration") from exc

    model = Model("learning-to-branch-set-cover")
    if hide_output:
        model.hideOutput(True)

    variables = [
        model.addVar(name=f"x_{j}", vtype="B", obj=float(instance.costs[j]))
        for j in range(instance.n_variables)
    ]
    for i in range(instance.n_constraints):
        cols = np.flatnonzero(instance.incidence[i])
        model.addCons(
            quicksum(variables[int(j)] for j in cols) >= int(instance.demands[i]),
            name=f"cover_{i}",
        )
    model.setMinimize()
    return model, variables
