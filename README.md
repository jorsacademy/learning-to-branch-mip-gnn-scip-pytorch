# Learning to Branch MIP GNN with SCIP and PyTorch

A from-scratch **Learning to Branch** research implementation that trains a bipartite graph neural network to imitate strong branching and then inserts the learned variable-selection policy back into SCIP's branch-and-bound loop through a PySCIPOpt branching-rule callback.

The repository is not a classifier detached from a solver. Its full pipeline is:

```text
MILP instances
    ↓
SCIP LP branching states
    ↓
real strong-branching candidate evaluation
    ↓
expert branching labels
    ↓
constraint-variable bipartite GNN
    ↓
offline imitation learning
    ↓
PySCIPOpt learned branching rule
    ↓
branch-and-bound benchmark
```

## Research lineage

The design follows the core research question studied by Gasse et al., *Exact Combinatorial Optimization with Graph Convolutional Neural Networks* (NeurIPS 2019): represent a MILP as a variable-constraint bipartite graph and learn a branching policy by imitating strong branching.

This repository is an independent educational implementation. It does not copy `learn2branch`, Ecole, PyEPO, or another repository's source/API layout.

## MILP family

The benchmark uses random minimum-cost **set covering** instances:

```text
minimize    sum_j c[j] x[j]

subject to  sum_j A[i,j] x[j] >= demand[i]    for every row i

x[j] ∈ {0,1}
```

The generator creates heterogeneous column costs and random sparse incidence structure while repairing rows only as needed to guarantee feasibility. No optimum is planted.

## Why set covering

Set covering has a natural MILP graph:

```text
constraint node i
      ↕  A[i,j]
variable node j
```

It also makes it possible to generate many independent train/validation/test MILPs with identical tensor dimensions while preserving nontrivial fractional LP relaxations.

## Expert data: real SCIP strong branching

At an LP branching node, the collector calls:

```text
getLPBranchCands()
startStrongbranch()
getVarStrongbranch(...)
getBranchScoreMultiple(...)
endStrongbranch()
```

for the current SCIP candidate variables.

Strong branching evaluates both child LPs and produces a score for each candidate. The highest-score candidate becomes the imitation-learning target.

The calls use `idempotent=True` so the data query is intended to preserve SCIP state rather than turning data generation into an uncontrolled solver modification.

After recording the state, the collector actually branches on the expert-selected variable. Training data therefore come from a strong-branching trajectory, not from independently sampled fractional vectors.

## Bipartite state representation

Constraint features include:

- demand relative to row degree;
- normalized row degree;
- normalized demand.

Variable features include:

- normalized objective coefficient;
- normalized column degree;
- cost per covered row;
- current LP solution;
- fractional part;
- reduced cost;
- current branching-candidate indicator.

Edge features include:

- binary incidence coefficient;
- incidence-weighted normalized variable cost.

The GNN never receives the expert answer as an input feature.

## GNN architecture

No graph-learning framework is required. Message passing is implemented directly in PyTorch.

```text
constraint embedding
variable embedding
edge embedding
       ↓
constraint-variable edge MLP
       ↓
incidence-masked messages
       ↓
aggregate to both node partitions
       ↓
residual update + LayerNorm
       ↓
variable branching logits
```

At inference time, all non-candidate variables are masked to `-∞`. The learned rule can therefore choose only from SCIP's current LP branching candidate set.

## Training objective

For every expert state:

```text
candidate logits
    ↓
mask non-candidates
    ↓
cross entropy
    ↓
strong-branching best candidate
```

Validation reports **expert imitation accuracy**.

This metric is useful for learning diagnostics but is not treated as the final solver KPI.

## Solver integration

The learned network is wrapped in a genuine PySCIPOpt `Branchrule`.

At every LP branching callback:

1. SCIP provides the current candidate variables;
2. dynamic LP features are extracted;
3. the GNN scores the full variable partition;
4. non-candidates are masked;
5. the highest-scoring candidate is mapped back to its SCIP variable;
6. `branchVarVal()` creates the child nodes.

If state extraction or candidate mapping fails, the callback returns `DIDNOTRUN`, allowing SCIP to continue with another rule instead of producing an invalid branch.

## Benchmarks

The same held-out MILP instances are solved with:

```text
SCIP default branching
SCIP pseudocost branching
strong-branching reference
learned GNN branching
```

Reported metrics:

- solved-instance count;
- branch-and-bound node count;
- SCIP solving time;
- final MIP gap;
- expert imitation accuracy.

The project does **not** infer solver improvement from classification accuracy.

A learned policy is only better if the actual branch-and-bound metrics support that conclusion.

## Strong-branching reference

A custom reference rule applies the same strong-branch scoring routine at every eligible LP node and branches on the best candidate.

This is intentionally expensive and serves as an expert/reference policy, not a claim that full strong branching is the preferred production setting.

PySCIPOpt's own documentation notes that a simplified strong-branching example does not reproduce every subtle interaction of SCIP's production branching rules. The same caveat applies here.

## Local verification

The pure graph/model code is testable without SCIP and covers:

- deterministic instance generation;
- row feasibility;
- graph tensor dimensions;
- incidence masking;
- candidate masking;
- GNN gradient flow;
- masked branching loss;
- NPZ expert-state round-trip.

When PySCIPOpt is installed, additional integration tests run a real SCIP model and a real strong-branching collector.

## GitHub Actions integration

CI installs current PySCIPOpt and CPU PyTorch, prints the actual SCIP/PySCIPOpt versions, then runs:

```text
pure graph/model self-test
        ↓
regression tests
        ↓
real SCIP solve
        ↓
real strong-branching expert collection
        ↓
GNN imitation training
        ↓
learned Branchrule callback
        ↓
default / pseudocost / strong / learned benchmark
```

The CI smoke benchmark is intentionally small. It validates mechanics and integration; it is not a publication-scale branching benchmark.

## Run

Install:

```bash
pip install -r requirements.txt
```

Pure self-test:

```bash
python learning_to_branch.py --self-test
```

All tests:

```bash
python -m unittest discover -s tests -v
```

A development integration experiment:

```bash
python learning_to_branch.py \
  --integration-smoke \
  --seed 42 \
  --constraints 24 \
  --variables 52 \
  --train-instances 6 \
  --validation-instances 2 \
  --test-instances 4 \
  --samples-per-instance 8 \
  --collection-node-limit 120 \
  --epochs 8 \
  --batch-size 16 \
  --hidden-dim 48 \
  --layers 2 \
  --time-limit 8
```

## Exactness and scope

Exact statements are deliberately narrow:

- SCIP solves each reported MILP according to its solver status and configured limits;
- expert targets are derived from explicit strong-branch evaluations of current SCIP candidates;
- the learned callback branches only on candidates supplied by SCIP.

Not claimed:

- the GNN reproduces strong branching perfectly;
- expert imitation accuracy guarantees a smaller B&B tree;
- the learned rule is globally optimal;
- the small synthetic benchmark establishes universal speedup;
- full strong branching in this repository reproduces every internal detail of SCIP's production `fullstrong`/`relpscost` implementations.

The main purpose is to demonstrate the complete learned-solver-control loop with visible failure modes and measurable solver consequences.

## References

- Gasse, Chételat, Ferroni, Charlin, Lodi. **Exact Combinatorial Optimization with Graph Convolutional Neural Networks.** NeurIPS 2019.
- SCIP / PySCIPOpt branching-rule and strong-branching documentation.
