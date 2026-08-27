# Learning to Branch MIP GNN with SCIP and PyTorch

A from-scratch **Learning to Branch** research implementation that trains a bipartite graph neural network to imitate strong branching and then inserts the learned variable-selection policy back into SCIP's branch-and-bound loop through a PySCIPOpt branching-rule callback.

The repository is not a classifier detached from a solver:

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
PySCIPOpt learned Branchrule
    ↓
actual branch-and-bound benchmark
```

## Research lineage

The design follows the core research question studied by Gasse et al., *Exact Combinatorial Optimization with Graph Convolutional Neural Networks* (NeurIPS 2019): represent a MILP as a variable-constraint bipartite graph and learn a branching policy by imitating strong branching.

This repository is an independent educational implementation. It does not copy `learn2branch`, Ecole, PyEPO, or another repository's source/API layout.

## MILP family

The benchmark uses minimum-cost set covering:

```text
minimize    sum_j c[j] x[j]

subject to  sum_j A[i,j] x[j] >= demand[i]    for every row i

x[j] ∈ {0,1}
```

Most of the instance is random sparse set covering with heterogeneous column costs. A small isolated five-variable odd-cycle covering core is added deliberately:

```text
x0 + x1 >= 1
x1 + x2 >= 1
x2 + x3 >= 1
x3 + x4 >= 1
x4 + x0 >= 1
```

Under the controlled solver configuration below, this core provides a reproducible fractional LP structure so that real branching callbacks are exercised. It is **not** an optimal integer solution planted into the data.

## Controlled branching-research mode

Expert collection and all four branching-policy benchmarks use the same SCIP research configuration:

```text
presolve          OFF
primal heuristics OFF
separation/cuts   OFF
```

The purpose is to isolate the effect of branching and prevent presolve, heuristics or cutting planes from eliminating the branching decisions being studied.

This scope matters: `default` in the benchmark means **SCIP's default branching policy inside this controlled branching-research configuration**. It is not a comparison against unrestricted full-default SCIP.

## Expert data: real SCIP strong branching

At an LP branching node, the collector obtains SCIP's actual candidate set and calls:

```text
getLPBranchCands()
startStrongbranch()
getVarStrongbranch(...)
getBranchScoreMultiple(...)
endStrongbranch()
```

Strong branching evaluates the down/up child LPs. The candidate with the best SCIP branching score becomes the imitation target. Calls use `idempotent=True` for data extraction so the query is intended to preserve solver state.

After recording the state, the collector actually branches on the expert-selected variable. Training samples therefore come from solver states visited along strong-branching trajectories rather than arbitrary fractional vectors.

## Bipartite state representation

Constraint features:

- demand relative to row degree;
- normalized row degree;
- normalized demand.

Variable features:

- normalized objective coefficient;
- normalized column degree;
- cost per covered row;
- current LP solution value;
- fractional part;
- reduced cost;
- branching-candidate indicator.

Edge features:

- binary incidence coefficient;
- incidence-weighted normalized variable cost.

SCIP may expose transformed variable names during solving. The implementation maps both original and transformed names back to the benchmark variable index instead of assuming that solver-side names remain unchanged.

The GNN never receives the expert answer as an input feature.

## Pure-PyTorch GNN

No graph-learning framework is required. Message passing is implemented directly in PyTorch:

```text
constraint embedding
variable embedding
edge embedding
       ↓
edge-message MLP
       ↓
incidence-masked messages
       ↓
aggregate to both node partitions
       ↓
residual update + LayerNorm
       ↓
variable branching logits
```

At inference, all non-candidate variables are masked to `-∞`, so the learned policy can select only a variable currently supplied by SCIP as an LP branching candidate.

## Imitation objective

For each expert state:

```text
GNN variable logits
      ↓
mask non-candidates
      ↓
cross entropy
      ↓
strong-branching best candidate
```

Validation reports expert-imitation accuracy. This is a learning diagnostic, not the final optimization KPI.

## Real solver integration

The trained network is wrapped in a genuine PySCIPOpt `Branchrule`.

At every eligible LP branching callback:

1. SCIP supplies the current candidates and LP values;
2. dynamic graph features are extracted;
3. the GNN scores variables;
4. non-candidates are masked;
5. the selected graph variable is mapped back to the current SCIP candidate object;
6. `branchVarVal()` creates the child nodes.

If no valid candidate can be mapped, the callback returns `DIDNOTRUN` so SCIP can safely fall back to another rule.

## Benchmark policies

Each held-out instance is solved under the same controlled solver settings with:

```text
SCIP default branching
SCIP pseudocost branching
custom strong-branching reference
learned GNN branching
```

Reported metrics are:

- solved-instance count;
- B&B node count;
- SCIP solving time;
- final MIP gap;
- expert-imitation accuracy.

The custom strong rule is an educational reference based on explicit `getVarStrongbranch` calls. It does not claim to reproduce every interaction of SCIP's production `fullstrong` or `relpscost` implementations.

## Regression tests

The suite covers:

- deterministic instance generation and row feasibility;
- graph feature dimensions and incidence representation;
- GNN shape and gradient flow;
- candidate masking;
- masked cross-entropy behavior;
- expert-state NPZ round trip;
- batched tensorization;
- a real PySCIPOpt set-cover solve;
- real strong-branching state collection from SCIP.

## Validated GitHub Actions run

GitHub Actions run `33105697429` completed successfully on Ubuntu 24.04 / CPython 3.12.14 with:

```text
PyTorch      2.13.0+cpu
PySCIPOpt    6.2.1
SCIP         10.0.2
NumPy        2.5.2
```

The pure self-test and all **9 regression/integration tests** passed. The CI run then collected real strong-branching states, trained the GNN and executed the learned PySCIPOpt branching callback in held-out SCIP solves.

CI smoke configuration:

```text
constraints                18
variables                  40
train instances             4
validation instances        2
test instances              3
expert samples/instance     5
collection node limit      80
GNN epochs                  5
hidden dimension           32
message-passing layers      2
time limit                  6 s
```

Observed learning result:

```text
collected expert states                 8
best validation expert accuracy       50.0%
```

Observed branch-and-bound result:

```text
policy       solved    mean nodes    mean time    mean final gap

default       3/3         1.67        0.001 s        0.00000
pseudocost    3/3         4.33        0.001 s        0.00000
strong        3/3         3.67        0.001 s        0.00000
learned       3/3        11.67        0.008 s        0.00000
```

This is intentionally reported as a **negative learned-policy result**. In this tiny CI experiment, the learned branching rule created a larger search tree and took longer than the controlled default policy. Only eight expert states were collected and the best validation imitation accuracy was 50%, so the run is far too small to support a solver-performance claim.

The CI result validates the end-to-end mechanics:

```text
real SCIP candidate states
→ real strong-branching labels
→ GNN training
→ learned Branchrule inference
→ actual B&B consequences
```

It does **not** validate a learned speedup.

The millisecond timings are runner/workload observations and are not portable performance guarantees.

Run: https://github.com/jorsacademy/learning-to-branch-mip-gnn-scip-pytorch/actions/runs/33105697429

## Run locally

Install:

```bash
pip install -r requirements.txt
```

Pure graph/model self-test:

```bash
python learning_to_branch.py --self-test
```

Tests:

```bash
python -m unittest discover -s tests -v
```

Integration experiment:

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

- SCIP solves each reported instance according to its solver status and configured limits;
- the successful CI smoke solved all 3/3 held-out instances to zero reported final MIP gap for every compared policy;
- expert labels are generated by explicit strong-branch evaluations of the current SCIP candidate set;
- the learned callback branches only on candidates supplied by SCIP.

Not claimed:

- the GNN reproduces strong branching perfectly;
- imitation accuracy guarantees a smaller B&B tree;
- the learned branching rule is globally optimal;
- the current smoke experiment demonstrates a speedup;
- the controlled benchmark represents full-default SCIP behavior;
- the synthetic set-cover results transfer directly to production MIPs;
- the custom Python strong-branching reference reproduces every internal detail of SCIP's production branching rules.

The purpose of the project is to expose the complete learned-solver-control loop, including cases where the learned policy is worse than the solver baseline.

## References

- Gasse, Chételat, Ferroni, Charlin, Lodi. **Exact Combinatorial Optimization with Graph Convolutional Neural Networks.** NeurIPS 2019.
- SCIP / PySCIPOpt branching-rule and strong-branching documentation.
