from __future__ import annotations
import numpy as np
from .graph import extract_scip_branch_state


def strong_branch_scores(scip, candidates, *, iteration_limit: int = 80):
    """
    Query SCIP strong branching without intentionally mutating solver state.

    Score each candidate with SCIP's own branching score aggregation.
    """
    lpobj = float(scip.getLPObjVal())
    scores = np.full(len(candidates), -np.inf, dtype=float)
    lperror = False

    scip.startStrongbranch()
    try:
        for i, var in enumerate(candidates):
            (
                down, up, downvalid, upvalid,
                downinf, upinf, _downconflict, _upconflict, err,
            ) = scip.getVarStrongbranch(
                var,
                int(iteration_limit),
                idempotent=True,
            )
            if err:
                lperror = True
                break

            cutoff_gain = max(abs(lpobj), 1.0) + 1e4
            if downinf:
                downgain = cutoff_gain
            elif downvalid:
                downgain = max(float(down) - lpobj, 0.0)
            else:
                downgain = 0.0

            if upinf:
                upgain = cutoff_gain
            elif upvalid:
                upgain = max(float(up) - lpobj, 0.0)
            else:
                upgain = 0.0

            scores[i] = float(scip.getBranchScoreMultiple(var, [downgain, upgain]))
    finally:
        scip.endStrongbranch()

    return scores, lperror


class StrongBranchCollector:
    """Factory namespace; real plugin class is created lazily with PySCIPOpt."""

    @staticmethod
    def make(instance, storage, *, max_samples=24, iteration_limit=80):
        from pyscipopt import Branchrule, SCIP_RESULT

        class Collector(Branchrule):
            def branchexeclp(self, allowaddcons):
                (
                    cands, sols, fracs, ncands, npriocands, _nimpl
                ) = self.model.getLPBranchCands()
                n = int(npriocands)
                if n <= 0:
                    return {"result": SCIP_RESULT.DIDNOTRUN}
                cands, sols, fracs = cands[:n], sols[:n], fracs[:n]

                scores, error = strong_branch_scores(
                    self.model, cands, iteration_limit=iteration_limit
                )
                if error or not np.any(np.isfinite(scores)):
                    return {"result": SCIP_RESULT.DIDNOTRUN}

                best_local = int(np.nanargmax(scores))
                if len(storage) < max_samples:
                    state = extract_scip_branch_state(
                        self.model,
                        instance,
                        cands,
                        sols,
                        fracs,
                        expert_scores=scores,
                    )
                    if state.expert_best is not None:
                        storage.append(state)

                self.model.branchVarVal(cands[best_local], sols[best_local])
                return {"result": SCIP_RESULT.BRANCHED}

        return Collector()
