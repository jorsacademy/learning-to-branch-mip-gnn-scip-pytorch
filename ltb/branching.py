from __future__ import annotations
import numpy as np
import torch
from .graph import extract_scip_branch_state, scip_variable_index
from .model import choose_candidate_index
from .expert import strong_branch_scores


class LearnedBranchRule:
    @staticmethod
    def make(instance, network):
        from pyscipopt import Branchrule, SCIP_RESULT

        class Rule(Branchrule):
            def branchexeclp(self, allowaddcons):
                cands, sols, fracs, ncands, npriocands, _ = self.model.getLPBranchCands()
                n = int(npriocands)
                if n <= 0:
                    return {"result": SCIP_RESULT.DIDNOTRUN}
                cands, sols, fracs = cands[:n], sols[:n], fracs[:n]
                state = extract_scip_branch_state(
                    self.model, instance, cands, sols, fracs
                )
                if not np.any(state.candidate_mask):
                    return {"result": SCIP_RESULT.DIDNOTRUN}

                with torch.no_grad():
                    logits = network(
                        torch.tensor(state.constraint_features)[None],
                        torch.tensor(state.variable_features)[None],
                        torch.tensor(state.edge_features)[None],
                    )[0]
                    mask = torch.tensor(state.candidate_mask)
                    j = choose_candidate_index(logits, mask)

                selected = next(
                    (
                        k for k, var in enumerate(cands)
                        if scip_variable_index(var, instance.n_variables) == j
                    ),
                    None,
                )
                if selected is None:
                    return {"result": SCIP_RESULT.DIDNOTRUN}
                self.model.branchVarVal(cands[selected], sols[selected])
                return {"result": SCIP_RESULT.BRANCHED}
        return Rule()


class StrongBranchRule:
    @staticmethod
    def make(iteration_limit=80):
        from pyscipopt import Branchrule, SCIP_RESULT

        class Rule(Branchrule):
            def branchexeclp(self, allowaddcons):
                cands, sols, fracs, ncands, npriocands, _ = self.model.getLPBranchCands()
                n = int(npriocands)
                if n <= 0:
                    return {"result": SCIP_RESULT.DIDNOTRUN}
                cands, sols = cands[:n], sols[:n]
                scores, error = strong_branch_scores(
                    self.model, cands, iteration_limit=iteration_limit
                )
                if error or not np.any(np.isfinite(scores)):
                    return {"result": SCIP_RESULT.DIDNOTRUN}
                best = int(np.nanargmax(scores))
                self.model.branchVarVal(cands[best], sols[best])
                return {"result": SCIP_RESULT.BRANCHED}
        return Rule()
