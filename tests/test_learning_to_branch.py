import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ltb.dataset import load_states_npz, save_states_npz
from ltb.graph import BranchState, static_graph_features, tensorize_states
from ltb.model import BranchingBipartiteGNN, choose_candidate_index, masked_branching_loss
from ltb.problem import generate_set_cover_instance


class LearningToBranchPureTests(unittest.TestCase):
    def _state(self, seed=1, best=3):
        instance = generate_set_cover_instance(
            seed=seed,
            n_constraints=7,
            n_variables=14,
        )
        c, vs, e = static_graph_features(instance)
        v = np.concatenate([vs, np.zeros((14, 4), dtype=np.float32)], axis=1)
        mask = np.zeros(14, dtype=bool)
        mask[[1, 3, 8, 11]] = True
        return BranchState(c, v, e, mask, None, best)

    def test_generator_reproducible_and_rows_coverable(self):
        a = generate_set_cover_instance(seed=10, n_constraints=9, n_variables=18)
        b = generate_set_cover_instance(seed=10, n_constraints=9, n_variables=18)
        np.testing.assert_array_equal(a.incidence, b.incidence)
        np.testing.assert_array_equal(a.costs, b.costs)
        np.testing.assert_array_equal(a.demands, b.demands)
        self.assertTrue(np.all(a.incidence.sum(axis=1) >= a.demands))

    def test_graph_feature_shapes(self):
        instance = generate_set_cover_instance(seed=11, n_constraints=8, n_variables=16)
        c, v, e = static_graph_features(instance)
        self.assertEqual(c.shape, (8, 3))
        self.assertEqual(v.shape, (16, 3))
        self.assertEqual(e.shape, (8, 16, 2))
        np.testing.assert_array_equal(e[..., 0], instance.incidence)

    def test_gnn_shape_and_gradient(self):
        state = self._state()
        c, v, e, mask, target = tensorize_states([state])
        model = BranchingBipartiteGNN(hidden_dim=24, layers=2)
        logits = model(c, v, e)
        self.assertEqual(tuple(logits.shape), (1, 14))
        loss = masked_branching_loss(logits, mask, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        ))

    def test_mask_prevents_non_candidate_choice(self):
        logits = torch.tensor([100.0, 1.0, 2.0, 3.0])
        mask = torch.tensor([False, True, False, True])
        self.assertEqual(choose_candidate_index(logits, mask), 3)

    def test_masked_loss_rejects_high_non_candidate_logit(self):
        logits = torch.tensor([[1000.0, 2.0, 1.0]], requires_grad=True)
        mask = torch.tensor([[False, True, True]])
        loss = masked_branching_loss(logits, mask, torch.tensor([1]))
        expected = torch.nn.functional.cross_entropy(
            torch.tensor([[2.0, 1.0]]),
            torch.tensor([0]),
        )
        self.assertAlmostEqual(float(loss.item()), float(expected.item()), places=6)

    def test_dataset_npz_roundtrip(self):
        states = [self._state(seed=20, best=3), self._state(seed=21, best=8)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "states.npz"
            save_states_npz(states, path)
            loaded = load_states_npz(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].expert_best, 3)
        self.assertEqual(loaded[1].expert_best, 8)
        np.testing.assert_array_equal(
            loaded[0].candidate_mask,
            states[0].candidate_mask,
        )

    def test_batch_tensorization(self):
        states = [self._state(seed=30, best=3), self._state(seed=31, best=8)]
        c, v, e, mask, target = tensorize_states(states)
        self.assertEqual(tuple(c.shape), (2, 7, 3))
        self.assertEqual(tuple(v.shape), (2, 14, 7))
        self.assertEqual(tuple(e.shape), (2, 7, 14, 2))
        self.assertEqual(tuple(mask.shape), (2, 14))
        self.assertEqual(tuple(target.shape), (2,))


class PySCIPOptIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import pyscipopt  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("PySCIPOpt unavailable in local runtime")

    def test_small_scip_model_solves(self):
        from ltb.problem import build_scip_model
        instance = generate_set_cover_instance(
            seed=90, n_constraints=10, n_variables=22
        )
        model, _ = build_scip_model(instance)
        model.setRealParam("limits/time", 10.0)
        model.optimize()
        self.assertIn("optimal", str(model.getStatus()).lower())

    def test_real_strong_branch_collector_produces_valid_states(self):
        from ltb.dataset import collect_expert_dataset
        states = collect_expert_dataset(
            n_instances=2,
            seed=91,
            n_constraints=12,
            n_variables=28,
            samples_per_instance=3,
            node_limit=60,
        )
        self.assertGreater(len(states), 0)
        for state in states:
            self.assertTrue(state.candidate_mask[state.expert_best])
            self.assertEqual(state.variable_features.shape[1], 7)


if __name__ == "__main__":
    unittest.main()
