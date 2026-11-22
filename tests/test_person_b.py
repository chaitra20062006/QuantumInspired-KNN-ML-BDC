"""Person B contract tests.  Run:  python -m unittest tests.test_person_b -v"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from shared.interfaces import check_optimizer_output, check_solution
from person_a.standins import make_toy_objective, standin_load_dataset
from person_b.classifier import predict
from person_b.fitness import fitness
from person_b import pso


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.Xtr, self.Xte, self.ytr, self.yte, self.folds = standin_load_dataset(seed=1)
        self.n = self.Xtr.shape[1]

    def test_one_label_per_test_row_and_memberships_sum_to_one(self):
        mask = np.zeros(self.n, bool); mask[:5] = True
        for fuzzy in (False, True):
            lab, mem = predict(self.Xtr, self.ytr, self.Xte, mask, 5, fuzzy)
            self.assertEqual(len(lab), len(self.Xte))
            self.assertEqual(mem.shape, (len(self.Xte), len(np.unique(self.ytr))))
            self.assertTrue(np.allclose(mem.sum(1), 1.0))

    def test_empty_mask_rejected(self):
        with self.assertRaises(ValueError):
            predict(self.Xtr, self.ytr, self.Xte, np.zeros(self.n, bool), 5, False)

    def test_k_larger_than_training_set_is_clamped(self):
        lab, _ = predict(self.Xtr[:7], self.ytr[:7], self.Xte, np.ones(self.n, bool), 99, True)
        self.assertEqual(len(lab), len(self.Xte))

    def test_fuzzy_membership_is_distance_weighted(self):
        Xtr = np.array([[0.0], [1.0], [10.0]]); ytr = np.array([0, 0, 1])
        Xte = np.array([[9.0]]); mask = np.array([True])
        _, crisp = predict(Xtr, ytr, Xte, mask, 3, False)
        _, fuzzy = predict(Xtr, ytr, Xte, mask, 3, True)
        self.assertAlmostEqual(crisp[0, 1], 1 / 3)          # equal votes
        self.assertGreater(fuzzy[0, 1], crisp[0, 1])        # near neighbour counts more


class TestFitness(unittest.TestCase):
    def test_float_in_unit_interval_crisp_and_fuzzy(self):
        Xtr, _, ytr, _, folds = standin_load_dataset(seed=1)
        for fuzzy in (False, True):
            v = fitness(np.ones(Xtr.shape[1], bool), 5, Xtr, ytr, folds, fuzzy)
            self.assertIsInstance(v, float); self.assertTrue(0.0 <= v <= 1.0)

    def test_empty_mask_scores_zero(self):
        Xtr, _, ytr, _, folds = standin_load_dataset(seed=1)
        self.assertEqual(fitness(np.zeros(Xtr.shape[1], bool), 5, Xtr, ytr, folds), 0.0)


class TestPSO(unittest.TestCase):
    N = 12

    def test_contract(self):
        check_optimizer_output(pso.run(make_toy_objective(self.N), self.N, 10, 0, pop_size=10), self.N, 10)

    def test_seed_reproducible(self):
        f = make_toy_objective(self.N)
        a, b = pso.run(f, self.N, 15, 3, pop_size=10), pso.run(f, self.N, 15, 3, pop_size=10)
        self.assertTrue((a[0] == b[0]).all() and a[1] == b[1]); self.assertEqual(a[2], b[2])

    def test_best_fitness_monotone(self):
        h = pso.run(make_toy_objective(self.N), self.N, 30, 1, pop_size=10)[2]
        b = [r["best_fitness"] for r in h]
        self.assertTrue(all(x <= y + 1e-12 for x, y in zip(b, b[1:])))

    def test_equal_evaluation_budget(self):
        h = pso.run(make_toy_objective(self.N), self.N, 12, 0, pop_size=8)[2]
        self.assertEqual(h[-1]["n_evals"], 8 * 12)

    def test_never_empty_mask(self):
        m, k, _ = pso.run(lambda m, k: 0.5, 3, 5, 0, pop_size=6)
        check_solution(m, k, 3)

    def test_beats_random_on_toy(self):
        n = 30; f = make_toy_objective(n, seed=1)
        rng = np.random.default_rng(0)
        rnd = np.mean([max(f(rng.random(n) < .5, 2 * rng.integers(16) + 1) for _ in range(20 * 40)) for _ in range(3)])
        p = np.mean([pso.run(f, n, 40, s)[2][-1]["best_fitness"] for s in range(3)])
        self.assertGreater(p, rnd)


if __name__ == "__main__":
    unittest.main()