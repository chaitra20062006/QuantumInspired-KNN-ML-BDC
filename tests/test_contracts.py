"""Contract + behaviour tests. Run:  python -m unittest discover -s tests -v   (pytest also works)."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from shared.interfaces import (check_optimizer_output, check_solution, decode_k, encode_k,
                               bind_fitness, serial_evaluate)
from person_a import qiea, ga
from person_a.baseline_knn import baseline_knn
from person_a.standins import make_toy_objective, standin_fitness, standin_load_dataset

N = 12


class TestInterfaces(unittest.TestCase):
    def test_k_roundtrip_and_odd(self):
        for k in range(1, 32, 2):
            self.assertEqual(decode_k(encode_k(k)), k)
        for v in range(16):
            self.assertEqual(decode_k([(v >> i) & 1 for i in (3, 2, 1, 0)]) % 2, 1)

    def test_serial_evaluate_len(self):
        pop = [(np.ones(N, bool), 3)] * 5
        self.assertEqual(len(serial_evaluate(pop, make_toy_objective(N))), 5)


class TestOptimizers(unittest.TestCase):
    def test_contract(self):
        f = make_toy_objective(N)
        for mod in (qiea, ga):
            out = mod.run(f, N, 10, 0, pop_size=10)
            check_optimizer_output(out, N, 10)

    def test_seed_reproducible(self):
        f = make_toy_objective(N)
        for mod in (qiea, ga):
            a, b = mod.run(f, N, 15, 3, pop_size=10), mod.run(f, N, 15, 3, pop_size=10)
            self.assertTrue((a[0] == b[0]).all() and a[1] == b[1])
            self.assertEqual(a[2], b[2])

    def test_best_fitness_monotone(self):
        f = make_toy_objective(N)
        for mod in (qiea, ga):
            h = mod.run(f, N, 30, 1, pop_size=10)[2]
            b = [r["best_fitness"] for r in h]
            self.assertTrue(all(x <= y + 1e-12 for x, y in zip(b, b[1:])))

    def test_equal_evaluation_budget(self):
        f = make_toy_objective(N)
        hq, hg = qiea.run(f, N, 12, 0, pop_size=8)[2], ga.run(f, N, 12, 0, pop_size=8)[2]
        self.assertEqual(hq[-1]["n_evals"], hg[-1]["n_evals"], 8 * 12)

    def test_qiea_beats_random_on_toy(self):
        n = 30
        f = make_toy_objective(n, seed=1)
        rng = np.random.default_rng(0)
        random_best = np.mean([max(f(rng.random(n) < .5, 2 * rng.integers(16) + 1) for _ in range(20 * 40))
                               for _ in range(3)])
        q = np.mean([qiea.run(f, n, 40, s)[2][-1]["best_fitness"] for s in range(3)])
        self.assertGreater(q, random_best)

    def test_never_empty_mask(self):
        f = lambda m, k: 0.5   # flat fitness
        for mod in (qiea, ga):
            m, k, _ = mod.run(f, 3, 5, 0, pop_size=6)
            check_solution(m, k, 3)


class TestStandins(unittest.TestCase):
    def test_fitness_float_in_unit_interval(self):
        Xtr, _, ytr, _, folds = standin_load_dataset()
        v = standin_fitness(np.ones(Xtr.shape[1], bool), 5, Xtr, ytr, folds)
        self.assertIsInstance(v, float); self.assertTrue(0.0 <= v <= 1.0)

    def test_qiea_on_real_standin(self):
        Xtr, Xte, ytr, yte, folds = standin_load_dataset()
        fn = bind_fitness(standin_fitness, Xtr, ytr, folds)
        out = qiea.run(fn, Xtr.shape[1], 3, 0, pop_size=6)
        check_optimizer_output(out, Xtr.shape[1], 3)

    def test_baseline_knn(self):
        Xtr, Xte, ytr, yte, _ = standin_load_dataset()
        r = baseline_knn(Xtr, ytr, Xte, yte)
        self.assertEqual(r["k"], 5); self.assertEqual(r["n_features"], Xtr.shape[1])
        self.assertGreater(r["accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
