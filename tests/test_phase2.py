import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from shared.interfaces import check_optimizer_output
from person_a import pso_standin
from person_a.components import get_components, fitness_for
from person_a.metrics import classification_metrics, generations_to_reach
from person_a.standins import make_toy_objective, standin_fitness_pred, standin_load_dataset, standin_predict
from person_a.baseline_knn import baseline_knn
from person_a.stats import holm, paired_tests


class TestMetricsStats(unittest.TestCase):
    def test_metrics(self):
        m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], np.array([[.9, .1], [.4, .6], [.2, .8], [.1, .9]]), 3)
        self.assertAlmostEqual(m["accuracy"], .75); self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["precision"], 2 / 3); self.assertAlmostEqual(m["auc"], 1.0); self.assertEqual(m["n_features"], 3)

    def test_gens_to_reach(self):
        h = [{"best_fitness": v} for v in (.5, .8, .99, 1.0, 1.0)]
        self.assertEqual(generations_to_reach(h), 2)

    def test_paired_tests(self):
        a = np.arange(10) + 1.0
        r = paired_tests(a + 1, a)
        self.assertEqual(r["wins"], 10); self.assertAlmostEqual(r["mean_diff"], 1.0); self.assertLess(r["wilcoxon_p"], .01)
        self.assertEqual(paired_tests(a, a)["wilcoxon_p"], 1.0)

    def test_holm(self):
        self.assertEqual(holm([.01, .04, .03]), [.03, .06, .06])


class TestStandinInterfaces(unittest.TestCase):
    def setUp(self):
        self.Xtr, self.Xte, self.ytr, self.yte, self.folds = standin_load_dataset(seed=1)

    def test_predict_contract(self):
        mask = np.zeros(self.Xtr.shape[1], bool); mask[:5] = True
        for fuzzy in (False, True):
            lab, mem = standin_predict(self.Xtr, self.ytr, self.Xte, mask, 5, fuzzy)
            self.assertEqual(len(lab), len(self.Xte)); self.assertTrue(np.allclose(mem.sum(1), 1))

    def test_fitness_contract(self):
        for fuzzy in (False, True):
            v = standin_fitness_pred(np.ones(self.Xtr.shape[1], bool), 5, self.Xtr, self.ytr, self.folds, fuzzy)
            self.assertIsInstance(v, float); self.assertTrue(0 <= v <= 1)

    def test_matches_sklearn_baseline(self):
        lab, _ = standin_predict(self.Xtr, self.ytr, self.Xte, np.ones(self.Xtr.shape[1], bool), 5, False)
        self.assertAlmostEqual((lab == self.yte).mean(), baseline_knn(self.Xtr, self.ytr, self.Xte, self.yte)["accuracy"], delta=0.02)

    def test_pso_contract(self):
        check_optimizer_output(pso_standin.run(make_toy_objective(12), 12, 8, 0, pop_size=8), 12, 8)

    def test_components_default_standins(self):
        c = get_components(); self.assertEqual(c["fitness_source"], "stand-in")
        self.assertTrue(callable(fitness_for(c, True)))


if __name__ == "__main__":
    unittest.main()
