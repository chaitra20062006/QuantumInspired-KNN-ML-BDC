"""Person C Tests: Data Loading, Leakage Prevention, Spark Evaluator, and LSH.
Run with:
    venv\\Scripts\\python.exe -m unittest tests.test_person_c -v
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from person_c.data_loader import load_dataset
from person_c.spark_evaluator import evaluate_population
from person_c.lsh import LSHApproximateKNN


class TestDataLoader(unittest.TestCase):
    def test_load_dataset_contract(self):
        X_tr, X_te, y_tr, y_te, folds = load_dataset("pima", seed=42)
        self.assertGreater(len(X_tr), 0)
        self.assertGreater(len(X_te), 0)
        self.assertEqual(len(X_tr), len(y_tr))
        self.assertEqual(len(X_te), len(y_te))
        self.assertEqual(len(folds), 5)

    def test_no_data_leakage_in_folds(self):
        X_tr, X_te, y_tr, y_te, folds = load_dataset("pima", seed=42)
        # Check that train folds and val folds are strictly disjoint
        for tr_idx, val_idx in folds:
            overlap = set(tr_idx).intersection(set(val_idx))
            self.assertEqual(len(overlap), 0, "CV folds leak validation indices into training!")

    def test_features_are_standardized(self):
        X_tr, _, _, _, _ = load_dataset("pima", seed=42)
        means = np.mean(X_tr, axis=0)
        stds = np.std(X_tr, axis=0)
        self.assertTrue(np.allclose(means, 0, atol=1e-1))
        self.assertTrue(np.allclose(stds, 1, atol=1e-1))


class TestSparkEvaluator(unittest.TestCase):
    def test_evaluator_length_and_range(self):
        n_features = 8
        pop = [
            (np.ones(n_features, dtype=bool), 3),
            (np.ones(n_features, dtype=bool), 5)
        ]
        
        def dummy_fn(mask, k):
            return float(mask.sum() / len(mask))

        scores = evaluate_population(pop, dummy_fn)
        self.assertEqual(len(scores), len(pop))
        for s in scores:
            self.assertTrue(0.0 <= s <= 1.0)


class TestLSH(unittest.TestCase):
    def test_lsh_predictions(self):
        rng = np.random.default_rng(42)
        X_tr = rng.normal(size=(50, 6))
        y_tr = rng.integers(0, 2, size=50)
        X_te = rng.normal(size=(10, 6))

        lsh = LSHApproximateKNN(n_bits=4, n_tables=2)
        lsh.fit(X_tr, y_tr)
        preds = lsh.predict(X_te, k=3)
        self.assertEqual(len(preds), len(X_te))


if __name__ == "__main__":
    unittest.main()