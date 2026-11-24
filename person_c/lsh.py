"""Person C: Locality Sensitive Hashing (LSH) for Approximate Nearest Neighbor Search.

Light task (1 pt): Alternative to exhaustive pairwise distance KNN on large datasets.
"""
from typing import Tuple
import numpy as np


class LSHApproximateKNN:
    """Random projection-based LSH classifier."""

    def __init__(self, n_bits: int = 8, n_tables: int = 3, seed: int = 42):
        self.n_bits = n_bits
        self.n_tables = n_tables
        self.rng = np.random.default_rng(seed)
        self.planes = None
        self.tables = None
        self.X_tr = None
        self.y_tr = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_tr = np.asarray(X, dtype=float)
        self.y_tr = np.asarray(y)
        n_features = self.X_tr.shape[1]

        # Generate random hyperplanes: (n_tables, n_bits, n_features)
        self.planes = [
            self.rng.normal(size=(self.n_bits, n_features))
            for _ in range(self.n_tables)
        ]

        # Build hash buckets
        self.tables = [{} for _ in range(self.n_tables)]
        for idx, x in enumerate(self.X_tr):
            for t in range(self.n_tables):
                h = self._hash(x, self.planes[t])
                self.tables[t].setdefault(h, []).append(idx)
        return self

    def _hash(self, x: np.ndarray, plane: np.ndarray) -> int:
        bits = (np.dot(plane, x) >= 0).astype(int)
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return val

    def predict(self, X_te: np.ndarray, k: int = 5) -> np.ndarray:
        X_te = np.asarray(X_te, dtype=float)
        preds = []
        for x in X_te:
            candidates = set()
            for t in range(self.n_tables):
                h = self._hash(x, self.planes[t])
                candidates.update(self.tables[t].get(h, []))

            if len(candidates) < k:
                # Fall back to random selection or all train samples if buckets are empty
                cand_indices = np.arange(len(self.X_tr))
            else:
                cand_indices = np.array(list(candidates))

            # Compute distances only against candidates
            cand_X = self.X_tr[cand_indices]
            dists = np.linalg.norm(cand_X - x, axis=1)
            nearest_k = cand_indices[np.argsort(dists)[:k]]
            cand_labels = self.y_tr[nearest_k]

            # Majority vote
            unique, counts = np.unique(cand_labels, return_counts=True)
            preds.append(unique[np.argmax(counts)])

        return np.array(preds)


def lsh_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, mask: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Drop-in matching the team's classifier interface."""
    mask = np.asarray(mask, dtype=bool)
    model = LSHApproximateKNN(n_bits=6, n_tables=3)
    model.fit(X_tr[:, mask], y_tr)
    labels = model.predict(X_te[:, mask], k=k)
    return labels, np.zeros((len(labels), len(np.unique(y_tr))))