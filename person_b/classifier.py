"""Person B: KNN / fuzzy KNN classifier.

Interface (agreed):
    predict(X_tr, y_tr, X_te, mask, k, fuzzy) -> labels, memberships

* mask  : boolean array, True = feature used (at least one True)
* k     : odd positive int (clamped to the number of training rows)
* fuzzy : False -> crisp majority vote; True -> distance-weighted fuzzy vote
* labels      : (n_test,) predicted class per test row
* memberships : (n_test, n_classes) rows sum to 1; column order = np.unique(y_tr)
"""
import numpy as np
from sklearn.metrics import pairwise_distances


def _scale(A, B):
    """Standardize using TRAINING statistics only (no leakage)."""
    mu, sd = A.mean(axis=0), A.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (A - mu) / sd, (B - mu) / sd


def predict(X_tr, y_tr, X_te, mask, k, fuzzy=False, m=2.0, eps=1e-12):
    X_tr = np.asarray(X_tr, dtype=float)
    X_te = np.asarray(X_te, dtype=float)
    y_tr = np.asarray(y_tr)
    mask = np.asarray(mask, dtype=bool)

    if mask.shape != (X_tr.shape[1],):
        raise ValueError("mask must be a boolean array with one entry per feature")
    if not mask.any():
        raise ValueError("mask must select at least one feature")
    if m <= 1:
        raise ValueError("fuzzifier m must be > 1")

    A, B = _scale(X_tr[:, mask], X_te[:, mask])
    classes = np.unique(y_tr)
    k = max(1, min(int(k), len(A)))

    D = pairwise_distances(B, A)                      # Euclidean, (n_test, n_train)
    if k < len(A):
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    else:
        idx = np.tile(np.arange(len(A)), (len(B), 1))
    d = np.take_along_axis(D, idx, axis=1)            # distances to the k neighbours
    neigh = y_tr[idx]                                 # labels of the k neighbours

    if fuzzy:
        w = 1.0 / np.maximum(d, eps) ** (2.0 / (m - 1.0))   # w_i = 1 / d_i^(2/(m-1))
    else:
        w = np.ones_like(d)                                   # equal votes = crisp KNN

    # membership(c) = sum(w_i * u_ic) / sum(w_i), with u_ic = 1 if neighbour i is class c
    mem = np.stack([(w * (neigh == c)).sum(axis=1) for c in classes], axis=1)
    mem = mem / w.sum(axis=1, keepdims=True)
    return classes[np.argmax(mem, axis=1)], mem