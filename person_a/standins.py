"""Stand-ins Person A uses until B (fitness, classifier) and C (data) deliver."""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_toy_objective(n_features: int, seed: int = 0, k_target: int = 7):
    """Toy fitness_fn(mask, k) in [0,1]: reward matching a hidden mask and K."""
    rng = np.random.default_rng(seed)
    target = rng.random(n_features) < 0.4
    target[0] = True

    def fn(mask, k):
        match = np.mean(np.asarray(mask) == target)
        k_score = 1.0 - min(abs(k - k_target), 30) / 30
        return float(0.8 * match + 0.2 * k_score)
    fn.target = target
    return fn


def standin_fitness(mask, k, X, y, folds, w1=0.9, w2=0.1):
    """Stand-in for B's fitness: w1*F1 + w2*(1 - selected/total), leakage-safe
    (scaler fit inside each CV training fold via a pipeline)."""
    mask = np.asarray(mask, dtype=bool)
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=int(k)))
    f1 = cross_val_score(model, X[:, mask], y, cv=folds, scoring="f1_weighted").mean()
    return float(w1 * f1 + w2 * (1 - mask.sum() / mask.size))


def standin_load_dataset(seed: int = 42, n_splits: int = 5):
    """Stand-in for C's load_dataset: Breast Cancer Wisconsin (built into sklearn).
    Returns X_train, X_test, y_train, y_test, folds (CV splits over the TRAIN set)."""
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X_tr, y_tr))
    return X_tr, X_te, y_tr, y_te, folds


# ---------------------------------------------------------------------------
# Phase 2 stand-ins that follow the AGREED interfaces exactly (B's classifier
# `predict` and B's `fitness`). Replace with the real versions via components.py.
# ---------------------------------------------------------------------------
from sklearn.metrics import f1_score as _f1, pairwise_distances as _pd


def standin_predict(X_tr, y_tr, X_te, mask, k, fuzzy=False, m=2.0):
    """predict(X_tr, y_tr, X_te, mask, k, fuzzy) -> labels, memberships.
    Crisp KNN: memberships = neighbour vote fractions.
    Fuzzy KNN: neighbour weight w = 1 / d^(2/(m-1)); membership = weighted vote share."""
    mask = np.asarray(mask, dtype=bool)
    A, B = X_tr[:, mask], X_te[:, mask]
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    A, B = (A - mu) / sd, (B - mu) / sd              # scaler fit on training data only
    classes = np.unique(y_tr)
    k = max(1, min(int(k), len(A)))
    D = _pd(B, A)
    idx = np.argpartition(D, k - 1, axis=1)[:, :k] if k < len(A) else np.tile(np.arange(len(A)), (len(B), 1))
    d = np.take_along_axis(D, idx, 1)
    lab = y_tr[idx]
    w = 1.0 / np.maximum(d, 1e-12) ** (2.0 / (m - 1.0)) if fuzzy else np.ones_like(d)
    mem = np.stack([(w * (lab == c)).sum(1) for c in classes], 1) / w.sum(1, keepdims=True)
    return classes[np.argmax(mem, 1)], mem


def _f1_auto(y_true, y_pred):
    return _f1(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "weighted", zero_division=0)


def standin_fitness_pred(mask, k, X, y, folds, fuzzy=False, w1=0.9, w2=0.1):
    """fitness(mask, k, X, y, folds) -> float in [0,1] = w1*F1 + w2*(1 - selected/total),
    mean over the supplied folds (indices refer to X, y)."""
    mask = np.asarray(mask, dtype=bool)
    f1 = np.mean([_f1_auto(y[te], standin_predict(X[tr], y[tr], X[te], mask, k, fuzzy)[0]) for tr, te in folds])
    return float(w1 * f1 + w2 * (1 - mask.sum() / mask.size))
