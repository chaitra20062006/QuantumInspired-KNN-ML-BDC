"""Metrics toolkit (used by all three people)."""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def classification_metrics(y_true, y_pred, scores=None, n_features=None) -> dict:
    """accuracy, precision, recall, F1 (+ AUC if class scores/memberships given, + #features).
    Binary: positive class = larger label; `scores` = (n, n_classes) memberships or 1-D positive score."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    binary = len(np.unique(y_true)) == 2
    avg = "binary" if binary else "weighted"
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
           "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
           "f1": f1_score(y_true, y_pred, average=avg, zero_division=0), "auc": np.nan}
    if scores is not None:
        s = np.asarray(scores)
        try:
            if binary:
                out["auc"] = roc_auc_score(y_true, s[:, -1] if s.ndim == 2 else s)
            else:
                out["auc"] = roc_auc_score(y_true, s, multi_class="ovr", average="weighted")
        except ValueError:
            pass
    if n_features is not None:
        out["n_features"] = int(n_features)
    return out


def generations_to_reach(history, frac: float = 0.99) -> int:
    """Convergence speed: first generation whose best-so-far fitness reaches
    `frac` of the run's final best fitness (lower = faster)."""
    best = np.array([h["best_fitness"] for h in history])
    return int(np.argmax(best >= frac * best[-1]))


def history_matrix(histories, key="best_fitness") -> np.ndarray:
    """list of histories -> (runs, generations) array for convergence curves."""
    return np.array([[h[key] for h in hist] for hist in histories])
