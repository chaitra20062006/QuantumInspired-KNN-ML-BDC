"""Person B: fitness function (the hub both other people call).

    fitness(mask, k, X, y, folds, fuzzy=False) -> float in [0, 1]

fitness = w1 * F1 + w2 * (1 - selected / total)      (w1 = 0.9, w2 = 0.1)

F1 is the mean over the supplied CV folds. Every fold trains on its own
training part only (scaling happens inside `predict`), so nothing leaks.
`folds` is a list of (train_idx, val_idx) pairs that index into X and y.
"""
import numpy as np
from sklearn.metrics import f1_score

from person_b.classifier import predict


def _f1(y_true, y_pred, pos_label):
    if len(np.unique(y_true)) <= 2 and pos_label is not None:
        return f1_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0)
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def fitness(mask, k, X, y, folds, fuzzy=False, w1=0.9, w2=0.1):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():                       # empty subset is invalid -> worst score
        return 0.0
    X, y = np.asarray(X), np.asarray(y)
    classes = np.unique(y)
    pos_label = classes[-1] if len(classes) == 2 else None   # larger label = "disease"

    scores = []
    for tr, va in folds:
        labels, _ = predict(X[tr], y[tr], X[va], mask, k, fuzzy)
        scores.append(_f1(y[va], labels, pos_label))
    f1 = float(np.mean(scores))

    value = w1 * f1 + w2 * (1.0 - mask.sum() / mask.size)
    return float(min(1.0, max(0.0, value)))