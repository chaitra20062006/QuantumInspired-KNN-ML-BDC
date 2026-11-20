"""Baseline KNN: all features, K = 5, Euclidean distance (the reference point)."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def baseline_knn(X_train, y_train, X_test, y_test, k: int = 5) -> dict:
    """Fit on train, score on held-out test. Scaler is fit on train only (no leakage)."""
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, metric="euclidean"))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    avg = "binary" if len(np.unique(y_train)) == 2 else "weighted"
    return {"accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, average=avg, zero_division=0),
            "recall": recall_score(y_test, pred, average=avg, zero_division=0),
            "f1": f1_score(y_test, pred, average=avg, zero_division=0),
            "n_features": int(X_train.shape[1]), "k": k}
