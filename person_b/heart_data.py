"""Person B: temporary Heart Disease (Cleveland) loader, used until Person C's pipeline is merged.

    load_dataset(name="heart", seed=42) -> X_train, X_test, y_train, y_test, folds

Expects the UCI file  data/processed.cleveland.data  (14 comma-separated columns, no header,
'?' = missing, last column = diagnosis 0..4).  Target is binarised: 0 = healthy, 1 = disease.
No leakage: the median used to fill missing values is computed on the TRAINING rows only.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "processed.cleveland.data")


def feature_names():
    return list(FEATURES)


def _read():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Put the UCI file here: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, header=None, names=FEATURES + ["target"], na_values="?")
    X = df[FEATURES].astype(float).to_numpy()
    y = (df["target"].astype(float).to_numpy() > 0).astype(int)
    return X, y


def load_dataset(name="heart", seed=42, n_splits=5, test_size=0.25):
    X, y = _read()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    med = np.nanmedian(X_tr, axis=0)                       # training statistics only
    X_tr = np.where(np.isnan(X_tr), med, X_tr)
    X_te = np.where(np.isnan(X_te), med, X_te)
    folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X_tr, y_tr))
    return X_tr, X_te, y_tr, y_te, folds