"""Integration layer: use the REAL components of B and C when available, else stand-ins.

Point to a real component with an env var "package.module:attribute":
  QIEA_LOAD_DATASET  C: load_dataset(name[, seed]) -> X_train, X_test, y_train, y_test, folds
  QIEA_PREDICT       B: predict(X_tr, y_tr, X_te, mask, k, fuzzy) -> labels, memberships
  QIEA_FITNESS       B: fitness(mask, k, X, y, folds[, fuzzy]) -> float
  QIEA_EVALUATOR     C: evaluate_population(population, fitness_fn) -> list[float]
  QIEA_PSO           B: run(fitness_fn, n_features, generations, seed) -> best_mask, best_k, history
Nothing set -> stand-ins. get_components() reports the source of each piece.
"""
import importlib, inspect, os
from functools import partial
from shared.interfaces import serial_evaluate
from . import pso_standin
from .standins import standin_fitness_pred, standin_load_dataset, standin_predict


def _load(env, fallback):
    spec = os.environ.get(env)
    if not spec:
        return fallback, "stand-in"
    mod, attr = spec.split(":")
    return getattr(importlib.import_module(mod), attr), spec


def _load_dataset_standin(name="breast_cancer", seed=42):
    return standin_load_dataset(seed=seed)


def get_components() -> dict:
    c = {}
    for key, env, fb in [("load_dataset", "QIEA_LOAD_DATASET", _load_dataset_standin),
                         ("predict", "QIEA_PREDICT", standin_predict),
                         ("fitness", "QIEA_FITNESS", standin_fitness_pred),
                         ("evaluate_population", "QIEA_EVALUATOR", serial_evaluate),
                         ("pso_run", "QIEA_PSO", pso_standin.run)]:
        c[key], c[key + "_source"] = _load(env, fb)
    return c


def load_data(comp, name, seed):
    fn = comp["load_dataset"]
    return fn(name, seed=seed) if "seed" in inspect.signature(fn).parameters else fn(name)


def fitness_for(comp, fuzzy: bool):
    """Return fitness(mask,k,X,y,folds) with the crisp/fuzzy switch applied."""
    fn = comp["fitness"]
    if "fuzzy" in inspect.signature(fn).parameters:
        return partial(fn, fuzzy=fuzzy)
    if fuzzy:
        raise RuntimeError("The supplied fitness has no `fuzzy` argument; ask Person B to add it.")
    return fn
