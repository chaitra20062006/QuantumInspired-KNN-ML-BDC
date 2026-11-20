"""Shared interfaces for QIEA-KNN (ONE file, agreed by all three people).

Solution format
    mask : 1-D boolean numpy array (True = feature selected, at least one True)
    k    : odd positive int
    population : list of (mask, k) pairs

Signatures
    Data        load_dataset(name) -> X_train, X_test, y_train, y_test, folds        (C)
    Classifier  predict(X_tr, y_tr, X_te, mask, k, fuzzy) -> labels, memberships    (B)
    Fitness     fitness(mask, k, X, y, folds) -> float in [0, 1]                    (B)
    Optimizer   run(fitness_fn, n_features, generations, seed)
                    -> best_mask, best_k, history                                   (A: QIEA, GA | B: PSO)
    Evaluator   evaluate_population(population, fitness_fn) -> list[float]          (C)

Convention used by optimizers: ``fitness_fn`` is the fitness with (X, y, folds)
already bound, i.e. ``fitness_fn(mask, k) -> float`` (see ``bind_fitness``).
``history`` is a list of dicts, one per generation:
    {"generation", "best_fitness", "mean_fitness", "n_evals"}   (best_fitness = best so far)

NOTE: decode_k below is a STAND-IN. Person B owns the real K encoding; the
default here (4 bits -> odd K in 1..31) is only a proposal to agree on.
"""
from typing import Callable, List, Sequence, Tuple
import numpy as np

Population = List[Tuple[np.ndarray, int]]
FitnessFn = Callable[[np.ndarray, int], float]

K_BITS_DEFAULT = 4


def decode_k(bits: Sequence[bool]) -> int:
    """K bits (MSB first) -> odd K = 2*value + 1."""
    value = 0
    for b in bits:
        value = (value << 1) | int(bool(b))
    return 2 * value + 1


def encode_k(k: int, k_bits: int = K_BITS_DEFAULT) -> np.ndarray:
    if k % 2 == 0 or k < 1 or (k - 1) // 2 >= 2 ** k_bits:
        raise ValueError(f"k={k} not representable with {k_bits} bits")
    v = (k - 1) // 2
    return np.array([(v >> (k_bits - 1 - i)) & 1 for i in range(k_bits)], dtype=bool)


def decode_chromosome(bits: np.ndarray, n_features: int) -> Tuple[np.ndarray, int]:
    """Binary string (features then K bits) -> (mask, k)."""
    return np.asarray(bits[:n_features], dtype=bool).copy(), decode_k(bits[n_features:])


def bind_fitness(fitness, X, y, folds) -> FitnessFn:
    """Turn fitness(mask,k,X,y,folds) into fitness_fn(mask,k)."""
    return lambda mask, k: fitness(mask, k, X, y, folds)


def serial_evaluate(population: Population, fitness_fn: FitnessFn) -> List[float]:
    """Default evaluator; Person C's Spark evaluator is a drop-in replacement."""
    return [float(fitness_fn(m, k)) for m, k in population]


def check_solution(mask, k, n_features) -> None:
    """Contract check for the agreed solution format."""
    mask = np.asarray(mask)
    assert mask.dtype == bool and mask.shape == (n_features,), "mask must be bool array of length n_features"
    assert mask.any(), "mask must select at least one feature"
    assert isinstance(k, (int, np.integer)) and k >= 1 and k % 2 == 1, "k must be an odd positive int"


def check_optimizer_output(out, n_features, generations) -> None:
    best_mask, best_k, history = out
    check_solution(best_mask, best_k, n_features)
    assert len(history) == generations, "history needs one entry per generation"
    for key in ("generation", "best_fitness", "mean_fitness", "n_evals"):
        assert key in history[0], f"history entries need '{key}'"
