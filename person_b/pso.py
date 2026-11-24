"""Person B: binary PSO baseline (Kennedy & Eberhart, 1997).

Same interface, chromosome layout, fitness and evaluation budget as QIEA and GA:
    run(fitness_fn, n_features, generations, seed) -> best_mask, best_k, history
Chromosome = [n_features feature bits | k_bits bits for K]. Budget = pop_size * generations.
"""
from typing import Callable, List, Optional, Tuple

import numpy as np

from shared.interfaces import K_BITS_DEFAULT, Population, decode_chromosome, serial_evaluate


def _repair(bits, n_features, rng):
    """Guarantee at least one selected feature (in place)."""
    for row in bits:
        if not row[:n_features].any():
            row[rng.integers(n_features)] = True
    return bits


def run(fitness_fn: Callable[[np.ndarray, int], float],
        n_features: int,
        generations: int,
        seed: int,
        pop_size: int = 20,
        k_bits: int = K_BITS_DEFAULT,
        w_start: float = 0.9,
        w_end: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        vmax: float = 4.0,
        evaluate_population: Optional[Callable[[Population, Callable], List[float]]] = None,
        ) -> Tuple[np.ndarray, int, list]:
    rng = np.random.default_rng(seed)
    evaluate = evaluate_population or serial_evaluate
    n_bits = n_features + k_bits

    x = _repair(rng.random((pop_size, n_bits)) < 0.5, n_features, rng)   # positions (0/1)
    v = rng.uniform(-vmax / 2, vmax / 2, (pop_size, n_bits))              # velocities
    pbest, pbest_fit = x.copy(), np.full(pop_size, -np.inf)
    gbest, gbest_fit = None, -np.inf
    history, n_evals = [], 0

    for g in range(generations):
        population = [decode_chromosome(b, n_features) for b in x]
        fits = np.asarray(evaluate(population, fitness_fn), dtype=float)
        n_evals += len(population)

        better = fits > pbest_fit                       # personal bests
        pbest[better], pbest_fit[better] = x[better], fits[better]
        i = int(np.argmax(fits))
        if fits[i] > gbest_fit:                         # global best
            gbest_fit, gbest = float(fits[i]), x[i].copy()

        history.append({"generation": g, "best_fitness": gbest_fit,
                        "mean_fitness": float(fits.mean()), "n_evals": n_evals})

        w = w_start - (w_start - w_end) * (g / max(1, generations - 1))   # decaying inertia
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = w * v + c1 * r1 * (pbest.astype(float) - x) + c2 * r2 * (gbest.astype(float) - x)
        v = np.clip(v, -vmax, vmax)
        x = _repair(rng.random(x.shape) < 1.0 / (1.0 + np.exp(-v)), n_features, rng)  # sigmoid -> bit

    best_mask, best_k = decode_chromosome(gbest, n_features)
    return best_mask, int(best_k), history