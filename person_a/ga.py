"""Plain binary Genetic Algorithm baseline.

Same chromosome layout, same fitness and the SAME evaluation budget as QIEA
(pop_size * generations evaluations): every generation evaluates pop_size
offspring; the previous elite replaces the worst offspring using its cached fitness.
"""
from typing import Callable, List, Optional, Tuple
import numpy as np
from shared.interfaces import (K_BITS_DEFAULT, Population, decode_chromosome,
                               serial_evaluate)


def _repair(bits, n_features, rng):
    for row in bits:
        if not row[:n_features].any():
            row[rng.integers(n_features)] = True
    return bits


def _tournament(fits, rng, size=3):
    idx = rng.integers(len(fits), size=size)
    return idx[np.argmax(np.asarray(fits)[idx])]


def run(fitness_fn: Callable[[np.ndarray, int], float],
        n_features: int,
        generations: int,
        seed: int,
        pop_size: int = 20,
        k_bits: int = K_BITS_DEFAULT,
        crossover_rate: float = 0.9,
        mutation_rate: Optional[float] = None,
        evaluate_population: Optional[Callable[[Population, Callable], List[float]]] = None,
        ) -> Tuple[np.ndarray, int, list]:
    rng = np.random.default_rng(seed)
    evaluate = evaluate_population or serial_evaluate
    n_bits = n_features + k_bits
    mutation_rate = 1.0 / n_bits if mutation_rate is None else mutation_rate

    pop = _repair(rng.random((pop_size, n_bits)) < 0.5, n_features, rng)
    best_bits, best_fit = None, -np.inf
    history, n_evals = [], 0
    elite_bits, elite_fit = None, -np.inf

    for g in range(generations):
        if g > 0:                                     # build offspring
            fits_prev = fits
            children = np.empty_like(pop)
            for c in range(pop_size):
                p1, p2 = pop[_tournament(fits_prev, rng)], pop[_tournament(fits_prev, rng)]
                if rng.random() < crossover_rate:     # uniform crossover
                    children[c] = np.where(rng.random(n_bits) < 0.5, p1, p2)
                else:
                    children[c] = p1
            flip = rng.random(children.shape) < mutation_rate
            pop = _repair(np.logical_xor(children, flip), n_features, rng)

        population = [decode_chromosome(b, n_features) for b in pop]
        fits = list(evaluate(population, fitness_fn))
        n_evals += len(population)

        if g > 0:                                     # elitism with cached fitness
            worst = int(np.argmin(fits))
            pop[worst], fits[worst] = elite_bits, elite_fit

        i = int(np.argmax(fits))
        elite_bits, elite_fit = pop[i].copy(), float(fits[i])
        if elite_fit > best_fit:
            best_fit, best_bits = elite_fit, elite_bits.copy()

        history.append({"generation": g, "best_fitness": best_fit,
                        "mean_fitness": float(np.mean(fits)), "n_evals": n_evals})

    best_mask, best_k = decode_chromosome(best_bits, n_features)
    return best_mask, int(best_k), history
