"""Quantum-Inspired Evolutionary Algorithm (QIEA) for joint feature + K search.

Each individual is a string of Q-bits. A Q-bit is (alpha, beta) with
alpha^2 + beta^2 = 1, stored as an angle theta: alpha = cos(theta),
beta = sin(theta), so P(bit = 1) = sin^2(theta).

Chromosome layout: [n_features feature bits | k_bits bits encoding K].

Steps per generation
  1. Observation (measurement): collapse every Q-bit into 0/1.
  2. Repair: guarantee at least one selected feature.
  3. Evaluate with the supplied fitness (serial, or Spark via evaluate_population).
  4. Track best-so-far.
  5. Rotation gate: rotate each worse individual's Q-bits toward the best-so-far bit.
  6. Quantum mutation: NOT gate (swap alpha/beta, theta -> pi/2 - theta) with small prob.
"""
from typing import Callable, List, Optional, Tuple
import numpy as np
from shared.interfaces import (K_BITS_DEFAULT, Population, decode_chromosome,
                               serial_evaluate)

THETA_MIN = 0.1                      # keeps P(1) within [~0.01, ~0.99]
THETA_MAX = np.pi / 2 - THETA_MIN


def init_qbits(pop_size: int, n_bits: int) -> np.ndarray:
    """Equal amplitudes: alpha = beta = 1/sqrt(2)  (theta = pi/4, P(1) = 0.5)."""
    return np.full((pop_size, n_bits), np.pi / 4)


def observe(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Collapse Q-bits to a binary population."""
    return rng.random(theta.shape) < np.sin(theta) ** 2


def repair(bits: np.ndarray, n_features: int, rng: np.random.Generator) -> np.ndarray:
    """Force at least one feature bit on (in place)."""
    for row in bits:
        if not row[:n_features].any():
            row[rng.integers(n_features)] = True
    return bits


def rotate(theta, bits, best_bits, fits, best_fit, delta):
    """Rotation gate: move Q-bits of worse individuals toward the best solution.
    Where the individual's bit differs from the best's bit, rotate theta
    toward 1 (best bit = 1) or toward 0 (best bit = 0)."""
    worse = np.asarray(fits) < best_fit
    differs = bits != best_bits[None, :]
    direction = np.where(best_bits[None, :], 1.0, -1.0)
    theta = theta + delta * direction * differs * worse[:, None]
    return np.clip(theta, THETA_MIN, THETA_MAX)


def quantum_mutation(theta, p_mut, rng):
    """NOT gate: swap alpha and beta (theta -> pi/2 - theta) on random Q-bits."""
    flip = rng.random(theta.shape) < p_mut
    return np.where(flip, np.pi / 2 - theta, theta)


def run(fitness_fn: Callable[[np.ndarray, int], float],
        n_features: int,
        generations: int,
        seed: int,
        pop_size: int = 20,
        k_bits: int = K_BITS_DEFAULT,
        delta_theta: float = 0.03 * np.pi,
        p_mut: float = 0.01,
        evaluate_population: Optional[Callable[[Population, Callable], List[float]]] = None,
        ) -> Tuple[np.ndarray, int, list]:
    """Optimizer interface: run(fitness_fn, n_features, generations, seed)
    -> best_mask, best_k, history.   Total evaluations = pop_size * generations."""
    rng = np.random.default_rng(seed)
    evaluate = evaluate_population or serial_evaluate
    n_bits = n_features + k_bits
    theta = init_qbits(pop_size, n_bits)

    best_bits, best_fit = None, -np.inf
    history, n_evals = [], 0

    for g in range(generations):
        bits = repair(observe(theta, rng), n_features, rng)
        population = [decode_chromosome(b, n_features) for b in bits]
        fits = evaluate(population, fitness_fn)
        n_evals += len(population)

        i = int(np.argmax(fits))
        if fits[i] > best_fit:                       # best-solution tracking
            best_fit, best_bits = float(fits[i]), bits[i].copy()

        history.append({"generation": g, "best_fitness": best_fit,
                        "mean_fitness": float(np.mean(fits)), "n_evals": n_evals})

        theta = rotate(theta, bits, best_bits, fits, best_fit, delta_theta)
        theta = quantum_mutation(theta, p_mut, rng)

    best_mask, best_k = decode_chromosome(best_bits, n_features)
    return best_mask, int(best_k), history
