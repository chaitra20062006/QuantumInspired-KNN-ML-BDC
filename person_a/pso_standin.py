"""STAND-IN for Person B's PSO baseline (binary PSO, Kennedy & Eberhart 1997 style),
so the full 5-method pipeline can run now. Same chromosome, fitness and budget as QIEA.
Replace with B's real PSO via components.py (env var QIEA_PSO)."""
import numpy as np
from shared.interfaces import K_BITS_DEFAULT, decode_chromosome, serial_evaluate


def run(fitness_fn, n_features, generations, seed, pop_size=20, k_bits=K_BITS_DEFAULT,
        w=0.7, c1=1.5, c2=1.5, vmax=4.0, evaluate_population=None):
    rng = np.random.default_rng(seed)
    evaluate = evaluate_population or serial_evaluate
    nb = n_features + k_bits

    def repair(x):
        for r in x:
            if not r[:n_features].any():
                r[rng.integers(n_features)] = True
        return x

    x = repair(rng.random((pop_size, nb)) < 0.5)
    v = rng.uniform(-1, 1, (pop_size, nb))
    pbest, pfit = x.copy(), np.full(pop_size, -np.inf)
    gbest, gfit = None, -np.inf
    hist, n_evals = [], 0
    for g in range(generations):
        fits = np.array(evaluate([decode_chromosome(b, n_features) for b in x], fitness_fn))
        n_evals += pop_size
        better = fits > pfit
        pbest[better], pfit[better] = x[better], fits[better]
        i = int(np.argmax(fits))
        if fits[i] > gfit:
            gfit, gbest = float(fits[i]), x[i].copy()
        hist.append({"generation": g, "best_fitness": gfit, "mean_fitness": float(fits.mean()), "n_evals": n_evals})
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = np.clip(w * v + c1 * r1 * (pbest.astype(float) - x) + c2 * r2 * (gbest.astype(float) - x), -vmax, vmax)
        x = repair(rng.random(x.shape) < 1 / (1 + np.exp(-v)))
    m, k = decode_chromosome(gbest, n_features)
    return m, int(k), hist
