"""Small, equal-effort hyper-parameter search for QIEA and GA (crisp KNN fitness).
Uses TUNING seeds (100+) with their own splits and only TRAIN-set CV fitness, so the
final evaluation seeds / test sets are never touched.  Writes results/tuning.json.
PSO is Person B's and is not tuned here."""
import argparse, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from shared.interfaces import bind_fitness
from person_a import qiea, ga
from person_a.components import get_components, fitness_for, load_data

ap = argparse.ArgumentParser()
ap.add_argument("--pop", type=int, default=20); ap.add_argument("--gens", type=int, default=20)
ap.add_argument("--tune_seeds", type=int, default=3)
a = ap.parse_args()
comp = get_components(); fit = fitness_for(comp, False)

QIEA_GRID = [dict(delta_theta=d * np.pi, p_mut=p) for d, p in itertools.product([0.02, 0.05, 0.08], [0.005, 0.02, 0.05])]
GA_GRID = [dict(mutation_rate=m, crossover_rate=c) for m, c in itertools.product([0.01, 0.03, 0.06], [0.9])]
data = {s: load_data(comp, "breast_cancer", 100 + s) for s in range(a.tune_seeds)}


def score(mod, params):
    vals = []
    for s, (Xtr, _, ytr, _, folds) in data.items():
        fn = bind_fitness(fit, Xtr, ytr, folds)
        vals.append(mod.run(fn, Xtr.shape[1], a.gens, 100 + s, pop_size=a.pop, **params)[2][-1]["best_fitness"])
    return float(np.mean(vals))


out = {}
for name, mod, grid in (("qiea", qiea, QIEA_GRID), ("ga", ga, GA_GRID)):
    res = []
    for p in grid:
        sc = score(mod, p); res.append({"params": {k: float(v) for k, v in p.items()}, "mean_train_fitness": sc})
        print(name, {k: round(float(v), 4) for k, v in p.items()}, f"{sc:.4f}", flush=True)
    best = max(res, key=lambda r: r["mean_train_fitness"])
    out[name] = {"best": best["params"], "grid": res}
    print(f"BEST {name}: {best}", flush=True)
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/tuning.json", "w"), indent=2)
