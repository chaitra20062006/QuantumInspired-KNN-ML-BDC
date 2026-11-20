"""Phase 1 demo: QIEA vs GA vs baseline KNN on Breast Cancer Wisconsin (stand-in data/fitness).
Usage: python scripts/run_phase1_demo.py [--seeds 5] [--pop 20] [--gens 20]
Writes results/phase1_demo_results.csv and results/phase1_convergence.png
"""
import argparse, csv, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from shared.interfaces import bind_fitness
from person_a import qiea, ga
from person_a.baseline_knn import baseline_knn
from person_a.standins import standin_fitness, standin_load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--pop", type=int, default=20)
ap.add_argument("--gens", type=int, default=20)
a = ap.parse_args()

Xtr, Xte, ytr, yte, folds = standin_load_dataset()
n = Xtr.shape[1]
fn = bind_fitness(standin_fitness, Xtr, ytr, folds)
rows, curves = [], {"QIEA": [], "GA": []}
base = baseline_knn(Xtr, ytr, Xte, yte)
print(f"Baseline KNN (all {n} features, K=5): test F1 = {base['f1']:.4f}")

for seed in range(a.seeds):
    for name, mod in (("QIEA", qiea), ("GA", ga)):
        t = time.time()
        mask, k, hist = mod.run(fn, n, a.gens, seed, pop_size=a.pop)
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(Xtr[:, mask], ytr)
        f1 = f1_score(yte, model.predict(Xte[:, mask]))
        curves[name].append([h["best_fitness"] for h in hist])
        rows.append({"method": name, "seed": seed, "train_cv_fitness": hist[-1]["best_fitness"],
                     "test_f1": f1, "n_features": int(mask.sum()), "k": k,
                     "n_evals": hist[-1]["n_evals"], "seconds": round(time.time() - t, 2)})
        print(f"{name:5s} seed {seed}: fitness={hist[-1]['best_fitness']:.4f} test_F1={f1:.4f} "
              f"features={int(mask.sum())}/{n} K={k}")

os.makedirs("results", exist_ok=True)
with open("results/phase1_demo_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

print("\nMean over seeds:")
for name in ("QIEA", "GA"):
    r = [x for x in rows if x["method"] == name]
    print(f"{name:5s} fitness={np.mean([x['train_cv_fitness'] for x in r]):.4f} "
          f"test_F1={np.mean([x['test_f1'] for x in r]):.4f} features={np.mean([x['n_features'] for x in r]):.1f}")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
for name, c in curves.items():
    c = np.array(c); x = np.arange(c.shape[1])
    plt.plot(x, c.mean(0), label=name); plt.fill_between(x, c.mean(0) - c.std(0), c.mean(0) + c.std(0), alpha=.2)
plt.xlabel("Generation"); plt.ylabel("Best-so-far fitness"); plt.legend()
plt.title("QIEA vs GA convergence (equal budget)"); plt.tight_layout()
plt.savefig("results/phase1_convergence.png", dpi=150)
print("Saved results/phase1_demo_results.csv and results/phase1_convergence.png")
