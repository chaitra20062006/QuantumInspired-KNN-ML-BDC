"""Full 5-method pipeline on Breast Cancer Wisconsin (Person A's dataset).

Methods (identical fitness, chromosome and evaluation budget for all searches):
  1 KNN baseline (all features, K=5)      2 KNN + GA      3 KNN + PSO
  4 QIEA-KNN                              5 QIEA-Fuzzy-KNN (fuzzy fitness + fuzzy classifier)
Each seed = a fresh stratified train/test split + CV folds + optimizer seed.
Selection uses TRAIN-set CV fitness only; all reported metrics are on the held-out TEST set.
Usage: python scripts/run_full_pipeline.py [--seeds 10] [--pop 20] [--gens 20]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from shared.interfaces import bind_fitness, check_optimizer_output
from person_a import qiea, ga
from person_a.components import get_components, fitness_for, load_data
from person_a.metrics import classification_metrics, generations_to_reach, history_matrix
from person_a.plotting import box_plot, convergence_plot
from person_a.stats import holm, mean_std, paired_tests

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--pop", type=int, default=20)
ap.add_argument("--gens", type=int, default=20); ap.add_argument("--dataset", default="breast_cancer")
a = ap.parse_args()

comp = get_components()
print({k: v for k, v in comp.items() if k.endswith("_source")})
tuned = json.load(open("results/tuning.json")) if os.path.exists("results/tuning.json") else {}
qp, gp = tuned.get("qiea", {}).get("best", {}), tuned.get("ga", {}).get("best", {})
print("QIEA params:", qp or "defaults", "| GA params:", gp or "defaults")
ev = comp["evaluate_population"]
SEARCH = {"KNN+GA": (ga.run, gp, False), "KNN+PSO": (comp["pso_run"], {}, False),
          "QIEA-KNN": (qiea.run, qp, False), "QIEA-Fuzzy-KNN": (qiea.run, qp, True)}

rows, hists = [], {m: [] for m in SEARCH}
for seed in range(a.seeds):
    Xtr, Xte, ytr, yte, folds = load_data(comp, a.dataset, seed)
    n = Xtr.shape[1]
    lab, mem = comp["predict"](Xtr, ytr, Xte, np.ones(n, bool), 5, False)
    rows.append({"method": "KNN baseline", "seed": seed, **classification_metrics(yte, lab, mem, n),
                 "k": 5, "seconds": 0.0, "gens_to_99": np.nan, "train_fitness": np.nan})
    for name, (run, params, fuzzy) in SEARCH.items():
        fn = bind_fitness(fitness_for(comp, fuzzy), Xtr, ytr, folds)
        t = time.time()
        out = run(fn, n, a.gens, seed, pop_size=a.pop, evaluate_population=ev, **params)
        check_optimizer_output(out, n, a.gens)
        mask, k, h = out
        lab, mem = comp["predict"](Xtr, ytr, Xte, mask, k, fuzzy)
        rows.append({"method": name, "seed": seed, **classification_metrics(yte, lab, mem, mask.sum()),
                     "k": k, "seconds": round(time.time() - t, 2), "gens_to_99": generations_to_reach(h),
                     "train_fitness": h[-1]["best_fitness"]})
        hists[name].append(h)
    print(f"seed {seed} done", flush=True)

os.makedirs("results", exist_ok=True)
df = pd.DataFrame(rows); df.to_csv("results/full_results.csv", index=False)
json.dump({m: [[x["best_fitness"] for x in h] for h in hs] for m, hs in hists.items()}, open("results/histories.json", "w"))

order = ["KNN baseline", "KNN+GA", "KNN+PSO", "QIEA-KNN", "QIEA-Fuzzy-KNN"]
cols = ["accuracy", "precision", "recall", "f1", "auc", "n_features", "k", "gens_to_99", "seconds"]
md = [f"# Results: {a.dataset}, {a.seeds} seeds, pop {a.pop} x {a.gens} gens (mean ± std, held-out test)\n",
      "| Method | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
for m in order:
    g = df[df.method == m]
    md.append(f"| {m} | " + " | ".join(("n/a" if g[c].isna().all() else mean_std(g[c], 3 if c in ("n_features", "k", "gens_to_99", "seconds") else 4)) for c in cols) + " |")

PAIRS = [("QIEA-KNN", "KNN baseline"), ("QIEA-KNN", "KNN+GA"), ("QIEA-KNN", "KNN+PSO"), ("QIEA-Fuzzy-KNN", "QIEA-KNN")]
md += ["\n## Paired tests over seeds (A - B)", "Wilcoxon signed-rank and paired t-test; Holm-adjusted over the 4 comparisons per metric.\n",
       "| Metric | A vs B | mean diff | W/T/L | Wilcoxon p | Holm p | t-test p |", "|---|---|---|---|---|---|---|"]
for met in ["f1", "recall", "n_features", "gens_to_99"]:
    res = []
    for x, y in PAIRS:
        gx, gy = df[df.method == x].sort_values("seed")[met].values, df[df.method == y].sort_values("seed")[met].values
        if np.isnan(gx).any() or np.isnan(gy).any():
            continue
        res.append((x, y, paired_tests(gx, gy)))
    adj = holm([r[2]["wilcoxon_p"] for r in res])
    for (x, y, r), hp in zip(res, adj):
        md.append(f"| {met} | {x} vs {y} | {r['mean_diff']:+.4f} | {r['wins']}/{r['ties']}/{r['losses']} | {r['wilcoxon_p']:.4f} | {hp:.4f} | {r['ttest_p']:.4f} |")
md.append("\nSearch-method comparisons on train fitness are only meaningful between methods using the same (crisp) fitness: KNN+GA, KNN+PSO, QIEA-KNN. QIEA-Fuzzy-KNN optimizes a different (fuzzy) fitness.")
open("results/summary.md", "w").write("\n".join(md)); print("\n".join(md))

H = {m: history_matrix(hists[m]) for m in SEARCH}
convergence_plot({m: H[m] for m in ["KNN+GA", "KNN+PSO", "QIEA-KNN"]}, "results/convergence_crisp.png",
                 "Convergence, crisp-KNN fitness (equal budget)")
for met, yl in (("f1", "Test F1"), ("n_features", "# selected features")):
    box_plot({m: df[df.method == m][met].values for m in order}, f"results/box_{met}.png", yl)
print("saved results/*")
