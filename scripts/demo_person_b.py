"""Person B demo: final QIEA-Fuzzy-KNN model on Heart Disease (Cleveland) + borderline-case analysis.

Usage (from the project root):  python scripts/demo_person_b.py [--seed 0] [--pop 20] [--gens 20]
Writes results/heart_demo_borderline.csv
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from shared.interfaces import bind_fitness
from person_a import qiea
from person_a.metrics import classification_metrics
from person_b.classifier import predict
from person_b.fitness import fitness
from person_b.heart_data import feature_names, load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--pop", type=int, default=20)
ap.add_argument("--gens", type=int, default=20); ap.add_argument("--border", type=float, default=0.75,
                help="a patient is 'borderline' if the top fuzzy membership is below this")
a = ap.parse_args()

Xtr, Xte, ytr, yte, folds = load_dataset("heart", seed=a.seed)
n, names = Xtr.shape[1], feature_names()
fn = bind_fitness(lambda m, k, X, y, f: fitness(m, k, X, y, f, fuzzy=True), Xtr, ytr, folds)

print(f"Searching with QIEA + fuzzy fitness (pop {a.pop} x {a.gens} gens) ...")
mask, k, hist = qiea.run(fn, n, a.gens, a.seed, pop_size=a.pop)
print(f"\nSelected {int(mask.sum())}/{n} features: {[nm for nm, m in zip(names, mask) if m]}")
print(f"Chosen K = {k}   |   best train-CV fitness = {hist[-1]['best_fitness']:.4f}")

full = np.ones(n, bool)
rows = []
for label, mk, kk, fz in (("KNN baseline (all feats, K=5, crisp)", full, 5, False),
                          ("QIEA-KNN (crisp)", mask, k, False),
                          ("QIEA-Fuzzy-KNN", mask, k, True)):
    lab, mem = predict(Xtr, ytr, Xte, mk, kk, fz)
    m = classification_metrics(yte, lab, mem, mk.sum())
    rows.append({"model": label, **{c: round(float(m[c]), 4) for c in ("accuracy", "precision", "recall", "f1", "auc")},
                 "n_features": int(mk.sum())})
print("\nHeld-out test results (single seed, so no significance claims):")
print(pd.DataFrame(rows).to_string(index=False))

# ---- borderline analysis: where does the fuzzy layer help? ----
lab_c, mem_c = predict(Xtr, ytr, Xte, mask, k, False)
lab_f, mem_f = predict(Xtr, ytr, Xte, mask, k, True)
top = mem_f.max(axis=1)
border = top < a.border
print(f"\nBorderline test patients (fuzzy top membership < {a.border}): {int(border.sum())} of {len(yte)}")
if border.any():
    print(f"  crisp accuracy on them: {(lab_c[border] == yte[border]).mean():.3f}")
    print(f"  fuzzy accuracy on them: {(lab_f[border] == yte[border]).mean():.3f}")
print(f"Labels where crisp and fuzzy disagree: {int((lab_c != lab_f).sum())}")

print("\nFirst 8 test patients (membership = [healthy, disease]):")
for i in range(min(8, len(yte))):
    print(f"  patient {i}: true={yte[i]}  fuzzy memberships={np.round(mem_f[i], 2)}  -> {lab_f[i]}"
          f"{'  (borderline)' if border[i] else ''}")

os.makedirs("results", exist_ok=True)
out = pd.DataFrame({"true": yte, "crisp_label": lab_c, "fuzzy_label": lab_f,
                    "p_disease_crisp": mem_c[:, -1].round(3), "p_disease_fuzzy": mem_f[:, -1].round(3),
                    "borderline": border})
out.to_csv("results/heart_demo_borderline.csv", index=False)
print("\nSaved results/heart_demo_borderline.csv")