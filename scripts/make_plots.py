"""Regenerate plots from results/full_results.csv and results/histories.json."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from person_a.plotting import box_plot, convergence_plot

df = pd.read_csv("results/full_results.csv"); H = json.load(open("results/histories.json"))
order = ["KNN baseline", "KNN+GA", "KNN+PSO", "QIEA-KNN", "QIEA-Fuzzy-KNN"]
convergence_plot({m: np.array(H[m]) for m in ["KNN+GA", "KNN+PSO", "QIEA-KNN"]}, "results/convergence_crisp.png",
                 "Convergence, crisp-KNN fitness (equal budget)")
for met, yl in (("f1", "Test F1"), ("n_features", "# selected features")):
    box_plot({m: df[df.method == m][met].values for m in order}, f"results/box_{met}.png", yl)
print("plots saved")
