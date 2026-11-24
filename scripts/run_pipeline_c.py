"""Person C Full Pipeline: Runs the 5 benchmark methods on the Pima Diabetes dataset."""
import argparse
import os
import sys
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.interfaces import bind_fitness
from person_a import qiea, ga
from person_b.classifier import predict
from person_b.fitness import fitness
from person_b.pso import run as pso_run
from person_c.data_loader import load_dataset
from person_c.spark_evaluator import evaluate_population
from person_a.metrics import classification_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--pop", type=int, default=10)
    parser.add_argument("--gens", type=int, default=10)
    args = parser.parse_args()

    results_dir = os.path.join("results", "pima")
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    methods = {
        "KNN+GA": lambda fn, n, s: ga.run(fn, n, args.gens, s, pop_size=args.pop),
        "KNN+PSO": lambda fn, n, s: pso_run(fn, n, args.gens, s, pop_size=args.pop),
        "QIEA-KNN": lambda fn, n, s: qiea.run(fn, n, args.gens, s, pop_size=args.pop, evaluate_population=evaluate_population),
    }

    print(f"Starting Person C pipeline over {args.seeds} seeds...")
    for seed in range(args.seeds):
        X_tr, X_te, y_tr, y_te, folds = load_dataset("pima", seed=seed)
        n_features = X_tr.shape[1]

        # 1. Baseline KNN
        base_pred, mem = predict(X_tr, y_tr, X_te, np.ones(n_features, bool), k=5, fuzzy=False)
        m = classification_metrics(y_te, base_pred, mem, n_features)
        rows.append({"method": "KNN Baseline", "seed": seed, **m})

        # 2. Search Methods
        fn = bind_fitness(fitness, X_tr, y_tr, folds)
        for name, run_fn in methods.items():
            t0 = time.time()
            mask, k, hist = run_fn(fn, n_features, seed)
            elapsed = time.time() - t0
            pred, mem = predict(X_tr, y_tr, X_te, mask, k=k, fuzzy=False)
            m = classification_metrics(y_te, pred, mem, int(mask.sum()))
            rows.append({"method": name, "seed": seed, "elapsed": round(elapsed, 2), "k": k, **m})

        print(f"Completed seed {seed + 1}/{args.seeds}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(results_dir, "pima_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults successfully saved to {out_csv}")
    print("\n--- Summary Performance (Mean F1) ---")
    print(df.groupby("method")["f1"].mean())


if __name__ == "__main__":
    main()