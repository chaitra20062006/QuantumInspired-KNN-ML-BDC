# QIEA-KNN: Person A, Phase 1 + Phase 2 deliverable (this zip contains both)

Person A = Search Algorithm Owner. Phase 1 = everything needed to run the search algorithms end to end
on stand-ins, so you are never blocked by Person B or C.

## Phase split (Person A, 8 points)
**Phase 1 (this zip): foundations + search engine**
- Shared interface file with the agreed solution format and contract checks (`shared/interfaces.py`)
- QIEA core (3): equal-amplitude q-bit init, observation, rotation gate toward best-so-far, NOT-gate quantum mutation, best tracking, joint feature + K encoding
- GA baseline (2): same fitness, same chromosome, same evaluation budget as QIEA
- Baseline KNN (1): all features, K=5, Euclidean
- Stand-ins: toy objective, sklearn CV fitness, sklearn Breast Cancer loader
- Contract/behaviour tests and a runnable demo

**Phase 2 (next)**
- Metrics, statistical tests and plotting toolkit (2): accuracy, precision, recall, F1, AUC, #features, Wilcoxon / paired t-test, mean ± std, convergence curves (used by all three)
- Swap stand-ins for the real versions: B's fitness and K decoding, C's data pipeline and folds, C's Spark `evaluate_population`
- Tune QIEA (rotation angle, mutation, population size) and run all 5 methods on Breast Cancer Wisconsin, 10+ seeds
- Literature search and write-up of Related Work and Proposed Method (QIEA part)

## Layout
```
shared/interfaces.py     agreed interfaces (one shared file)
person_a/qiea.py         QIEA  run(fitness_fn, n_features, generations, seed) -> best_mask, best_k, history
person_a/ga.py           GA baseline, same signature
person_a/baseline_knn.py default KNN reference
person_a/standins.py     stand-ins for B's fitness and C's data
tests/test_contracts.py  contract tests
scripts/run_phase1_demo.py  QIEA vs GA vs baseline demo
results/                 demo CSV and convergence plot
```

## Run
```
pip install -r requirements.txt
python -m unittest discover -s tests -v        # 11 tests (pytest also works)
python scripts/run_phase1_demo.py --seeds 5 --pop 20 --gens 20
```

## Plugging in teammates' parts
- **B's fitness:** `fn = bind_fitness(real_fitness, X, y, folds)`, then `qiea.run(fn, n_features, gens, seed)`.
- **C's Spark evaluator:** `qiea.run(..., evaluate_population=spark_evaluate)`. Same for `ga.run`.
- **K encoding:** `decode_k` in `shared/interfaces.py` is a proposal (4 bits -> odd K in 1..31). B owns the final version; agree it before merging. Pass `k_bits=` to match.

## Assumptions to confirm with the team
1. Optimizers receive `fitness_fn(mask, k)` with X, y, folds already bound.
2. `history` is a list of dicts `{generation, best_fitness, mean_fitness, n_evals}`.
3. Budget = pop_size x generations evaluations for both QIEA and GA (B's PSO should match).

## Phase 2 (added in this zip)
New modules in `person_a/`: `metrics.py` (accuracy, precision, recall, F1, AUC, #features, generations-to-99%),
`stats.py` (Wilcoxon, paired t-test, Holm correction, mean ± std), `plotting.py` (convergence curves, boxplots),
`components.py` (use B's and C's real code via env vars, stand-ins otherwise), plus stand-ins that follow the agreed
interfaces exactly: `standins.standin_predict` / `standin_fitness_pred` (crisp + fuzzy KNN) and `pso_standin.py`.
```
python scripts/tune_qiea.py          # small grid for QIEA and GA on separate tuning seeds -> results/tuning.json
python scripts/run_full_pipeline.py  # 5 methods x 10 seeds -> results/summary.md, full_results.csv, plots
python scripts/make_plots.py         # regenerate plots from saved results
python -m unittest discover -s tests # 20 tests
```
Switch to the real components when ready, e.g.
`QIEA_FITNESS=person_b.fitness:fitness QIEA_PREDICT=person_b.knn:predict QIEA_EVALUATOR=person_c.spark:evaluate_population python scripts/run_full_pipeline.py`
(module paths are examples; use your teammates' actual ones). The PSO, fuzzy KNN and fitness in this zip are
**stand-ins written by Person A**, so the five-method numbers must be re-run after integration.

## Results with stand-ins (Breast Cancer Wisconsin, 10 seeds, pop 20 x 20 gens, held-out test)
See `results/summary.md` for the full table and tests. Headline, mean over seeds:
all methods reach test F1 of about 0.963 to 0.970 with 6 to 7 features instead of 30, while the no-selection KNN
baseline has F1 0.9755. Differences between QIEA-KNN, GA and PSO were not statistically significant (Wilcoxon
p > 0.05 for F1; QIEA used fewer features than PSO, p = 0.023 uncorrected, 0.070 after Holm).
Selection lowered recall versus the baseline (p = 0.006, Holm 0.023). QIEA-KNN did not converge faster than GA:
it needed 2.8 more generations to reach 99% of its final fitness (p = 0.15, not significant) although it ended with
slightly higher mean training fitness. Fuzzy KNN did not significantly change F1, recall or features here.
These hypotheses (fewer features with equal F1, faster convergence than GA, better recall) are therefore
**not confirmed** on this dataset and budget; report them as they are.

## Limitations to state
- PSO, fuzzy KNN and the fitness are stand-ins; the K encoding is a proposal.
- Tuning gave QIEA and GA small grids only; the best QIEA setting was at the grid edge. PSO was not tuned.
- Small budget (400 evaluations per run); each seed uses its own split, so variance includes data-split variance.
- `gens_to_99` is relative to each run's own final value, so a method that keeps improving looks slower.
