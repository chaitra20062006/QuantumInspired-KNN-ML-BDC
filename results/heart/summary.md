# Results: heart, 10 seeds, pop 20 x 20 gens (mean ± std, held-out test)

| Method | accuracy | precision | recall | f1 | auc | n_features | k | gens_to_99 | seconds |
|---|---|---|---|---|---|---|---|---|---|
| KNN baseline | 0.8263 ± 0.0315 | 0.8328 ± 0.0390 | 0.7829 ± 0.0788 | 0.8042 ± 0.0445 | 0.8859 ± 0.0370 | 13.000 ± 0.000 | 5.000 ± 0.000 | n/a | 0.000 ± 0.000 |
| KNN+GA | 0.8066 ± 0.0388 | 0.8175 ± 0.0450 | 0.7514 ± 0.0914 | 0.7793 ± 0.0553 | 0.8661 ± 0.0300 | 5.100 ± 0.738 | 25.200 ± 6.286 | 5.500 ± 3.894 | 12.120 ± 3.959 |
| KNN+PSO | 0.7921 ± 0.0451 | 0.8004 ± 0.0659 | 0.7400 ± 0.0908 | 0.7647 ± 0.0565 | 0.8537 ± 0.0472 | 5.500 ± 1.179 | 23.000 ± 8.165 | 5.500 ± 2.415 | 11.443 ± 4.261 |
| QIEA-KNN | 0.7908 ± 0.0399 | 0.7920 ± 0.0617 | 0.7514 ± 0.0894 | 0.7663 ± 0.0510 | 0.8615 ± 0.0485 | 5.200 ± 1.476 | 18.800 ± 9.682 | 6.600 ± 4.115 | 10.985 ± 3.935 |
| QIEA-Fuzzy-KNN | 0.7776 ± 0.0466 | 0.7748 ± 0.0569 | 0.7343 ± 0.0797 | 0.7516 ± 0.0539 | 0.8375 ± 0.0452 | 5.600 ± 1.776 | 20.000 ± 6.944 | 8.000 ± 3.859 | 11.675 ± 4.163 |

## Paired tests over seeds (A - B)
Wilcoxon signed-rank and paired t-test; Holm-adjusted over the 4 comparisons per metric.

| Metric | A vs B | mean diff | W/T/L | Wilcoxon p | Holm p | t-test p |
|---|---|---|---|---|---|---|
| f1 | QIEA-KNN vs KNN baseline | -0.0380 | 1/0/9 | 0.0039 | 0.0156 | 0.0044 |
| f1 | QIEA-KNN vs KNN+GA | -0.0130 | 2/1/7 | 0.2031 | 0.6094 | 0.3316 |
| f1 | QIEA-KNN vs KNN+PSO | +0.0016 | 4/1/5 | 0.8203 | 1.0000 | 0.8845 |
| f1 | QIEA-Fuzzy-KNN vs QIEA-KNN | -0.0147 | 5/1/4 | 0.5703 | 1.0000 | 0.3853 |
| recall | QIEA-KNN vs KNN baseline | -0.0314 | 1/3/6 | 0.0625 | 0.2500 | 0.0318 |
| recall | QIEA-KNN vs KNN+GA | -0.0000 | 3/5/2 | 1.0000 | 1.0000 | 1.0000 |
| recall | QIEA-KNN vs KNN+PSO | +0.0114 | 5/2/3 | 0.5547 | 1.0000 | 0.4790 |
| recall | QIEA-Fuzzy-KNN vs QIEA-KNN | -0.0171 | 2/4/4 | 0.5312 | 1.0000 | 0.4679 |
| n_features | QIEA-KNN vs KNN baseline | -7.8000 | 0/0/10 | 0.0020 | 0.0078 | 0.0000 |
| n_features | QIEA-KNN vs KNN+GA | +0.1000 | 4/2/4 | 1.0000 | 1.0000 | 0.8321 |
| n_features | QIEA-KNN vs KNN+PSO | -0.3000 | 2/3/5 | 0.7188 | 1.0000 | 0.6044 |
| n_features | QIEA-Fuzzy-KNN vs QIEA-KNN | +0.4000 | 5/2/3 | 0.7500 | 1.0000 | 0.6193 |
| gens_to_99 | QIEA-KNN vs KNN+GA | +1.1000 | 5/2/3 | 0.6016 | 1.0000 | 0.4347 |
| gens_to_99 | QIEA-KNN vs KNN+PSO | +1.1000 | 4/0/6 | 0.5410 | 1.0000 | 0.3748 |
| gens_to_99 | QIEA-Fuzzy-KNN vs QIEA-KNN | +1.4000 | 5/0/5 | 0.3398 | 1.0000 | 0.4120 |

Search-method comparisons on train fitness are only meaningful between methods using the same (crisp) fitness: KNN+GA, KNN+PSO, QIEA-KNN. QIEA-Fuzzy-KNN optimizes a different (fuzzy) fitness.