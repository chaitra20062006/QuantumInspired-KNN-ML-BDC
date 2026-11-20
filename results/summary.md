# Results: breast_cancer, 10 seeds, pop 20 x 20 gens (mean ± std, held-out test)

| Method | accuracy | precision | recall | f1 | auc | n_features | k | gens_to_99 | seconds |
|---|---|---|---|---|---|---|---|---|---|
| KNN baseline | 0.9685 ± 0.0111 | 0.9585 ± 0.0178 | 0.9933 ± 0.0078 | 0.9755 ± 0.0084 | 0.9844 ± 0.0107 | 30.000 ± 0.000 | 5.000 ± 0.000 | n/a | 0.000 ± 0.000 |
| KNN+GA | 0.9580 ± 0.0151 | 0.9561 ± 0.0219 | 0.9789 ± 0.0152 | 0.9671 ± 0.0116 | 0.9846 ± 0.0131 | 6.700 ± 1.160 | 14.600 ± 7.412 | 6.700 ± 2.497 | 5.053 ± 0.143 |
| KNN+PSO | 0.9531 ± 0.0190 | 0.9573 ± 0.0148 | 0.9689 ± 0.0202 | 0.9630 ± 0.0151 | 0.9850 ± 0.0106 | 7.400 ± 1.265 | 12.800 ± 6.630 | 5.100 ± 4.202 | 5.072 ± 0.127 |
| QIEA-KNN | 0.9615 ± 0.0129 | 0.9649 ± 0.0120 | 0.9744 ± 0.0149 | 0.9696 ± 0.0102 | 0.9848 ± 0.0151 | 5.800 ± 1.135 | 13.400 ± 8.884 | 9.500 ± 4.528 | 4.953 ± 0.056 |
| QIEA-Fuzzy-KNN | 0.9601 ± 0.0114 | 0.9599 ± 0.0141 | 0.9778 ± 0.0174 | 0.9686 ± 0.0090 | 0.9872 ± 0.0081 | 5.800 ± 1.317 | 15.600 ± 7.604 | 7.700 ± 2.908 | 5.083 ± 0.311 |

## Paired tests over seeds (A - B)
Wilcoxon signed-rank and paired t-test; Holm-adjusted over the 4 comparisons per metric.

| Metric | A vs B | mean diff | W/T/L | Wilcoxon p | Holm p | t-test p |
|---|---|---|---|---|---|---|
| f1 | QIEA-KNN vs KNN baseline | -0.0059 | 3/0/7 | 0.0840 | 0.3359 | 0.0954 |
| f1 | QIEA-KNN vs KNN+GA | +0.0024 | 5/1/4 | 0.4961 | 0.9922 | 0.5335 |
| f1 | QIEA-KNN vs KNN+PSO | +0.0066 | 7/0/3 | 0.2754 | 0.8262 | 0.1694 |
| f1 | QIEA-Fuzzy-KNN vs QIEA-KNN | -0.0010 | 6/1/3 | 0.9102 | 0.9922 | 0.7818 |
| recall | QIEA-KNN vs KNN baseline | -0.0189 | 1/0/9 | 0.0059 | 0.0234 | 0.0012 |
| recall | QIEA-KNN vs KNN+GA | -0.0044 | 3/2/5 | 0.3984 | 1.0000 | 0.4790 |
| recall | QIEA-KNN vs KNN+PSO | +0.0056 | 5/2/3 | 0.5703 | 1.0000 | 0.3809 |
| recall | QIEA-Fuzzy-KNN vs QIEA-KNN | +0.0033 | 6/2/2 | 0.4375 | 1.0000 | 0.6044 |
| n_features | QIEA-KNN vs KNN baseline | -24.2000 | 0/0/10 | 0.0020 | 0.0078 | 0.0000 |
| n_features | QIEA-KNN vs KNN+GA | -0.9000 | 1/4/5 | 0.1250 | 0.2500 | 0.0676 |
| n_features | QIEA-KNN vs KNN+PSO | -1.6000 | 1/2/7 | 0.0234 | 0.0703 | 0.0107 |
| n_features | QIEA-Fuzzy-KNN vs QIEA-KNN | +0.0000 | 4/3/3 | 1.0000 | 1.0000 | 1.0000 |
| gens_to_99 | QIEA-KNN vs KNN+GA | +2.8000 | 6/1/3 | 0.1523 | 0.3047 | 0.1323 |
| gens_to_99 | QIEA-KNN vs KNN+PSO | +4.4000 | 7/1/2 | 0.0859 | 0.2578 | 0.0677 |
| gens_to_99 | QIEA-Fuzzy-KNN vs QIEA-KNN | -1.8000 | 2/2/6 | 0.2188 | 0.3047 | 0.2068 |

Search-method comparisons on train fitness are only meaningful between methods using the same (crisp) fitness: KNN+GA, KNN+PSO, QIEA-KNN. QIEA-Fuzzy-KNN optimizes a different (fuzzy) fitness.