"""Statistical tests and mean +/- std reporting."""
import numpy as np
from scipy import stats


def mean_std(x, digits=4) -> str:
    x = np.asarray(x, dtype=float)
    return f"{np.nanmean(x):.{digits}f} ± {np.nanstd(x, ddof=1) if len(x) > 1 else 0.0:.{digits}f}"


def paired_tests(a, b) -> dict:
    """Paired comparison of two methods over the same seeds/splits (a - b).
    Returns Wilcoxon signed-rank p, paired t-test p, mean difference, wins/ties/losses."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    res = {"mean_diff": float(d.mean()), "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
           "losses": int((d < 0).sum()), "wilcoxon_p": 1.0, "ttest_p": 1.0}
    if np.any(d != 0):
        try:
            res["wilcoxon_p"] = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
        except ValueError:
            pass
        if d.std(ddof=1) > 0:
            res["ttest_p"] = float(stats.ttest_rel(a, b).pvalue)
    return res


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (for several comparisons in one family)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
