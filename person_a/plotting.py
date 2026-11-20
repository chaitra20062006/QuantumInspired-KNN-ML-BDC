"""Plotting toolkit: convergence curves (mean ± std) and metric boxplots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def convergence_plot(curves: dict, path, title="Convergence (best-so-far fitness)", ylabel="Fitness"):
    """curves: {method: (runs, generations) array}."""
    plt.figure(figsize=(6.5, 4.2))
    for name, c in curves.items():
        c = np.asarray(c); x = np.arange(c.shape[1]); m, s = c.mean(0), c.std(0)
        plt.plot(x, m, label=name); plt.fill_between(x, m - s, m + s, alpha=0.15)
    plt.xlabel("Generation"); plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def box_plot(data: dict, path, ylabel, title=None):
    """data: {method: list of values over seeds}."""
    plt.figure(figsize=(6.5, 4.2))
    plt.boxplot(list(data.values()), showmeans=True)
    plt.xticks(range(1, len(data) + 1), list(data.keys()))
    plt.ylabel(ylabel); plt.xticks(rotation=20); plt.grid(alpha=.3)
    if title: plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
