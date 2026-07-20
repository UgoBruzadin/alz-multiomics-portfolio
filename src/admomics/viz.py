"""Plotting helpers. Matplotlib-only, no seaborn dependency required for figures."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})


def qq_plot(pvalues, ax=None, title="QQ plot"):
    ax = ax or plt.gca()
    p = np.sort(pvalues.dropna().to_numpy())
    exp = -np.log10((np.arange(1, len(p) + 1) - 0.5) / len(p))
    obs = -np.log10(np.clip(p, 1e-300, 1))
    ax.plot([0, exp.max()], [0, exp.max()], "--", color="grey", lw=1)
    ax.scatter(exp, obs, s=10, alpha=0.7)
    ax.set(xlabel="Expected -log10(p)", ylabel="Observed -log10(p)", title=title)
    return ax


def apoe_forest(strat: pd.DataFrame, group_col="ancestry", ax=None,
                title="APOE*4 odds ratio by ancestry"):
    """Forest plot of stratified APOE ORs with 95% CIs."""
    ax = ax or plt.gca()
    d = strat.sort_values("or_")
    y = np.arange(len(d))
    ax.errorbar(
        d["or_"], y,
        xerr=[d["or_"] - d["ci_low"], d["ci_high"] - d["or_"]],
        fmt="o", capsize=3, color="#2b6cb0",
    )
    ax.axvline(1.0, ls="--", color="grey", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d[group_col])
    ax.set(xlabel="Odds ratio per APOE*4 allele (95% CI)", title=title)
    return ax


def block_auroc_bar(block_df: pd.DataFrame, ax=None, title="Integration: AUROC by strategy"):
    ax = ax or plt.gca()
    d = block_df.sort_values("auroc")
    y = np.arange(len(d))
    err = [d["auroc"] - d["auroc_lo"], d["auroc_hi"] - d["auroc"]]
    colors = ["#dd6b20" if "fusion" in m else "#4a5568" for m in d["model"]]
    ax.barh(y, d["auroc"], xerr=err, color=colors, alpha=0.85, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["model"])
    ax.set_xlim(0.5, 1.0)
    ax.set(xlabel="Cross-validated AUROC", title=title)
    return ax


def importance_bar(imp_df: pd.DataFrame, ax=None, title="Top permutation importances"):
    ax = ax or plt.gca()
    d = imp_df.sort_values("importance").tail(15)
    palette = {"transcriptomics": "#38a169", "proteomics": "#805ad5",
               "genetics": "#e53e3e"}
    colors = [palette.get(b, "#718096") for b in d["block"]]
    ax.barh(np.arange(len(d)), d["importance"], color=colors, alpha=0.9)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["feature"], fontsize=8)
    ax.set(xlabel="Mean AUROC drop when permuted", title=title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    ax.legend(handles, palette.keys(), fontsize=8, loc="lower right")
    return ax


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
