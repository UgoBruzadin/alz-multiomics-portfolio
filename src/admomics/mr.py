"""Two-sample Mendelian randomization for causal-gene / drug-target nomination.

MR uses genetic variants as instruments to estimate the causal effect of a
molecular exposure (e.g. a protein's abundance, from pQTLs) on AD risk (from
GWAS). Combined with colocalization, this is the lab's engine for turning
associations into directional, causal, druggable hypotheses.

Implements the three workhorse estimators:
  * IVW (inverse-variance weighted) -- the primary estimate
  * MR-Egger -- intercept tests for directional pleiotropy
  * Weighted median -- robust to up to 50% invalid instruments
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class MRResult:
    method: str
    estimate: float
    se: float
    pvalue: float
    n_snps: int
    intercept: float = np.nan
    intercept_p: float = np.nan

    def as_row(self) -> dict:
        return {
            "method": self.method,
            "estimate": self.estimate,
            "or_": float(np.exp(self.estimate)),
            "se": self.se,
            "pvalue": self.pvalue,
            "n_snps": self.n_snps,
            "egger_intercept": self.intercept,
            "egger_intercept_p": self.intercept_p,
        }


def _harmonize(
    exposure: pd.DataFrame, outcome: pd.DataFrame, snp="snp"
) -> pd.DataFrame:
    """Merge instrument effects for exposure and outcome on shared SNPs."""
    return exposure.merge(outcome, on=snp, suffixes=("_exp", "_out"))


def mr_ivw(bx, by, sx, sy) -> MRResult:
    """Inverse-variance weighted estimate (fixed effects)."""
    w = 1.0 / (sy**2)
    beta = np.sum(w * bx * by) / np.sum(w * bx**2)
    se = np.sqrt(1.0 / np.sum(w * bx**2))
    p = 2 * norm.sf(abs(beta / se))
    return MRResult("IVW", beta, se, p, len(bx))


def mr_egger(bx, by, sx, sy) -> MRResult:
    """MR-Egger regression; intercept != 0 flags directional pleiotropy."""
    w = 1.0 / (sy**2)
    X = np.column_stack([np.ones_like(bx), bx])
    W = np.diag(w)
    xtwx = X.T @ W @ X
    coef = np.linalg.solve(xtwx, X.T @ W @ by)
    resid = by - X @ coef
    dof = max(len(bx) - 2, 1)
    sigma2 = (resid @ W @ resid) / dof
    cov = sigma2 * np.linalg.inv(xtwx)
    slope, intercept = coef[1], coef[0]
    slope_se = np.sqrt(cov[1, 1])
    int_se = np.sqrt(cov[0, 0])
    p = 2 * norm.sf(abs(slope / slope_se))
    ip = 2 * norm.sf(abs(intercept / int_se))
    return MRResult("MR-Egger", slope, slope_se, p, len(bx), intercept, ip)


def mr_weighted_median(bx, by, sx, sy, n_boot: int = 1000, seed: int = 0) -> MRResult:
    """Weighted-median estimate; robust to <=50% invalid instruments."""
    ratios = by / bx
    weights = (bx**2) / (sy**2)

    def _wm(r, w):
        order = np.argsort(r)
        r, w = r[order], w[order]
        cw = np.cumsum(w) - 0.5 * w
        cw /= np.sum(w)
        below = np.searchsorted(cw, 0.5) - 1
        below = np.clip(below, 0, len(r) - 2)
        return r[below] + (r[below + 1] - r[below]) * (0.5 - cw[below]) / (
            cw[below + 1] - cw[below]
        )

    est = _wm(ratios, weights)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        bxi = rng.normal(bx, sx)
        byi = rng.normal(by, sy)
        boot[i] = _wm(byi / bxi, (bxi**2) / (sy**2))
    se = np.std(boot)
    p = 2 * norm.sf(abs(est / se)) if se > 0 else np.nan
    return MRResult("Weighted median", est, se, p, len(bx))


def run_mr(
    instruments: pd.DataFrame,
    beta_exp="beta_exp",
    se_exp="se_exp",
    beta_out="beta_out",
    se_out="se_out",
) -> pd.DataFrame:
    """Run all three estimators on a harmonized instrument table."""
    bx = instruments[beta_exp].to_numpy()
    by = instruments[beta_out].to_numpy()
    sx = instruments[se_exp].to_numpy()
    sy = instruments[se_out].to_numpy()
    results = [
        mr_ivw(bx, by, sx, sy),
        mr_egger(bx, by, sx, sy),
        mr_weighted_median(bx, by, sx, sy),
    ]
    return pd.DataFrame([r.as_row() for r in results])


def simulate_instruments(
    n_snps: int = 25,
    true_effect: float = 0.3,
    pleiotropy: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate MR instruments with a known causal effect for validation."""
    rng = np.random.default_rng(seed)
    bx = rng.uniform(0.05, 0.3, n_snps) * rng.choice([-1, 1], n_snps)
    sx = rng.uniform(0.01, 0.03, n_snps)
    plei = rng.normal(0, pleiotropy, n_snps)
    sy = rng.uniform(0.01, 0.03, n_snps)
    by = true_effect * bx + plei + rng.normal(0, sy)
    return pd.DataFrame(
        {"snp": [f"rs{i}" for i in range(n_snps)],
         "beta_exp": bx, "se_exp": sx, "beta_out": by, "se_out": sy}
    )
