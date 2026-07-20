"""Bayesian colocalization via approximate Bayes factors (Giambartolomei et al.).

Colocalization asks: does a GWAS association and a molecular QTL (eQTL / pQTL) at
the same locus share a single causal variant? A high posterior for H4 (shared
causal variant) is evidence that the gene/protein mediates the GWAS signal, which
is how the lab moves from a GWAS hit to a candidate causal gene / drug target.

This is a faithful implementation of the coloc "abf" method operating on
per-variant summary statistics (beta, varbeta) for two traits over the same set
of SNPs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _approx_bf(beta: np.ndarray, varbeta: np.ndarray, sd_prior: float) -> np.ndarray:
    """Wakefield approximate log Bayes factor per SNP for one trait."""
    z2 = beta**2 / varbeta
    r = sd_prior**2 / (sd_prior**2 + varbeta)
    return 0.5 * (np.log(1 - r) + r * z2)


def _logsumexp(x: np.ndarray) -> float:
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def _logdiffexp(a: float, b: float) -> float:
    """log(exp(a) - exp(b)) for a > b, numerically stable."""
    if b >= a:
        return -np.inf
    return a + np.log1p(-np.exp(b - a))


def coloc_abf(
    df: pd.DataFrame,
    beta1: str = "beta_gwas",
    varbeta1: str = "varbeta_gwas",
    beta2: str = "beta_qtl",
    varbeta2: str = "varbeta_qtl",
    p1: float = 1e-4,
    p2: float = 1e-4,
    p12: float = 1e-5,
    sd_prior1: float = 0.2,
    sd_prior2: float = 0.15,
) -> dict:
    """Colocalization posterior probabilities PP.H0..PP.H4 for a locus.

    Priors p1/p2/p12 are the standard coloc defaults: prior that a SNP is
    associated with trait 1 only, trait 2 only, or both.

    Returns a dict with the five posteriors; PP.H4 near 1 means "shared causal
    variant" (colocalized).
    """
    d = df.dropna(subset=[beta1, varbeta1, beta2, varbeta2]).copy()
    labf1 = _approx_bf(d[beta1].to_numpy(), d[varbeta1].to_numpy(), sd_prior1)
    labf2 = _approx_bf(d[beta2].to_numpy(), d[varbeta2].to_numpy(), sd_prior2)

    # per-hypothesis log evidence, summing over configurations of causal SNPs
    l1 = _logsumexp(labf1)                       # some SNP causal for trait 1
    l2 = _logsumexp(labf2)                       # some SNP causal for trait 2
    l_shared = _logsumexp(labf1 + labf2)         # same SNP causal for both (H4)
    # H3 (distinct causal SNPs) = product of marginals minus the shared term:
    #   sum_{i != j} ABF1_i * ABF2_j = (sum_i ABF1_i)(sum_j ABF2_j) - sum_i ABF1_i*ABF2_i
    l3 = _logdiffexp(l1 + l2, l_shared)

    log_h = np.array(
        [
            0.0,                       # H0: no causal variant
            np.log(p1) + l1,           # H1: causal for trait 1 only
            np.log(p2) + l2,           # H2: causal for trait 2 only
            np.log(p1) + np.log(p2) + l3,  # H3: distinct causal variants
            np.log(p12) + l_shared,    # H4: shared causal variant
        ]
    )
    denom = _logsumexp(log_h)
    pp = np.exp(log_h - denom)
    return {
        "PP.H0": pp[0],
        "PP.H1": pp[1],
        "PP.H2": pp[2],
        "PP.H3": pp[3],
        "PP.H4": pp[4],
        "n_snps": int(len(d)),
    }


def simulate_locus(
    n_snps: int = 60,
    shared_causal: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate GWAS + QTL summary stats over a locus for a coloc demo.

    If ``shared_causal`` the same SNP drives both traits (should yield high
    PP.H4); otherwise the causal variants differ (should yield high PP.H3).
    """
    rng = np.random.default_rng(seed)
    maf = rng.uniform(0.05, 0.5, n_snps)
    varbeta = 1.0 / (2 * maf * (1 - maf) * 4000)  # ~ 1/(2p(1-p)N)
    se = np.sqrt(varbeta)
    # small null noise everywhere...
    beta_g = rng.normal(0, se)
    beta_q = rng.normal(0, se)
    causal_g = rng.integers(n_snps)
    causal_q = causal_g if shared_causal else (causal_g + n_snps // 2) % n_snps
    # ...and a deterministic strong effect at each causal SNP (z ~ 6.5)
    beta_g[causal_g] = 6.5 * se[causal_g]
    beta_q[causal_q] = 6.5 * se[causal_q]
    return pd.DataFrame(
        {
            "snp": [f"rs{i}" for i in range(n_snps)],
            "beta_gwas": beta_g,
            "varbeta_gwas": varbeta,
            "beta_qtl": beta_q,
            "varbeta_qtl": varbeta,
        }
    )
