"""APOE genotype calling and stratified risk estimation.

APOE is the strongest common genetic risk factor for late-onset AD. The lab's
signature finding (Belloy et al., JAMA Neurology 2023) is that the APOE*4 effect
is *not* uniform: it varies by ancestry, sex, and age. This module (a) calls the
epsilon genotype from the two defining SNPs and (b) estimates APOE*4 odds ratios
stratified by ancestry and sex, which is the analysis that produced that finding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import APOE_HAPLOTYPE_MAP


def call_apoe_genotype(rs429358: pd.Series, rs7412: pd.Series) -> pd.Series:
    """Map per-allele calls at the two APOE SNPs to an epsilon genotype string.

    Inputs are strings of two alleles each, e.g. rs429358="TC", rs7412="TT".
    Returns e.g. "e3/e4". This is illustrative of the calling logic; the
    simulator provides genotypes directly, but a real pipeline starts here.
    """
    def _call(a1: str, a2: str) -> str:
        alleles1 = list(a1)
        alleles2 = list(a2)
        haps = sorted(
            APOE_HAPLOTYPE_MAP.get((x, y), "e?")
            for x, y in zip(alleles1, alleles2)
        )
        return "/".join(haps)

    return pd.Series(
        [_call(a, b) for a, b in zip(rs429358, rs7412)], index=rs429358.index
    )


def apoe_dosage(genotype: pd.Series, allele: str = "e4") -> pd.Series:
    """Count copies of an epsilon allele (0/1/2) from an 'eX/eY' genotype string."""
    return genotype.str.count(allele)


def stratified_apoe_or(
    pheno: pd.DataFrame,
    strata: tuple = ("ancestry",),
    dosage_col: str = "APOE_e4_dosage",
    outcome: str = "AD",
    adjust: tuple = ("age", "PC1", "PC2", "PC3", "PC4"),
) -> pd.DataFrame:
    """Per-stratum, per-allele APOE odds ratios via adjusted logistic regression.

    For each stratum (e.g. each ancestry, or each ancestry x sex cell) fit:
        logit(AD) ~ APOE_dosage + age + PCs
    and report OR, 95% CI, and p. This reproduces the *shape* of the
    ancestry-stratified APOE analysis.
    """
    rows = []
    for keys, sub in pheno.groupby(list(strata)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        if sub[outcome].nunique() < 2 or len(sub) < 50:
            continue
        adj = [c for c in adjust if c in sub.columns]
        X = sub[[dosage_col] + adj].to_numpy(dtype=float)
        X = sm.add_constant(X, has_constant="add")
        y = sub[outcome].to_numpy()
        try:
            res = sm.Logit(y, X).fit(disp=0, maxiter=100)
            beta = res.params[1]
            se = res.bse[1]
            row = dict(zip(strata, keys))
            row.update(
                dict(
                    n=len(sub),
                    n_cases=int(sub[outcome].sum()),
                    or_=np.exp(beta),
                    ci_low=np.exp(beta - 1.96 * se),
                    ci_high=np.exp(beta + 1.96 * se),
                    beta=beta,
                    se=se,
                    p=res.pvalues[1],
                )
            )
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


def apoe_ancestry_interaction(
    pheno: pd.DataFrame,
    dosage_col: str = "APOE_e4_dosage",
    outcome: str = "AD",
    ref_ancestry: str = "EUR",
) -> pd.DataFrame:
    """Formal test that APOE*4 effect differs by ancestry (dosage x ancestry).

    Returns interaction coefficients relative to the reference ancestry; a
    significant term is evidence of ancestry-dependent APOE risk.
    """
    df = pheno.copy()
    df["ancestry"] = pd.Categorical(
        df["ancestry"],
        categories=[ref_ancestry]
        + [a for a in df["ancestry"].unique() if a != ref_ancestry],
    )
    dummies = pd.get_dummies(df["ancestry"], prefix="anc", drop_first=True).astype(
        float
    )
    inter = dummies.mul(df[dosage_col].to_numpy(), axis=0)
    inter.columns = [f"{c}_x_e4" for c in dummies.columns]
    adj = [c for c in ("age", "PC1", "PC2", "PC3", "PC4") if c in df.columns]
    X = pd.concat(
        [df[[dosage_col] + adj].reset_index(drop=True), dummies.reset_index(drop=True),
         inter.reset_index(drop=True)],
        axis=1,
    )
    X = sm.add_constant(X, has_constant="add")
    res = sm.Logit(df[outcome].to_numpy(), X.astype(float)).fit(disp=0, maxiter=200)
    out = pd.DataFrame(
        {"term": res.params.index, "beta": res.params.values,
         "se": res.bse.values, "p": res.pvalues.values}
    )
    return out[out["term"].str.contains("_x_e4")].reset_index(drop=True)
