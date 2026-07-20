"""Association testing utilities: QC + covariate-adjusted logistic GWAS.

In production an AD lab would run PLINK2 / REGENIE on hundreds of thousands of
variants; here we implement the same statistical model (covariate-adjusted
logistic regression) in statsmodels so the demo runs anywhere and the logic is
transparent. The covariate set -- age, sex, and ancestry PCs -- is exactly what
you would use to control for population stratification in a real GWAS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

DEFAULT_COVARS = ["age", "PC1", "PC2", "PC3", "PC4"]


def variant_qc(
    genotypes: pd.DataFrame,
    maf_min: float = 0.01,
    call_rate_min: float = 0.98,
    hwe_p_min: float = 1e-6,
) -> pd.DataFrame:
    """Return a per-variant QC table with pass/fail flags.

    Filters mirror standard GWAS QC: minor-allele frequency, call rate, and a
    Hardy-Weinberg equilibrium chi-square test.
    """
    geno = genotypes.drop(columns=["subject_id"], errors="ignore")
    n = len(geno)
    rows = []
    for snp in geno.columns:
        col = geno[snp]
        call_rate = col.notna().mean()
        maf = np.nanmean(col) / 2.0
        maf = min(maf, 1 - maf)
        # HWE: observed vs expected genotype counts under HWE
        obs = col.value_counts().reindex([0, 1, 2]).fillna(0).to_numpy()
        p = (2 * obs[0] + obs[1]) / (2 * obs.sum()) if obs.sum() else np.nan
        exp = np.array([p**2, 2 * p * (1 - p), (1 - p) ** 2]) * obs.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.nansum((obs - exp) ** 2 / np.where(exp > 0, exp, np.nan))
        from scipy.stats import chi2 as chi2_dist

        hwe_p = 1 - chi2_dist.cdf(chi2, df=1)
        passed = (
            (maf >= maf_min)
            and (call_rate >= call_rate_min)
            and (hwe_p >= hwe_p_min)
        )
        rows.append(
            dict(snp=snp, maf=maf, call_rate=call_rate, hwe_p=hwe_p, pass_qc=passed)
        )
    return pd.DataFrame(rows)


def _logit_assoc(y, X):
    """Fit a logistic regression, returning (beta, se, p) for the last column."""
    Xc = sm.add_constant(X, has_constant="add")
    try:
        res = sm.Logit(y, Xc).fit(disp=0, maxiter=100)
        # np.asarray works whether statsmodels returns a Series or ndarray
        beta = float(np.asarray(res.params)[-1])
        se = float(np.asarray(res.bse)[-1])
        p = float(np.asarray(res.pvalues)[-1])
        return beta, se, p
    except Exception:
        return np.nan, np.nan, np.nan


def run_gwas(
    genotypes: pd.DataFrame,
    pheno: pd.DataFrame,
    covars: list[str] | None = None,
    outcome: str = "AD",
    add_sex: bool = True,
) -> pd.DataFrame:
    """Covariate-adjusted logistic GWAS across all variants.

    Returns one row per variant with effect size (log-OR), SE, and p-value.
    """
    covars = list(covars or DEFAULT_COVARS)
    geno = genotypes.drop(columns=["subject_id"], errors="ignore")
    df = pheno.copy()
    if add_sex:
        df["sex_bin"] = (df["sex"] == "F").astype(int)
        covar_cols = covars + ["sex_bin"]
    else:
        covar_cols = covars
    y = df[outcome].to_numpy()
    base = df[covar_cols].to_numpy(dtype=float)

    results = []
    for snp in geno.columns:
        X = np.column_stack([base, geno[snp].to_numpy(dtype=float)])
        beta, se, p = _logit_assoc(y, X)
        results.append(
            dict(snp=snp, beta=beta, se=se, or_=np.exp(beta), p=p, n=len(y))
        )
    out = pd.DataFrame(results)
    out["neglog10p"] = -np.log10(out["p"].clip(lower=1e-300))
    return out.sort_values("p").reset_index(drop=True)


def sex_stratified_gwas(
    genotypes: pd.DataFrame,
    pheno: pd.DataFrame,
    covars: list[str] | None = None,
    outcome: str = "AD",
) -> pd.DataFrame:
    """Run the GWAS separately in females and males and test for heterogeneity.

    This is the core of the lab's sex-dimorphism thrust: fit sex-specific effects
    and quantify their difference with a two-sample z-test of betas.
    """
    covars = list(covars or DEFAULT_COVARS)
    out = {}
    for sex_label in ("F", "M"):
        sub = pheno[pheno["sex"] == sex_label]
        g = genotypes[genotypes["subject_id"].isin(sub["subject_id"])]
        out[sex_label] = run_gwas(
            g, sub, covars=covars, outcome=outcome, add_sex=False
        ).set_index("snp")

    f, m = out["F"], out["M"]
    merged = f[["beta", "se", "p"]].join(
        m[["beta", "se", "p"]], lsuffix="_F", rsuffix="_M"
    )
    # z-test for difference in betas between sexes (Cochran-style)
    denom = np.sqrt(merged["se_F"] ** 2 + merged["se_M"] ** 2)
    merged["z_diff"] = (merged["beta_F"] - merged["beta_M"]) / denom
    from scipy.stats import norm

    merged["p_diff"] = 2 * (1 - norm.cdf(merged["z_diff"].abs()))
    return merged.sort_values("p_diff").reset_index()


def genomic_inflation(pvalues: pd.Series) -> float:
    """Lambda_GC: median chi-square / expected median. ~1.0 means well-calibrated."""
    from scipy.stats import chi2

    p = pd.Series(pvalues).dropna().clip(lower=1e-300, upper=1.0)
    obs = chi2.isf(p, df=1)
    return float(np.median(obs) / chi2.isf(0.5, df=1))
