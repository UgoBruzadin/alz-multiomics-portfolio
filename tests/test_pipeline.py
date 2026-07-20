"""End-to-end sanity tests: each module should recover its planted signal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from admomics import apoe, coloc, gwas, integrate, ml, mr, simulate, sumstats
from admomics.config import SimConfig


@pytest.fixture(scope="module")
def cohort():
    # smaller cohort keeps the test suite fast
    cfg = SimConfig(n_subjects=1500, n_common_snps=150, n_genes=120, n_proteins=80)
    return simulate.simulate_cohort(cfg)


# --- simulation -----------------------------------------------------------
def test_cohort_shapes(cohort):
    n = len(cohort.pheno)
    assert cohort.genotypes.shape[0] == n
    assert cohort.expression.shape[0] == n
    assert cohort.proteomics.shape[0] == n
    assert set(cohort.pheno["ancestry"]).issubset({"EUR", "AFR", "EAS", "AMR"})
    assert cohort.pheno["AD"].between(0, 1).all()


def test_apoe_dosage_range(cohort):
    assert cohort.pheno["APOE_e4_dosage"].between(0, 2).all()


# --- APOE stratification --------------------------------------------------
def test_apoe_or_positive_and_ancestry_varies(cohort):
    strat = apoe.stratified_apoe_or(cohort.pheno, strata=("ancestry",))
    # APOE*4 should be a risk allele (OR > 1) overall in most strata
    assert (strat["or_"] > 1).sum() >= len(strat) - 1
    # EUR effect should exceed AFR effect (planted attenuation)
    ors = strat.set_index("ancestry")["or_"]
    if {"EUR", "AFR"}.issubset(ors.index):
        assert ors["EUR"] > ors["AFR"]


def test_apoe_genotype_calling():
    # subject 0: rs429358=T/T, rs7412=T/C -> haplotypes (T,T)=e2 and (T,C)=e3
    # subject 1: rs429358=T/C, rs7412=C/C -> haplotypes (T,C)=e3 and (C,C)=e4
    g = apoe.call_apoe_genotype(pd.Series(["TT", "TC"]), pd.Series(["TC", "CC"]))
    assert g.iloc[0] == "e2/e3"          # order-normalized (sorted)
    assert g.iloc[1] == "e3/e4"


# --- GWAS -----------------------------------------------------------------
def test_gwas_runs_and_calibrated(cohort):
    qc = gwas.variant_qc(cohort.genotypes)
    assert qc["pass_qc"].sum() > 0
    res = gwas.run_gwas(cohort.genotypes, cohort.pheno)
    assert {"beta", "se", "p"}.issubset(res.columns)
    lam = gwas.genomic_inflation(res["p"])
    # null background genome -> inflation near 1
    assert 0.7 < lam < 1.5


# --- coloc ----------------------------------------------------------------
def test_coloc_shared_vs_distinct():
    shared = coloc.coloc_abf(coloc.simulate_locus(shared_causal=True, seed=1))
    distinct = coloc.coloc_abf(coloc.simulate_locus(shared_causal=False, seed=2))
    assert shared["PP.H4"] > 0.5
    assert distinct["PP.H3"] > distinct["PP.H4"]
    for pp in (shared, distinct):
        total = sum(pp[k] for k in ["PP.H0", "PP.H1", "PP.H2", "PP.H3", "PP.H4"])
        assert abs(total - 1.0) < 1e-6


# --- MR -------------------------------------------------------------------
def test_mr_recovers_true_effect():
    inst = mr.simulate_instruments(true_effect=0.3, pleiotropy=0.0, seed=3)
    res = mr.run_mr(inst).set_index("method")
    ivw = res.loc["IVW", "estimate"]
    assert abs(ivw - 0.3) < 0.1  # recovers planted causal effect
    assert res.loc["MR-Egger", "n_snps"] == len(inst)


def test_mr_egger_flags_pleiotropy():
    inst = mr.simulate_instruments(true_effect=0.0, pleiotropy=0.1, seed=5)
    res = mr.run_mr(inst).set_index("method")
    # Egger intercept should be estimated (non-nan)
    assert not np.isnan(res.loc["MR-Egger", "egger_intercept"])


# --- ML integration -------------------------------------------------------
def test_integration_beats_chance_and_fusion_helps(cohort):
    om = integrate.build_omics_matrix(
        cohort.pheno, cohort.expression, cohort.proteomics, genotypes=cohort.genotypes
    )
    bdf = ml.evaluate_blocks(om, model_key="logreg_l2", n_splits=3)
    fusion = bdf.loc[bdf["model"].str.contains("fusion"), "auroc"].iloc[0]
    best_block = bdf.loc[~bdf["model"].str.contains("fusion"), "auroc"].max()
    assert fusion > 0.6  # integration is informative
    # early fusion should be at least competitive with the best single block
    assert fusion >= best_block - 0.03


def test_sex_stratified_performance_runs(cohort):
    om = integrate.build_omics_matrix(
        cohort.pheno, cohort.expression, cohort.proteomics
    )
    perf = ml.sex_stratified_performance(om, n_splits=3)
    assert set(perf["sex"]) == {"F", "M", "overall"}
    assert perf["auroc"].between(0, 1).all()


# --- sumstats -------------------------------------------------------------
def test_example_sumstats_parse():
    ss = sumstats.load_example_sumstats()
    assert {"snp", "beta", "se", "pval", "varbeta"}.issubset(ss.columns)
    assert (ss["pval"].dropna() <= 1).all()
    # planted APOE-region signal should be genome-wide-ish significant
    assert ss["pval"].min() < 1e-6


def test_gwas_catalog_url_format():
    url = sumstats.gwas_catalog_url("GCST90027158")
    assert url.startswith("https://ftp.ebi.ac.uk")
    assert "GCST90027158" in url
