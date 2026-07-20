"""Simulate a multi-omics AD cohort with biologically motivated, *planted* signal.

Why simulate? Real AD cohorts (ADSP, ADNI, UK Biobank) are access-restricted, so
a portfolio pipeline that anyone can run end-to-end needs synthetic data. The
signal is planted deliberately so that downstream analyses recover known
structure:

  * APOE*4 increases AD risk, and its effect size depends on genetic ancestry
    (attenuated in AFR, elevated in EAS) -- mirroring Belloy et al. 2023.
  * A subset of "causal genes" drive AD liability; some act in a sex-biased way.
  * Transcriptomic and proteomic layers are downstream molecular readouts of the
    causal genes plus noise, so multi-omics integration is genuinely informative.

The output is a bundle of pandas / numpy objects plus a tidy phenotype table.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    APOE4_LOGOR_BY_ANCESTRY,
    APOE2_LOGOR,
    APOE_ALLELE_FREQ,
    ANCESTRY_GROUPS,
    DEFAULT_SIM,
    SimConfig,
)


@dataclass
class Cohort:
    """Container for a simulated multi-omics cohort."""

    pheno: pd.DataFrame          # subject-level: AD, sex, age, ancestry, PCs, APOE
    genotypes: pd.DataFrame      # subjects x common SNPs (0/1/2 dosage)
    expression: pd.DataFrame     # subjects x genes
    proteomics: pd.DataFrame     # subjects x proteins
    causal_genes: list[str]      # ground-truth causal gene names
    sex_biased_genes: list[str]  # ground-truth sex-biased genes
    snp_info: pd.DataFrame       # per-SNP metadata (chrom, pos, ref/alt, MAF)


def _draw_apoe(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw APOE epsilon genotypes as e4 dosage (0/1/2) via allele sampling."""
    alleles = list(APOE_ALLELE_FREQ)
    freqs = np.array([APOE_ALLELE_FREQ[a] for a in alleles])
    freqs = freqs / freqs.sum()
    a1 = rng.choice(alleles, size=n, p=freqs)
    a2 = rng.choice(alleles, size=n, p=freqs)
    e4 = (a1 == "e4").astype(int) + (a2 == "e4").astype(int)
    e2 = (a1 == "e2").astype(int) + (a2 == "e2").astype(int)
    geno = np.array([f"{x}/{y}" for x, y in zip(a1, a2)])
    return e4, e2, geno


def simulate_cohort(cfg: SimConfig = DEFAULT_SIM) -> Cohort:
    """Generate a :class:`Cohort` with reproducible, planted multi-omics signal."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_subjects

    # --- ancestry ---------------------------------------------------------
    ancestry = rng.choice(
        ANCESTRY_GROUPS, size=n, p=np.array(cfg.ancestry_props)
    )

    # --- background genotypes with ancestry-structured allele frequencies --
    # Each ancestry gets its own MAF vector; genotypes are drawn binomially.
    # This creates real population structure that PCA can recover.
    base_maf = rng.uniform(0.05, 0.5, size=cfg.n_common_snps)
    anc_shift = {
        a: rng.normal(0, 0.08, size=cfg.n_common_snps) for a in ANCESTRY_GROUPS
    }
    geno = np.zeros((n, cfg.n_common_snps), dtype=np.int8)
    for a in ANCESTRY_GROUPS:
        idx = np.where(ancestry == a)[0]
        maf_a = np.clip(base_maf + anc_shift[a], 0.01, 0.99)
        geno[idx] = rng.binomial(2, maf_a, size=(len(idx), cfg.n_common_snps))

    snp_ids = [f"rs{1000000 + i}" for i in range(cfg.n_common_snps)]
    snp_info = pd.DataFrame(
        {
            "snp": snp_ids,
            "chrom": rng.integers(1, 23, size=cfg.n_common_snps),
            "pos": rng.integers(1, 250_000_000, size=cfg.n_common_snps),
            "maf": base_maf,
        }
    )
    genotypes = pd.DataFrame(geno, columns=snp_ids)

    # --- ancestry principal components (top 4) ---------------------------
    # Standardize and SVD to get genotype PCs, the standard ancestry covariate.
    g = genotypes.to_numpy(dtype=float)
    g = (g - g.mean(0)) / (g.std(0) + 1e-8)
    u, s, _ = np.linalg.svd(g, full_matrices=False)
    pcs = u[:, :4] * s[:4]
    pcs = (pcs - pcs.mean(0)) / (pcs.std(0) + 1e-8)

    # --- APOE + demographics ---------------------------------------------
    e4, e2, apoe_geno = _draw_apoe(rng, n)
    sex = rng.choice(["F", "M"], size=n, p=[cfg.female_prop, 1 - cfg.female_prop])
    age = np.clip(rng.normal(72, 8, size=n), 55, 95)

    # --- causal genes drive a latent AD liability ------------------------
    gene_ids = [f"GENE{i:03d}" for i in range(cfg.n_genes)]
    causal_idx = rng.choice(cfg.n_genes, size=cfg.n_causal_genes, replace=False)
    causal_genes = [gene_ids[i] for i in causal_idx]
    sex_biased_genes = list(
        rng.choice(causal_genes, size=cfg.n_sex_biased_genes, replace=False)
    )

    # latent "true" activity of each causal gene per subject
    true_activity = rng.normal(0, 1, size=(n, cfg.n_causal_genes))
    effect_sizes = rng.normal(0.5, 0.15, size=cfg.n_causal_genes) * rng.choice(
        [-1, 1], size=cfg.n_causal_genes
    )

    # --- assemble log-odds of AD -----------------------------------------
    lin = np.full(n, np.log(cfg.base_prevalence / (1 - cfg.base_prevalence)))
    lin += 0.04 * (age - 72)                       # age effect

    # APOE effects, ancestry-dependent for e4 (the Belloy signature)
    e4_logor = np.array([APOE4_LOGOR_BY_ANCESTRY[a] for a in ancestry])
    lin += e4_logor * e4
    lin += APOE2_LOGOR * e2

    # causal-gene effects, with sex bias on a subset
    is_female = (sex == "F").astype(float)
    for j, gene in enumerate(causal_genes):
        contrib = effect_sizes[j] * true_activity[:, j]
        if gene in sex_biased_genes:
            contrib = contrib * (1 + cfg.sex_bias_logor * is_female)
        lin += contrib

    prob = 1 / (1 + np.exp(-lin))
    ad = rng.binomial(1, prob)

    # --- molecular layers are downstream readouts of causal activity ------
    # Expression: causal genes reflect true activity (+ noise); others noise.
    expr = rng.normal(0, 1, size=(n, cfg.n_genes))
    for j, ci in enumerate(causal_idx):
        expr[:, ci] = true_activity[:, j] + rng.normal(0, 0.6, size=n)
    expression = pd.DataFrame(expr, columns=gene_ids)

    # Proteomics: a noisy, partly-overlapping molecular layer. The first
    # n_causal proteins mirror causal-gene activity with more measurement noise
    # (proteomics is noisier but closer to phenotype -> complementary signal).
    protein_ids = [f"PROT{i:03d}" for i in range(cfg.n_proteins)]
    prot = rng.normal(0, 1, size=(n, cfg.n_proteins))
    for j in range(min(cfg.n_causal_genes, cfg.n_proteins)):
        prot[:, j] = 0.8 * true_activity[:, j] + rng.normal(0, 0.9, size=n)
    proteomics = pd.DataFrame(prot, columns=protein_ids)

    # --- tidy phenotype table --------------------------------------------
    pheno = pd.DataFrame(
        {
            "subject_id": [f"S{i:05d}" for i in range(n)],
            "AD": ad.astype(int),
            "sex": sex,
            "age": np.round(age, 1),
            "ancestry": ancestry,
            "APOE_geno": apoe_geno,
            "APOE_e4_dosage": e4,
            "APOE_e2_dosage": e2,
            "PC1": pcs[:, 0],
            "PC2": pcs[:, 1],
            "PC3": pcs[:, 2],
            "PC4": pcs[:, 3],
        }
    )

    for df in (genotypes, expression, proteomics):
        df.insert(0, "subject_id", pheno["subject_id"].to_numpy())

    return Cohort(
        pheno=pheno,
        genotypes=genotypes,
        expression=expression,
        proteomics=proteomics,
        causal_genes=causal_genes,
        sex_biased_genes=sex_biased_genes,
        snp_info=snp_info,
    )


def save_cohort(cohort: Cohort, out_dir) -> None:
    """Persist a cohort to parquet/csv for downstream scripts."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cohort.pheno.to_csv(out / "pheno.csv", index=False)
    cohort.genotypes.to_parquet(out / "genotypes.parquet", index=False)
    cohort.expression.to_parquet(out / "expression.parquet", index=False)
    cohort.proteomics.to_parquet(out / "proteomics.parquet", index=False)
    cohort.snp_info.to_csv(out / "snp_info.csv", index=False)
    pd.Series(cohort.causal_genes).to_csv(
        out / "causal_genes.csv", index=False, header=["gene"]
    )
    pd.Series(cohort.sex_biased_genes).to_csv(
        out / "sex_biased_genes.csv", index=False, header=["gene"]
    )
