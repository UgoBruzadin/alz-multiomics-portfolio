# Data sources

## Why the cohort is simulated

Individual-level AD multi-omics data live in access-restricted repositories:

- **ADSP** (Alzheimer's Disease Sequencing Project) — dbGaP application required.
- **ADNI** — data-use agreement and application required.
- **UK Biobank** — approved application and access fee required.

A portfolio pipeline that anyone can `git clone && make all` cannot ship those
data. So `admomics.simulate` generates a synthetic cohort with **planted,
biologically-motivated signal**:

- APOE\*4 raises AD risk with an **ancestry-dependent effect size**
  (attenuated in AFR, elevated in EAS), so `apoe` recovers a realistic gradient.
- A set of **causal genes** drives a latent AD liability; a subset act in a
  **sex-biased** way, so the sex-stratified and integration analyses have real
  structure to find.
- Transcriptomic and proteomic layers are **downstream molecular readouts** of
  the causal genes plus noise, making multi-omics integration genuinely
  informative (and the two layers partially complementary).

Because the ground truth is known, the test suite can assert that each method
recovers it — a stronger correctness guarantee than eyeballing plots.

## Real public GWAS summary statistics (open access, no application)

The `coloc` and `mr` modules can consume **published, open-access** AD GWAS
summary statistics. These are aggregate statistics, not individual-level data,
and are freely downloadable from the EBI GWAS Catalog:

| Study | Catalog accession | Notes |
|---|---|---|
| Bellenguez et al. 2022, *Nat Genet* | `GCST90027158` | Largest EADB AD/dementia GWAS meta-analysis |
| Wightman et al. 2021, *Nat Genet* | `GCST90012877` | Large AD GWAS meta-analysis |
| Kunkle et al. 2019, *Nat Genet* | IGAP stage 1 | Classic AD GWAS |

Download locally with:

```bash
python scripts/download_sumstats.py bellenguez2022
python scripts/download_sumstats.py wightman2021
```

`admomics.sumstats.parse_sumstats` harmonizes the GWAS-Catalog harmonized columns
(`hm_beta`, `standard_error`, `p_value`, …) into a canonical schema
(`snp, chrom, pos, effect_allele, other_allele, beta, se, pval, eaf, varbeta`).

> The offline example (`data/external/example_ad_sumstats.tsv`) is a small,
> synthetic APOE-region file so the pipeline and CI run without network access.
> It contains a planted chr19/APOE signal so lookups and colocalization return
> something meaningful.

## Extending to real molecular QTLs

For a real analysis, the simulated QTL layer would be replaced with public
molecular QTL resources — e.g. brain eQTLs (ROSMAP/AMP-AD, MetaBrain, GTEx) and
plasma/brain pQTLs — matched to the GWAS on `chrom:pos:alleles` before running
`coloc` and `mr`. The harmonization interface in `sumstats.py` is the intended
entry point for that swap.
