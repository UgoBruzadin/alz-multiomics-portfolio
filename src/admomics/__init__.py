"""admomics: a compact, reproducible Alzheimer's disease multi-omics toolkit.

The package is organized around the analytical stack used in AD genetics /
multi-omics labs:

    simulate  -> generate a cohort with genotypes, multi-omics, and phenotype
    sumstats  -> fetch / parse public GWAS summary statistics
    gwas      -> QC + (sex/ancestry-aware) association testing
    apoe      -> APOE genotype calling + stratified risk (APOE x ancestry x sex)
    coloc     -> Bayesian colocalization (approximate Bayes factors)
    mr        -> two-sample Mendelian randomization (IVW / Egger / weighted median)
    integrate -> assemble a subject x multi-omic-feature matrix
    ml        -> cross-validated multi-omics integration models + interpretation

Everything runs on simulated data out of the box, so the full pipeline is
reproducible without access-restricted cohorts (ADSP / ADNI / UK Biobank).
"""

__version__ = "0.1.0"

from . import apoe, coloc, gwas, integrate, ml, mr, simulate, sumstats  # noqa: F401

__all__ = [
    "simulate",
    "sumstats",
    "gwas",
    "apoe",
    "coloc",
    "mr",
    "integrate",
    "ml",
    "__version__",
]
