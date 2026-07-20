"""Project-wide configuration: paths, biological constants, and defaults.

Keeping these in one place makes the pipeline reproducible and makes it obvious
which numbers are "knobs" versus real biology.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (repo-relative, resolved at import time)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

for _d in (DATA_DIR, EXTERNAL_DIR, RESULTS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Biology: APOE is defined by two SNPs whose haplotypes give e2/e3/e4.
# rs429358 (C = e4-defining) and rs7412 (T = e2-defining).
# The (rs429358, rs7412) allele pair maps to an epsilon allele:
#   (T, T) -> e2   (T, C) -> e3   (C, C) -> e4   (C, T) -> e1 (rare)
# ---------------------------------------------------------------------------
APOE_SNPS = ("rs429358", "rs7412")

# haplotype (rs429358_allele, rs7412_allele) -> epsilon allele
APOE_HAPLOTYPE_MAP = {
    ("T", "T"): "e2",
    ("T", "C"): "e3",
    ("C", "C"): "e4",
    ("C", "T"): "e1",
}

# Approximate epsilon-allele frequencies in a European-ancestry reference,
# used only to seed the simulator.
APOE_ALLELE_FREQ = {"e2": 0.08, "e3": 0.78, "e4": 0.14}

# Ancestry groups modeled in the simulator. The APOE*4 effect is deliberately
# made ancestry-dependent to mirror Belloy et al. (JAMA Neurology, 2023):
# APOE*4 risk is attenuated in African ancestry and elevated in East Asian
# ancestry relative to European ancestry.
ANCESTRY_GROUPS = ("EUR", "AFR", "EAS", "AMR")

# Per-e4-allele log-odds for AD, by ancestry (illustrative, signal for the demo).
APOE4_LOGOR_BY_ANCESTRY = {
    "EUR": 1.10,   # ~OR 3.0 per allele
    "AFR": 0.55,   # attenuated
    "EAS": 1.35,   # elevated
    "AMR": 0.95,
}
APOE2_LOGOR = -0.45  # protective, roughly ancestry-shared in this demo


@dataclass(frozen=True)
class SimConfig:
    """Parameters controlling the synthetic cohort."""

    n_subjects: int = 4000
    n_common_snps: int = 500          # background genome for GWAS/PCA
    n_genes: int = 300                # transcriptomic layer
    n_proteins: int = 200             # proteomic layer
    n_causal_genes: int = 12          # genes with true effect on AD liability
    ancestry_props: tuple = (0.62, 0.20, 0.10, 0.08)  # EUR, AFR, EAS, AMR
    female_prop: float = 0.58         # AD cohorts skew female
    base_prevalence: float = 0.30
    # sex-dimorphic loci: a subset of causal genes act more strongly in females
    n_sex_biased_genes: int = 4
    sex_bias_logor: float = 0.60      # extra log-OR in the stronger sex
    seed: int = 20260719

    ancestry_groups: tuple = field(default=ANCESTRY_GROUPS)


DEFAULT_SIM = SimConfig()
