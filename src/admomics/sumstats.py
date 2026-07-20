"""Fetch and parse public GWAS summary statistics.

Real, published AD GWAS summary statistics are openly available (no application
needed) from the GWAS Catalog and the study authors. We provide a downloader and
a harmonizing parser so the coloc / MR modules can run against *real* GWAS
effect sizes, while the omics side stays simulated.

Recommended public AD GWAS sumstats (all open-access):
  * Bellenguez et al. 2022, Nat Genet -- GWAS Catalog GCST90027158
  * Wightman et al. 2021, Nat Genet   -- GWAS Catalog GCST90012877
  * Kunkle  et al. 2019, Nat Genet    -- IGAP stage 1

Because sandboxed CI has no network access to EBI, ``load_example_sumstats``
returns a small bundled table so the pipeline is fully reproducible offline; the
downloader is what you run locally to pull the full files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import EXTERNAL_DIR

# canonical column names we harmonize everything to
CANONICAL = ["snp", "chrom", "pos", "effect_allele", "other_allele",
             "beta", "se", "pval", "eaf"]

# Public AD GWAS catalog identifiers (harmonized files live under these accessions)
GWAS_CATALOG = {
    "bellenguez2022": "GCST90027158",
    "wightman2021": "GCST90012877",
}


def gwas_catalog_url(accession: str) -> str:
    """Build the GWAS Catalog harmonized-sumstats FTP URL for an accession.

    Accession like 'GCST90027158' -> the range bucket + harmonized .tsv.gz path.
    Run the downloader locally; EBI is not reachable from CI sandboxes.
    """
    num = int(accession.replace("GCST", ""))
    lower = (num // 1000) * 1000 + 1
    upper = lower + 999
    bucket = f"GCST{lower:08d}-GCST{upper:08d}"
    return (
        "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
        f"{bucket}/{accession}/harmonised/{accession}.h.tsv.gz"
    )


def download_sumstats(study: str, dest: Path | None = None) -> Path:
    """Download harmonized sumstats for a known study key (network required)."""
    import urllib.request

    if study not in GWAS_CATALOG:
        raise KeyError(f"Unknown study '{study}'. Known: {list(GWAS_CATALOG)}")
    url = gwas_catalog_url(GWAS_CATALOG[study])
    dest = Path(dest or (EXTERNAL_DIR / f"{study}.h.tsv.gz"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (documented, user-run)
    return dest


# Map common GWAS-Catalog harmonized columns to our canonical schema.
_COLMAP = {
    "hm_rsid": "snp", "variant_id": "snp", "rsid": "snp",
    "hm_chrom": "chrom", "chromosome": "chrom",
    "hm_pos": "pos", "base_pair_location": "pos",
    "hm_effect_allele": "effect_allele", "effect_allele": "effect_allele",
    "hm_other_allele": "other_allele", "other_allele": "other_allele",
    "hm_beta": "beta", "beta": "beta",
    "standard_error": "se", "se": "se",
    "p_value": "pval", "p": "pval", "pval": "pval",
    "hm_effect_allele_frequency": "eaf", "effect_allele_frequency": "eaf",
}


def parse_sumstats(path_or_df) -> pd.DataFrame:
    """Read and harmonize a sumstats file (or DataFrame) to the canonical schema."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df, sep=None, engine="python")
    df = df.rename(columns={c: _COLMAP.get(c, c) for c in df.columns})
    for col in CANONICAL:
        if col not in df.columns:
            df[col] = np.nan
    df = df[CANONICAL]
    df["varbeta"] = df["se"] ** 2
    return df


def load_example_sumstats() -> pd.DataFrame:
    """Return the small bundled example sumstats (offline-safe)."""
    path = EXTERNAL_DIR / "example_ad_sumstats.tsv"
    if not path.exists():
        write_example_sumstats(path)
    return parse_sumstats(path)


def write_example_sumstats(path: Path | None = None, seed: int = 7) -> Path:
    """Create a tiny, realistic AD-locus sumstats file for offline demos.

    Includes an APOE-region signal (chr19) and a nearby null region so coloc /
    lookups have something meaningful to work with without downloading GB files.
    """
    rng = np.random.default_rng(seed)
    path = Path(path or (EXTERNAL_DIR / "example_ad_sumstats.tsv"))
    n = 200
    pos = np.sort(rng.integers(44_900_000, 45_500_000, n))  # APOE region, hg38-ish
    eaf = rng.uniform(0.05, 0.5, n)
    se = 1.0 / np.sqrt(2 * eaf * (1 - eaf) * 60000)
    beta = rng.normal(0, se)
    # plant a strong APOE-region signal
    lead = np.argmin(np.abs(pos - 45_411_941))  # ~APOE
    beta[lead] += 8 * se[lead]
    for off in (-2, -1, 1, 2):
        j = np.clip(lead + off, 0, n - 1)
        beta[j] += 4 * se[j]
    z = beta / se
    from scipy.stats import norm

    pval = 2 * norm.sf(np.abs(z))
    df = pd.DataFrame(
        {
            "snp": [f"rs{9000000 + i}" for i in range(n)],
            "chrom": 19,
            "pos": pos,
            "effect_allele": rng.choice(list("ACGT"), n),
            "other_allele": rng.choice(list("ACGT"), n),
            "beta": beta,
            "se": se,
            "pval": pval,
            "eaf": eaf,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return path
