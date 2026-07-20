"""Download real public AD GWAS summary statistics (run locally; needs network).

Usage:
    python scripts/download_sumstats.py bellenguez2022
    python scripts/download_sumstats.py wightman2021

These are open-access files from the EBI GWAS Catalog -- no data application
needed. The coloc/MR pipeline will then use real GWAS effect sizes.
"""
from __future__ import annotations

import sys

from admomics.sumstats import GWAS_CATALOG, download_sumstats, gwas_catalog_url


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in GWAS_CATALOG:
        print("Available studies:")
        for k, acc in GWAS_CATALOG.items():
            print(f"  {k:16s} {acc}  {gwas_catalog_url(acc)}")
        print("\nUsage: python scripts/download_sumstats.py <study>")
        return
    study = sys.argv[1]
    print(f"Downloading {study} from {gwas_catalog_url(GWAS_CATALOG[study])}")
    path = download_sumstats(study)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
