"""Covariate-adjusted logistic GWAS and sex-stratified association."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from admomics import gwas, viz
from admomics.config import DATA_DIR, FIG_DIR, RESULTS_DIR


def main() -> None:
    pheno = pd.read_csv(DATA_DIR / "processed" / "pheno.csv")
    genos = pd.read_parquet(DATA_DIR / "processed" / "genotypes.parquet")

    qc = gwas.variant_qc(genos)
    keep = qc.loc[qc["pass_qc"], "snp"].tolist()
    print(f"QC: {qc['pass_qc'].sum()}/{len(qc)} variants pass")

    genos_qc = genos[["subject_id"] + keep]
    res = gwas.run_gwas(genos_qc, pheno)
    lam = gwas.genomic_inflation(res["p"])
    print(f"Genomic inflation lambda_GC = {lam:.3f}")
    print(res.head(8).round(4).to_string(index=False))

    sexdiff = gwas.sex_stratified_gwas(genos_qc, pheno)
    print("\nTop sex-heterogeneous variants:")
    print(sexdiff.head(6).round(4).to_string(index=False))

    res.to_csv(RESULTS_DIR / "gwas_results.csv", index=False)
    sexdiff.to_csv(RESULTS_DIR / "gwas_sex_stratified.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    viz.qq_plot(res["p"], ax=ax, title=f"GWAS QQ (lambda={lam:.2f})")
    viz.savefig(fig, FIG_DIR / "gwas_qq.png")
    print(f"\nFigure -> {FIG_DIR / 'gwas_qq.png'}")


if __name__ == "__main__":
    main()
