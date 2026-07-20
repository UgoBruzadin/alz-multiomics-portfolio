"""Multi-omics ML integration -- the analytical centerpiece.

Compares per-block, early-fusion, and late-fusion (stacked) integration under
stratified CV; then interprets the winning model with block ablation, grouped
permutation importance, and sex-stratified performance.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from admomics import integrate, ml, viz
from admomics.config import DATA_DIR, FIG_DIR, RESULTS_DIR


def main() -> None:
    proc = DATA_DIR / "processed"
    pheno = pd.read_csv(proc / "pheno.csv")
    expr = pd.read_parquet(proc / "expression.parquet")
    prot = pd.read_parquet(proc / "proteomics.parquet")
    genos = pd.read_parquet(proc / "genotypes.parquet")

    om = integrate.build_omics_matrix(pheno, expr, prot, genotypes=genos)
    print(f"Design matrix: {om.X.shape[0]} subjects x {om.X.shape[1]} features")
    print(f"Blocks: { {k: len(v) for k, v in om.blocks.items()} }")

    # 1) per-block vs early fusion, across model families
    all_rows = []
    for model_key in ("logreg_l2", "random_forest"):
        bdf = ml.evaluate_blocks(om, model_key=model_key)
        bdf["family"] = model_key
        all_rows.append(bdf)
    block_df = pd.concat(all_rows, ignore_index=True)
    print("\nIntegration performance (AUROC):")
    print(block_df.round(3).to_string(index=False))

    # 2) late fusion / stacking
    late = ml.late_fusion_stack(om)
    print(f"\nLate fusion (stacked) AUROC: {late.auroc:.3f} "
          f"[{late.auroc_ci[0]:.3f}, {late.auroc_ci[1]:.3f}]")

    # 3) block ablation
    abl = ml.block_ablation(om)
    print("\nBlock ablation (leave-one-omic-out):")
    print(abl.round(3).to_string(index=False))

    # 4) grouped permutation importance
    imp = ml.permutation_importance_grouped(om, n_repeats=4, n_splits=4, top_n=20)
    print("\nTop features by permutation importance:")
    print(imp.round(4).to_string(index=False))

    # 5) sex-stratified performance
    sex_perf = ml.sex_stratified_performance(om)
    print("\nSex-stratified AUROC:")
    print(sex_perf.round(3).to_string(index=False))

    # persist
    block_df.to_csv(RESULTS_DIR / "ml_integration_blocks.csv", index=False)
    abl.to_csv(RESULTS_DIR / "ml_block_ablation.csv", index=False)
    imp.to_csv(RESULTS_DIR / "ml_permutation_importance.csv", index=False)
    sex_perf.to_csv(RESULTS_DIR / "ml_sex_performance.csv", index=False)

    # figures
    lr = block_df[block_df["family"] == "logreg_l2"].copy()
    lr = pd.concat([lr, pd.DataFrame([late.as_row() | {"family": "logreg_l2"}])],
                   ignore_index=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    viz.block_auroc_bar(lr, ax=ax)
    viz.savefig(fig, FIG_DIR / "ml_integration_auroc.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    viz.importance_bar(imp, ax=ax)
    viz.savefig(fig, FIG_DIR / "ml_permutation_importance.png")
    print(f"\nFigures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
