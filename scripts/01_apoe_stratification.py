"""APOE*4 risk stratified by ancestry and sex, plus a formal interaction test.

Reproduces the *shape* of Belloy et al. (JAMA Neurology, 2023): the APOE*4 effect
on AD is ancestry-dependent.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from admomics import apoe, viz
from admomics.config import DATA_DIR, FIG_DIR, RESULTS_DIR


def main() -> None:
    pheno = pd.read_csv(DATA_DIR / "processed" / "pheno.csv")

    by_anc = apoe.stratified_apoe_or(pheno, strata=("ancestry",))
    by_anc_sex = apoe.stratified_apoe_or(pheno, strata=("ancestry", "sex"))
    inter = apoe.apoe_ancestry_interaction(pheno)

    print("APOE*4 OR by ancestry:")
    print(by_anc.round(3).to_string(index=False))
    print("\nAncestry x APOE*4 interaction (vs EUR):")
    print(inter.round(4).to_string(index=False))

    by_anc.to_csv(RESULTS_DIR / "apoe_or_by_ancestry.csv", index=False)
    by_anc_sex.to_csv(RESULTS_DIR / "apoe_or_by_ancestry_sex.csv", index=False)
    inter.to_csv(RESULTS_DIR / "apoe_ancestry_interaction.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    viz.apoe_forest(by_anc, ax=ax)
    viz.savefig(fig, FIG_DIR / "apoe_or_by_ancestry.png")
    print(f"\nFigure -> {FIG_DIR / 'apoe_or_by_ancestry.png'}")


if __name__ == "__main__":
    main()
