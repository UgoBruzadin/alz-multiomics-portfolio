"""Colocalization + Mendelian randomization: from association to causal target.

The GWAS side uses public AD summary statistics (bundled example offline; the
real Bellenguez/Wightman files via scripts/download_sumstats.py). The molecular
QTL side is simulated, illustrating the lab's proteogenomic causal-inference
workflow end to end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from admomics import coloc, mr
from admomics.config import RESULTS_DIR
from admomics.sumstats import load_example_sumstats


def main() -> None:
    # --- colocalization: shared vs distinct causal variants ---------------
    shared = coloc.simulate_locus(shared_causal=True, seed=1)
    distinct = coloc.simulate_locus(shared_causal=False, seed=2)
    pp_shared = coloc.coloc_abf(shared)
    pp_distinct = coloc.coloc_abf(distinct)
    print("Coloc (shared causal variant):")
    print({k: round(v, 3) for k, v in pp_shared.items()})
    print("Coloc (distinct causal variants):")
    print({k: round(v, 3) for k, v in pp_distinct.items()})

    # tie coloc to a real public GWAS locus (APOE region in the example file)
    ss = load_example_sumstats().dropna(subset=["beta", "se"])
    top = ss.sort_values("pval").head(1)
    print(f"\nPublic-sumstats lead variant: {top['snp'].iloc[0]} "
          f"(chr{int(top['chrom'].iloc[0])}:{int(top['pos'].iloc[0])}, "
          f"p={top['pval'].iloc[0]:.2e})")

    # --- MR: causal effect of a protein exposure on AD --------------------
    instruments = mr.simulate_instruments(true_effect=0.30, pleiotropy=0.0, seed=3)
    mr_res = mr.run_mr(instruments)
    print("\nMR (true causal log-OR = 0.30):")
    print(mr_res.round(4).to_string(index=False))

    # a pleiotropic case where Egger intercept should flag bias
    plei = mr.simulate_instruments(true_effect=0.0, pleiotropy=0.08, seed=4)
    mr_plei = mr.run_mr(plei)
    print("\nMR with directional pleiotropy (true effect = 0):")
    print(mr_plei.round(4).to_string(index=False))

    pd.DataFrame([pp_shared, pp_distinct],
                 index=["shared", "distinct"]).to_csv(RESULTS_DIR / "coloc_results.csv")
    mr_res.to_csv(RESULTS_DIR / "mr_results.csv", index=False)
    mr_plei.to_csv(RESULTS_DIR / "mr_pleiotropy_results.csv", index=False)


if __name__ == "__main__":
    main()
