"""Generate and persist the synthetic multi-omics cohort."""
from __future__ import annotations

from admomics.config import DATA_DIR, DEFAULT_SIM
from admomics.simulate import save_cohort, simulate_cohort


def main() -> None:
    print(f"Simulating cohort (n={DEFAULT_SIM.n_subjects}) ...")
    cohort = simulate_cohort(DEFAULT_SIM)
    out = DATA_DIR / "processed"
    save_cohort(cohort, out)
    ph = cohort.pheno
    print(f"  AD prevalence: {ph['AD'].mean():.3f}")
    print(f"  ancestry mix : {ph['ancestry'].value_counts().to_dict()}")
    print(f"  female frac  : {(ph['sex'] == 'F').mean():.3f}")
    print(f"  causal genes : {cohort.causal_genes}")
    print(f"  sex-biased   : {cohort.sex_biased_genes}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
