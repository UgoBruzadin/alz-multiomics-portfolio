# From EEG / signal processing to genetics & multi-omics

This document is for reviewers evaluating whether a computational-neuroscience
background (EEG, signal processing, computational modeling) transfers to
population genetics, functional genomics, and multi-omics. The short answer is
that the *statistical and engineering core is shared*; the domain-specific
vocabulary is the part that is learned quickly. Below I make the mapping
explicit, and point to where each skill is exercised in this repository.

## The transferable core

| Skill I already use in EEG / signal work | Its genetics / multi-omics counterpart | Where in this repo |
|---|---|---|
| High-dimensional inference across thousands of channels/time-frequency features | Genome-wide / transcriptome-wide association across 10⁴–10⁷ features | `gwas.run_gwas`, `ml.permutation_importance_grouped` |
| Multiple-comparison control (cluster correction, FDR over the time-frequency plane) | Genome-wide significance, FDR over variants/genes | `gwas` p-values, `gwas.genomic_inflation` (λ_GC calibration) |
| Controlling nuisance structure (referencing, artifact/ICA components, subject random effects) | Controlling population stratification with genotype PCs and covariates | `gwas` covariate model, `simulate` PCA-derived ancestry PCs |
| Leakage-safe cross-validation for subject-level decoding | Identical CV discipline for polygenic / multi-omic prediction | `ml.make_pipeline`, `ml.cv_evaluate`, `ml.late_fusion_stack` |
| Feature *group* attribution (which band / region drives decoding) rather than over-reading single features | Which *omic block* / gene set drives prediction | `ml.evaluate_blocks`, `ml.block_ablation`, grouped importances |
| Multimodal fusion (EEG + MEG + behavior) | Multi-omics fusion (genetics + transcriptomics + proteomics) | `integrate.build_omics_matrix`, early vs late fusion in `ml` |
| Generative / forward modeling (simulate signals to validate a decoder) | Simulating genotypes + molecular layers with planted effects to validate a pipeline | `simulate.simulate_cohort`, `coloc.simulate_locus`, `mr.simulate_instruments` |
| Reproducible, tested analysis pipelines | Same — arguably higher stakes given consortium data | `tests/`, CI, packaging |

## What is genuinely new — and how I'm closing the gap

Being honest about the delta matters more than pretending there isn't one. The
genuinely domain-specific pieces are:

1. **Genetic data formats and tooling** — PLINK/BGEN/VCF, imputation panels,
   REGENIE/SAIGE for biobank-scale mixed models. This repo implements the
   *statistical model* (covariate-adjusted logistic association) in Python so the
   logic is transparent; in a lab setting I would run the same model in
   PLINK2/REGENIE. Learning the file formats and job-scheduling is days, not
   months, once the statistics are familiar.
2. **Causal-inference genetics** — colocalization and Mendelian randomization.
   These are new *names* for machinery I find intuitive: coloc is a Bayesian
   model-comparison over causal-variant configurations (`coloc.py`); MR is
   instrumental-variables regression with pleiotropy diagnostics (`mr.py`). I
   implemented both from summary statistics here to show I understand them at the
   level of the estimator, not just the software.
3. **Molecular biology context** — cell-type-specific effects, QTL resources
   (eQTL/pQTL/sQTL), pathway interpretation. This is reading and mentorship, and
   the part I am most eager to learn on the job.

## Why the ML-integration angle is the right contribution

Multi-omics integration is fundamentally a **high-dimensional, multimodal signal
problem** — exactly the shape of problem EEG decoding trains you for. The hard
parts are not "run a classifier": they are avoiding leakage across correlated
molecular layers, attributing signal to the right block rather than to a
correlated confounder, calibrating uncertainty over folds, and checking that
performance is equitable across strata (e.g. sex). Those are the habits this
repo demonstrates, and they are the habits a signal-processing background
instills. The centerpiece script (`scripts/04_ml_integration.py`) is where that
argument is made concretely.
