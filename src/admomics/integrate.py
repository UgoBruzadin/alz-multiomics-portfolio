"""Assemble a subject x multi-omic feature matrix for integrative modeling.

This is the plumbing that turns separate genetic / transcriptomic / proteomic
tables into a single aligned design matrix, while tracking which "block" each
feature belongs to. Block membership is what lets us later ask *which omic layer*
carries the predictive signal -- the central question in multi-omics integration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OmicsMatrix:
    X: pd.DataFrame            # subjects x features, index = subject_id
    y: pd.Series              # outcome aligned to X
    blocks: dict[str, list]   # block name -> feature columns
    meta: pd.DataFrame        # subject-level covariates (sex, ancestry, ...)

    @property
    def feature_blocks(self) -> pd.Series:
        """A Series mapping every feature to its omic block."""
        m = {}
        for block, cols in self.blocks.items():
            for c in cols:
                m[c] = block
        return pd.Series(m, name="block")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("subject_id") if "subject_id" in df.columns else df


def build_omics_matrix(
    pheno: pd.DataFrame,
    expression: pd.DataFrame,
    proteomics: pd.DataFrame,
    genotypes: pd.DataFrame | None = None,
    apoe_features: bool = True,
    outcome: str = "AD",
) -> OmicsMatrix:
    """Combine omic layers into one aligned matrix keyed by subject_id.

    Genotype block, if included, is restricted to a manageable set of variants
    to keep the demo fast; in practice you'd feed a polygenic feature set or
    fine-mapped variants.
    """
    ph = pheno.set_index("subject_id")
    expr = _clean(expression)
    prot = _clean(proteomics)

    blocks: dict[str, list] = {}
    parts = []

    # transcriptomics
    expr = expr.add_prefix("expr__")
    blocks["transcriptomics"] = list(expr.columns)
    parts.append(expr)

    # proteomics
    prot = prot.add_prefix("prot__")
    blocks["proteomics"] = list(prot.columns)
    parts.append(prot)

    # genetics (optional): APOE dosages + a slice of common variants
    if apoe_features:
        gen = ph[["APOE_e4_dosage", "APOE_e2_dosage"]].add_prefix("gen__")
        blocks.setdefault("genetics", []).extend(gen.columns)
        parts.append(gen)
    if genotypes is not None:
        g = _clean(genotypes)
        g = g.iloc[:, :50].add_prefix("gen__")  # a manageable slice
        blocks.setdefault("genetics", []).extend(g.columns)
        parts.append(g)

    X = pd.concat(parts, axis=1)
    X = X.loc[ph.index]  # align order
    y = ph[outcome]
    meta = ph[["sex", "age", "ancestry", "PC1", "PC2", "PC3", "PC4"]]
    return OmicsMatrix(X=X, y=y, blocks=blocks, meta=meta)
