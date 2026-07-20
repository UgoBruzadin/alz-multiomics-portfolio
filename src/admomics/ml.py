"""Machine-learning multi-omics integration -- the analytical centerpiece.

The question is not just "can we predict AD?" but "how do omic layers combine,
and which features / blocks drive the signal?" We implement the three canonical
integration strategies and evaluate them with nested-safe, stratified CV:

  * per-block models        -- baseline: each omic alone
  * early fusion            -- concatenate all blocks, one model
  * late fusion (stacking)  -- per-block models + a meta-learner

We add the interpretation tooling a reviewer expects: cross-validated AUROC/AUPRC
with confidence intervals, permutation importance, block ablation, and
sex-stratified performance (relevant to the lab's sex-dimorphism focus).

Design choices that translate directly from EEG / signal-processing ML:
  * leakage-safe pipelines (scaling fit inside each CV fold),
  * stratified resampling and CI estimation over folds,
  * feature-group attribution rather than single-feature over-interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .integrate import OmicsMatrix


def make_pipeline(model=None) -> Pipeline:
    """A leakage-safe pipeline: impute -> scale -> classifier, per fold."""
    if model is None:
        model = LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced"
        )
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", model),
        ]
    )


MODELS = {
    "logreg_l2": lambda: LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced"
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=150, max_depth=12, n_jobs=-1, class_weight="balanced_subsample",
        random_state=0,
    ),
    "grad_boost": lambda: GradientBoostingClassifier(random_state=0),
}


@dataclass
class CVResult:
    label: str
    auroc: float
    auroc_ci: tuple
    auprc: float
    fold_auroc: list = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "model": self.label,
            "auroc": self.auroc,
            "auroc_lo": self.auroc_ci[0],
            "auroc_hi": self.auroc_ci[1],
            "auprc": self.auprc,
        }


def _ci(vals, alpha=0.05):
    vals = np.asarray(vals)
    lo = np.percentile(vals, 100 * alpha / 2)
    hi = np.percentile(vals, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def cv_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory=None,
    n_splits: int = 5,
    seed: int = 0,
    label: str = "model",
) -> CVResult:
    """Stratified k-fold evaluation returning AUROC (with fold-CI) and AUPRC."""
    model_factory = model_factory or MODELS["logreg_l2"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    Xv, yv = X.to_numpy(dtype=float), y.to_numpy()
    fold_auc, oof_true, oof_score = [], [], []
    for tr, te in skf.split(Xv, yv):
        pipe = make_pipeline(model_factory())
        pipe.fit(Xv[tr], yv[tr])
        score = pipe.predict_proba(Xv[te])[:, 1]
        fold_auc.append(roc_auc_score(yv[te], score))
        oof_true.append(yv[te])
        oof_score.append(score)
    oof_true = np.concatenate(oof_true)
    oof_score = np.concatenate(oof_score)
    return CVResult(
        label=label,
        auroc=float(np.mean(fold_auc)),
        auroc_ci=_ci(fold_auc),
        auprc=float(average_precision_score(oof_true, oof_score)),
        fold_auroc=[float(a) for a in fold_auc],
    )


def evaluate_blocks(
    om: OmicsMatrix, model_key: str = "logreg_l2", n_splits: int = 5, seed: int = 0
) -> pd.DataFrame:
    """Per-block vs early-fusion performance -- the core integration comparison."""
    rows = []
    for block, cols in om.blocks.items():
        res = cv_evaluate(
            om.X[cols], om.y, MODELS[model_key], n_splits, seed, label=f"{block} only"
        )
        rows.append(res.as_row())
    res_all = cv_evaluate(
        om.X, om.y, MODELS[model_key], n_splits, seed, label="early fusion (all omics)"
    )
    rows.append(res_all.as_row())
    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)


def late_fusion_stack(
    om: OmicsMatrix, n_splits: int = 5, seed: int = 0
) -> CVResult:
    """Late fusion: per-block base learners feed a logistic meta-learner.

    Base-model out-of-fold predictions become meta-features, avoiding leakage.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = om.y.to_numpy()
    blocks = list(om.blocks)
    fold_auc, oof_true, oof_score = [], [], []
    for tr, te in skf.split(om.X.to_numpy(), y):
        # inner OOF predictions on the training split for each block
        meta_tr = np.zeros((len(tr), len(blocks)))
        meta_te = np.zeros((len(te), len(blocks)))
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for bi, block in enumerate(blocks):
            Xb = om.X[om.blocks[block]].to_numpy(dtype=float)
            # build OOF preds for the training rows
            for itr, ite in inner.split(Xb[tr], y[tr]):
                pipe = make_pipeline(MODELS["logreg_l2"]())
                pipe.fit(Xb[tr][itr], y[tr][itr])
                meta_tr[ite, bi] = pipe.predict_proba(Xb[tr][ite])[:, 1]
            # fit on full training split, predict test split
            pipe = make_pipeline(MODELS["logreg_l2"]())
            pipe.fit(Xb[tr], y[tr])
            meta_te[:, bi] = pipe.predict_proba(Xb[te])[:, 1]
        meta = LogisticRegression(max_iter=2000, class_weight="balanced")
        meta.fit(meta_tr, y[tr])
        score = meta.predict_proba(meta_te)[:, 1]
        fold_auc.append(roc_auc_score(y[te], score))
        oof_true.append(y[te])
        oof_score.append(score)
    return CVResult(
        "late fusion (stacked)",
        float(np.mean(fold_auc)),
        _ci(fold_auc),
        float(average_precision_score(np.concatenate(oof_true),
                                      np.concatenate(oof_score))),
        [float(a) for a in fold_auc],
    )


def block_ablation(
    om: OmicsMatrix, model_key: str = "logreg_l2", n_splits: int = 5, seed: int = 0
) -> pd.DataFrame:
    """Leave-one-block-out: how much AUROC is lost by dropping each omic layer."""
    full = cv_evaluate(om.X, om.y, MODELS[model_key], n_splits, seed, "full").auroc
    rows = [{"dropped_block": "(none)", "auroc": full, "delta": 0.0}]
    for block in om.blocks:
        keep = [c for b, cols in om.blocks.items() if b != block for c in cols]
        auc = cv_evaluate(
            om.X[keep], om.y, MODELS[model_key], n_splits, seed, f"drop {block}"
        ).auroc
        rows.append({"dropped_block": block, "auroc": auc, "delta": auc - full})
    return pd.DataFrame(rows)


def permutation_importance_grouped(
    om: OmicsMatrix,
    model_key: str = "logreg_l2",
    n_repeats: int = 10,
    n_splits: int = 5,
    seed: int = 0,
    top_n: int = 20,
) -> pd.DataFrame:
    """Permutation importance aggregated to individual features and their block.

    Importance = drop in held-out AUROC when a feature is shuffled, averaged over
    CV folds and repeats. Reported with the owning omic block for interpretation.
    """
    rng = np.random.default_rng(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X, y = om.X.to_numpy(dtype=float), om.y.to_numpy()
    cols = list(om.X.columns)
    imp = np.zeros(len(cols))
    counts = 0
    for tr, te in skf.split(X, y):
        pipe = make_pipeline(MODELS[model_key]())
        pipe.fit(X[tr], y[tr])
        base = roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1])
        Xte = X[te].copy()
        for j in range(len(cols)):
            drops = []
            for _ in range(n_repeats):
                saved = Xte[:, j].copy()
                Xte[:, j] = rng.permutation(saved)
                drops.append(base - roc_auc_score(y[te], pipe.predict_proba(Xte)[:, 1]))
                Xte[:, j] = saved
            imp[j] += np.mean(drops)
        counts += 1
    imp /= counts
    blocks = om.feature_blocks
    out = (
        pd.DataFrame({"feature": cols, "importance": imp})
        .assign(block=lambda d: d["feature"].map(blocks))
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return out.head(top_n)


def sex_stratified_performance(
    om: OmicsMatrix, model_key: str = "logreg_l2", n_splits: int = 5, seed: int = 0
) -> pd.DataFrame:
    """Train on all, evaluate AUROC within each sex -- checks for disparity.

    Relevant to the lab's sex-dimorphism focus: a model can integrate omics well
    overall but perform unevenly across sexes, which matters for equitable risk
    prediction.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X, y = om.X.to_numpy(dtype=float), om.y.to_numpy()
    sex = om.meta["sex"].to_numpy()
    scores = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        pipe = make_pipeline(MODELS[model_key]())
        pipe.fit(X[tr], y[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]
    rows = []
    for s in ("F", "M"):
        m = sex == s
        rows.append(
            {"sex": s, "n": int(m.sum()), "n_cases": int(y[m].sum()),
             "auroc": float(roc_auc_score(y[m], scores[m]))}
        )
    rows.append(
        {"sex": "overall", "n": len(y), "n_cases": int(y.sum()),
         "auroc": float(roc_auc_score(y, scores))}
    )
    return pd.DataFrame(rows)
