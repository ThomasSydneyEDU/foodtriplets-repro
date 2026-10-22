#!/usr/bin/env python3
"""compute_family_generalization.py

Stage 5 (Compute): Leave-pairs-out CV family-blocked regression for FoodTriplets2.

Runs cross-validated Non-Negative Least Squares (NNLS) regression separately
for each model-family block and additive combination thereof, to quantify how
much variance in the held-out empirical RDM is explained by each family alone
and in combination.

Cross-validation is performed over stimulus pairs on a single fixed-effects
group RDM (all participants pooled): each fold holds out a disjoint subset of
observed pairs, fits weights on the remaining pairs, and scores the held-out
pairs. This keeps the unit of analysis consistent with the univariate,
partial-Mantel and Shapley analyses (earlier versions cross-validated over
participants).

NNLS (scipy.optimize.nnls) is used instead of OLS to constrain all model
weights to β ≥ 0. Within-family RDMs are often highly collinear (e.g.
AI Savoriness and AI Saltiness ρ=0.76), and unconstrained OLS assigns large
negative weights to suppress collinear predictors — theoretically incoherent
in RSA, where a negative weight implies that more-different items on a feature
are judged more similar. NNLS assigns zero to the weaker of two collinear
predictors instead, which is the standard approach in modern RSA literature
(e.g. Khaligh-Razavi & Kriegeskorte, 2014).

Family structure
----------------
  ai_appraisal    : AI calorie_density, AI healthiness
  ai_flavour      : AI sweetness, saltiness, sourness, bitterness,
                    savoriness, fattiness, spiciness
  human_appraisal : Human calorie_density, healthiness, appeal
  human_flavour   : Human sweetness, saltiness, sourness, bitterness,
                    savoriness, fattiness, spiciness
  visual          : ResNet-50 conv1 PCs (ll_resnet50_conv1_pcs10)
  food_classif.   : food_classification (standalone categorical)

Blocks run
----------
Individual families + additive combinations:
  - each family alone
  - ai_appraisal + ai_flavour  (and +visual, full AI)
  - human_appraisal + human_flavour  (and +visual, full Human)
  - food_classification + visual
  - ai_appraisal + ai_flavour + food_classification  (semantic + category)
  - ai_full + food_classification                     (full AI + category)
  - human equivalents of the above two
  - human_full + transformation                       (transformation beyond
                                                       semantic + visual)

Outputs (written to compute_outputs/stage5_family_generalization/)
------------------------------------------------------------------
  cv_family_regression_fits.csv
      fold, model_set, display_label, n_models, rho_test, r2_test, n_pairs
  cv_family_manifest.json
      provenance: model stems, block definitions, run settings

Usage
-----
  python analysis_code/compute_family_generalization.py
  python analysis_code/compute_family_generalization.py --kfold 5 --seed 42
  python analysis_code/compute_family_generalization.py --outdir custom/path/

  # Run only specific blocks (fast isolation mode):
  python analysis_code/compute_family_generalization.py \\
      --only-blocks ai_appraisal+ai_flavour+food_classification,ai_full+food_classification \\
      --append

  # Run and append only the no-category transformation comparison:
  python analysis_code/compute_family_generalization.py \\
      --transformation-beyond-semantic-visual-only

  # --append merges new block results into the existing CSV rather than overwriting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
SUMMARY_CSV     = HERE / "derivatives" / "summary.csv"
MODEL_RDM_DIR   = HERE / "derivatives" / "model_rdms_canon"
DEFAULT_OUTDIR  = HERE / "compute_outputs" / "stage5_family_generalization"

# ---------------------------------------------------------------------------
# Family membership  (NPZ stem → family key)
# ---------------------------------------------------------------------------
FAMILIES: Dict[str, List[str]] = {
    "ai_appraisal": [
        "rdm_ai_calorie_density",
        "rdm_ai_healthiness",
    ],
    "ai_flavour": [
        "rdm_ai_sweetness",
        "rdm_ai_saltiness",
        "rdm_ai_sourness",
        "rdm_ai_bitterness",
        "rdm_ai_savoriness",
        "rdm_ai_fattiness",
        "rdm_ai_spiciness",
    ],
    "human_appraisal": [
        "rdm_human_calorie_density",
        "rdm_human_healthiness",
        "rdm_human_appeal",
    ],
    "human_flavour": [
        "rdm_human_sweetness",
        "rdm_human_saltiness",
        "rdm_human_sourness",
        "rdm_human_bitterness",
        "rdm_human_savoriness",
        "rdm_human_fattiness",
        "rdm_human_spiciness",
    ],
    "visual": [
        "rdm_ll_resnet50_conv1_pcs10",
        "rdm_ll_hsv_s_mean",
        "rdm_ll_lbp_hist",
        "rdm_ll_resnet50_layer3_pcs10",
    ],
    "food_classification": [
        "rdm_food_classification",
    ],
    # --- Transformation families -------------------------------------------
    # REVISION 2026-09-10 (revision/transformation-models): NOVA_ordinal is no
    # longer the sole operationalisation of food transformation. Each measure is
    # its own family so it can be entered alone or alongside the others, and the
    # three together form the "transformation" family for a family-wise test that
    # is immune to their mutual collinearity (rho(NvT, Transformation_score) = 0.85).
    "NOVA_ordinal": [
        "rdm_NOVA_ordinal",
    ],
    "natural_vs_transformed": [
        "rdm_natural_vs_transformed",
    ],
    "Transformation_score": [
        "rdm_Transformation_score",
    ],
    "transformation": [
        "rdm_NOVA_ordinal",
        "rdm_natural_vs_transformed",
        "rdm_Transformation_score",
    ],
    # --- Alternative taxonomies (symmetric reporting; reviewer point RB-4a) ---
    "Category_WHO_10": [
        "rdm_Category_WHO_10",
    ],
    "Category_Culinary_9": [
        "rdm_Category_Culinary_9",
    ],
}

# Cross-check family membership against the canonical policy lists so this module
# cannot drift from the univariate scan / variance attribution.
try:
    from reference.policy_defaults import (
        TRANSFORMATION_MODELS as _POLICY_TRANSFORMATION,
        TAXONOMY_MODELS as _POLICY_TAXONOMY,
    )
    _declared = {k for k in FAMILIES if k in set(_POLICY_TRANSFORMATION) | set(_POLICY_TAXONOMY)}
    _expected = set(_POLICY_TRANSFORMATION) | set(_POLICY_TAXONOMY)
    _missing = _expected - _declared
    if _missing:
        raise ValueError(
            "compute_family_generalization.FAMILIES is missing policy models: "
            f"{sorted(_missing)}. Add them here and to BLOCKS below."
        )
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Blocks: (model_set_key, display_label, [family_keys_to_combine])
# ---------------------------------------------------------------------------
BLOCKS: List[Tuple[str, str, List[str]]] = [
    # --- AI ---
    ("ai_appraisal",
     "AI Appraisal",
     ["ai_appraisal"]),
    ("ai_flavour",
     "AI Sensory",
     ["ai_flavour"]),
    ("ai_appraisal+ai_flavour",
     "AI Appraisal + Sensory",
     ["ai_appraisal", "ai_flavour"]),
    ("ai_appraisal+visual",
     "AI Appraisal + Visual",
     ["ai_appraisal", "visual"]),
    ("ai_flavour+visual",
     "AI Sensory + Visual",
     ["ai_flavour", "visual"]),
    ("ai_full",
     "AI Full (Appraisal + Sensory + Visual)",
     ["ai_appraisal", "ai_flavour", "visual"]),
    # --- Human ---
    ("human_appraisal",
     "Human Appraisal",
     ["human_appraisal"]),
    ("human_flavour",
     "Human Sensory",
     ["human_flavour"]),
    ("human_appraisal+human_flavour",
     "Human Appraisal + Sensory",
     ["human_appraisal", "human_flavour"]),
    ("human_appraisal+visual",
     "Human Appraisal + Visual",
     ["human_appraisal", "visual"]),
    ("human_flavour+visual",
     "Human Sensory + Visual",
     ["human_flavour", "visual"]),
    ("human_full",
     "Human Full (Appraisal + Sensory + Visual)",
     ["human_appraisal", "human_flavour", "visual"]),
    # --- Visual standalone ---
    ("visual",
     "Visual",
     ["visual"]),
    # --- Food classification ---
    ("food_classification",
     "Food Classification",
     ["food_classification"]),
    ("food_classification+visual",
     "Food Classification + Visual",
     ["food_classification", "visual"]),
    # --- Semantic + Category combinations (hierarchy test) ---
    ("ai_appraisal+ai_flavour+food_classification",
     "AI Semantic + Category",
     ["ai_appraisal", "ai_flavour", "food_classification"]),
    ("ai_full+food_classification",
     "AI Full + Category",
     ["ai_appraisal", "ai_flavour", "visual", "food_classification"]),
    ("human_appraisal+human_flavour+food_classification",
     "Human Semantic + Category",
     ["human_appraisal", "human_flavour", "food_classification"]),
    ("human_full+food_classification",
     "Human Full + Category",
     ["human_appraisal", "human_flavour", "visual", "food_classification"]),
    # --- NOVA_ordinal (processing level; Monteiro et al. 2016) ---
    ("NOVA_ordinal",
     "NOVA Processing Level",
     ["NOVA_ordinal"]),
    ("human_full+NOVA_ordinal",
     "Human Appraisal + Sensory + Visual + NOVA",
     ["human_appraisal", "human_flavour", "visual", "NOVA_ordinal"]),
    ("ai_full+NOVA_ordinal",
     "AI Appraisal + Sensory + Visual + NOVA",
     ["ai_appraisal", "ai_flavour", "visual", "NOVA_ordinal"]),
    ("human_full+food_classification+NOVA_ordinal",
     "Human Appraisal + Sensory + Visual + Category + NOVA",
     ["human_appraisal", "human_flavour", "visual", "food_classification", "NOVA_ordinal"]),
    ("ai_full+food_classification+NOVA_ordinal",
     "AI Appraisal + Sensory + Visual + Category + NOVA",
     ["ai_appraisal", "ai_flavour", "visual", "food_classification", "NOVA_ordinal"]),

    # --- REVISION 2026-09-10 (revision/transformation-models) -----------------
    # Each transformation measure alone; all three jointly beyond human semantic
    # + visual predictors; and each measure (plus the joint family) beyond the
    # full human semantic + visual + category model. The no-category joint block
    # separates overlap with continuous predictors from overlap with category.
    ("natural_vs_transformed",
     "Natural vs Transformed",
     ["natural_vs_transformed"]),
    ("Transformation_score",
     "Transformation Score",
     ["Transformation_score"]),
    ("transformation",
     "Transformation (all three)",
     ["transformation"]),
    ("human_full+transformation",
     "Human Appraisal + Sensory + Visual + Transformation (all three)",
     ["human_appraisal", "human_flavour", "visual", "transformation"]),
    ("human_full+food_classification+natural_vs_transformed",
     "Human Full + Category + Natural vs Transformed",
     ["human_appraisal", "human_flavour", "visual", "food_classification", "natural_vs_transformed"]),
    ("human_full+food_classification+Transformation_score",
     "Human Full + Category + Transformation Score",
     ["human_appraisal", "human_flavour", "visual", "food_classification", "Transformation_score"]),
    ("human_full+food_classification+transformation",
     "Human Full + Category + Transformation (all three)",
     ["human_appraisal", "human_flavour", "visual", "food_classification", "transformation"]),

    # --- Alternative taxonomies (symmetric reporting; RB-4a) ------------------
    # The same category increment computed under each scheme, so the choice of
    # taxonomy is shown to be presentational rather than inferential.
    ("Category_WHO_10",
     "Category (WHO 10)",
     ["Category_WHO_10"]),
    ("Category_Culinary_9",
     "Category (Culinary 9)",
     ["Category_Culinary_9"]),
    ("human_full+Category_WHO_10",
     "Human Full + Category (WHO 10)",
     ["human_appraisal", "human_flavour", "visual", "Category_WHO_10"]),
    ("human_full+Category_Culinary_9",
     "Human Full + Category (Culinary 9)",
     ["human_appraisal", "human_flavour", "visual", "Category_Culinary_9"]),
    ("human_full+Category_WHO_10+transformation",
     "Human Full + Category (WHO 10) + Transformation",
     ["human_appraisal", "human_flavour", "visual", "Category_WHO_10", "transformation"]),
    ("human_full+Category_Culinary_9+transformation",
     "Human Full + Category (Culinary 9) + Transformation",
     ["human_appraisal", "human_flavour", "visual", "Category_Culinary_9", "transformation"]),
]

# ---------------------------------------------------------------------------
# AI RATING FAMILIES — REMOVED FROM THE PIPELINE (2026-09-11)
#
# The ai_* rating models are not reported in the manuscript (see the note in
# reference/policy_defaults.py). compute_model_evaluation drops them via
# EXCLUDE_MODELS, but this module builds its families and blocks from its own
# literals above, so it would otherwise keep fitting AI blocks that nothing
# consumes and writing AI rows into cv_family_summary.csv.
#
# The definitions above are left in place rather than deleted so the family
# composition stays readable and the removal is one line to undo. Filtering
# here means any block that references an AI family disappears with it.
_AI_FAMILIES = ("ai_appraisal", "ai_flavour")
FAMILIES = {k: v for k, v in FAMILIES.items() if k not in _AI_FAMILIES}
BLOCKS = [b for b in BLOCKS if not any(f in _AI_FAMILIES for f in b[2])]


# ---------------------------------------------------------------------------
# Utilities  (self-contained — adapted from compute_model_generalization.py)
# ---------------------------------------------------------------------------

@dataclass
class EmpiricalRDM:
    D: np.ndarray
    labels: List[str]


def _vectorize_upper(D: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    iu, ju = np.triu_indices_from(D, k=1)
    return D[iu, ju].astype(float), iu, ju


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return np.nan
    xs = pd.Series(x[m]).rank(method="average").to_numpy(dtype=float)
    ys = pd.Series(y[m]).rank(method="average").to_numpy(dtype=float)
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    denom = float(np.sqrt((xs * xs).sum()) * np.sqrt((ys * ys).sum()))
    return float((xs * ys).sum() / denom) if denom > 0 else np.nan


def _to_bool_series(s: pd.Series) -> pd.Series:
    def _cell(v) -> bool:
        if v is None:
            return False
        try:
            if pd.isna(v):
                return False
        except Exception:
            pass
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v) != 0
        if isinstance(v, (float, np.floating)):
            return float(v) != 0.0
        return str(v).strip().lower() in {"1", "1.0", "true", "t", "yes", "y", "on"}
    return s.fillna(False).apply(_cell).astype(bool)


def _build_rdm(df: pd.DataFrame, labels: Optional[List[str]] = None) -> EmpiricalRDM:
    needed = {"StimA_file", "StimB_file", "StimC_file", "choice_file"}
    if not needed.issubset(df.columns):
        raise KeyError(f"summary missing columns: {needed - set(df.columns)}")

    def _base(x: object) -> str:
        return os.path.basename(str(x))

    if labels is None:
        labs: set = set()
        for col in ("StimA_file", "StimB_file", "StimC_file", "choice_file"):
            labs.update(_base(v) for v in df[col].dropna().astype(str) if v.strip())
        labels = sorted(labs)

    n = len(labels)
    idx_of = {fn: i for i, fn in enumerate(labels)}
    sim = np.zeros((n, n), dtype=float)
    cnt = np.zeros((n, n), dtype=float)

    for _, r in df.iterrows():
        if pd.isna(r["choice_file"]):
            continue
        a, b, c = _base(r["StimA_file"]), _base(r["StimB_file"]), _base(r["StimC_file"])
        choice = _base(r["choice_file"])
        if a not in idx_of or b not in idx_of or c not in idx_of:
            continue
        ia, ib, ic = idx_of[a], idx_of[b], idx_of[c]
        if choice == a:
            i, j = sorted((ib, ic))
        elif choice == b:
            i, j = sorted((ia, ic))
        elif choice == c:
            i, j = sorted((ia, ib))
        else:
            continue
        sim[i, j] += 1.0
        for p1, p2 in ((ia, ib), (ia, ic), (ib, ic)):
            if p1 != p2:
                u, v = sorted((p1, p2))
                cnt[u, v] += 1.0

    sim += sim.T;  cnt += cnt.T
    with np.errstate(invalid="ignore", divide="ignore"):
        p_sim = np.where(cnt > 0, sim / cnt, np.nan)
    D = 1.0 - p_sim
    np.fill_diagonal(D, np.nan)
    return EmpiricalRDM(D=D, labels=labels)


def _load_npz(path: Path) -> Tuple[np.ndarray, List[str]]:
    z = np.load(str(path), allow_pickle=True)
    key = next((k for k in ("data", "D", "rdm") if k in z.files), None)
    if key is None:
        raise KeyError(f"{path}: no data key found. Keys={z.files}")
    D = np.asarray(z[key]).astype(float)
    if "labels" not in z.files:
        raise KeyError(f"{path}: missing 'labels'")
    labels = [str(x) for x in np.asarray(z["labels"]).astype(str).tolist()]
    return D, labels


def _align(D: np.ndarray, src_labels: List[str], tgt_labels: List[str]) -> np.ndarray:
    idx = {str(l): i for i, l in enumerate(src_labels)}
    missing = [l for l in tgt_labels if str(l) not in idx]
    if missing:
        raise KeyError(f"RDM missing target labels: {missing[:5]}")
    order = [idx[str(l)] for l in tgt_labels]
    return D[np.ix_(order, order)].astype(float)


def _split_kfold_pairs(pair_indices: np.ndarray, k: int, seed: int
                       ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Leave-pairs-out CV: split observed pair indices into k disjoint test folds.

    Returns a list of (train_idx, test_idx) arrays. Each observed pair appears in
    exactly one test fold; training uses all other observed pairs. This performs
    cross-validation over stimulus pairs on a single fixed-effects group RDM,
    rather than over participants, keeping the unit of analysis consistent with
    the univariate, partial-Mantel and Shapley analyses.
    """
    idx = np.asarray(pair_indices)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idx)
    test_folds = np.array_split(perm, k)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        te = test_folds[i]
        if k > 1:
            tr = np.concatenate([test_folds[j] for j in range(k) if j != i])
        else:
            tr = te
        splits.append((np.sort(tr), np.sort(te)))
    return splits


def _ols_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
                 ridge_alpha: float = 0.0) -> np.ndarray:
    """OLS prediction (kept for reference; not used in primary analyses)."""
    n, p = X_tr.shape
    Xb = np.hstack([X_tr, np.ones((n, 1))])
    Xbt = np.hstack([X_te, np.ones((X_te.shape[0], 1))])
    A = Xb.T @ Xb
    if ridge_alpha > 0:
        A[:p, :p] += ridge_alpha * np.eye(p)
    b = Xb.T @ y_tr
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(Xb, y_tr, rcond=None)[0]
    return Xbt @ beta


def _nnls_fit(X: np.ndarray, y: np.ndarray,
              max_iter: int = 3000, tol: float = 1e-10) -> np.ndarray:
    """Non-negative least squares: scipy if available, else projected gradient.

    Returns beta (weights only; intercept is handled externally by the caller).
    """
    try:
        from scipy.optimize import nnls as _sp
        beta, _ = _sp(X, y)
        return beta
    except ImportError:
        pass
    ATA  = X.T @ X
    ATy  = X.T @ y
    step = 1.0 / (np.linalg.norm(ATA, "fro") + 1e-12)
    beta = np.zeros(X.shape[1], dtype=float)
    for _ in range(max_iter):
        b_new = np.maximum(0.0, beta - step * (ATA @ beta - ATy))
        if np.linalg.norm(b_new - beta) < tol:
            return b_new
        beta = b_new
    return beta


def _nnls_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    """Predict using Non-Negative Least Squares (NNLS).

    Constrains all model weights to β ≥ 0, preventing theoretically incoherent
    negative weights on collinear RDM predictors. A free intercept is handled by
    centering y on the training mean before fitting and adding it back afterward,
    so the non-negativity constraint applies only to the model predictors.
    """
    y_mean = float(y_tr.mean())
    beta   = _nnls_fit(X_tr, y_tr - y_mean)
    return X_te @ beta + y_mean


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary",    default=str(SUMMARY_CSV))
    p.add_argument("--model-rdms", default=str(MODEL_RDM_DIR))
    p.add_argument("--outdir",     default=str(DEFAULT_OUTDIR))
    p.add_argument("--kfold",      type=int, default=5)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--min-pairs",  type=int, default=50)
    p.add_argument("--pool",       default="all",
                   choices=["all", "prolific", "sona"],
                   help="Participant pool to include (default: all)")
    p.add_argument("--fixed-dense-only", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep only fixed+dense trials (default: True). "
                        "Pass --no-fixed-dense-only to use all trials.")
    p.add_argument(
        "--empirical-rdm",
        type=str,
        default=None,
        help="Optional path to an empirical triplet RDM .npz to use instead of building from summary.csv.",
    )
    p.add_argument(
        "--use-completed-nn-seed",
        type=int,
        default=None,
        help="Convenience option: use derivatives/triplet_rdm_completed_nn_seed{SEED}.npz as the empirical RDM.",
    )
    block_selection = p.add_mutually_exclusive_group()
    block_selection.add_argument(
        "--only-blocks",
        default=None,
        help="Comma-separated list of block keys to run (e.g. "
             "'human_full,human_full+food_classification'). "
             "When set, only the listed blocks are executed. "
             "Combine with --append to merge into the existing CSV.",
    )
    block_selection.add_argument(
        "--transformation-beyond-semantic-visual-only",
        action="store_true",
        default=False,
        help="Run only human_full and human_full+transformation, testing the joint "
             "transformation family beyond semantic and visual predictors without "
             "category. Results are merged into the existing CSV automatically.",
    )
    p.add_argument("--append", action="store_true", default=False,
                   help="Append new block results to the existing CSV rather than "
                        "overwriting it. Rows for blocks being re-run are replaced.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.transformation_beyond_semantic_visual_only:
        args.only_blocks = "human_full,human_full+transformation"
        # This convenience mode adds the new contrast to an existing full
        # analysis without discarding any previously computed blocks.
        args.append = True

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Resolve Empirical RDM Source
    # ------------------------------------------------------------------
    empirical_path = None
    if args.use_completed_nn_seed is not None:
        empirical_path = HERE / "derivatives" / f"triplet_rdm_completed_nn_seed{args.use_completed_nn_seed}.npz"
    elif args.empirical_rdm is not None:
        empirical_path = Path(args.empirical_rdm)

    external_rdm = None
    if empirical_path is not None:
        if not empirical_path.exists():
            print(f"[ERROR] Empirical RDM not found: {empirical_path}")
            return 1
        D_ext, labels_ext = _load_npz(empirical_path)
        external_rdm = EmpiricalRDM(D=D_ext, labels=labels_ext)
        print(f"[INFO] Using external empirical RDM: {empirical_path} ({len(labels_ext)} items)")

    # ------------------------------------------------------------------
    # 2. Load summary and identify subjects (if building from trials)
    # ------------------------------------------------------------------
    summary_path = Path(args.summary)
    df = None
    subjects = []
    if external_rdm is None:
        if not summary_path.exists():
            print(f"[ERROR] summary.csv not found: {summary_path}")
            return 1
        df = pd.read_csv(summary_path, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]

        # Pool filter
        if args.pool != "all" and "pool" in df.columns:
            df = df[df["pool"].astype(str).str.lower() == args.pool.lower()].copy()

        # Fixed+dense filter
        if args.fixed_dense_only and ("is_fixed" in df.columns or "is_dense" in df.columns):
            is_fixed = _to_bool_series(df["is_fixed"]) if "is_fixed" in df.columns \
                       else pd.Series(False, index=df.index)
            is_dense = _to_bool_series(df["is_dense"]) if "is_dense" in df.columns \
                       else pd.Series(False, index=df.index)
            df = df[is_fixed | is_dense].copy()

        if "SubID" not in df.columns:
            print("[ERROR] summary.csv missing SubID column")
            return 1
        df["SubID"] = df["SubID"].astype(str).str.strip()
        df = df[df["SubID"].notna() & (df["SubID"] != "")].copy()

        subjects = sorted(df["SubID"].unique().tolist())
        print(f"[INFO] Subjects: {len(subjects)} | Trials: {len(df)}")
    else:
        print("[INFO] Skipping summary loading (using external RDM).")

    # ------------------------------------------------------------------
    # 3. Build global label universe
    # ------------------------------------------------------------------
    if external_rdm is not None:
        labels = external_rdm.labels
    else:
        rdm_full = _build_rdm(df)
        labels = rdm_full.labels
    
    n_items = len(labels)
    print(f"[INFO] Food items: {n_items}")

    # ------------------------------------------------------------------
    # 4. Resolve which blocks to run
    # ------------------------------------------------------------------
    if args.only_blocks:
        requested = {k.strip() for k in args.only_blocks.split(",")}
        all_keys  = {bk for bk, _, _ in BLOCKS}
        unknown   = requested - all_keys
        if unknown:
            print(f"[ERROR] Unknown block key(s): {unknown}")
            print(f"        Available: {sorted(all_keys)}")
            return 1
        active_blocks = [(bk, lbl, fks) for bk, lbl, fks in BLOCKS
                         if bk in requested]
        print(f"[INFO] Isolation mode — running {len(active_blocks)} block(s): "
              f"{[bk for bk,_,_ in active_blocks]}")
    else:
        active_blocks = BLOCKS

    # ------------------------------------------------------------------
    # 5. Load model RDMs needed by the active blocks
    # ------------------------------------------------------------------
    model_dir = Path(args.model_rdms)
    needed_stems: set[str] = set()
    for _, _, fam_keys in active_blocks:
        for fk in fam_keys:
            if fk in FAMILIES:
                needed_stems.update(FAMILIES[fk])

    loaded_rdms: Dict[str, np.ndarray] = {}
    for stem in sorted(needed_stems):
        npz_path = model_dir / f"{stem}.npz"
        if not npz_path.exists():
            # Try with rdm_ prefix if not present
            if not stem.startswith("rdm_"):
                npz_path = model_dir / f"rdm_{stem}.npz"
            
            if not npz_path.exists():
                print(f"[WARN] Model RDM not found, skipping: {stem}")
                continue

        try:
            D_raw, m_labels = _load_npz(npz_path)
            loaded_rdms[stem] = _align(D_raw, m_labels, labels)
        except Exception as e:
            print(f"[WARN] Could not load/align {stem}: {e}")
    print(f"[INFO] Loaded {len(loaded_rdms)}/{len(needed_stems)} model RDMs")

    # Check the families used by the active blocks. In isolation mode this
    # avoids warnings about deliberately unrequested model families.
    needed_family_keys = {
        fk for _, _, fam_keys in active_blocks for fk in fam_keys
    }
    available_families: set[str] = set()
    for fk, stems in FAMILIES.items():
        if fk not in needed_family_keys:
            continue
        if all(s in loaded_rdms for s in stems):
            available_families.add(fk)
        else:
            missing = [s for s in stems if s not in loaded_rdms]
            print(f"[WARN] Family '{fk}' incomplete — missing: {missing}")

    # ------------------------------------------------------------------
    # 6. Build leave-pairs-out CV splits (or single shot for external RDM)
    # ------------------------------------------------------------------
    # Cross-validation is performed over stimulus pairs on a single
    # fixed-effects group RDM (all participants pooled), consistent with the
    # univariate, partial-Mantel and Shapley analyses. Each fold holds out a
    # disjoint subset of observed pairs; weights are fit on the remaining pairs
    # and scored on the held-out pairs. This replaces the earlier
    # participant-fold scheme, which conflated model generalisation with
    # group-RDM reliability.
    iu_global, ju_global = np.triu_indices(n_items, k=1)
    n_pairs_total = iu_global.size
    fold_specs: List[Tuple[np.ndarray, np.ndarray, str]] = []

    if external_rdm is not None:
        print("[INFO] Single-shot evaluation on external RDM (no cross-validation).")
        v_all = external_rdm.D[iu_global, ju_global]
        full_mask = np.ones(n_pairs_total, dtype=bool)
        fold_specs.append((full_mask, full_mask, "full"))
    else:
        v_all = rdm_full.D[iu_global, ju_global]
        obs_pair_idx = np.flatnonzero(np.isfinite(v_all))
        print(f"[INFO] Building {args.kfold}-fold leave-pairs-out CV over "
              f"{obs_pair_idx.size:,} observed pairs...")
        pair_folds = _split_kfold_pairs(obs_pair_idx, k=args.kfold, seed=args.seed)
        for fold_idx, (tr_idx, te_idx) in enumerate(pair_folds, start=1):
            train_mask = np.zeros(n_pairs_total, dtype=bool)
            test_mask  = np.zeros(n_pairs_total, dtype=bool)
            train_mask[tr_idx] = True
            test_mask[te_idx]  = True
            fold_specs.append((train_mask, test_mask, f"fold_{fold_idx}"))
            print(f"  Fold {fold_idx}: train_pairs={tr_idx.size:,}  "
                  f"test_pairs={te_idx.size:,}")

    # ------------------------------------------------------------------
    # 7. Run CV for each block (fold RDMs already cached)
    # ------------------------------------------------------------------
    rows: List[Dict[str, object]] = []

    for block_key, display_label, fam_keys in active_blocks:
        # Check all families for this block are available
        if not all(fk in available_families for fk in fam_keys):
            skip = [fk for fk in fam_keys if fk not in available_families]
            print(f"[SKIP] Block '{block_key}' — missing families: {skip}")
            continue

        # Collect model vectors for this block (aligned, upper-triangle)
        block_stems: List[str] = []
        for fk in fam_keys:
            block_stems.extend(FAMILIES[fk])
        
        block_vecs_list = []
        for s in block_stems:
            if s in loaded_rdms:
                block_vecs_list.append(loaded_rdms[s][iu_global, ju_global])
        
        if not block_vecs_list:
            continue
            
        block_vecs = np.vstack(block_vecs_list).T  # (n_pairs, n_models)
        n_models = len(block_vecs_list)
        ok_mod   = np.all(np.isfinite(block_vecs), axis=1)

        print(f"[INFO] Block '{block_key}' | {n_models} models | families: {fam_keys}")

        for train_mask, test_mask, fold_tag in fold_specs:
            # Disjoint train/test pairs that are observed in the group RDM and
            # finite across all model predictors in this block.
            tr_ok = train_mask & ok_mod & np.isfinite(v_all)
            te_ok = test_mask  & ok_mod & np.isfinite(v_all)

            n_tr = int(tr_ok.sum())
            n_te = int(te_ok.sum())
            if n_tr < args.min_pairs or n_te < args.min_pairs:
                rows.append({
                    "fold": fold_tag,
                    "model_set": block_key,
                    "display_label": display_label,
                    "n_models": n_models,
                    "rho_test": np.nan,
                    "r2_test": np.nan,
                    "n_pairs": n_te,
                })
                continue

            X_tr = block_vecs[tr_ok]
            X_te = block_vecs[te_ok]
            y_tr = v_all[tr_ok]
            y_te = v_all[te_ok]

            # Standardize predictors on TRAIN-pair statistics, apply to test
            mu = X_tr.mean(axis=0)
            sd = X_tr.std(axis=0)
            sd[sd == 0] = 1.0
            Xz_tr = (X_tr - mu) / sd
            Xz_te = (X_te - mu) / sd

            # NNLS fit on train pairs, predict held-out pairs
            yhat   = _nnls_predict(Xz_tr, y_tr, Xz_te)
            rho_te = _spearman(yhat, y_te)
            r2_te  = _r2(y_te, yhat)

            rows.append({
                "fold": fold_tag,
                "model_set": block_key,
                "display_label": display_label,
                "n_models": n_models,
                "rho_test": float(rho_te) if np.isfinite(rho_te) else np.nan,
                "r2_test": float(r2_te) if np.isfinite(r2_te) else np.nan,
                "n_pairs": n_te,
            })

        # Print fold summary for this block
        block_rows = [r for r in rows if r["model_set"] == block_key]
        mean_rho = float(np.nanmean([r["rho_test"] for r in block_rows]))
        mean_r2  = float(np.nanmean([r["r2_test"]  for r in block_rows]))
        print(f"  → mean ρ={mean_rho:.3f}  mean R²={mean_r2:.3f}")

    # ------------------------------------------------------------------
    # 8. Write outputs
    # ------------------------------------------------------------------
    new_df   = pd.DataFrame(rows)
    # Handle output folder naming based on RDM mode
    rdm_mode_tag = "empirical" if external_rdm is None else "filled"
    if args.fixed_dense_only and external_rdm is None:
        rdm_mode_tag = "fixed_dense"
    
    outdir_final = outdir / rdm_mode_tag
    outdir_final.mkdir(parents=True, exist_ok=True)
    
    csv_path = outdir_final / "cv_family_regression_fits.csv"

    if args.append and csv_path.exists():
        existing = pd.read_csv(csv_path)
        # Drop any old rows for blocks we just re-ran (avoid duplicates)
        new_keys = set(new_df["model_set"].unique())
        existing = existing[~existing["model_set"].isin(new_keys)]
        out_df = pd.concat([existing, new_df], ignore_index=True)
        print(f"\n[INFO] Appended {len(new_df)} rows to existing "
              f"{len(existing)} rows ({len(new_keys)} block(s) refreshed)")
    else:
        out_df = new_df

    out_df.to_csv(csv_path, index=False)
    print(f"[OK] Wrote: {csv_path}")

    # Summary table (mean ± SD across folds) — covers ALL blocks present in CSV
    summary_rows = []
    for block_key, display_label, fam_keys in BLOCKS:
        sub = out_df[out_df["model_set"] == block_key]
        if sub.empty:
            continue
        rhos = sub["rho_test"].dropna().to_numpy()
        r2s  = sub["r2_test"].dropna().to_numpy()
        # n_models might vary if append is used, just take the first
        n_mod = int(sub["n_models"].iloc[0])
        summary_rows.append({
            "model_set": block_key,
            "display_label": display_label,
            "n_models": n_mod,
            "rho_mean": float(np.mean(rhos)) if len(rhos) else np.nan,
            "rho_sd":   float(np.std(rhos))  if len(rhos) else np.nan,
            "r2_mean":  float(np.mean(r2s))  if len(r2s)  else np.nan,
            "r2_sd":    float(np.std(r2s))   if len(r2s)  else np.nan,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path_out = outdir_final / "cv_family_summary.csv"
    summary_df.to_csv(summary_path_out, index=False)
    print(f"[OK] Wrote: {summary_path_out}")

    # Manifest
    manifest = {
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "cv_scheme":        "single_shot" if external_rdm is not None else "leave_pairs_out",
        "kfold":            args.kfold,
        "seed":             args.seed,
        "pool":             args.pool,
        "fixed_dense_only": args.fixed_dense_only,
        "rdm_mode":         rdm_mode_tag,
        "n_subjects":       len(subjects),
        "n_items":          n_items,
        "mode":             ("transformation_beyond_semantic_visual_only"
                             if args.transformation_beyond_semantic_visual_only
                             else "append" if (args.append and args.only_blocks)
                             else "full"),
        "blocks_run":       [bk for bk, _, _ in active_blocks],
        "families":         FAMILIES,
        "blocks":           [{"key": k, "label": l, "families": f} for k, l, f in BLOCKS],
        "outputs": {
            "cv_family_regression_fits": str(csv_path),
            "cv_family_summary":         str(summary_path_out),
        },
    }
    manifest_path = outdir_final / "cv_family_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[OK] Wrote: {manifest_path}")

    # Print final summary table
    print("\n=== Family CV summary (mean ρ across folds) ===")
    for _, row in summary_df.iterrows():
        print(f"  {row['display_label']:<45s}  ρ={row['rho_mean']:.3f} ± {row['rho_sd']:.3f}"
              f"  R²={row['r2_mean']:.3f} ± {row['r2_sd']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
