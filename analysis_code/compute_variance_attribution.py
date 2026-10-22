#!/usr/bin/env python3
"""compute_variance_attribution.py

Stage 4 (Compute): Variance attribution for triplet RDM regression predictors.

Purpose
-------
Provides a conservative complement to univariate correlations by estimating
*unique* and *incremental* explained variance of predictor RDM vectors.

It computes:
1) Partial Spearman Correlations (Unique RSA): 
   Correlation between a model and the empirical RDM, controlling for a set of baseline 
   control models (e.g., low-level visual features).
2) Shapley / Hierarchical Partitioning (order-agnostic):
   Average marginal R² contribution of each predictor over many random predictor orderings.

Inputs
------
Expects the pairwise vector table produced by compute_model_evaluation.py:
  analysis_code/compute_outputs/stage2_model_evaluation/<rdm_mode>/<source>/pairwise_vectors.csv

Three rdm_mode variants are used in the paper (pass via --outdir or rely on pipeline):
  fixed_dense  → 209-item observed-only RDM (703 pairs) — primary / main text
  filled       → 350-item NN-completed RDM (seed 19)    — appendix
  empirical    → full observed-only RDM (all pairs)     — reference / diagnostic

Required columns:
  - emp : empirical dissimilarity vector
  - used_in_regression : boolean-like flag (True/False/1/0)

All other numeric columns are treated as candidate predictors unless excluded.

By default, predictors are filtered using the model-selection policy in model_selection.py:
  - Human vs AI predictors are separated (default: human only, see --source).
  - Conceptual contrasts (e.g., "Sweet vs savory") are excluded unless --include-contrasts is set.
  - Visual predictors are included unless --no-visual is set.
  - Control predictors are included only if --controls is set.
  - To disable all policy filtering and treat all numeric columns as predictors, use --no-policy.

Relevant CLI flags:
  --source                : Which rating source to include for appraisal/flavour (human/ai/mixed).
  --include-contrasts     : Include conceptual contrasts if present.
  --controls              : Include control predictors if present.
  --no-visual             : Exclude low-level visual predictors.
  --no-policy             : Disable model-selection policy filtering (legacy behavior).
  --timestamped           : Create a timestamped subfolder under the output directory.

Outputs
-------
Writes CSV summaries and JSON manifests to an output directory.

Examples
--------
# Shapley-style attribution over all predictors in the file (excluding i,j,emp)
python analysis_code/compute_variance_attribution.py \
  --pairwise-csv analysis_code/compute_outputs/stage2_model_evaluation/fixed_dense/<source>/pairwise_vectors.csv \
  --method shapley --n-orderings 300 --bootstrap 200

# Shapley attribution but treat low-level models as "controls" (entered first)
python analysis_code/compute_variance_attribution.py \
  --pairwise-csv analysis_code/compute_outputs/stage2_model_evaluation/fixed_dense/<source>/pairwise_vectors.csv \
  --method shapley --n-orderings 400 \
  --controls-prefixes ll_,rdm_ll_

# Ordered incremental (define an order explicitly)
python analysis_code/compute_variance_attribution.py \
  --pairwise-csv analysis_code/compute_outputs/stage2_model_evaluation/fixed_dense/<source>/pairwise_vectors.csv \
  --method ordered \
  --ordered "aware_ai_savoriness_0_100,human_mean_umami,ll_hog_pcs10"

# Classical regression (standardized OLS) over chosen predictors
python analysis_code/compute_variance_attribution.py \
  --pairwise-csv analysis_code/compute_outputs/stage2_model_evaluation/fixed_dense/<source>/pairwise_vectors.csv \
  --method ols --rank-transform \
  --exclude "i,j,used_in_regression"
"""

from __future__ import annotations

import argparse
import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Tuple, Dict

import numpy as np
import pandas as pd

# Central model-selection policy — INTENTIONALLY NOT WIRED UP.
#
# Documented 2026-09-10 (revision/transformation-models). This block previously read
# as a failed import (`try: select_models = None`) with no import statement in it, so
# `select_models` was always None and the strict whitelist branch further down
# (`if not args.no_policy and select_models is not None`) has never executed. Every
# published variance attribution was produced by the permissive fallback branch, which
# drops only the opposite rating source and keeps all 23 predictors.
#
# DO NOT "fix" this by importing analysis_helpers.select_models without also widening
# analysis_helpers.VISUAL. That whitelist contains only the 4 compact visual models,
# whereas the manuscript Methods state that all 11 visual feature RDMs enter the
# Shapley. Activating it would silently drop 7 visual predictors and change published
# numbers. The behaviour below is deliberate and is what reproduces the paper.
select_models = None  # see note above; the strict-policy branch is intentionally inert

# Administrative / QC columns that should never be treated as model predictors
try:
    from reference.policy_defaults import CONTROL_COLS as _POLICY_CONTROL_COLS
except Exception:
    _POLICY_CONTROL_COLS: frozenset[str] = frozenset({
        "portion_size_ok", "quality", "plate_rim_visible", "file_exists",
        "label_match", "label_confidence", "qc_issues", "qc_reasons", "qc_at", "qc_model",
    })


# ---------------------------
# Utilities
# ---------------------------

def _as_bool_series(s: pd.Series) -> pd.Series:
    """Robust conversion for common boolean-ish encodings."""
    if s.dtype == bool:
        return s

    def _cell(v) -> bool:
        if v is None:
            return False
        try:
            if pd.isna(v):
                return False
        except Exception:
            pass
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, np.integer)):
            return int(v) != 0
        if isinstance(v, (float, np.floating)):
            return float(v) != 0.0
        vv = str(v).strip().lower()
        return vv in {"1", "true", "t", "yes", "y", "on"}

    return s.apply(_cell).astype(bool)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return x * 0.0
    return (x - mu) / sd



def _partial_correlation(y, x, Z=None):
    """Partial Spearman correlation between y and x, controlling for columns in Z."""
    from scipy.stats import rankdata
    
    def _rank_vec(v):
        return rankdata(v, method="average")
    
    yr = _rank_vec(y)
    xr = _rank_vec(x)
    
    if Z is not None and Z.shape[1] > 0:
        Zr = np.column_stack([_rank_vec(Z[:, j]) for j in range(Z.shape[1])])
        # Regress out Zr from yr
        beta_y, *_ = np.linalg.lstsq(Zr, yr, rcond=None)
        yr_resid = yr - Zr @ beta_y
        # Regress out Zr from xr
        beta_x, *_ = np.linalg.lstsq(Zr, xr, rcond=None)
        xr_resid = xr - Zr @ beta_x
        
        # Pearson on residuals is partial correlation
        def _corr(a, b):
            a_z = (a - a.mean()) / a.std()
            b_z = (b - b.mean()) / b.std()
            return (a_z * b_z).mean()
        
        return float(_corr(yr_resid, xr_resid))
    else:
        # Standard Spearman
        from scipy.stats import spearmanr
        return float(spearmanr(y, x).correlation)

def bootstrap_partial_correlation(y, x, Z=None, n_boot=1000, seed=123):
    rng = np.random.default_rng(seed)
    n = len(y)
    
    true_rho = _partial_correlation(y, x, Z)
    
    rhos = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Zb = Z[idx, :] if Z is not None else None
        rhos[b] = _partial_correlation(y[idx], x[idx], Zb)
        
    rhos_perm = np.empty(n_boot)
    for b in range(n_boot):
        x_perm = x.copy()
        rng.shuffle(x_perm)
        rhos_perm[b] = _partial_correlation(y, x_perm, Z)
        
    p_val = np.mean(np.abs(rhos_perm) >= np.abs(true_rho))
        
    return float(true_rho), float(np.nanpercentile(rhos, 2.5)), float(np.nanpercentile(rhos, 97.5)), float(p_val)

def _r2_ols(y: np.ndarray, X: np.ndarray, rank_transform: bool = False) -> float:
    """R² from OLS on standardized y and standardized X (columns).
    Optionally rank-transform y and X before z-scoring/OLS.
    """
    y = np.asarray(y, dtype=float)
    # Optional: rank-transform (Spearman-style) before z-scoring/OLS.
    # This makes the attribution more comparable to rank-based RSA/Mantel analyses.
    if rank_transform:
        y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if X.size == 0:
        return 0.0

    # z-score
    X_z = np.asarray(X, dtype=float)
    if rank_transform:
        X_z = np.column_stack([
            pd.Series(X_z[:, j]).rank(method="average").to_numpy(dtype=float)
            for j in range(X_z.shape[1])
        ])
    y_z = _zscore(y)
    X_z = np.column_stack([_zscore(X_z[:, j]) for j in range(X_z.shape[1])])

    # if all-zero predictors after zscore, return 0
    if np.allclose(X_z, 0.0):
        return 0.0

    beta, *_ = np.linalg.lstsq(X_z, y_z, rcond=None)
    y_hat = X_z @ beta
    ss_res = float(np.sum((y_z - y_hat) ** 2))
    ss_tot = float(np.sum((y_z - np.mean(y_z)) ** 2))
    if ss_tot <= 0:
        return 0.0
    r2 = 1.0 - (ss_res / ss_tot)
    # numerical guard
    return float(max(min(r2, 1.0), 0.0))


def _select_predictor_columns(
    df: pd.DataFrame,
    exclude: Iterable[str],
    include_controls: bool = False,
) -> List[str]:
    exclude_set = {c.strip() for c in exclude if c.strip()}
    base_exclude = {"i", "j", "emp", "used_in_regression", "sweet_vs_savory"}
    exclude_set |= base_exclude
    # Always exclude administrative/QC columns unless the caller explicitly opts in
    if not include_controls:
        exclude_set |= _POLICY_CONTROL_COLS

    cols: List[str] = []
    for c in df.columns:
        if c in exclude_set:
            continue
        # keep numeric columns only
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _split_controls_and_targets(predictors: List[str], controls_prefixes: List[str]) -> Tuple[List[str], List[str]]:
    if not controls_prefixes:
        return [], predictors
    controls = [p for p in predictors if any(p.startswith(px) for px in controls_prefixes)]
    targets = [p for p in predictors if p not in controls]
    return controls, targets



def _make_outdir(
    root: Path | None,
    *,
    timestamped: bool = False,
    stage_dir: str = "stage4_variance_attribution",
    subdir: str | None = None,
) -> Path:
    """Create/return output directory.

    Default behavior is a *stable* stage folder under `analysis_code/results/<stage_dir>`.
    If `timestamped=True`, a timestamped subfolder is created under that stage folder.

    If `subdir` is provided (e.g., "human" or "ai"), outputs are written under
    `.../<stage_dir>/<subdir>/` to avoid overwriting across policy runs.
    """
    if root is None:
        base = Path("analysis_code") / "results" / stage_dir
    else:
        base = root

    if subdir:
        base = base / subdir

    if timestamped:
        base = base / datetime.now().strftime("%Y-%m-%d_%H%M%S")

    base.mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------
# Attribution methods
# ---------------------------

def shapley_attribution(
    y: np.ndarray,
    X: Dict[str, np.ndarray],
    controls: List[str] | None = None,
    n_orderings: int = 300,
    rank_transform: bool = False,
    seed: int = 123,
) -> Dict[str, float]:
    """Approximate Shapley values via random orderings.

    If `controls` are provided, they are always included first (baseline) and
    Shapley values are computed for the remaining predictors, i.e. marginal
    contributions *beyond the controls*.
    """
    rng = np.random.default_rng(seed)

    controls = controls or []
    pred_names = [k for k in X.keys() if k not in controls]

    # baseline R2 with controls
    X_ctrl = np.column_stack([X[c] for c in controls]) if controls else np.empty((len(y), 0))
    base_r2 = _r2_ols(y, X_ctrl, rank_transform=rank_transform) if controls else 0.0

    contrib = {k: 0.0 for k in pred_names}

    if len(pred_names) == 0:
        return {"__baseline_R2__": float(base_r2)}

    for _ in range(n_orderings):
        order = list(pred_names)
        rng.shuffle(order)

        cur_cols = list(controls)
        cur_X = X_ctrl
        cur_r2 = base_r2

        for p in order:
            cur_cols.append(p)
            # append column efficiently
            if cur_X.size == 0:
                cur_X = X[p][:, None]
            else:
                cur_X = np.column_stack([cur_X, X[p]])
            new_r2 = _r2_ols(y, cur_X, rank_transform=rank_transform)
            contrib[p] += (new_r2 - cur_r2)
            cur_r2 = new_r2

    # average
    for k in contrib:
        contrib[k] /= float(n_orderings)

    contrib["__baseline_R2__"] = float(base_r2)
    contrib["__total_R2_with_all__"] = float(_r2_ols(y, np.column_stack([X[c] for c in controls + pred_names]), rank_transform=rank_transform))
    return contrib


def ordered_incremental(
    y: np.ndarray,
    X: Dict[str, np.ndarray],
    ordered: List[str],
    rank_transform: bool = False,
) -> List[Dict[str, float]]:
    """Compute incremental R2 as predictors are added in the given order."""
    rows: List[Dict[str, float]] = []
    cur_X = np.empty((len(y), 0))
    cur_r2 = 0.0

    for step, name in enumerate(ordered, start=1):
        if name not in X:
            rows.append({"step": step, "predictor": name, "delta_R2": np.nan, "R2": np.nan})
            continue
        if cur_X.size == 0:
            cur_X = X[name][:, None]
        else:
            cur_X = np.column_stack([cur_X, X[name]])
        new_r2 = _r2_ols(y, cur_X, rank_transform=rank_transform)
        rows.append({"step": step, "predictor": name, "delta_R2": float(new_r2 - cur_r2), "R2": float(new_r2)})
        cur_r2 = new_r2

    return rows


def bootstrap_shapley(
    y: np.ndarray,
    X: Dict[str, np.ndarray],
    controls: List[str],
    n_orderings: int,
    n_boot: int,
    rank_transform: bool = False,
    seed: int = 123,
) -> pd.DataFrame:
    """Bootstrap CIs for Shapley contributions by resampling rows."""
    rng = np.random.default_rng(seed)
    names = [k for k in X.keys() if k not in controls]
    if not names:
        return pd.DataFrame(columns=["predictor", "mean", "ci95_lo", "ci95_hi"]) 

    B = int(n_boot)
    vals = {k: np.empty(B, dtype=float) for k in names}
    base = np.empty(B, dtype=float)
    tot = np.empty(B, dtype=float)

    n = len(y)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        Xb = {k: v[idx] for k, v in X.items()}
        out = shapley_attribution(yb, Xb, controls=controls, n_orderings=n_orderings, rank_transform=rank_transform, seed=int(seed + 10 + b))
        base[b] = out.get("__baseline_R2__", np.nan)
        tot[b] = out.get("__total_R2_with_all__", np.nan)
        for k in names:
            vals[k][b] = out.get(k, np.nan)

    rows = []
    for k in names:
        rows.append({
            "predictor": k,
            "mean": float(np.nanmean(vals[k])),
            "ci95_lo": float(np.nanpercentile(vals[k], 2.5)),
            "ci95_hi": float(np.nanpercentile(vals[k], 97.5)),
        })

    rows.append({
        "predictor": "__baseline_R2__",
        "mean": float(np.nanmean(base)),
        "ci95_lo": float(np.nanpercentile(base, 2.5)),
        "ci95_hi": float(np.nanpercentile(base, 97.5)),
    })
    rows.append({
        "predictor": "__total_R2_with_all__",
        "mean": float(np.nanmean(tot)),
        "ci95_lo": float(np.nanpercentile(tot, 2.5)),
        "ci95_hi": float(np.nanpercentile(tot, 97.5)),
    })

    return pd.DataFrame(rows)




# ---------------------------
# CLI
# ---------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Variance attribution for pairwise_vectors regression predictors")
    p.add_argument("--pairwise-csv", type=str, required=True, help="Path to pairwise_vectors.csv")
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help=(
            "Output root directory. Default is a stable stage folder: "
            "analysis_code/results/stage4_variance_attribution/<source>/ (or /no_policy if --no-policy). "
            "Use --timestamped to create a timestamped subfolder under that stage folder."
        ),
    )

    p.add_argument(
        "--timestamped",
        action="store_true",
        help="If set, create a timestamped subfolder under the default stage output directory.",
    )

    p.add_argument("--method", type=str, choices=["shapley", "ordered", "ols", "partial", "suite"], default="shapley")
    p.add_argument("--exclude", type=str, default="", help="Comma-separated columns to exclude as predictors")

    p.add_argument(
        "--controls-prefixes",
        type=str,
        default="",
        help=(
            "Comma-separated prefixes treated as controls (always entered first). "
            "Example: ll_,rdm_ll_"
        ),
    )

    p.add_argument("--n-orderings", type=int, default=300, help="Number of random predictor orderings (shapley)")
    p.add_argument("--seed", type=int, default=123, help="RNG seed")
    p.add_argument("--bootstrap", type=int, default=0, help="Bootstrap iterations for CI (0 disables)")

    p.add_argument("--ordered", type=str, default="", help="Comma-separated predictor order for --method ordered")

    p.add_argument(
        "--rank-transform",
        action="store_true",
        default=True,
        help="Rank-transform emp and predictors before OLS/R² (Spearman-style). "
             "ON by default to match the Spearman ρ metric used in Stages 3 and 5. "
             "Use --no-rank-transform to revert to raw OLS R².",
    )
    p.add_argument(
        "--no-rank-transform",
        action="store_false",
        dest="rank_transform",
        help="Disable rank-transform and use raw OLS R² (legacy behaviour, not recommended for RSA).",
    )

    p.add_argument(
        "--no-policy",
        action="store_true",
        help=(
            "Disable model-selection policy filtering and treat all numeric columns as predictors "
            "(legacy behavior)."
        ),
    )
    p.add_argument(
        "--source",
        choices=["human", "ai", "mixed"],
        default="human",
        help="Model policy: which rating source to include for appraisal+flavour (default: human).",
    )
    p.add_argument(
        "--rdm-mode",
        choices=["empirical", "filled", "fixed_dense"],
        default="empirical",
        help="RDM mode for output directory resolution (default: empirical).",
    )
    p.add_argument(
        "--no-visual",
        action="store_true",
        help="Model policy: exclude low-level visual predictors (ll_*).",
    )
    p.add_argument(
        "--controls",
        action="store_true",
        help="Model policy: include control predictors (e.g., portion_size_ok, quality) if present.",
    )
    p.add_argument(
        "--include-contrasts",
        action="store_true",
        help="Model policy: include supplementary conceptual contrasts (Natural vs transformed) if present.",
    )

    return p.parse_args(argv)


# ---------------------------
# Model selection policy helpers
# ---------------------------

def _resolve_policy_keys_to_columns(keys: List[str], available_cols: List[str]) -> List[str]:
    """Map model-selection keys to column names in pairwise_vectors.csv.

    We support a few historical naming variants:
      - AI keys like `aware_ai_sweetness_0_100` may appear as `sweetness_0_100`.
      - Some scripts store models with a leading `rdm_` prefix.
    """
    avail = set(available_cols)
    out: List[str] = []

    def _add_if_exists(c: str) -> None:
        if c in avail and c not in out:
            out.append(c)

    for k in keys:
        k = str(k)
        _add_if_exists(k)
        _add_if_exists(f"rdm_{k}")

        # AI alias: ai_<name> -> <name>
        if k.startswith("ai_"):
            _add_if_exists(k[len("ai_"):])
            _add_if_exists(f"rdm_{k[len('ai_'):]}")

    return out


import sys
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from reference.policy_defaults import get_compute_dir

def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    pairwise_csv = Path(args.pairwise_csv).expanduser().resolve()
    if not pairwise_csv.exists():
        raise FileNotFoundError(f"pairwise CSV not found: {pairwise_csv}")

    # Use central get_compute_dir, or honour an explicit --outdir override.
    # The rdm_mode segment ("empirical", "fixed_dense", or "filled") is passed by the caller
    # (compute_model_evaluation.py) via --outdir when running on NN-completed data.
    policy_subdir = "mixed" if bool(args.no_policy) else str(args.source)
    if args.outdir is not None:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        outdir = get_compute_dir("stage4_variance_attribution", args.rdm_mode, policy_subdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pairwise_csv)
    if "emp" not in df.columns:
        raise ValueError("pairwise CSV missing required column: emp")
    if "used_in_regression" not in df.columns:
        raise ValueError("pairwise CSV missing required column: used_in_regression")

    used = _as_bool_series(df["used_in_regression"])
    df = df.loc[used].copy()

    # Keep only finite emp
    df["emp"] = pd.to_numeric(df["emp"], errors="coerce")
    df = df.loc[np.isfinite(df["emp"])].copy()

    exclude = [x.strip() for x in str(args.exclude).split(",") if x.strip()]
    predictors = _select_predictor_columns(df, exclude=exclude, include_controls=bool(args.controls))

    # ---------------------------
    # Model-selection policy filtering (optional)
    # ---------------------------
    if not args.no_policy and select_models is not None:
        specs = select_models(
            source_mode=args.source,  # type: ignore[arg-type]
            include_visual=(not args.no_visual),
            include_controls=bool(args.controls),
            include_supplementary_contrasts=bool(args.include_contrasts),
        )
        policy_keys = [s.key for s in specs]
        allowed = set(_resolve_policy_keys_to_columns(policy_keys, predictors))
        predictors = [p for p in predictors if p in allowed]

        if not predictors:
            raise ValueError(
                "After applying model-selection policy, no predictors remained. "
                "Check that pairwise_vectors.csv contains columns matching the selected models, "
                "or run with --no-policy."
            )
    elif not args.no_policy:
        # Fallback source filter when the full model-selection module is unavailable.
        # Human and AI rating columns are separated so that each source analysis uses
        # only its own ratings (plus visual controls and conceptual contrasts).
        # Pooling both sets in one model inflates collinearity because matched dimensions
        # (e.g., aware_ai_spiciness / human_spiciness) measure the same construct and would
        # suppress each other's apparent contribution.
        src = str(getattr(args, "source", "human")).lower()
        if src in ("human", "ai"):
            keep_prefix = f"{src}_"
            drop_prefix = "ai_" if src == "human" else "human_"
            filtered = []
            for p in predictors:
                if p.startswith(drop_prefix):
                    continue   # drop the other source's ratings
                filtered.append(p)
            n_before = len(predictors)
            predictors = filtered
            n_dropped = n_before - len(predictors)
            if n_dropped:
                print(
                    f"[INFO] source={src}: dropped {n_dropped} predictor(s) from the "
                    f"opposite rating source (pass --no-policy to disable this filter)"
                )
        # source == "mixed" → keep everything (no filter needed)

    if not predictors:
        raise ValueError("No numeric predictor columns found after exclusions.")

    controls_prefixes = [x.strip() for x in str(args.controls_prefixes).split(",") if x.strip()]
    controls, targets = _split_controls_and_targets(predictors, controls_prefixes=controls_prefixes)

    # Assemble arrays
    y = df["emp"].to_numpy(dtype=float)

    X: Dict[str, np.ndarray] = {}
    for c in predictors:
        X[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)

    # Finite mask across y and all X
    m = np.isfinite(y)
    for v in X.values():
        m &= np.isfinite(v)

    y = y[m]
    X = {k: v[m] for k, v in X.items()}

    # Report
    print("=== Variance Attribution ===")
    print(f"Input:  {pairwise_csv}")
    print(f"Outdir: {outdir}")
    print(f"Rows used: {len(y):,}")
    print(f"Predictors: {len(X)} (controls={len(controls)} via prefixes={controls_prefixes})")
    if args.no_policy:
        print("[INFO] Policy: disabled (--no-policy)")
    else:
        print(
            f"[INFO] Policy: source={args.source} | visual={'off' if args.no_visual else 'on'} "
            f"| controls={'on' if args.controls else 'off'} | contrasts={'on' if args.include_contrasts else 'off'}"
        )

    # ---------------------------
    
    # ---------------------------
    # Partial Correlation (RSA Unique)
    # ---------------------------
    if args.method in ["partial", "suite"]:
        print(f"--- Computing Partial Correlations (Unique RSA — leave-one-out) ---")
        rows = []
        # For each target predictor, Z = all other predictors (controls + remaining targets).
        # This gives the true partial Mantel ρ: unique variance after all others are partialed out.
        all_predictors = controls + targets  # full set, preserving order

        for name in targets:
            x_vec = X[name]
            others = [c for c in all_predictors if c != name]
            Z_loo = np.column_stack([X[c] for c in others]) if others else None
            rho, lo, hi, p_val = bootstrap_partial_correlation(y, x_vec, Z=Z_loo, n_boot=max(1, int(args.bootstrap)), seed=int(args.seed))
            print(f"  {name:25s} partial_rho={rho:.3f} (95% CI: {lo:.3f}, {hi:.3f}, p={p_val:.3g})")
            rows.append({
                "model": name,
                "rho_partial": rho,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "p_perm": p_val
            })
        
        df_partial = pd.DataFrame(rows).sort_values("rho_partial", ascending=False)
        out_csv = outdir / "model_unique.csv"
        df_partial.to_csv(out_csv, index=False)
        print(f"[OK] Wrote: {out_csv}")

    # Classical regression (OLS)
    # ---------------------------
    if args.method == "ols":
        # Fit OLS with all predictors (controls + targets if any). This is a convenient
        # "rank the models" view that complements the order-agnostic Shapley values.
        # Note: Uses the same rank_transform option to align with Spearman-style analyses.

        pred_names = list(X.keys())
        Xmat = np.column_stack([X[k] for k in pred_names]) if pred_names else np.empty((len(y), 0))

        def _ols_fit(yv: np.ndarray, Xm: np.ndarray, rank_transform: bool) -> Tuple[np.ndarray, float, float]:
            """Return (beta, R2, adj_R2) for standardized OLS."""
            if Xm.size == 0:
                return np.zeros((0,), dtype=float), 0.0, 0.0

            yy = np.asarray(yv, dtype=float)
            XX = np.asarray(Xm, dtype=float)

            if rank_transform:
                yy = pd.Series(yy).rank(method="average").to_numpy(dtype=float)
                XX = np.column_stack([
                    pd.Series(XX[:, j]).rank(method="average").to_numpy(dtype=float)
                    for j in range(XX.shape[1])
                ])

            yy_z = _zscore(yy)
            XX_z = np.column_stack([_zscore(XX[:, j]) for j in range(XX.shape[1])])

            # Guard: if predictors collapse to zero
            if np.allclose(XX_z, 0.0):
                return np.zeros((XX_z.shape[1],), dtype=float), 0.0, 0.0

            beta, *_ = np.linalg.lstsq(XX_z, yy_z, rcond=None)
            yhat = XX_z @ beta
            ss_res = float(np.sum((yy_z - yhat) ** 2))
            ss_tot = float(np.sum((yy_z - np.mean(yy_z)) ** 2))
            r2 = 0.0 if ss_tot <= 0 else float(max(min(1.0 - ss_res / ss_tot, 1.0), 0.0))

            n = int(len(yy_z))
            p = int(XX_z.shape[1])
            denom = max(n - p - 1, 1)
            adj = float(1.0 - (1.0 - r2) * (n - 1) / denom)
            return beta.astype(float), r2, adj

        beta, r2, r2_adj = _ols_fit(y, Xmat, rank_transform=bool(args.rank_transform))

        # VIF (collinearity diagnostic) computed on standardized (and rank-transformed if requested) X
        # We compute it on the same transformed design matrix used for fitting.
        def _design_matrix_for_vif(Xm: np.ndarray, rank_transform: bool) -> np.ndarray:
            XX = np.asarray(Xm, dtype=float)
            if XX.size == 0:
                return XX
            if rank_transform:
                XX = np.column_stack([
                    pd.Series(XX[:, j]).rank(method="average").to_numpy(dtype=float)
                    for j in range(XX.shape[1])
                ])
            XX_z = np.column_stack([_zscore(XX[:, j]) for j in range(XX.shape[1])])
            return XX_z

        XX_z = _design_matrix_for_vif(Xmat, rank_transform=bool(args.rank_transform))
        vifs: List[float] = []
        if XX_z.size == 0:
            vifs = []
        elif XX_z.shape[1] == 1:
            vifs = [float("nan")]
        else:
            for j in range(XX_z.shape[1]):
                yj = XX_z[:, j]
                Xo = np.delete(XX_z, j, axis=1)
                try:
                    bj, *_ = np.linalg.lstsq(Xo, yj, rcond=None)
                    yhatj = Xo @ bj
                    ss_res_j = float(np.sum((yj - yhatj) ** 2))
                    ss_tot_j = float(np.sum((yj - np.mean(yj)) ** 2))
                    r2_j = 0.0 if ss_tot_j <= 0 else float(max(min(1.0 - ss_res_j / ss_tot_j, 1.0), 0.0))
                    vif = float("inf") if (1.0 - r2_j) <= 1e-12 else float(1.0 / (1.0 - r2_j))
                except Exception:
                    vif = float("nan")
                vifs.append(vif)

        # Optional bootstrap CIs on betas (row-resampling)
        beta_ci_lo = [float("nan")] * len(pred_names)
        beta_ci_hi = [float("nan")] * len(pred_names)
        if int(args.bootstrap) > 0 and len(pred_names) > 0:
            rng = np.random.default_rng(int(args.seed) + 2026)
            B = int(args.bootstrap)
            n = len(y)
            betas_b = np.empty((B, len(pred_names)), dtype=float)
            for b in range(B):
                idx = rng.integers(0, n, size=n)
                bb, _, _ = _ols_fit(y[idx], Xmat[idx, :], rank_transform=bool(args.rank_transform))
                betas_b[b, :] = bb
            beta_ci_lo = np.nanpercentile(betas_b, 2.5, axis=0).tolist()
            beta_ci_hi = np.nanpercentile(betas_b, 97.5, axis=0).tolist()

        reg_rows = []
        for i, name in enumerate(pred_names):
            reg_rows.append({
                "predictor": name,
                "beta": float(beta[i]),
                "beta_ci95_lo": float(beta_ci_lo[i]) if i < len(beta_ci_lo) else float("nan"),
                "beta_ci95_hi": float(beta_ci_hi[i]) if i < len(beta_ci_hi) else float("nan"),
                "vif": float(vifs[i]) if i < len(vifs) else float("nan"),
            })

        reg_df = pd.DataFrame(reg_rows)
        reg_df = reg_df.sort_values("beta", ascending=False)
        out_csv = outdir / "variance_attribution_ols.csv"
        reg_df.to_csv(out_csv, index=False)
        print(f"[OK] Wrote: {out_csv}")

        # Also write a tiny metrics file
        metrics = {
            "pairwise_csv": str(pairwise_csv),
            "rows_used": int(len(y)),
            "method": "ols",
            "rank_transform": bool(args.rank_transform),
            "predictors": pred_names,
            "R2": float(r2),
            "adj_R2": float(r2_adj),
        }
        mpath = outdir / "variance_attribution_ols_manifest.json"
        mpath.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[OK] Wrote: {mpath}")

        # Plot absolute betas for readability
        plot_df = reg_df.copy()
        plot_df["mean"] = plot_df["beta"].abs()

        print(f"OLS R²={r2:.4f} | adj.R²={r2_adj:.4f}")
        print("Top predictors by |beta|:")
        top = plot_df.sort_values("mean", ascending=False).head(10)
        for _, rr in top.iterrows():
            print(f"  {rr['predictor']}: beta={rr['beta']:.4f}")

        return 0

    if args.method == "ordered":
        ordered = [x.strip() for x in str(args.ordered).split(",") if x.strip()]
        if not ordered:
            raise ValueError("--method ordered requires --ordered with at least one predictor name")

        rows = ordered_incremental(y, X, ordered=ordered, rank_transform=bool(args.rank_transform))
        df_out = pd.DataFrame(rows)
        out_csv = outdir / "variance_attribution_ordered.csv"
        df_out.to_csv(out_csv, index=False)
        print(f"[OK] Wrote: {out_csv}")

        # Plot
        # use delta_R2 bar plot
        df_plot = df_out.dropna(subset=["delta_R2"]).copy()
        df_plot = df_plot.rename(columns={"predictor": "predictor", "delta_R2": "mean"})
        return 0

    if args.method in ["shapley", "suite"]:
        # Shapley / hierarchical
        contrib = shapley_attribution(
            y=y,
            X=X,
            controls=controls,
            n_orderings=int(args.n_orderings),
            rank_transform=bool(args.rank_transform),
            seed=int(args.seed),
        )
    
        df_contrib = pd.DataFrame(
            [{"predictor": k, "mean": float(v)} for k, v in contrib.items()]
        ).sort_values("mean", ascending=False)
    
        out_csv = outdir / "variance_attribution_shapley.csv"
        df_contrib.to_csv(out_csv, index=False)
        print(f"[OK] Wrote: {out_csv}")
    
        if int(args.bootstrap) > 0:
            df_ci = bootstrap_shapley(
                y=y,
                X=X,
                controls=controls,
                n_orderings=max(30, int(args.n_orderings) // 5),
                n_boot=int(args.bootstrap),
                rank_transform=bool(args.rank_transform),
                seed=int(args.seed) + 999,
            )
            out_ci = outdir / "variance_attribution_shapley_ci.csv"
            df_ci.to_csv(out_ci, index=False)
            print(f"[OK] Wrote: {out_ci}")
            df_plot = df_ci.rename(columns={"predictor": "predictor", "mean": "mean"})
        else:
            pass

        # Write a small manifest
        manifest = {
            "pairwise_csv": str(pairwise_csv),
            "rows_used": int(len(y)),
            "method": "shapley",
            "n_orderings": int(args.n_orderings),
            "controls_prefixes": controls_prefixes,
            "controls": controls,
            "predictors": list(X.keys()),
            "rank_transform": bool(args.rank_transform),
            "outputs": {
                "variance_attribution_shapley_csv": str(out_csv),
            },
        }
        if int(args.bootstrap) > 0:
            manifest["outputs"]["variance_attribution_shapley_ci_csv"] = str(outdir / "variance_attribution_shapley_ci.csv")
    
        mpath = outdir / "variance_attribution_manifest.json"
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[OK] Wrote: {mpath}")
    
        # Console summary
        _metric = "ρ²" if bool(args.rank_transform) else "R²"
        base = float(contrib.get("__baseline_R2__", np.nan))
        tot = float(contrib.get("__total_R2_with_all__", np.nan))
        print(f"Baseline {_metric} (controls only): {base:.4f}")
        print(f"Total {_metric} (controls + all targets): {tot:.4f}")
        top = df_contrib[~df_contrib["predictor"].str.startswith("__")].head(10)
        print(f"\nTop Δ{_metric} contributors:")
        for _, r in top.iterrows():
            print(f"  {r['predictor']}: {r['mean']:.4f}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())