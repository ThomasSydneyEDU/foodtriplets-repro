#!/usr/bin/env python3
"""compute_category_residuals.py

Stage (Compute): Residual structure analysis for FoodTriplets2.

Asks whether the variance left unexplained by the best-fitting full-feature
model concentrates within food categories rather than between them.

Rationale
---------
The full-feature NNLS model (10 human semantic dimensions + 11 visual feature
RDMs + food classification) explains ρ² ≈ 0.226 of the behavioural RDM.
If the remaining ~77% is noise, the residuals should be unstructured with
respect to category membership. If instead the residuals are systematically
larger for within-category pairs, it suggests that fine-grained, within-
category structure (e.g. flavour differences among grains, or texture
differences among animal proteins) accounts for the unexplained variance —
a structured gap that better within-category models might close.

Analysis
--------
1. Build the group empirical RDM from all trials (full dataset).
2. Fit the full-feature NNLS model (same 22 predictors as Stage 6
   full-feature Shapley) on all observed pairs.
3. Compute pair-wise residuals: residual = empirical − predicted.
4. Classify each pair as within-category or between-category using the
   food_classification column from the canonical CSV.
5. Compare mean absolute residual (MAR) for within vs. between pairs:
   - Observed MAR difference (within − between).
   - Permutation test: shuffle category labels across items (5000
     permutations) to generate a null distribution; report p-value and
     95% CI of the observed difference.
   - Mann-Whitney U test as a non-parametric complement.
6. Repeat for signed residuals to check directionality:
   - Positive residual → model underestimates dissimilarity for this pair.
   - Negative residual → model overestimates dissimilarity.

Outputs (written to compute_outputs/stage_category_residuals/)
--------------------------------------------------------------
  residual_pairs.csv
      i, j, label_i, label_j, category_i, category_j, pair_type,
      empirical, predicted, residual, abs_residual
  residual_summary.csv
      pair_type, n_pairs, mean_abs_residual, sd_abs_residual,
      mean_signed_residual, sd_signed_residual
  residual_test.json
      observed_diff, p_value, ci_low, ci_high, mannwhitney_U,
      mannwhitney_p, n_permutations
  fig_residual_by_category.png
      Violin/strip plot of absolute residuals split by pair type.
  manifest.json

Usage
-----
  python analysis_code/compute_category_residuals.py
  python analysis_code/compute_category_residuals.py --n-perms 1000
  python analysis_code/compute_category_residuals.py --no-fixed-dense-only
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
SUMMARY_CSV         = HERE / "derivatives" / "summary.csv"
MODEL_RDM_DIR       = HERE / "derivatives" / "model_rdms_canon"
CANONICAL_CSV       = HERE / "reference" / "Foodpictures_information_dynamic.csv"
DEFAULT_OUTDIR      = HERE / "compute_outputs" / "stage_category_residuals"

# Try alternate canonical CSV locations if default not found
_ALT_CANONICAL_PATHS = [
    HERE / "derivatives" / "Foodpictures_information_canonical.csv",
    HERE.parent / "uploads" / "Foodpictures_information_canonical.csv",
    HERE.parent / "Foodpictures_information_canonical.csv",
]

# ---------------------------------------------------------------------------
# Full-feature predictor set (22 RDMs — mirrors Stage 6 full-feature Shapley)
# ---------------------------------------------------------------------------
FULL_FEATURE_STEMS: List[str] = [
    # Human semantic (10)
    "rdm_human_sweetness",
    "rdm_human_saltiness",
    "rdm_human_sourness",
    "rdm_human_bitterness",
    "rdm_human_savoriness",
    "rdm_human_fattiness",
    "rdm_human_spiciness",
    "rdm_human_healthiness",
    "rdm_human_calorie_density",
    "rdm_human_appeal",
    # Visual (11)
    "rdm_ll_mean_luminance",
    "rdm_ll_rms_contrast",
    "rdm_ll_hsv_s_mean",
    "rdm_ll_edge_energy",
    "rdm_ll_lab_mean",
    "rdm_ll_lab_std",
    "rdm_ll_hog_pcs10",
    "rdm_ll_lbp_hist",
    "rdm_ll_power_radial30",
    "rdm_ll_resnet50_conv1_pcs10",
    "rdm_ll_resnet50_layer3_pcs10",
    # Category (1)
    "rdm_food_classification",
]

# ---------------------------------------------------------------------------
# Utilities  (adapted from compute_family_generalization.py)
# ---------------------------------------------------------------------------

@dataclass
class EmpiricalRDM:
    D: np.ndarray
    labels: List[str]


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

    sim += sim.T
    cnt += cnt.T
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


def _nnls_numpy(A: np.ndarray, b: np.ndarray,
                max_iter: int = 2000, tol: float = 1e-10) -> Tuple[np.ndarray, float]:
    """Non-negative least squares via projected gradient descent (pure NumPy).

    Minimises ‖Ax − b‖₂  subject to  x ≥ 0.
    Step size is 1 / ‖AᵀA‖₂ (spectral norm upper bound via Frobenius norm).
    Falls back automatically to scipy.optimize.nnls if available.
    """
    try:
        from scipy.optimize import nnls as _sp_nnls
        x, resid = _sp_nnls(A, b)
        return x, float(resid)
    except ImportError:
        pass

    ATA = A.T @ A
    ATb = A.T @ b
    step = 1.0 / (np.linalg.norm(ATA, "fro") + 1e-12)
    x = np.zeros(A.shape[1], dtype=float)
    for _ in range(max_iter):
        grad = ATA @ x - ATb
        x_new = np.maximum(0.0, x - step * grad)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    resid = float(np.linalg.norm(A @ x - b))
    return x, resid


def _nnls_predict(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit NNLS on (X, y) and return (predicted_y, beta).

    Standardises X on its own mean/SD. Centres y before fitting.
    Returns predictions in the original y scale.
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    y_mean = float(y.mean())
    beta, _ = _nnls_numpy(Xz, y - y_mean)
    y_hat = Xz @ beta + y_mean
    return y_hat, beta


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return np.nan
    xs = pd.Series(x[m]).rank(method="average").to_numpy(dtype=float).copy()
    ys = pd.Series(y[m]).rank(method="average").to_numpy(dtype=float).copy()
    xs -= xs.mean()
    ys -= ys.mean()
    denom = float(np.sqrt((xs * xs).sum()) * np.sqrt((ys * ys).sum()))
    return float((xs * ys).sum() / denom) if denom > 0 else np.nan


# ---------------------------------------------------------------------------
# Category loading
# ---------------------------------------------------------------------------

def _load_category_map(canonical_path: Path) -> Dict[str, str]:
    """Return {filename -> food_classification} from the canonical CSV."""
    df = pd.read_csv(canonical_path)
    if "filename" not in df.columns or "food_classification" not in df.columns:
        raise KeyError("canonical CSV missing 'filename' or 'food_classification'")
    return dict(zip(df["filename"].astype(str), df["food_classification"].astype(str)))


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def _permutation_test(
    abs_resid: np.ndarray,
    within_mask: np.ndarray,
    n_perms: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    """Permute category labels and test whether within-category MAR exceeds
    between-category MAR.

    Rather than shuffling pair labels (which would break the pairwise
    dependence structure), we shuffle the item-level category assignments
    and recompute the within/between mask from scratch.

    Returns (observed_diff, p_value, ci_low_95, ci_high_95).
    observed_diff = MAR_within - MAR_between (positive = within > between).
    p_value is one-tailed: P(perm_diff >= observed_diff).
    ci_low/high are the 2.5/97.5 percentiles of the null distribution.
    """
    obs_diff = float(abs_resid[within_mask].mean() - abs_resid[~within_mask].mean())

    rng = np.random.default_rng(seed)
    n_pairs = len(within_mask)
    n_within = int(within_mask.sum())
    perm_diffs = np.empty(n_perms, dtype=float)
    for p in range(n_perms):
        # Randomly assign n_within pairs as "within" — preserves counts
        perm_mask = np.zeros(n_pairs, dtype=bool)
        perm_mask[rng.choice(n_pairs, size=n_within, replace=False)] = True
        perm_diffs[p] = abs_resid[perm_mask].mean() - abs_resid[~perm_mask].mean()

    p_val = float((perm_diffs >= obs_diff).mean())
    ci_lo = float(np.percentile(perm_diffs, 2.5))
    ci_hi = float(np.percentile(perm_diffs, 97.5))
    return obs_diff, p_val, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(pairs_df: pd.DataFrame, out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping plot")
        return

    within  = pairs_df.loc[pairs_df["pair_type"] == "within",  "abs_residual"].to_numpy()
    between = pairs_df.loc[pairs_df["pair_type"] == "between", "abs_residual"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # --- Panel A: violin plot of |residual| by pair type ---
    ax = axes[0]
    colors = ["#4393c3", "#d6604d"]
    data_groups = [within, between]
    group_labels = ["Within-category", "Between-category"]

    parts = ax.violinplot(data_groups, positions=[0, 1], showmedians=True,
                          showextrema=False)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.72)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.8)

    for xi, arr, col in zip([0, 1], data_groups, colors):
        mean = arr.mean()
        se   = arr.std() / np.sqrt(len(arr))
        ax.errorbar(xi, mean, yerr=se, fmt="D", color=col,
                    markerfacecolor="white", markeredgewidth=1.8,
                    markersize=7, capsize=4, zorder=5,
                    label=f"{group_labels[xi]}\n(mean={mean:.3f})")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(group_labels, fontsize=10)
    ax.set_ylabel("Absolute residual  |empirical − predicted|", fontsize=9)
    ax.set_title("A.  Residual magnitude by pair type", fontsize=10, pad=8)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel B: mean |residual| per food category (within-pairs only) ---
    ax2 = axes[1]
    cat_stats = (pairs_df[pairs_df["pair_type"] == "within"]
                 .groupby("category_i")["abs_residual"]
                 .agg(mean="mean", sem=lambda x: x.std() / np.sqrt(len(x)), count="count")
                 .sort_values("mean", ascending=True))

    y_pos = np.arange(len(cat_stats))
    ax2.barh(y_pos, cat_stats["mean"], xerr=cat_stats["sem"],
             color="#4393c3", alpha=0.80, capsize=3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(cat_stats.index, fontsize=9)
    ax2.set_xlabel("Mean absolute residual  (within-category pairs)", fontsize=9)
    ax2.set_title("B.  Within-category residual by food category", fontsize=10, pad=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for yi, (_, row) in zip(y_pos, cat_stats.iterrows()):
        ax2.text(row["mean"] + row["sem"] + 0.001, yi,
                 f"n = {int(row['count'])}", va="center", fontsize=7.5, color="gray")

    fig.tight_layout(pad=2.5)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote: {out_png}")


def _plot_semantic_residuals(df: pd.DataFrame, out_png: Path) -> None:
    """Forest plot of Spearman ρ between each semantic dimension and the
    category-model residuals.  Error bars are 95% bootstrap CIs.
    Bars are coloured by sign and filled/open by significance (p < 0.05).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[WARN] matplotlib not available — skipping semantic residual plot")
        return

    df = df.sort_values("rho", ascending=True).reset_index(drop=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=(8, max(4.0, n * 0.55 + 1.5)))
    y = np.arange(n)

    for i, row in df.iterrows():
        rho = row["rho"]
        ci_lo = row["ci_low_95"]
        ci_hi = row["ci_high_95"]
        sig   = bool(row["significant"])
        color = "#d6604d" if rho > 0 else "#4393c3"
        alpha = 0.90 if sig else 0.45
        marker = "D" if sig else "o"
        ax.errorbar(rho, i,
                    xerr=[[rho - ci_lo], [ci_hi - rho]],
                    fmt=marker, color=color, alpha=alpha,
                    markerfacecolor=color if sig else "white",
                    markeredgecolor=color, markeredgewidth=1.5,
                    markersize=7, capsize=4, linewidth=1.2, zorder=3)
        ax.text(ci_hi + 0.004, i,
                f"ρ={rho:+.3f}{'*' if sig else ''}",
                va="center", ha="left", fontsize=7.5, color="black")

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["predictor"].tolist(), fontsize=9)
    ax.set_xlabel("Spearman ρ with category-model residuals (95% bootstrap CI)",
                  fontsize=9)
    ax.set_title(
        "Semantic dimensions vs. category-model residuals\n"
        "(after accounting for food classification)",
        fontsize=10, pad=8,
    )

    # Right-margin label: category-only R²
    r2_cat = float(df["category_only_r2"].iloc[0])
    ax.text(0.99, 0.02,
            f"Category-only R² = {r2_cat:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="gray")

    # Legend
    handles = [
        mpatches.Patch(color="#d6604d", label="Positive ρ (filled = p < 0.05)"),
        mpatches.Patch(color="#4393c3", label="Negative ρ (filled = p < 0.05)"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right", framealpha=0.85)

    # Widen x-axis slightly
    lo = df["ci_low_95"].min()
    hi = df["ci_high_95"].max()
    span = hi - lo
    ax.set_xlim(lo - 0.05 * span - 0.01, hi + 0.15 * span + 0.01)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=2.0)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote: {out_png}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary",    default=str(SUMMARY_CSV))
    p.add_argument("--model-rdms", default=str(MODEL_RDM_DIR))
    p.add_argument("--canonical",  default=str(CANONICAL_CSV))
    p.add_argument("--outdir",     default=str(DEFAULT_OUTDIR))
    p.add_argument("--n-perms",    type=int, default=5000,
                   help="Permutations for label-shuffle test (default: 5000)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--pool",       default="all",
                   choices=["all", "prolific", "sona"])
    p.add_argument("--fixed-dense-only", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Restrict to fixed+dense trials only (default: False — "
                        "use all trials for maximum pair coverage)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load summary and build empirical RDM
    # ------------------------------------------------------------------
    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"[ERROR] summary.csv not found: {summary_path}")
        return 1

    df = pd.read_csv(summary_path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if args.pool != "all" and "pool" in df.columns:
        df = df[df["pool"].astype(str).str.lower() == args.pool.lower()].copy()

    if args.fixed_dense_only and ("is_fixed" in df.columns or "is_dense" in df.columns):
        is_fixed = _to_bool_series(df["is_fixed"]) if "is_fixed" in df.columns \
                   else pd.Series(False, index=df.index)
        is_dense = _to_bool_series(df["is_dense"]) if "is_dense" in df.columns \
                   else pd.Series(False, index=df.index)
        df = df[is_fixed | is_dense].copy()

    print(f"[INFO] Trials loaded: {len(df)}")
    rdm_full = _build_rdm(df)
    labels   = rdm_full.labels
    n_items  = len(labels)
    print(f"[INFO] Food items: {n_items}")

    # ------------------------------------------------------------------
    # 2. Load category labels
    # ------------------------------------------------------------------
    canonical_path = Path(args.canonical)
    if not canonical_path.exists():
        for alt in _ALT_CANONICAL_PATHS:
            if alt.exists():
                canonical_path = alt
                break
    if not canonical_path.exists():
        print(f"[ERROR] Canonical CSV not found: {canonical_path}")
        return 1

    cat_map         = _load_category_map(canonical_path)
    item_categories = [cat_map.get(lbl, "Unknown") for lbl in labels]
    n_unknown       = sum(c == "Unknown" for c in item_categories)
    if n_unknown > 0:
        print(f"[WARN] {n_unknown}/{n_items} items missing from canonical CSV")
    print(f"[INFO] Categories: {sorted(set(item_categories))}")

    # ------------------------------------------------------------------
    # 3. Load full-feature model RDMs
    # ------------------------------------------------------------------
    model_dir     = Path(args.model_rdms)
    loaded_rdms: Dict[str, np.ndarray] = {}
    missing_stems = []
    for stem in FULL_FEATURE_STEMS:
        npz_path = model_dir / f"{stem}.npz"
        if not npz_path.exists():
            print(f"[WARN] Model RDM not found, skipping: {npz_path}")
            missing_stems.append(stem)
            continue
        try:
            D_raw, m_labels = _load_npz(npz_path)
            loaded_rdms[stem] = _align(D_raw, m_labels, labels)
        except KeyError as e:
            print(f"[WARN] Could not align {stem}: {e}")
            missing_stems.append(stem)

    n_loaded = len(loaded_rdms)
    print(f"[INFO] Loaded {n_loaded}/{len(FULL_FEATURE_STEMS)} model RDMs")
    if missing_stems:
        print(f"[WARN] Missing/skipped: {missing_stems}")
    if n_loaded == 0:
        print("[ERROR] No model RDMs loaded — cannot proceed")
        return 1

    # ------------------------------------------------------------------
    # 4. Build pair vectors (upper triangle, observed pairs only)
    # ------------------------------------------------------------------
    iu, ju  = np.triu_indices(n_items, k=1)
    emp_vec = rdm_full.D[iu, ju]

    model_matrix = np.vstack([loaded_rdms[s][iu, ju]
                               for s in loaded_rdms]).T  # (n_pairs, n_models)

    ok_emp = np.isfinite(emp_vec)
    ok_mod = np.all(np.isfinite(model_matrix), axis=1)
    ok     = ok_emp & ok_mod

    n_pairs = int(ok.sum())
    print(f"[INFO] Valid pairs for regression: {n_pairs} / {len(emp_vec)}")

    emp_ok   = emp_vec[ok]
    model_ok = model_matrix[ok]
    iu_ok    = iu[ok]
    ju_ok    = ju[ok]

    # ------------------------------------------------------------------
    # 5. Fit NNLS model and compute residuals
    # ------------------------------------------------------------------
    print(f"[INFO] Fitting NNLS model ({n_loaded} predictors)...")
    y_hat, beta = _nnls_predict(model_ok, emp_ok)
    residuals    = emp_ok - y_hat
    abs_residuals = np.abs(residuals)

    rho_fit = _spearman(y_hat, emp_ok)
    ss_res  = float(np.sum(residuals ** 2))
    ss_tot  = float(np.sum((emp_ok - emp_ok.mean()) ** 2))
    r2_fit  = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    print(f"[INFO] Full-feature model fit: ρ = {rho_fit:.4f}, R² = {r2_fit:.4f}")

    stems_used = list(loaded_rdms.keys())
    nonzero = [(stems_used[i], float(beta[i]))
               for i in range(len(beta)) if beta[i] > 1e-6]
    print(f"[INFO] Non-zero NNLS weights ({len(nonzero)}/{len(beta)}):")
    for stem, w in sorted(nonzero, key=lambda x: -x[1]):
        print(f"       {stem:<48s}  β = {w:.4f}")

    # ------------------------------------------------------------------
    # 6. Build pair-level DataFrame with category labels
    # ------------------------------------------------------------------
    cat_i     = [item_categories[i] for i in iu_ok]
    cat_j     = [item_categories[j] for j in ju_ok]
    pair_type = ["within" if ci == cj else "between"
                 for ci, cj in zip(cat_i, cat_j)]

    pairs_df = pd.DataFrame({
        "i":            iu_ok,
        "j":            ju_ok,
        "label_i":      [labels[i] for i in iu_ok],
        "label_j":      [labels[j] for j in ju_ok],
        "category_i":   cat_i,
        "category_j":   cat_j,
        "pair_type":    pair_type,
        "empirical":    emp_ok,
        "predicted":    y_hat,
        "residual":     residuals,
        "abs_residual": abs_residuals,
    })

    n_within  = int((pairs_df["pair_type"] == "within").sum())
    n_between = int((pairs_df["pair_type"] == "between").sum())
    print(f"[INFO] Within-category pairs:  {n_within}")
    print(f"[INFO] Between-category pairs: {n_between}")

    # ------------------------------------------------------------------
    # 7. Descriptive statistics by pair type
    # ------------------------------------------------------------------
    summary_rows = []
    for ptype in ["within", "between"]:
        sub = pairs_df[pairs_df["pair_type"] == ptype]
        summary_rows.append({
            "pair_type":            ptype,
            "n_pairs":              len(sub),
            "mean_abs_residual":    float(sub["abs_residual"].mean()),
            "sd_abs_residual":      float(sub["abs_residual"].std()),
            "median_abs_residual":  float(sub["abs_residual"].median()),
            "mean_signed_residual": float(sub["residual"].mean()),
            "sd_signed_residual":   float(sub["residual"].std()),
        })
    summary_df = pd.DataFrame(summary_rows)
    print("\n[INFO] Residual summary:")
    print(summary_df.to_string(index=False))

    # Per-category breakdown
    cat_summary = (pairs_df[pairs_df["pair_type"] == "within"]
                   .groupby("category_i")["abs_residual"]
                   .agg(n="count", mean="mean", sd="std", median="median")
                   .sort_values("mean", ascending=False))
    print("\n[INFO] Within-category MAR by food category:")
    print(cat_summary.to_string())

    # ------------------------------------------------------------------
    # 8. Permutation test
    # ------------------------------------------------------------------
    print(f"\n[INFO] Running permutation test ({args.n_perms} permutations)...")
    within_mask = (pairs_df["pair_type"] == "within").to_numpy()
    obs_diff, p_val, ci_lo, ci_hi = _permutation_test(
        abs_residuals, within_mask, args.n_perms, args.seed
    )
    print(f"[INFO] Observed MAR difference (within − between): {obs_diff:+.6f}")
    print(f"[INFO] Permutation p-value (one-tailed):           {p_val:.4f}")
    print(f"[INFO] Null 95% CI:                                [{ci_lo:.6f}, {ci_hi:.6f}]")

    # ------------------------------------------------------------------
    # 9. Mann-Whitney U test
    # ------------------------------------------------------------------
    try:
        from scipy.stats import mannwhitneyu
        stat_mw, p_mw = mannwhitneyu(
            abs_residuals[within_mask],
            abs_residuals[~within_mask],
            alternative="greater",
        )
        print(f"[INFO] Mann-Whitney U (within > between): U = {stat_mw:.1f}, p = {p_mw:.4f}")
    except ImportError:
        stat_mw, p_mw = np.nan, np.nan
        print("[WARN] scipy not available — Mann-Whitney test skipped")

    # ------------------------------------------------------------------
    # 10. Semantic dimensions vs. category residuals
    #
    # Question: after accounting for food category alone, do continuous
    # semantic dimensions (healthiness, savoriness, etc.) explain variance
    # in the residuals? This tests whether semantic dimensions carry
    # information about food similarity beyond broad category membership.
    #
    # Procedure:
    #   a) Fit a category-only NNLS model → category residuals
    #   b) Spearman-correlate each semantic predictor with those residuals
    #   c) Bootstrap 95% CIs (resample pairs); permutation p-values
    #      (shuffle each predictor vector independently)
    # ------------------------------------------------------------------
    SEMANTIC_STEMS = [
        "rdm_human_sweetness",
        "rdm_human_saltiness",
        "rdm_human_sourness",
        "rdm_human_bitterness",
        "rdm_human_savoriness",
        "rdm_human_fattiness",
        "rdm_human_spiciness",
        "rdm_human_healthiness",
        "rdm_human_calorie_density",
        "rdm_human_appeal",
    ]
    available_semantic = [s for s in SEMANTIC_STEMS if s in loaded_rdms]
    missing_semantic   = [s for s in SEMANTIC_STEMS if s not in loaded_rdms]
    if missing_semantic:
        print(f"[WARN] Semantic stems missing for category-residual analysis: {missing_semantic}")

    print(f"\n[INFO] Fitting category-only NNLS model...")
    cat_stem = "rdm_food_classification"
    if cat_stem not in loaded_rdms:
        print(f"[WARN] {cat_stem} not loaded — skipping semantic residual analysis")
        cat_resid_rows = []
    else:
        cat_vec = loaded_rdms[cat_stem][iu_ok[np.arange(len(iu_ok))], ju_ok[np.arange(len(ju_ok))]]
        # Use the already-filtered ok pairs
        cat_X = cat_vec.reshape(-1, 1)
        cat_yhat, cat_beta = _nnls_predict(cat_X, emp_ok)
        cat_residuals = emp_ok - cat_yhat

        rho_cat = _spearman(cat_yhat, emp_ok)
        ss_cat  = float(np.sum(cat_residuals ** 2))
        r2_cat  = 1.0 - ss_cat / ss_tot if ss_tot > 0 else np.nan
        print(f"[INFO] Category-only model fit: ρ = {rho_cat:.4f}, R² = {r2_cat:.4f}")
        print(f"[INFO] Variance unexplained by category: {1.0 - r2_cat:.4f}")

        # Correlate each semantic predictor with the category residuals
        N_BOOT  = 1000
        N_PERM  = 1000
        rng_sem = np.random.default_rng(args.seed + 1)
        n_ok    = len(emp_ok)

        cat_resid_rows = []
        print(f"\n[INFO] Semantic dimensions vs. category residuals "
              f"({N_BOOT} bootstrap, {N_PERM} permutations each):")

        for stem in available_semantic:
            pred_vec = loaded_rdms[stem][iu, ju][ok]  # align to same ok mask

            # Observed Spearman ρ
            rho_obs = _spearman(pred_vec, cat_residuals)

            # Bootstrap 95% CI
            boot_rhos = np.empty(N_BOOT, dtype=float)
            for b in range(N_BOOT):
                idx = rng_sem.integers(0, n_ok, size=n_ok)
                boot_rhos[b] = _spearman(pred_vec[idx], cat_residuals[idx])
            ci_lo_sem = float(np.percentile(boot_rhos, 2.5))
            ci_hi_sem = float(np.percentile(boot_rhos, 97.5))

            # Permutation p-value (two-tailed: |perm_rho| >= |obs_rho|)
            perm_rhos = np.empty(N_PERM, dtype=float)
            for p in range(N_PERM):
                perm_rhos[p] = _spearman(rng_sem.permutation(pred_vec), cat_residuals)
            p_perm = float((np.abs(perm_rhos) >= abs(rho_obs)).mean())

            short_name = stem.replace("rdm_human_", "")
            sig = "*" if p_perm < 0.05 else " "
            print(f"       {short_name:<20s}  ρ = {rho_obs:+.4f}  "
                  f"95% CI [{ci_lo_sem:+.4f}, {ci_hi_sem:+.4f}]  "
                  f"p = {p_perm:.3f} {sig}")

            cat_resid_rows.append({
                "predictor":    short_name,
                "stem":         stem,
                "rho":          rho_obs,
                "ci_low_95":    ci_lo_sem,
                "ci_high_95":   ci_hi_sem,
                "p_perm":       p_perm,
                "significant":  p_perm < 0.05,
                "category_only_r2": r2_cat,
            })

    # ------------------------------------------------------------------
    # 11. Write outputs
    # ------------------------------------------------------------------
    pairs_path = outdir / "residual_pairs.csv"
    pairs_df.to_csv(pairs_path, index=False)
    print(f"\n[OK] Wrote: {pairs_path}")

    summary_path_out = outdir / "residual_summary.csv"
    summary_df.to_csv(summary_path_out, index=False)
    print(f"[OK] Wrote: {summary_path_out}")

    cat_summary_path = outdir / "residual_by_category.csv"
    cat_summary.to_csv(cat_summary_path)
    print(f"[OK] Wrote: {cat_summary_path}")

    test_dict = {
        "observed_diff_within_minus_between": obs_diff,
        "p_value_one_tailed":                 p_val,
        "null_ci_low_95":                     ci_lo,
        "null_ci_high_95":                    ci_hi,
        "n_permutations":                     args.n_perms,
        "mannwhitney_U":        float(stat_mw) if np.isfinite(stat_mw) else None,
        "mannwhitney_p_one_tailed": float(p_mw) if np.isfinite(p_mw) else None,
        "n_within_pairs":       n_within,
        "n_between_pairs":      n_between,
        "full_model_rho":       rho_fit,
        "full_model_r2":        r2_fit,
        "n_predictors_loaded":  n_loaded,
        "n_predictors_missing": len(missing_stems),
    }
    test_path = outdir / "residual_test.json"
    test_path.write_text(json.dumps(test_dict, indent=2))
    print(f"[OK] Wrote: {test_path}")

    _plot(pairs_df, outdir / "fig_residual_by_category.png")

    # --- Semantic vs. category residuals outputs ---
    sem_csv_path  = outdir / "semantic_vs_category_residuals.csv"
    sem_plot_path = outdir / "fig_semantic_vs_category_residuals.png"
    if cat_resid_rows:
        sem_df = pd.DataFrame(cat_resid_rows)
        sem_df.to_csv(sem_csv_path, index=False)
        print(f"[OK] Wrote: {sem_csv_path}")
        _plot_semantic_residuals(sem_df, sem_plot_path)
        category_only_r2  = float(sem_df["category_only_r2"].iloc[0])
        category_only_rho = rho_cat  # defined in step 10 block
    else:
        print("[WARN] cat_resid_rows is empty — semantic residual outputs skipped")
        category_only_r2  = None
        category_only_rho = None

    manifest = {
        "timestamp":            datetime.now().isoformat(timespec="seconds"),
        "pool":                 args.pool,
        "fixed_dense_only":     args.fixed_dense_only,
        "n_items":              n_items,
        "n_pairs_total":        n_pairs,
        "n_pairs_within":       n_within,
        "n_pairs_between":      n_between,
        "n_perms":              args.n_perms,
        "seed":                 args.seed,
        "predictors_loaded":    list(loaded_rdms.keys()),
        "predictors_missing":   missing_stems,
        "full_model_rho":       rho_fit,
        "full_model_r2":        r2_fit,
        "category_only_rho":    category_only_rho,
        "category_only_r2":     category_only_r2,
        "outputs": {
            "residual_pairs":                  str(pairs_path),
            "residual_summary":                str(summary_path_out),
            "residual_by_category":            str(cat_summary_path),
            "residual_test":                   str(test_path),
            "semantic_vs_category_residuals":  str(sem_csv_path) if cat_resid_rows else None,
            "fig_semantic_vs_category":        str(sem_plot_path) if cat_resid_rows else None,
        },
    }
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[OK] Wrote: {manifest_path}")

    # ------------------------------------------------------------------
    # 12. Final interpretation summary
    # ------------------------------------------------------------------
    mar_within  = summary_df.loc[summary_df["pair_type"] == "within",  "mean_abs_residual"].iloc[0]
    mar_between = summary_df.loc[summary_df["pair_type"] == "between", "mean_abs_residual"].iloc[0]
    print("\n=== Interpretation guide ===")
    print(f"  Full-feature model fit:        ρ = {rho_fit:.4f},  R² = {r2_fit:.4f}")
    print(f"  MAR within-category:           {mar_within:.4f}")
    print(f"  MAR between-category:          {mar_between:.4f}")
    print(f"  Difference (within − between): {obs_diff:+.4f}")
    print(f"  Permutation p (one-tailed):    {p_val:.4f}")
    if obs_diff > 0 and p_val < 0.05:
        print("  RESULT: Residuals concentrate WITHIN categories.")
        print("  Interpretation: Unexplained variance is structured — the model resolves")
        print("  between-category differences well but leaves fine-grained within-category")
        print("  similarity (e.g. grain-to-grain or protein-to-protein differences) unaccounted.")
        print("  Candidate missing dimensions: texture, cultural familiarity, specific flavour notes.")
    elif obs_diff > 0 and p_val >= 0.05:
        print("  RESULT: Trend toward within-category concentration but not significant (p >= 0.05).")
    else:
        print("  RESULT: No within-category concentration.")
        print("  Interpretation: Residuals appear unstructured — unexplained variance is likely noise.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
