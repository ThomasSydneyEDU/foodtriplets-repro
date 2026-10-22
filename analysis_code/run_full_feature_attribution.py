#!/usr/bin/env python3
"""run_full_feature_attribution.py

Supplementary Shapley + partial-Spearman attribution that combines:
  - human-rated food attributes (from the existing pairwise_vectors.csv)
  - low-level visual feature RDMs  (ll_*)
  - food category model            (rdm_food_classification)

This script is READ-ONLY with respect to all existing output folders.
New outputs go to:
  compute_outputs/stage4_variance_attribution_full/empirical/human/
  results/stage6_variance_attribution_full/empirical/human/

Usage
-----
  cd <repo_root>
  python analysis_code/run_full_feature_attribution.py [--n-orderings 300] [--n-boot 200]
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DERIVATIVES_DIR  = HERE / "derivatives"
MODEL_RDMS_DIR   = DERIVATIVES_DIR / "model_rdms_canon"
EMPIRICAL_RDM_NPZ = DERIVATIVES_DIR / "triplet_rdm_by_filename.npz"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Full-feature Shapley attribution")
    p.add_argument("--n-orderings", type=int, default=300,
                   help="Shapley: number of random predictor orderings (default 300)")
    p.add_argument("--n-boot",      type=int, default=50,
                   help="Bootstrap resamples for CIs (default 50)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--include-conceptual-contrasts", action="store_true",
                   help="Also include Sweet-vs-savory and Natural-vs-transformed RDMs")
    p.add_argument("--rdm-mode", type=str, default="fixed_dense",
                   choices=["empirical", "fixed_dense", "filled"],
                   help="RDM mode for input/output paths (default: fixed_dense)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Statistics helpers (self-contained, no dependency on compute_* scripts)
# ---------------------------------------------------------------------------

def _zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    mu, sd = np.nanmean(v), np.nanstd(v)
    return v * 0.0 if (not np.isfinite(sd) or sd == 0) else (v - mu) / sd


def _prerank(v: np.ndarray) -> np.ndarray:
    """Rank-transform a vector once (average ties), then z-score."""
    r = pd.Series(np.asarray(v, dtype=float)).rank(method="average").to_numpy(float)
    return _zscore(r)


def _r2_from_preranked(y_z: np.ndarray, X_cols_z: list[np.ndarray],
                        idx: list[int] | None = None) -> float:
    """R² from OLS on already z-scored (pre-ranked) y and selected X columns.

    idx: column indices to use (None = all). Avoids rebuilding matrices in the loop.
    """
    if idx is not None:
        cols = [X_cols_z[i] for i in idx]
    else:
        cols = X_cols_z
    if not cols:
        return 0.0
    X_z = np.column_stack(cols)
    if np.allclose(X_z, 0.0):
        return 0.0
    beta, *_ = np.linalg.lstsq(X_z, y_z, rcond=None)
    ss_res = float(np.sum((y_z - X_z @ beta) ** 2))
    ss_tot = float(np.sum(y_z ** 2))          # y_z is already mean-centred (z-scored)
    return float(max(0.0, min(1.0, 1.0 - ss_res / ss_tot))) if ss_tot > 0 else 0.0


def _shapley(y: np.ndarray, X_dict: dict, n_orderings: int = 300,
             rank_transform: bool = True, seed: int = 42) -> dict:
    """Approximate Shapley ΔR² via random predictor orderings.

    Pre-ranks y and each predictor column *once*, then runs plain OLS in the loop
    — much faster than re-ranking inside every OLS call.
    """
    rng   = np.random.default_rng(seed)
    names = list(X_dict.keys())
    n_p   = len(names)

    # Pre-rank (z-scored) vectors — done once
    y_z   = _prerank(y) if rank_transform else _zscore(y)
    X_z   = [_prerank(X_dict[k]) if rank_transform else _zscore(X_dict[k]) for k in names]
    name_to_idx = {k: i for i, k in enumerate(names)}

    contrib = {k: 0.0 for k in names}

    for _ in range(n_orderings):
        order = list(range(n_p))
        rng.shuffle(order)
        included: list[int] = []
        cur_r2 = 0.0
        for ci in order:
            included.append(ci)
            new_r2 = _r2_from_preranked(y_z, X_z, included)
            contrib[names[ci]] += new_r2 - cur_r2
            cur_r2 = new_r2

    for k in contrib:
        contrib[k] /= float(n_orderings)

    contrib["__baseline_R2__"]       = 0.0
    contrib["__total_R2_with_all__"] = _r2_from_preranked(y_z, X_z)
    return contrib


def _bootstrap_shapley(y: np.ndarray, X_dict: dict, n_orderings: int,
                        n_boot: int, rank_transform: bool = True, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    names = list(X_dict.keys())
    vals  = {k: np.empty(n_boot) for k in names}
    tot   = np.empty(n_boot)
    n     = len(y)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb  = y[idx]
        Xb  = {k: v[idx] for k, v in X_dict.items()}
        out = _shapley(yb, Xb, n_orderings=max(10, n_orderings // 30),
                       rank_transform=rank_transform, seed=int(seed + 10 + b))
        tot[b] = out["__total_R2_with_all__"]
        for k in names:
            vals[k][b] = out.get(k, np.nan)

    rows = []
    for k in names:
        rows.append({
            "predictor": k,
            "mean":    float(np.nanmean(vals[k])),
            "ci95_lo": float(np.nanpercentile(vals[k], 2.5)),
            "ci95_hi": float(np.nanpercentile(vals[k], 97.5)),
        })
    rows.append({"predictor": "__total_R2_with_all__",
                 "mean":    float(np.nanmean(tot)),
                 "ci95_lo": float(np.nanpercentile(tot, 2.5)),
                 "ci95_hi": float(np.nanpercentile(tot, 97.5))})
    return pd.DataFrame(rows)


def _partial_spearman_preranked(yr_z: np.ndarray, xr_z: np.ndarray,
                                 Zr_z: np.ndarray | None = None) -> float:
    """Partial Spearman on pre-ranked, z-scored vectors. Much faster in loops."""
    if Zr_z is not None and Zr_z.shape[1] > 0:
        yr_res = yr_z - Zr_z @ np.linalg.lstsq(Zr_z, yr_z, rcond=None)[0]
        xr_res = xr_z - Zr_z @ np.linalg.lstsq(Zr_z, xr_z, rcond=None)[0]
    else:
        yr_res, xr_res = yr_z, xr_z
    a = yr_res - yr_res.mean()
    b = xr_res - xr_res.mean()
    denom = np.std(a) * np.std(b)
    return float((a * b).mean() / denom) if denom > 0 else 0.0


def _bootstrap_partial(y_raw, x_raw, Z_raw, n_boot=200, seed=42):
    """Bootstrap CI and permutation p-value for partial Spearman.

    Pre-ranks raw inputs once; bootstrap re-samples rows of the ranked data.
    """
    rng = np.random.default_rng(seed)
    n   = len(y_raw)

    # Pre-rank everything once
    yr_z = _prerank(y_raw)
    xr_z = _prerank(x_raw)
    Zr_z = (np.column_stack([_prerank(Z_raw[:, j]) for j in range(Z_raw.shape[1])])
            if Z_raw is not None and Z_raw.shape[1] > 0 else None)

    true_rho = _partial_spearman_preranked(yr_z, xr_z, Zr_z)

    boot = np.empty(n_boot)
    perm = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Zb  = Zr_z[idx] if Zr_z is not None else None
        boot[b] = _partial_spearman_preranked(yr_z[idx], xr_z[idx], Zb)
        xp = xr_z.copy(); rng.shuffle(xp)
        perm[b] = _partial_spearman_preranked(yr_z, xp, Zr_z)

    p_val = float(np.mean(np.abs(perm) >= abs(true_rho)))
    return (true_rho,
            float(np.nanpercentile(boot, 2.5)),
            float(np.nanpercentile(boot, 97.5)),
            p_val)

# ---------------------------------------------------------------------------
# RDM loading & alignment helpers
# ---------------------------------------------------------------------------

def _load_npz_rdm(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load an RDM npz; return (matrix, labels_list)."""
    d = np.load(path, allow_pickle=True)
    # Try several key names used across the codebase
    for key in ("rdm", "data", "D"):
        if key in d:
            mat = d[key].astype(float)
            break
    else:
        raise KeyError(f"No recognised RDM key in {path}. Keys: {list(d.keys())}")
    labels = list(d["labels"].tolist()) if "labels" in d else []
    labels = [str(x) for x in labels]
    return mat, labels


def _align_rdm_to_emp(model_mat: np.ndarray, model_labels: list[str],
                      emp_labels: list[str]) -> np.ndarray:
    """Re-order model_mat rows/cols to match emp_labels ordering.
    Stimuli missing from the model get NaN rows/cols."""
    emp_base  = [os.path.basename(str(x)) for x in emp_labels]
    mod_base  = [os.path.basename(str(x)) for x in model_labels]
    mod_idx   = {lbl: i for i, lbl in enumerate(mod_base)}

    n = len(emp_labels)
    out = np.full((n, n), np.nan)
    for i, li in enumerate(emp_base):
        for j, lj in enumerate(emp_base):
            mi = mod_idx.get(li)
            mj = mod_idx.get(lj)
            if mi is not None and mj is not None:
                out[i, j] = model_mat[mi, mj]
    return out


def _extract_pairwise_vec(mat: np.ndarray, i_arr: np.ndarray, j_arr: np.ndarray) -> np.ndarray:
    """Pull mat[i, j] for each (i, j) pair."""
    return np.array([mat[ii, jj] for ii, jj in zip(i_arr, j_arr)], dtype=float)

# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

_PRETTY = {
    # low-level visual
    "ll_mean_luminance":        "Luminance (mean)",
    "ll_rms_contrast":          "Contrast (RMS)",
    "ll_hsv_s_mean":            "Colour saturation",
    "ll_edge_energy":           "Edge energy",
    "ll_lab_mean":              "Lab colour (mean)",
    "ll_lab_std":               "Lab colour (std)",
    "ll_hog_pcs10":             "Shape/texture (HOG PCs)",
    "ll_lbp_hist":              "Texture (LBP)",
    "ll_power_radial30":        "Spatial frequency",
    "ll_resnet50_conv1_pcs10":  "DNN conv1 (ResNet-50)",
    "ll_resnet50_layer3_pcs10": "DNN layer3 (ResNet-50)",
    # category
    "food_classification":      "Food category",
    # conceptual contrasts (natural_vs_transformed excluded from Shapley)
    # human-rated
    "human_healthiness":    "Healthiness",
    "human_sweetness":      "Sweetness",
    "human_calorie_density":"Calorie density",
    "human_savoriness":     "Savoriness",
    "human_saltiness":      "Saltiness",
    "human_fattiness":      "Fattiness",
    "human_appeal":         "Appeal",
    "human_spiciness":      "Spiciness",
    "human_sourness":       "Sourness",
    "human_bitterness":     "Bitterness",
}

_GROUP = {
    "ll_":       ("Low-level visual", "#6baed6"),
    "food_clas": ("Food category",    "#31a354"),
    "concept_":  ("Conceptual",       "#fd8d3c"),
    "human_":    ("Human-rated",      "#e6550d"),
}

def _pretty(name: str) -> str:
    return _PRETTY.get(name, name.replace("_", " ").title())

def _group_color(name: str) -> str:
    for prefix, (_, color) in _GROUP.items():
        if name.startswith(prefix):
            return color
    return "#999999"

def _group_label(name: str) -> str:
    for prefix, (label, _) in _GROUP.items():
        if name.startswith(prefix):
            return label
    return "Other"

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_shapley(df_ci: pd.DataFrame, out_path: Path, total_r2: float) -> None:
    """Horizontal bar chart of Shapley Δρ² with 95% CI error bars."""
    df = df_ci[~df_ci["predictor"].str.startswith("__")].copy()
    df = df.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(4, 0.45 * len(df))))
    colors = [_group_color(p) for p in df["predictor"]]
    y_pos  = np.arange(len(df))
    xerr_lo = np.clip(df["mean"].to_numpy() - df["ci95_lo"].to_numpy(), 0, None)
    xerr_hi = np.clip(df["ci95_hi"].to_numpy() - df["mean"].to_numpy(), 0, None)
    xerr    = np.array([xerr_lo, xerr_hi])
    ax.barh(y_pos, df["mean"], color=colors, alpha=0.85, height=0.65)
    ax.errorbar(df["mean"], y_pos, xerr=xerr, fmt="none",
                color="black", linewidth=1.0, capsize=3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_pretty(p) for p in df["predictor"]], fontsize=9)
    ax.set_xlabel("Shapley Δρ² (mean ± 95% CI)", fontsize=10)
    ax.set_title(f"Feature importance for food discrimination\n(total R² = {total_r2:.3f})", fontsize=11)

    # Legend for groups
    seen = {}
    for p, c in zip(df["predictor"], colors):
        gl = _group_label(p)
        if gl not in seen:
            seen[gl] = c
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85) for c in seen.values()]
    ax.legend(handles, list(seen.keys()), fontsize=8, loc="lower right")

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


def _plot_partial(df_unique: pd.DataFrame, out_path: Path) -> None:
    """Forest plot of partial Spearman ρ with 95% CI and significance marker."""
    df = df_unique.copy().sort_values("rho_partial", ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(4, 0.45 * len(df))))
    colors = [_group_color(p) for p in df["model"]]
    y_pos  = np.arange(len(df))
    xerr_lo = np.clip(df["rho_partial"].to_numpy() - df["ci95_lo"].to_numpy(), 0, None)
    xerr_hi = np.clip(df["ci95_hi"].to_numpy() - df["rho_partial"].to_numpy(), 0, None)
    xerr    = np.array([xerr_lo, xerr_hi])
    ax.barh(y_pos, df["rho_partial"], color=colors, alpha=0.85, height=0.65)
    ax.errorbar(df["rho_partial"], y_pos, xerr=xerr, fmt="none",
                color="black", linewidth=1.0, capsize=3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_pretty(p) for p in df["model"]], fontsize=9)
    ax.set_xlabel("Partial Spearman ρ (unique, ± 95% CI)", fontsize=10)
    ax.set_title("Unique variance — all features\n(controlling for all others)", fontsize=11)

    # n.s. markers
    for yi, (_, row) in zip(y_pos, df.iterrows()):
        if row["p_perm"] >= 0.05:
            ax.text(row["ci95_hi"] + 0.002, yi, "n.s.", va="center", fontsize=7, color="grey")

    # Legend
    seen = {}
    for p, c in zip(df["model"], colors):
        gl = _group_label(p)
        if gl not in seen:
            seen[gl] = c
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85) for c in seen.values()]
    ax.legend(handles, list(seen.keys()), fontsize=8, loc="lower right")

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    out_compute = HERE / "compute_outputs" / "stage4_variance_attribution_full" / args.rdm_mode / "human"
    out_results = HERE / "results"  / "stage6_variance_attribution_full"  / args.rdm_mode / "human"

    out_compute.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load existing pairwise vectors (human-rated attributes + emp)    #
    # ------------------------------------------------------------------ #
    existing_pairwise_csv = (
        HERE / "compute_outputs" / "stage2_model_evaluation" / args.rdm_mode / "human" / "pairwise_vectors.csv"
    )
    print(f"\n[1/5] Loading pairwise_vectors.csv from {existing_pairwise_csv} …")
    if not existing_pairwise_csv.exists():
        sys.exit(f"[ERROR] Not found: {existing_pairwise_csv}")

    pv = pd.read_csv(existing_pairwise_csv)
    print(f"      Loaded {len(pv)} rows, {len(pv.columns)} columns")

    # Filter to regression rows only (used_in_regression == True)
    used_col = "used_in_regression"
    if used_col in pv.columns:
        mask = pv[used_col].astype(str).str.lower().isin({"1", "true", "t", "yes"})
        pv = pv[mask].reset_index(drop=True)
        print(f"      After used_in_regression filter: {len(pv)} rows")

    i_arr = pv["i"].to_numpy(int)
    j_arr = pv["j"].to_numpy(int)
    y     = pv["emp"].to_numpy(float)

    # ------------------------------------------------------------------ #
    # 2. Load empirical RDM to get stimulus label ordering                #
    # ------------------------------------------------------------------ #
    print(f"\n[2/5] Loading empirical RDM labels …")
    if not EMPIRICAL_RDM_NPZ.exists():
        sys.exit(f"[ERROR] Empirical RDM not found: {EMPIRICAL_RDM_NPZ}")

    emp_npz    = np.load(EMPIRICAL_RDM_NPZ, allow_pickle=True)
    emp_labels = [str(x) for x in emp_npz["labels"].tolist()]
    print(f"      {len(emp_labels)} stimuli in empirical RDM")

    # ------------------------------------------------------------------ #
    # 3. Collect model RDMs to append                                     #
    # ------------------------------------------------------------------ #
    print(f"\n[3/5] Loading model RDMs from {MODEL_RDMS_DIR} …")

    # ll_* models to include (all present in model_rdms_canon/)
    ll_candidates = [
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
    ]
    extra_candidates = ["rdm_food_classification"]
    # natural_vs_transformed excluded from Shapley
    # (its family-wise contribution is negligible after controlling for semantic dimensions)

    all_candidates = ll_candidates + extra_candidates

    # The column name in pairwise_vectors will strip the "rdm_" prefix
    new_columns: dict[str, np.ndarray] = {}

    for stem in all_candidates:
        npz_path = MODEL_RDMS_DIR / f"{stem}.npz"
        if not npz_path.exists():
            print(f"      [SKIP] Not found: {npz_path.name}")
            continue
        try:
            mat, mod_labels = _load_npz_rdm(npz_path)
        except Exception as e:
            print(f"      [WARN] Could not load {npz_path.name}: {e}")
            continue

        aligned = _align_rdm_to_emp(mat, mod_labels, emp_labels)
        vec     = _extract_pairwise_vec(aligned, i_arr, j_arr)
        nan_frac = np.isnan(vec).mean()
        if nan_frac > 0.5:
            print(f"      [SKIP] {npz_path.name}: >{nan_frac*100:.0f}% NaN after alignment")
            continue

        col_name = stem.removeprefix("rdm_")  # e.g. "ll_mean_luminance"
        new_columns[col_name] = vec
        print(f"      [OK]   {col_name}  (NaN: {nan_frac*100:.1f}%)")

    print(f"      Added {len(new_columns)} new predictor columns")

    # ------------------------------------------------------------------ #
    # 4. Build predictor dict — existing human_* + new columns           #
    # ------------------------------------------------------------------ #
    print(f"\n[4/5] Building predictor set …")

    # Columns to skip in pairwise_vectors.csv
    skip_cols = {"i", "j", "emp", "used_in_regression",
                 "sweet_vs_savory",  # removed from all analyses (redundant with continuous dims)
                 # exclude ai_* to keep analysis focused on human ratings + visual
                 *[c for c in pv.columns if "ai_" in c]}

    X_dict: dict[str, np.ndarray] = {}

    # Add existing human-rated columns
    for col in pv.columns:
        if col in skip_cols:
            continue
        vec = pd.to_numeric(pv[col], errors="coerce").to_numpy(float)
        if np.isnan(vec).mean() > 0.5:
            print(f"      [SKIP] {col}: too many NaN")
            continue
        X_dict[col] = vec
        print(f"      [human] {col}")

    # Add new columns (ll_* and food_classification)
    for col, vec in new_columns.items():
        if col in X_dict:
            print(f"      [DUP]  {col} already present, skipping")
            continue
        # Fill residual NaN with column median
        med = np.nanmedian(vec)
        vec = np.where(np.isnan(vec), med, vec)
        X_dict[col] = vec
        print(f"      [new]  {col}")

    predictor_names = list(X_dict.keys())
    print(f"\n      Total predictors: {len(predictor_names)}")

    # ------------------------------------------------------------------ #
    # 5a. Shapley attribution                                             #
    # ------------------------------------------------------------------ #
    print(f"\n[5a/5] Running Shapley attribution "
          f"({args.n_orderings} orderings, {args.n_boot} bootstrap reps) …")

    shapley_mean = _shapley(y, X_dict, n_orderings=args.n_orderings,
                            rank_transform=True, seed=args.seed)

    print(f"\n  Total R²: {shapley_mean['__total_R2_with_all__']:.4f}")

    # Bootstrap CIs
    shapley_ci_df = _bootstrap_shapley(y, X_dict,
                                       n_orderings=args.n_orderings,
                                       n_boot=args.n_boot,
                                       rank_transform=True,
                                       seed=args.seed)

    # Merge point estimates from full-data run into CI table
    ci_map = dict(zip(shapley_ci_df["predictor"], shapley_ci_df["mean"]))
    for k in predictor_names:
        if k in shapley_mean:
            ci_map[k] = shapley_mean[k]  # prefer full-data estimate for the mean

    shapley_df = pd.DataFrame([
        {"predictor": k, "mean": shapley_mean.get(k, np.nan)}
        for k in ["__total_R2_with_all__", "__baseline_R2__"] + predictor_names
    ])

    # Final CI df: use point estimates from full run, CIs from bootstrap
    shapley_ci_out = shapley_ci_df.copy()
    for k in predictor_names:
        if k in shapley_mean:
            shapley_ci_out.loc[shapley_ci_out["predictor"] == k, "mean"] = shapley_mean[k]

    # ------------------------------------------------------------------ #
    # 5b. Partial Spearman (each predictor, controlling for all others)   #
    # ------------------------------------------------------------------ #
    print(f"\n[5b/5] Running partial Spearman (unique variance) …")

    unique_rows = []
    names_arr = predictor_names
    X_all = np.column_stack([X_dict[k] for k in names_arr])

    for idx, name in enumerate(names_arr):
        x_focal = X_all[:, idx]
        # Control matrix = all other predictors
        other_idx = [j for j in range(len(names_arr)) if j != idx]
        Z = X_all[:, other_idx] if other_idx else None
        rho, lo, hi, pval = _bootstrap_partial(y, x_focal, Z,
                                               n_boot=args.n_boot, seed=args.seed)
        unique_rows.append({"model": name, "rho_partial": rho,
                             "ci95_lo": lo, "ci95_hi": hi, "p_perm": pval})
        sig = "" if pval < 0.05 else "  [n.s.]"
        print(f"      {_pretty(name):35s}  ρ={rho:+.3f}  p={pval:.3f}{sig}")

    unique_df = pd.DataFrame(unique_rows).sort_values("rho_partial", ascending=False)

    # ------------------------------------------------------------------ #
    # Save CSVs                                                           #
    # ------------------------------------------------------------------ #
    shapley_df.to_csv(out_compute / "variance_attribution_shapley.csv", index=False)
    shapley_ci_out.to_csv(out_compute / "variance_attribution_shapley_ci.csv", index=False)
    unique_df.to_csv(out_compute / "model_unique.csv", index=False)

    manifest = {
        "pairwise_csv": str(existing_pairwise_csv),
        "rows_used":    int(len(pv)),
        "method":       "shapley",
        "n_orderings":  args.n_orderings,
        "n_boot":       args.n_boot,
        "predictors":   predictor_names,
        "rank_transform": True,
        "note": (
            "Full-feature run: human-rated food attributes + low-level visual RDMs "
            "+ food category RDM. Writes to _full folders; existing outputs untouched."
        ),
    }
    (out_compute / "variance_attribution_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\n[OK] Compute outputs written to: {out_compute}")

    # ------------------------------------------------------------------ #
    # Plots                                                               #
    # ------------------------------------------------------------------ #
    total_r2 = float(shapley_mean["__total_R2_with_all__"])
    _plot_shapley(shapley_ci_out, out_results / "fig_shapley_full_features.png", total_r2)
    _plot_partial(unique_df,       out_results / "fig_partial_spearman_full_features.png")

    # ------------------------------------------------------------------ #
    # Markdown summary                                                    #
    # ------------------------------------------------------------------ #
    sorted_shap = shapley_df[~shapley_df["predictor"].str.startswith("__")]\
                  .sort_values("mean", ascending=False)
    sorted_uniq = unique_df.sort_values("rho_partial", ascending=False)

    md_lines = [
        "# Full-Feature Variance Attribution (human ratings + visual + category)\n",
        f"Total R² (all predictors): **{total_r2:.4f}**\n",
        "## Shapley Δρ² (feature ranking)\n",
        "| Rank | Feature | Group | Δρ² |",
        "|------|---------|-------|-----|",
    ]
    for rank, (_, row) in enumerate(sorted_shap.iterrows(), 1):
        ci_row = shapley_ci_out[shapley_ci_out["predictor"] == row["predictor"]]
        ci_str = ""
        if not ci_row.empty:
            lo = float(ci_row["ci95_lo"].iloc[0])
            hi = float(ci_row["ci95_hi"].iloc[0])
            ci_str = f" [{lo:.4f}, {hi:.4f}]"
        md_lines.append(
            f"| {rank} | {_pretty(row['predictor'])} | {_group_label(row['predictor'])} "
            f"| {row['mean']:.4f}{ci_str} |"
        )

    md_lines += [
        "\n## Partial Spearman ρ (unique contributions, controlling for all others)\n",
        "| Rank | Feature | Group | ρ | 95% CI | p |",
        "|------|---------|-------|---|--------|---|",
    ]
    for rank, (_, row) in enumerate(sorted_uniq.iterrows(), 1):
        sig = "" if row["p_perm"] < 0.05 else " ⚠️ n.s."
        md_lines.append(
            f"| {rank} | {_pretty(row['model'])} | {_group_label(row['model'])} "
            f"| {row['rho_partial']:+.3f} | [{row['ci95_lo']:+.3f}, {row['ci95_hi']:+.3f}] "
            f"| {row['p_perm']:.3f}{sig} |"
        )

    md_lines += [
        "\n## Notes",
        "- Human-rated food attributes sourced from existing `pairwise_vectors.csv`.",
        "- Low-level visual RDMs loaded from `derivatives/model_rdms_canon/rdm_ll_*.npz`.",
        "- Food category RDM: `derivatives/model_rdms_canon/rdm_food_classification.npz`.",
        "- Shapley ΔR² sums to ≈ total R² (up to sampling noise in random orderings).",
        "- Partial ρ controls for ALL other predictors simultaneously (conservative).",
        "- `ai_*` rating columns excluded to keep analysis focused on human perception + visual.",
        "- **This run does not overwrite any existing output.**",
    ]

    summary_path = out_results / "stage6_full_feature_attribution_summary.md"
    summary_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] Summary written: {summary_path}")
    print(f"[OK] Results written to: {out_results}\n")

    # Print top 5 to console
    print("=" * 60)
    print("TOP FEATURES BY SHAPLEY Δρ²")
    print("=" * 60)
    for rank, (_, row) in enumerate(sorted_shap.head(10).iterrows(), 1):
        print(f"  {rank:2d}. {_pretty(row['predictor']):35s}  Δρ² = {row['mean']:.4f}")
    print(f"\n  Total R² = {total_r2:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
