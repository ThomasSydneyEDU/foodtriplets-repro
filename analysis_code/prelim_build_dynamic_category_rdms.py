#!/usr/bin/env python3
"""
prelim_build_dynamic_category_rdms.py

Build additional category model RDMs from Foodpictures_information_dynamic.csv
and save them to derivatives/model_rdms_canon/ so that compute_model_evaluation.py
picks them up automatically alongside the main model set.

This script is supplementary to prelim_build_model_rdms.py and does NOT modify
the canonical pipeline. It reads from the dynamic reference CSV only.

Models built
------------
Categorical (0 = same group, 1 = different group):
    Category_WHO_10      — 10-group WHO food classification
    Category_Intuitive_7 — 7-group intuitive classification
    Category_Culinary_9  — 9-group culinary classification
    Category_NOVA_4      — 4-group NOVA processing classification

Continuous (absolute pairwise difference):
    Transformation_score — continuous 0-100 transformation/processing score
    NOVA_ordinal         — NOVA group treated as an ordered scale (|NOVA_i - NOVA_j|)

Usage
-----
    python analysis_code/prelim_build_dynamic_category_rdms.py
    python analysis_code/prelim_build_dynamic_category_rdms.py --outdir path/to/custom/dir
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# --- Policy defaults ---
try:
    from reference.policy_defaults import (
        DERIVATIVES_DIR as _DERIV_DIR,
        REFERENCE_DIR as _REF_DIR,
        DYNAMIC_INFO_CSV as _DYNAMIC_INFO_CSV,
        EMPIRICAL_RDM_NPZ as _EMPIRICAL_RDM_NPZ,
    )
except Exception:
    _HERE = Path(os.path.dirname(os.path.abspath(__file__)))
    _DERIV_DIR = _HERE / "derivatives"
    _REF_DIR = _HERE / "reference"
    _DYNAMIC_INFO_CSV = _REF_DIR / "Foodpictures_information_dynamic.csv"
    _EMPIRICAL_RDM_NPZ = _DERIV_DIR / "triplet_rdm_by_filename.npz"

HERE = os.path.dirname(os.path.abspath(__file__))
DYNAMIC_CSV = str(_DYNAMIC_INFO_CSV)
TRIPLET_RDM_NPZ = str(_EMPIRICAL_RDM_NPZ)
DEFAULT_OUTDIR = str(_DERIV_DIR / "model_rdms_canon")


# ---------------------------------------------------------------------------
# RDM builders
# ---------------------------------------------------------------------------

def build_categorical_rdm(labels, info_df, column):
    """0/1 RDM: 0 = same category, 1 = different; NaN if missing."""
    n = len(labels)
    D = np.zeros((n, n), dtype=float)
    cat_map = pd.Series(info_df[column].values,
                        index=info_df["filename"].astype(str)).to_dict()
    cats = np.array([cat_map.get(lab, np.nan) for lab in labels], dtype=object)
    for i in range(n):
        ci = cats[i]
        for j in range(i + 1, n):
            cj = cats[j]
            if pd.isna(ci) or pd.isna(cj):
                val = np.nan
            else:
                val = 0.0 if ci == cj else 1.0
            D[i, j] = D[j, i] = val
    np.fill_diagonal(D, 0.0)
    return D


def build_continuous_rdm(labels, info_df, column):
    """Absolute pairwise difference RDM for a continuous column."""
    s = pd.to_numeric(
        info_df.set_index("filename").reindex(labels)[column],
        errors="coerce",
    )
    v = s.to_numpy(dtype=float)
    D = np.abs(v[:, None] - v[None, :])
    np.fill_diagonal(D, np.nan)
    return D


# ---------------------------------------------------------------------------
# Save helper (matches format expected by compute_model_evaluation.py)
# ---------------------------------------------------------------------------

def save_rdm(rdm, labels, key, source_column, kind, outdir):
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"rdm_{key}.npz")
    np.savez_compressed(
        out_path,
        data=np.asarray(rdm, dtype=np.float32),
        labels=np.asarray(labels, dtype=object),
        key=str(key),
        source_column=str(source_column),
        kind=str(kind),
    )
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: derivatives/model_rdms_canon)")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = os.path.abspath(args.outdir) if args.outdir else DEFAULT_OUTDIR

    # --- Load canonical label ordering from empirical RDM ---
    if not os.path.exists(TRIPLET_RDM_NPZ):
        raise FileNotFoundError(
            f"Empirical RDM not found: {TRIPLET_RDM_NPZ}\n"
            "Run prelim_preprocess_data.py first."
        )
    z = np.load(TRIPLET_RDM_NPZ, allow_pickle=True)
    labels = [os.path.basename(str(x)) for x in list(z["labels"])]
    print(f"Loaded {len(labels)} canonical labels from empirical RDM.")

    # --- Load dynamic CSV ---
    if not os.path.exists(DYNAMIC_CSV):
        raise FileNotFoundError(f"Dynamic CSV not found: {DYNAMIC_CSV}")
    info = pd.read_csv(DYNAMIC_CSV, encoding="utf-8-sig")
    info["filename"] = info["filename"].astype(str).apply(os.path.basename)
    n_raw = len(info)
    info = info.drop_duplicates(subset="filename", keep="first")
    if len(info) < n_raw:
        print(f"[NOTE] Dropped {n_raw - len(info)} duplicate filename row(s) from CSV.")
    print(f"Loaded dynamic CSV: {len(info)} rows, columns: {list(info.columns)}")

    # --- Define models to build ---
    categorical_cols = [
        "Category_WHO_10",
        "Category_Intuitive_7",
        "Category_Culinary_9",
        "Category_NOVA_4",
    ]
    continuous_cols = [
        "Transformation_score",
    ]

    # NOVA_4 also as ordinal-continuous (|NOVA_i - NOVA_j|)
    nova_ordinal_col = "Category_NOVA_4"

    built = []

    # --- Categorical RDMs ---
    print("\n--- Building categorical RDMs ---")
    for col in categorical_cols:
        if col not in info.columns:
            print(f"[SKIP] Column not found: {col}")
            continue
        try:
            D = build_categorical_rdm(labels, info, col)
            path = save_rdm(D, labels, key=col, source_column=col,
                            kind="categorical", outdir=outdir)
            n_missing = np.isnan(D).sum() // 2
            print(f"[OK]   {col} -> {path}  (missing pairs: {n_missing})")
            built.append(col)
        except Exception as e:
            print(f"[ERROR] {col}: {e}")

    # --- Continuous RDMs ---
    print("\n--- Building continuous RDMs ---")
    for col in continuous_cols:
        if col not in info.columns:
            print(f"[SKIP] Column not found: {col}")
            continue
        try:
            D = build_continuous_rdm(labels, info, col)
            path = save_rdm(D, labels, key=col, source_column=col,
                            kind="continuous", outdir=outdir)
            n_missing = int(np.isnan(D[np.triu_indices(len(labels), k=1)]).sum())
            print(f"[OK]   {col} -> {path}  (missing pairs: {n_missing})")
            built.append(col)
        except Exception as e:
            print(f"[ERROR] {col}: {e}")

    # --- NOVA as ordinal-continuous ---
    print("\n--- Building NOVA ordinal-continuous RDM ---")
    if nova_ordinal_col in info.columns:
        try:
            nova_info = info.copy()
            nova_info[nova_ordinal_col] = pd.to_numeric(
                nova_info[nova_ordinal_col], errors="coerce"
            )
            D = build_continuous_rdm(labels, nova_info, nova_ordinal_col)
            key = "NOVA_ordinal"
            path = save_rdm(D, labels, key=key, source_column=nova_ordinal_col,
                            kind="continuous", outdir=outdir)
            n_missing = int(np.isnan(D[np.triu_indices(len(labels), k=1)]).sum())
            print(f"[OK]   {key} -> {path}  (missing pairs: {n_missing})")
            built.append(key)
        except Exception as e:
            print(f"[ERROR] NOVA ordinal: {e}")

    print(f"\nDone. Built {len(built)} RDMs in: {outdir}")
    print("These will be picked up automatically by compute_model_evaluation.py.")


if __name__ == "__main__":
    main()
