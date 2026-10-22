#!/usr/bin/env python3
"""
prelim_build_model_rdms.py

Precompute model RDMs from the food-stimulus info CSVs.

Sources:
- Human ratings (human_*) and AI ratings (blind_ai_*):
  `reference.policy_defaults.DYNAMIC_INFO_CSV` (Foodpictures_information_dynamic.csv)
  — the dynamic CSV is the source of truth for all rating models. It is produced
  by the PAFID pipeline (run_qc.py for aware/QC, rate_images.py for blind).
- Categorical contrasts (natural_vs_transformed, food_classification):
  `reference.policy_defaults.INFO_CSV` (canonical CSV).

CSV header names are used directly as model keys (no local remapping layer).

Outputs are written to the derivatives/model_rdms_canon directory (or as specified)
for downstream analysis.
"""

import os
import argparse
import numpy as np
import pandas as pd
from analysis_helpers import (
    load_info,
    build_model_rdm,
    canonicalize_key,
    resolve_info_col,
)

# Optional: if model selection has been merged into analysis_helpers, use it.
try:
    from analysis_helpers import select_models  # type: ignore
except Exception:
    select_models = None  # type: ignore

# --- Configuration (policy defaults) ---
from pathlib import Path

try:
    # Preferred: central defaults
    from reference.policy_defaults import (
        DERIVATIVES_DIR as _DERIV_DIR,
        REFERENCE_DIR as _REF_DIR,
        INFO_CSV as _INFO_CSV,
        DYNAMIC_INFO_CSV as _DYNAMIC_INFO_CSV,
        EMPIRICAL_RDM_NPZ as _EMPIRICAL_RDM_NPZ,
        DEFAULT_MODEL_POLICY,
    )
except Exception:
    _DERIV_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "derivatives"))
    _REF_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference"))
    _INFO_CSV = _REF_DIR / "Foodpictures_information_dynamic.csv"
    _DYNAMIC_INFO_CSV = _REF_DIR / "Foodpictures_information_dynamic.csv"
    _EMPIRICAL_RDM_NPZ = _DERIV_DIR / "triplet_rdm_by_filename.npz"

HERE = os.path.dirname(os.path.abspath(__file__))
DERIVATIVES_DIR = str(_DERIV_DIR)
MODEL_RDMS_DIR = str(_DERIV_DIR / "model_rdms_canon")
os.makedirs(MODEL_RDMS_DIR, exist_ok=True)

INFO_CSV = str(_INFO_CSV)

# Human ratings are wired through the dynamic reference CSV (source of truth
# for human_* columns). AI (blind) ratings remain in the canonical CSV.
DYNAMIC_INFO_CSV = str(_DYNAMIC_INFO_CSV)


def load_dynamic_info(csv_path: str) -> pd.DataFrame:
    """Load the dynamic reference CSV.

    Handles the UTF-8 BOM in the header, normalizes filename to basename, and
    drops duplicate filename rows (keep first) so reindex-by-filename is safe.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing dynamic info CSV at {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    df["filename"] = df["filename"].astype(str).apply(os.path.basename)
    n_raw = len(df)
    df = df.drop_duplicates(subset="filename", keep="first")
    if len(df) < n_raw:
        print(f"[NOTE] Dropped {n_raw - len(df)} duplicate filename row(s) from dynamic CSV.")

    # Generic ai_* mapping (mirrors analysis_helpers.load_info): map the
    # requested AI-rating prefix to generic ai_* columns so model keys from
    # select_models() resolve. Pipeline default is blind (run_full_pipeline.sh
    # exports TRIPLETS_AI_MODE=blind).
    ai_mode = os.environ.get("TRIPLETS_AI_MODE", "aware")
    prefix = "aware_ai_" if ai_mode == "aware" else "blind_ai_"
    for trait in ["calorie_density", "healthiness", "sweetness", "saltiness", "sourness",
                  "bitterness", "savoriness", "fattiness", "spiciness"]:
        src = f"{prefix}{trait}"
        if src in df.columns:
            df[f"ai_{trait}"] = df[src]
    return df

# Ensure consistent label ordering using the empirical triplet RDM labels if available
TRIPLET_RDM_NPZ = str(_EMPIRICAL_RDM_NPZ)

# Rating models: use canonical CSV header names directly.
# Uses blind AI ratings (model does not see food name) — set TRIPLETS_AI_MODE=aware to override.
AI_RATING_KEYS = [
    "blind_ai_sweetness",
    "blind_ai_saltiness",
    "blind_ai_sourness",
    "blind_ai_bitterness",
    "blind_ai_savoriness",
    "blind_ai_fattiness",
    "blind_ai_spiciness",
    "blind_ai_healthiness",
    "blind_ai_calorie_density",
]

HUMAN_RATING_KEYS = [
    "human_sweetness",
    "human_saltiness",
    "human_sourness",
    "human_bitterness",
    "human_savoriness",
    "human_fattiness",
    "human_spiciness",
    "human_healthiness",
    "human_calorie_density",
    "human_appeal",
]


def save_model_rdm_npz_local(
    rdm: np.ndarray,
    labels,
    key: str,
    source_column: str,
    kind: str,
    outdir: str,
):
    """Save a single canonical model RDM NPZ.

    This deliberately avoids any legacy alias writing. The filename is always
    `rdm_<key>.npz`, where `key` is the canonical CSV-header-based model name.
    """
    os.makedirs(outdir, exist_ok=True)
    canon = canonicalize_key(key)
    out_path = os.path.join(outdir, f"rdm_{canon}.npz")
    np.savez_compressed(
        out_path,
        data=np.asarray(rdm, dtype=np.float32),
        labels=np.asarray(labels, dtype=object),
        key=str(canon),
        source_column=str(source_column),
        kind=str(kind),
    )
    return [out_path]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute model RDMs from the stimuli info CSV")
    p.add_argument("--source", choices=["human", "ai", "mixed"], default=(getattr(DEFAULT_MODEL_POLICY, "source", "mixed") if "DEFAULT_MODEL_POLICY" in globals() else "mixed"),
                   help="Which rating sources to build (default: mixed).")
    p.add_argument("--include-contrasts", action=argparse.BooleanOptionalAction, default=True,
                   help="Also build supplementary conceptual contrast RDMs (Natural vs transformed, Sweet vs savory) if present.")
    p.add_argument("--include-food-classification", action="store_true",
                   help="Also build Food classification categorical RDM if present.")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory for RDMs (default: analysis_code/derivatives/model_rdms_canon)")
    return p.parse_args(argv)


def build_continuous_rdm(labels, info_df, column):
    """
    Build an n x n RDM from a continuous column using absolute differences.
    Returns an ndarray with NaN on the diagonal.
    """
    if column not in info_df.columns:
        raise KeyError(f"Column not in info: {column}")
    
    # Extract the column values aligned with the provided labels
    s = pd.to_numeric(info_df.set_index("filename").reindex(labels)[column], errors="coerce")
    v = s.to_numpy(dtype=float)
    
    # Compute absolute pairwise differences
    D = np.abs(v[:, None] - v[None, :])
    np.fill_diagonal(D, np.nan)
    return D


def main():
    args = parse_args()
    global MODEL_RDMS_DIR
    if args.outdir is not None:
        MODEL_RDMS_DIR = os.path.abspath(os.path.expanduser(args.outdir))
        os.makedirs(MODEL_RDMS_DIR, exist_ok=True)

    print(f"Loading info from: {INFO_CSV}")
    info_df = load_info(INFO_CSV)

    print(f"Loading human ratings from: {DYNAMIC_INFO_CSV}")
    dynamic_df = load_dynamic_info(DYNAMIC_INFO_CSV)

    # Determine the canonical label ordering
    if os.path.exists(TRIPLET_RDM_NPZ):
        print(f"Loading canonical labels from: {TRIPLET_RDM_NPZ}")
        z = np.load(TRIPLET_RDM_NPZ, allow_pickle=True)
        labels = [os.path.basename(str(x)) for x in list(z["labels"])]
    else:
        print("Triplet RDM not found. Using labels directly from the info CSV.")
        labels = list(info_df["filename"].astype(str).apply(os.path.basename))

    print(f"Total labels (stimuli): {len(labels)}")

    # 1. Build Categorical Models (policy-driven)
    categorical_to_build = []
    if args.include_contrasts:
        categorical_to_build += ["natural_vs_transformed"]  # sweet_vs_savory removed (redundant with continuous dims)
    if args.include_food_classification:
        categorical_to_build += ["food_classification"]

    if categorical_to_build:
        print("\n--- Building Categorical RDMs ---")
        for col in categorical_to_build:
            if col in info_df.columns:
                try:
                    D = build_model_rdm(labels, info_df, col)
                    canon = canonicalize_key(col)
                    saved = save_model_rdm_npz_local(
                        rdm=D,
                        labels=labels,
                        key=canon,
                        source_column=str(col),
                        kind="categorical",
                        outdir=MODEL_RDMS_DIR,
                    )
                    print(f"[OK] Saved: {saved[0]}  (col={col} -> key={canon})")
                except Exception as e:
                    print(f"[ERROR] Failed to build '{col}': {e}")
            else:
                print(f"[SKIP] Column not found: {col}")
    else:
        print("\n--- Skipping Categorical RDMs (no contrasts or food classification requested) ---")

    # 2. Build AI/Human Rating Models (Continuous, policy-driven)
    source = args.source

    # If model-selection policy is available, prefer it so model keys stay aligned
    # with the canonical CSV header names.
    if select_models is not None:
        try:
            specs = select_models(
                source_mode=source,
                include_visual=False,  # this script focuses on ratings; visual models are built elsewhere
                include_controls=False,
                include_supplementary_contrasts=False,
            )
            # Keep only appraisal/flavour sources (human/ai) — ignore conceptual/controls
            keys = [s.key for s in specs if getattr(s, "source", "") in ("human", "ai")]
            # Split by source
            ai_keys = [k for k in keys if str(k).startswith("ai")]
            human_keys = [k for k in keys if str(k).startswith("human")]
        except Exception:
            ai_keys = AI_RATING_KEYS if source in ("ai", "mixed") else []
            human_keys = HUMAN_RATING_KEYS if source in ("human", "mixed") else []
    else:
        ai_keys = AI_RATING_KEYS if source in ("ai", "mixed") else []
        human_keys = HUMAN_RATING_KEYS if source in ("human", "mixed") else []

    if ai_keys:
        print("\n--- Building AI Rating RDMs (from dynamic CSV) ---")
        for key in ai_keys:
            col = resolve_info_col(dynamic_df, key)
            if col is None:
                print(f"[SKIP] Column not found in dynamic CSV for key={key}")
                continue
            try:
                D = build_continuous_rdm(labels, dynamic_df, col)
                canon = canonicalize_key(key)
                saved = save_model_rdm_npz_local(
                    rdm=D,
                    labels=labels,
                    key=canon,
                    source_column=str(col),
                    kind="continuous",
                    outdir=MODEL_RDMS_DIR,
                )
                print(f"[OK] Saved: {saved[0]}  (col={col} -> key={canon})")
            except Exception as e:
                print(f"[ERROR] Failed to build key={key} (col={col}): {e}")

    if human_keys:
        print("\n--- Building Human Rating RDMs (from dynamic CSV) ---")
        for key in human_keys:
            col = resolve_info_col(dynamic_df, key)
            if col is None:
                print(f"[SKIP] Column not found in dynamic CSV for key={key}")
                continue
            try:
                D = build_continuous_rdm(labels, dynamic_df, col)
                canon = canonicalize_key(key)
                saved = save_model_rdm_npz_local(
                    rdm=D,
                    labels=labels,
                    key=canon,
                    source_column=str(col),
                    kind="continuous",
                    outdir=MODEL_RDMS_DIR,
                )
                print(f"[OK] Saved: {saved[0]}  (col={col} -> key={canon})")
            except Exception as e:
                print(f"[ERROR] Failed to build key={key} (col={col}): {e}")

    print(f"\nFinished building all RDMs in: {MODEL_RDMS_DIR}")

if __name__ == "__main__":
    main()
