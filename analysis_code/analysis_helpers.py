#!/usr/bin/env python3
"""
analysis_helpers.py

A library of helper functions for the foodtriplets analysis project.
This module consolidates common functions for data loading, processing,
RSA, and model output handling from various scripts in the project.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import random
from typing import Callable, Optional, Tuple, List, Dict, Iterable, Literal
import ast
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
try:
    from scipy.spatial.distance import squareform as _squareform
except ImportError:
    _squareform = None  # type: ignore[assignment]


def squareform(M, checks=True):
    if _squareform is None:
        raise ImportError("scipy is required for squareform. Install with: pip install scipy")
    return _squareform(M, checks=checks)


# -----------------------------------------------------------------------------
# Path Management
# -----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(HERE, 'processed')
DERIVATIVES_DIR = os.path.join(HERE, 'derivatives')
#
# Define the path to the stimulus information sheet
_STIM_INFO_CANDIDATES = [
    os.path.join(HERE, 'reference', 'Foodpictures_information_dynamic.csv'),
]
STIMULI_INFO = None
for _p in _STIM_INFO_CANDIDATES:
    _abs = os.path.abspath(_p)
    if os.path.exists(_abs):
        STIMULI_INFO = _abs
        break
if STIMULI_INFO is None:
    STIMULI_INFO = os.path.abspath(_STIM_INFO_CANDIDATES[0])
TRIPLET_RDM_NPZ = os.path.join(PROCESSED_DIR, 'triplet_rdm_by_filename.npz')

def basename_only(path_str):
    if pd.isna(path_str):
        return path_str
    return os.path.basename(str(path_str))


def resolve_path(path: str) -> str:
    """Resolve a path relative to the repository root."""
    if os.path.exists(path):
        return path
    repo_root = os.path.dirname(HERE)
    candidate = os.path.join(repo_root, path)
    if os.path.exists(candidate):
        return candidate
    return path

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Basic RSA & RDM Utilities
# -----------------------------------------------------------------------------


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rho (rank→Pearson), assumes no NaNs."""
    rx = pd.Series(x).rank(method='average').to_numpy()
    ry = pd.Series(y).rank(method='average').to_numpy()
    x0 = (rx - rx.mean()) / (rx.std(ddof=0) + 1e-12)
    y0 = (ry - ry.mean()) / (ry.std(ddof=0) + 1e-12)
    return float(np.clip(np.mean(x0 * y0), -1.0, 1.0))

def vectorize_upper(M: np.ndarray) -> np.ndarray:
    iu, ju = np.triu_indices_from(M, k=1)
    return M[iu, ju]

def mantel_spearman(emp_M: np.ndarray,
                    mod_M: np.ndarray,
                    n_perm: int = 5000,
                    seed: Optional[int] = 1234) -> dict:
    """Mantel-style permutation test for Spearman RSA."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    v_emp = vectorize_upper(emp_M)
    v_mod = vectorize_upper(mod_M)

    mask = ~np.isnan(v_emp) & ~np.isnan(v_mod)
    if not np.any(mask):
        return {"rho": np.nan, "p_perm": np.nan, "n_perm": int(n_perm)}

    rho_obs = _spearman(v_emp[mask], v_mod[mask])

    n = emp_M.shape[0]
    idx = np.arange(n)
    ge = 0
    for _ in range(n_perm):
        perm = np.random.permutation(idx)
        mod_p = mod_M[perm][:, perm]
        v_modp = vectorize_upper(mod_p)
        rho_p = _spearman(v_emp[mask], v_modp[mask])
        if abs(rho_p) >= abs(rho_obs):
            ge += 1
    p_perm = (ge + 1) / (n_perm + 1)
    return {"rho": float(rho_obs), "p_perm": float(p_perm), "n_perm": int(n_perm)}

def bootstrap_ci_spearman(emp_M: np.ndarray,
                           mod_M: np.ndarray,
                           n_boot: int = 2000,
                           seed: Optional[int] = 5678,
                           ci: float = 0.95) -> Tuple[float, float]:
    """Bootstrap CI for Spearman RSA by resampling observed pairs."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    v_emp = vectorize_upper(emp_M)
    v_mod = vectorize_upper(mod_M)
    mask = ~np.isnan(v_emp) & ~np.isnan(v_mod)
    if not np.any(mask):
        return (np.nan, np.nan)
    x = v_emp[mask]
    y = v_mod[mask]
    m = len(x)
    idx = np.arange(m)
    rhos = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        samp = np.random.choice(idx, size=m, replace=True)
        rhos[b] = _spearman(x[samp], y[samp])
    alpha = (1.0 - ci) / 2.0
    return (float(np.quantile(rhos, alpha)), float(np.quantile(rhos, 1.0 - alpha)))

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_triplet_rdm(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)

    # --- Robust key support across pipeline versions ---
    # Common keys used: 'data' (old), 'rdm' (models), 'D' (some scripts), 'matrix' (misc)
    if "data" in z:
        D = z["data"]
    elif "rdm" in z:
        D = z["rdm"]
    elif "D" in z:
        D = z["D"]
    elif "matrix" in z:
        D = z["matrix"]
    else:
        raise KeyError(
            f"Empirical RDM archive {npz_path} does not contain a known matrix key. "
            f"Found keys: {list(z.keys())}. Expected one of: data, rdm, D, matrix."
        )

    D = np.asarray(D, dtype=float)

    # Labels are optional in some saved artifacts
    if "labels" in z:
        labels = z["labels"]
        labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    else:
        labels = None

    return D, labels

def normalize_info_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy column names (e.g., _0_100, human_mean_) to canonical names."""
    df = df.copy()

    # Alias Category_Intuitive_7 to food_classification if missing
    if 'food_classification' not in df.columns and 'Category_Intuitive_7' in df.columns:
        df['food_classification'] = df['Category_Intuitive_7']

    # Canonicalize filename column
    if 'filename' not in df.columns:
        if 'image_file' in df.columns:
            df = df.rename(columns={'image_file': 'filename'})
        else:
            raise KeyError("Info CSV must contain 'filename' or 'image_file' column")

    # Canonicalize category column
    if 'category' not in df.columns:
        for c in ['Category', 'food_category', 'FoodCategory', 'food_classification']:
            if c in df.columns:
                df['category'] = df[c]
                break

    # --- Attribute Mapping Fallbacks ---
    # The canonical info CSV uses human_ prefix instead of _0_100 suffixes.
    # We map them here so that scripts expecting the old names still work.
    attr_map = {
        "healthiness_0_100": "human_healthiness",
        "savoriness_0_100": "human_savoriness",
        "calorie_density_0_100": "human_calorie_density",
        "sweetness_0_100": "human_sweetness",
        "saltiness_0_100": "human_saltiness",
        "sourness_0_100": "human_sourness",
        "bitterness_0_100": "human_bitterness",
        "fatty_flavour_0_100": "human_fattiness",
        "spiciness_0_100": "human_spiciness",
        # Some scripts also expect human_mean_ prefixes from an intermediate version
        "human_mean_healthiness": "human_healthiness",
        "human_mean_umami": "human_savoriness",
        "human_mean_caloriedensity": "human_calorie_density",
        "human_mean_sweet": "human_sweetness",
        "human_mean_salty": "human_saltiness",
        "human_mean_sour": "human_sourness",
        "human_mean_bitter": "human_bitterness",
        "human_mean_fatty": "human_fattiness",
        "human_mean_spicy": "human_spiciness",
        "human_mean_appeal": "human_appeal",
    }
    # Also map AI columns if missing
    for trait in ["healthiness", "savoriness", "calorie_density", "sweetness", "saltiness", "sourness", "bitterness", "fattiness", "spiciness"]:
        old_ai = f"{trait}_0_100"
        if trait == "fattiness": old_ai = "fatty_flavour_0_100"
        new_ai = f"ai_{trait}"
        attr_map[old_ai] = new_ai

    for old, new in attr_map.items():
        if old not in df.columns and new in df.columns:
            df[old] = df[new]

    # Normalize filenames to basenames (match triplet labels)
    df['filename'] = df['filename'].astype(str).apply(lambda p: os.path.basename(p)).str.strip()
    df['stem'] = df['filename'].apply(lambda p: os.path.splitext(p)[0])

    return df


# Columns whose source of truth is the dynamic reference CSV
# (Foodpictures_information_dynamic.csv, maintained by the PAFID pipeline:
# run_qc.py writes aware ratings + categories + human merge, rate_images.py
# writes blind ratings). load_info() overrides these columns from the dynamic
# CSV whenever the canonical info sheet is loaded, so all consumers (prelim +
# compute scripts) see the dynamic values without per-script wiring.
DYNAMIC_OVERRIDE_PREFIXES = ("human_", "blind_")
# Exact-name model columns also owned by the dynamic CSV (categorical models).
DYNAMIC_OVERRIDE_COLUMNS = (
    "natural_vs_transformed",
    "sweet_vs_savory",
)
# food_classification is legacy and fully replaced by Category_Intuitive_7
# (7-way intuitive classification; splits Protein into Animal/Plant protein).
# PAFID no longer exports food_classification. To avoid renaming the model key
# across all downstream scripts, load_info() aliases the dynamic CSV's
# Category_Intuitive_7 values INTO the food_classification column — so every
# script that references "food_classification" transparently uses the new
# 7-way classification.
FOOD_CLASSIFICATION_SOURCE = "Category_Intuitive_7"


def _apply_dynamic_rating_overrides(df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    """Override rating/category model columns from the dynamic reference CSV (by filename)."""
    try:
        from reference.policy_defaults import DYNAMIC_INFO_CSV as _DYN_CSV
        dyn_path = str(_DYN_CSV)
    except Exception:
        dyn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "reference", "Foodpictures_information_dynamic.csv")

    if not os.path.exists(dyn_path):
        return df
    # Don't override when the dynamic CSV itself is being loaded.
    if os.path.abspath(str(csv_path)) == os.path.abspath(dyn_path):
        return df
    if "filename" not in df.columns:
        return df

    dyn = pd.read_csv(dyn_path, encoding="utf-8-sig")
    dyn.columns = [str(c).strip() for c in dyn.columns]
    if "filename" not in dyn.columns:
        return df
    dyn["__key"] = dyn["filename"].astype(str).map(os.path.basename).str.strip()
    dyn = dyn.drop_duplicates(subset="__key", keep="first").set_index("__key")

    cols = [c for c in dyn.columns
            if c.startswith(DYNAMIC_OVERRIDE_PREFIXES) or c in DYNAMIC_OVERRIDE_COLUMNS]
    if not cols:
        return df

    key = df["filename"].astype(str).map(os.path.basename).str.strip()
    for c in cols:
        df[c] = key.map(dyn[c]).values

    # Alias: food_classification := Category_Intuitive_7 (see note above).
    if FOOD_CLASSIFICATION_SOURCE in dyn.columns:
        df["food_classification"] = key.map(dyn[FOOD_CLASSIFICATION_SOURCE]).values
        print(f"[INFO] load_info: food_classification aliased to {FOOD_CLASSIFICATION_SOURCE} (dynamic CSV)")

    print(f"[INFO] load_info: {len(cols)} rating/category column(s) overridden from {os.path.basename(dyn_path)}")
    return df


def load_info(csv_path: str = STIMULI_INFO) -> pd.DataFrame:
    """Load the stimulus information CSV.

    Supports both legacy schemas (column `filename`) and the current stimulus sheet
    schema (column `image_file`). Always returns a dataframe containing:
      - filename: basename of image file
      - stem: filename without extension

    Rating columns (human_*, blind_*) are overridden from the dynamic reference
    CSV (see DYNAMIC_OVERRIDE_PREFIXES) so the dynamic CSV is the single source
    of truth for ratings.

    Also normalizes common underscore-vs-space variants for factor columns used
    by model RDM builders.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing info CSV at {csv_path}")

    df = pd.read_csv(csv_path)

    # Strip header whitespace
    df.columns = [str(c).strip() for c in df.columns]

    # Override rating columns from the dynamic reference CSV (single source of
    # truth). Must run BEFORE the AI-mode mapping below so generic ai_* columns
    # inherit the dynamic values.
    df = _apply_dynamic_rating_overrides(df, csv_path)

    # Dynamic AI Mode Mapping
    ai_mode = os.environ.get("TRIPLETS_AI_MODE", "aware")
    prefix = "aware_ai_" if ai_mode == "aware" else "blind_ai_"
    
    # Map the requested prefix back to the generic 'ai_' columns for analysis
    for trait in ["calorie_density", "healthiness", "sweetness", "saltiness", "sourness", "bitterness", "savoriness", "fattiness", "spiciness"]:
        source_col = f"{prefix}{trait}"
        generic_col = f"ai_{trait}"
        if source_col in df.columns:
            df[generic_col] = df[source_col]
            
    # Also map observed_food if requested
    obs_prefix = "aware_observed_" if ai_mode == "aware" else "blind_observed_"
    if f"{obs_prefix}food" in df.columns:
        df["observed_food"] = df[f"{obs_prefix}food"]


    df = normalize_info_columns(df)

    return df

def load_summary(path: str) -> pd.DataFrame:
    """Loads and validates the summary.csv file."""
    df = pd.read_csv(path)
    # Harmonize column names
    colmap = {c: c.strip() for c in df.columns}
    df.rename(columns=colmap, inplace=True)
    
    has_list_format = "images_onscreen" in df.columns
    has_split_cols = {"StimA_file", "StimB_file", "StimC_file", "choice_file"}.issubset(df.columns)
    if not (has_list_format or has_split_cols):
        raise ValueError(
            "summary.csv must have either 'images_onscreen' + 'choice_image', "
            "or the split columns 'StimA_file','StimB_file','StimC_file','choice_file'."
        )
    if has_list_format and ("choice_image" not in df.columns) and not {"p_ab", "p_ac", "p_bc"}.issubset(df.columns):
        raise ValueError("When using 'images_onscreen', provide 'choice_image' or probability columns p_ab,p_ac,p_bc.")
    return df


def load_summary_csv(path: str) -> pd.DataFrame:
    """Backward-compatible alias used by older scripts.

    Some analysis scripts historically imported `load_summary_csv` from other modules.
    Keep this thin wrapper here so all scripts can share the same loader.
    """
    return load_summary(path)

def load_labels_lookup(labels_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    """Loads the labels_lookup.csv file."""
    if labels_csv is None or not labels_csv.exists():
        return None
    df = pd.read_csv(labels_csv)
    if not {"filename", "label"}.issubset(set(df.columns)):
        return None
    df = df.copy()
    df["filename"] = df["filename"].astype(str).map(os.path.basename)
    return df

# -----------------------------------------------------------------------------
# Triplet Processing
# -----------------------------------------------------------------------------

def parse_images_list(s: str | list) -> list[str]:
    """
    Parse images_onscreen column which is stored like "['a.jpg','b.jpg','c.jpg']".
    Falls back to a best-effort comma split if ast.literal_eval fails.
    """
    if isinstance(s, list):
        return [str(x) for x in s]
    if not isinstance(s, str):
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return [str(x) for x in v]
    except Exception:
        pass
    # Fallback heuristic
    s2 = s.strip().strip('[]')
    parts = [p.strip().strip("'").strip('"') for p in s2.split(",")]
    return [p for p in parts if p][:3]


def build_filename_index(summary_df: pd.DataFrame, labels_lookup_path: str = None) -> Tuple[List[str], Dict[str, int]]:
    """Builds an index of filenames from summary or labels lookup."""
    filenames = []
    if labels_lookup_path and os.path.exists(labels_lookup_path):
        try:
            lookup = pd.read_csv(labels_lookup_path)
            if "filename" in lookup.columns:
                filenames = lookup["filename"].astype(str).tolist()
        except Exception:
            filenames = []
    if not filenames:
        seen = set()
        if "images_onscreen" in summary_df.columns:
            for imgs_raw in summary_df["images_onscreen"]:
                imgs = parse_images_list(imgs_raw)
                for x in imgs:
                    seen.add(basename_only(x))
        else:
            for col in ("StimA_file", "StimB_file", "StimC_file"):
                if col in summary_df.columns:
                    seen.update(basename_only(x) for x in summary_df[col].astype(str).tolist())
        filenames = sorted(seen)
    idx_of = {fn: i for i, fn in enumerate(filenames)}
    return filenames, idx_of

# -----------------------------------------------------------------------------
# Model Building & Output
# -----------------------------------------------------------------------------

def build_model_rdm(labels: List[str], info_df: pd.DataFrame, column: str) -> np.ndarray:
    """Return an n×n 0/1 RDM: 0 if same category, 1 if different; NaN if missing."""
    n = len(labels)
    D = np.zeros((n, n), dtype=float)
    cat_map = pd.Series(info_df[column].values, index=info_df['filename'].astype(str)).to_dict()
    cats = np.array([cat_map.get(lab, np.nan) for lab in labels], dtype=object)
    for i in range(n):
        ci = cats[i]
        for j in range(i+1, n):
            cj = cats[j]
            if pd.isna(ci) or pd.isna(cj):
                val = np.nan
            else:
                val = 0.0 if ci == cj else 1.0
            D[i, j] = D[j, i] = val
    np.fill_diagonal(D, 0.0)
    return D

def ensure_symmetric_zero_diag(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M)
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 0.0)
    return M.astype(np.float32)

def save_dsm(df: pd.DataFrame, prefix: str, enable_parquet: bool = False) -> list[str]:
    """Saves a DSM in multiple formats."""
    saved_paths = []
    assert (df.index == df.columns).all(), "Index/columns must be the same labels and order."
    labels = df.index.to_numpy()
    M = ensure_symmetric_zero_diag(df.to_numpy())

    df_clean = pd.DataFrame(M, index=labels, columns=labels)

    csv_path = f"{prefix}.csv.gz"
    df_clean.to_csv(csv_path, index=True)
    saved_paths.append(csv_path)

    condensed = squareform(M, checks=False).astype(np.float32)
    npz_path = f"{prefix}.npz"
    np.savez_compressed(npz_path, condensed=condensed, labels=labels)
    saved_paths.append(npz_path)

    return saved_paths

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_heatmaps(triplet_rdm: np.ndarray,
                  D_nat: Optional[np.ndarray] = None,
                  labels: Optional[List[str]] = None,
                  figsize: Tuple[int, int] = (15, 5)):
    """Return a matplotlib figure with heatmaps for interactive use."""
    import matplotlib.pyplot as plt

    n_panels = 1 + int(D_nat is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    ims = []
    ims.append(axes[0].imshow(triplet_rdm, cmap='viridis', interpolation='nearest'))
    axes[0].set_title('Triplet RDM')

    k = 1
    if D_nat is not None:
        ims.append(axes[k].imshow(D_nat, cmap='viridis', interpolation='nearest'))
        axes[k].set_title('Model: Natural vs Transformed')
        k += 1

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(ims[0], ax=axes, orientation='vertical', fraction=0.02)
    fig.tight_layout()
    return fig, axes


# -----------------------------------------------------------------------------
# Label utilities for figures
# -----------------------------------------------------------------------------

MODEL_LABEL_MAP = {
    "B": "Blue intensity (B)",
    "P": "Pleasure (P)",
    "E": "Excitement (E)",
    "AN": "Angularity (AN)",
    "S": "Saturation (S)",
}

def expand_model_label(name: str) -> str:
    """Return a human-readable label for figure axes/titles."""
    return MODEL_LABEL_MAP.get(name, name)

def apply_label_map(names: List[str]) -> List[str]:
    """Vectorized helper to map a sequence of model codes to pretty labels."""
    return [expand_model_label(n) for n in names]


# -----------------------------------------------------------------------------
# Split-half noise ceiling (wrapper)
# -----------------------------------------------------------------------------

def split_half_noise_ceiling(labels: List[str],
                              summary_csv: Optional[str] = None,
                              n_iter: int = 100,
                              seed: Optional[int] = 1234,
                              builder: Optional[Callable[[List[str], pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]] = None
                              ) -> float:
    """Estimate split-half reliability of the behavioural RDM."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if summary_csv is None:
        summary_csv = os.path.join(PROCESSED_DIR, 'summary.csv')

    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Missing summary.csv at {summary_csv}")

    all_df = pd.read_csv(summary_csv)

    if builder is None:
        try:
            from analysis_code.triplet_prelim import build_pairwise_similarity_from_summary as _default_builder
            builder = _default_builder
        except Exception as e:
            raise ImportError(
                "No builder provided and could not import 'build_pairwise_similarity_from_summary' "
                "from TripletAnalysisPrelim.py. Pass a builder= callable."
            ) from e

    rhos = []
    for _ in range(n_iter):
        perm = np.random.permutation(len(all_df))
        mid = len(all_df) // 2
        df1 = all_df.iloc[perm[:mid]]
        df2 = all_df.iloc[perm[mid:]]
        _, rdm1 = builder(labels, df1)
        _, rdm2 = builder(labels, df2)
        
        v1 = vectorize_upper(rdm1.to_numpy() if isinstance(rdm1, pd.DataFrame) else rdm1)
        v2 = vectorize_upper(rdm2.to_numpy() if isinstance(rdm2, pd.DataFrame) else rdm2)
        
        mask = ~np.isnan(v1) & ~np.isnan(v2)
        if np.any(mask):
            rhos.append(_spearman(v1[mask], v2[mask]))
            
    if not rhos:
        return float('nan')
    mean_rho = float(np.mean(rhos))
    return mean_rho



# -----------------------------------------------------------------------------
# Model selection policy (canonical keys + aliases)
# -----------------------------------------------------------------------------

# This section folds the former standalone `model_selection.py` into analysis_helpers
# so analysis scripts can import a single helpers module.
#
# Key idea:
# - Internal model identifiers are CANONICAL keys (stable across refactors).
# - On-disk artifacts (derivatives/model_rdms_canon/rdm_<key>.npz) may use legacy keys.
# - `possible_disk_keys()` provides a canonical->legacy lookup order to locate files.

Source = Literal["human", "ai", "mixed", "none"]
Family = Literal["appraisal", "flavour", "visual", "conceptual", "other"]

# -----------------------------------------------------------------------------
# Canonical model keys (paper-facing; no __mean / no __0_100)
# -----------------------------------------------------------------------------

# Appraisal / valuation
APPRAISAL_HUMAN: List[str] = [
    "human_healthiness",
    "human_calorie_density",
    "human_appeal",
]
APPRAISAL_AI: List[str] = [
    "ai_healthiness",
    "ai_calorie_density",
]

# Flavour / taste
FLAVOUR_HUMAN: List[str] = [
    "human_sweetness",
    "human_saltiness",
    "human_sourness",
    "human_bitterness",
    "human_savoriness",
    "human_fattiness",
    "human_spiciness",
]
FLAVOUR_AI: List[str] = [
    "ai_sweetness",
    "ai_saltiness",
    "ai_sourness",
    "ai_bitterness",
    "ai_savoriness",
    "ai_fattiness",
    "ai_spiciness",
]

# Visual / low-level models (explicit list; single-underscore canonical keys)
# ll_edge_energy is intentionally excluded — it was dropped from the paper
# and is listed in policy_defaults.EXCLUDE_VISUAL_MODELS.
VISUAL: List[str] = [
    "ll_mean_luminance",
    "ll_rms_contrast",
    "ll_hsv_s_mean",
    "ll_lab_mean",
    "ll_lab_std",
    "ll_power_radial30",
    "ll_hog_pcs10",
    "ll_lbp_hist",
    "ll_resnet50_conv1_pcs10",
    "ll_resnet50_layer3_pcs10",
]

# Demoted / supplementary conceptual contrasts (excluded from unique-variance by default)
# REVISION 2026-09-10 (revision/transformation-models): this list is no longer
# defined here. It is imported from reference/policy_defaults.py so that the MDS
# model space (Fig 3), the univariate scan, the family-wise regression (Fig 5) and
# the variance attribution all select the same models. Previously this list held
# only the four taxonomy/NOVA keys, which meant natural_vs_transformed and
# Transformation_score were filtered out of Fig 3 and the Shapley even when they
# were present in pairwise_vectors.csv.
try:
    from reference.policy_defaults import CONCEPTUAL_SUPPLEMENTARY as _POLICY_CONCEPTUAL
    CONCEPTUAL_SUPPLEMENTARY: List[str] = list(_POLICY_CONCEPTUAL)
except Exception:  # pragma: no cover - fallback must mirror policy_defaults
    CONCEPTUAL_SUPPLEMENTARY: List[str] = [
        "food_classification",
        "Category_WHO_10",
        "Category_Culinary_9",
        "NOVA_ordinal",
        "natural_vs_transformed",
        "Transformation_score",
    ]

# Controls (optional)
CONTROLS_DEFAULT: List[str] = [
    "portion_size_ok",
    "quality",
]

# -----------------------------------------------------------------------------
# Straight-through naming (no remapping)
# -----------------------------------------------------------------------------

def canonicalize_key(key: str) -> str:
    """Return model keys unchanged (except for stripping a leading 'rdm_')."""
    k = str(key)
    if k.startswith("rdm_"):
        k = k[4:]
    return k


def canonicalize_keys(keys: Iterable[str]) -> List[str]:
    return [canonicalize_key(k) for k in keys]


def possible_disk_keys(key: str) -> List[str]:
    """Return the single expected disk key for a model (no legacy aliases)."""
    return [canonicalize_key(key)]


def possible_source_columns(key: str) -> List[str]:
    """Return the single expected source/info-sheet column for a model."""
    return [canonicalize_key(key)]


def resolve_info_col(info_df: pd.DataFrame, model_key: str) -> str | None:
    """Resolve an info-sheet column name for a given model key with no remapping."""
    k = canonicalize_key(model_key)
    return k if k in info_df.columns else None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: Family
    source: Source
    role: Literal["primary", "supplementary", "control"]


def _dedupe(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _family_source_for_key(key: str) -> Tuple[Family, Source]:
    """Infer (family, source) from a key (canonicalized)."""
    k = canonicalize_key(key)

    # visual / deep visual
    if k.startswith("vis_") or k.startswith("dl_") or k.startswith("ll_"):
        return ("visual", "none")

    # human ratings
    if k.startswith("human_"):
        if k in APPRAISAL_HUMAN:
            return ("appraisal", "human")
        if k in FLAVOUR_HUMAN:
            return ("flavour", "human")
        return ("other", "human")

    # AI ratings
    if k.startswith("ai_"):
        if k in APPRAISAL_AI:
            return ("appraisal", "ai")
        if k in FLAVOUR_AI:
            return ("flavour", "ai")
        return ("other", "ai")

    # conceptual contrasts
    if k.startswith("concept_"):
        return ("conceptual", "none")

    # controls
    if k in CONTROLS_DEFAULT:
        return ("other", "none")

    return ("other", "none")


def build_model_catalog() -> Dict[str, ModelSpec]:
    """Return a catalog of canonical model keys and their metadata."""
    catalog: Dict[str, ModelSpec] = {}

    def add(keys: Iterable[str], family: Family, source: Source, role: str):
        for k in keys:
            catalog[k] = ModelSpec(key=k, family=family, source=source, role=role)  # type: ignore[arg-type]

    add(APPRAISAL_HUMAN, "appraisal", "human", "primary")
    add(APPRAISAL_AI, "appraisal", "ai", "primary")
    add(FLAVOUR_HUMAN, "flavour", "human", "primary")
    add(FLAVOUR_AI, "flavour", "ai", "primary")
    add(VISUAL, "visual", "none", "primary")

    add(CONCEPTUAL_SUPPLEMENTARY, "conceptual", "none", "supplementary")
    add(CONTROLS_DEFAULT, "other", "none", "control")

    return catalog


def select_models(
    *,
    source_mode: Source = "human",
    include_visual: bool = True,
    include_controls: bool = False,
    include_supplementary_contrasts: bool = False,
    extra_include: Optional[List[str]] = None,
    extra_exclude: Optional[List[str]] = None,
) -> List[ModelSpec]:
    """Select model specs under the current paper policy.

    Parameters
    ----------
    source_mode:
      - "human": include human appraisal+flavour (no AI)
      - "ai": include AI appraisal+flavour (no human)
      - "mixed": include both human and AI (not recommended for unique variance)
    """

    extra_include = extra_include or []
    extra_exclude = extra_exclude or []

    cat = build_model_catalog()

    chosen: List[str] = []

    if source_mode in ("human", "mixed"):
        chosen += APPRAISAL_HUMAN + FLAVOUR_HUMAN
    if source_mode in ("ai", "mixed"):
        chosen += APPRAISAL_AI + FLAVOUR_AI

    if include_visual:
        chosen += VISUAL

    if include_controls:
        chosen += CONTROLS_DEFAULT

    if include_supplementary_contrasts:
        chosen += CONCEPTUAL_SUPPLEMENTARY

    chosen += [canonicalize_key(k) for k in extra_include]

    exclude_set = set(canonicalize_key(x) for x in extra_exclude)

    # Honour the policy exclusion list. Added 2026-09-11: EXCLUDE_MODELS was
    # only ever applied inside compute_model_evaluation, so select_models kept
    # handing excluded models to every other consumer (Fig 3 MDS, the variance
    # attribution gate, the model catalog summaries). With the ai_* ratings now
    # excluded by policy, that gap would have left AI models in the selection
    # while the scan skipped them: two different answers to "what is the model
    # set". policy_defaults is the single source of truth, so read it here too.
    try:
        from reference.policy_defaults import EXCLUDE_MODELS as _POLICY_EXCLUDE
        exclude_set |= {canonicalize_key(x) for x in _POLICY_EXCLUDE}
    except Exception:
        pass

    chosen = [c for c in chosen if c not in exclude_set]

    chosen = _dedupe(chosen)

    specs: List[ModelSpec] = []
    for k in chosen:
        k_c = canonicalize_key(k)
        if k_c in cat:
            s = cat[k_c]
            specs.append(ModelSpec(key=k_c, family=s.family, source=s.source, role=s.role))
        else:
            fam, src = _family_source_for_key(k_c)
            specs.append(ModelSpec(key=k_c, family=fam, source=src, role="supplementary"))

    return specs


def summarize_model_selection(specs: List[ModelSpec]) -> Dict[str, object]:
    """Summarize selected models for logging/manifests."""
    by_family: Dict[str, List[str]] = {}
    by_role: Dict[str, List[str]] = {}
    by_source: Dict[str, List[str]] = {}

    for s in specs:
        by_family.setdefault(s.family, []).append(s.key)
        by_role.setdefault(s.role, []).append(s.key)
        by_source.setdefault(s.source, []).append(s.key)

    return {
        "n_models": len(specs),
        "families": {k: len(v) for k, v in by_family.items()},
        "roles": {k: len(v) for k, v in by_role.items()},
        "sources": {k: len(v) for k, v in by_source.items()},
        "by_family": by_family,
        "by_role": by_role,
        "by_source": by_source,
        "models": [
            {**asdict(s), "disk_key_candidates": possible_disk_keys(s.key)}
            for s in specs
        ],
    }

# Backward-compatible alias expected by plotting scripts

def summarize(specs: List[ModelSpec]) -> Dict[str, object]:
    return summarize_model_selection(specs)


# -----------------------------------------------------------------------------
# Model RDM artifact naming (canonical keys)
# -----------------------------------------------------------------------------

def model_rdms_dir(base_dir: Optional[str] = None) -> Path:
    """Return the directory where pre-computed model RDMs live.

    By convention this is: analysis_code/derivatives/model_rdms_canon
    (the single canonical source for all model RDMs used in downstream analyses)
    """
    if base_dir is None:
        base_dir = DERIVATIVES_DIR
    return Path(base_dir) / "model_rdms_canon"


def canonical_model_rdm_filename(canonical_key: str) -> str:
    """Filename for a model RDM saved under the canonical-key convention."""
    k = canonicalize_key(canonical_key)
    return f"rdm_{k}.npz"


def find_model_rdm_path(
    key: str,
    base_dir: Optional[str] = None,
) -> Optional[Path]:
    """Locate a model RDM on disk using canonical/legacy fallbacks.

    Search order (first hit wins):
      1) rdm_<canonical>.npz
      2) rdm_<legacy>.npz for each legacy alias of the canonical key
      3) <candidate>_rdm.npz (older low-level naming)

    This lets analysis scripts adopt canonical keys immediately while still
    reading older artifacts.
    """
    d = model_rdms_dir(base_dir)
    if not d.exists():
        return None

    for cand in possible_disk_keys(key):
        # canonical convention
        p = d / f"rdm_{cand}.npz"
        if p.exists():
            return p

        # older convention used by some low-level scripts
        p2 = d / f"{cand}_rdm.npz"
        if p2.exists():
            return p2

    return None


def save_model_rdm_npz(
    *,
    rdm: np.ndarray,
    labels: List[str],
    key: str,
    out_dir: Optional[str] = None,
    outdir: Optional[str] = None,
    source_column: Optional[str] = None,
    kind: Optional[str] = None,
    also_write_legacy: bool = False,
    legacy_keys: Optional[List[str]] = None,
) -> Path:
    """Save a pre-computed model RDM using the canonical-key naming convention.

    Writes: derivatives/model_rdms_canon/rdm_<canonical_key>.npz

    The saved NPZ includes metadata fields:
      - canonical_key
      - source_column (optional)
      - kind (optional)

    If `also_write_legacy=True`, optionally writes additional copies using
    provided `legacy_keys` (or the default legacy aliases for this key) to ease
    transition for older scripts.
    """
    # Backwards-compatible alias
    if out_dir is None and outdir is not None:
        out_dir = outdir
    if out_dir is None:
        out_dir = str(model_rdms_dir())
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    canon = canonicalize_key(key)
    fname = canonical_model_rdm_filename(canon)
    path = outp / fname

    np.savez_compressed(
        path,
        rdm=np.asarray(rdm, dtype=float),
        labels=np.asarray(labels, dtype=object),
        canonical_key=str(canon),
        source_column=(None if source_column is None else str(source_column)),
        kind=(None if kind is None else str(kind)),
    )

    if also_write_legacy:
        if legacy_keys is None:
            legacy_keys = sorted(_CANON_TO_LEGACY.get(canon, []))
        for lk in legacy_keys:
            # Prefer rdm_<legacy>.npz for backwards compatibility
            legacy_path = outp / f"rdm_{lk}.npz"
            if legacy_path.resolve() == path.resolve():
                continue
            try:
                shutil.copyfile(path, legacy_path)
            except Exception:
                # best-effort; do not fail the pipeline for legacy copies
                pass

    return path
