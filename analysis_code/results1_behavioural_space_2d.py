#!/usr/bin/env python3
"""results1_behavioural_space_2d.py

Visualisation of the *empirical* food similarity space.

Creates 2D embeddings of the empirical triplet RDM and overlays key attributes:
- Figure A: colour by Category
- Figure B: colour by savoriness (0–100) and size by healthiness (0–100)
- correlates each available AI/human rating column with x and y (UMAP and PCA)

Outputs are written to: analysis_code/results/stage1_behavioural_space/

Usage:
  python analysis_code/results1_behavioural_space_2d.py

Optional:
  python analysis_code/results1_behavioural_space_2d.py --method mds
  python analysis_code/results1_behavioural_space_2d.py --method umap
  python analysis_code/results1_behavioural_space_2d.py --size-by human_healthiness

Notes:
- The *default* empirical RDM is: analysis_code/derivatives/triplet_rdm_by_filename.npz
- You can override this with --empirical-rdm or --use-completed-nn-seed.
- Expects labels at: analysis_code/derivatives/labels_lookup.csv
- Expects info at: reference/Foodpictures_information_canonical.csv

"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from analysis_helpers import load_info
from analysis_helpers import normalize_info_columns

def select_rating_columns(info: pd.DataFrame) -> list[str]:
    """Select numeric rating columns (human/ai) for correlation analysis.

    We include:
    - ai_* columns
    - human_* columns
    - *_0_100 columns
    - columns containing 'mean_' that are numeric

    Exclude clearly non-rating fields.
    """
    exclude = {
        "trial_index", "rt", "rt_ms", "time_elapsed", "sonaCode", "prolificPID",
        "image_file", "filename", "file", "Description", "Food", "category", "Category",
        "Natural_vs_transformed", "Natural vs transformed", "natural_vs_transformed",
        "_file", "_name", "_category",
    }

    cols: list[str] = []
    for c in info.columns:
        if c in exclude:
            continue
        cl = str(c).lower()
        if cl.startswith("ai_") or cl.startswith("human_") or cl.endswith("_0_100") or ("mean_" in cl):
            cols.append(c)

    # keep only those that can be numeric
    keep: list[str] = []
    for c in cols:
        s = pd.to_numeric(info[c], errors="coerce")
        if s.notna().sum() >= 10:  # at least some data
            keep.append(c)
    return keep


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    """Spearman correlation using pandas ranks (no scipy dependency)."""
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    m = aa.notna() & bb.notna()
    if int(m.sum()) < 10:
        return float("nan")
    ra = aa[m].rank(method="average")
    rb = bb[m].rank(method="average")
    return float(ra.corr(rb))


def coord_rating_correlations(coords_df: pd.DataFrame, rating_cols: list[str], prefix: str) -> pd.DataFrame:
    """Compute Spearman correlations between x/y and each rating column."""
    rows = []
    for c in rating_cols:
        rows.append({
            "embedding": prefix,
            "rating": c,
            "rho_x": spearman_corr(coords_df[c], coords_df["x"]),
            "rho_y": spearman_corr(coords_df[c], coords_df["y"]),
            "n": int(pd.to_numeric(coords_df[c], errors="coerce").notna().sum()),
        })
    out = pd.DataFrame(rows)
    # convenient sorting fields
    out["abs_rho_x"] = out["rho_x"].abs()
    out["abs_rho_y"] = out["rho_y"].abs()
    return out


def print_top_correlations(df: pd.DataFrame, k: int = 8, title: str | None = None) -> None:
    """Print the top correlations for X and Y.

    If `title` is provided, prints a header line before the table.
    """
    if df is None or df.empty:
        return

    if title:
        print(f"\n[INFO] {title}")

    print("[INFO] Top |rho| with X:")
    topx = df.sort_values("abs_rho_x", ascending=False).head(k)
    for _, r in topx.iterrows():
        print(f"  {r['rating']}: rho_x={r['rho_x']:.3f} (n={int(r['n'])})")

    print("[INFO] Top |rho| with Y:")
    topy = df.sort_values("abs_rho_y", ascending=False).head(k)
    for _, r in topy.iterrows():
        print(f"  {r['rating']}: rho_y={r['rho_y']:.3f} (n={int(r['n'])})")


# --- Helper to split ratings into human/ai/other groups ---
def split_rating_columns(rating_cols: list[str]) -> dict[str, list[str]]:
    """Split rating columns into human vs ai (and other)."""
    human = [c for c in rating_cols if str(c).lower().startswith("human_")]
    ai = [c for c in rating_cols if str(c).lower().startswith("ai_")]
    other = [c for c in rating_cols if c not in set(human) and c not in set(ai)]
    return {"human": human, "ai": ai, "other": other}


HERE = Path(__file__).resolve()
SCRIPT_DIR = HERE.parent
REPO_ROOT = SCRIPT_DIR.parent
DERIV_DIR = SCRIPT_DIR / "derivatives"
RESULTS_DIR = SCRIPT_DIR / "results"

DEFAULT_RDM_NPZ = DERIV_DIR / "triplet_rdm_by_filename.npz"
DEFAULT_LABELS_CSV = DERIV_DIR / "labels_lookup.csv"

try:
    from reference.policy_defaults import INFO_CSV as _POLICY_INFO_CSV
except Exception:
    _POLICY_INFO_CSV = None
INFO_CSV = Path(_POLICY_INFO_CSV) if _POLICY_INFO_CSV is not None else SCRIPT_DIR / "reference" / "Foodpictures_information_dynamic.csv"


def run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _basename(x: str) -> str:
    return os.path.basename(str(x).strip())


def load_empirical_rdm(npz_path: Path) -> Tuple[np.ndarray, Optional[list[str]]]:
    """Load an empirical RDM from an .npz file.

    Returns:
      - D: (n,n) float matrix
      - labels: optional list[str] if present in the npz
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing empirical RDM: {npz_path}")
    npz = np.load(npz_path, allow_pickle=True)
    keys = list(npz.keys())

    # Prefer common keys
    M = None
    for k in ["D", "data", "rdm", "RDM", "D_emp", "D_target", "D_TARGET", "matrix"]:
        if k in keys:
            M = np.asarray(npz[k], dtype=float)
            break
    if M is None:
        # fall back to first array-like entry
        M = np.asarray(npz[keys[0]], dtype=float)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Expected square matrix in {npz_path.name}; got {M.shape}")

    labels: Optional[list[str]] = None
    if "labels" in keys:
        try:
            raw = npz["labels"]
            raw_list = raw.tolist() if hasattr(raw, "tolist") else list(raw)
            labels = [_basename(x) for x in raw_list]
        except Exception:
            labels = None

    return M, labels


def load_labels(labels_csv: Path, n_expected: int) -> list[str]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Missing labels CSV: {labels_csv}")
    df = pd.read_csv(labels_csv)

    # Try common column names
    for c in ["filename", "image_file", "label", "stim", "file"]:
        if c in df.columns:
            labels = df[c].astype(str).apply(_basename).tolist()
            break
    else:
        # fallback: first column
        labels = df.iloc[:, 0].astype(str).apply(_basename).tolist()

    if len(labels) != n_expected:
        # If labels_lookup has two columns (e.g., idx, filename), try the last column
        if df.shape[1] >= 2:
            labels2 = df.iloc[:, -1].astype(str).apply(_basename).tolist()
            if len(labels2) == n_expected:
                labels = labels2

    if len(labels) != n_expected:
        raise ValueError(f"Label count mismatch: labels={len(labels)} vs RDM={n_expected}")

    return labels


def infer_natural_column(info: pd.DataFrame) -> Optional[str]:
    # Prefer explicit binary or categorical column names if present
    candidates = [
        "Natural_vs_transformed",
        "Natural vs transformed",
        "natural_vs_transformed",
        "naturalness",
        "Naturalness",
    ]
    for c in candidates:
        if c in info.columns:
            return c
    return None


def to_natural_binary(series: pd.Series) -> pd.Series:
    """Return True for Natural, False for Transformed, NaN for unknown."""
    s = series.copy()
    # Normalize
    s = s.astype(str).str.strip().str.lower()
    s = s.replace({"nan": "", "none": "", "": ""})

    # common encodings
    natural = {"natural", "raw", "unprocessed", "0", "n", "nat"}
    transformed = {"transformed", "prepared", "processed", "cooked", "1", "t", "trans"}

    out = pd.Series([np.nan] * len(s), index=s.index, dtype=float)
    out[s.isin(natural)] = 1.0
    out[s.isin(transformed)] = 0.0

    # If looks numeric (0/1)
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().any():
        # interpret 0=natural, 1=transformed (common), but allow either; here we assume 0=natural
        # If your coding is opposite, just swap below.
        out[num == 0] = 1.0
        out[num == 1] = 0.0

    return out


def build_info_table(labels: list[str], info_csv: Path, size_by: str) -> pd.DataFrame:
    if not info_csv.exists():
        raise FileNotFoundError(f"Missing info CSV: {info_csv}")

    info = load_info(info_csv)

    # Identify filename column
    fname_col = None
    for c in ["image_file", "filename", "file", "Image", "image"]:
        if c in info.columns:
            fname_col = c
            break
    if fname_col is None:
        raise KeyError("Info CSV must contain an image filename column (e.g., image_file)")

    info = normalize_info_columns(info)

    info["_file"] = info[fname_col].astype(str).apply(_basename)

    # Human-readable label (prefer Food/Description if present)
    name_col = None
    for c in ["Food", "food", "Description", "description", "Label", "label"]:
        if c in info.columns:
            name_col = c
            break
    if name_col is not None:
        info["_name"] = info[name_col].astype(str).str.strip()
    else:
        info["_name"] = info["_file"].astype(str)

    # Naturalness
    nat_col = infer_natural_column(info)
    if nat_col is None:
        raise KeyError("Could not find Natural vs transformed column in info CSV")
    info["_natural_bin"] = to_natural_binary(info[nat_col])

    # Category (for colouring)
    cat_col = None
    for c in ["category", "Category", "food_category", "FoodCategory", "food_classification"]:
        if c in info.columns:
            cat_col = c
            break
    if cat_col is not None:
        info["_category"] = info[cat_col].astype(str).str.strip()
        info["_category"] = info["_category"].replace({"nan": "", "None": "", "": ""})
    else:
        info["_category"] = ""

    # Size-by
    if size_by not in info.columns:
        raise KeyError(f"Requested size-by column '{size_by}' not found in info CSV")

    # Select rating columns (AI/human) for later correlation analyses
    rating_cols = select_rating_columns(info)

    # Ensure core overlays are present
    for must in ["human_savoriness", "human_healthiness"]:
        if must in info.columns and must not in rating_cols:
            rating_cols.append(must)

    # Join onto labels
    meta = pd.DataFrame({"filename": labels})
    cols_for_merge = ["_file", "_name", "_natural_bin", "_category", size_by] + rating_cols
    # de-duplicate while preserving order
    seen = set()
    cols_for_merge = [c for c in cols_for_merge if not (c in seen or seen.add(c))]

    meta = meta.merge(
        info[cols_for_merge].rename(columns={"_file": "filename"}),
        on="filename",
        how="left",
    )

    # Clean name fallback
    meta["_name"] = meta["_name"].fillna("").astype(str).str.strip()
    meta["display_name"] = meta.apply(
        lambda r: r["_name"] if r["_name"] not in {"", "nan", "None"} else Path(str(r["filename"])).stem.replace("-", " "),
        axis=1,
    )

    # Normalise size-by to numeric
    meta[size_by] = pd.to_numeric(meta[size_by], errors="coerce")
    for c in rating_cols:
        if c in meta.columns:
            meta[c] = pd.to_numeric(meta[c], errors="coerce")

    # Create readable category label
    meta["natural_label"] = meta["_natural_bin"].map({1.0: "Natural", 0.0: "Transformed"})

    # Category label for colouring (fallback to 'Unknown')
    meta["category_label"] = meta["_category"].astype(str).str.strip()
    meta.loc[meta["category_label"].isin(["", "nan", "None"]), "category_label"] = "Unknown"

    return meta


def embed_mds(D: np.ndarray, random_state: int = 0, n_init: int = 10, nonmetric: bool = False) -> Tuple[np.ndarray, float]:
    """2D MDS embedding from a precomputed distance matrix.

    Returns
    -------
    coords : (n, 2) array of 2D coordinates
    stress : normalised stress value (sklearn stress_ after normalisation)
    """
    try:
        from sklearn.manifold import MDS
    except Exception as e:
        raise RuntimeError("scikit-learn is required for MDS. Install with: pip install scikit-learn") from e

    D_arr = np.asarray(D, dtype=float)
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=int(random_state),
        n_init=int(n_init),
        max_iter=3000,
        normalized_stress=False,  # normalise manually so values are on Kruskal's 0–1 scale
        metric=(not bool(nonmetric)),
    )
    coords = mds.fit_transform(D_arr, init=None)
    mask = np.isfinite(D_arr) & ~np.eye(D_arr.shape[0], dtype=bool)
    ss_diss = float(np.sum(D_arr[mask] ** 2))
    stress_norm = float(mds.stress_) / ss_diss if ss_diss > 0 else float(mds.stress_)
    return coords, stress_norm


def compute_mds_stress_curve(
    D: np.ndarray,
    max_dims: int = 6,
    random_state: int = 0,
    n_init: int = 4,
) -> pd.DataFrame:
    """Fit MDS for n_components = 1 … max_dims; return a DataFrame of stress values.

    Uses fewer initialisations than the primary 2D fit to keep runtime short.
    Normalised stress is used throughout for comparability.
    """
    try:
        from sklearn.manifold import MDS
    except Exception as e:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn") from e

    D_arr = np.asarray(D, dtype=float)
    # Normalisation denominator: sum of squared observed (finite, non-diagonal) dissimilarities
    # This puts stress on Kruskal's 0–1 scale regardless of sklearn version behaviour.
    mask = np.isfinite(D_arr) & ~np.eye(D_arr.shape[0], dtype=bool)
    ss_diss = float(np.sum(D_arr[mask] ** 2))

    rows = []
    for n in range(1, max_dims + 1):
        m = MDS(
            n_components=n,
            dissimilarity="precomputed",
            random_state=int(random_state),
            n_init=int(n_init),
            max_iter=2000,
            normalized_stress=False,  # use raw stress_ and normalise manually below
        )
        m.fit(D_arr, init=None)
        stress_norm = float(m.stress_) / ss_diss if ss_diss > 0 else float(m.stress_)
        rows.append({"n_dims": n, "stress_raw": float(m.stress_), "stress_normalised": stress_norm})
        print(f"  [stress curve] {n}D  raw={m.stress_:.1f}  normalised={stress_norm:.4f}")
    return pd.DataFrame(rows)


def plot_mds_stress_curve(stress_df: pd.DataFrame, stress_2d: float, out_png: Path) -> None:
    """Scree-style plot of MDS stress vs number of dimensions."""
    import matplotlib.pyplot as plt

    # Use stress_normalised column if present, else fall back to stress
    y_col = "stress_normalised" if "stress_normalised" in stress_df.columns else "stress"

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(stress_df["n_dims"], stress_df[y_col], "o-", color="steelblue", linewidth=1.5, markersize=5)

    # Highlight the 2D solution used in the paper
    ax.scatter([2], [stress_2d], s=80, color="tomato", zorder=5, label=f"2D (stress = {stress_2d:.4f})")

    ax.set_xlabel("Number of dimensions", fontsize=10)
    ax.set_ylabel("Normalised stress", fontsize=10)
    ax.set_xticks(stress_df["n_dims"].tolist())
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_png.name}")


def embed_pca_from_rdm(D: np.ndarray, random_state: int = 0) -> np.ndarray:
    """PCA on the row-wise dissimilarity profiles.

    Treat each item as a vector of distances to all other items (rows of D),
    then apply PCA to project to 2D.
    """
    try:
        from sklearn.decomposition import PCA
    except Exception as e:
        raise RuntimeError("scikit-learn is required for PCA. Install with: pip install scikit-learn") from e

    X = np.asarray(D, dtype=float)
    # Ensure finite values for PCA; fill NaN with column means (rare)
    if not np.isfinite(X).all():
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(col_mean, inds[1])

    pca = PCA(n_components=2, random_state=int(random_state))
    return pca.fit_transform(X)


def embed_umap(D: np.ndarray, random_state: int = 0) -> Optional[np.ndarray]:
    try:
        import umap  # type: ignore
    except Exception:
        return None

    reducer = umap.UMAP(
        n_components=2,
        metric="precomputed",
        random_state=int(random_state),
        n_neighbors=min(30, max(5, D.shape[0] // 10)),
        min_dist=0.2,
    )
    return reducer.fit_transform(D)


# --- Helper functions for plotting overlays ---

def category_color_map(categories: pd.Series) -> dict:
    """Map category values to distinct colours using matplotlib default cycle."""
    import matplotlib.pyplot as plt
    cats = pd.Series(categories).fillna("Unknown").astype(str)
    uniq = sorted([c for c in cats.unique().tolist() if c])
    cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if not cycle:
        cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    return {cat: cycle[i % len(cycle)] for i, cat in enumerate(uniq)}


def scale_sizes(vals: pd.Series, min_size: float = 30.0, max_size: float = 220.0) -> np.ndarray:
    v = pd.to_numeric(vals, errors="coerce")
    if v.notna().sum() == 0:
        return np.full(len(vals), (min_size + max_size) / 2.0)
    vmin, vmax = float(np.nanmin(v.values)), float(np.nanmax(v.values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.full(len(vals), (min_size + max_size) / 2.0)
    norm = (v - vmin) / (vmax - vmin)
    norm = norm.fillna(0.5)
    return (min_size + (max_size - min_size) * norm).to_numpy(dtype=float)


def category_markers(categories: pd.Series) -> dict:
    """Assign marker shapes per category.

    We hard-code common categories for readability and fall back to cycling shapes for any extras.
    """
    cats = pd.Series(categories).fillna("Unknown").astype(str)
    uniq = sorted([c for c in cats.unique().tolist() if c])

    fixed = {
        "Fruit": "^",        # triangle up
        "Vegetable": "v",    # triangle down (requested)
        "Dish": "s",         # square
        "Protein": "P",      # filled plus
        "Grain": "D",        # diamond
        "Dessert": "o",      # circle
        "Unknown": "o",
    }

    # Remaining categories get assigned from the pool (avoiding those already used)
    pool = ["X", "<", ">", "*", "h", "H", "p"]
    used = set(fixed.values())
    pool = [m for m in pool if m not in used]

    out = {}
    j = 0
    for cat in uniq:
        if cat in fixed:
            out[cat] = fixed[cat]
        else:
            out[cat] = pool[j % len(pool)] if pool else "o"
            j += 1
    return out



# --- Spatial outlier detection ---

def flag_spatial_outliers(
    coords_df: pd.DataFrame,
    z_thresh: float = 2.0,
    min_cat_size: int = 4,
) -> pd.DataFrame:
    """Flag items that are spatial outliers *within their own category*.

    For each category with at least `min_cat_size` members:
      1. Compute the category centroid (mean x, mean y).
      2. Compute each item's Euclidean distance from that centroid.
      3. Flag items whose distance exceeds  mean + z_thresh * std
         of within-category distances.

    Items in tiny categories (< min_cat_size) are never flagged so that
    singletons/pairs don't generate spurious outliers.

    Adds three columns to a copy of `coords_df`:
      - ``centroid_x`` / ``centroid_y`` : category centroid
      - ``within_cat_dist``             : distance from centroid
      - ``is_spatial_outlier``          : bool flag
      - ``outlier_z``                   : z-score within category (NaN if not computed)

    Returns the annotated DataFrame.
    """
    df = coords_df.copy()
    df["centroid_x"] = np.nan
    df["centroid_y"] = np.nan
    df["within_cat_dist"] = np.nan
    df["outlier_z"] = np.nan
    df["is_spatial_outlier"] = False

    cat_col = "category_label"
    if cat_col not in df.columns:
        return df  # nothing to do

    for cat, grp in df.groupby(cat_col):
        idx = grp.index
        xs = pd.to_numeric(grp["x"], errors="coerce")
        ys = pd.to_numeric(grp["y"], errors="coerce")
        valid = xs.notna() & ys.notna()
        if valid.sum() < min_cat_size:
            continue

        cx = float(xs[valid].mean())
        cy = float(ys[valid].mean())
        dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

        mu = float(dists[valid].mean())
        sd = float(dists[valid].std(ddof=1)) if valid.sum() > 1 else 0.0

        df.loc[idx, "centroid_x"] = cx
        df.loc[idx, "centroid_y"] = cy
        df.loc[idx, "within_cat_dist"] = dists

        if sd > 0:
            z = (dists - mu) / sd
            df.loc[idx, "outlier_z"] = z
            df.loc[idx[valid], "is_spatial_outlier"] = z[valid] > z_thresh
        # if sd == 0 all items are at the centroid — no outliers

    return df


def save_outlier_report(coords_df: pd.DataFrame, out_path: Path, method_tag: str) -> None:
    """Write a CSV of flagged outlier items; print a summary to stdout."""
    outliers = coords_df[coords_df["is_spatial_outlier"] == True].copy()
    cols = [c for c in ["display_name", "filename", "category_label",
                        "x", "y", "within_cat_dist", "outlier_z"] if c in outliers.columns]
    outliers = outliers[cols].sort_values(["outlier_z"], ascending=False)
    outliers.to_csv(out_path, index=False)

    if outliers.empty:
        print(f"[INFO] {method_tag}: no spatial outliers detected (wrote {out_path})")
    else:
        print(f"[INFO] {method_tag}: {len(outliers)} spatial outlier(s) flagged (wrote {out_path})")
        for _, r in outliers.iterrows():
            name = r.get("display_name", r.get("filename", "?"))
            cat  = r.get("category_label", "?")
            z    = r.get("outlier_z", float("nan"))
            x, y = r.get("x", float("nan")), r.get("y", float("nan"))
            print(f"  [{cat}] {name}  x={x:.3f}  y={y:.3f}  z={z:.2f}")


# --- Label extreme points helper ---
def add_extreme_labels(ax, df: pd.DataFrame, n: int) -> None:
    """Label extreme items on x and y axes (top/bottom n on each), de-duplicated."""
    import numpy as _np

    if n is None or int(n) <= 0:
        return
    n = int(n)

    if "x" not in df.columns or "y" not in df.columns:
        return

    d = df.copy()
    # require finite coords
    m = _np.isfinite(d["x"].to_numpy()) & _np.isfinite(d["y"].to_numpy())
    d = d.loc[m].copy()
    if d.empty:
        return

    # choose extremes
    picks = []
    for col in ["x", "y"]:
        d_sorted = d.sort_values(col, ascending=True)
        picks.extend(d_sorted.head(n).index.tolist())
        picks.extend(d_sorted.tail(n).index.tolist())

    # de-duplicate while preserving order
    seen = set()
    idxs = []
    for i in picks:
        if i in seen:
            continue
        seen.add(i)
        idxs.append(i)

    # label offsets based on data range
    xr = float(d["x"].max() - d["x"].min())
    yr = float(d["y"].max() - d["y"].min())
    dx = 0.01 * xr if _np.isfinite(xr) and xr > 0 else 0.02
    dy = 0.01 * yr if _np.isfinite(yr) and yr > 0 else 0.02

    for i in idxs:
        r = d.loc[i]
        name = str(r.get("display_name", r.get("filename", "")))
        if not name:
            continue
        ax.text(
            float(r["x"]) + dx,
            float(r["y"]) + dy,
            name,
            fontsize=8,
            alpha=0.9,
        )


def add_named_labels(ax, df: pd.DataFrame, names: list[str]) -> None:
    """Label specific food items by name on the given axes.

    Parameters
    ----------
    ax : matplotlib Axes
    df : DataFrame with columns 'x', 'y', and 'display_name' (or 'filename')
    names : list of case-insensitive substrings to match against display_name
    """
    import numpy as _np

    if not names or "x" not in df.columns or "y" not in df.columns:
        return

    xr = float(df["x"].max() - df["x"].min()) if len(df) > 1 else 1.0
    yr = float(df["y"].max() - df["y"].min()) if len(df) > 1 else 1.0
    dx = 0.012 * xr if _np.isfinite(xr) and xr > 0 else 0.02
    dy = 0.012 * yr if _np.isfinite(yr) and yr > 0 else 0.02

    name_col = "display_name" if "display_name" in df.columns else "filename"
    names_lower = [n.lower() for n in names]

    for _, row in df.iterrows():
        label = str(row.get(name_col, ""))
        if not label:
            continue
        if any(n in label.lower() for n in names_lower):
            ax.annotate(
                label,
                xy=(float(row["x"]), float(row["y"])),
                xytext=(float(row["x"]) + dx, float(row["y"]) + dy),
                fontsize=11,
                alpha=0.95,
                arrowprops=dict(arrowstyle="-", color="black", lw=0.6, alpha=0.6),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )


_OUTLIER_LABELS = ["Cheese (brie)", "sweet breakfast cereal", "French fries (chips)",
                   "Garden salad", "Meatloaf", "Fruit leather"]


def add_top_extreme_labels(ax, df: pd.DataFrame, n: int = 10) -> None:
    """Label the N foods with the highest outlier_z score from the pre-saved
    stimuli_mds_category_outliers_empirical.csv (within-category MDS outliers).

    Falls back to centroid-distance selection if the CSV cannot be found.
    """
    import numpy as _np

    if n <= 0 or "x" not in df.columns or "y" not in df.columns:
        return

    outlier_csv = HERE.parent / "results" / "stage1_behavioural_space" / "stimuli_mds_category_outliers_empirical.csv"

    top_names: list[str] = []
    if outlier_csv.exists():
        outlier_df = pd.read_csv(outlier_csv)
        if "outlier_z" in outlier_df.columns and "display_name" in outlier_df.columns:
            top_names = (
                outlier_df.sort_values("outlier_z", ascending=False)
                .head(n)["display_name"]
                .tolist()
            )

    d = df.copy()
    finite_mask = _np.isfinite(d["x"].to_numpy()) & _np.isfinite(d["y"].to_numpy())
    d = d.loc[finite_mask].copy()
    if d.empty:
        return

    name_col = "display_name" if "display_name" in d.columns else "filename"

    if top_names:
        top = d.loc[d[name_col].isin(top_names)]
    else:
        # Fallback: centroid distance
        xs, ys = d["x"].to_numpy(), d["y"].to_numpy()
        cx, cy = xs.mean(), ys.mean()
        d = d.copy()
        d["_dist"] = _np.sqrt((d["x"] - cx) ** 2 + (d["y"] - cy) ** 2)
        top = d.nlargest(n, "_dist")

    if top.empty:
        return

    xs_all = d["x"].to_numpy()
    ys_all = d["y"].to_numpy()
    xr = float(xs_all.max() - xs_all.min()) if len(d) > 1 else 1.0
    yr = float(ys_all.max() - ys_all.min()) if len(d) > 1 else 1.0
    dx = 0.012 * xr if _np.isfinite(xr) and xr > 0 else 0.02
    dy = 0.012 * yr if _np.isfinite(yr) and yr > 0 else 0.02

    # --- collision-avoiding label placement (added 2026-09-15) ---------------
    # Every label used to go at the same (+dx, +dy) offset, so labels for points
    # that sit close together printed on top of each other. In the September
    # rebuild "Potato chips" and "Meatloaf" landed within a few hundredths of a
    # unit and were illegible. Try eight directions at two radii and take the
    # first that clears the labels already placed; fall back to the original
    # offset if nothing fits, which keeps the old behaviour as the worst case.
    _fs = 9
    _char_w = 0.0062 * xr          # approx text width per character, data units
    _line_h = 0.030 * yr           # approx text height, data units
    _placed: list[tuple[float, float, float, float]] = []

    def _overlaps(box, boxes) -> bool:
        x0, y0, x1, y1 = box
        for b0, c0, b1, c1 in boxes:
            if x0 < b1 and b0 < x1 and y0 < c1 and c0 < y1:
                return True
        return False

    _dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]

    for _, row in top.iterrows():
        label = str(row.get(name_col, ""))
        if not label:
            continue
        px, py = float(row["x"]), float(row["y"])
        w = max(len(label), 1) * _char_w
        h = _line_h
        best = None
        for scale in (1.0, 2.0, 3.2):
            for ux, uy in _dirs:
                tx = px + ux * dx * scale * (1.0 if ux == 0 else 1.0)
                ty = py + uy * dy * scale
                # anchor the box on the side the label runs toward
                x0 = tx if ux >= 0 else tx - w
                y0 = ty - h / 2.0
                box = (x0, y0, x0 + w, y0 + h)
                if not _overlaps(box, _placed):
                    best = (tx, ty, box, ux)
                    break
            if best is not None:
                break
        if best is None:
            tx, ty = px + dx, py + dy
            box = (tx, ty - h / 2.0, tx + w, ty + h / 2.0)
            ux = 1
        else:
            tx, ty, box, ux = best
        _placed.append(box)
        ax.annotate(
            label,
            xy=(px, py),
            xytext=(tx, ty),
            fontsize=_fs,
            alpha=0.95,
            ha="left" if ux >= 0 else "right",
            va="center",
            arrowprops=dict(arrowstyle="-", color="black", lw=0.6, alpha=0.6),
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )


def plot_space_category(coords: np.ndarray, meta: pd.DataFrame, title: str, out_png: Path, label_extremes: int = 0) -> None:
    """Figure A: colour by category (categorical), single marker."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    df = meta.copy()
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    cats = df.get("category_label", pd.Series(["Unknown"] * len(df)))
    cats = cats.fillna("Unknown").astype(str)
    cmap = category_color_map(cats)
    colors = cats.map(cmap).fillna("tab:gray")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(df["x"], df["y"], s=100, c=colors, alpha=0.80, edgecolor="white", linewidth=0.4, marker='o')

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    add_extreme_labels(ax, df, int(label_extremes))
    add_top_extreme_labels(ax, df, n=10)

    # Legend (categorical) — keep it spaced
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label=cat, markerfacecolor=cmap.get(cat, 'tab:gray'), markersize=8)
        for cat in sorted(cmap.keys())
    ]
    ax.legend(
        handles=legend_elems,
        title="Category",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        title_fontsize=10,
        ncol=1,
        labelspacing=0.8,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_space_factors(coords: np.ndarray, meta: pd.DataFrame, title: str, out_png: Path, label_extremes: int = 0) -> None:
    """Figure B: colour by savoriness, size by healthiness, single marker."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _SAVORY_SWEET_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "savory_sweet", ["#FFB300", "#B71C1C"]  # amber/gold (sweet, low) → deep red (savory, high)
    )

    df = meta.copy()
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    if "human_savoriness" not in df.columns:
        raise KeyError("human_savoriness not found in merged metadata; cannot colour points")
    if "human_healthiness" not in df.columns:
        raise KeyError("human_healthiness not found in merged metadata; cannot size points")

    col_vals = pd.to_numeric(df["human_savoriness"], errors="coerce")
    sizes = scale_sizes(df["human_healthiness"], min_size=35.0, max_size=240.0)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(
        df["x"], df["y"],
        s=sizes,
        c=col_vals,
        cmap=_SAVORY_SWEET_CMAP,
        vmin=0,
        vmax=100,
        marker='o',
        alpha=0.80,
        edgecolor="white",
        linewidth=0.4,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    add_extreme_labels(ax, df, int(label_extremes))

    sm = plt.cm.ScalarMappable(cmap=_SAVORY_SWEET_CMAP, norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.08)
    cbar.set_label("Savouriness (amber=sweet → red=savoury)", fontsize=10)

    # Size legend for healthiness
    size_legend_vals = [25, 50, 75, 100]
    size_legend_sizes = scale_sizes(pd.Series(size_legend_vals), min_size=35.0, max_size=240.0)
    size_handles = [
        plt.scatter([], [], s=s, color="grey", alpha=0.6, edgecolor="white", linewidth=0.4, label=str(v))
        for s, v in zip(size_legend_sizes, size_legend_vals)
    ]
    size_legend = ax.legend(
        handles=size_handles,
        title="Healthiness",
        frameon=False,
        loc="lower left",
        fontsize=8,
        title_fontsize=9,
        labelspacing=0.9,
        handletextpad=0.6,
        scatterpoints=1,
    )
    ax.add_artist(size_legend)

    # Reserve the same right margin as the category plot so the two figures align when placed side-by-side.
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_space_combined(coords: np.ndarray, meta: pd.DataFrame, title: str, out_png: Path, label_extremes: int = 0) -> None:
    """Combined figure: category panel (top) + factors panel (bottom), stacked vertically."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as mcolors
    _SAVORY_SWEET_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "savory_sweet", ["#FFB300", "#B71C1C"]  # amber/gold (sweet, low) → deep red (savory, high)
    )
    from matplotlib.lines import Line2D

    df = meta.copy()
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    if "human_savoriness" not in df.columns:
        raise KeyError("human_savoriness not found in metadata; cannot build combined figure")
    if "human_healthiness" not in df.columns:
        raise KeyError("human_healthiness not found in metadata; cannot build combined figure")

    # ── Layout: 3 rows (A, B, C-images), leave right margin for legends ────
    fig = plt.figure(figsize=(11, 20))
    # A and B are equal height; C (2×5 image grid) gets proportional height
    gs = gridspec.GridSpec(
        3, 1,
        figure=fig,
        height_ratios=[5, 5, 3],
        hspace=0.08,
        left=0.04, right=0.80, top=0.97, bottom=0.03,
    )
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # ── Panel A — category (colour by food category) ────────────────────────
    cats = df.get("category_label", pd.Series(["Unknown"] * len(df)))
    cats = cats.fillna("Unknown").astype(str)
    cmap_cat = category_color_map(cats)
    colors = cats.map(cmap_cat).fillna("tab:gray")

    ax_top.scatter(
        df["x"], df["y"],
        s=100, c=colors,
        alpha=0.80,
        edgecolor="white", linewidth=0.4, marker='o',
    )
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.text(-0.01, 1.01, "A", transform=ax_top.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="right")

    add_extreme_labels(ax_top, df, int(label_extremes))
    add_top_extreme_labels(ax_top, df, n=10)

    # Category legend — placed outside right of top panel
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label=cat,
               markerfacecolor=cmap_cat.get(cat, 'tab:gray'), markersize=8)
        for cat in sorted(cmap_cat.keys())
    ]
    ax_top.legend(
        handles=legend_elems,
        title="Category",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        fontsize=9,
        title_fontsize=10,
        ncol=1,
        labelspacing=0.8,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    # ── Panel B — factors (savouriness colour, healthiness size) ─────────────
    col_vals = pd.to_numeric(df["human_savoriness"], errors="coerce")
    sizes = scale_sizes(df["human_healthiness"], min_size=35.0, max_size=240.0)

    sc = ax_bot.scatter(
        df["x"], df["y"],
        s=sizes, c=col_vals,
        cmap=_SAVORY_SWEET_CMAP, vmin=0, vmax=100,
        marker='o', alpha=0.80,
        edgecolor="white", linewidth=0.4,
    )
    ax_bot.set_xticks([])
    ax_bot.set_yticks([])
    ax_bot.text(-0.01, 1.01, "B", transform=ax_bot.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="right")

    add_extreme_labels(ax_bot, df, int(label_extremes))

    # Colour bar anchored to bottom panel
    sm = plt.cm.ScalarMappable(cmap=_SAVORY_SWEET_CMAP, norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_bot, fraction=0.03, pad=0.02, aspect=25)
    cbar.set_label("Savouriness (amber=sweet → red=savoury)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Size legend for healthiness — placed in lower-left of bottom panel
    size_legend_vals = [25, 50, 75, 100]
    size_legend_sizes = scale_sizes(pd.Series(size_legend_vals), min_size=35.0, max_size=240.0)
    size_handles = [
        plt.scatter([], [], s=s, color="grey", alpha=0.6,
                    edgecolor="white", linewidth=0.4, label=str(v))
        for s, v in zip(size_legend_sizes, size_legend_vals)
    ]
    ax_bot.legend(
        handles=size_handles,
        title="Healthiness",
        frameon=False,
        loc="lower left",
        fontsize=8,
        title_fontsize=9,
        labelspacing=0.9,
        handletextpad=0.6,
        scatterpoints=1,
    )

    # ── Panel C — top-10 outlier images in rank order ────────────────────────
    import matplotlib.image as _mpimg

    _outlier_csv = HERE.parent / "results" / "stage1_behavioural_space" / "stimuli_mds_category_outliers_empirical.csv"
    _img_dir = HERE.parent.parent / "expt_stimuli" / "images"
    _n_imgs = 10

    if _outlier_csv.exists() and _img_dir.exists():
        _odf = (
            pd.read_csv(_outlier_csv)
            .sort_values("outlier_z", ascending=False)
            .head(_n_imgs)
            .reset_index(drop=True)
        )
        # 2 rows × 5 columns
        _gs_imgs = gridspec.GridSpecFromSubplotSpec(
            2, 5, subplot_spec=gs[2], wspace=0.04, hspace=0.35,
        )
        _img_axes = []
        for _i, _row in _odf.iterrows():
            _r, _c = divmod(_i, 5)
            _ax = fig.add_subplot(_gs_imgs[_r, _c])
            _img_axes.append(_ax)
            _img_path = _img_dir / _row["filename"]
            if _img_path.exists():
                _img = _mpimg.imread(str(_img_path))
                _ax.imshow(_img)
            else:
                _ax.set_facecolor("#eeeeee")
            _ax.set_xticks([])
            _ax.set_yticks([])
            for _spine in _ax.spines.values():
                _spine.set_visible(False)
            _name = str(_row.get("display_name", _row.get("filename", "")))
            _ax.set_xlabel(
                f"{_i + 1}. {_name}",
                fontsize=9, labelpad=3, ha="center",
            )
        # Panel label "C" above the first image
        if _img_axes:
            _img_axes[0].text(
                -0.12, 1.12, "C",
                transform=_img_axes[0].transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="right",
            )

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2D embedding of empirical triplet RDM with attribute overlays")
    p.add_argument("--method", type=str, default="mds", choices=["mds", "both", "umap", "pca"],
                   help="Embedding method (default: mds — the primary publication figure; use 'both' for exploratory comparison)")
    p.add_argument(
        "--size-by",
        type=str,
        default="human_savoriness",
        help="(legacy) Unused for plotting; plot uses colour=human_savoriness and size=human_healthiness"
    )
    p.add_argument("--random-state", type=int, default=0, help="Random state")
    p.add_argument("--mds-n-init", type=int, default=10,
                   help="MDS: number of random initialisations (higher = more stable)")
    p.add_argument("--mds-nonmetric", action="store_true",
                   help="Use non-metric MDS (rank-preserving)")
    p.add_argument(
        "--empirical-rdm",
        type=str,
        default=None,
        help=(
            "Optional path to an empirical RDM .npz to use instead of the default "
            "analysis_code/derivatives/triplet_rdm_by_filename.npz"
        ),
    )
    p.add_argument(
        "--use-completed-nn-seed",
        type=int,
        default=None,
        help=(
            "Convenience: use analysis_code/derivatives/triplet_rdm_completed_nn_seed{SEED}.npz as the empirical RDM. "
            "Overrides --empirical-rdm if both are provided."
        ),
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(RESULTS_DIR / "stage1_behavioural_space"),
        help="Output directory (default: analysis_code/results/stage1_behavioural_space)",
    )
    p.add_argument(
        "--label-extremes",
        type=int,
        default=0,
        help="Label the N most extreme items on each axis (x and y). 0 disables.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    label_extremes = int(getattr(args, "label_extremes", 0) or 0)

    # --- Empirical RDM selection (default vs overridden) ---
    rdm_path = DEFAULT_RDM_NPZ
    if args.empirical_rdm:
        rdm_path = Path(os.path.expanduser(args.empirical_rdm)).resolve()
    if args.use_completed_nn_seed is not None:
        rdm_path = (DERIV_DIR / f"triplet_rdm_completed_nn_seed{int(args.use_completed_nn_seed)}.npz").resolve()

    print(f"[INFO] Empirical RDM: {rdm_path}")

    # Human-readable tag for figure titles (so we never mix up observed vs completed)
    if args.use_completed_nn_seed is not None:
        rdm_tag = f"NN-completed RDM (seed {int(args.use_completed_nn_seed)})"
    elif args.empirical_rdm:
        rdm_tag = f"Empirical RDM override ({rdm_path.name})"
    else:
        rdm_tag = "Observed-only empirical RDM"

    # Output suffix to avoid confusing empirical vs completed (filled) embeddings
    out_suffix = "_filled" if args.use_completed_nn_seed is not None else "_empirical"

    def _with_suffix(filename: str) -> str:
        """Insert out_suffix before the file extension (or append if none)."""
        p = Path(filename)
        if p.suffix:
            return f"{p.stem}{out_suffix}{p.suffix}"
        return f"{filename}{out_suffix}"

    D, labels_from_npz = load_empirical_rdm(rdm_path)

    # Prefer labels inside the empirical RDM (prevents label-order mismatches)
    if labels_from_npz is not None and len(labels_from_npz) == D.shape[0]:
        labels = labels_from_npz
    else:
        labels = load_labels(DEFAULT_LABELS_CSV, n_expected=D.shape[0])

    size_by = args.size_by
    try:
        meta = build_info_table(labels, INFO_CSV, size_by=size_by)
    except KeyError:
        # fallback to healthiness
        if size_by != "human_healthiness":
            size_by = "human_healthiness"
            meta = build_info_table(labels, INFO_CSV, size_by=size_by)
        else:
            raise

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save meta used
    meta_out = outdir / _with_suffix("stimuli_meta_used.csv")
    meta.to_csv(meta_out, index=False)

    # Rating columns available for correlation analysis
    rating_cols = [c for c in meta.columns if (str(c).lower().startswith("ai_") or str(c).lower().startswith("human_") or str(c).lower().endswith("_0_100") or ("mean_" in str(c).lower()))]
    rating_cols = [c for c in rating_cols if c not in {"x", "y"}]
    rating_groups = split_rating_columns(rating_cols)

    # --- Sanity check: warn if any items lack a category label (should be none) ---
    unknown = meta[meta["category_label"] == "Unknown"]
    if not unknown.empty:
        print(f"[WARN] {len(unknown)} item(s) have no category label — check the info CSV:")
        for _, r in unknown[["filename", "display_name"]].head(15).iterrows():
            print(f"  - {r['display_name']}  ({r['filename']})")
    else:
        print("[INFO] All items have a recognised category label.")

    # --- PCA embedding + correlations ---
    if args.method in {"both", "pca"}:
        coords_pca = embed_pca_from_rdm(D, random_state=args.random_state)
        coords_pca[:, 1] *= -1
        pca_df = meta.copy()
        pca_df["x"] = coords_pca[:, 0]
        pca_df["y"] = coords_pca[:, 1]
        pca_df = flag_spatial_outliers(pca_df)
        pca_coords_path = outdir / _with_suffix("stimuli_pca_2d_coords.csv")
        pca_df.to_csv(pca_coords_path, index=False)
        pca_outliers_path = outdir / _with_suffix("stimuli_pca_category_outliers.csv")
        save_outlier_report(pca_df, pca_outliers_path, method_tag="PCA")

        corr_pca = coord_rating_correlations(pca_df, rating_cols=rating_cols, prefix="pca")
        corr_pca_path = outdir / _with_suffix("coord_rating_corr_pca.csv")
        corr_pca.to_csv(corr_pca_path, index=False)
        print(f"[OK] Wrote: {corr_pca_path}")
        # Print correlations split by source
        if rating_groups["human"]:
            print_top_correlations(corr_pca[corr_pca["rating"].isin(rating_groups["human"])], k=10, title="PCA: Human ratings")
        if rating_groups["ai"]:
            print_top_correlations(corr_pca[corr_pca["rating"].isin(rating_groups["ai"])], k=10, title="PCA: AI ratings")

        # Plot PCA space: category-only
        pca_cat_png = outdir / _with_suffix("stimuli_pca_category.png")
        plot_space_category(
            coords_pca,
            meta=meta,
            title=f"Empirical food space (PCA on RDM rows)\ncolour=Category\n{rdm_tag}",
            out_png=pca_cat_png,
            label_extremes=label_extremes,
        )
        print(f"[OK] Wrote: {pca_cat_png}")

        # Plot PCA space: factors
        pca_factors_png = outdir / _with_suffix("stimuli_pca_factors.png")
        plot_space_factors(
            coords_pca,
            meta=meta,
            title=f"Empirical food space (PCA on RDM rows)\ncolour=Savouriness, size=Healthiness\n{rdm_tag}",
            out_png=pca_factors_png,
            label_extremes=label_extremes,
        )
        print(f"[OK] Wrote: {pca_factors_png}")

    # MDS
    if args.method in {"both", "mds"}:
        coords_mds, mds_stress = embed_mds(
            D,
            random_state=args.random_state,
            n_init=args.mds_n_init,
            nonmetric=bool(args.mds_nonmetric),
        )
        coords_mds[:, 1] *= -1
        print(f"[INFO] MDS 2D normalised stress = {mds_stress:.4f}")
        coords_df = meta.copy()
        coords_df["x"] = coords_mds[:, 0]
        coords_df["y"] = coords_mds[:, 1]
        coords_df = flag_spatial_outliers(coords_df)
        mds_coords_path = outdir / _with_suffix("stimuli_mds_2d_coords.csv")
        coords_df.to_csv(mds_coords_path, index=False)
        mds_outliers_path = outdir / _with_suffix("stimuli_mds_category_outliers.csv")
        save_outlier_report(coords_df, mds_outliers_path, method_tag="MDS")

        corr_mds = coord_rating_correlations(coords_df, rating_cols=rating_cols, prefix="mds")
        corr_mds_path = outdir / _with_suffix("coord_rating_corr_mds.csv")
        corr_mds.to_csv(corr_mds_path, index=False)
        print(f"[OK] Wrote: {corr_mds_path}")
        # Print correlations split by source
        if rating_groups["human"]:
            print_top_correlations(corr_mds[corr_mds["rating"].isin(rating_groups["human"])], k=10, title="MDS: Human ratings")
        if rating_groups["ai"]:
            print_top_correlations(corr_mds[corr_mds["rating"].isin(rating_groups["ai"])], k=10, title="MDS: AI ratings")

        # Ordered list of NATURAL foods by x-coordinate (MDS)
        nat_mds = coords_df[coords_df["natural_label"] == "Natural"].copy()
        nat_mds = nat_mds.sort_values("x", ascending=True)
        nat_mds_out = outdir / _with_suffix("natural_sorted_by_x_mds.csv")
        nat_mds[["display_name", "filename", "x", "y", "human_savoriness"]].to_csv(nat_mds_out, index=False)
        print(f"[INFO] Wrote natural-by-x list (MDS): {nat_mds_out}  (n={len(nat_mds)})")
        # Show a quick intuition slice
        if len(nat_mds) > 0:
            print("[MDS] Natural foods at low-x end (first 10):")
            for _, r in nat_mds.head(10).iterrows():
                print(f"  {r['display_name']} | x={r['x']:.3f} | human_savoriness={r['human_savoriness']}")
            print("[MDS] Natural foods at high-x end (last 10):")
            for _, r in nat_mds.tail(10).iterrows():
                print(f"  {r['display_name']} | x={r['x']:.3f} | human_savoriness={r['human_savoriness']}")

        # Plot MDS space: category-only
        mds_cat_png = outdir / _with_suffix("stimuli_mds_category.png")
        plot_space_category(
            coords_mds,
            meta=meta,
            title="Food similarity space (MDS)",
            out_png=mds_cat_png,
            label_extremes=label_extremes,
        )
        print(f"[OK] Wrote: {mds_cat_png}")

        # Plot MDS space: factors
        mds_factors_png = outdir / _with_suffix("stimuli_mds_factors.png")
        plot_space_factors(
            coords_mds,
            meta=meta,
            title="Food similarity space (MDS)",
            out_png=mds_factors_png,
            label_extremes=label_extremes,
        )
        print(f"[OK] Wrote: {mds_factors_png}")

        # Plot MDS space: combined (factors top, category bottom)
        mds_combined_png = outdir / _with_suffix("stimuli_mds_combined.png")
        plot_space_combined(
            coords_mds,
            meta=meta,
            title="Food similarity space (MDS)",
            out_png=mds_combined_png,
            label_extremes=label_extremes,
        )
        print(f"[OK] Wrote: {mds_combined_png}")

        # MDS stress curve (dims 1–6) — supplementary diagnostic
        print("[INFO] Computing MDS stress curve (dims 1–6) ...")
        stress_df = compute_mds_stress_curve(
            D, max_dims=6, random_state=args.random_state, n_init=4
        )
        stress_csv = outdir / _with_suffix("mds_stress_curve.csv")
        stress_df.to_csv(stress_csv, index=False)
        print(f"[OK] Wrote: {stress_csv}")
        stress_png = outdir / _with_suffix("mds_stress_curve.png")
        plot_mds_stress_curve(stress_df, mds_stress, stress_png)
        print(f"[OK] Wrote: {stress_png}")

    # UMAP
    if args.method in {"both", "umap"}:
        coords_umap = embed_umap(D, random_state=args.random_state)
        if coords_umap is None:
            print("[WARN] UMAP not available (install with: pip install umap-learn). Skipping UMAP.")
        else:
            coords_umap[:, 1] *= -1
            coords_df = meta.copy()
            coords_df["x"] = coords_umap[:, 0]
            coords_df["y"] = coords_umap[:, 1]
            coords_df = flag_spatial_outliers(coords_df)
            umap_coords_path = outdir / _with_suffix("stimuli_umap_2d_coords.csv")
            coords_df.to_csv(umap_coords_path, index=False)
            umap_outliers_path = outdir / _with_suffix("stimuli_umap_category_outliers.csv")
            save_outlier_report(coords_df, umap_outliers_path, method_tag="UMAP")

            corr_umap = coord_rating_correlations(coords_df, rating_cols=rating_cols, prefix="umap")
            corr_umap_path = outdir / _with_suffix("coord_rating_corr_umap.csv")
            corr_umap.to_csv(corr_umap_path, index=False)
            print(f"[OK] Wrote: {corr_umap_path}")
            # Print correlations split by source
            if rating_groups["human"]:
                print_top_correlations(corr_umap[corr_umap["rating"].isin(rating_groups["human"])], k=10, title="UMAP: Human ratings")
            if rating_groups["ai"]:
                print_top_correlations(corr_umap[corr_umap["rating"].isin(rating_groups["ai"])], k=10, title="UMAP: AI ratings")

            # Ordered list of NATURAL foods by x-coordinate (UMAP)
            nat_umap = coords_df[coords_df["natural_label"] == "Natural"].copy()
            nat_umap = nat_umap.sort_values("x", ascending=True)
            nat_umap_out = outdir / _with_suffix("natural_sorted_by_x_umap.csv")
            nat_umap[["display_name", "filename", "x", "y", "human_savoriness"]].to_csv(nat_umap_out, index=False)
            print(f"[INFO] Wrote natural-by-x list (UMAP): {nat_umap_out}  (n={len(nat_umap)})")
            if len(nat_umap) > 0:
                print("[UMAP] Natural foods at low-x end (first 10):")
                for _, r in nat_umap.head(10).iterrows():
                    print(f"  {r['display_name']} | x={r['x']:.3f} | human_savoriness={r['human_savoriness']}")
                print("[UMAP] Natural foods at high-x end (last 10):")
                for _, r in nat_umap.tail(10).iterrows():
                    print(f"  {r['display_name']} | x={r['x']:.3f} | human_savoriness={r['human_savoriness']}")

            # Plot UMAP space: category-only
            umap_cat_png = outdir / _with_suffix("stimuli_umap_category.png")
            plot_space_category(
                coords_umap,
                meta=meta,
                title=f"Empirical food space (UMAP)\ncolour=Category\n{rdm_tag}",
                out_png=umap_cat_png,
                label_extremes=label_extremes,
            )
            print(f"[OK] Wrote: {umap_cat_png}")

            # Plot UMAP space: factors
            umap_factors_png = outdir / _with_suffix("stimuli_umap_factors.png")
            plot_space_factors(
                coords_umap,
                meta=meta,
                title=f"Empirical food space (UMAP)\ncolour=Savouriness, size=Healthiness\n{rdm_tag}",
                out_png=umap_factors_png,
                label_extremes=label_extremes,
            )
            print(f"[OK] Wrote: {umap_factors_png}")

    print(f"[INFO] Results folder: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
