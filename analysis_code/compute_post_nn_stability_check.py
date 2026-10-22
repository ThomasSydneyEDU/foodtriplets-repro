# Add future import at the top, after comments/docstring (no shebang present)
from __future__ import annotations

# Goal
# ----
# Quantify whether independently trained NN embeddings (different seeds)
# produce *high and tight* solutions (i.e., near-identical relational structure),
# or whether there are outliers / appreciable variability.

# What we measure
# ---------------
# For each seed, we build a *seed RDM* from the learned component scores
# (e.g., nn_comp1..nn_compK) using a multivariate distance:
#   - cosine distance (default) or euclidean.

# We then compute seed×seed similarity as Spearman( vec_upper(RDM_seed_i), vec_upper(RDM_seed_j) ).

# Outputs (default stable folder)
# ------------------------------
# analysis_code/compute_outputs/stage0_nn_seed_stability/
#   - nn_seed_similarity.csv              # seed×seed Spearman matrix
#   - nn_seed_similarity_long.csv         # long-format (pairs)
#   - nn_seed_similarity_summary.json     # summary stats + outlier flags
#   - fig_nn_seed_similarity_heatmap.png
#   - fig_nn_seed_similarity_distribution.png
#   - (optional) nn_seed_to_empirical.csv # seed→empirical Spearman, if --empirical-rdm provided
#   - (optional) nn_seed<SEED>_embedding_rdm.npz   # multivariate RDM from all selected components
#   - (optional) nn_seed<SEED>_component_rdms/     # per-component RDMs (|ci-cj|)
def _export_seed_rdms(
    *,
    seed: int,
    df_scores: pd.DataFrame,
    common_files: List[str],
    comp_cols_used: List[str],
    metric: str,
    outdir: Path,
    export_n_comp: Optional[int] = None,
) -> None:
    """Export RDMs derived from a specific seed’s component scores.

    Writes:
      - nn_seed<SEED>_embedding_rdm.npz  (multivariate RDM from all exported components)
      - nn_seed<SEED>_component_rdms/rdm_nn_seed<SEED>_<comp>.npz  (per-component |ci-cj|)

    Notes:
      - labels are the common filenames (intersection across seeds).
      - matrices use NaN on the diagonal, consistent with the stability script.
    """
    seed = int(seed)

    # Filter to the seed and align to common_files
    g = df_scores[df_scores["seed"].astype(int) == seed].copy()
    if g.empty:
        raise ValueError(f"Requested export seed {seed} not found in scores.")

    g = g.set_index("filename").reindex(common_files)

    comp_cols = list(comp_cols_used)
    if export_n_comp is not None:
        comp_cols = comp_cols[: max(1, min(int(export_n_comp), len(comp_cols)))]

    X = g[comp_cols].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        # Keep only rows that are finite across selected components
        ok = np.isfinite(X).all(axis=1)
        kept_files = [f for f, keep in zip(common_files, ok) if keep]
        g = g.loc[kept_files]
        X = g[comp_cols].to_numpy(dtype=float)
        labels = kept_files
    else:
        labels = list(common_files)

    # 1) Multivariate embedding RDM
    D_embed = _dist_matrix_from_components(X, metric=metric)
    embed_path = outdir / f"nn_seed{seed}_embedding_rdm.npz"
    np.savez_compressed(
        embed_path,
        rdm=D_embed.astype(np.float32),
        labels=np.array(labels, dtype=object),
        seed=seed,
        metric=str(metric),
        n_comp=int(len(comp_cols)),
        comp_cols=np.array(comp_cols, dtype=object),
    )

    # 2) Per-component RDMs
    comp_dir = outdir / f"nn_seed{seed}_component_rdms"
    comp_dir.mkdir(parents=True, exist_ok=True)

    for c in comp_cols:
        v = g[c].to_numpy(dtype=float)
        Dc = np.abs(v[:, None] - v[None, :]).astype(np.float32)
        np.fill_diagonal(Dc, np.nan)
        out_path = comp_dir / f"rdm_nn_seed{seed}_{c}.npz"
        np.savez_compressed(
            out_path,
            rdm=Dc,
            labels=np.array(labels, dtype=object),
            seed=seed,
            component=str(c),
            kind="nn_component_absdiff",
        )

    print(f"[OK] Exported embedding RDM: {embed_path}")
    print(f"[OK] Exported {len(comp_cols)} component RDMs to: {comp_dir}")


# --- Export best seed to model_rdms ---
def _rank_components_by_variance(
    seed: int,
    df_scores: pd.DataFrame,
    comp_cols_used: List[str],
) -> Tuple[List[str], List[Dict[str, object]]]:
    """Rank NN components by post-centering variance (descending).

    Mean-centering removes global activation magnitude (all NN scores are
    non-negative) so variance reflects how strongly a component discriminates
    among foods — analogous to eigenvalue ordering in PCA.  This criterion is
    fully independent of behavioral data, avoiding circularity when the
    components are later used as predictors in RSA.
    """
    g = df_scores[df_scores["seed"].astype(int) == int(seed)].copy()
    comp_cols = [c for c in comp_cols_used if c in g.columns]
    centered = g[comp_cols].subtract(g[comp_cols].mean(axis=0), axis=1)
    var_series = centered.var(axis=0).sort_values(ascending=False)
    ranked_cols = var_series.index.tolist()
    rows = [
        {
            "seed": int(seed),
            "component": str(c),
            "rank": int(i),
            "variance_centered": float(var_series[c]),
            "ordering": "variance",
        }
        for i, c in enumerate(ranked_cols, start=1)
    ]
    return ranked_cols, rows


def _export_best_seed_to_model_rdms(
    *,
    seed: int,
    df_scores: pd.DataFrame,
    common_files: List[str],
    comp_cols_used: List[str],
    metric: str,
    model_rdms_dir: Path,
    top_k: Optional[int] = None,
    export_embed_rdm: bool = False,
    empirical_npz: Optional[Path] = None,
) -> Dict[str, object]:
    """Export per-component (and optionally multivariate) RDMs into derivatives/model_rdms.

    Components are ranked by post-centering variance (highest first) — the same
    criterion used in the sorted-components CSV — so the two outputs are consistent.
    This is unbiased with respect to behavioral data, avoiding circularity when the
    exported RDMs are later used as predictors in RSA.

    Files:
      - rdm_nn_seed<SEED>_compXX.npz (per-component |ci-cj|)
      - (optional) rdm_nn_seed<SEED>_embedding_<metric>.npz (multivariate distance RDM)

    Returns a small manifest dict describing what was written.
    """
    seed = int(seed)
    model_rdms_dir = model_rdms_dir.resolve()
    model_rdms_dir.mkdir(parents=True, exist_ok=True)

    # Align to common_files
    g = df_scores[df_scores["seed"].astype(int) == seed].copy()
    if g.empty:
        raise ValueError(f"Requested export seed {seed} not found in scores.")
    g = g.set_index("filename").reindex(common_files)

    # Rank by variance (unbiased w.r.t. behavioral data)
    ranked_comp_cols, comp_rank_rows = _rank_components_by_variance(
        seed=int(seed),
        df_scores=df_scores,
        comp_cols_used=comp_cols_used,
    )
    comp_cols = list(ranked_comp_cols)
    if top_k is not None:
        comp_cols = comp_cols[: max(1, min(int(top_k), len(comp_cols)))]
        comp_rank_rows = [r for r in comp_rank_rows if int(r["rank"]) <= int(len(comp_cols))]

    # Keep only rows that are finite across selected components
    X = g[comp_cols].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        ok = np.isfinite(X).all(axis=1)
        kept_files = [f for f, keep in zip(common_files, ok) if keep]
        g = g.loc[kept_files]
        X = g[comp_cols].to_numpy(dtype=float)
        labels = kept_files
    else:
        labels = list(common_files)

    written = []

    # Optional multivariate embedding RDM
    embed_path = None
    if export_embed_rdm:
        D_embed = _dist_matrix_from_components(X, metric=metric)
        embed_path = model_rdms_dir / f"rdm_nn_seed{seed}_embedding_{metric}.npz"
        np.savez_compressed(
            embed_path,
            rdm=D_embed.astype(np.float32),
            labels=np.array(labels, dtype=object),
            seed=seed,
            metric=str(metric),
            n_comp=int(len(comp_cols)),
            comp_cols=np.array(comp_cols, dtype=object),
            nn_seed=int(seed),
            selected_nn_seed=int(seed),
            kind="nn_embedding_multivariate",
            canonical_key=f"nn_seed{seed}_embedding_{metric}",
        )
        written.append(str(embed_path))

    # Per-component RDMs
    for idx, c in enumerate(comp_cols, start=1):
        v = g[c].to_numpy(dtype=float)
        Dc = np.abs(v[:, None] - v[None, :]).astype(np.float32)
        np.fill_diagonal(Dc, np.nan)
        out_path = model_rdms_dir / f"rdm_nn_seed{seed}_comp{idx:02d}.npz"
        np.savez_compressed(
            out_path,
            rdm=Dc,
            labels=np.array(labels, dtype=object),
            seed=seed,
            component=str(c),
            component_index=int(idx),
            nn_seed=int(seed),
            selected_nn_seed=int(seed),
            component_rank=int(idx),
            kind="nn_component_absdiff",
            canonical_key=f"nn_seed{seed}_comp{idx:02d}",
        )
        written.append(str(out_path))

    manifest = {
        "seed": int(seed),
        "selected_nn_seed": int(seed),
        "metric": str(metric),
        "top_k": int(top_k) if top_k is not None else None,
        "export_embed_rdm": bool(export_embed_rdm),
        "embed_path": str(embed_path) if embed_path is not None else None,
        "n_items": int(len(labels)),
        "n_comp_exported": int(len(comp_cols)),
        "component_ordering": "variance",
        "component_rankings": comp_rank_rows,
        "files": written,
    }
    return manifest

# Usage
# -----
#   python analysis_code/results0b_nn_seed_stability.py

# Common:
#   python analysis_code/results0b_nn_seed_stability.py --metric cosine --n-comp 5
#   python analysis_code/results0b_nn_seed_stability.py --empirical-rdm analysis_code/derivatives/triplet_rdm_by_filename.npz

# Notes
# -----
# - Expects NN scores at: analysis_code/derivatives/nn_runs/nn_scores_all_seeds.csv
#   (or seed_*/nn_scores.csv). You can override with --scores.

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from reference.policy_defaults import (
        DERIVATIVES_DIR as POLICY_DERIVATIVES_DIR,
        RESULTS_DIR as POLICY_RESULTS_DIR,
        COMPUTE_DIR as POLICY_COMPUTE_DIR,
        DEFAULT_NN_SEED as POLICY_DEFAULT_NN_SEED,
        stage_outdir as policy_stage_outdir,
    )
except Exception:
    POLICY_DERIVATIVES_DIR = None
    POLICY_RESULTS_DIR = None
    POLICY_COMPUTE_DIR = None
    POLICY_DEFAULT_NN_SEED = 1
    policy_stage_outdir = None

import numpy as np
import pandas as pd


# -----------------------------
# Paths
# -----------------------------

HERE = Path(__file__).resolve()
SCRIPT_DIR = HERE.parent
DERIV_DIR = Path(POLICY_DERIVATIVES_DIR) if POLICY_DERIVATIVES_DIR is not None else SCRIPT_DIR / "derivatives"
COMPUTE_DIR = Path(POLICY_COMPUTE_DIR) if POLICY_COMPUTE_DIR is not None else SCRIPT_DIR / "compute_outputs"

DEFAULT_OUTDIR = COMPUTE_DIR / "stage0_nn_seed_stability"
DEFAULT_NN_RUNS_DIR = DERIV_DIR / "nn_runs"
DEFAULT_SCORES_AGG = DEFAULT_NN_RUNS_DIR / "nn_scores_all_seeds.csv"


# -----------------------------
# Stats helpers
# -----------------------------

def _spearman_vec(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation (average ranks) between two 1D arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float("nan")
    rx = pd.Series(x[m]).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y[m]).rank(method="average").to_numpy(dtype=float)
    sx = rx.std(ddof=0)
    sy = ry.std(ddof=0)
    if sx == 0 or sy == 0:
        return float("nan")
    rx = (rx - rx.mean()) / sx
    ry = (ry - ry.mean()) / sy
    return float(np.dot(rx, ry) / len(rx))


def _vectorize_upper(M: np.ndarray) -> np.ndarray:
    iu, ju = np.triu_indices_from(M, k=1)
    return M[iu, ju]


def _dist_matrix_from_components(X: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Build an RDM from component scores.

    X: [n_items, n_comp]
    metric: 'cosine' or 'euclidean'
    """
    X = np.asarray(X, dtype=float)
    # center columns for numerical stability (does not change cosine direction materially)
    X = X - np.nanmean(X, axis=0, keepdims=True)

    if metric == "cosine":
        eps = 1e-12
        norms = np.sqrt((X * X).sum(axis=1, keepdims=True)) + eps
        U = X / norms
        S = U @ U.T
        D = 1.0 - S
    elif metric == "euclidean":
        xx = (X ** 2).sum(axis=1, keepdims=True)
        D2 = xx + xx.T - 2.0 * (X @ X.T)
        D2[D2 < 0] = 0.0
        D = np.sqrt(D2, dtype=float)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    np.fill_diagonal(D, np.nan)
    return D


# -----------------------------
# IO helpers
# -----------------------------

def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _ensure_outdir(outdir: Path, timestamped: bool) -> Path:
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if timestamped:
        outdir = outdir / _timestamp()
        outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _find_scores_files(nn_runs_dir: Path) -> List[Path]:
    """Return per-seed nn_scores.csv paths, if present."""
    out: List[Path] = []
    if not nn_runs_dir.exists():
        return out
    for p in sorted(nn_runs_dir.glob("seed_*/nn_scores.csv")):
        out.append(p)
    return out


def _load_scores(scores_path: Optional[Path]) -> pd.DataFrame:
    """Load NN scores into a single dataframe containing 'seed' and 'filename'.

    Priority / behavior:
      1) If --scores is provided, load that file exactly.
      2) Otherwise, try BOTH:
         - aggregated: derivatives/nn_runs/nn_scores_all_seeds.csv
         - per-seed:   derivatives/nn_runs/seed_*/nn_scores.csv
         and take the UNION (de-duplicated on seed+filename).

    Rationale:
      The aggregated file can lag behind (e.g., new seeds exist on disk but the
      aggregated CSV wasn't refreshed). Unioning ensures we see all available seeds.
    """
    # 1) Explicit override
    if scores_path is not None:
        df = pd.read_csv(scores_path)
        if "seed" not in df.columns:
            raise ValueError(f"Scores file missing 'seed' column: {scores_path}")
        if "filename" not in df.columns:
            raise ValueError(f"Scores file missing 'filename' column: {scores_path}")
        # normalize types
        df = df.copy()
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
        df["filename"] = df["filename"].astype(str)
        return df

    frames: List[pd.DataFrame] = []

    # 2a) Aggregated (if present)
    if DEFAULT_SCORES_AGG.exists():
        try:
            df_agg = pd.read_csv(DEFAULT_SCORES_AGG)
            if "seed" in df_agg.columns and "filename" in df_agg.columns:
                df_agg = df_agg.copy()
                df_agg["seed"] = pd.to_numeric(df_agg["seed"], errors="coerce")
                df_agg["filename"] = df_agg["filename"].astype(str)
                frames.append(df_agg)
        except Exception:
            pass

    # 2b) Per-seed folders (if present)
    per_seed = _find_scores_files(DEFAULT_NN_RUNS_DIR)
    for p in per_seed:
        seed_str = p.parent.name.split("_", 1)[-1]
        try:
            seed_val = int(seed_str)
        except Exception:
            seed_val = None
        try:
            df_s = pd.read_csv(p)
        except Exception:
            continue
        if "filename" not in df_s.columns:
            continue
        df_s = df_s.copy()
        df_s.insert(0, "seed", seed_val)
        df_s["seed"] = pd.to_numeric(df_s["seed"], errors="coerce")
        df_s["filename"] = df_s["filename"].astype(str)
        frames.append(df_s)

    if not frames:
        raise FileNotFoundError(
            "Could not find NN scores. Expected either:\n"
            f"  - {DEFAULT_SCORES_AGG}\n"
            f"  - {DEFAULT_NN_RUNS_DIR}/seed_*/nn_scores.csv\n"
            "Or provide --scores PATH."
        )

    df_all = pd.concat(frames, ignore_index=True)

    # De-duplicate exact repeats (common when aggregated + per-seed overlap)
    if "seed" in df_all.columns and "filename" in df_all.columns:
        df_all = df_all.drop_duplicates(subset=["seed", "filename"], keep="last")

    # Keep only finite seeds
    df_all = df_all[df_all["seed"].notna()].copy()
    df_all["seed"] = df_all["seed"].astype(int)

    return df_all


def _load_empirical_rdm(npz_path: Path) -> Tuple[np.ndarray, Optional[List[str]]]:
    npz = np.load(npz_path, allow_pickle=True)
    keys = list(npz.keys())
    for k in ["data", "rdm", "D", "matrix"]:
        if k in keys:
            D = np.asarray(npz[k], dtype=float)
            break
    else:
        D = np.asarray(npz[keys[0]], dtype=float)
    labels = None
    if "labels" in keys:
        lab = npz["labels"]
        labels = lab.tolist() if hasattr(lab, "tolist") else list(lab)
    return D, labels

def _rank_components_by_empirical_fit(
    *,
    seed: int,
    df_scores: pd.DataFrame,
    common_files: List[str],
    comp_cols_used: List[str],
    empirical_npz: Optional[Path],
) -> Tuple[List[str], List[Dict[str, object]]]:
    """Return component columns ranked by univariate fit to the empirical RDM."""
    comp_cols = list(comp_cols_used)
    meta_rows: List[Dict[str, object]] = []

    def _fallback_rows(ordering: str = "original") -> Tuple[List[str], List[Dict[str, object]]]:
        rows: List[Dict[str, object]] = []
        for i, c in enumerate(comp_cols, start=1):
            rows.append({
                "seed": int(seed),
                "component": str(c),
                "rank": int(i),
                "rho_empirical": np.nan,
                "ordering": ordering,
            })
        return comp_cols, rows

    if empirical_npz is None or (not Path(empirical_npz).exists()):
        return _fallback_rows("original")

    try:
        D_emp, emp_labels = _load_empirical_rdm(Path(empirical_npz))
    except Exception:
        return _fallback_rows("original")

    labels = [os.path.basename(str(x)) for x in common_files]
    D_emp = np.asarray(D_emp, dtype=float)

    if emp_labels is not None:
        emp_labels = [os.path.basename(str(x)) for x in emp_labels]
        if set(emp_labels) == set(labels) and len(emp_labels) == len(labels):
            idx = [emp_labels.index(f) for f in labels]
            D_emp = D_emp[np.ix_(idx, idx)]
        elif D_emp.shape[0] != len(labels) or D_emp.shape[1] != len(labels):
            return _fallback_rows("original")
    else:
        if D_emp.shape[0] != len(labels) or D_emp.shape[1] != len(labels):
            return _fallback_rows("original")

    v_emp = _vectorize_upper(D_emp)

    g = df_scores[df_scores["seed"].astype(int) == int(seed)].copy()
    g["filename"] = g["filename"].astype(str).apply(os.path.basename)
    g = g.set_index("filename").reindex(labels)

    scored: List[Tuple[str, float]] = []
    for c in comp_cols:
        v = g[c].to_numpy(dtype=float)
        Dc = np.abs(v[:, None] - v[None, :]).astype(float)
        np.fill_diagonal(Dc, np.nan)
        rho = _spearman_vec(_vectorize_upper(Dc), v_emp)
        scored.append((str(c), float(rho) if np.isfinite(rho) else np.nan))

    order_lookup = {c: i for i, c in enumerate(comp_cols)}
    scored_sorted = sorted(
        scored,
        key=lambda x: (
            -(x[1]) if np.isfinite(x[1]) else np.inf,
            order_lookup[x[0]],
        ),
    )
    ranked_cols = [c for c, _ in scored_sorted]
    for i, (c, rho) in enumerate(scored_sorted, start=1):
        meta_rows.append({
            "seed": int(seed),
            "component": str(c),
            "rank": int(i),
            "rho_empirical": float(rho) if np.isfinite(rho) else np.nan,
            "ordering": "empirical_fit",
        })
    return ranked_cols, meta_rows
# -----------------------------
# Core analysis
# -----------------------------

@dataclass
class OutlierReport:
    threshold: float
    n_pairs: int
    n_below_threshold: int
    below_threshold_pairs: List[Tuple[int, int, float]]


def build_seed_rdms(
    df_scores: pd.DataFrame,
    metric: str,
    n_comp: Optional[int],
    seeds_keep: Optional[List[int]],
) -> Tuple[Dict[int, np.ndarray], List[str], List[str]]:
    """Return {seed: RDM}, along with the filenames order and component columns used."""

    df = df_scores.copy()

    # Basic validation
    if "seed" not in df.columns or "filename" not in df.columns:
        raise ValueError("Scores must contain columns: seed, filename")

    # Identify component columns
    comp_cols = [c for c in df.columns if str(c).startswith("nn_comp")]
    if not comp_cols:
        raise ValueError("No nn_comp* columns found in scores.")

    def _key(c: str) -> int:
        try:
            return int(str(c).replace("nn_comp", ""))
        except Exception:
            return 10**9

    comp_cols = sorted(comp_cols, key=_key)
    if n_comp is not None:
        comp_cols = comp_cols[: max(1, min(int(n_comp), len(comp_cols)))]

    # Filter seeds
    seeds = sorted([s for s in df["seed"].dropna().unique().tolist() if str(s) != "nan"])
    seeds = [int(s) for s in seeds]
    if seeds_keep:
        keep = set(int(x) for x in seeds_keep)
        seeds = [s for s in seeds if s in keep]

    if len(seeds) < 2:
        raise ValueError(f"Need >=2 seeds to compute stability, got: {seeds}")

    # Align filenames across seeds (intersection)
    by_seed = {s: df[df["seed"] == s].copy() for s in seeds}
    file_sets = [set(g["filename"].astype(str).tolist()) for g in by_seed.values()]
    common = sorted(set.intersection(*file_sets))
    if len(common) < 10:
        raise ValueError(f"Too few common filenames across seeds: n_common={len(common)}")

    # Build RDM per seed in the same file order
    rdms: Dict[int, np.ndarray] = {}
    for s in seeds:
        g = by_seed[s].set_index("filename").reindex(common)
        X = g[comp_cols].to_numpy(dtype=float)
        # Seeds may retain different numbers of dimensions; the aggregate CSV
        # pads shorter seeds with NaN in the extra columns.  Drop NaN *columns*
        # (dims this seed didn't keep) first, so we don't accidentally empty
        # the filename list by failing row-wise finite checks.
        col_ok = np.isfinite(X).all(axis=0)
        if not col_ok.all():
            X = X[:, col_ok]
        if X.shape[1] == 0:
            raise ValueError(f"Seed {s} has no finite component columns after NaN-column drop.")
        # After column filtering, any residual NaN rows are genuine missing items
        row_ok = np.isfinite(X).all(axis=1)
        if not row_ok.all():
            common = [f for f, keep in zip(common, row_ok) if keep]
            X = X[row_ok]
        rdms[s] = _dist_matrix_from_components(X, metric=metric)

    return rdms, common, comp_cols


def seed_similarity_matrix(rdms: Dict[int, np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seeds = sorted(rdms.keys())
    V = {s: _vectorize_upper(rdms[s]) for s in seeds}

    M = np.full((len(seeds), len(seeds)), np.nan, dtype=float)
    rows = []
    for i, si in enumerate(seeds):
        for j, sj in enumerate(seeds):
            if j < i:
                continue
            r = _spearman_vec(V[si], V[sj]) if si != sj else 1.0
            M[i, j] = M[j, i] = r
            if si != sj:
                rows.append({"seed_i": si, "seed_j": sj, "rho": float(r)})

    mat = pd.DataFrame(M, index=seeds, columns=seeds)
    long = pd.DataFrame(rows).sort_values(["rho"], ascending=False)
    return mat, long


def outlier_check(long_df: pd.DataFrame, threshold: float) -> OutlierReport:
    below = long_df[long_df["rho"] < float(threshold)].copy()
    pairs = [(int(r.seed_i), int(r.seed_j), float(r.rho)) for r in below.itertuples(index=False)]
    return OutlierReport(
        threshold=float(threshold),
        n_pairs=int(len(long_df)),
        n_below_threshold=int(len(below)),
        below_threshold_pairs=pairs,
    )


def plot_heatmap(mat: pd.DataFrame, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    A = mat.to_numpy(dtype=float)
    fig = plt.figure(figsize=(6.2, 5.4))
    ax = fig.add_subplot(111)
    im = ax.imshow(A, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_xticklabels([str(x) for x in mat.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(x) for x in mat.index])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman ρ")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(long_df: pd.DataFrame, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    vals = long_df["rho"].to_numpy(dtype=float)
    fig = plt.figure(figsize=(6.2, 3.8))
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=12)
    ax.set_title(title)
    ax.set_xlabel("Seed-pair Spearman ρ")
    ax.set_ylabel("Count")
    # annotate
    if len(vals) > 0 and np.isfinite(vals).any():
        ax.text(
            0.02,
            0.98,
            f"n_pairs={len(vals)}\nmin={np.nanmin(vals):.4f}\nmed={np.nanmedian(vals):.4f}\nmax={np.nanmax(vals):.4f}",
            transform=ax.transAxes,
            va="top",
        )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def seed_to_empirical(rdms: Dict[int, np.ndarray], empirical_npz: Path) -> pd.DataFrame:
    D_emp, labels = _load_empirical_rdm(empirical_npz)
    if D_emp.ndim != 2 or D_emp.shape[0] != D_emp.shape[1]:
        raise ValueError(f"Empirical RDM must be square, got {D_emp.shape}")

    # If labels are present, we can align by basename; otherwise assume already aligned.
    # (In FoodTriplets2 the seed RDM items are 209 labels; empirical could be 209 or 350 depending on preprocessing.)
    rows = []
    v_emp = _vectorize_upper(D_emp)

    for seed, D in rdms.items():
        if D.shape != D_emp.shape:
            # cannot safely align without labels; skip
            rows.append({"seed": int(seed), "rho": np.nan, "note": f"shape_mismatch seed={D.shape} emp={D_emp.shape}"})
            continue
        v = _vectorize_upper(D)
        rho = _spearman_vec(v, v_emp)
        rows.append({"seed": int(seed), "rho": float(rho), "note": ""})

    return pd.DataFrame(rows).sort_values("rho", ascending=False)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NN embedding stability across seeds")
    p.add_argument(
        "--scores",
        type=str,
        default=None,
        help=(
            "Optional path to nn_scores_all_seeds.csv. If omitted, tries "
            "analysis_code/derivatives/nn_runs/nn_scores_all_seeds.csv then seed_*/nn_scores.csv."
        ),
    )
    p.add_argument(
        "--metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric used to build seed RDMs from component scores.",
    )
    p.add_argument(
        "--n-comp",
        type=int,
        default=None,
        help="Limit to first K nn_comp columns (default: use all available).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of seeds to include (e.g., --seeds 0 1 2 3 4).",
    )
    p.add_argument(
        "--default-seed",
        type=int,
        default=int(POLICY_DEFAULT_NN_SEED),
        help="Representative/default NN seed from policy (informational unless used by export selection helpers).",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(DEFAULT_OUTDIR),
        help=f"Output directory (default: {DEFAULT_OUTDIR}).",
    )
    p.add_argument(
        "--timestamped",
        action="store_true",
        help="If set, create a timestamped subfolder under --outdir.",
    )
    p.add_argument(
        "--outlier-threshold",
        type=float,
        default=0.95,
        help="Flag seed-pairs with rho below this threshold.",
    )
    p.add_argument(
        "--empirical-rdm",
        type=str,
        default=None,
        help="Optional: compute seed→empirical Spearman using this empirical RDM NPZ.",
    )
    # Export/selection CLI options
    p.add_argument(
        "--export-seed-rdms",
        type=int,
        default=None,
        help=(
            "Optional: export RDMs for a specific seed based on its component scores. "
            "Writes one multivariate embedding RDM plus per-component RDMs (absolute differences)."
        ),
    )
    p.add_argument(
        "--export-outdir",
        type=str,
        default=None,
        help=(
            "Where to write exported RDMs. Default: <outdir>/nn_seed<SEED>_component_rdms/ "
            "and <outdir>/nn_seed<SEED>_embedding_rdm.npz."
        ),
    )
    p.add_argument(
        "--export-n-comp",
        type=int,
        default=None,
        help=(
            "When exporting per-component RDMs, limit to the first K components (default: export all used)."
        ),
    )
    # New CLI options for best seed/model RDMs
    p.add_argument(
        "--select-best-seed",
        action="store_true",
        help="Select the best/representative seed using a centrality criterion (overrides --export-seed-rdms unless --export-seed-rdms is explicitly provided).",
    )
    p.add_argument(
        "--best-seed-metric",
        choices=["median", "mean"],
        default="median",
        help="Centrality metric across seed-pairs (median or mean Spearman to other seeds).",
    )
    p.add_argument(
        "--export-best-to-model-rdms",
        action="store_true",
        help="If set, export best seed component RDMs into analysis_code/derivatives/model_rdms as rdm_nn_seed<SEED>_compXX.npz (plus optional multivariate embedding RDM).",
    )
    p.add_argument(
        "--export-top-k",
        type=int,
        default=None,
        help="When exporting to model_rdms, export only the first K components (after nn_comp sorting).",
    )
    p.add_argument(
        "--export-embed-rdm",
        action="store_true",
        help="Also export a multivariate embedding RDM for the best seed into model_rdms as rdm_nn_seed<SEED>_embedding_<metric>.npz.",
    )
    return p.parse_args(argv)
def _select_best_seed_from_similarity(mat: pd.DataFrame, metric: str = "median") -> Tuple[int, pd.DataFrame]:
    """Select a representative/best seed by centrality.

    Centrality is computed as the median (default) or mean similarity of a seed
    to all other seeds (excluding the diagonal). Returns (best_seed, table).
    """
    seeds = [int(s) for s in mat.index.tolist()]
    A = mat.to_numpy(dtype=float)
    # Exclude diagonal
    scores = []
    for i, s in enumerate(seeds):
        row = A[i, :].copy()
        row[i] = np.nan
        if metric == "mean":
            sc = float(np.nanmean(row))
        else:
            sc = float(np.nanmedian(row))
        scores.append({"seed": int(s), "centrality": sc})
    df = pd.DataFrame(scores).sort_values(["centrality", "seed"], ascending=[False, True]).reset_index(drop=True)
    best_seed = int(df.iloc[0]["seed"])
    return best_seed, df


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    outdir = _ensure_outdir(Path(args.outdir), bool(args.timestamped))

    # Load scores
    scores_path = Path(args.scores).resolve() if args.scores else None
    df = _load_scores(scores_path)

    # Build seed RDMs
    rdms, common_files, comp_cols = build_seed_rdms(
        df_scores=df,
        metric=str(args.metric),
        n_comp=args.n_comp,
        seeds_keep=args.seeds,
    )
    print(f"[INFO] Detected seeds: {sorted(rdms.keys())}")

    mat, long_df = seed_similarity_matrix(rdms)

    # Save
    mat_path = outdir / "nn_seed_similarity.csv"
    long_path = outdir / "nn_seed_similarity_long.csv"
    mat.to_csv(mat_path, index=True)
    long_df.to_csv(long_path, index=False)

    # Compute best seed by centrality
    best_seed, centrality_df = _select_best_seed_from_similarity(mat, metric=args.best_seed_metric)
    centrality_path = outdir / "nn_seed_centrality.csv"
    centrality_df.to_csv(centrality_path, index=False)

    # Summary
    vals = long_df["rho"].to_numpy(dtype=float)
    summary = {
        "n_seeds": int(len(rdms)),
        "seeds": [int(s) for s in sorted(rdms.keys())],
        "n_common_items": int(len(common_files)),
        "metric": str(args.metric),
        "n_comp_used": int(len(comp_cols)),
        "comp_cols": [str(c) for c in comp_cols],
        "pairwise": {
            "n_pairs": int(len(vals)),
            "min": float(np.nanmin(vals)) if len(vals) else float("nan"),
            "median": float(np.nanmedian(vals)) if len(vals) else float("nan"),
            "max": float(np.nanmax(vals)) if len(vals) else float("nan"),
            "p05": float(np.nanpercentile(vals, 5)) if len(vals) else float("nan"),
            "p95": float(np.nanpercentile(vals, 95)) if len(vals) else float("nan"),
            "iqr": float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)) if len(vals) else float("nan"),
        },
        "best_seed": int(best_seed),
        "seed_centrality": centrality_df.to_dict(orient="records"),
    }

    print(f"[INFO] Best seed by centrality ({args.best_seed_metric}): {best_seed}")
    print(f"[INFO] Policy default representative seed: {int(args.default_seed)}")

    out_rep = outlier_check(long_df, threshold=float(args.outlier_threshold))
    summary["outlier_check"] = asdict(out_rep)

    # Optional seed→empirical
    if args.empirical_rdm:
        emp_path = Path(args.empirical_rdm).resolve()
        if emp_path.exists():
            df_emp = seed_to_empirical(rdms, emp_path)
            df_emp.to_csv(outdir / "nn_seed_to_empirical.csv", index=False)
            summary["seed_to_empirical"] = {
                "path": str(emp_path),
                "rows": df_emp.to_dict(orient="records"),
            }

    (outdir / "nn_seed_similarity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Figures
    plot_heatmap(mat, outdir / "fig_nn_seed_similarity_heatmap.png", title="NN seed×seed similarity (Spearman ρ)")
    plot_distribution(long_df, outdir / "fig_nn_seed_similarity_distribution.png", title="NN seed-pair similarity distribution")

    # Determine export_seed for export-seed-rdms
    export_seed = None
    if args.export_seed_rdms is not None:
        export_seed = int(args.export_seed_rdms)
    elif args.select_best_seed:
        export_seed = int(best_seed)

    # Optional: export RDMs for a specific seed for downstream analyses (to results outdir)
    if export_seed is not None:
        export_root = Path(args.export_outdir).resolve() if args.export_outdir else outdir
        _export_seed_rdms(
            seed=int(export_seed),
            df_scores=df,
            common_files=common_files,
            comp_cols_used=comp_cols,
            metric=str(args.metric),
            outdir=export_root,
            export_n_comp=args.export_n_comp,
        )

    # Optional: export best seed to model_rdms for downstream analyses
    if args.export_best_to_model_rdms:
        model_rdms_dir = DERIV_DIR / "model_rdms_canon"
        manifest = _export_best_seed_to_model_rdms(
            seed=int(best_seed),
            df_scores=df,
            common_files=common_files,
            comp_cols_used=comp_cols,
            metric=str(args.metric),
            model_rdms_dir=model_rdms_dir,
            top_k=args.export_top_k,
            export_embed_rdm=bool(args.export_embed_rdm),
            empirical_npz=Path(args.empirical_rdm).resolve() if args.empirical_rdm else None,
        )
        # Write manifest
        manifest_path = outdir / "nn_best_seed_export_to_model_rdms.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        summary["export_to_model_rdms"] = manifest

    # Console
    print(f"[OK] Outdir: {outdir}")
    print(f"[INFO] Seeds: {summary['seeds']} | n_common_items={summary['n_common_items']} | metric={summary['metric']} | n_comp={summary['n_comp_used']}")
    pw = summary["pairwise"]
    print(
        "[INFO] Pairwise seed similarity: "
        f"n={pw['n_pairs']} | min={pw['min']:.4f} | med={pw['median']:.4f} | p05={pw['p05']:.4f} | p95={pw['p95']:.4f} | max={pw['max']:.4f} | IQR={pw['iqr']:.4f}"
    )

    if out_rep.n_below_threshold > 0:
        print(f"[WARN] {out_rep.n_below_threshold}/{out_rep.n_pairs} seed-pairs below threshold {out_rep.threshold:.3f}:")
        for si, sj, r in out_rep.below_threshold_pairs[:15]:
            print(f"  - ({si},{sj}) rho={r:.4f}")
        if len(out_rep.below_threshold_pairs) > 15:
            print(f"  ... and {len(out_rep.below_threshold_pairs) - 15} more")
    else:
        print(f"[OK] No seed-pairs below threshold {out_rep.threshold:.3f} (high & tight).")

    # -----------------------------
    # Extract Sorted Components for Best Seed
    # -----------------------------
    best_seed = summary.get("best_seed")
    try:
        if best_seed is not None:
            best_seed_dir = DEFAULT_NN_RUNS_DIR / f"seed_{best_seed}"
            scores_csv = best_seed_dir / "nn_scores.csv"
            if scores_csv.exists():
                print(f"[INFO] Exporting sorted components for best seed {best_seed}")
                df_scores = pd.read_csv(scores_csv)
                comp_cols = [c for c in df_scores.columns if c.startswith("nn_comp")]

                if comp_cols:
                    # Mean-centre each component so sorting reflects component-specific
                    # structure, not global activation magnitude.  All raw NN scores are
                    # non-negative (softmax / relu outputs), so without centering the
                    # rank ordering is dominated by overall food "activation" level and
                    # looks nearly identical across components.
                    df_centered = df_scores[comp_cols].subtract(df_scores[comp_cols].mean(axis=0), axis=1)

                    # Sort components by their variance (descending) so the most
                    # informative / distinctive components appear first in the output.
                    comp_variance = df_centered.var(axis=0).sort_values(ascending=False)
                    sorted_comp_cols = comp_variance.index.tolist()

                    out_cols = {}
                    for col in sorted_comp_cols:
                        order = df_centered[col].sort_values(ascending=False).index
                        out_cols[col] = df_scores.loc[order, "filename"].tolist()

                    out_df = pd.DataFrame(out_cols)
                    out_path = outdir / f"nn_seed{best_seed}_sorted_components.csv"
                    out_df.to_csv(out_path, index=False)
                    print(f"[OK] Saved sorted components (mean-centred, by variance) for best seed {best_seed} to: {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to export sorted components for best seed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
