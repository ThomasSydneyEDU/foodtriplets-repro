#!/usr/bin/env python3
"""compute_data_qc.py

Stage 0 compute: data quality control and reliability metrics.

Runs the slow computations once and writes stable CSVs + a summary JSON to
  analysis_code/compute_outputs/stage0_data_qc/

These outputs are consumed by:
  - results0_data_qc_reliability_pooling.py  (plots / paper-ready figures)
  - compute_model_evaluation.py              (noise ceiling annotation on model scan)

Outputs
-------
  compute_outputs/stage0_data_qc/
    pool_summary.csv               participant counts and trial stats per pool
    catch_error_rates.csv          group-level catch trial error rates
    noise_ceiling_split_half.csv   per-iteration split-half rho and rho_sb
    loso_stability.csv             per-subject LOSO stability
    data_quality_summary.json      canonical summary with noise ceiling value

Usage
-----
  python analysis_code/compute_data_qc.py
  python analysis_code/compute_data_qc.py --pool all --fixed-dense-only --n-iter 400
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (via policy_defaults where available)
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
import sys
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from reference.policy_defaults import (
        DERIVATIVES_DIR as _POLICY_DERIV,
        COMPUTE_DIR as _POLICY_COMPUTE,
        SUMMARY_CSV as _POLICY_SUMMARY_CSV,
    )
    DERIV_DIR    = Path(_POLICY_DERIV)
    COMPUTE_DIR  = Path(_POLICY_COMPUTE)
    SUMMARY_CSV  = Path(_POLICY_SUMMARY_CSV)
except Exception:
    DERIV_DIR   = HERE / "derivatives"
    COMPUTE_DIR = HERE / "compute_outputs"
    SUMMARY_CSV = DERIV_DIR / "summary.csv"

STAGE_OUTDIR = COMPUTE_DIR / "stage0_data_qc"

try:
    from analysis_helpers import load_summary_csv  # type: ignore
except Exception:
    def load_summary_csv(path: str) -> pd.DataFrame:  # type: ignore
        return pd.read_csv(path, low_memory=False)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class QCConfig:
    pool: str
    fixed_dense_only: bool
    exclude_catch_ids: list[int]
    n_iter: int
    seed: int


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_bool_series(s: pd.Series) -> pd.Series:
    """Robust boolean parsing for columns that may contain True/False, 1/0, strings, NaN."""
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
        return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return s.apply(_cell).astype(bool)


def _spearman_vec(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via rank->Pearson. Returns NaN if insufficient variance."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    xr = pd.Series(x[m]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y[m]).rank(method="average").to_numpy(dtype=float)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    xr = (xr - xr.mean()) / (xr.std(ddof=0) + 1e-12)
    yr = (yr - yr.mean()) / (yr.std(ddof=0) + 1e-12)
    return float(np.mean(xr * yr))


def _ensure_required(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Core computations  (same logic as results0; kept in sync)
# ---------------------------------------------------------------------------

def filter_summary(
    df: pd.DataFrame,
    pool: str,
    fixed_dense_only: bool,
    exclude_catch_ids: list[int],
) -> pd.DataFrame:
    """Apply consistent filters for QC and reliability analyses."""
    df = df.copy()
    if pool.lower() != "all":
        if "pool" not in df.columns:
            raise ValueError("summary.csv has no 'pool' column; cannot filter by pool")
        df = df[df["pool"].astype(str).str.lower() == pool.lower()].copy()
    if fixed_dense_only:
        if "is_fixed" not in df.columns and "is_dense" not in df.columns:
            raise ValueError("fixed_dense_only requested but summary.csv has no is_fixed/is_dense columns")
        is_fixed = _parse_bool_series(df.get("is_fixed", pd.Series([False] * len(df))))
        is_dense = _parse_bool_series(df.get("is_dense", pd.Series([False] * len(df))))
        df = df.loc[(is_fixed | is_dense)].copy()
    if exclude_catch_ids:
        if "is_catch" not in df.columns:
            return df
        is_catch = _parse_bool_series(df["is_catch"])
        if "catch_id" in df.columns:
            cid = pd.to_numeric(df["catch_id"], errors="coerce")
            df = df.loc[~(is_catch & cid.isin(exclude_catch_ids))].copy()
        else:
            raise ValueError("exclude_catch_ids requested but summary.csv has no 'catch_id' column")
    return df


def pool_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pool-level participant summary."""
    if "SubID" not in df.columns:
        raise ValueError("summary.csv missing SubID column")
    d = df.copy()
    d["SubID"] = d["SubID"].astype(str)
    g = d.groupby(["pool", "SubID"], as_index=False).size().rename(columns={"size": "n_trials"})
    if "passed_attention" in d.columns:
        pa = _parse_bool_series(d["passed_attention"]).groupby([d["pool"], d["SubID"]]).all().reset_index()
        pa.columns = ["pool", "SubID", "passed_attention"]
        g = g.merge(pa, on=["pool", "SubID"], how="left")
    rows = []
    for pool_name, sub in g.groupby("pool"):
        n_subs = int(sub["SubID"].nunique())
        row = {
            "pool": pool_name,
            "n_participants": n_subs,
            "median_trials": float(sub["n_trials"].median()) if n_subs else float("nan"),
            "p05_trials": float(np.percentile(sub["n_trials"], 5)) if n_subs else float("nan"),
            "p95_trials": float(np.percentile(sub["n_trials"], 95)) if n_subs else float("nan"),
            "min_trials": int(sub["n_trials"].min()) if n_subs else 0,
            "max_trials": int(sub["n_trials"].max()) if n_subs else 0,
        }
        if "passed_attention" in sub.columns:
            passed = sub["passed_attention"].fillna(False)
            row.update({
                "n_passed_attention": int(passed.sum()),
                "n_failed_attention": int((~passed).sum()),
                "pass_rate": float(passed.mean()) if len(passed) else float("nan"),
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pool")


def catch_error_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Group-level catch trial error rates."""
    if "is_catch" not in df.columns:
        return pd.DataFrame(columns=["catch_id", "n", "n_incorrect", "error_rate"]).astype({"catch_id": "Int64"})
    is_catch = _parse_bool_series(df["is_catch"]).to_numpy()
    d = df.loc[is_catch].copy()
    if d.empty:
        return pd.DataFrame(columns=["catch_id", "n", "n_incorrect", "error_rate"]).astype({"catch_id": "Int64"})
    if "correct" not in d.columns:
        raise ValueError("catch trials present but 'correct' column missing")
    if "catch_id" not in d.columns:
        corr = _parse_bool_series(d["correct"]).to_numpy()
        n = int(len(d))
        n_inc = int((~corr).sum())
        return pd.DataFrame({"catch_id": [pd.NA], "n": [n], "n_incorrect": [n_inc], "error_rate": [n_inc / max(n, 1)]})
    d["catch_id"] = pd.to_numeric(d["catch_id"], errors="coerce").astype("Int64")
    d["correct_bool"] = _parse_bool_series(d["correct"]).astype(bool)
    rows = []
    for cid, sub in d.groupby("catch_id"):
        n = int(len(sub))
        n_inc = int((~sub["correct_bool"]).sum())
        rows.append({"catch_id": cid, "n": n, "n_incorrect": n_inc, "error_rate": float(n_inc / max(n, 1))})
    return pd.DataFrame(rows).sort_values("catch_id")


def build_pairwise_from_summary(df: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Build D = 1 - p_sim from odd-one-out trials; also returns count matrix."""
    _ensure_required(df, ["StimA_file", "StimB_file", "StimC_file", "choice_file"], "build_pairwise")
    labels = [os.path.basename(str(x)) for x in labels]
    idx = {fn: i for i, fn in enumerate(labels)}
    n = len(labels)
    sim = np.zeros((n, n), dtype=float)
    cnt = np.zeros((n, n), dtype=float)

    def _b(x): return os.path.basename(str(x))

    for _, r in df.iterrows():
        a, b, c = _b(r["StimA_file"]), _b(r["StimB_file"]), _b(r["StimC_file"])
        if pd.isna(r["choice_file"]):
            continue
        choice = _b(r["choice_file"])
        if a not in idx or b not in idx or c not in idx:
            continue
        ia, ib, ic = idx[a], idx[b], idx[c]
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
            u, v = sorted((p1, p2))
            cnt[u, v] += 1.0

    sim = sim + sim.T
    cnt = cnt + cnt.T
    with np.errstate(invalid="ignore", divide="ignore"):
        p_sim = np.where(cnt > 0, sim / cnt, np.nan)
    D = 1.0 - p_sim
    np.fill_diagonal(D, np.nan)
    np.fill_diagonal(cnt, 0.0)
    return D, cnt


def split_half_noise_ceiling(
    df: pd.DataFrame,
    labels: list[str],
    n_iter: int,
    seed: int,
) -> pd.DataFrame:
    """Participant split-half reliability with Spearman-Brown correction.

    Returns per-iteration rho (raw) and rho_sb (SB-corrected full-group estimate).
    This is the canonical noise ceiling for the model evaluation.
    """
    if "SubID" not in df.columns:
        raise ValueError("split_half_noise_ceiling requires SubID")
    subs = df["SubID"].astype(str).str.strip().replace({"nan": "", "None": ""}).fillna("")
    df = df.assign(SubID=subs)
    unique_subs = [s for s in df["SubID"].unique().tolist() if s not in {"", "nan", "None"}]
    if len(unique_subs) < 4:
        raise ValueError(f"Not enough unique SubIDs to split-half (n={len(unique_subs)})")

    rng = np.random.default_rng(seed)
    iu, ju = np.triu_indices(len(labels), k=1)
    rows = []

    for it in range(n_iter):
        s = unique_subs.copy()
        rng.shuffle(s)
        mid = len(s) // 2
        df1 = df[df["SubID"].isin(set(s[:mid]))]
        df2 = df[df["SubID"].isin(set(s[mid:]))]
        D1, C1 = build_pairwise_from_summary(df1, labels)
        D2, C2 = build_pairwise_from_summary(df2, labels)
        m = (C1[iu, ju] > 0) & (C2[iu, ju] > 0)
        if m.sum() < 20:
            continue
        rho = _spearman_vec(D1[iu, ju][m], D2[iu, ju][m])
        if np.isnan(rho):
            continue
        rho_sb = (2.0 * rho) / (1.0 + rho) if np.isfinite(rho) else np.nan
        rows.append({"iter": it + 1, "rho": float(rho), "rho_sb": float(rho_sb), "overlap_pairs": int(m.sum())})

    return pd.DataFrame(rows)


def pool_rdm_overlap_corr(
    df: pd.DataFrame,
    labels: list[str],
    pool_a: str,
    pool_b: str,
) -> dict:
    """Correlation between pool-level RDMs on overlapping observed pairs."""
    if "pool" not in df.columns:
        raise ValueError("summary.csv missing pool column")
    Da, Ca = build_pairwise_from_summary(df[df["pool"].astype(str).str.lower() == pool_a.lower()], labels)
    Db, Cb = build_pairwise_from_summary(df[df["pool"].astype(str).str.lower() == pool_b.lower()], labels)
    iu, ju = np.triu_indices(len(labels), k=1)
    m = (Ca[iu, ju] > 0) & (Cb[iu, ju] > 0)
    rho = _spearman_vec(Da[iu, ju][m], Db[iu, ju][m])
    return {"rho": float(rho), "overlap_pairs": int(m.sum()), "pool_a": pool_a, "pool_b": pool_b}


def loso_stability(
    df: pd.DataFrame,
    labels: list[str],
    pool: Optional[str] = None,
) -> pd.DataFrame:
    """Leave-one-subject-out stability: correlation of full RDM vs each LOO RDM."""
    if "SubID" not in df.columns:
        raise ValueError("LOSO requires SubID")
    subs = df["SubID"].astype(str).str.strip().replace({"nan": "", "None": ""}).fillna("")
    df = df.assign(SubID=subs)
    unique_subs = [s for s in df["SubID"].unique().tolist() if s not in {"", "nan", "None"}]
    D_full, C_full = build_pairwise_from_summary(df, labels)
    iu, ju = np.triu_indices(len(labels), k=1)
    rows = []
    for sid in unique_subs:
        D_loo, C_loo = build_pairwise_from_summary(df[df["SubID"] != sid], labels)
        m = (C_full[iu, ju] > 0) & (C_loo[iu, ju] > 0)
        if m.sum() < 20:
            continue
        rho = _spearman_vec(D_full[iu, ju][m], D_loo[iu, ju][m])
        delta = float(np.nanmean(np.abs(D_full[iu, ju][m] - D_loo[iu, ju][m])))
        rows.append({"SubID": sid, "rho": float(rho), "mean_abs_delta": delta, "overlap_pairs": int(m.sum())})
    out = pd.DataFrame(rows)
    if pool is not None:
        out.insert(0, "pool", pool)
    return out


def loso_ceiling(
    df: pd.DataFrame,
    labels: list[str],
    min_overlap_pairs: int = 20,
) -> pd.DataFrame:
    """LOSO noise ceiling: correlate each individual's RDM with the pooled N-1 RDM.

    This is a diagnostic estimate only — it tends to be inflated because a single
    participant contributes very little to the N-1 pool, so the pool is nearly
    identical to the full dataset.  Use split_half_noise_ceiling (Method 1) as the
    canonical ceiling for model evaluation.

    Returns per-subject rows with columns: SubID, rho, n_overlap_pairs, used.
    """
    if "SubID" not in df.columns:
        raise ValueError("loso_ceiling requires SubID")
    subs = df["SubID"].astype(str).str.strip().replace({"nan": "", "None": ""}).fillna("")
    df = df.assign(SubID=subs)
    unique_subs = [s for s in df["SubID"].unique().tolist() if s not in {"", "nan", "None"}]

    iu, ju = np.triu_indices(len(labels), k=1)
    rows = []
    rhos: list[float] = []
    overlap_counts: list[int] = []

    for sid in sorted(unique_subs):
        df_sub  = df[df["SubID"] == sid]
        df_rest = df[df["SubID"] != sid]
        if df_sub.empty or df_rest.empty:
            continue
        try:
            D_sub,  C_sub  = build_pairwise_from_summary(df_sub,  labels)
            D_rest, C_rest = build_pairwise_from_summary(df_rest, labels)
        except Exception:
            continue
        m = (C_sub[iu, ju] > 0) & (C_rest[iu, ju] > 0)
        n_ov = int(m.sum())
        if n_ov < min_overlap_pairs:
            rows.append({"SubID": sid, "rho": float("nan"), "n_overlap_pairs": n_ov, "used": False})
            continue
        v_sub  = D_sub [iu, ju][m].astype(float)
        v_rest = D_rest[iu, ju][m].astype(float)
        ok = np.isfinite(v_sub) & np.isfinite(v_rest)
        if ok.sum() < min_overlap_pairs:
            rows.append({"SubID": sid, "rho": float("nan"), "n_overlap_pairs": int(ok.sum()), "used": False})
            continue
        rho = _spearman_vec(v_sub[ok], v_rest[ok])
        rows.append({"SubID": sid, "rho": float(rho) if np.isfinite(rho) else float("nan"),
                     "n_overlap_pairs": int(ok.sum()), "used": bool(np.isfinite(rho))})
        if np.isfinite(rho):
            rhos.append(float(rho))
            overlap_counts.append(int(ok.sum()))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RDM coverage statistics
# ---------------------------------------------------------------------------

def rdm_coverage_stats(cnt: np.ndarray, labels: list[str]) -> dict:
    """Compute RDM sampling coverage statistics from the pairwise count matrix.

    Parameters
    ----------
    cnt : np.ndarray
        Symmetric pairwise count matrix of shape (n, n).  cnt[i, j] is the
        number of times items i and j co-appeared in the same triplet (after
        symmetrisation by build_pairwise_from_summary).
    labels : list[str]
        Item labels corresponding to the rows/columns of cnt.

    Returns
    -------
    dict
        Coverage statistics suitable for JSON serialisation and manuscript
        reporting, including total responses, % of possible triplets sampled,
        pairwise coverage, and per-pair observation counts.

    Notes
    -----
    Each triplet (a, b, c) contributes +1 to cnt for all three pairs
    (a,b), (a,c), (b,c).  After symmetrisation the upper-triangle sum
    therefore equals 3 × total_triplet_responses.
    """
    n = len(labels)
    iu, ju = np.triu_indices(n, k=1)
    counts_upper = cnt[iu, ju]

    # Recover total triplet responses from the count matrix
    total_triplet_responses = int(round(float(counts_upper.sum()) / 3.0))

    # Combinatoric denominators
    n_pairs_possible    = n * (n - 1) // 2          # C(n, 2)
    n_triplets_possible = n * (n - 1) * (n - 2) // 6  # C(n, 3)

    # Pairwise coverage
    observed_mask    = counts_upper > 0
    n_pairs_observed = int(observed_mask.sum())
    pct_pairs_observed = (
        100.0 * n_pairs_observed / n_pairs_possible
        if n_pairs_possible > 0 else float("nan")
    )

    # % of all possible unique triplets sampled
    pct_triplets_sampled = (
        100.0 * total_triplet_responses / n_triplets_possible
        if n_triplets_possible > 0 else float("nan")
    )

    # Per-pair observation counts (observed pairs only)
    obs_per_pair = counts_upper[observed_mask]
    if len(obs_per_pair) > 0:
        mean_obs   = float(np.mean(obs_per_pair))
        median_obs = float(np.median(obs_per_pair))
        min_obs    = int(np.min(obs_per_pair))
        max_obs    = int(np.max(obs_per_pair))
    else:
        mean_obs = median_obs = float("nan")
        min_obs  = max_obs    = 0

    return {
        "n_items":                        n,
        "n_pairs_possible":               n_pairs_possible,
        "n_triplets_possible":            n_triplets_possible,
        "total_triplet_responses":        total_triplet_responses,
        "pct_triplets_sampled":           round(pct_triplets_sampled, 4),
        "n_pairs_observed":               n_pairs_observed,
        "pct_pairs_observed":             round(pct_pairs_observed, 2),
        # NOTE: obs-per-pair statistics below are inflated by fixed/dense trials
        # (fixed trials appear for every participant; dense trials oversample a
        # 30-item subset).  Do not use mean_obs as a representative coverage
        # figure for the full RDM without accounting for trial-type imbalance.
        "mean_obs_per_observed_pair":     round(mean_obs, 2),
        "median_obs_per_observed_pair":   round(median_obs, 1),
        "min_obs_per_observed_pair":      min_obs,
        "max_obs_per_observed_pair":      max_obs,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_labels(df0: pd.DataFrame) -> list[str]:
    _ensure_required(df0, ["StimA_file", "StimB_file", "StimC_file"], "summary.csv")
    return sorted(set(
        pd.concat([df0["StimA_file"].astype(str), df0["StimB_file"].astype(str), df0["StimC_file"].astype(str)])
        .map(os.path.basename).tolist()
    ))


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 0 compute: data QC and reliability")
    ap.add_argument("--pool", default="all", help="Pool filter: all | prolific | sona (default: all)")
    ap.add_argument("--fixed-dense-only", action="store_true",
                    help="Restrict analyses to fixed OR dense trials (recommended for noise ceiling)")
    ap.add_argument("--exclude-catch-ids", default="",
                    help="Comma-separated catch IDs to exclude")
    ap.add_argument("--n-iter", type=int, default=400,
                    help="Split-half iterations (default: 400)")
    ap.add_argument("--seed", type=int, default=1234, help="RNG seed (default: 1234)")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Override output directory (default: compute_outputs/stage0_data_qc/)")
    ap.add_argument("--no-loso-ceiling", action="store_true",
                    help="Skip the LOSO ceiling diagnostic (individual vs N-1 pooled). "
                         "The primary split-half ceiling is always computed.")
    args = ap.parse_args()

    exclude_ids: list[int] = []
    if str(args.exclude_catch_ids).strip():
        exclude_ids = [int(x.strip()) for x in str(args.exclude_catch_ids).split(",") if x.strip()]

    cfg = QCConfig(
        pool=str(args.pool),
        fixed_dense_only=bool(args.fixed_dense_only),
        exclude_catch_ids=exclude_ids,
        n_iter=int(args.n_iter),
        seed=int(args.seed),
    )

    if args.outdir:
        outdir = Path(args.outdir)
    elif cfg.fixed_dense_only:
        # Use a separate output directory so the fixed-dense noise ceiling does
        # not overwrite the full-dataset one consumed by the standard Stage 3 run.
        outdir = STAGE_OUTDIR.parent / "stage0_data_qc_fixed_dense"
    else:
        outdir = STAGE_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output dir: {outdir}")

    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"summary.csv not found: {SUMMARY_CSV}")

    df0 = load_summary_csv(str(SUMMARY_CSV))
    df  = filter_summary(df0, pool=cfg.pool, fixed_dense_only=cfg.fixed_dense_only, exclude_catch_ids=cfg.exclude_catch_ids)
    # When fixed_dense_only is active, derive labels only from the filtered subset
    # so the noise ceiling is computed over the same 209-item space used in Stage 5.
    # Otherwise, use the full label universe (df0) as before.
    labels = _build_labels(df if cfg.fixed_dense_only else df0)

    print(f"[INFO] Rows after filters: {len(df):,} | labels: {len(labels)}")

    # 1) Pool summary
    print("[INFO] Computing pool summary...")
    ps = pool_summary(df if "pool" in df.columns else df.assign(pool="unknown"))
    ps.to_csv(outdir / "pool_summary.csv", index=False)
    print(f"[OK]  pool_summary.csv ({len(ps)} rows)")

    # 2) RDM coverage statistics (overall + per trial type)
    print("[INFO] Computing RDM coverage statistics...")
    coverage: dict = {}
    coverage_by_type: dict = {}
    try:
        _, cnt_full = build_pairwise_from_summary(df, labels)
        coverage = rdm_coverage_stats(cnt_full, labels)

        # Per-trial-type breakdowns — use the same full label set so all
        # coverage percentages share the same denominator (all possible pairs).
        type_filters = {
            "random":     ("is_random", True),
            "fixed":      ("is_fixed",  True),
            "dense":      ("is_dense",  True),
            "fixed_dense": None,  # handled separately below
        }

        def _filter_by_col(df_in: pd.DataFrame, col: str, val: bool) -> pd.DataFrame:
            if col not in df_in.columns:
                return pd.DataFrame(columns=df_in.columns)
            return df_in[_parse_bool_series(df_in[col]) == val].copy()

        # Random-only: trials that are neither fixed nor dense.
        # NOTE: is_random may be unreliable (all-False in some data exports);
        # the safe definition is the complement of fixed and dense.
        _is_fixed_mask = _parse_bool_series(df.get("is_fixed", pd.Series([False] * len(df))))
        _is_dense_mask = _parse_bool_series(df.get("is_dense", pd.Series([False] * len(df))))
        df_random = df[~_is_fixed_mask & ~_is_dense_mask].copy()
        _, cnt_random = build_pairwise_from_summary(df_random, labels)
        coverage_by_type["random"] = rdm_coverage_stats(cnt_random, labels)

        # Fixed-only
        df_fixed = _filter_by_col(df, "is_fixed", True)
        _, cnt_fixed = build_pairwise_from_summary(df_fixed, labels)
        coverage_by_type["fixed"] = rdm_coverage_stats(cnt_fixed, labels)

        # Dense-only
        df_dense = _filter_by_col(df, "is_dense", True)
        _, cnt_dense = build_pairwise_from_summary(df_dense, labels)
        coverage_by_type["dense"] = rdm_coverage_stats(cnt_dense, labels)

        # Fixed+Dense combined (the subset used for model evaluation)
        is_fd = (
            _parse_bool_series(df.get("is_fixed", pd.Series([False] * len(df)))) |
            _parse_bool_series(df.get("is_dense", pd.Series([False] * len(df))))
        )
        df_fd = df[is_fd].copy()
        _, cnt_fd = build_pairwise_from_summary(df_fd, labels)
        coverage_by_type["fixed_dense"] = rdm_coverage_stats(cnt_fd, labels)

        # Save all to CSV
        rows = [{"trial_type": "all", **coverage}]
        for ttype, stats in coverage_by_type.items():
            rows.append({"trial_type": ttype, **stats})
        pd.DataFrame(rows).to_csv(outdir / "rdm_coverage.csv", index=False)

        # Console summary
        print(f"[OK]  rdm_coverage.csv")
        header = f"  {'trial_type':<12} {'responses':>12} {'%triplets':>10} {'pairs_obs':>10} {'%pairs':>8} {'mean_obs':>9} {'median_obs':>11}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for ttype, stats in [("all", coverage)] + list(coverage_by_type.items()):
            print(
                f"  {ttype:<12} "
                f"{stats['total_triplet_responses']:>12,} "
                f"{stats['pct_triplets_sampled']:>10.2f} "
                f"{stats['n_pairs_observed']:>10,} "
                f"{stats['pct_pairs_observed']:>8.1f} "
                f"{stats['mean_obs_per_observed_pair']:>9.1f} "
                f"{stats['median_obs_per_observed_pair']:>11.1f}"
            )
        print("  (mean/median obs per observed pair; random row is unbiased estimate of sparse coverage)")

    except Exception as e:
        print(f"[WARN] RDM coverage computation failed: {e}")

    # 3) Catch error rates
    print("[INFO] Computing catch error rates...")
    cer = catch_error_rates(df)
    cer.to_csv(outdir / "catch_error_rates.csv", index=False)
    print(f"[OK]  catch_error_rates.csv ({len(cer)} rows)")

    # 4) Split-half noise ceiling (the slow one)
    print(f"[INFO] Computing split-half noise ceiling ({cfg.n_iter} iterations)...")
    nc = split_half_noise_ceiling(df, labels=labels, n_iter=cfg.n_iter, seed=cfg.seed)
    nc.to_csv(outdir / "noise_ceiling_split_half.csv", index=False)

    if len(nc) > 0:
        nc_mean     = float(nc["rho_sb"].mean())
        nc_mean_raw = float(nc["rho"].mean())
        nc_ci       = (float(np.percentile(nc["rho_sb"], 2.5)), float(np.percentile(nc["rho_sb"], 97.5)))
        nc_pairs    = float(nc["overlap_pairs"].mean())
    else:
        nc_mean = nc_mean_raw = float("nan")
        nc_ci = (float("nan"), float("nan"))
        nc_pairs = float("nan")

    print(f"[OK]  noise ceiling (SB-corrected): mean={nc_mean:.3f}  CI95=({nc_ci[0]:.3f},{nc_ci[1]:.3f})  raw={nc_mean_raw:.3f}  iters={len(nc)}")

    # 4) Pool comparison (SONA vs Prolific) if both pools present
    pool_comp = None
    if "pool" in df0.columns:
        pools = set(df0["pool"].astype(str).str.lower().unique())
        if {"sona", "prolific"}.issubset(pools):
            print("[INFO] Computing pool comparison (SONA vs Prolific)...")
            df_pool = filter_summary(df0, pool="all", fixed_dense_only=True, exclude_catch_ids=[])
            pool_comp = pool_rdm_overlap_corr(df_pool, labels=labels, pool_a="sona", pool_b="prolific")
            print(f"[OK]  SONA vs Prolific: rho={pool_comp['rho']:.3f}  overlap_pairs={pool_comp['overlap_pairs']}")

    # 5) LOSO stability (full RDM vs LOO RDM — how much does each subject shift the group?)
    print("[INFO] Computing LOSO stability...")
    loso = loso_stability(df, labels=labels, pool=(cfg.pool if cfg.pool.lower() != "all" else None))
    loso.to_csv(outdir / "loso_stability.csv", index=False)
    if not loso.empty:
        print(f"[OK]  LOSO stability: median rho={float(loso['rho'].median()):.4f}  n_subjects={loso['SubID'].nunique()}")

    # 6) LOSO ceiling diagnostic (individual RDM vs N-1 pooled — inflated; diagnostic only)
    loso_ceil_df = pd.DataFrame()
    loso_ceil_summary: dict = {}
    if not args.no_loso_ceiling:
        print("[INFO] Computing LOSO ceiling diagnostic (individual vs N-1 pooled)...")
        loso_ceil_df = loso_ceiling(df, labels=labels)
        loso_ceil_df.to_csv(outdir / "loso_ceiling.csv", index=False)
        used = loso_ceil_df[loso_ceil_df["used"]]
        if not used.empty:
            _lc_rhos = used["rho"].to_numpy(dtype=float)
            loso_ceil_summary = {
                "mean_rho":              float(np.nanmean(_lc_rhos)),
                "ci95_lo":               float(np.nanpercentile(_lc_rhos, 2.5)),
                "ci95_hi":               float(np.nanpercentile(_lc_rhos, 97.5)),
                "n_subjects_used":       int(len(_lc_rhos)),
                "median_overlap_pairs":  float(used["n_overlap_pairs"].median()),
                "note":                  "inflated_diagnostic_only",
            }
            print(f"[OK]  LOSO ceiling: mean rho={loso_ceil_summary['mean_rho']:.3f}  "
                  f"CI95=({loso_ceil_summary['ci95_lo']:.3f},{loso_ceil_summary['ci95_hi']:.3f})  "
                  f"n={loso_ceil_summary['n_subjects_used']}  "
                  f"[diagnostic only — inflated]")
    else:
        print("[INFO] LOSO ceiling diagnostic skipped (--no-loso-ceiling).")

    # 7) Summary JSON — canonical output read by compute_model_evaluation.py
    summary = {
        "config": asdict(cfg),
        "inputs": {
            "summary_csv": str(SUMMARY_CSV),
            "n_rows_loaded": int(len(df0)),
            "n_rows_after_filters": int(len(df)),
            "n_labels": int(len(labels)),
        },
        "noise_ceiling": {
            "mean_sb":       nc_mean,
            "ci95_lo_sb":    nc_ci[0],
            "ci95_hi_sb":    nc_ci[1],
            "mean_raw":      nc_mean_raw,
            "n_iters_used":  int(len(nc)),
            "mean_overlap_pairs": nc_pairs,
            "method":        "split_half_spearman_brown",
        },
        "rdm_coverage": coverage if coverage else None,
        "rdm_coverage_by_trial_type": coverage_by_type if coverage_by_type else None,
        "pool_comparison": pool_comp,
        "catch": {
            "n_catch_rows": int(_parse_bool_series(
                df.get("is_catch", pd.Series([False] * len(df)))
            ).sum()) if "is_catch" in df.columns else 0,
            "catch_ids": cer["catch_id"].dropna().astype(int).tolist()
                if ("catch_id" in cer.columns and not cer.empty) else [],
        },
        "loso": {
            "n_subjects":           int(loso["SubID"].nunique()) if not loso.empty else 0,
            "rho_median":           float(loso["rho"].median()) if not loso.empty else float("nan"),
            "rho_p05":              float(np.percentile(loso["rho"], 5)) if not loso.empty else float("nan"),
            "rho_p95":              float(np.percentile(loso["rho"], 95)) if not loso.empty else float("nan"),
            "mean_abs_delta_median": float(loso["mean_abs_delta"].median())
                if ("mean_abs_delta" in loso.columns and not loso.empty) else float("nan"),
        },
        "loso_ceiling": loso_ceil_summary if loso_ceil_summary else None,
        "outputs": {
            "pool_summary_csv":       str(outdir / "pool_summary.csv"),
            "catch_error_rates_csv":  str(outdir / "catch_error_rates.csv"),
            "noise_ceiling_csv":      str(outdir / "noise_ceiling_split_half.csv"),
            "loso_stability_csv":     str(outdir / "loso_stability.csv"),
            "loso_ceiling_csv":       str(outdir / "loso_ceiling.csv") if not args.no_loso_ceiling else None,
            "rdm_coverage_csv":       str(outdir / "rdm_coverage.csv"),
        },
    }

    json_path = outdir / "data_quality_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK]  data_quality_summary.json")
    print(f"\n[DONE] All outputs written to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
