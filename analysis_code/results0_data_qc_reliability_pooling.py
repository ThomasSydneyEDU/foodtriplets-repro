#!/usr/bin/env python3
"""results0_data_qc_reliability_pooling.py

Step 0 (Results): Data quality / reliability plots.

What this script does
---------------------
Reads pre-computed outputs from ``compute_data_qc.py`` and generates
publication-quality figures.  All heavy computation lives in that script;
this one is plotting-only.

Outputs two figures intended for the Supplementary Materials:
  1. Split-half reliability distribution (basis for the noise ceiling reference line)
  2. LOSO ceiling (individual vs N-1 group; shows range of inter-individual variability)

Note: The LOSO stability figure (full vs leave-one-out, ρ ≈ 0.999) is intentionally
omitted — with N=294 participants, removing one participant has a trivially small effect
on the group RDM and the figure communicates nothing useful. Catch-trial error rates
are also omitted (used for participant screening only; not reported in the paper).

Required inputs (written by compute_data_qc.py)
------------------------------------------------
  compute_outputs/stage0_data_qc/data_quality_summary.json
  compute_outputs/stage0_data_qc/pool_summary.csv
  compute_outputs/stage0_data_qc/noise_ceiling_split_half.csv
  compute_outputs/stage0_data_qc/loso_ceiling.csv   (optional)

Outputs
-------
  results/stage0_data_qc_reliability_pooling/
    noise_ceiling_split_half.png
    loso_ceiling.png          (if loso_ceiling.csv present)
    pool_summary.csv          (copied from compute stage)
    noise_ceiling_split_half.csv  (copied from compute stage)
    provenance.json

Example
-------
  python analysis_code/results0_data_qc_reliability_pooling.py
  python analysis_code/results0_data_qc_reliability_pooling.py --no-plots
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from reference.policy_defaults import (
        COMPUTE_DIR as POLICY_COMPUTE_DIR,
        RESULTS_DIR as POLICY_RESULTS_DIR,
        stage_outdir as policy_stage_outdir,
    )
except Exception:
    POLICY_COMPUTE_DIR = None
    POLICY_RESULTS_DIR = None
    policy_stage_outdir = None

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _get_compute_stage_dir(override: str | None = None) -> Path:
    if override:
        return Path(override)
    if POLICY_COMPUTE_DIR is not None:
        return Path(POLICY_COMPUTE_DIR) / "stage0_data_qc"
    return HERE / "compute_outputs" / "stage0_data_qc"


def _get_results_stage_dir() -> Path:
    if policy_stage_outdir is not None:
        out = Path(policy_stage_outdir("stage0_data_qc_reliability_pooling"))
    elif POLICY_RESULTS_DIR is not None:
        out = Path(POLICY_RESULTS_DIR) / "stage0_data_qc_reliability_pooling"
    else:
        out = HERE / "results" / "stage0_data_qc_reliability_pooling"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_catch_error_rates(df_rates: pd.DataFrame, out_png: Path) -> None:
    if df_rates.empty or "catch_id" not in df_rates.columns:
        print("[WARN] catch_error_rates: empty or missing catch_id — skipping plot")
        return
    d = df_rates.dropna(subset=["error_rate"]).sort_values("error_rate", ascending=False)
    if d.empty:
        return

    fig = plt.figure(figsize=(7, max(2.5, 0.35 * len(d) + 1)))
    ax = fig.add_subplot(111)
    y = np.arange(len(d))
    ax.barh(y, d["error_rate"].to_numpy())
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"catch_{int(x)}" if pd.notna(x) else "catch" for x in d["catch_id"].tolist()]
    )
    ax.set_xlabel("Error rate")
    ax.set_title("Catch trial error rates (group-level)")
    ax.set_xlim(0, max(0.15, float(d["error_rate"].max() * 1.1)))
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_png.name}")


def plot_noise_ceiling(df_nc: pd.DataFrame, out_png: Path) -> None:
    if df_nc.empty:
        print("[WARN] noise_ceiling: empty — skipping plot")
        return
    col = "rho_sb" if "rho_sb" in df_nc.columns else "rho"
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(111)
    ax.hist(df_nc[col].dropna().to_numpy(), bins=25)
    ax.set_xlabel("Split-half Spearman ρ (Spearman-Brown corrected)" if col == "rho_sb" else "Split-half Spearman ρ")
    ax.set_ylabel("Count")
    ax.set_title("Split-half reliability distribution")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_png.name}")


def plot_loso(df_loso: pd.DataFrame, out_png: Path, title: str = "LOSO stability distribution") -> None:
    if df_loso.empty or "rho" not in df_loso.columns:
        print(f"[WARN] loso: empty or missing rho column — skipping {out_png.name}")
        return
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(111)
    ax.hist(df_loso["rho"].dropna().to_numpy(), bins=25)
    ax.set_xlabel("ρ(full, leave-one-out)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_png.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Results step 0: data QC and reliability plots (reads from compute_data_qc.py outputs)"
    )
    ap.add_argument(
        "--compute-dir",
        default=None,
        help="Override path to compute_outputs/stage0_data_qc/ directory",
    )
    ap.add_argument("--no-plots", action="store_true", help="Skip figure generation (CSV copying still runs)")
    args = ap.parse_args()

    compute_dir = _get_compute_stage_dir(args.compute_dir)
    qc_json = compute_dir / "data_quality_summary.json"

    if not qc_json.exists():
        print(
            f"[ERROR] Required file not found: {qc_json}\n"
            "        Run compute_data_qc.py first to generate QC outputs.",
            file=sys.stderr,
        )
        return 1

    # Load canonical JSON
    with open(qc_json, encoding="utf-8") as fh:
        cjson = json.load(fh)

    # Resolve CSV paths — prefer 'outputs' dict in JSON, fall back to defaults
    outputs = cjson.get("outputs", {})

    def _csv(key: str, fallback: str) -> Path:
        v = outputs.get(key)
        return Path(v) if v else compute_dir / fallback

    pool_csv  = _csv("pool_summary_csv",        "pool_summary.csv")
    catch_csv = _csv("catch_error_rates_csv",    "catch_error_rates.csv")
    nc_csv    = _csv("noise_ceiling_csv",        "noise_ceiling_split_half.csv")
    loso_ceil_csv = _csv("loso_ceiling_csv",     "loso_ceiling.csv")

    # Check mandatory CSVs
    missing = [p for p in (pool_csv, nc_csv) if not p.exists()]
    if missing:
        print(
            "[ERROR] Some expected compute outputs are missing:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\n        Re-run compute_data_qc.py to regenerate them.",
            file=sys.stderr,
        )
        return 1

    # Load DataFrames
    ps        = pd.read_csv(pool_csv)
    nc        = pd.read_csv(nc_csv)
    loso_ceil = pd.read_csv(loso_ceil_csv) if loso_ceil_csv.exists() else pd.DataFrame()

    # Results output directory
    outdir = _get_results_stage_dir()
    print(f"[INFO] Writing results to: {outdir}")

    # Copy CSVs into results dir so it is self-contained
    for src, name in [
        (pool_csv, "pool_summary.csv"),
        (nc_csv,   "noise_ceiling_split_half.csv"),
    ]:
        shutil.copy2(src, outdir / name)

    # Plots
    if not args.no_plots:
        plot_noise_ceiling(nc, outdir / "noise_ceiling_split_half.png")
        if not loso_ceil.empty:
            plot_loso(loso_ceil, outdir / "loso_ceiling.png",
                      title="LOSO ceiling (individual vs N-1 pooled)")

    # Provenance JSON
    nc_blob       = cjson.get("noise_ceiling", {})
    loso_ceil_blob = cjson.get("loso_ceiling", {})
    pool_comp     = cjson.get("pool_comparison", {})

    provenance = {
        "generated": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "compute_source": str(compute_dir),
        "noise_ceiling": {
            "mean_sb":      nc_blob.get("mean_sb"),
            "ci95_lo_sb":   nc_blob.get("ci95_lo_sb"),
            "ci95_hi_sb":   nc_blob.get("ci95_hi_sb"),
            "mean_raw":     nc_blob.get("mean_raw"),
            "n_iters_used": nc_blob.get("n_iters_used", len(nc)),
            "method":       "split_half_spearman_brown",
        },
        "pool_comparison": pool_comp if pool_comp else None,
        "loso_ceiling": loso_ceil_blob if loso_ceil_blob else None,
        "outputs": {
            "pool_summary_csv":  str(outdir / "pool_summary.csv"),
            "noise_ceiling_csv": str(outdir / "noise_ceiling_split_half.csv"),
        },
    }

    prov_path = outdir / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    # Console summary
    print("=== Results0: Data quality / reliability ===")
    print(f"  Source: {compute_dir}")
    if nc_blob:
        print(
            f"  Noise ceiling (SB-corrected): "
            f"mean={nc_blob.get('mean_sb', float('nan')):.3f}  "
            f"CI95=({nc_blob.get('ci95_lo_sb', float('nan')):.3f}, "
            f"{nc_blob.get('ci95_hi_sb', float('nan')):.3f})  "
            f"iters={nc_blob.get('n_iters_used', len(nc))}"
        )
    if pool_comp:
        print(
            f"  Pool comparison (SONA vs Prolific): ρ={pool_comp.get('rho', float('nan')):.3f}  "
            f"overlap_pairs={pool_comp.get('overlap_pairs', '?')}"
        )
    print(f"[OK] Wrote results to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
