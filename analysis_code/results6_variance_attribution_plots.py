#!/usr/bin/env python3
"""results6_variance_attribution_plots.py

Stage 6 (Results): Conditional association (partial Spearman RSA) and Shapley variance attribution.

Reads per-source outputs from compute_variance_attribution.py and produces
publication-ready figures. Run separately for --source human and --source ai
(or --source both, the default). The mixed source run is not reported.

Inputs (per source):
  compute_outputs/stage2_model_evaluation/<rdm_mode>/<source>/manifest.json
  compute_outputs/stage4_variance_attribution/<rdm_mode>/<source>/model_unique.csv
  compute_outputs/stage4_variance_attribution/<rdm_mode>/<source>/variance_attribution_shapley.csv
  compute_outputs/stage4_variance_attribution/<rdm_mode>/<source>/variance_attribution_shapley_ci.csv

Outputs (per source):
  results/stage6_variance_attribution_plots/<rdm_mode>/<source>/fig_model_unique.png
  results/stage6_variance_attribution_plots/<rdm_mode>/<source>/fig_variance_attribution_shapley.png
  results/stage6_variance_attribution_plots/<rdm_mode>/<source>/stage6_variance_attribution_summary.md

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from reference.policy_defaults import get_compute_dir, get_results_dir

# Re-use visual styling from results3
from results3_model_evaluation_plots import (
    _pretty_model_name,
    _model_group,
    _GROUP_COLORS,
    _group_order,
    _group_marker,
    _set_pub_axes,
    _draw_group_separators,
    _coerce_numeric,
    _nn_seed_from_name,
    _policy_from_manifest,
    _model_group_with_policy,
)

# ------------------------------
# Helpers & Formatting
# ------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read CSV: {path} ({e})")
        return None


def _read_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Could not read JSON: {path} ({e})")
        return None

def _trait_key_from_model_name(raw_name: str) -> Optional[str]:
    """Map a raw model key to a comparable 'trait' label for Human vs AI comparison."""
    if raw_name is None:
        return None
    s = str(raw_name)

    if s.startswith("rdm_"):
        s = s[4:]
    if s.startswith("ai_"):
        s = s[3:]
        s = s.split("_0_100")[0]
        trait_map = {
            "spiciness": "spicy",
            "sweetness": "sweet",
            "bitterness": "bitter",
            "saltiness": "salty",
            "sourness": "sour",
            "savoriness": "savoury",
            "fatty_flavour": "fatty",
        }
        s = trait_map.get(s, s)
        return s

    if s.startswith("human_"):
        s = s[6:]
        if s.startswith("mean_"):
            s = s[5:]
        trait_map = {
            "spiciness": "spicy",
            "sweetness": "sweet",
            "bitterness": "bitter",
            "saltiness": "salty",
            "sourness": "sour",
            "savoriness": "savoury",
            "fatty_flavour": "fatty",
        }
        s = trait_map.get(s, s)
        return s

    return None

# ------------------------------
# Plotting functions
# ------------------------------

def plot_unique_forest(
    df: pd.DataFrame,
    out_png: Path,
    title: str = "Unique contribution (partial RSA)",
    top_n: int = 25,
    group_by_type: bool = True,
    policy: Optional[dict] = None,
    nn_seed: int = -1,
) -> None:
    df = df.copy()
    if "model" not in df.columns or "rho_partial" not in df.columns:
        raise ValueError("Unique df must contain columns: model, rho_partial")

    df = _coerce_numeric(df, ["rho_partial", "ci95_lo", "ci95_hi", "p_perm"])
    df = df[np.isfinite(df["rho_partial"])].copy()

    if nn_seed >= 0 and "model" in df.columns:
        keep_mask = []
        for m in df["model"].astype(str).tolist():
            grp = _model_group(m)
            if grp != "NN":
                keep_mask.append(True)
                continue
            seed_here = _nn_seed_from_name(m)
            keep_mask.append(seed_here is None or int(seed_here) == int(nn_seed))
        df = df.loc[np.asarray(keep_mask, dtype=bool)].copy()

    def is_conceptual(modelkey: str) -> bool:
        k = modelkey
        if k.startswith("rdm_"):
            k = k[4:]
        return ("vs" in k or "contrast" in k or "natural_vs" in k)
        
    include_contrasts = False
    if policy is not None:
        include_contrasts = policy.get("_include_contrasts_in_unique", False)
    if not include_contrasts:
        df = df[~df["model"].apply(is_conceptual)]
        
    if policy is not None and policy.get("selected_keys", None) is not None and policy.get("_policy_only", True):
        selected_keys_set = set(policy["selected_keys"])
        def key_in_selected(modelkey: str) -> bool:
            k = modelkey
            if k.startswith("rdm_"):
                k = k[4:]
            return (k in selected_keys_set) or (modelkey in selected_keys_set)
        df = df[df["model"].apply(key_in_selected)]

    df["_raw_model"] = df["model"].astype(str)
    if policy is not None and len(policy) > 0:
        df["_group"] = df["_raw_model"].map(lambda m: _model_group_with_policy(m, policy))
    else:
        df["_group"] = df["_raw_model"].map(_model_group)
        
    if group_by_type:
        order_map = _group_order()
        df["_gord"] = df["_group"].map(lambda g: order_map.get(str(g), 99))
        df["abs_r"] = df["rho_partial"].abs()
        df = df.sort_values(["_gord", "abs_r", "rho_partial"], ascending=[True, False, False]).drop(columns=["_gord", "abs_r"]).head(top_n)
    else:
        df["abs_r"] = df["rho_partial"].abs()
        df = df.sort_values(["abs_r", "rho_partial"], ascending=[False, False]).drop(columns=["abs_r"]).head(top_n)

    df_plot = df.copy()
    df_plot["_pretty"] = df_plot["model"].astype(str).map(_pretty_model_name)

    n = len(df_plot)
    if n == 0:
        return
        
    y_pos = np.arange(n)[::-1]
    df_plot["_y"] = y_pos

    labels_plot = df_plot["_pretty"].tolist()
    has_ci = ("ci95_lo" in df_plot.columns) and ("ci95_hi" in df_plot.columns) and np.isfinite(df_plot["ci95_lo"]).any() and np.isfinite(df_plot["ci95_hi"]).any()

    fig_h = max(5.5, 0.28 * len(labels_plot) + 1.5)
    fig = plt.figure(figsize=(10, fig_h))
    ax = fig.add_subplot(111)

    if has_ci:
        for grp in df_plot["_group"].unique().tolist():
            sub = df_plot[df_plot["_group"] == grp]
            sub_rho = sub["rho_partial"].to_numpy(dtype=float)
            sub_lo = sub["ci95_lo"].to_numpy(dtype=float)
            sub_hi = sub["ci95_hi"].to_numpy(dtype=float)
            sub_err_lo = sub_rho - sub_lo
            sub_err_hi = sub_hi - sub_rho
            grp_alpha = 0.6 if str(grp) in {"Conceptual (supp)", "Conceptual"} else 1.0
            ax.errorbar(
                sub_rho,
                sub["_y"].to_numpy(dtype=int),
                xerr=[sub_err_lo, sub_err_hi],
                fmt=_group_marker(str(grp)),
                capsize=3,
                markersize=5,
                linewidth=1,
                label=str(grp),
                alpha=grp_alpha,
            )
    else:
        for grp in df_plot["_group"].unique().tolist():
            sub = df_plot[df_plot["_group"] == grp]
            grp_alpha = 0.6 if str(grp) in {"Conceptual (supp)", "Conceptual"} else 1.0
            ax.plot(
                sub["rho_partial"].to_numpy(dtype=float),
                sub["_y"].to_numpy(dtype=int),
                _group_marker(str(grp)),
                markersize=5,
                label=str(grp),
                alpha=grp_alpha,
            )

    _draw_group_separators(ax, df_plot["_group"].astype(str).tolist(), df_plot["_y"].to_numpy(dtype=float))
    ax.axvline(0, linewidth=1, linestyle="--")
    ax.set_yticks(df_plot["_y"].to_numpy(dtype=int))
    ax.set_yticklabels(labels_plot)
    ax.set_xlabel("Partial Spearman ρ with behavioural RDM")
    ax.set_title(title)

    _set_pub_axes(ax)

    handles, labels_leg = ax.get_legend_handles_labels()
    order_map = _group_order()
    order = np.argsort([order_map.get(lbl, 99) for lbl in labels_leg])
    handles = [handles[i] for i in order]
    labels_leg = [labels_leg[i] for i in order]

    leg = ax.legend(
        handles,
        labels_leg,
        title="Family",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(5, max(1, len(labels_leg))),
        frameon=True,
    )
    leg._legend_box.align = "left"

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_unique_human_vs_ai_compare(
    df_human: pd.DataFrame,
    df_ai: pd.DataFrame,
    out_png: Path,
    top_n: int = 15,
    title: str = "Unique contribution comparison: Human-only vs AI-only",
) -> None:
    dh = df_human.copy()
    da = df_ai.copy()

    for d in (dh, da):
        d["rho_partial"] = pd.to_numeric(d.get("rho_partial"), errors="coerce")
        if "ci95_lo" in d.columns:
            d["ci95_lo"] = pd.to_numeric(d["ci95_lo"], errors="coerce")
        if "ci95_hi" in d.columns:
            d["ci95_hi"] = pd.to_numeric(d["ci95_hi"], errors="coerce")

    dh["trait"] = dh["model"].astype(str).map(_trait_key_from_model_name)
    da["trait"] = da["model"].astype(str).map(_trait_key_from_model_name)

    dh = dh[dh["trait"].notna() & np.isfinite(dh["rho_partial"])].copy()
    da = da[da["trait"].notna() & np.isfinite(da["rho_partial"])].copy()

    dh = dh.sort_values("rho_partial", ascending=False).drop_duplicates("trait")
    da = da.sort_values("rho_partial", ascending=False).drop_duplicates("trait")

    m = dh.merge(da, on="trait", how="inner", suffixes=("_human", "_ai"))
    if m.empty:
        print("[WARN] No overlapping traits found between human and ai unique results; skipping compare plot.")
        return

    m["abs_max"] = np.maximum(m["rho_partial_human"].abs(), m["rho_partial_ai"].abs())
    
    def trait_family(tr: str) -> str:
        t = str(tr)
        if t in {"healthiness", "calorie_density", "appeal", "valuation", "value"}:
            return "Food attributes"
        return "Taste"

    m["family"] = m["trait"].map(trait_family)
    
    m = m.sort_values(["family", "abs_max"], ascending=[True, False]).copy()
    m = m.head(int(top_n)).copy()

    def pretty_trait(t: str) -> str:
        tt = str(t).strip()
        if tt in ("food_classification", "food classification"):
            return "Intuitive category"
        tt = tt.replace("calorie_density", "calorie density")
        tt = tt.replace("_", " ")
        tt = " ".join(tt.split())
        return tt[:1].upper() + tt[1:] if tt else tt

    m["trait_pretty"] = m["trait"].map(pretty_trait)

    fig_h = max(5.5, 0.32 * len(m) + 1.5)
    fig = plt.figure(figsize=(10, fig_h))
    ax = fig.add_subplot(111)

    y = np.arange(len(m))[::-1]
    x_h = m["rho_partial_human"].to_numpy(dtype=float)
    x_a = m["rho_partial_ai"].to_numpy(dtype=float)

    fams_in_order = ["Food attributes", "Taste"]
    fam_counts = [int((m["family"] == f).sum()) for f in fams_in_order]
    if sum(1 for c in fam_counts if c > 0) >= 2:
        first_block = next((i for i, c in enumerate(fam_counts) if c > 0), None)
        second_block = next((i for i, c in enumerate(fam_counts) if c > 0 and i != first_block), None)
        if first_block is not None and second_block is not None:
            n_first = fam_counts[first_block]
            boundary_y = (len(m) - n_first) - 0.5
            ax.axhline(boundary_y, linewidth=0.8, alpha=0.35)
            ax.text(
                0.0,
                1.02,
                "Food attributes (top) • Taste (bottom)",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=10,
            )

    jitter = 0.14
    y_h = y + jitter
    y_a = y - jitter

    for yi_h, yi_a, x1, x2 in zip(y_h, y_a, x_h, x_a):
        ax.plot([x1, x2], [(yi_h + yi_a) / 2.0, (yi_h + yi_a) / 2.0], linewidth=1, alpha=0.35)
        ax.plot([x1, x1], [yi_a, yi_h], linewidth=0.7, alpha=0.25)

    ax.plot(x_h, y_h, "s", markersize=6, label="Human-only")
    ax.plot(x_a, y_a, "^", markersize=6, label="AI-only")

    has_ci = (
        ("ci95_lo_human" in m.columns) and ("ci95_hi_human" in m.columns)
        and ("ci95_lo_ai" in m.columns) and ("ci95_hi_ai" in m.columns)
    )
    if has_ci:
        lo_h = pd.to_numeric(m["ci95_lo_human"], errors="coerce").to_numpy(dtype=float)
        hi_h = pd.to_numeric(m["ci95_hi_human"], errors="coerce").to_numpy(dtype=float)
        lo_a = pd.to_numeric(m["ci95_lo_ai"], errors="coerce").to_numpy(dtype=float)
        hi_a = pd.to_numeric(m["ci95_hi_ai"], errors="coerce").to_numpy(dtype=float)

        ax.errorbar(x_h, y_h, xerr=[x_h - lo_h, hi_h - x_h], fmt="none", capsize=2, linewidth=1)
        ax.errorbar(x_a, y_a, xerr=[x_a - lo_a, hi_a - x_a], fmt="none", capsize=2, linewidth=1)

    ax.axvline(0, linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(m["trait_pretty"].tolist())
    ax.set_xlabel("Partial Spearman ρ")
    ax.set_title(title)

    _set_pub_axes(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        title="Unique variance run",
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shapley_bar(df: pd.DataFrame, out_png: Path, title: str = "Shapley ΔR² (mean)", value_col: str = "mean") -> None:
    d = df.copy()
    if value_col not in d.columns:
        return
    d = d[~d["predictor"].str.startswith("__")].copy()
    d = d.sort_values(value_col, ascending=False)
    if d.empty:
        return

    # Use pretty names and group colours
    d["_pretty"] = d["predictor"].astype(str).map(_pretty_model_name)
    d["_group"] = d["predictor"].astype(str).map(_model_group)
    d["_color"] = d["_group"].map(lambda g: _GROUP_COLORS.get(str(g), "#888888"))

    fig = plt.figure(figsize=(10, max(4, 0.28 * len(d) + 1)))
    ax = fig.add_subplot(111)

    y = np.arange(len(d))

    has_ci = ("ci95_lo" in d.columns) and ("ci95_hi" in d.columns)
    x = d[value_col].to_numpy(dtype=float)
    colors = d["_color"].tolist()

    if has_ci:
        err_lo = np.maximum(0, x - d["ci95_lo"].to_numpy(dtype=float))
        err_hi = np.maximum(0, d["ci95_hi"].to_numpy(dtype=float) - x)
        ax.barh(y, x, xerr=[err_lo, err_hi], color=colors, capsize=3, alpha=0.85)
    else:
        ax.barh(y, x, color=colors, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(d["_pretty"].tolist())
    ax.invert_yaxis()
    # Infer metric label from title (set at call site based on rank_transform flag)
    _ax_metric = "Δρ²" if "ρ²" in str(title) else "ΔR²"
    ax.set_xlabel(f"Shapley {_ax_metric} (incremental variance)")
    ax.set_title(title)

    _set_pub_axes(ax)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_butterfly(
    df_shapley_ci: pd.DataFrame,
    df_unique: pd.DataFrame,
    out_png: Path,
    p_thresh: float = 0.05,
    shapley_metric: str = "Δρ²",
) -> None:
    """Compact back-to-back (butterfly) chart.

    Left bars  → Shapley Δρ² (feature importance; bars grow LEFT from centre)
    Right bars → Partial Spearman ρ (conditional association; bars grow RIGHT from centre)
    Labels sit between the two panels.

    Rows: predictors with p_perm < p_thresh in the partial Spearman RSA,
    sorted by Shapley Δρ² — largest at top (y = n-1), smallest at bottom (y = 0).
    """
    import matplotlib.patches as mpatches

    # ── Merge ───────────────────────────────────────────────────────────────
    sh = df_shapley_ci[~df_shapley_ci["predictor"].str.startswith("__")].copy()
    sh = sh.rename(columns={"mean": "shapley_mean",
                             "ci95_lo": "sh_lo", "ci95_hi": "sh_hi"})

    un = df_unique.copy()
    un = un.rename(columns={"model": "predictor"})

    df = sh.merge(un[["predictor", "rho_partial", "ci95_lo", "ci95_hi", "p_perm"]],
                  on="predictor", how="inner")

    # ── Drop predictors excluded from the paper (dropped from analysis_helpers.VISUAL
    #    but may still appear in older compute outputs)
    _PLOT_EXCLUDE: set[str] = {"ll_edge_energy"}
    df = df[~df["predictor"].isin(_PLOT_EXCLUDE)].copy()

    # ── Labels & colours ─────────────────────────────────────────────────────
    df["_pretty"] = df["predictor"].map(_pretty_model_name)
    df["_group"]  = df["predictor"].map(_model_group)
    df["_color"]  = df["_group"].map(lambda g: _GROUP_COLORS.get(str(g), "#888888"))

    # ── Filter predictors for display ───────────────────────────────────────
    # We include all non-visual predictors, but restrict visual predictors to 
    # the core 4 used in the model visualisation plot.
    # Accept both bare ll_ names and rdm_ll_ prefixed names (pairwise_vectors uses rdm_ll_)
    core_visual = {
        "ll_resnet50_layer3_pcs10", "rdm_ll_resnet50_layer3_pcs10",
        "ll_resnet50_conv1_pcs10",  "rdm_ll_resnet50_conv1_pcs10",
        "ll_hsv_s_mean",            "rdm_ll_hsv_s_mean",
        "ll_lbp_hist",              "rdm_ll_lbp_hist",
    }

    def keep_predictor(row):
        if row["_group"] == "Visual":
            return row["predictor"] in core_visual
        return True
        
    df = df[df.apply(keep_predictor, axis=1)].copy()

    if df.empty:
        print(f"[WARN] plot_butterfly: no predictors matched filter criteria.")
        return

    # ── Sort: largest Shapley at top → ascending so row 0 = bottom (smallest) ──
    df = df.sort_values("shapley_mean", ascending=True).reset_index(drop=True)

    n      = len(df)
    y      = np.arange(n)          # y=0 → bottom row (smallest shapley), y=n-1 → top
    colors = df["_color"].tolist()  # colors[i] matches y[i]

    # ── Layout: two panels, labels on RIGHT side of left panel ───────────────
    fig_h = max(4.0, n * 0.45 + 1.8)
    # 3-column layout: [left bars | centre labels | right bars]
    fig = plt.figure(figsize=(11, fig_h))
    gs  = fig.add_gridspec(1, 3, wspace=0,
                           width_ratios=[4, 1.8, 4])
    ax_l   = fig.add_subplot(gs[0])
    ax_mid = fig.add_subplot(gs[1])
    ax_r   = fig.add_subplot(gs[2])

    BAR_H = 0.6

    # ── Left panel: Shapley Δρ² bars grow LEFTWARD from zero at right edge ───
    sh_vals  = df["shapley_mean"].to_numpy(dtype=float)
    sh_lo_err = (sh_vals - df["sh_lo"].to_numpy(dtype=float)).clip(min=0)
    sh_hi_err = (df["sh_hi"].to_numpy(dtype=float) - sh_vals).clip(min=0)

    # Bars as negative values so they extend left; right edge is x=0
    ax_l.barh(y, -sh_vals, color=colors, alpha=0.85, height=BAR_H)
    ax_l.errorbar(-sh_vals, y,
                  xerr=[sh_lo_err, sh_hi_err],
                  fmt="none", color="#333333", linewidth=0.9, capsize=2.5)

    ax_l.axvline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=0)
    max_sh = sh_vals.max() * 1.22
    ax_l.set_xlim(-max_sh, 0)
    # Tick labels as positive numbers (negate stored tick positions)
    n_ticks = 4
    tick_pos = -np.linspace(0, sh_vals.max(), n_ticks)
    ax_l.set_xticks(tick_pos)
    ax_l.set_xticklabels([f"{abs(v):.3f}" for v in tick_pos], fontsize=8)
    ax_l.set_xlabel(f"Shapley {shapley_metric}", fontsize=9)

    # No y-tick labels on left panel — labels go in the centre column
    ax_l.set_yticks(y)
    ax_l.set_yticklabels([])
    ax_l.tick_params(axis="y", left=False, right=False)

    ax_l.set_title(f"Feature importance\n(Shapley {shapley_metric})", fontsize=9, pad=6)
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["left"].set_visible(False)
    ax_l.spines["right"].set_visible(False)

    # ── Centre column: labels only ────────────────────────────────────────────
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(-0.5, n - 0.5)
    for spine in ax_mid.spines.values():
        spine.set_visible(False)
    ax_mid.tick_params(left=False, right=False, bottom=False, top=False,
                       labelleft=False, labelright=False, labelbottom=False)
    for i, (label, col) in enumerate(zip(df["_pretty"].tolist(), colors)):
        ax_mid.text(0.5, i, label, ha="center", va="center",
                    fontsize=9, color=col)

    # ── Right panel: Partial Spearman ρ bars grow RIGHTWARD from zero ────────
    rho_vals = df["rho_partial"].to_numpy(dtype=float)
    rho_lo   = rho_vals - df["ci95_lo"].to_numpy(dtype=float)
    rho_hi   = df["ci95_hi"].to_numpy(dtype=float) - rho_vals

    ax_r.barh(y, rho_vals, color=colors, alpha=0.85, height=BAR_H)
    ax_r.errorbar(rho_vals, y,
                  xerr=[rho_lo, rho_hi],
                  fmt="none", color="#333333", linewidth=0.9, capsize=2.5)
    ax_r.axvline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=0)

    neg_extent = min(rho_vals.min(), 0)
    pos_extent = rho_vals.max()
    ax_r.set_xlim(neg_extent * 1.4 - 0.01, pos_extent * 1.25)
    ax_r.set_xlabel("Partial Spearman ρ", fontsize=9)
    ax_r.tick_params(axis="x", labelsize=8)

    ax_r.set_yticks(y)
    ax_r.set_yticklabels([])
    ax_r.tick_params(axis="y", left=False, right=False)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    ax_r.spines["left"].set_visible(False)
    ax_r.set_ylim(-0.5, n - 0.5)
    ax_l.set_ylim(-0.5, n - 0.5)

    ax_r.set_title("Conditional association\n(partial Spearman ρ)", fontsize=9, pad=6)

    # ── Legend ───────────────────────────────────────────────────────────────
    seen: dict = {}
    for grp, col in zip(df["_group"].tolist(), colors):
        if grp not in seen:
            seen[grp] = col
    handles = [mpatches.Patch(color=c, label=g) for g, c in seen.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote butterfly figure: {out_png}")


# ------------------------------
# CLI
# ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rdm-mode",
        type=str,
        default="empirical",
        choices=["empirical", "filled", "fixed_dense"],
        help="Whether to read from the empirical, filled, or fixed_dense RDM outputs (default: empirical)",
    )
    p.add_argument(
        "--source",
        type=str,
        default="both",
        choices=["human", "ai", "mixed", "both"],
        help="Model source to evaluate: human, ai, mixed, or both (default: both)",
    )
    p.add_argument("--top-n", type=int, default=25, help="How many models to display in plots (default 25)")
    p.add_argument(
        "--nn-seed",
        type=int,
        default=None,
        help="Which NN seed to show in plots/heatmap (-1 = all seeds). Default: read from manifest.json if available.",
    )
    p.add_argument(
        "--no-group-by-type",
        action="store_true",
        help="Do not group y-axis by model type; just sort by rho.",
    )
    p.add_argument(
        "--compare-unique",
        action="store_true",
        help=(
            "Create an additional figure comparing unique-variance (partial rho) for matched traits "
            "between Human-only and AI-only runs."
        ),
    )
    p.add_argument(
        "--no-ai",
        action="store_true",
        default=False,
        help="Restrict output to human source only; skip AI and Human vs AI comparison figures.",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.no_ai:
        sources = ["human"]
    elif args.source == "both":
        sources = ["human", "ai"]
    else:
        sources = [args.source]

    for source in sources:
        # 1. Gather partial mantel inputs
        indir_stage2 = get_compute_dir("stage2_model_evaluation", args.rdm_mode, source)
        # 2. Gather shapley inputs
        # For shapley, we typically use the same mode/source
        indir_stage4 = get_compute_dir("stage4_variance_attribution", args.rdm_mode, source)

        outdir = get_results_dir("stage6_variance_attribution_plots", args.rdm_mode, source)
        _ensure_dir(outdir)

        unique_path = indir_stage4 / "model_unique.csv"
        mani_path = indir_stage2 / "manifest.json"
        shapley_mani_path = indir_stage4 / "variance_attribution_manifest.json"

        shapley_path = indir_stage4 / "variance_attribution_shapley.csv"
        shapley_ci_path = indir_stage4 / "variance_attribution_shapley_ci.csv"

        df_unique = _read_csv_if_exists(unique_path)
        manifest = _read_json_if_exists(mani_path)
        shapley_manifest = _read_json_if_exists(shapley_mani_path)
        # Determine correct metric label for Shapley outputs
        _rank_transform = bool((shapley_manifest or {}).get("rank_transform", False))
        _shapley_metric = "Δρ²" if _rank_transform else "ΔR²"

        df_shapley = _read_csv_if_exists(shapley_path)
        df_shapley_ci = _read_csv_if_exists(shapley_ci_path)

        # Reconcile NN seed / policy
        resolved_nn_seed = -1
        if args.nn_seed is not None:
            resolved_nn_seed = int(args.nn_seed)
        else:
            try:
                resolved_nn_seed = int(manifest.get("selected_nn_seed")) if manifest is not None and manifest.get("selected_nn_seed") is not None else -1
            except Exception:
                resolved_nn_seed = -1

        policy = _policy_from_manifest(manifest) if manifest else {}

        # --- 1. Plot partial Spearman RSA (model_unique.csv) ---
        if df_unique is not None and not df_unique.empty:
            out_png = outdir / "fig_model_unique.png"
            plot_unique_forest(
                df_unique,
                out_png,
                title=f"Unique contribution (partial RSA) — {source}",
                top_n=int(args.top_n),
                group_by_type=(not args.no_group_by_type),
                policy=policy if policy else None,
                nn_seed=int(resolved_nn_seed),
            )
            print(f"[OK] Wrote: {out_png}")
        else:
            print(f"[NOTE] {source}: model_unique.csv not found (or empty) — skipping partial Spearman plot")

        # --- 2. Plot Human vs AI compare (if requested) ---
        if bool(args.compare_unique):
            df_h = _read_csv_if_exists(get_compute_dir("stage4_variance_attribution", args.rdm_mode, "human") / "model_unique.csv")
            df_a = _read_csv_if_exists(get_compute_dir("stage4_variance_attribution", args.rdm_mode, "ai") / "model_unique.csv")

            if df_h is None or df_h.empty or df_a is None or df_a.empty:
                print("[NOTE] --compare-unique requested, but could not find both human and ai model_unique.csv.")
            else:
                out_png = outdir / "fig_unique_human_vs_ai.png"
                plot_unique_human_vs_ai_compare(df_h, df_a, out_png, top_n=min(15, int(args.top_n)))
                print(f"[OK] Wrote: {out_png}")

        # --- 3. Plot Shapley variance attribution ---
        if df_shapley is not None and not df_shapley.empty:
            # Merge CI if it exists
            if df_shapley_ci is not None and not df_shapley_ci.empty:
                df_s = df_shapley.merge(df_shapley_ci[["predictor", "ci95_lo", "ci95_hi"]], on="predictor", how="left")
            else:
                df_s = df_shapley.copy()

            out_png = outdir / "fig_variance_attribution_shapley.png"
            plot_shapley_bar(df_s, out_png, title=f"Shapley {_shapley_metric} (mean) — {source}")
            print(f"[OK] Wrote: {out_png}")
        else:
            print(f"[NOTE] {source}: variance_attribution_shapley.csv not found — skipping Shapley plot")

        # --- 3b. Primary butterfly chart (Fig 6 — Shapley left, partial ρ right) ---
        # Reads from the same pairwise_vectors-based compute outputs as the bar plots
        # above. This is the combined Figure 6 as described in the manuscript.
        if (source == "human"
                and df_shapley_ci is not None and not df_shapley_ci.empty
                and df_unique is not None and not df_unique.empty):
            out_png = outdir / "fig_shapley_butterfly.png"
            plot_butterfly(df_shapley_ci, df_unique, out_png,
                           p_thresh=0.05, shapley_metric=_shapley_metric)
        elif source == "human":
            print(f"[NOTE] {source}: Shapley CI or partial ρ missing — skipping primary butterfly chart")

        # --- 3c. Full-feature butterfly (all 11 visual features; supplementary) ---
        full_unique_path     = HERE / "compute_outputs" / "stage4_variance_attribution_full" / args.rdm_mode / source / "model_unique.csv"
        full_shapley_ci_path = HERE / "compute_outputs" / "stage4_variance_attribution_full" / args.rdm_mode / source / "variance_attribution_shapley_ci.csv"
        df_full_unique    = _read_csv_if_exists(full_unique_path)
        df_full_shapley_ci = _read_csv_if_exists(full_shapley_ci_path)
        if df_full_unique is not None and df_full_shapley_ci is not None:
            full_out_dir = HERE / "results" / "stage6_variance_attribution_full" / args.rdm_mode / source
            full_out_dir.mkdir(parents=True, exist_ok=True)
            out_png = full_out_dir / "fig_shapley_butterfly.png"
            plot_butterfly(df_full_shapley_ci, df_full_unique, out_png,
                           p_thresh=0.05, shapley_metric=_shapley_metric)
        else:
            print(f"[NOTE] Full-feature data not found for source={source} at rdm_mode={args.rdm_mode} — skipping full-feature butterfly chart")

        # --- 4. Write simple summary markdown ---
        lines = [f"# Stage 6 — Unique Variance Attribution ({source})\n\n"]

        if df_unique is not None and not df_unique.empty:
            df = df_unique.copy()
            df = _coerce_numeric(df, ["rho_partial", "ci95_lo", "ci95_hi", "p_perm"])
            df = df[np.isfinite(df["rho_partial"])].copy()
            df["abs_r"] = df["rho_partial"].abs()
            df = df.sort_values(["abs_r", "rho_partial"], ascending=[False, False]).drop(columns=["abs_r"]).head(args.top_n)

            lines.append("## Conditional associations (partial Spearman RSA)\n")
            lines.append("| Model | partial ρ | CI95 lo | CI95 hi | p |\n")
            lines.append("|---|---:|---:|---:|---:|\n")
            for _, r in df.iterrows():
                m = _pretty_model_name(r.get("model", ""))
                rho = r.get("rho_partial", np.nan)
                lo = r.get("ci95_lo", np.nan)
                hi = r.get("ci95_hi", np.nan)
                p = r.get("p_perm", np.nan)
                lines.append(
                    "| {} | {} | {} | {} | {} |\n".format(
                        m,
                        f"{float(rho):.3f}" if np.isfinite(rho) else "",
                        f"{float(lo):.3f}" if np.isfinite(lo) else "",
                        f"{float(hi):.3f}" if np.isfinite(hi) else "",
                        f"{float(p):.3g}" if np.isfinite(p) else "",
                    )
                )
            lines.append("\n")

        if df_shapley is not None and not df_shapley.empty:
            lines.append(f"## Shapley variance attribution ({_shapley_metric})\n")
            lines.append(f"| Predictor | {_shapley_metric} (mean) |\n")
            lines.append("|---|---:|\n")
            df_s = df_shapley[~df_shapley["predictor"].str.startswith("__")].copy()
            df_s = df_s.sort_values("mean", ascending=False).head(args.top_n)
            for _, r in df_s.iterrows():
                m = _pretty_model_name(r.get("predictor", ""))
                mean_val = r.get("mean", np.nan)
                lines.append(
                    "| {} | {} |\n".format(
                        m,
                        f"{float(mean_val):.4f}" if np.isfinite(mean_val) else "",
                    )
                )
            lines.append("\n")

        out_md = outdir / "stage6_variance_attribution_summary.md"
        out_md.write_text("".join(lines), encoding="utf-8")
        print(f"[OK] Wrote: {out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
