#!/usr/bin/env python3
"""results5_family_generalization_plots.py

Stage 5 (Results): Group-wise / family-blocked CV regression plots.

Reads outputs from compute_family_generalization.py and produces a
publication-ready figure summarising how much variance each model family
(and family combination) explains in the held-out empirical RDM.

Inputs:
  compute_outputs/stage5_family_generalization/
  - cv_family_regression_fits.csv   (per-fold rows)
  - cv_family_summary.csv           (mean ± SD across folds)

Outputs:
  results/stage5_family_generalization_plots/
  - fig_family_cv_rho.png           horizontal bar chart, ρ ± 95% CI
  - fig_family_cv_rho_compact.png   two-panel publication figure when --compact is used
  - stage5_family_generalization_summary.md

Usage:
  python analysis_code/results5_family_generalization_plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import argparse

HERE = Path(__file__).resolve().parent

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rdm-mode",
        type=str,
        default="fixed_dense",
        choices=["empirical", "fixed_dense", "filled"],
        help="RDM mode for input/output resolution (default: fixed_dense)",
    )
    p.add_argument(
        "--include-ai",
        action="store_true",
        default=False,
        help="Include AI-only family rows (ai_*) on the bar chart.",
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help=("Produce the compact two-panel publication view: family fits on the left "
              "and paired fold-wise increments on the right."),
    )
    p.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Optional stage5_family_generalization input root (before the RDM-mode subfolder).",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Optional output directory. Useful for isolated experimental figures.",
    )
    return p.parse_args()

INPUT_DIR  = HERE / "compute_outputs" / "stage5_family_generalization"
RESULTS_DIR = HERE / "results" / "stage5_family_generalization_plots"

# ---------------------------------------------------------------------------
# Visual design
# ---------------------------------------------------------------------------

# Group palette — mirrors the scheme used in results3/4
_GROUP_COLORS: dict[str, str] = {
    "AI":                      "#ff7f00",   # ColorBrewer orange (AI ratings)
    "Human":                   "#377eb8",   # ColorBrewer blue (Human ratings)
    "Visual":                  "#4daf4a",   # ColorBrewer green (visual)
    "Intuitive category":      "#e41a1c",   # ColorBrewer red (categorical)
    "NOVA Processing Level":   "#984ea3",   # ColorBrewer purple (processing level)
    # REVISION 2026-09-10: transformation is no longer NOVA alone. All three
    # operationalisations share the purple family so the section reads as one
    # construct tested three ways.
    "Transformation":          "#984ea3",   # purple (all transformation measures)
    "Alternative taxonomy":    "#f4a3a2",   # muted red — same construct as category, different partition
}

# Display order (top → bottom on the horizontal bar chart)
# Narrative: Visual (floor) → Semantic (AI then Human) → NOVA modest → Category
_BLOCK_ORDER: list[str] = [
    # ── Visual (floor) ─────────────────────────────────────────────
    "visual",
    # ── AI semantic families ────────────────────────────────────────
    "ai_appraisal",
    "ai_flavour",
    "ai_appraisal+ai_flavour",
    "ai_full",
    # ── Human semantic families ─────────────────────────────────────
    "human_appraisal",
    "human_flavour",
    "human_appraisal+human_flavour",
    "human_full",
    # ── Transformation: three operationalisations, all modest ───────
    # REVISION 2026-09-10 (revision/transformation-models). Any block key absent
    # from this list is SILENTLY DROPPED from the figure (see the filter at
    # `order = [k for k in _BLOCK_ORDER if k in present]`), so new BLOCKS entries
    # in compute_family_generalization.py must be added here too. The consistency
    # check in test_model_policy_consistency.py enforces this.
    "NOVA_ordinal",
    "natural_vs_transformed",
    "Transformation_score",
    "transformation",
    "human_full+NOVA_ordinal",
    "human_full+food_classification+natural_vs_transformed",
    "human_full+food_classification+Transformation_score",
    "human_full+food_classification+transformation",
    "ai_full+NOVA_ordinal",
    # ── Category: what actually adds variance ───────────────────────
    "food_classification",
    "human_full+food_classification",
    "ai_full+food_classification",
    # ── Alternative taxonomies (symmetric reporting; RB-4a) ─────────
    "Category_WHO_10",
    "Category_Culinary_9",
    "human_full+Category_WHO_10",
    "human_full+Category_Culinary_9",
]

# Blocks that ARE computed by compute_family_generalization.py and written to
# cv_family_regression_fits.csv, but are deliberately NOT drawn in the main
# figure. Declared explicitly (rather than by omission from _BLOCK_ORDER) so the
# distinction between "left out on purpose" and "forgotten" is reviewable, and so
# test_model_policy_consistency.py can enforce that every computed block is
# accounted for one way or the other.
#
# The first nine are pre-existing: intermediate two-family combinations that the
# figure's narrative does not need. The last two were added on
# revision/transformation-models: they are the taxonomy x transformation
# robustness cross-checks, which belong in the supplementary table rather than
# the main figure.
# AI entries removed 2026-09-11 with the AI rating families themselves; the
# consistency test flags stale entries here, which is how this was caught.
_INTENTIONALLY_NOT_PLOTTED: set[str] = {
    "human_appraisal+visual",
    "human_flavour+visual",
    "food_classification+visual",
    "human_appraisal+human_flavour+food_classification",
    "human_full+food_classification+NOVA_ordinal",
    # No-category transformation contrast; held for the simplified targeted
    # figure rather than added to the legacy all-block figure.
    "human_full+transformation",
    "human_full+Category_WHO_10+transformation",
    "human_full+Category_Culinary_9+transformation",
}

# Keys that begin a new figure section (thicker separator line drawn above)
_SECTION_STARTS = {
    "ai_appraisal",
    "human_appraisal",
    "NOVA_ordinal",
    "food_classification",
    "Category_WHO_10",
}

# Map block key → source group for colouring
_BLOCK_GROUP: dict[str, str] = {
    # Visual
    "visual":                              "Visual",
    # AI semantic
    "ai_appraisal":                        "AI",
    "ai_flavour":                          "AI",
    "ai_appraisal+ai_flavour":             "AI",
    "ai_full":                             "AI",
    # Human semantic
    "human_appraisal":                     "Human",
    "human_flavour":                       "Human",
    "human_appraisal+human_flavour":       "Human",
    "human_full":                          "Human",
    # NOVA processing level — all NOVA bars (including combinations) are purple
    "NOVA_ordinal":                        "Transformation",
    "human_full+NOVA_ordinal":             "Transformation",
    "ai_full+NOVA_ordinal":                "Transformation",
    "natural_vs_transformed":              "Transformation",
    "Transformation_score":                "Transformation",
    "transformation":                      "Transformation",
    "human_full+food_classification+natural_vs_transformed": "Transformation",
    "human_full+food_classification+Transformation_score":   "Transformation",
    "human_full+food_classification+transformation":         "Transformation",
    # Category — all category bars (including combinations) are red
    "food_classification":                 "Intuitive category",
    "human_full+food_classification":      "Intuitive category",
    "ai_full+food_classification":         "Intuitive category",
    # Alternative taxonomies
    "Category_WHO_10":                     "Alternative taxonomy",
    "Category_Culinary_9":                 "Alternative taxonomy",
    "human_full+Category_WHO_10":          "Alternative taxonomy",
    "human_full+Category_Culinary_9":      "Alternative taxonomy",
}

# Override display labels — use explicit family names throughout, no "Full"
_LABEL_OVERRIDES: dict[str, str] = {
    "ai_full":                           "AI Appraisal + Sensory + Visual",
    "human_full":                        "Human Appraisal + Sensory + Visual",
    "human_full+NOVA_ordinal":           "Human Appraisal + Sensory + Visual + NOVA",
    "ai_full+NOVA_ordinal":              "AI Appraisal + Sensory + Visual + NOVA",
    "food_classification":               "Intuitive category",
    "human_full+food_classification":    "Human Appraisal + Sensory + Visual + Category",
    "ai_full+food_classification":       "AI Appraisal + Sensory + Visual + Category",
    # REVISION 2026-09-10 — transformation measures named by construct, not by
    # instrument, so the figure does not read as three attempts at the same thing.
    "NOVA_ordinal":                      "Industrial processing (NOVA)",
    "natural_vs_transformed":            "Whole food vs composite dish",
    "Transformation_score":              "Graded culinary transformation",
    "transformation":                    "Transformation (all three)",
    "human_full+food_classification+natural_vs_transformed":
        "Human Appraisal + Sensory + Visual + Category + Whole food vs dish",
    "human_full+food_classification+Transformation_score":
        "Human Appraisal + Sensory + Visual + Category + Graded transformation",
    "human_full+food_classification+transformation":
        "Human Appraisal + Sensory + Visual + Category + Transformation (all three)",
    "Category_WHO_10":                   "Category (WHO 10)",
    "Category_Culinary_9":               "Category (Culinary 9)",
    "human_full+Category_WHO_10":        "Human Appraisal + Sensory + Visual + Category (WHO 10)",
    "human_full+Category_Culinary_9":    "Human Appraisal + Sensory + Visual + Category (Culinary 9)",
}

# Hatched for combined/multi-family blocks
_COMBINED_KEYS = {
    "ai_appraisal+ai_flavour",
    "ai_full",
    "human_appraisal+human_flavour",
    "human_full",
    "human_full+NOVA_ordinal",
    "ai_full+NOVA_ordinal",
    "human_full+food_classification",
    "ai_full+food_classification",
    "transformation",
    "human_full+food_classification+natural_vs_transformed",
    "human_full+food_classification+Transformation_score",
    "human_full+food_classification+transformation",
    "human_full+Category_WHO_10",
    "human_full+Category_Culinary_9",
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_family_bars(
    df_folds: pd.DataFrame,
    out_png: Path,
    title: str = "Group-wise CV regression (Spearman ρ on held-out RDM)",
    noise_ceiling: float | None = None,
) -> None:
    """Horizontal bar chart of mean ρ ± 95% CI per family block."""

    present = set(df_folds["model_set"].unique())
    order = [k for k in _BLOCK_ORDER if k in present]

    # Compute mean and 95% CI (t-based) per block
    stats: dict[str, dict] = {}
    k_folds = df_folds["fold"].nunique()
    for key in order:
        sub = df_folds[df_folds["model_set"] == key]["rho_test"].dropna()
        n = len(sub)
        mean = float(sub.mean())
        # 95% CI using t-distribution (conservative for small k)
        if n > 1:
            from math import sqrt
            # t critical for 95% two-tailed, df = n-1
            # For k=5: t(4, 0.975) ≈ 2.776; approximate with 2.0 for k>=10
            t_crit = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                      6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 2.0)
            ci = t_crit * float(sub.std(ddof=1)) / sqrt(n)
        else:
            ci = 0.0
        label = _LABEL_OVERRIDES.get(
            key, df_folds[df_folds["model_set"] == key]["display_label"].iloc[0]
        )
        stats[key] = {"mean": mean, "ci": ci, "label": label}

    n_bars = len(order)
    bar_h = 0.68
    fig_h = max(6.0, n_bars * 0.60 + 2.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    # ── Alternating white / light-grey section bands ───────────────────────
    section_indices: list[list[int]] = []
    current_section: list[int] = []
    for i, key in enumerate(order):
        if key in _SECTION_STARTS and current_section:
            section_indices.append(current_section)
            current_section = []
        current_section.append(i)
    if current_section:
        section_indices.append(current_section)

    _SECTION_SHADES = ["white", "#eeeeee"]   # white / light grey alternating
    for s_idx, indices in enumerate(section_indices):
        y_lo = n_bars - 1 - max(indices) - 0.5
        y_hi = n_bars - 1 - min(indices) + 0.5
        ax.axhspan(y_lo, y_hi, facecolor=_SECTION_SHADES[s_idx % 2],
                   edgecolor="none", zorder=0)

    # Thick black separator lines between sections
    for i, key in enumerate(order):
        if key in _SECTION_STARTS:
            ax.axhline(n_bars - i - 0.5, color="black", linewidth=1.2, zorder=1)

    # ── Bars ──────────────────────────────────────────────────────────────
    for i, key in enumerate(order):
        yi = n_bars - 1 - i          # invert so first block is at top
        grp  = _BLOCK_GROUP[key]
        color = _GROUP_COLORS.get(grp, "steelblue")
        is_combined = key in _COMBINED_KEYS

        # Combined bars: same colour as group, hatched; solid bars: full opacity
        hatch    = "//" if is_combined else None
        alpha    = 0.85
        edgecol  = color   # always match bar colour for a clean look

        m  = stats[key]["mean"]
        ci = stats[key]["ci"]

        ax.barh(yi, m, height=bar_h, color=color, alpha=alpha,
                edgecolor=edgecol, linewidth=0.8, hatch=hatch, zorder=2)
        ax.errorbar(m, yi, xerr=ci, fmt="none",
                    color="black", linewidth=1.1, capsize=3, zorder=3)
        if noise_ceiling and np.isfinite(noise_ceiling) and noise_ceiling > 0:
            pct = int(round(100 * m / noise_ceiling))
            label_str = f"{m:.3f} ({pct}%)"
        else:
            label_str = f"{m:.3f}"
        ax.text(m + ci + 0.005, yi, label_str,
                va="center", ha="left", fontsize=7.5, color="black", zorder=4)

    # Y-axis tick labels (coloured and italicised for combined blocks)
    ytick_labels = [stats[k]["label"] for k in reversed(order)]
    ax.set_yticks(np.arange(n_bars))
    ax.set_yticklabels(ytick_labels, fontsize=8.5)
    for tick, key in zip(ax.get_yticklabels(), reversed(order)):
        grp = _BLOCK_GROUP[key]
        tick.set_color(_GROUP_COLORS.get(grp, "black"))
        if key in _COMBINED_KEYS:
            tick.set_style("italic")

    ax.set_xlabel("Cross-validated Spearman ρ  (mean ± 95% CI over folds)", fontsize=9)

    # X-axis: fit tightly to data — long bars like v2
    max_val = max(s["mean"] + s["ci"] for s in stats.values())
    ax.set_xlim(left=0, right=round(max_val + 0.08, 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.set_title(title, fontsize=10, pad=10)

    # Legend: one swatch per family + hatched swatch for combinations
    c = _GROUP_COLORS
    handles = [
        mpatches.Patch(color=c["Visual"],                label="Visual features"),
        mpatches.Patch(color=c["Human"],                 label="Human ratings"),
        mpatches.Patch(color=c["AI"],                    label="AI ratings"),
        mpatches.Patch(color=c["NOVA Processing Level"],  label="NOVA processing level"),
        mpatches.Patch(color=c["Intuitive category"],    label="Intuitive category"),
        mpatches.Patch(facecolor="white", hatch="//",
                       edgecolor="#888888",              label="Combination block"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="upper right",
              framealpha=0.90, edgecolor="lightgray")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote: {out_png}")


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    """Return mean and descriptive t-based fold-interval half-width."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    n = len(values)
    mean = float(values.mean())
    if n <= 1:
        return mean, 0.0
    from math import sqrt
    t_crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
              5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
              9: 2.262}.get(n - 1, 2.0)
    return mean, t_crit * float(values.std(ddof=1)) / sqrt(n)


def plot_family_compact(df_folds: pd.DataFrame, out_png: Path, rdm_mode: str) -> None:
    """Two-panel publication view separating total fit from incremental gain."""
    family_rows = [
        ("visual", "Visual", "Visual"),
        ("human_appraisal", "Human appraisal", "Human"),
        ("human_flavour", "Human sensory", "Human"),
        ("human_appraisal+human_flavour", "Human appraisal + sensory", "Human"),
        ("human_full", "Human appraisal + sensory + visual", "Human"),
        ("transformation", "Transformation (all three)", "Transformation"),
        ("food_classification", "Intuitive category", "Intuitive category"),
        ("Category_WHO_10", "WHO category", "Alternative taxonomy"),
        ("Category_Culinary_9", "Culinary category", "Alternative taxonomy"),
    ]
    increment_rows = [
        ("human_full", "human_appraisal+human_flavour", "Visual beyond human semantic", "Visual"),
        ("human_full+food_classification", "human_full", "Intuitive category\nbeyond semantic + visual", "Intuitive category"),
        ("human_full+Category_WHO_10", "human_full", "WHO category\nbeyond semantic + visual", "Alternative taxonomy"),
        ("human_full+Category_Culinary_9", "human_full", "Culinary category\nbeyond semantic + visual", "Alternative taxonomy"),
        ("human_full+transformation", "human_full", "Transformation beyond\nsemantic + visual", "Transformation"),
        ("human_full+food_classification+transformation", "human_full+food_classification", "Transformation beyond semantic + visual\n+ Intuitive category", "Transformation"),
        ("human_full+Category_WHO_10+transformation", "human_full+Category_WHO_10", "Transformation beyond semantic + visual\n+ WHO category", "Transformation"),
        ("human_full+Category_Culinary_9+transformation", "human_full+Category_Culinary_9", "Transformation beyond semantic + visual\n+ Culinary category", "Transformation"),
    ]

    present = set(df_folds["model_set"])
    missing = [k for k, _, _ in family_rows if k not in present]
    missing += [k for a, b, _, _ in increment_rows for k in (a, b) if k not in present]
    if missing:
        raise ValueError(f"Compact family figure is missing required model sets: {sorted(set(missing))}")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14.4, 7.5), gridspec_kw={"wspace": 0.82})

    # Panel A: total held-out fit.
    for pos, (key, label, group) in enumerate(reversed(family_rows)):
        vals = df_folds.loc[df_folds["model_set"] == key, "rho_test"]
        mean, ci = _mean_ci(vals)
        color = _GROUP_COLORS[group]
        ax_a.barh(pos, mean, height=0.64, color=color, alpha=0.88, edgecolor=color)
        if ci > 0:
            ax_a.errorbar(mean, pos, xerr=ci, fmt="none", color="#222222", capsize=3, linewidth=1.0)
        ax_a.text(mean + ci + 0.008, pos, f"{mean:.3f}", va="center", fontsize=8.5)
    ax_a.set_yticks(range(len(family_rows)))
    ax_a.set_yticklabels([x[1] for x in reversed(family_rows)], fontsize=9)
    ax_a.set_xlim(0, 0.72 if rdm_mode == "fixed_dense" else 0.86)
    ax_a.set_xlabel("Cross-validated Spearman ρ" if rdm_mode == "fixed_dense" else "Spearman ρ")
    ax_a.set_title("A   Predictive performance", loc="left", fontsize=11, fontweight="bold")
    ax_a.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.7)
    ax_a.set_axisbelow(True)

    # Panel B: paired fold-wise increments. This is the inferential comparison.
    pivot = df_folds.pivot(index="fold", columns="model_set", values="rho_test")
    increment_stats = []
    for augmented, baseline, label, group in increment_rows:
        delta = pivot[augmented] - pivot[baseline]
        mean, ci = _mean_ci(delta)
        augmented_mean = float(pivot[augmented].mean())
        increment_stats.append((label, group, mean, ci, augmented_mean))
    for pos, (label, group, mean, ci, augmented_mean) in enumerate(reversed(increment_stats)):
        color = _GROUP_COLORS[group]
        ax_b.barh(pos, mean, height=0.64, color=color, alpha=0.88, edgecolor=color)
        if ci > 0:
            ax_b.errorbar(mean, pos, xerr=ci, fmt="none", color="#222222", capsize=3, linewidth=1.0)
        x_text = mean + ci + 0.002 if mean >= 0 else mean - ci - 0.002
        ax_b.text(x_text, pos, f"{mean:+.3f}  (total {augmented_mean:.3f})", va="center",
                  ha="left" if mean >= 0 else "right", fontsize=8.5)
    ax_b.set_yticks(range(len(increment_stats)))
    ax_b.set_yticklabels([x[0] for x in reversed(increment_stats)], fontsize=9)
    ax_b.axvline(0, color="#555555", linewidth=0.9)
    if rdm_mode == "fixed_dense":
        ax_b.set_xlim(-0.025, 0.085)
        ax_b.set_xlabel("Incremental held-out Spearman Δρ\n(mean ± descriptive fold-based interval)")
    else:
        ax_b.set_xlim(-0.025, 0.085)
        ax_b.set_xlabel("Incremental Spearman Δρ (single-shot sensitivity)")
    ax_b.set_title("B   Increment beyond the baseline model", loc="left", fontsize=11, fontweight="bold")
    ax_b.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.7)
    ax_b.set_axisbelow(True)

    for ax in (ax_a, ax_b):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8.5)

    handles = [
        mpatches.Patch(color=_GROUP_COLORS["Human"], label="Human ratings"),
        mpatches.Patch(color=_GROUP_COLORS["Visual"], label="Visual features"),
        mpatches.Patch(color=_GROUP_COLORS["Transformation"], label="Transformation"),
        mpatches.Patch(color=_GROUP_COLORS["Intuitive category"], label="Intuitive category"),
        mpatches.Patch(color=_GROUP_COLORS["Alternative taxonomy"], label="Alternative category"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        "Family-wise regression predicting the behavioural RDM"
        + ("" if rdm_mode == "fixed_dense" else " (filled sensitivity)"),
        fontsize=12, y=0.985,
    )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.91, bottom=0.13)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote compact figure: {out_png}")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def write_markdown(df_folds: pd.DataFrame, df_summary: pd.DataFrame,
                   out_md: Path) -> None:
    lines: list[str] = []
    lines.append("# Family-Wise CV Regression — Results Summary\n\n")
    lines.append(
        "Cross-validated NNLS regression (5-fold) predicting held-out empirical RDM "
        "similarity from model-family subsets. ρ = Spearman correlation between "
        "predicted and observed dissimilarities on held-out pairs. "
        "Values are mean ± SD across folds.\n\n"
    )

    lines.append("| Model set | N models | ρ (mean ± SD) | R² (mean ± SD) |\n")
    lines.append("|---|---:|---:|---:|\n")

    present = set(df_folds["model_set"].unique())
    order = [k for k in _BLOCK_ORDER if k in present]
    repeated_folds = df_folds["fold"].nunique() > 1
    if "human_full+transformation" in present:
        insert_at = order.index("transformation") + 1
        order.insert(insert_at, "human_full+transformation")

    for key in order:
        row = df_summary[df_summary["model_set"] == key]
        if row.empty:
            continue
        row = row.iloc[0]
        label = str(row["display_label"])
        n_mod = int(row["n_models"])
        rho_str = f"{row['rho_mean']:.3f} ± {row['rho_sd']:.3f}"
        r2_str  = f"{row['r2_mean']:.3f} ± {row['r2_sd']:.3f}"
        lines.append(f"| {label} | {n_mod} | {rho_str} | {r2_str} |\n")

    lines.append("\n---\n\n")
    lines.append("## Key observations\n\n")

    # Auto-generate some key observations from the data
    def _rho(key: str) -> float:
        r = df_summary[df_summary["model_set"] == key]
        return float(r["rho_mean"].iloc[0]) if not r.empty else np.nan

    # REVISION 2026-09-11: this block used to be written around the AI rating
    # families. Those are no longer computed (see reference/policy_defaults.py),
    # so every AI lookup returned NaN and the section rendered sentences like
    # "AI outperforms Human throughout (rho=nan vs rho=nan)" underneath a correct
    # table. The observations are now drawn from the families the paper reports,
    # and each one is emitted only if the rows it cites are actually present.
    hu_app   = _rho("human_appraisal")
    hu_flav  = _rho("human_flavour")
    hu_comb  = _rho("human_appraisal+human_flavour")
    hu_full  = _rho("human_full")
    hu_trans = _rho("human_full+transformation")
    vis      = _rho("visual")
    food_cls = _rho("food_classification")
    hu_cat   = _rho("human_full+food_classification")

    def _have(*vals) -> bool:
        return all(v == v for v in vals)   # NaN != NaN

    if _have(hu_flav, hu_app):
        lines.append(
            f"**Human flavour > human appraisal** (ρ={hu_flav:.3f} vs ρ={hu_app:.3f}): "
            f"taste/flavour ratings account for more perceptual variance than "
            f"evaluative ratings (healthiness, calorie density) alone.\n\n"
        )
    if _have(hu_full, hu_comb):
        lines.append(
            f"**Visual adds negligible variance**: Human Full ρ={hu_full:.3f} vs "
            f"Human Appraisal+Sensory ρ={hu_comb:.3f} (Δρ={hu_full - hu_comb:+.3f}). "
            f"Low-level visual features are essentially redundant once the "
            f"semantic ratings are included.\n\n"
        )
    if _have(hu_trans, hu_full):
        qualifier = (
            "this fold-variable estimate is treated descriptively"
            if repeated_folds
            else "this single-shot filled-space sensitivity estimate is descriptive"
        )
        lines.append(
            f"**Transformation before category is included**: adding the joint "
            f"transformation family to Human Full increased mean held-out fit from "
            f"ρ={hu_full:.3f} to ρ={hu_trans:.3f} "
            f"(Δρ={hu_trans - hu_full:+.3f}); {qualifier}.\n\n"
        )
    if _have(vis):
        lines.append(
            f"**Visual features alone are weak**: ρ={vis:.3f}, well below every "
            f"rating family.\n\n"
        )
    if _have(food_cls, hu_comb):
        lines.append(
            f"**The Intuitive category is remarkably competitive**: ρ={food_cls:.3f} as a "
            f"single categorical predictor, against ρ={hu_comb:.3f} for the full human "
            f"appraisal and flavour suite. Basic category membership captures a "
            f"substantial portion of the perceptual structure.\n\n"
        )
    if _have(hu_cat, hu_full):
        lines.append(
            f"**Category adds to the semantic families**: Human Full + Category "
            f"ρ={hu_cat:.3f} against Human Full ρ={hu_full:.3f} "
            f"(Δρ={hu_cat - hu_full:+.3f}).\n\n"
        )

    out_md.write_text("".join(lines), encoding="utf-8")
    print(f"[OK] Wrote: {out_md}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    input_root = Path(args.input_dir).expanduser().resolve() if args.input_dir else INPUT_DIR
    in_dir = input_root / args.rdm_mode
    out_dir = Path(args.outdir).expanduser().resolve() if args.outdir else RESULTS_DIR / args.rdm_mode
    _ensure_dir(out_dir)

    fits_path    = in_dir / "cv_family_regression_fits.csv"
    summary_path = in_dir / "cv_family_summary.csv"

    if not fits_path.exists():
        print(f"[ERROR] Per-fold fits not found: {fits_path}")
        return 1
    if not summary_path.exists():
        print(f"[ERROR] Summary CSV not found: {summary_path}")
        return 1

    df_folds   = pd.read_csv(fits_path)
    df_summary = pd.read_csv(summary_path)

    # Drop AI-only families by default unless explicitly requested
    if not getattr(args, "include_ai", False):
        ai_keys = {k for k in _BLOCK_ORDER if k.startswith("ai_")}
        before = len(df_folds)
        df_folds   = df_folds[~df_folds["model_set"].isin(ai_keys)].copy()
        df_summary = df_summary[~df_summary["model_set"].isin(ai_keys)].copy()
        print(f"[INFO] AI families dropped by default; {before - len(df_folds)} AI rows removed.")

    # Load noise ceiling from the matching model evaluation manifest
    noise_ceiling: float | None = None

    if args.rdm_mode == "filled":
        noise_ceiling = None

    # Figure
    if args.compact:
        plot_family_compact(
            df_folds,
            out_png=out_dir / "fig_family_cv_rho_compact.png",
            rdm_mode=args.rdm_mode,
        )
    else:
        plot_family_bars(
            df_folds,
            out_png=out_dir / "fig_family_cv_rho.png",
            title=f"Family-wise CV regression: Spearman ρ on held-out RDM ({args.rdm_mode})",
            noise_ceiling=noise_ceiling,
        )

    # Markdown
    write_markdown(
        df_folds, df_summary,
        out_md=out_dir / "stage5_family_generalization_summary.md",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
