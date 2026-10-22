

#!/usr/bin/env python3
"""results3_model_evaluation_plots.py

Stage 3 (Results): Plot and summarize model evaluation outputs.

This script is intentionally *fast* and deterministic. It reads artifacts produced by
`compute_model_evaluation.py` and produces publication-ready figures + a short markdown summary.

Division of labour
------------------
- This script (results3) handles the **univariate** model scan.
- results6_variance_attribution_plots.py handles unique variance / Shapley attribution.

Expected inputs (from compute_model_evaluation.py):
  analysis_code/compute_outputs/stage2_model_evaluation/<rdm_mode>/<source>/
  - model_fits_univariate.csv   (required)
  - model_model_corr.csv        (optional — model-model correlation heatmap)
  - manifest.json               (optional — noise ceiling, policy)

Outputs:
  analysis_code/results/stage3_model_evaluation_plots/<rdm_mode>/<source>/
  - fig_model_scan_univariate.png
  - fig_model_model_corr.png (optional)
  - stage3_model_evaluation_summary.md

Usage (default — reads from mixed compute output):
  python analysis_code/results3_model_evaluation_plots.py

Override source or RDM mode:
  python analysis_code/results3_model_evaluation_plots.py --source human --rdm-mode empirical

Notes:
- Uses matplotlib only (no seaborn).
- If a compute manifest.json contains a policy section, this script will use that policy
  for grouping and filtering, and include conceptual contrasts in univariate only when
  present and marked as supplementary.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import sys
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from reference.policy_defaults import get_compute_dir, get_results_dir

# ------------------------------
# Helpers
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


def _pick_inputs(indir: Path) -> dict:
    """Return exact paths for expected inputs under indir."""
    out: dict = {}
    p1 = indir / "model_fits_univariate.csv"
    p2 = indir / "model_fits.csv"
    out["univariate"] = p1 if p1.exists() else (p2 if p2.exists() else None)
    
    
    p = indir / "model_model_corr.csv"
    out["corr"] = p if p.exists() else None
    
    p = indir / "manifest.json"
    out["manifest"] = p if p.exists() else None
    return out


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out



_UK_DISPLAY_SPELLING = (
    ("savoriness", "savouriness"),
    ("savory", "savoury"),
    ("flavor", "flavour"),
    ("color", "colour"),
    ("behavior", "behaviour"),
    ("neighbor", "neighbour"),
)


def _to_uk(s: str) -> str:
    """Convert a DISPLAY string to UK spelling.

    Internal model keys keep their US forms (human_savoriness, sweet_vs_savory),
    because renaming them would ripple through every cached output. Only what a
    reader sees is converted: Cognition accepts American or British usage but
    not a mixture, and this manuscript is British throughout.
    """
    if not s:
        return s
    for us, uk in _UK_DISPLAY_SPELLING:
        s = s.replace(us, uk).replace(us.capitalize(), uk.capitalize())
    return s


def _pretty_model_name(name: str) -> str:
    """UK-spelling wrapper around _pretty_model_name_raw (see _to_uk)."""
    return _to_uk(_pretty_model_name_raw(name))


def _pretty_model_name_raw(name: str) -> str:
    """Aggressively prettify model names for figures.
    - Strips leading 'rdm_' and 'll_' prefixes
    - Human models: 'human_mean_foo' -> 'Human Foo' (removes 'mean')
    - AI models: 'ai_<trait>' -> 'AI <trait>' using the canonical name from file
      (no trait renaming — keeps *iness forms so labels match human counterparts)
      Trailing '_0_100' scale suffix is stripped if present.
    - Low-level: underscores->spaces, drop trailing 'pcs10'->'PCs10'
    - NN_completed_seedX -> 'NN filled (seed X)'
    """
    if name is None:
        return ""
    s = str(name)
    # NN filled
    if s.startswith("NN_completed_seed"):
        seed = s.replace("NN_completed_seed", "")
        return f"NN filled (seed {seed.strip('_')})"
    # Strip leading rdm_
    if s.startswith("rdm_"):
        s = s[len("rdm_") :]
    # Categorical reference models
    if s == "food_classification":
        return "Intuitive category"
    if s == "NOVA_ordinal":
        return "NOVA processing level"
    if s == "Category_WHO_10":
        return "WHO food groups"
    if s == "Category_Culinary_9":
        return "Culinary groups"
    # Strip leading ll_
    if s.startswith("ll_"):
        s = s[len("ll_") :]
    # Human models
    if s.startswith("human_"):
        s = s[len("human_") :]
        if s.startswith("mean_"):
            s = s[len("mean_") :]
        s = s.replace("caloriedensity", "calorie density")
        s = s.replace("_", " ")
        s = " ".join(s.split())
        return "Human " + s.capitalize()
    # AI models — use canonical name from file (no trait renaming)
    if s.startswith("ai_"):
        s = s[len("ai_"):]
        # Strip trailing scale suffix if present (e.g. _0_100)
        trait = s.split("_0_100")[0] if "_0_100" in s else s
        trait = trait.replace("_", " ")
        trait = " ".join(trait.split())
        return f"AI {trait.capitalize()}"
    # Low-level models
    if s.startswith("ll_"):
        s = s[len("ll_") :]
    # If endswith pcs10, capitalize P
    if s.endswith("pcs10"):
        s = s[:-5] + "PCs10"
    # Conceptual contrasts (e.g. natural_vs_transformed) — strip 'concept_' prefix if present
    # so rdm_concept_natural_vs_transformed and rdm_natural_vs_transformed get the same label
    if s.startswith("concept_"):
        s = s[len("concept_"):]
    if "vs" in s:
        s = s.replace("_vs_", " vs ")
        s = s.replace("_", " ")
        s = " ".join(s.split())
        return s.capitalize()
    # Generic: underscores to spaces
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s


# --- NN seed extraction helper ---
def _nn_seed_from_name(raw_name: str) -> Optional[int]:
    """Extract NN seed from names like 'nn_seed21_comp01' or 'NN (seed 21) / ...'."""
    s = str(raw_name)
    if s.startswith("rdm_"):
        s = s[4:]
    # Canonical exported names: nn_seed21_comp01 / nn_seed21_embedding_cosine
    if s.startswith("nn_seed"):
        tail = s[len("nn_seed"):]
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            try:
                return int(digits)
            except Exception:
                return None
    # Older pretty names: NN (seed 21) / comp01
    marker = "(seed "
    if marker in s:
        try:
            tail = s.split(marker, 1)[1]
            digits = ""
            for ch in tail:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            return int(digits) if digits else None
        except Exception:
            return None
    return None



def _draw_group_separators(ax: plt.Axes, groups: list[str], y_vals: np.ndarray) -> None:
    """Draw light horizontal separator lines where model families change."""
    if len(groups) <= 1:
        return
    for i in range(1, len(groups)):
        if str(groups[i]) != str(groups[i - 1]):
            ax.axhline((y_vals[i - 1] + y_vals[i]) / 2.0, linewidth=0.8, alpha=0.25)


# --- Special separator for within-group boundaries (e.g., before unique human appeal) ---
def _draw_special_separators(ax: plt.Axes, labels_plot: list[str], y_vals: np.ndarray) -> None:
    """Draw extra separators for special within-group boundaries.

    Currently used to place an additional separator directly under the unique human
    appeal rating, which has no AI counterpart.
    """
    if len(labels_plot) <= 1:
        return
    labels_norm = [str(x).strip().lower() for x in labels_plot]
    for i in range(len(labels_norm) - 1):
        if labels_norm[i] == "human appeal":
            ax.axhline((y_vals[i] + y_vals[i + 1]) / 2.0, linewidth=0.8, alpha=0.25)
            break


# --- Model group helpers and policy helpers ---

def _policy_from_manifest(manifest: Optional[dict]) -> dict:
    """Extract plotting policy from manifest, if present."""
    if manifest is None or "policy" not in manifest:
        return {}
    pol = manifest.get("policy", {})
    out = {}
    # source
    out["source"] = pol.get("source", pol.get("model_source", "unknown"))
    # visual
    out["include_visual"] = pol.get("include_visual", pol.get("keep_visual_models", False))
    # controls
    out["include_controls"] = pol.get("include_controls", False)
    # contrasts
    out["include_contrasts"] = pol.get("include_contrasts", pol.get("include_conceptual_contrasts", False))
    # selected keys
    if "selected_models" in pol:
        out["selected_keys"] = list(pol["selected_models"])
    elif "models" in pol and isinstance(pol["models"], list):
        out["selected_keys"] = [m["key"] if isinstance(m, dict) and "key" in m else m for m in pol["models"]]
    else:
        out["selected_keys"] = None
    return out


def _model_group_with_policy(raw_name: str, policy: Optional[dict]) -> str:
    """Group models by new policy-aware families if policy is provided, else fallback."""
    s = str(raw_name)
    # Conceptual contrasts (supplementary)
    # REVISION 2026-09-10 (revision/transformation-models): membership comes from
    # reference/policy_defaults so figure grouping cannot drift from model selection.
    # Previously natural_vs_transformed was grouped only by an incidental "_vs_"
    # substring match and Transformation_score reached this group solely via the
    # catch-all fallback at the end of this function.
    try:
        from reference.policy_defaults import CONCEPTUAL_SUPPLEMENTARY as _POLICY_CONCEPTUAL
        conceptual_keys = list(_POLICY_CONCEPTUAL) + ["natural_vs_processed"]
    except Exception:  # pragma: no cover - must mirror policy_defaults
        conceptual_keys = [
            "food_classification",
            "Category_WHO_10",
            "Category_Culinary_9",
            "NOVA_ordinal",
            "natural_vs_transformed",
            "Transformation_score",
            "natural_vs_processed",
        ]
    # Remove rdm_ prefix for matching
    key = s
    if key.startswith("rdm_"):
        key = key[4:]
    # Policy grouping
    if policy is not None and len(policy) > 0:
        # Map keys to families if possible
        # For now, use labels:
        # Appraisal/valuation (Human), Flavour/taste (Human), Appraisal/valuation (AI), Flavour/taste (AI), Visual, NN, Conceptual (supp)
        # Heuristics:
        if key.startswith("ll_") or "lowlevel" in key or "visual" in key:
            return "Visual"
        # Categorical reference models — explicit before the generic conceptual fallback
        if key == "food_classification":
            return "Conceptual (supp)"
        if key in conceptual_keys or "_vs_" in key.lower() or key.lower().startswith("concept_"):
            return "Conceptual (supp)"
        if key.startswith("human_") or key.startswith("mean_"):
            if any(x in key for x in ["liking", "pleasantness", "preference", "value", "valuation", "enjoyment"]):
                return "Human ratings"
            else:
                return "Human ratings"
        if key.startswith("ai_"):
            if any(x in key for x in ["liking", "pleasantness", "preference", "value", "valuation", "enjoyment"]):
                return "AI ratings"
            else:
                return "AI ratings"
        if key.startswith("ll_") or "lowlevel" in key or "visual" in key:
            return "Visual"
        if "NN" in key or key.startswith("NN") or key.startswith("nn_") or key.startswith("nn_seed"):
            return "NN"
        # fallback: unknown
        return "Conceptual (supp)"
    # Fallback to legacy grouping
    return _model_group(raw_name)


def _model_group(raw_name: str) -> str:
    """Coarse model family for ordering/legend (legacy fallback)."""
    s = str(raw_name)
    # Strip rdm_ prefix once so all checks below work for both prefixed and bare names
    key = s[4:] if s.startswith("rdm_") else s
    if key.startswith("ll_") or " ll_" in key:
        return "Visual"
    if key.startswith("ai_"):
        return "AI"
    if key.startswith("human_"):
        return "Human"
    if key.startswith("nn_") or key.startswith("NN") or "NN" in key:
        return "NN"
    # Categorical reference models (explicit before generic "vs" fallback)
    if key == "food_classification":
        return "Conceptual"
    # Conceptual contrasts
    if "vs" in key or "contrast" in key:
        return "Conceptual"
    return "Conceptual"

# Visual models shown in the main figure. All other ll_* models are suppressed
# from the figure and reported in the supplementary table only.
_VISUAL_FIGURE_KEEP: set[str] = {"ll_resnet50_conv1_pcs10", "ll_lbp_hist", "ll_resnet50_layer3_pcs10", "ll_hsv_s_mean"}

def _is_suppressed_visual(raw_name: str) -> bool:
    """Return True for ll_* visual models that should not appear in the main figure."""
    k = str(raw_name)
    if k.startswith("rdm_"):
        k = k[4:]
    return k.startswith("ll_") and k not in _VISUAL_FIGURE_KEEP


def _group_order() -> dict[str, int]:
    # Group order for the univariate figure:
    # Human ratings, AI ratings, Conceptual contrasts, Visual, NN
    return {
        "Human ratings": 0,
        "AI ratings": 1,
        "Conceptual (supp)": 2,
        "Visual": 3,
        "NN": 4,
        # fallback legacy
        "Human": 0,
        "AI": 1,
        "Conceptual": 2,
    }

def _group_marker(group: str) -> str:
    # Uniform circles everywhere — colour alone distinguishes groups,
    # consistent with the family-wise and variance-attribution figures.
    return "o"


# Shared colour palette — matches stage2 heatmap and stage3 model-model heatmap
_GROUP_COLORS: dict[str, str] = {
    "Human ratings":    "#377eb8",
    "AI ratings":       "#ff7f00",
    "Conceptual (supp)":"#e41a1c",
    "Visual":           "#4daf4a",
    "NN":               "#984ea3",
    # fallback legacy keys
    "Human":            "#377eb8",
    "AI":               "#ff7f00",
    "Conceptual":       "#e41a1c",
}


def _group_color(group: str) -> str:
    return _GROUP_COLORS.get(group, "#888888")


def _set_pub_axes(ax: plt.Axes) -> None:
    # Light, clean axes (do not hard-code colors)
    ax.grid(True, axis="x", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)


# ------------------------------
# Plots
# ------------------------------


def plot_univariate_scan(
    df: pd.DataFrame,
    out_png: Path,
    title: str = "Model fit (univariate RSA)",
    top_n: int = 50,
    xcol: str = "rho",
    ci_lo: str = "ci95_lo",
    ci_hi: str = "ci95_hi",
    annotate_ceiling: Optional[float] = None,
    group_by_type: bool = True,
    policy: Optional[dict] = None,
    nn_seed: int = -1,
) -> None:
    df = df.copy()

    # normalize column names across possible variants
    if xcol not in df.columns and "spearman_r" in df.columns:
        xcol = "spearman_r"
    if ci_lo not in df.columns and "ci95_lo" in df.columns:
        ci_lo = "ci95_lo"
    if ci_hi not in df.columns and "ci95_hi" in df.columns:
        ci_hi = "ci95_hi"

    need = ["model", xcol]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Univariate df missing required column: {c}")

    df = _coerce_numeric(df, [xcol, ci_lo, ci_hi, "p_perm", "q_fdr", "r_over_ceiling"])

    # Drop NaNs
    df = df[np.isfinite(df[xcol])].copy()

    # NN filtering: nn_seed < 0 (default) = exclude all NN models;
    # nn_seed >= 0 = keep only NN models from that specific seed.
    if "model" in df.columns:
        keep_mask = []
        for m in df["model"].astype(str).tolist():
            grp = _model_group(m)
            if grp != "NN":
                keep_mask.append(True)
            elif nn_seed >= 0:
                seed_here = _nn_seed_from_name(m)
                keep_mask.append(seed_here is None or int(seed_here) == int(nn_seed))
            else:
                keep_mask.append(False)  # nn_seed=-1: drop all NN
        df = df.loc[np.asarray(keep_mask, dtype=bool)].copy()

    # Policy-aware filtering
    if policy is not None and len(policy) > 0 and policy.get("_policy_only", True):
        selected_keys = policy.get("selected_keys", None)
        include_contrasts = policy.get("_include_contrasts_in_univariate", False) or policy.get("include_contrasts", False)
        # Remove rdm_ prefix for matching
        def is_conceptual(modelkey: str) -> bool:
            k = modelkey
            if k.startswith("rdm_"):
                k = k[4:]
            return ("vs" in k or "contrast" in k or "natural_vs" in k)

        def is_nn_model(modelkey: str) -> bool:
            k = modelkey
            if k.startswith("rdm_"):
                k = k[4:]
            return (
                "NN" in k
                or k.startswith("NN")
                or k.startswith("nn_")
                or k.startswith("nn_seed")
            )

        if selected_keys is not None:
            # Allow both with and without rdm_ prefix in selected_keys
            selected_keys_set = set(selected_keys)

            def key_in_selected(modelkey: str) -> bool:
                k = modelkey
                if k.startswith("rdm_"):
                    k = k[4:]
                return (k in selected_keys_set) or (modelkey in selected_keys_set)

            df = df[
                df["model"].apply(
                    lambda m: (
                        key_in_selected(str(m))
                        or is_nn_model(str(m))
                        or (is_conceptual(str(m)) and include_contrasts)
                    )
                )
            ]
        else:
            if not include_contrasts:
                df = df[~df["model"].apply(is_conceptual)]

    # Drop weak visual models from figure — only the representative visual model
    # (ll_resnet50_conv1_pcs10) is shown; the rest are reported in the supplementary table.
    df = df[~df["model"].astype(str).map(_is_suppressed_visual)].copy()

    # Sort by effect size
    df = df.sort_values(xcol, ascending=False)

    def is_nn_model(modelkey: str) -> bool:
        k = str(modelkey)
        if k.startswith("rdm_"):
            k = k[4:]
        return (
            "NN" in k
            or k.startswith("NN")
            or k.startswith("nn_")
            or k.startswith("nn_seed")
        )

    # Keep the top_n models overall, but always retain NN models that survived
    # the earlier filtering steps so they appear in the univariate figure.
    df_top = df.head(top_n).copy()
    df_nn = df[df["model"].astype(str).apply(is_nn_model)].copy()

    if len(df_nn) > 0:
        df_pos = pd.concat([df_top, df_nn], axis=0, ignore_index=True)
        df_pos = df_pos.drop_duplicates(subset=["model"], keep="first")
        df_pos = df_pos.sort_values(xcol, ascending=False)
    else:
        df_pos = df_top.copy()

    df_pos["_raw_model"] = df_pos["model"].astype(str)
    # Grouping
    if policy is not None and len(policy) > 0:
        df_pos["_group"] = df_pos["_raw_model"].map(lambda m: _model_group_with_policy(m, policy))
    else:
        df_pos["_group"] = df_pos["_raw_model"].map(_model_group)

    # Compute pretty names early so we can build within-group sort keys
    df_pos["_pretty"] = df_pos["model"].astype(str).map(_pretty_model_name)

    if group_by_type:
        order_map = _group_order()
        df_pos["_gord"] = df_pos["_group"].map(lambda g: order_map.get(str(g), 99))
        # Within-group ordering:
        #   Human ratings — alphabetical by pretty name; 'appeal' goes last (has no AI counterpart)
        #   AI ratings    — alphabetical by pretty name
        #   Other groups  — sort by ρ descending (Visual, NN, Conceptual)
        def _make_sort_keys(df):
            s_str, s_num, s_sub = [], [], []
            for _, row in df.iterrows():
                grp = str(row["_group"])
                pretty = str(row["_pretty"]).lower()
                if grp in ("Human ratings", "Human"):
                    is_appeal = (pretty == "human appeal") or pretty.endswith(" appeal")
                    s_str.append(pretty)
                    s_num.append(0.0)
                    # appeal sorts last within the Human group (1 > 0) → lowest y → bottom of block
                    s_sub.append(1 if is_appeal else 0)
                elif grp in ("AI ratings", "AI"):
                    s_str.append(pretty)
                    s_num.append(0.0)
                    s_sub.append(0)
                else:
                    s_str.append("")
                    try:
                        s_num.append(-float(row[xcol]))
                    except Exception:
                        s_num.append(0.0)
                    s_sub.append(0)
            return s_str, s_num, s_sub

        _ss, _sn, _ssub = _make_sort_keys(df_pos)
        df_pos["_sort_str"] = _ss
        df_pos["_sort_num"] = _sn
        df_pos["_sort_sub"] = _ssub
        df_pos = df_pos.sort_values(["_gord", "_sort_sub", "_sort_str", "_sort_num"], ascending=[True, True, True, True])
    else:
        df_pos = df_pos.sort_values(xcol, ascending=False)

    # Final plotting order (top-to-bottom): df_pos rows
    df_plot = df_pos.copy()
    df_plot["_pretty"] = df_plot["model"].astype(str).map(_pretty_model_name)

    # De-duplicate by pretty name (keeping the best performing one)
    df_plot = df_plot.sort_values(xcol, ascending=False).drop_duplicates(subset=["_pretty"]).copy()

    # Re-apply grouping sort if needed
    if group_by_type:
        # Recompute sort keys after dedup (pretty names already up to date)
        _ss2, _sn2, _ssub2 = _make_sort_keys(df_plot)
        df_plot["_sort_str"] = _ss2
        df_plot["_sort_num"] = _sn2
        df_plot["_sort_sub"] = _ssub2
        _drop_cols = [c for c in ["_gord", "_sort_str", "_sort_num", "_sort_sub"] if c in df_plot.columns]
        df_plot = df_plot.sort_values(["_gord", "_sort_sub", "_sort_str", "_sort_num"], ascending=[True, True, True, True]).drop(columns=_drop_cols).copy()
    else:
        df_plot = df_plot.sort_values(xcol, ascending=False).copy()

    # y positions: first row at top
    n = len(df_plot)
    y_pos = np.arange(n)[::-1]
    df_plot["_y"] = y_pos

    labels_plot = df_plot["_pretty"].tolist()

    # CIs are optional
    has_ci = (ci_lo in df_plot.columns) and (ci_hi in df_plot.columns) and np.isfinite(df_plot[ci_lo]).any() and np.isfinite(df_plot[ci_hi]).any()

    # Figure
    fig_h = max(5.5, 0.28 * len(labels_plot) + 1.5)
    fig = plt.figure(figsize=(10, fig_h))
    ax = fig.add_subplot(111)

    # Plot by group (keeps correct label<->point mapping via stored y positions)
    if has_ci:
        for grp in df_plot["_group"].unique().tolist():
            sub = df_plot[df_plot["_group"] == grp]
            sub_rho = sub[xcol].to_numpy(dtype=float)
            sub_lo = sub[ci_lo].to_numpy(dtype=float) if ci_lo in sub.columns else np.full(len(sub), np.nan)
            sub_hi = sub[ci_hi].to_numpy(dtype=float) if ci_hi in sub.columns else np.full(len(sub), np.nan)
            sub_err_lo = sub_rho - sub_lo
            sub_err_hi = sub_hi - sub_rho
            grp_alpha = 0.6 if str(grp) in {"Conceptual (supp)", "Conceptual"} else 1.0
            ax.errorbar(
                sub_rho,
                sub["_y"].to_numpy(dtype=int),
                xerr=[sub_err_lo, sub_err_hi],
                fmt=_group_marker(str(grp)),
                color=_group_color(str(grp)),
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
                sub[xcol].to_numpy(dtype=float),
                sub["_y"].to_numpy(dtype=int),
                _group_marker(str(grp)),
                color=_group_color(str(grp)),
                markersize=5,
                label=str(grp),
                alpha=grp_alpha,
            )

    _draw_group_separators(ax, df_plot["_group"].astype(str).tolist(), df_plot["_y"].to_numpy(dtype=float))
    _draw_special_separators(ax, labels_plot, df_plot["_y"].to_numpy(dtype=float))
    ax.axvline(0, linewidth=1, linestyle="--")

    # Noise ceiling — vertical reference line so readers can judge model fits against ceiling
    if annotate_ceiling is not None and np.isfinite(annotate_ceiling):
        ax.axvline(
            annotate_ceiling,
            linewidth=1.2,
            linestyle=":",
            color="gray",
            zorder=0,
            label=f"Noise ceiling ({annotate_ceiling:.3f})",
        )

    ax.set_yticks(df_plot["_y"].to_numpy(dtype=int))
    ax.set_yticklabels(labels_plot)
    ax.set_xlabel("Spearman ρ with behavioural RDM")
    ax.set_title(title)

    # Colour-code y-axis tick labels by group (matches family-wise figure).
    # y_pos is descending (top row = highest y value), so tick order matches
    # df_plot sorted by _y descending.
    group_list = df_plot.sort_values("_y", ascending=False)["_group"].astype(str).tolist()
    for tick, grp in zip(ax.get_yticklabels(), group_list):
        tick.set_color(_group_color(grp))

    _set_pub_axes(ax)

    # Legend inside the plot (lower right), consistent with family-wise figure
    handles, labels_leg = ax.get_legend_handles_labels()
    order_map = _group_order()
    order = np.argsort([order_map.get(lbl, 99) for lbl in labels_leg])
    handles = [handles[i] for i in order]
    labels_leg = [labels_leg[i] for i in order]

    ax.legend(
        handles,
        labels_leg,
        title="Family",
        loc="upper right",
        framealpha=0.9,
        edgecolor="#cccccc",
        fontsize=8,
    )

    fig.tight_layout(pad=1.8)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_model_corr_heatmap(
    df_corr: pd.DataFrame,
    out_png: Path,
    title: str = "Model–model similarity (Spearman)",
    max_labels: int = 60,
    nn_seed: int = -1,
) -> None:
    """Heatmap of model-model Spearman correlations.

    Models are sorted by family group. White separator lines divide family
    blocks. Tick labels are colour-coded by family. nn_seed=-1 (default)
    excludes all NN models; nn_seed >= 0 includes only that specific seed.
    """
    if df_corr is None or df_corr.empty:
        return

    df_corr = df_corr.copy()
    df_corr.index = df_corr.index.astype(str)
    df_corr.columns = df_corr.columns.astype(str)

    if df_corr.shape[0] != df_corr.shape[1]:
        print("[WARN] model_model_corr.csv not square; skipping heatmap")
        return

    # --- NN filtering: nn_seed < 0 (default) = exclude all NN;
    # nn_seed >= 0 = keep only NN from that specific seed ---
    raw = df_corr.index.tolist()
    keep = []
    for n in raw:
        grp = _model_group(n)
        if grp != "NN":
            keep.append(n)
        elif nn_seed >= 0:
            seed_here = _nn_seed_from_name(n)
            if seed_here is None or int(seed_here) == int(nn_seed):
                keep.append(n)
        # else nn_seed=-1: drop all NN
    df_corr = df_corr.loc[keep, keep]

    # --- Sort by family group, then alphabetically within group ---
    # human_appeal has no AI counterpart — sort it last within the Human block
    raw_names = df_corr.index.tolist()
    order_map = _group_order()
    def _heatmap_sort_key(name: str):
        grp = _model_group(name)
        pretty = _pretty_model_name(name).lower()
        is_appeal = 1 if (grp in {"Human", "Human ratings"} and "appeal" in pretty) else 0
        return (order_map.get(grp, 99), is_appeal, name)
    sorted_idx = sorted(range(len(raw_names)), key=lambda i: _heatmap_sort_key(raw_names[i]))
    df_corr = df_corr.iloc[sorted_idx, :].iloc[:, sorted_idx]

    # --- Optionally shrink ---
    if df_corr.shape[0] > max_labels:
        df_corr = df_corr.iloc[:max_labels, :max_labels]

    # --- Build labels and group tags ---
    raw_names = df_corr.index.tolist()
    labs = [_pretty_model_name(x) for x in raw_names]
    groups = [_model_group(x) for x in raw_names]
    n = len(labs)

    # --- Family colours ---
    group_colors = {
        "Conceptual": "#e41a1c",
        "Human": "#377eb8",
        "AI": "#ff7f00",
        "Visual": "#4daf4a",
        "NN": "#984ea3",
    }

    # --- Figure size scales with number of models ---
    fig_size = max(10, 0.22 * n + 3)
    fig = plt.figure(figsize=(fig_size, fig_size * 0.92))
    ax = fig.add_subplot(111)

    M = df_corr.to_numpy(dtype=float)
    im = ax.imshow(M, interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    ax.set_title(title)

    # --- Ticks ---
    tick_fs = max(6, 9 - n // 12)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labs, rotation=90, fontsize=tick_fs)
    ax.set_yticklabels(labs, fontsize=tick_fs)
    ax.tick_params(axis="both", which="both", length=0)

    # --- Colour tick labels by family ---
    for tick, grp in zip(ax.get_xticklabels(), groups):
        tick.set_color(group_colors.get(grp, "black"))
    for tick, grp in zip(ax.get_yticklabels(), groups):
        tick.set_color(group_colors.get(grp, "black"))

    # --- White separator lines between family blocks ---
    prev_grp = groups[0]
    for i in range(1, n):
        if groups[i] != prev_grp:
            ax.axhline(i - 0.5, color="white", linewidth=1.5)
            ax.axvline(i - 0.5, color="white", linewidth=1.5)
            prev_grp = groups[i]

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------
# Markdown summary
# ------------------------------

def write_summary_md(
    out_md: Path,
    df_uni: Optional[pd.DataFrame],
    manifest: Optional[dict],
    top_n: int = 15,
) -> None:
    lines: list[str] = []
    lines.append("# Stage 3 — Model evaluation (Univariate)\n")

    # Noise ceiling
    ceiling = None
    if manifest is not None:
        ceiling = (
            manifest.get("metrics", {}).get("noise_ceiling_mean")
            or manifest.get("metrics", {}).get("noise_ceiling_scan_mean")
        )
    if ceiling is not None:
        try:
            ceiling_f = float(ceiling)
            lines.append(f"**Noise ceiling (if available):** {ceiling_f:.3f}\n")
        except Exception:
            pass

    # Univariate
    if df_uni is not None and not df_uni.empty:
        df = df_uni.copy()
        # normalize x col
        xcol = "rho" if "rho" in df.columns else ("spearman_r" if "spearman_r" in df.columns else None)
        if xcol is not None:
            df = _coerce_numeric(df, [xcol, "ci95_lo", "ci95_hi", "p_perm", "q_fdr", "r_over_ceiling"])
            df["_pretty"] = df["model"].astype(str).map(_pretty_model_name)
            df = df[np.isfinite(df[xcol])].sort_values(xcol, ascending=False).drop_duplicates(subset=["_pretty"])
            lines.append("## Univariate model fits (top models)\n")
            show = df.head(top_n)
            # Build a small markdown table
            cols = ["_pretty", xcol]
            for c in ("ci95_lo", "ci95_hi", "p_perm", "q_fdr", "r_over_ceiling"):
                if c in show.columns:
                    cols.append(c)
            show = show[cols].copy()

            lines.append("| Model | ρ | CI95 lo | CI95 hi | p | q | ρ/ceiling |\n")
            lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
            for _, r in show.iterrows():
                m = r.get("_pretty", "")
                rho = r.get(xcol, np.nan)
                lo = r.get("ci95_lo", np.nan)
                hi = r.get("ci95_hi", np.nan)
                p = r.get("p_perm", np.nan)
                q = r.get("q_fdr", np.nan)
                rc = r.get("r_over_ceiling", np.nan)
                lines.append(
                    "| {} | {:.3f} | {} | {} | {} | {} | {} |\n".format(
                        m,
                        float(rho) if np.isfinite(rho) else float("nan"),
                        f"{float(lo):.3f}" if np.isfinite(lo) else "",
                        f"{float(hi):.3f}" if np.isfinite(hi) else "",
                        f"{float(p):.3g}" if np.isfinite(p) else "",
                        f"{float(q):.3g}" if np.isfinite(q) else "",
                        f"{float(rc):.3f}" if np.isfinite(rc) else "",
                    )
                )
            lines.append("\n")

    out_md.write_text("".join(lines), encoding="utf-8")


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
        default="mixed",
        choices=["human", "ai", "mixed"],
        help="Model source to evaluate: human, ai, or mixed (default: mixed; matches the univariate_all compute run)",
    )
    p.add_argument("--top-n", type=int, default=50, help="How many models to display in plots (default 50)")
    p.add_argument(
        "--max-corr-labels",
        type=int,
        default=60,
        help="Max number of model labels to show in model-model correlation heatmap (default 60)",
    )
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
        "--include-contrasts-in-univariate",
        action="store_true",
        help="Include conceptual contrasts in univariate plot (overrides policy; default: only if policy includes them)",
    )
    p.add_argument(
        "--policy-only",
        action="store_true",
        default=None,
        help="When policy exists, restrict to policy-selected models (default: True). If set and no policy, keep all models.",
    )
    p.add_argument(
        "--no-ai",
        action="store_true",
        default=False,
        help="Drop all AI rating models (ai_*) from the univariate plot.",
    )

    return p.parse_args()

def main() -> int:
    args = parse_args()
    indir = get_compute_dir("stage2_model_evaluation", args.rdm_mode, args.source)
    outdir = get_results_dir("stage3_model_evaluation_plots", args.rdm_mode, args.source)
    _ensure_dir(outdir)

    # Use recursive input discovery
    inputs = _pick_inputs(indir)
    uni_path = inputs["univariate"]
    corr_path = inputs["corr"]
    mani_path = inputs["manifest"]

    if uni_path is not None:
        print(f"[INFO] Using univariate fits: {uni_path}")
    else:
        print(f"[WARN] Could not find model_fits_univariate.csv or model_fits.csv under: {indir}")

    if corr_path is not None:
        print(f"[INFO] Using model-model corr: {corr_path}")
    if mani_path is not None:
        print(f"[INFO] Using manifest: {mani_path}")

    df_uni = _read_csv_if_exists(uni_path) if uni_path is not None else None
    # model_model_corr.csv is always a square matrix with the first column as the row index;
    # read with index_col=0 directly to avoid "Unnamed: 0" ambiguity.
    df_corr = None
    if corr_path is not None:
        try:
            df_corr = pd.read_csv(corr_path, index_col=0)
            df_corr.index = df_corr.index.astype(str)
            df_corr.columns = df_corr.columns.astype(str)
        except Exception as e:
            print(f"[WARN] Could not read model_model_corr.csv: {e}")

    manifest = _read_json_if_exists(mani_path) if mani_path is not None else None

    # Resolve NN seed: CLI overrides manifest; otherwise read from manifest; fallback = -1 (show all).
    if args.nn_seed is not None:
        resolved_nn_seed = int(args.nn_seed)
    else:
        try:
            resolved_nn_seed = int(manifest.get("selected_nn_seed")) if manifest is not None and manifest.get("selected_nn_seed") is not None else -1
        except Exception:
            resolved_nn_seed = -1

    # Policy from manifest
    policy = _policy_from_manifest(manifest)
    # Thread CLI flags into policy
    # --policy-only: if set (or default True), enforce policy filtering; if not set, ignore policy
    if args.policy_only is None:
        policy["_policy_only"] = True
    else:
        policy["_policy_only"] = args.policy_only
    # --include-contrasts-in-univariate
    policy["_include_contrasts_in_univariate"] = bool(args.include_contrasts_in_univariate)

    ceiling = None
    if manifest is not None:
        ceiling = (
            manifest.get("metrics", {}).get("noise_ceiling_mean")
            or manifest.get("metrics", {}).get("noise_ceiling_scan_mean")
        )
        try:
            ceiling = float(ceiling) if ceiling is not None else None
        except Exception:
            ceiling = None

    if args.rdm_mode == "filled":
        ceiling = None

    # --- Drop AI models if requested ---
    if args.no_ai and df_uni is not None:
        before = len(df_uni)
        df_uni = df_uni[~df_uni["model"].astype(str).str.startswith("ai_")].copy()
        print(f"[INFO] --no-ai: dropped {before - len(df_uni)} AI model rows.")

    # --- Plots ---
    if df_uni is None or df_uni.empty:
        print(f"[ERROR] No univariate fits found. Looked under: {indir}")
    else:
        out_png = outdir / "fig_model_scan_univariate.png"
        plot_univariate_scan(
            df_uni,
            out_png,
            title="Model fit (univariate RSA)",
            top_n=int(args.top_n),
            annotate_ceiling=ceiling,
            group_by_type=(not args.no_group_by_type),
            policy=policy if policy else None,
            nn_seed=int(resolved_nn_seed),
        )
        print(f"[OK] Wrote: {out_png}")

    if df_corr is not None and not df_corr.empty:
        out_png = outdir / "fig_model_model_corr.png"
        plot_model_corr_heatmap(df_corr, out_png, max_labels=int(args.max_corr_labels), nn_seed=int(resolved_nn_seed))
        print(f"[OK] Wrote: {out_png}")
    else:
        print("[NOTE] model_model_corr.csv not found (or empty) — skipping model-model heatmap")

    # --- Summary MD ---
    out_md = outdir / "stage3_model_evaluation_summary.md"
    write_summary_md(out_md, df_uni, manifest, top_n=15)
    print(f"[OK] Wrote: {out_md}")

    # Also copy the key input filenames used (for provenance)
    prov = {
        "indir": str(indir.resolve()),
        "inputs": {
            "univariate": str(uni_path.resolve()) if uni_path is not None else None,
            "model_corr": str(corr_path.resolve()) if corr_path is not None else None,
            "manifest": str(mani_path.resolve()) if mani_path is not None else None,
        },
        "outputs": {
            "fig_univariate": str((outdir / "fig_model_scan_univariate.png").resolve())
            if (outdir / "fig_model_scan_univariate.png").exists()
            else None,
            "fig_corr": str((outdir / "fig_model_model_corr.png").resolve())
            if (outdir / "fig_model_model_corr.png").exists()
            else None,
            "summary_md": str(out_md.resolve()) if out_md.exists() else None,
        },
    }
    (outdir / "stage3_model_evaluation_provenance.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())