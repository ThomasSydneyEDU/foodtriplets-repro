#!/usr/bin/env python3
"""results2_model_space_2d.py

Visualise similarity between *model* RDMs in 2D.

What it does
------------
1) Loads every `.npz` file under `analysis_code/derivatives/model_rdms_canon/`.
   Each `.npz` is expected to contain either:
     - a square matrix under a key like 'D', 'rdm', or the first array in the file, OR
     - a condensed vector under a key like 'vec' (rare; handled).
2) Vectorises each RDM (upper triangle, k=1).
3) Computes pairwise similarity between model vectors using Spearman correlation
   on overlapping finite entries.
4) Converts similarity -> distance: dist = 1 - rho (clipped to [0, 2]).
5) Runs Metric MDS (2D) on the distance matrix.
6) Plots models as points, coloured by inferred source from filename (ai/human/other).

Outputs
-------
- analysis_code/results/stage2_model_visualisation/<policy_tag>/model_similarity_matrix.csv
- analysis_code/results/stage2_model_visualisation/<policy_tag>/model_mds_2d.png
- analysis_code/results/stage2_model_visualisation/<policy_tag>/model_mds_2d_coords.csv
- analysis_code/results/stage2_model_visualisation/<policy_tag>/model_visualisation_manifest.json

Usage
-----
python analysis_code/results_model_visualisations.py

Optional:
python analysis_code/results_model_visualisations.py --metric spearman --random-state 0

Notes
-----
- Spearman is robust to monotonic transforms and is appropriate for comparing RDM structure.
- If two models have too little overlap in finite entries, rho is set to 0 (distance 1).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

try:
    # Preferred import when running from project root as a package
    from analysis_code.analysis_helpers import select_models, summarize, ModelSpec
    _HAS_POLICY = True
except ImportError:
    try:
        # Fallback when running the script directly from within analysis_code/
        from analysis_helpers import select_models, summarize, ModelSpec
        _HAS_POLICY = True
    except ImportError:
        try:
            # Backward-compatible legacy fallback
            from analysis_code.legacy.model_selection import select_models, summarize, ModelSpec
            _HAS_POLICY = True
        except ImportError:
            _HAS_POLICY = False


def infer_source_tag(stem: str) -> str:
    s = stem.lower()
    if re.match(r"rdm_nn_seed\d+_comp\d+", s) or re.match(r"rdm_nn_seed\d+_embedding", s):
        return "nn"
    if s.startswith("rdm_ai_") or "_ai_" in s:
        return "ai"
    if s.startswith("rdm_human_") or "_human_" in s:
        return "human"
    return "other"


def nice_label_from_stem(stem: str) -> str:
    """Turn a filename stem into a concise, readable label."""
    s = stem
    # Strip common prefixes
    for pref in ["rdm_", "model_", "triplet_"]:
        if s.startswith(pref):
            s = s[len(pref) :]

    # NN component models: e.g. nn_seed21_comp01 -> "NN comp 1"
    m = re.match(r"nn_seed\d+_comp(\d+)$", s)
    if m:
        return f"NN comp {int(m.group(1))}"

    # Low-level models: drop the raw `ll_` prefix.
    if s.startswith("ll_"):
        s = s[len("ll_") :]

    # Conceptual contrast models: drop `concept_` prefix
    if s.startswith("concept_"):
        s = s[len("concept_") :]

    # Explicit overrides for category schema models
    if s == "Category_WHO_10":
        return "WHO groups"
    if s == "Category_Culinary_9":
        return "Culinary groups"
    if s == "food_classification":
        return "Intuitive category"
    if s == "NOVA_ordinal":
        return "NOVA level"

    # Make readable
    s = s.replace("_0_100", "")
    s = s.replace("_", " ")

    # Human rating models: drop the word "mean" for tighter labels
    # e.g., "human mean sweet" -> "human sweet"
    if s.startswith("human mean "):
        s = "human " + s[len("human mean "):]

    # Fix known compound: caloriedensity -> calorie density
    s = s.replace("caloriedensity", "calorie density")

    return s


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _stem_to_key(stem: str) -> str:
    if stem.startswith("rdm_"):
        stem = stem[4:]
    # Strip concept_ prefix so rdm_concept_natural_vs_transformed → natural_vs_transformed
    if stem.startswith("concept_"):
        stem = stem[len("concept_"):]

    # Normalise capitalisation to match policy keys (e.g. Natural_vs_transformed → natural_vs_transformed)
    stem = stem.lower()
    # analysis_helpers uses ai_* as canonical keys — no prefix remapping needed
    return stem


def _policy_tag(args) -> str:
    if args.no_policy:
        return "no_policy"
    return f"{args.source}_visual_{'off' if args.no_visual else 'on'}_controls_{'on' if args.controls else 'off'}_contrasts_{'on' if args.include_contrasts else 'off'}"


def load_rdm_npz(path: Path) -> np.ndarray:
    """Load an RDM matrix from a .npz file.

    Supports a few conventions:
    - key 'D'
    - key 'rdm'
    - key 'RDM'
    - key 'vec' (condensed upper triangle)
    - otherwise first array in npz
    """
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.keys())

    # Prefer common keys
    for k in ["D", "rdm", "RDM", "d", "matrix", "M"]:
        if k in keys:
            arr = npz[k]
            return np.asarray(arr)

    if "vec" in keys:
        vec = np.asarray(npz["vec"]).astype(float)
        # infer n from vec length: n(n-1)/2 = L
        L = len(vec)
        n = int((1 + np.sqrt(1 + 8 * L)) / 2)
        if n * (n - 1) // 2 != L:
            raise ValueError(f"Cannot infer square size from vec length {L} in {path.name}")
        M = np.full((n, n), np.nan, dtype=float)
        iu, ju = np.triu_indices(n, k=1)
        M[iu, ju] = vec
        M[ju, iu] = vec
        np.fill_diagonal(M, 0.0)
        return M

    if not keys:
        raise ValueError(f"Empty npz: {path}")

    arr = npz[keys[0]]
    return np.asarray(arr)


def vectorize_upper(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got shape {mat.shape}")
    n = mat.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    return mat[iu, ju].astype(float)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation using pandas ranking (handles ties, faster than manual loop)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 3:
        return np.nan

    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float, copy=True)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float, copy=True)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    if denom == 0:
        return np.nan
    return float(np.sum(rx * ry) / denom)


def build_distance_matrix(vectors: Dict[str, np.ndarray], min_overlap: int = 50) -> Tuple[List[str], np.ndarray, np.ndarray]:
    names = sorted(vectors.keys())
    k = len(names)
    sim = np.zeros((k, k), dtype=float)

    for i in range(k):
        sim[i, i] = 1.0
        for j in range(i + 1, k):
            a = vectors[names[i]]
            b = vectors[names[j]]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < int(min_overlap):
                rho = 0.0
            else:
                rho = spearman_rho(a[ok], b[ok])
                if not np.isfinite(rho):
                    rho = 0.0
            sim[i, j] = rho
            sim[j, i] = rho

    # distance = 1 - rho, clip to [0, 2]
    dist = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    return names, sim, dist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2D MDS visualisation of model RDM similarity")
    p.add_argument(
        "--model-rdms-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "derivatives" / "model_rdms_canon"),
        help="Directory containing .npz model RDMs",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "stage2_model_visualisation"),
        help="Output directory (default: analysis_code/results/stage2_model_visualisation)",
    )
    p.add_argument("--timestamped", action="store_true", help="Create timestamped subfolder under the stage folder")
    p.add_argument("--random-state", type=int, default=0, help="Random state for MDS")
    p.add_argument("--mds-n-init", type=int, default=10, help="MDS: number of random initialisations (higher = more stable)")
    p.add_argument(
        "--label-mode",
        type=str,
        default="smart",
        choices=["smart", "all", "none"],
        help="Labeling mode: smart=try repel labels (adjustText if available), all=label everything, none=no labels",
    )
    p.add_argument(
        "--min-overlap",
        type=int,
        default=50,
        help="Minimum overlapping finite entries to compute Spearman rho (else rho=0).",
    )
    p.add_argument("--no-policy", action="store_true", help="Disable model selection policy and load all models")
    p.add_argument(
        "--source",
        type=str,
        choices=["human", "ai", "mixed"],
        default="mixed",
        help="Source mode for model selection policy",
    )
    p.add_argument("--no-visual", action="store_true", help="Exclude visual models from selection policy")
    p.add_argument("--controls", action="store_true", help="Include control models in selection policy")
    p.add_argument("--no-contrasts", dest="include_contrasts", action="store_false",
                   help="Exclude supplementary contrast/category models (food classification, sweet/savory, natural/transformed)")
    p.set_defaults(include_contrasts=True)
    p.add_argument(
        "--include",
        type=str,
        default="",
        help="Comma-separated extra keys to include in selection policy",
    )
    p.add_argument(
        "--exclude",
        type=str,
        default="ll_mean_luminance,ll_rms_contrast,ll_edge_energy,ll_lab_mean,ll_lab_std,ll_power_radial30,ll_hog_pcs10",
        help="Comma-separated keys to exclude in selection policy (default: drop low-information visual models)",
    )
    p.add_argument(
        "--color-by",
        type=str,
        choices=["source", "family"],
        default="family",
        help="Color points by source or family",
    )
    p.add_argument(
        "--nn-seed",
        type=int,
        default=-1,
        help="Which NN seed's component RDMs to include (disabled by default; set to non-negative to include NN components).",
    )
    p.add_argument(
        "--nn-n-comp",
        type=int,
        default=5,
        help="How many NN components to include (default: 5, i.e. comp01–05).",
    )
    p.add_argument("--no-nn", action="store_true", help="Exclude NN component RDMs entirely (NN models are excluded by default).")
    p.add_argument(
        "--seeds",
        type=int,
        default=100,
        help="Number of random seeds for sklearn MDS optimisation (default: 100). Ignored for classical MDS.",
    )
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore any cached MDS coordinates and recompute from scratch.",
    )
    return p.parse_args()


def plot_embedding(meta: pd.DataFrame, title: str, out_png: Path, label_mode: str = "smart", color_by: str = "family") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Normalise labels so AI and human flavour dimensions use the same -ness form.
    # nice_label_from_stem() historically shortened a few AI labels (bitter, sweet, spicy)
    # but kept the human form long (bitterness, sweetness, spiciness).
    _LABEL_NORM = {
        "ai bitter":  "ai bitterness",
        "ai sweet":   "ai sweetness",
        "ai spicy":   "ai spiciness",
        "ai sour":    "ai sourness",
        "ai salty":   "ai saltiness",
        "ai fatty":   "ai fattiness",
    }
    meta = meta.copy()
    meta["label"] = meta["label"].map(lambda lbl: _LABEL_NORM.get(str(lbl), str(lbl)))

    # Colour by SOURCE — consistent with all other figures in the paper:
    #   Human=#377eb8 (blue), AI=#ff7f00 (orange), Visual=#4daf4a (green),
    #   Category/Conceptual=#e41a1c (red).
    # Marker shape encodes DIMENSION FAMILY within each source, providing a
    # secondary visual cue without conflicting with the source colour scheme.
    SOURCE_COLOR_MAP = {
        "ai":    "#ff7f00",   # orange  — matches family-wise and univariate figures
        "human": "#377eb8",   # blue
        "none":  "#4daf4a",   # green   — visual / conceptual models (source="none")
        "other": "#888888",
    }
    CONCEPTUAL_COLOR = "#e41a1c"   # red override for conceptual/food-classification models

    def _point_color(src: str, fam: str) -> str:
        if src == "none" and fam == "conceptual":
            return CONCEPTUAL_COLOR
        return SOURCE_COLOR_MAP.get(src, "#888888")

    # Shape encodes family type so readers can distinguish appraisal vs flavour
    # within each coloured source group.
    marker_map = {
        "appraisal":  "o",   # circle
        "flavour":    "^",   # triangle
        "visual":     "D",   # diamond
        "conceptual": "s",   # square
        "nn":         "X",
        "other":      "P",
    }

    # Keep color_by param for backward compatibility but ignore it — always colour by source
    _ = color_by  # unused

    fig, ax = plt.subplots(figsize=(12, 8))

    # --- Connecting lines between matched AI/human pairs ---
    if "key" in meta.columns:
        ai_rows = meta[meta["source"] == "ai"].copy()
        human_rows = meta[meta["source"] == "human"].copy()
        ai_concept = {re.sub(r"^(?:blind_)?ai_", "", str(k)): row for k, row in ai_rows.set_index("key").iterrows()}
        human_concept = {re.sub(r"^human_", "", str(k)): row for k, row in human_rows.set_index("key").iterrows()}
        for concept, ai_row in ai_concept.items():
            if concept in human_concept:
                h_row = human_concept[concept]
                ax.plot(
                    [float(ai_row["x"]), float(h_row["x"])],
                    [float(ai_row["y"]), float(h_row["y"])],
                    color="grey", linewidth=0.8, alpha=0.4, zorder=0,
                )

    # Determine unique groups for color_by
    groups = meta[color_by].fillna("other").unique()
    groups = sorted(groups)

    # Determine unique sources for marker shapes if available
    if "source" in meta.columns:
        sources = meta["source"].fillna("other").unique()
    else:
        sources = ["other"]

    # Plot points: colour = source, marker = family
    for _, r in meta.iterrows():
        src = str(r.get("source", "other"))
        fam = str(r.get("family", "other"))
        c = _point_color(src, fam)
        m = marker_map.get(fam, "o")
        ax.scatter(float(r["x"]), float(r["y"]), s=100, alpha=0.9, c=c, marker=m, zorder=2)

    # --- Legend: one entry per (source, family) combo actually present ---
    FAMILY_LABEL = {"appraisal": "Appraisal", "flavour": "Sensory", "visual": "Visual",
                    "conceptual": "Conceptual", "nn": "NN", "other": "Other"}
    SOURCE_LABEL = {"ai": "AI", "human": "Human", "none": "", "other": ""}
    LEGEND_ORDER = [  # preferred display order
        ("ai",    "appraisal"), ("ai",    "flavour"),
        ("human", "appraisal"), ("human", "flavour"),
        ("none",  "conceptual"), ("none", "visual"),
    ]
    seen_combos = set()
    legend_elements = []
    # First add in preferred order
    for (src, fam) in LEGEND_ORDER:
        if meta[(meta["source"].fillna("other") == src) & (meta["family"].fillna("other") == fam)].empty:
            continue
        combo = (src, fam)
        if combo in seen_combos:
            continue
        seen_combos.add(combo)
        c = _point_color(src, fam)
        m = marker_map.get(fam, "o")
        src_str = SOURCE_LABEL.get(src, src)
        fam_str = FAMILY_LABEL.get(fam, fam.capitalize())
        label = f"{src_str} {fam_str}".strip()
        legend_elements.append(Line2D([0], [0], marker=m, color="w", label=label,
                                      markerfacecolor=c, markersize=9))
    # Then any remaining combos not in preferred order
    for _, r in meta.iterrows():
        src = str(r.get("source", "other"))
        fam = str(r.get("family", "other"))
        combo = (src, fam)
        if combo in seen_combos:
            continue
        seen_combos.add(combo)
        c = _point_color(src, fam)
        m = marker_map.get(fam, "o")
        src_str = SOURCE_LABEL.get(src, src)
        fam_str = FAMILY_LABEL.get(fam, fam.capitalize())
        label = f"{src_str} {fam_str}".strip()
        legend_elements.append(Line2D([0], [0], marker=m, color="w", label=label,
                                      markerfacecolor=c, markersize=9))

    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=len(legend_elements), frameon=False, fontsize=9)

    if label_mode != "none":
        BASE_DX = 8   # horizontal gap between dot and label text (points)

        # Vertical nudges (points) — add per-label tweaks here once the default layout is known.
        LABEL_DY: dict = {}

        x_vals = meta["x"].dropna().values
        x_mid = float(np.median(x_vals))   # items right of median → label on left, and vice versa

        # Labels forced to a specific side regardless of x position.
        FORCE_LEFT: set = set()
        FORCE_RIGHT: set = set()

        # Full overrides: (dx, dy, ha, va) — use sparingly for tight clusters.
        # Manual de-overlapping (2026-06-12): below = (0, -10, center, top),
        # above = (0, 8, center, bottom), right = (8, 0, left, center).
        LABEL_OVERRIDE: dict = {
            "ai sweetness":       (0, -10, "center", "top"),     # below
            "human sourness":     (0,   8, "center", "bottom"),  # above
            "Intuitive category": (0, -10, "center", "top"),     # below
            "Culinary groups":    ( 8,  0, "left",   "center"),  # right
            "WHO groups":         (8,   0, "left",   "center"),  # right
            "ai saltiness":       (0, -10, "center", "top"),     # below
            "ai healthiness":     (0,   8, "center", "bottom"),  # above
            "ai calorie density": (0, -10, "center", "top"),     # below
        }

        for _, r in meta.iterrows():
            lab = str(r.get("label", ""))
            x = float(r["x"])
            dy = LABEL_DY.get(lab, 0)

            if lab in LABEL_OVERRIDE:
                dx, dy, ha, va = LABEL_OVERRIDE[lab]
            elif x > x_mid or lab in FORCE_LEFT:
                dx, ha, va = -BASE_DX, "right", "center"
            elif x <= x_mid or lab in FORCE_RIGHT:
                dx, ha, va = BASE_DX, "left", "center"
            else:
                dx, ha, va = BASE_DX, "left", "center"

            ax.annotate(
                lab,
                xy=(float(r["x"]), float(r["y"])),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=9,
                alpha=0.9,
            )

    # ax.set_xlabel("Dim 1", fontsize=11)
    # ax.set_ylabel("Dim 2", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_similarity_heatmap(
    sim: np.ndarray,
    names: List[str],
    meta: pd.DataFrame,
    out_png: Path,
    sort_by: str = "family",
) -> None:
    """Plot a sorted similarity matrix heatmap (Spearman rho between model RDMs).

    Rows/columns are sorted by family (then source) so related models cluster
    visually.  A thin colour strip on the left and top encodes family membership.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import BoundaryNorm
    import matplotlib.gridspec as gridspec

    FAMILY_COLORS = {
        "appraisal": "#4e79a7",
        "flavour":   "#f28e2b",
        "visual":    "#59a14f",
        "conceptual":"#e15759",
        "control":   "#b07aa1",
        "nn":        "#76b7b2",
        "other":     "#bab0ac",
    }

    # Build an index lookup: stem -> row in meta
    meta_idx = meta.set_index("stem") if "stem" in meta.columns else meta

    # Determine sort order: source -> family -> label
    # Matches stage3 convention: Human → AI → Conceptual → Visual → NN
    FAMILY_ORDER = ["appraisal", "flavour", "conceptual", "visual", "nn", "control", "other"]
    SOURCE_ORDER = ["human", "ai", "none", "other"]
    def _sort_key(stem: str):
        row = meta_idx.loc[stem] if stem in meta_idx.index else {}
        fam = str(row.get("family", "other")) if hasattr(row, "get") else "other"
        src = str(row.get("source", "other")) if hasattr(row, "get") else "other"
        lbl = str(row.get("label", stem)) if hasattr(row, "get") else stem
        fam_rank = FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else len(FAMILY_ORDER)
        src_rank = SOURCE_ORDER.index(src) if src in SOURCE_ORDER else len(SOURCE_ORDER)
        # human_appeal has no AI counterpart — sort it last within the human block
        # to match stage3 convention (is_appeal sentinel = 1 pushes it down)
        is_appeal = 1 if (src == "human" and "appeal" in lbl) else 0
        return (src_rank, fam_rank, is_appeal, lbl)

    order = sorted(range(len(names)), key=lambda i: _sort_key(names[i]))
    sorted_names = [names[i] for i in order]
    sorted_sim = sim[np.ix_(order, order)]

    # Nice labels for tick marks
    nice = [nice_label_from_stem(n) for n in sorted_names]
    family_labels = []
    for n in sorted_names:
        row = meta_idx.loc[n] if n in meta_idx.index else {}
        fam = str(row.get("family", "other")) if hasattr(row, "get") else "other"
        family_labels.append(fam)

    k = len(sorted_names)
    label_fs = 9  # fixed readable size; bbox_inches='tight' handles overflow
    cell = max(0.42, min(0.52, 12.0 / k))
    heat_size = max(10, k * cell)

    # No family colour strips — label colours carry source information.
    # Layout: [heatmap | colorbar]; add generous height for rotated x labels.
    fig = plt.figure(figsize=(heat_size + 2.8, heat_size + 1.5))
    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[heat_size, 0.45],
        wspace=0.04,
    )
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])

    # --- Heatmap ---
    # viridis (0→1): model-model correlations are almost all positive so a sequential
    # colourmap uses its full range; the old diverging RdBu_r wasted half on negatives.
    im = ax_heat.imshow(
        sorted_sim, aspect="auto", interpolation="nearest",
        cmap="viridis", vmin=0.0, vmax=1.0,
    )

    # Colour tick labels by source/family — matches stage3 ColorBrewer palette:
    #   Human=#377eb8, AI=#ff7f00, Visual=#4daf4a, Conceptual=#e41a1c
    SOURCE_LABEL_COLORS = {"human": "#377eb8", "ai": "#ff7f00", "none": "#4daf4a", "other": "#888888"}
    CONCEPTUAL_COLOR = "#e41a1c"  # red for food-classification / contrast models (matches stage3)

    def _tick_color(stem: str) -> str:
        row = meta_idx.loc[stem] if stem in meta_idx.index else {}
        src = str(row.get("source", "other")) if hasattr(row, "get") else "other"
        fam = str(row.get("family", "other")) if hasattr(row, "get") else "other"
        if src == "none" and fam == "conceptual":
            return CONCEPTUAL_COLOR
        return SOURCE_LABEL_COLORS.get(src, "#888888")

    tick_colors = [_tick_color(n) for n in sorted_names]

    # Also track source_labels for divider logic (visual + conceptual both have src="none")
    def _group_key(stem: str) -> str:
        """Return a grouping key for drawing thick dividers between Human/AI/non-rating blocks."""
        row = meta_idx.loc[stem] if stem in meta_idx.index else {}
        src = str(row.get("source", "other")) if hasattr(row, "get") else "other"
        fam = str(row.get("family", "other")) if hasattr(row, "get") else "other"
        if src in ("human", "ai"):
            return src
        return f"none_{fam}"  # distinguish visual from conceptual within source="none"

    group_labels = [_group_key(n) for n in sorted_names]

    ax_heat.set_xticks(range(k))
    ax_heat.set_xticklabels(nice, rotation=90, ha="center", fontsize=label_fs)
    ax_heat.set_yticks(range(k))
    ax_heat.set_yticklabels(nice, fontsize=label_fs)
    for tick, col in zip(ax_heat.get_xticklabels(), tick_colors):
        tick.set_color(col)
    for tick, col in zip(ax_heat.get_yticklabels(), tick_colors):
        tick.set_color(col)
    ax_heat.tick_params(axis="both", length=0)
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")

    # Thin cell grid lines
    for v in np.arange(-0.5, k, 1):
        ax_heat.axhline(v, color="white", linewidth=0.3, alpha=0.4)
        ax_heat.axvline(v, color="white", linewidth=0.3, alpha=0.4)

    # Thicker dividers at group boundaries (Human | AI | Conceptual | Visual)
    for i in range(1, k):
        if group_labels[i] != group_labels[i - 1]:
            ax_heat.axhline(i - 0.5, color="white", linewidth=2.0, alpha=0.95)
            ax_heat.axvline(i - 0.5, color="white", linewidth=2.0, alpha=0.95)

    # --- Colorbar ---
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label("Spearman ρ", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # --- Legend: source/family label colours — matches stage3 ColorBrewer palette ---
    source_patches = [
        mpatches.Patch(color="#377eb8", label="Human"),
        mpatches.Patch(color="#ff7f00", label="AI"),
        mpatches.Patch(color="#e41a1c", label="Category"),
        mpatches.Patch(color="#4daf4a", label="Visual"),
    ]
    ax_heat.legend(
        handles=source_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
        fontsize=10,
    )

    fig.suptitle("Model RDM similarity (Spearman ρ)", fontsize=12, y=1.01)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()

    model_dir = Path(args.model_rdms_dir).expanduser().resolve()
    if not model_dir.exists():
        print(f"[ERROR] Model RDM directory not found: {model_dir}")
        return 1

    base_out = Path(args.outdir).expanduser().resolve()
    outdir = base_out / _policy_tag(args)
    if args.timestamped:
        outdir = outdir / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(model_dir.glob("*.npz"))
    if not files:
        print(f"[ERROR] No .npz files found under {model_dir}")
        return 1

    # By default, exclude NN model files from the base load.
    # They are only added back later if explicitly requested via --nn-seed >= 0 and not --no-nn.
    include_nn = (not args.no_nn) and (args.nn_seed >= 0)
    if not include_nn:
        files = [fp for fp in files if infer_source_tag(fp.stem) != "nn"]

    # Deduplicate by resolved key: if multiple files share a key, prefer the
    # canonical rdm_concept_* version over legacy capitalised files.
    seen_keys: dict = {}
    for fp in files:
        key = _stem_to_key(fp.stem)
        if key not in seen_keys:
            seen_keys[key] = fp
        else:
            # Prefer the file whose stem starts with "rdm_concept_" or "rdm_ll_" (canonical)
            existing = seen_keys[key]
            if fp.stem.startswith("rdm_concept_") or (
                not existing.stem.startswith("rdm_concept_") and fp.stem < existing.stem
            ):
                print(f"[INFO] Dedup: preferring {fp.name} over {existing.name} (same key: {key!r})")
                seen_keys[key] = fp
            else:
                print(f"[INFO] Dedup: skipping {fp.name} (duplicate of {existing.name}, key: {key!r})")
    files = list(seen_keys.values())

    if not args.no_policy and not _HAS_POLICY:
        print("[WARN] Could not import policy helpers from analysis_helpers; proceeding as --no-policy. Family labels will default to 'other', so color-by-family may appear grey.")
        args.no_policy = True

    if not args.no_policy:
        specs = select_models(
            source_mode=args.source,
            include_visual=not args.no_visual,
            include_controls=args.controls,
            include_supplementary_contrasts=args.include_contrasts,
            extra_include=_parse_csv_list(args.include),
            extra_exclude=_parse_csv_list(args.exclude),
        )
        spec_by_key = {s.key.lower(): s for s in specs}  # lowercase to match _stem_to_key output
        # Filter files by keys in spec_by_key
        filtered_files = []
        for fp in files:
            key = _stem_to_key(fp.stem)
            if key in spec_by_key:
                filtered_files.append(fp)
        files = filtered_files
        if not include_nn:
            files = [fp for fp in files if infer_source_tag(fp.stem) != "nn"]
        if not files:
            print(f"[ERROR] No model RDM files matched selection policy in {model_dir}")
            return 1

    # Inject NN component RDMs only if explicitly requested (unless --no-nn or --nn-seed < 0)
    nn_files: List[Path] = []
    if include_nn:
        nn_stems_already = {fp.stem for fp in files}
        n_already = 0
        for k in range(1, args.nn_n_comp + 1):
            nn_fp = model_dir / f"rdm_nn_seed{args.nn_seed}_comp{k:02d}.npz"
            if not nn_fp.exists():
                continue
            if nn_fp.stem in nn_stems_already:
                n_already += 1
            else:
                nn_files.append(nn_fp)
        if nn_files:
            print(f"[INFO] Injecting {len(nn_files)} NN component RDM(s) for seed {args.nn_seed}")
        elif n_already > 0:
            print(f"[INFO] {n_already} NN component RDM(s) for seed {args.nn_seed} already included in full load")
        else:
            print(f"[WARN] No NN component RDMs found for seed {args.nn_seed} in {model_dir} — run results0b with --export-best-to-model-rdms")

    print(f"[INFO] Loading {len(files) + len(nn_files)} model RDM(s) from: {model_dir}")

    vectors: Dict[str, np.ndarray] = {}
    meta_rows: List[Dict[str, str]] = []

    for fp in files + nn_files:
        stem = fp.stem
        try:
            M = load_rdm_npz(fp)
            v = vectorize_upper(M)
            vectors[stem] = v
            source = infer_source_tag(stem)
            if source == "nn":
                family = "nn"
                role = "primary"
            elif not args.no_policy:
                key = _stem_to_key(stem)
                spec = spec_by_key.get(key)
                if spec is None:
                    family = "other"
                    role = "supplementary"
                else:
                    family = spec.family
                    source = spec.source
                    role = spec.role
            else:
                family = "other"
                role = "supplementary"
            meta_rows.append(
                {
                    "stem": stem,
                    "key": _stem_to_key(stem),
                    "file": fp.name,
                    "source": source,
                    "label": nice_label_from_stem(stem),
                    "family": family,
                    "role": role,
                }
            )
        except Exception as e:
            print(f"[WARN] Skipping {fp.name}: {e}")

    if len(vectors) < 3:
        print("[ERROR] Need at least 3 valid model RDMs to run MDS.")
        return 1

    names, sim, dist = build_distance_matrix(vectors, min_overlap=int(args.min_overlap))

    # Save similarity matrix
    sim_df = pd.DataFrame(sim, index=names, columns=names)
    sim_path = outdir / "model_similarity_matrix.csv"
    sim_df.to_csv(sim_path, index=True)
    print(f"[OK] Wrote: {sim_path}")

    # -----------------------------------------------------------------------
    # Classical MDS fallback (numpy-only) — used when sklearn is unavailable.
    # Defined here (before stress curve) so both sections can reference them.
    # -----------------------------------------------------------------------

    # sklearn MDS compatibility: `init` and `normalized_stress` were added in
    # newer versions.  Build kwargs dynamically so the code works on any version.
    def _mds_kwargs(n_components: int, random_state: int, **extra) -> dict:
        """Return MDS constructor kwargs compatible with the installed sklearn."""
        kw: dict = dict(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=1,
            max_iter=3000,
            metric=True,
        )
        try:
            import sklearn
            from packaging.version import Version
            _ver = Version(sklearn.__version__)
        except Exception:
            _ver = None
        # normalized_stress added in sklearn 1.2
        try:
            from sklearn.manifold import MDS as _MDS_test
            import inspect
            _sig = inspect.signature(_MDS_test.__init__)
            if "normalized_stress" in _sig.parameters:
                kw["normalized_stress"] = "auto"
            if "init" in _sig.parameters:
                kw["init"] = "random"
        except Exception:
            pass
        kw.update(extra)
        return kw

    def _classical_mds(D, n_components=2):
        """Return (coords, sorted_eigenvalues) via double-centering of D²."""
        n = D.shape[0]
        D2 = D ** 2
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ D2 @ H
        vals, vecs = np.linalg.eigh(B)
        idx = np.argsort(vals)[::-1]
        vals, vecs = vals[idx], vecs[:, idx]
        coords = vecs[:, :n_components] * np.sqrt(np.maximum(vals[:n_components], 0))
        return coords, vals

    def _normalised_stress(D, coords):
        """Kruskal normalised stress between dissimilarity matrix D and coords."""
        d_hat = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
        mask = np.triu(np.ones(D.shape, dtype=bool), k=1)
        d, dh = D[mask], d_hat[mask]
        return float(np.sqrt(np.sum((d - dh) ** 2) / np.sum(dh ** 2)))

    # --- Dimensionality check: stress curve for 1–5 dimensions ---
    print("[INFO] Computing MDS stress curve (1–5 dimensions) …")
    try:
        _use_sklearn_sc = True
        try:
            from sklearn.manifold import MDS as _MDS
        except Exception:
            _use_sklearn_sc = False

        rng_sc = np.random.default_rng(int(args.random_state))
        stress_rows = []
        prev_stress = None
        for n_dim in range(1, 6):
            if _use_sklearn_sc:
                sc_seeds = rng_sc.integers(0, 2**31, size=20).tolist()
                best_s = np.inf
                for seed in sc_seeds:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
                        _mds = _MDS(**_mds_kwargs(n_components=n_dim, random_state=int(seed)))
                        _mds.fit(dist)
                    if _mds.stress_ < best_s:
                        best_s = _mds.stress_
            else:
                # Classical MDS fallback for stress curve
                c_sc, _ = _classical_mds(dist, n_components=n_dim)
                best_s = _normalised_stress(dist, c_sc)
            drop = (prev_stress - best_s) if prev_stress is not None else float("nan")
            stress_rows.append({"dims": n_dim, "stress": best_s, "stress_drop": drop})
            print(f"  {n_dim}D  stress = {best_s:.6f}"
                  + (f"  (drop = {drop:.4f})" if not np.isnan(drop) else ""))
            prev_stress = best_s
        stress_df = pd.DataFrame(stress_rows)
        stress_path = outdir / "model_mds_stress_curve.csv"
        stress_df.to_csv(stress_path, index=False)
        print(f"[OK] Wrote stress curve: {stress_path}")

        # Plot stress curve
        import matplotlib.pyplot as _plt
        fig_sc, ax_sc = _plt.subplots(figsize=(5, 3.5))
        ax_sc.plot(stress_df["dims"], stress_df["stress"], "o-", color="steelblue", lw=1.5)
        ax_sc.set_xlabel("Number of dimensions", fontsize=11)
        ax_sc.set_ylabel("Normalised stress", fontsize=11)
        ax_sc.set_xticks(stress_df["dims"])
        ax_sc.set_title("Model MDS stress curve", fontsize=11)
        fig_sc.tight_layout()
        sc_png = outdir / "model_mds_stress_curve.png"
        fig_sc.savefig(sc_png, dpi=150)
        _plt.close(fig_sc)
        print(f"[OK] Wrote stress curve plot: {sc_png}")
    except Exception as e:
        print(f"[WARN] Stress curve skipped: {e}")

    # ── Coordinate cache ────────────────────────────────────────────────────────
    # Key the cache on an MD5 hash of the distance matrix so the stored solution
    # is reused whenever the input data hasn't changed.  Pass --force-recompute
    # to ignore the cache (e.g. after changing the model set deliberately).
    import hashlib, json as _json
    _dist_hash = hashlib.md5(dist.tobytes()).hexdigest()[:16]
    _cache_path = outdir / "model_mds_coords_cache.json"

    def _load_cache():
        if not _cache_path.exists():
            return None
        try:
            cached = _json.loads(_cache_path.read_text())
            if cached.get("dist_hash") != _dist_hash:
                print("[INFO] MDS cache exists but input data changed — recomputing.")
                return None
            coords = np.array(cached["coords"])
            stress = float(cached["stress"])
            print(f"[INFO] Loaded cached MDS coordinates (hash={_dist_hash}, stress={stress:.6f}).")
            return coords, stress
        except Exception as e:
            print(f"[WARN] Cache load failed ({e}) — recomputing.")
            return None

    def _save_cache(coords, stress):
        try:
            _cache_path.write_text(_json.dumps({
                "dist_hash": _dist_hash,
                "stress": float(stress),
                "coords": coords.tolist(),
            }))
            print(f"[INFO] MDS coordinates cached to {_cache_path.name}")
        except Exception as e:
            print(f"[WARN] Could not write cache: {e}")

    # Run MDS — prefer sklearn (iterative refinement); fall back to classical MDS
    _sklearn_available = True
    try:
        from sklearn.manifold import MDS
    except Exception:
        _sklearn_available = False
        print("[WARN] scikit-learn not found — using classical MDS (numpy fallback). "
              "Install scikit-learn for iterative metric MDS: pip install scikit-learn")

    # Try cache first (unless --force-recompute)
    _cached = None if args.force_recompute else _load_cache()

    if _cached is not None:
        best_coords, best_stress = _cached
    elif _sklearn_available:
        n_seeds = max(int(args.seeds), int(args.mds_n_init), 10)
        rng = np.random.default_rng(int(args.random_state))
        seeds = rng.integers(0, 2**31, size=n_seeds).tolist()
        best_coords, best_stress = None, np.inf
        print(f"[INFO] Running sklearn metric MDS with {n_seeds} random seeds; "
              "keeping lowest-stress solution …")
        for seed in seeds:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
                mds = MDS(**_mds_kwargs(n_components=2, random_state=int(seed)))
                c = mds.fit_transform(dist)
            if mds.stress_ < best_stress:
                best_stress = mds.stress_
                best_coords = c
        _save_cache(best_coords, best_stress)
    else:
        # Classical MDS is deterministic — run once (seeds are irrelevant)
        print("[INFO] Running classical MDS (numpy) …")
        best_coords, _ = _classical_mds(dist, n_components=2)
        best_stress = _normalised_stress(dist, best_coords)
        _save_cache(best_coords, best_stress)

    coords_mds = best_coords
    print(f"[INFO] Best normalised stress: {best_stress:.6f}")

    # Canonicalise axis orientation so the figure is stable across runs:
    #   • visual models (green) anchor to the LEFT  (negative x)
    #   • appraisal models (healthiness, calorie density) sit at the TOP (positive y)
    meta_tmp = pd.DataFrame(meta_rows).set_index("stem").reindex(names)
    vis_idx = [i for i, n in enumerate(names) if meta_tmp.loc[n, "family"] == "visual"] if "family" in meta_tmp.columns else []
    apr_idx = [i for i, n in enumerate(names) if meta_tmp.loc[n, "family"] == "appraisal"] if "family" in meta_tmp.columns else []
    if vis_idx and coords_mds[vis_idx, 0].mean() > 0:
        coords_mds[:, 0] *= -1   # flip x so visual cluster is on the left
    if apr_idx and coords_mds[apr_idx, 1].mean() < 0:
        coords_mds[:, 1] *= -1   # flip y so appraisal cluster is at the top

    # Merge meta for MDS
    meta = pd.DataFrame(meta_rows)
    meta = meta.set_index("stem").reindex(names).reset_index()
    meta_mds = meta.copy()
    meta_mds["x"] = coords_mds[:, 0]
    meta_mds["y"] = coords_mds[:, 1]

    # Save MDS coords
    coords_mds_path = outdir / "model_mds_2d_coords.csv"
    meta_mds.to_csv(coords_mds_path, index=False)
    print(f"[OK] Wrote: {coords_mds_path}")

    # Plot MDS
    fig_mds_path = outdir / "model_mds_2d.png"
    plot_embedding(meta_mds, title="Model RDM similarity (2D MDS)", out_png=fig_mds_path, label_mode=args.label_mode, color_by=args.color_by)
    print(f"[OK] Wrote: {fig_mds_path}")

    # Plot similarity heatmap — same information as MDS but shows all pairwise rho values
    # directly, without the distortion introduced by collapsing to 2 dimensions.
    fig_heatmap_path = outdir / "model_similarity_heatmap.png"
    plot_similarity_heatmap(sim, names, meta, fig_heatmap_path)
    print(f"[OK] Wrote: {fig_heatmap_path}")

    # Write manifest JSON
    manifest = {
        "policy": {
            "no_policy": args.no_policy,
            "source": args.source,
            "no_visual": args.no_visual,
            "controls": args.controls,
            "include_contrasts": args.include_contrasts,
            "include": _parse_csv_list(args.include),
            "exclude": _parse_csv_list(args.exclude),
        },
        "num_files_found": len(list(model_dir.glob("*.npz"))),
        "num_files_loaded": len(vectors),
        "min_overlap": args.min_overlap,
        "label_mode": args.label_mode,
        "random_state": args.random_state,
        "mds_n_init": args.mds_n_init,
        "color_by": args.color_by,
        "models": [
            {
                "stem": r["stem"],
                "key": r.get("key", ""),
                "file": r["file"],
                "label": r["label"],
                "source": r["source"],
                "family": r["family"],
                "role": r["role"],
            }
            for r in meta_rows
        ],
    }
    manifest["outputs"] = {
        "model_similarity_matrix_csv": str(sim_path),
        "model_mds_2d_coords_csv":     str(coords_mds_path),
        "model_mds_2d_png":            str(fig_mds_path),
        "model_similarity_heatmap_png": str(fig_heatmap_path),
    }
    manifest_path = outdir / "model_visualisation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Wrote: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())