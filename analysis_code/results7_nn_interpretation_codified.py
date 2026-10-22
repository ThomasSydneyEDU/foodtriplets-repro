#!/usr/bin/env python3
"""results7_nn_interpretation_codified.py

Stage 7 (Results): Vision-grounded interpretation of the 24 SPoSE components.

SPECTRAL INTERPRETATION STRATEGY:
  1. Provide the LLM with the FULL rank-ordered list of 350 food names.
  2. Provide 5 tiered image bands (Top, 75th, 50th, 25th, Lowest).
  3. LLM Task: Semantic Hypothesis (Names) -> Visual Refinement (Images).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# =============================================================================
# Paths
# =============================================================================
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NN_DIR_TEMPLATE = HERE / "derivatives" / "nn_runs" / "seed_{seed}"
INFO_CSV = HERE / "reference" / "Foodpictures_information_dynamic.csv"
IMG_DIR = REPO / "expt_stimuli" / "images"
OUT_DIR = HERE / "results" / "stage7_nn_interpretation"
CACHE_PATH = OUT_DIR / "ai_interpretations_cache.json"


# =============================================================================
# Plotting / analysis constants
# =============================================================================
N_TOP_PLOT = 10
FDR_ALPHA = 0.05
FIG_VMAX = 0.75


# =============================================================================
# Prompt
# =============================================================================
SYSTEM_INSTRUCTION = """You are an expert in food perception and human psychology. You are analyzing a single component (latent dimension) from a SPoSE (Sparse Positive Similarity Embedding) model.

TECHNICAL CONTEXT:
- SPoSE is a non-negative and sparse model.
- FEATURE DETECTOR: Highly sparse (e.g. 70%+ of items are at or near zero). Only the top items share a specific property. The median and lowest items are just a random "background" of foods that lack this property.
- CONTINUOUS AXIS: Low sparsity (e.g. <30% at zero). There is a meaningful gradient where the property transitions into a meaningful opposite (e.g. "Juicy" vs "Dry" or "Raw" vs "Cooked").

TASK:
1. PHASE 1 (Semantic & Statistical): Review the numerical distribution statistics and the ordered list of 350 food names. 
    - If the stats show high sparsity and the median/bottom names are a diverse mix sharing no property other than *lacking* the top feature, classify as a "detector".
    - If the loadings decrease smoothly and the names show a clear transition from one pole to its opposite, classify as an "axis".
2. PHASE 2 (Visual): Review the 5 tiered groups of images to confirm your classification.
3. FINAL ANALYSIS: 
    - Explain the ordering principle.
    - Classify the type: "detector" (categorical presence/absence) or "axis" (dimensional gradient).
    - Assess the "Visual Weight": Did the images reveal a "hidden" perceptual axis (e.g. color, shape) that the semantic names didn't capture?
    - LABELING RULE: 
        - If "detector": provide a 2-4 word categorical label naming the specific feature (e.g., "Leafy Green Vegetables").
        - If "axis": provide a description framed as a continuum/gradient (e.g., "Gradient from Raw to Cooked" or "Continuum of Juicy to Dry"). WARNING: Look closely at the lowest items to find the true common denominator for the opposing pole.

Return your response in JSON format.
"""

CLOSING_PROMPT_TEMPLATE = """Analyze the SPoSE component {comp_id} (rank {rank} of {total}).

Return ONLY valid JSON with exactly these keys:
- names_hypothesis: string, hypothesis based on names + loadings
- visual_refinement: string, how the 15 images influenced your view
- visual_weight: "high" | "medium" | "low"
- relationship_type: "detector" | "axis"
- ordering_explanation: string, final explanation of the sequence
- label: string, 3-5 word label
- confidence: "high" | "medium" | "low"
"""


# =============================================================================
# Gemini client
# =============================================================================
def get_gemini_client():
    """Vertex AI (preferred) or AI Studio API key — same logic as PAFID's run_qc.py."""
    try:
        from google import genai  # type: ignore
    except Exception:
        raise RuntimeError(
            "Missing google-genai. Install with: pip install -U google-genai"
        )

    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"true", "1", "yes"}
    if use_vertex:
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        if not project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is not set. Export it first, e.g. 'export GOOGLE_CLOUD_PROJECT=usyd-llm'"
            )
        print(f"[INFO] Using Gemini via Vertex AI: project={project}, location={location}")
        return genai.Client(vertexai=True, project=project, location=location)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Gemini credentials: set GOOGLE_GENAI_USE_VERTEXAI=True (+ GOOGLE_CLOUD_PROJECT) "
            "for Vertex AI, or export GEMINI_API_KEY for AI Studio."
        )
    return genai.Client(api_key=api_key)


def _backoff(attempt: int) -> None:
    time.sleep(min(60, 2 ** attempt) + 0.25 * attempt)


def _extract_text(response: Any) -> str:
    txt = getattr(response, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    for cand in getattr(response, "candidates", []) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", []) or []:
            t = getattr(part, "text", None)
            if isinstance(t, str) and t.strip():
                return t.strip()
    raise RuntimeError("Could not extract text from Gemini response.")


def _strip_json_fences(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        lines = t.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    if "{" in t and "}" in t:
        t = t[t.find("{"): t.rfind("}") + 1].strip()
    return t


# =============================================================================
# Image helpers
# =============================================================================
def _load_image_part(image_path: Path) -> Dict[str, Any]:
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    data_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return {"inline_data": {"mime_type": mime, "data": data_b64}}


# =============================================================================
# Anchor selection
# =============================================================================
def pick_spectral_anchors(
    merged: pd.DataFrame,
    comp: str,
    n_per_band: int,
) -> Dict[str, Any]:
    """Return a dict with 5 image bands, the full list of names, and distribution stats."""
    df = merged[["filename", "food", comp]].dropna()
    df = df.assign(_loading=df[comp].astype(float))
    
    # Calculate distribution stats
    loadings = df["_loading"].values
    n = len(loadings)
    max_val = np.max(loadings)
    # Sparsity: % items < 5% of max loading (effectively zero/noise)
    sparsity = (loadings < 0.05 * max_val).sum() / n
    # Top-Heavy Ratio: Mean of Top 10 / Mean of All
    top_10_mean = np.mean(np.sort(loadings)[::-1][:10])
    avg_mean = np.mean(loadings)
    top_ratio = top_10_mean / avg_mean if avg_mean > 0 else 0
    med_val = np.median(loadings)

    df_sorted = df.sort_values("_loading", ascending=False).reset_index(drop=True)

    def _pack(indices):
        out = []
        for idx in indices:
            r = df_sorted.iloc[idx]
            out.append({"filename": str(r["filename"]), "loading": float(r["_loading"]), "food": str(r["food"])})
        return out

    # 1. Full ordered names list with loadings
    names_list = [f"{i+1}. {r['food']} (Loading: {r['_loading']:.4f})" for i, r in df_sorted.iterrows()]

    # 2. 5 bands of images (3 per band typically)
    bands = {
        "top": _pack(range(n_per_band)),
        "p75": _pack(range(n // 4, n // 4 + n_per_band)),
        "p50": _pack(range(n // 2 - n_per_band // 2, n // 2 + n_per_band // 2 + (n_per_band % 2))),
        "p25": _pack(range(3 * n // 4, 3 * n // 4 + n_per_band)),
        "lowest": _pack(range(n - n_per_band, n)),
    }

    return {
        "names_list": names_list,
        "bands": bands,
        "stats": {
            "sparsity": sparsity,
            "top_ratio": top_ratio,
            "median_value": med_val,
            "max_value": max_val
        }
    }


def spectral_fingerprint(spectral_data: Dict[str, Any]) -> str:
    """Stable hash of the spectral data structure."""
    bands = spectral_data["bands"]
    payload = json.dumps({
        "names": spectral_data["names_list"][:10] + spectral_data["names_list"][-10:], # Check ends for stability
        "top": [a["filename"] for a in bands["top"]],
        "p75": [a["filename"] for a in bands["p75"]],
        "p50": [a["filename"] for a in bands["p50"]],
        "p25": [a["filename"] for a in bands["p25"]],
        "lowest": [a["filename"] for a in bands["lowest"]],
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


# =============================================================================
# Cache
# =============================================================================
def load_cache() -> Dict[str, Any]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception as e:
            print(f"[WARN] Could not parse cache at {CACHE_PATH}: {e}. Starting fresh.")
    return {}


def save_cache(cache: Dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


# =============================================================================
# Gemini call for one component
# =============================================================================
def query_gemini_for_component(
    client: Any,
    comp_id: str,
    rank: int,
    total: int,
    spectral_data: Dict[str, Any],
    model: str,
    max_attempts: int = 4,
) -> Dict[str, Any]:
    """Call Gemini with stats, names, and images."""
    parts: List[Dict[str, Any]] = []

    # 1. Distribution Stats
    s = spectral_data["stats"]
    stats_text = (
        f"DISTRIBUTION STATISTICS for component {comp_id}:\n"
        f"- Sparsity: {s['sparsity']:.1%} of foods have essentially zero loading.\n"
        f"- Top-Heavy Ratio: The Top 10 items are {s['top_ratio']:.1f}x more influential than the average food.\n"
        f"- Median Loading: {s['median_value']:.4f} (Max loading is {s['max_value']:.4f})\n\n"
        "CLASSIFICATION GUIDE:\n"
        "- If Sparsity > 70% and Median is near zero, it is likely a FEATURE DETECTOR.\n"
        "- If Sparsity < 40% and Median is significant, it is likely a CONTINUOUS AXIS.\n\n"
    )
    parts.append({"text": stats_text})

    # 2. Names List
    names_text = "FULL ORDERED LIST OF 350 FOOD NAMES (Rank 1 to 350):\n" + "\n".join(spectral_data["names_list"])
    parts.append({"text": names_text})

    # 2. Images
    def _append_band(header: str, anchors: List[Dict[str, Any]]) -> None:
        parts.append({"text": f"\n\n--- {header} ---"})
        for a in anchors:
            img_path = IMG_DIR / a["filename"]
            if not img_path.exists():
                continue
            parts.append(_load_image_part(img_path))

    bands = spectral_data["bands"]
    _append_band("TOP IMAGES (Ranks 1-3)", bands["top"])
    _append_band("75th PERCENTILE IMAGES", bands["p75"])
    _append_band("50th PERCENTILE (MEDIAN) IMAGES", bands["p50"])
    _append_band("25th PERCENTILE IMAGES", bands["p25"])
    _append_band("LOWEST IMAGES (Ranks 348-350)", bands["lowest"])

    parts.append({
        "text": "\n\n" + CLOSING_PROMPT_TEMPLATE.format(
            comp_id=comp_id, rank=rank, total=total
        )
    })

    contents = [{"role": "user", "parts": parts}]
    config = {
        "system_instruction": SYSTEM_INSTRUCTION,
        "response_mime_type": "application/json",
        "temperature": 0.0,
    }

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config
            )
            raw = _extract_text(response)
            parsed = json.loads(_strip_json_fences(raw))
            # Ensure keys exist
            for key in ("label", "relationship_type", "confidence", "visual_weight", "names_hypothesis", "ordering_explanation"):
                parsed.setdefault(key, "")
            return parsed
        except Exception as e:
            print(f"  [WARN] Gemini attempt {attempt + 1}/{max_attempts} failed for "
                  f"{comp_id}: {e}")
            if attempt < max_attempts - 1:
                _backoff(attempt)
    raise RuntimeError(f"Gemini failed for {comp_id} after {max_attempts} attempts")


# =============================================================================
# Tier assignment (variance-rank-based; data-driven cutoff candidates are an
# obvious future refinement, but kept simple here)
# =============================================================================
def get_tier(rank: int) -> str:
    if rank <= 8:
        return "Tier 1: Core Categories"
    if rank <= 13:
        return "Tier 2: Form & Texture"
    return "Tier 3: Perceptual Detail"


# =============================================================================
# Stats helpers
# =============================================================================
def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    if n == 0:
        return p_values.copy()
    order = np.argsort(p_values)
    ranks = np.argsort(order) + 1
    adjusted = p_values * n / ranks
    adj_sorted = adjusted[order]
    for i in range(n - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    out = np.empty(n)
    out[order] = adj_sorted
    return np.minimum(out, 1.0)


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=20, help="SPoSE seed to interpret (default 20 = best-by-centrality, 2026-09 retrain)")
    p.add_argument("--model", type=str, default="gemini-2.5-pro",
                   help="Gemini model name (default gemini-2.5-pro)")
    p.add_argument("--n-per-band", type=int, default=3,
                   help="Number of anchor images per band. Default 3.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-query Gemini for every component (ignore cache).")
    p.add_argument("--no-llm", action="store_true",
                   help="Use only cached labels; never call Gemini.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N components (for smoke testing).")
    args = p.parse_args()

    nn_dir = Path(str(NN_DIR_TEMPLATE).format(seed=args.seed))
    scores_path = nn_dir / "nn_scores.csv"
    if not scores_path.exists():
        print(f"[ERROR] nn_scores.csv not found at {scores_path}", file=sys.stderr)
        return 1
    if not INFO_CSV.exists():
        print(f"[ERROR] Canonical info CSV not found at {INFO_CSV}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading scores from {scores_path}")
    scores = pd.read_csv(scores_path)
    info = pd.read_csv(INFO_CSV)
    info.columns = [str(c).strip() for c in info.columns]
    kept = [c for c in scores.columns if c.startswith("nn_comp")]
    merged = pd.merge(scores[["filename"] + kept], info, on="filename")
    variances = merged[kept].var().sort_values(ascending=False)
    rank_map = {c: i + 1 for i, c in enumerate(variances.index)}
    n_total = len(variances)
    print(f"[INFO] {n_total} components found. Variance range: "
          f"{variances.iloc[0]:.3f} (top) to {variances.iloc[-1]:.4f} (bottom).")

    # Filter to only human rating columns.
    # Exclude PAFID dimensions collected in the database but not used in this study
    # (e.g. familiarity), so component "best rating" correlations match the manuscript.
    EXCLUDE_RATINGS = {"human_familiarity"}
    rating_cols = [
        c for c in info.columns
        if c.startswith("human_") and c not in EXCLUDE_RATINGS
    ]

    cache = load_cache()
    client = None
    if not args.no_llm:
        try:
            client = get_gemini_client()
            print(f"[INFO] Gemini client ready (model: {args.model}).")
        except Exception as e:
            print(f"[WARN] No Gemini client ({e}). Falling back to cache only.")
            client = None

    # Iterate components in variance-rank order
    components_in_order = list(variances.index)
    if args.limit is not None:
        components_in_order = components_in_order[: args.limit]
        print(f"[INFO] --limit {args.limit}: only processing first {args.limit} components.")

    rows: List[Dict[str, Any]] = []
    for comp in components_in_order:
        rank = rank_map[comp]

        # ----- Anchors --------------------------------------------------------
        spectral_data = pick_spectral_anchors(merged, comp, args.n_per_band)
        fp = spectral_fingerprint(spectral_data)
        cache_key = f"seed_{args.seed}/{comp}_spectral"
        cached = cache.get(cache_key)

        # ----- AI label (cache or query) --------------------------------------
        ai_record: Dict[str, Any] = {}
        if cached and cached.get("fingerprint") == fp and not args.overwrite:
            ai_record = cached
            print(f"[CACHE] Rank {rank:>2}  {comp:<11} → using cached label.")
        elif client is None:
            print(f"[SKIP ] Rank {rank:>2}  {comp:<11} → no cache and no API client; placeholder.")
            ai_record = {"label": "(label pending — re-run with --overwrite and GEMINI_API_KEY set)",
                         "relationship_type": "", "confidence": "", "visual_weight": "",
                         "fingerprint": fp, "model": "", "timestamp": int(time.time())}
        else:
            print(f"[QUERY] Rank {rank:>2}  {comp:<11} → Gemini ({args.model}); 350 names + 15 images.")
            parsed = query_gemini_for_component(client, comp, rank, n_total, spectral_data, args.model)
            ai_record = {
                **parsed,
                "fingerprint": fp,
                "model": args.model,
                "timestamp": int(time.time()),
            }
            cache[cache_key] = ai_record
            save_cache(cache)

        # ----- Best rating + ρ ------------------------------------------------
        corrs = {}
        for r in rating_cols:
            rho, _ = spearmanr(merged[comp], merged[r], nan_policy="omit")
            corrs[r] = rho
        best_r = max(corrs, key=lambda k: abs(corrs[k]))
        max_rho = corrs[best_r]

        def _get_names(anchors, n=2):
            return ", ".join([a["food"] for a in anchors[:n]])

        rows.append({
            "Rank": rank,
            "Component": comp,
            "LLM Interpretation": ai_record.get("label", ""),
            "Type": ai_record.get("relationship_type", ""),
            "Sparsity": round(spectral_data["stats"]["sparsity"], 3),
            "Visual Weight": ai_record.get("visual_weight", ""),
            "Top Loading": _get_names(spectral_data["bands"]["top"], 2),
            "Lowest": _get_names(spectral_data["bands"]["lowest"], 2),
            "Best Rating": best_r.replace("human_", ""),
            "Rho": round(float(max_rho), 2),
        })

    df_final = pd.DataFrame(rows).sort_values("Rank")

    out_csv = OUT_DIR / "spose_interpretation_codified.csv"
    df_final.to_csv(out_csv, index=False)
    print(f"[OK] Wrote: {out_csv}")

    md_lines: List[str] = []
    md_lines.append("**Table 2. Structural inventory of SPoSE components.** Components are partitioned by their LLM-assisted classification (continuous axis vs feature detector) and ranked by item-wise variance within each group. LLM annotations were generated by Gemini 2.5 Pro from the full 350-item rank-ordered name list with numerical loadings plus anchor images across five percentile bands. Each component is classified as either a **continuous axis** (a meaningful gradient between two semantically opposed poles) or a **feature detector** (a sparse positive feature whose low-loading items primarily lack the property, reflecting SPoSE's non-negativity). \"Visual Weight\" (Vis. Wt) indicates how much the images added beyond the names alone. The annotations are descriptive aids rather than independent statistical evidence. Continuous axes predominate among the highest-variance components and capture broad gradients of food perception. Feature detectors become dominant as variance declines, encoding progressively narrower perceptual and categorical features that are not well captured by the candidate rating dimensions.\n\n")
    md_lines.append("| Rank | Top Foods | Lowest Foods | LLM Annotation | Best human rating (max ρ) | Vis. Wt |\n")
    md_lines.append("|---|---|---|---|---|---|\n")

    # Filter into groups
    axes_df = df_final[df_final["Type"] == "axis"].copy()
    detectors_df = df_final[df_final["Type"] == "detector"].copy()

    md_lines.append(f"| <td colspan=6>🔵 **Continuous Axes ({len(axes_df)} components, ranked by variance)**</td> |\n")
    for _, r in axes_df.iterrows():
        label = str(r["LLM Interpretation"]).replace("Savory", "Savoury").replace("savoriness", "savouriness").replace("Brown Colored Foods", "Brown-coloured Foods")
        best_rating = str(r["Best Rating"]).replace("savoriness", "savouriness").replace("calorie_density", "calorie density")
        md_lines.append(
            f"| {r['Rank']} | {r['Top Loading']} | {r['Lowest']} | {label} | "
            f"{best_rating} ({r['Rho']}) | {r['Visual Weight']} |\n"
        )

    md_lines.append(f"| <td colspan=6>🟠 **Feature Detectors ({len(detectors_df)} components, ranked by variance)**</td> |\n")
    for _, r in detectors_df.iterrows():
        label = str(r["LLM Interpretation"]).replace("Savory", "Savoury").replace("savoriness", "savouriness").replace("Brown Colored Foods", "Brown-coloured Foods")
        best_rating = str(r["Best Rating"]).replace("savoriness", "savouriness").replace("calorie_density", "calorie density")
        md_lines.append(
            f"| {r['Rank']} | {r['Top Loading']} | {r['Lowest']} | {label} | "
            f"{best_rating} ({r['Rho']}) | {r['Visual Weight']} |\n"
        )
    out_md = OUT_DIR / "spose_interpretation_table.md"
    out_md.write_text("".join(md_lines))
    print(f"[OK] Wrote: {out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
