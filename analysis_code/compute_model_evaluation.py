"""compute_model_evaluation.py

Stage 2 (Compute): Evaluate model representations against empirical similarity data.

This script runs the core univariate RSA scans (Spearman rank correlations) between 
candidate model RDMs and the behavioral (empirical or NN-completed) RDM.

It also exports `pairwise_vectors.csv` which contains the unwrapped upper-triangle 
vectors for the empirical data and all evaluated models. This file is required by 
`compute_variance_attribution.py` for downstream Unique Variance and Shapley analyses.

When run with `--run-paper-suite`, it orchestrates the entire pipeline:
1. Univariate evaluation for mixed models.
2. Univariate evaluation for human-only models -> triggers `compute_variance_attribution.py`.
3. Univariate evaluation for ai-only models -> triggers `compute_variance_attribution.py`.

Outputs strictly to data folders:
  analysis_code/compute_outputs/stage2_model_evaluation/<rdm_mode>/<source>/
"""

import os, sys, json
from pathlib import Path
from datetime import datetime
from scipy.stats import rankdata

# Ensure local imports work when running the script from analysis_code/
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# --- Local analysis utilities ---
from analysis_helpers import (
    load_triplet_rdm, load_info, build_model_rdm,
    mantel_spearman, bootstrap_ci_spearman,
    plot_heatmaps,
)

# Central model-selection policy (explicit, reviewable)
try:
    from analysis_helpers import select_models
except Exception:
    select_models = None  # type: ignore

# Policy defaults (canonical, reviewable)
try:
    from reference.policy_defaults import (
        EXCLUDE_VISUAL_MODELS,
        EXCLUDE_MODELS,
        INFO_CSV as POLICY_INFO_CSV,
        DERIVATIVES_DIR as POLICY_DERIVATIVES_DIR,
        RESULTS_DIR as POLICY_RESULTS_DIR,
        OUTPUTS_DIR as POLICY_OUTPUTS_DIR,
        COMPUTE_DIR as POLICY_COMPUTE_DIR,
        SUMMARY_CSV as POLICY_SUMMARY_CSV,
        EMPIRICAL_RDM_NPZ as POLICY_EMPIRICAL_RDM_NPZ,
        get_compute_dir as policy_get_compute_dir,
        DEFAULT_SOURCE as POLICY_DEFAULT_SOURCE,
        DEFAULT_INCLUDE_VISUAL as POLICY_DEFAULT_INCLUDE_VISUAL,
        DEFAULT_INCLUDE_CONTROLS as POLICY_DEFAULT_INCLUDE_CONTROLS,
        DEFAULT_INCLUDE_CONCEPTUAL_IN_UNIVARIATE as POLICY_DEFAULT_INCLUDE_CONCEPTUAL_IN_UNIVARIATE,
        DEFAULT_INCLUDE_CONCEPTUAL_IN_UNIQUE as POLICY_DEFAULT_INCLUDE_CONCEPTUAL_IN_UNIQUE,
        CONTROL_COLS as POLICY_CONTROL_COLS,
        ALWAYS_INCLUDE_MODELS as POLICY_ALWAYS_INCLUDE_MODELS,
        CONCEPTUAL_SUPPLEMENTARY as POLICY_CONCEPTUAL_SUPPLEMENTARY,
        TRANSFORMATION_MODELS as POLICY_TRANSFORMATION_MODELS,
    )
except Exception:
    POLICY_ALWAYS_INCLUDE_MODELS = frozenset({
        "food_classification", "Category_WHO_10", "Category_Culinary_9",
        "NOVA_ordinal", "natural_vs_transformed", "Transformation_score",
    })
    POLICY_CONCEPTUAL_SUPPLEMENTARY = tuple(sorted(POLICY_ALWAYS_INCLUDE_MODELS))
    POLICY_TRANSFORMATION_MODELS = ("NOVA_ordinal", "natural_vs_transformed", "Transformation_score")
    POLICY_INFO_CSV = None
    POLICY_DERIVATIVES_DIR = None
    POLICY_RESULTS_DIR = None
    POLICY_OUTPUTS_DIR = None
    POLICY_COMPUTE_DIR = None
    POLICY_SUMMARY_CSV = None
    POLICY_EMPIRICAL_RDM_NPZ = None
    policy_get_compute_dir = None
    POLICY_DEFAULT_SOURCE = "human"
    POLICY_DEFAULT_INCLUDE_VISUAL = True
    POLICY_DEFAULT_INCLUDE_CONTROLS = False
    POLICY_DEFAULT_INCLUDE_CONCEPTUAL_IN_UNIVARIATE = True
    POLICY_DEFAULT_INCLUDE_CONCEPTUAL_IN_UNIQUE = False
    POLICY_CONTROL_COLS = frozenset({"portion_size_ok", "quality", "plate_rim_visible",
                                     "file_exists", "label_match", "label_confidence",
                                     "qc_issues", "qc_reasons", "qc_at", "qc_model"})
    EXCLUDE_VISUAL_MODELS = frozenset()
    EXCLUDE_MODELS = frozenset({"sweet_vs_savory", "Sweet_vs_savory",
                                "sweet vs savory", "Sweet vs savory"})


def _pair_counts_from_summary(summary_csv: str, labels: list[str]) -> "np.ndarray":
    """
    Return an (n x n) symmetric matrix of observation counts for each pair,
    based on triplet odd-one-out trials in `summary_csv`.
    """
    import pandas as _pd
    import numpy as _np
    labels = [os.path.basename(str(x)) for x in labels]
    idx_of = {fn: i for i, fn in enumerate(labels)}
    n = len(labels)
    cnt = _np.zeros((n, n), dtype=float)

    df = _pd.read_csv(summary_csv)
    needed = {"StimA_file","StimB_file","StimC_file","choice_file"}
    if not needed.issubset(df.columns):
        raise ValueError(f"summary_csv missing required columns: {needed}")

    def _base(x): return os.path.basename(str(x))

    for _, r in df.iterrows():
        a, b, c = _base(r["StimA_file"]), _base(r["StimB_file"]), _base(r["StimC_file"])
        if a not in idx_of or b not in idx_of or c not in idx_of:
            continue
        ia, ib, ic = idx_of[a], idx_of[b], idx_of[c]
        # A pair is "observed" if it co-occurred in the same triplet trial — regardless of
        # whether it was chosen as the similar pair.  Counting only the winning pair was a
        # bug: pairs that always lost (high dissimilarity) got count=0 and were incorrectly
        # masked as unobserved.  Count all three co-occurring pairs instead.
        for pi, pj in ((ia, ib), (ia, ic), (ib, ic)):
            i, j = min(pi, pj), max(pi, pj)
            cnt[i, j] += 1.0

    cnt = cnt + cnt.T
    _np.fill_diagonal(cnt, 0.0)
    return cnt

# === Noise ceiling is precomputed by compute_data_qc.py (split-half + SB, Method 1) ===
# The ceiling value is loaded from data_quality_summary.json at startup (see below).
# The two functions that used to live here (_build_pairwise_from_df,
# compute_noise_ceiling_split_half, compute_noise_ceiling_loso) have been moved to
# compute_data_qc.py, which is the single source of truth for all ceiling computation.


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(name))


# ---------------------------------------------------------------------------
# Variance Attribution helpers (Shapley ΔR²)
# Ported from results5_variance_attribution_plots.py so compute runs are
# self-contained and results scripts only do plotting.
# ---------------------------------------------------------------------------

def _va_zscore(x):
    import numpy as _np
    x = _np.asarray(x, dtype=float)
    mu = _np.nanmean(x)
    sd = _np.nanstd(x)
    if not _np.isfinite(sd) or sd == 0:
        return x * 0.0
    return (x - mu) / sd


def _va_r2_ols(y, X, rank_transform=True):
    """R² from OLS on (optionally rank-transformed) z-scored y and X columns."""
    import numpy as _np
    import pandas as _pd
    y = _np.asarray(y, dtype=float)
    if X.size == 0:
        return 0.0
    if rank_transform:
        y = _pd.Series(y).rank(method="average").to_numpy(dtype=float)
        X = _np.column_stack([
            _pd.Series(X[:, j]).rank(method="average").to_numpy(dtype=float)
            for j in range(X.shape[1])
        ])
    y_z = _va_zscore(y)
    X_z = _np.column_stack([_va_zscore(X[:, j]) for j in range(X.shape[1])])
    if _np.allclose(X_z, 0.0):
        return 0.0
    beta, *_ = _np.linalg.lstsq(X_z, y_z, rcond=None)
    y_hat = X_z @ beta
    ss_res = float(_np.sum((y_z - y_hat) ** 2))
    ss_tot = float(_np.sum((y_z - _np.mean(y_z)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return float(max(min(1.0 - ss_res / ss_tot, 1.0), 0.0))


def _va_shapley(y, X_dict, controls=None, n_orderings=300, rank_transform=True, seed=123):
    """Approximate Shapley ΔR² via random predictor orderings.

    Parameters
    ----------
    y : 1-D array  — empirical dissimilarity vector
    X_dict : dict[str -> 1-D array]  — predictor vectors (keyed by model name)
    controls : list[str]  — names always entered first (baseline; e.g. visual models)
    n_orderings : int  — number of random orderings to average over
    rank_transform : bool  — rank-transform before OLS (Spearman-style, recommended)
    seed : int

    Returns
    -------
    dict  — {predictor_name: mean_delta_R2, '__baseline_R2__': float,
             '__total_R2_with_all__': float}
    """
    import numpy as _np
    rng = _np.random.default_rng(seed)
    controls = list(controls or [])
    pred_names = [k for k in X_dict.keys() if k not in controls]

    X_ctrl = _np.column_stack([X_dict[c] for c in controls]) if controls else _np.empty((len(y), 0))
    base_r2 = _va_r2_ols(y, X_ctrl, rank_transform=rank_transform) if controls else 0.0

    contrib = {k: 0.0 for k in pred_names}
    if not pred_names:
        return {"__baseline_R2__": float(base_r2)}

    for _ in range(n_orderings):
        order = list(pred_names)
        rng.shuffle(order)
        cur_X = X_ctrl
        cur_r2 = base_r2
        for p in order:
            if cur_X.size == 0:
                cur_X = X_dict[p][:, None]
            else:
                cur_X = _np.column_stack([cur_X, X_dict[p]])
            new_r2 = _va_r2_ols(y, cur_X, rank_transform=rank_transform)
            contrib[p] += new_r2 - cur_r2
            cur_r2 = new_r2

    for k in contrib:
        contrib[k] /= float(n_orderings)

    all_X = _np.column_stack([X_dict[c] for c in controls + pred_names])
    contrib["__baseline_R2__"] = float(base_r2)
    contrib["__total_R2_with_all__"] = float(_va_r2_ols(y, all_X, rank_transform=rank_transform))
    return contrib


def _va_bootstrap_shapley(y, X_dict, controls, n_orderings, n_boot, rank_transform=True, seed=123):
    """Bootstrap CIs for Shapley values (row-resampling)."""
    import numpy as _np
    rng = _np.random.default_rng(seed)
    names = [k for k in X_dict.keys() if k not in controls]
    if not names:
        import pandas as _pd
        return _pd.DataFrame(columns=["predictor", "mean", "ci95_lo", "ci95_hi"])

    vals = {k: _np.empty(n_boot, dtype=float) for k in names}
    base_arr = _np.empty(n_boot, dtype=float)
    tot_arr = _np.empty(n_boot, dtype=float)
    n = len(y)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        Xb = {k: v[idx] for k, v in X_dict.items()}
        out = _va_shapley(yb, Xb, controls=controls, n_orderings=max(30, n_orderings // 5),
                          rank_transform=rank_transform, seed=int(seed + 10 + b))
        base_arr[b] = out.get("__baseline_R2__", _np.nan)
        tot_arr[b] = out.get("__total_R2_with_all__", _np.nan)
        for k in names:
            vals[k][b] = out.get(k, _np.nan)

    import pandas as _pd
    rows = []
    for k in names:
        rows.append({
            "predictor": k,
            "mean": float(_np.nanmean(vals[k])),
            "ci95_lo": float(_np.nanpercentile(vals[k], 2.5)),
            "ci95_hi": float(_np.nanpercentile(vals[k], 97.5)),
        })
    rows.append({"predictor": "__baseline_R2__",
                 "mean": float(_np.nanmean(base_arr)),
                 "ci95_lo": float(_np.nanpercentile(base_arr, 2.5)),
                 "ci95_hi": float(_np.nanpercentile(base_arr, 97.5))})
    rows.append({"predictor": "__total_R2_with_all__",
                 "mean": float(_np.nanmean(tot_arr)),
                 "ci95_lo": float(_np.nanpercentile(tot_arr, 2.5)),
                 "ci95_hi": float(_np.nanpercentile(tot_arr, 97.5))})
    return _pd.DataFrame(rows)


# Project directories

DERIVATIVES_DIR = str(POLICY_DERIVATIVES_DIR) if POLICY_DERIVATIVES_DIR is not None else os.path.join(HERE, "derivatives")
RESULTS_DIR     = str(POLICY_RESULTS_DIR) if POLICY_RESULTS_DIR is not None else os.path.join(HERE, "results")
OUTPUTS_DIR     = str(POLICY_OUTPUTS_DIR) if POLICY_OUTPUTS_DIR is not None else os.path.join(HERE, "outputs")
for _p in (DERIVATIVES_DIR, RESULTS_DIR, OUTPUTS_DIR):
    os.makedirs(_p, exist_ok=True)

# Stage folder (stable, non-timestamped)
# Will be set after CLI args are parsed / empirical RDM is resolved
RUN_ID = "stage3_model_evaluation"

# Info sheet / core inputs
INFO_CSV = str(POLICY_INFO_CSV) if POLICY_INFO_CSV is not None else os.path.join(HERE, "reference", "Foodpictures_information_dynamic.csv")
TRIPLET_RDM_NPZ = str(POLICY_EMPIRICAL_RDM_NPZ) if POLICY_EMPIRICAL_RDM_NPZ is not None else os.path.join(DERIVATIVES_DIR, "triplet_rdm_by_filename.npz")
SUMMARY_CSV = str(POLICY_SUMMARY_CSV) if POLICY_SUMMARY_CSV is not None else os.path.join(DERIVATIVES_DIR, "summary.csv")

print("DERIVATIVES_DIR:", DERIVATIVES_DIR)
print("INFO_CSV:", INFO_CSV)
print("TRIPLET_RDM_NPZ:", TRIPLET_RDM_NPZ)
if not os.path.exists(INFO_CSV):
    print(f"[ERROR] INFO_CSV not found at: {INFO_CSV}")
    print("[HINT] Expected it under ../expt_stimuli/ or ../stimuli/. Update INFO_CSV candidates if you moved it elsewhere.")

# Optional: skip recomputation if core inputs have not changed since last run
SKIP_IF_UP_TO_DATE = False

# === Global verbosity and summary storage ===
VERBOSE = True  # toggle to True for detailed path prints

import argparse
parser = argparse.ArgumentParser(description="Triplet Model Testing")
parser.add_argument("--run-all", action="store_true", help="Run all analyses")
parser.add_argument(
    "--run-source-suite",
    action="store_true",
    help="Run a default source suite consisting of human, ai, and mixed evaluation runs.",
)
# --- Paper suite runner ---
parser.add_argument(
    "--run-paper-suite",
    action="store_true",
    help="Run the main paper suite: mixed univariate only, plus human-only and ai-only univariate+unique runs.",
)
parser.add_argument(
    "--skip-unique",
    action="store_true",
    help="Run the univariate model scan only and skip the unique-variance step.",
)
parser.add_argument("--run-univariate", action="store_true", help="Run univariate model scans (Mantel)")
parser.add_argument("--run-regression", action="store_true", help="Run k-model regression and VP")
parser.add_argument("--run-nn-mapping", action="store_true", help="Run NN component mapping")

# --- Model-selection policy (primary analyses) ---
parser.add_argument(
    "--source",
    choices=["human", "ai", "mixed"],
    default=str(POLICY_DEFAULT_SOURCE),
    help=(
        "Model policy: which rating source to include for appraisal+flavour. "
        "Default: human. Use ai for AI-only. Use mixed only for supplementary analyses."
    ),
)
parser.add_argument(
    "--no-visual",
    action="store_true",
    help="Model policy: exclude low-level visual models (ll_*).",
)
parser.add_argument(
    "--include-nn-models",
    action="store_true",
    dest="include_nn_models",
    help=(
        "Include NN seed component RDMs (nn_seed*) in the univariate RSA scan. "
        "Off by default: NN components are excluded from the paper's quantitative "
        "analyses and used only for the MDS / interpretation figures."
    ),
)
parser.add_argument(
    "--controls",
    action="store_true",
    help="Model policy: include control predictors if present (e.g., portion_size_ok, quality).",
)

# --- Policy for conceptual contrasts (supplementary models) ---
parser.add_argument(
    "--include-conceptual-contrasts",
    action="store_true",
    help=(
        "Include supplementary conceptual contrast models (e.g., 'natural_vs_transformed') in the model set. "
        "Default behavior follows policy (included in univariate only when enabled; excluded from unique analyses unless explicitly allowed)."
    ),
)
# Backwards-compatible alias
parser.add_argument(
    "--include-contrasts",
    dest="include_conceptual_contrasts",
    action="store_true",
    help="Alias for --include-conceptual-contrasts.",
)
# --- Policy: conceptual contrasts can be included in univariate scan, but are excluded from unique-variance and variance partitioning unless explicitly allowed
parser.add_argument(
    "--allow-conceptual-in-unique",
    action="store_true",
    help=(
        "Allow conceptual contrast models to participate in unique-variance analyses (partial Mantel / variance attribution) "
        "and variance partitioning. Default behavior follows policy and excludes them from unique-variance unless explicitly allowed."
    ),
)

# --- Add support for overriding empirical RDM ---
parser.add_argument(
    "--empirical-rdm",
    type=str,
    default=None,
    help=(
        "Optional path to an empirical triplet RDM .npz to use instead of the default "
        "derivatives/triplet_rdm_by_filename.npz. Useful for NN-completed matrices."
    ),
)

parser.add_argument(
    "--fixed-dense-only",
    action="store_true",
    default=False,
    help=(
        "Build the empirical RDM exclusively from fixed+dense trials "
        "(summary.csv rows where is_fixed=True or is_dense=True). "
        "Matches the 209-item subset used in Stage 5 (compute_family_generalization.py), "
        "enabling a direct apples-to-apples comparison of univariate and multivariate results. "
        "Overrides the pre-built NPZ. The default noise-ceiling JSON path is also updated "
        "to point to the fixed-dense-specific file (stage0_data_qc_fixed_dense/)."
    ),
)

parser.add_argument(
    "--use-completed-nn-seed",
    type=int,
    default=None,
    help=(
        "Convenience option: use derivatives/triplet_rdm_completed_nn_seed{SEED}.npz as the empirical RDM. "
        "Overrides --empirical-rdm if both are provided."
    ),
)

# --- Option to filter which NN seed's precomputed model RDMs are included ---
# Auto-detect default from derivatives/nn_runs/selected_seed.json (written by
# compute_nn_embedding_decomp.py --select). Falls back to None (all seeds).
_NN_RUNS_ROOT = (
    os.path.join(str(POLICY_DERIVATIVES_DIR), "nn_runs")
    if POLICY_DERIVATIVES_DIR is not None
    else os.path.join(HERE, "derivatives", "nn_runs")
)
_SELECTED_SEED_JSON = os.path.join(_NN_RUNS_ROOT, "selected_seed.json")
_DEFAULT_NN_SEED: int | None = None
if os.path.exists(_SELECTED_SEED_JSON):
    try:
        with open(_SELECTED_SEED_JSON, encoding="utf-8") as _ssf:
            _DEFAULT_NN_SEED = int(json.load(_ssf)["selected_seed"])
    except Exception:
        pass  # silently ignore malformed JSON — user can still pass --nn-seed explicitly

parser.add_argument(
    "--nn-seed",
    type=int,
    default=_DEFAULT_NN_SEED,
    help=(
        "NN seed to include from precomputed model RDMs (e.g., nn_seed21_*). "
        f"Default: auto-detected from {_SELECTED_SEED_JSON} "
        f"(currently: {_DEFAULT_NN_SEED}). "
        "Pass explicitly to override. If None, all available NN seed models are included."
    ),
)

# --- Runtime controls (speed/stability) ---
parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed for permutations/bootstraps")
parser.add_argument(
    "--fast",
    action="store_true",
    help=(
        "Fast mode: reduce permutations/bootstraps/ceiling iterations for quicker, more stable exploratory runs. "
        "(Overrides n_perm, n_boot, n_perm_unique, n_boot_unique to smaller values.)"
    ),
)

parser.add_argument(
    "--outdir",
    type=str,
    default=None,
    help=(
        "Optional output directory for this evaluation run. If not provided, outputs go to "
        "analysis_code/results/stage3_model_evaluation/<run_tag>/ and analysis_code/outputs/stage3_model_evaluation/<run_tag>/"
    ),
)
parser.add_argument(
    "--output-tag",
    type=str,
    default=None,
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--run-type",
    choices=["univariate_all", "human_unique", "ai_unique"],
    default="univariate_all",
    help=(
        "Preset run type. Configures source, analysis flags, and output folder automatically so the "
        "three standard runs never overwrite each other. Default: univariate_all.\n"
        "  univariate_all  → source=mixed,  univariate scan only (no unique),  "
        "folder: …/empirical/mixed/\n"
        "  human_unique    → source=human,  univariate + unique variance + Shapley, "
        "folder: …/empirical/human/\n"
        "  ai_unique       → source=ai,     univariate + unique variance + Shapley, "
        "folder: …/empirical/ai/\n"
        "Any explicitly-passed flag (--source, --output-tag, --skip-unique, --outdir) "
        "still takes precedence over the preset."
    ),
)

# Univariate Mantel / scan
parser.add_argument("--n-perm", type=int, default=3000, help="Permutations for univariate Mantel tests")
parser.add_argument("--n-boot", type=int, default=1500, help="Bootstraps for CI on univariate Mantel tests")

# Unique variance (partial Mantel analogue)
parser.add_argument("--n-perm-unique", type=int, default=3000, help="Permutations for partial/unique tests")
parser.add_argument("--n-boot-unique", type=int, default=2000, help="Bootstraps for CI in partial/unique tests")

# Pairwise vector export controls
parser.add_argument(
    "--pairwise-include-precomputed",
    action="store_true",
    help=(
        "When writing pairwise_vectors.csv (regression step), also add columns for each precomputed model RDM "
        "loaded from derivatives/model_rdms (and the two base categorical models). Column names are prefixed with 'rdm_'."
    ),
)
parser.add_argument(
    "--pairwise-precomputed-filter",
    type=str,
    default="",
    help=(
        "Optional comma-separated prefixes to filter which precomputed models are included in pairwise_vectors.csv. "
        "Example: 'll_,ai_,human_' to include low-level + AI + human models. Empty string includes all precomputed."
    ),
)
_DEFAULT_NC_JSON = str(
    (Path(POLICY_COMPUTE_DIR) / "stage0_data_qc" / "data_quality_summary.json")
    if POLICY_COMPUTE_DIR is not None
    else Path(__file__).resolve().parent / "compute_outputs" / "stage0_data_qc" / "data_quality_summary.json"
)
parser.add_argument(
    "--noise-ceiling-json",
    default=_DEFAULT_NC_JSON,
    metavar="PATH",
    help=(
        "Path to data_quality_summary.json written by compute_data_qc.py. "
        "When this file exists the precomputed noise ceiling (mean_sb / ci95_lo_sb / ci95_hi_sb) "
        "is loaded directly, skipping recomputation. "
        f"Default: {_DEFAULT_NC_JSON}"
    ),
)


args, unknown = parser.parse_known_args()

# --- Apply --run-type preset (before all other defaults) ---
# Only override fields the user did NOT explicitly provide on the CLI.
_explicit = set(sys.argv[1:])  # raw tokens for presence checks

if getattr(args, "run_type", None) is not None:
    _rt = args.run_type

    # Presets for each run type
    _PRESETS = {
        "univariate_all": {
            "source": "mixed",
            "skip_unique": True,
            "run_univariate": True,
            "output_tag": "univariate_all",
        },
        "human_unique": {
            "source": "human",
            "skip_unique": False,
            "run_univariate": True,
            "output_tag": "human_unique",
        },
        "ai_unique": {
            "source": "ai",
            "skip_unique": False,
            "run_univariate": True,
            "output_tag": "ai_unique",
        },
    }
    _preset = _PRESETS[_rt]

    # source: apply unless --source was explicitly passed
    if "--source" not in _explicit:
        args.source = _preset["source"]

    # skip_unique: apply unless --skip-unique was explicitly passed
    if "--skip-unique" not in _explicit:
        args.skip_unique = _preset["skip_unique"]

    # run_univariate: always enable (safe to force on)
    args.run_univariate = True

    # output_tag: apply unless --output-tag was explicitly passed
    if "--output-tag" not in _explicit and not getattr(args, "output_tag", None):
        args.output_tag = _preset["output_tag"]

    # include_conceptual_contrasts: apply preset only if user didn't pass the flag explicitly
    if "--include-conceptual-contrasts" not in _explicit and "--include_conceptual_contrasts" not in _explicit:
        if "include_conceptual_contrasts" in _preset:
            args.include_conceptual_contrasts = _preset["include_conceptual_contrasts"]

    print(f"[INFO] --run-type={_rt}: source={args.source}, skip_unique={args.skip_unique}, "
          f"output_tag={args.output_tag}, conceptual_contrasts={getattr(args, 'include_conceptual_contrasts', '?')}")

# Apply policy-aligned defaults when flags are absent.
if POLICY_DEFAULT_INCLUDE_VISUAL is False:
    args.no_visual = True if not getattr(args, "no_visual", False) else args.no_visual
if POLICY_DEFAULT_INCLUDE_CONTROLS is True:
    args.controls = True if not getattr(args, "controls", False) else args.controls
# Apply the policy default only when neither the CLI nor a --run-type preset has already
# set include_conceptual_contrasts.  The preset uses a sentinel so we can distinguish
# "not set yet" (None) from "explicitly disabled" (False).
_contrasts_locked_by_preset = (
    getattr(args, "run_type", None) is not None
    and "include_conceptual_contrasts" in _PRESETS.get(getattr(args, "run_type", ""), {})
)
if POLICY_DEFAULT_INCLUDE_CONCEPTUAL_IN_UNIVARIATE is True and not _contrasts_locked_by_preset:
    if not bool(getattr(args, "include_conceptual_contrasts", False)):
        args.include_conceptual_contrasts = True

# Policy manifest (logged in stdout and later written into manifest.json)
POLICY = {
    "source": getattr(args, "source", "human"),
    "include_visual": (not getattr(args, "no_visual", False)),
    "include_controls": bool(getattr(args, "controls", False)),
    "include_conceptual_contrasts": bool(getattr(args, "include_conceptual_contrasts", False)),
    "allow_conceptual_in_unique": bool(getattr(args, "allow_conceptual_in_unique", False)),
}
if VERBOSE:
    print(
        "[INFO] Model policy: "
        f"source={POLICY['source']} | visual={'on' if POLICY['include_visual'] else 'off'} | "
        f"controls={'on' if POLICY['include_controls'] else 'off'} | "
        f"contrasts={'on' if POLICY['include_conceptual_contrasts'] else 'off'} | "
        f"contrasts_in_unique={'on' if POLICY['allow_conceptual_in_unique'] else 'off'}"
    )
    if getattr(args, "nn_seed", None) is not None:
        print(f"[INFO] NN seed filter: using precomputed NN models from seed {int(args.nn_seed)} only")

# Apply fast mode overrides
if getattr(args, "fast", False):
    args.n_perm = min(int(args.n_perm), 800)
    args.n_boot = min(int(args.n_boot), 400)
    args.n_perm_unique = min(int(args.n_perm_unique), 800)
    args.n_boot_unique = min(int(args.n_boot_unique), 600)

if VERBOSE:
    print(
        f"[INFO] Runtime: seed={args.seed} | n_perm={args.n_perm} n_boot={args.n_boot} | "
        f"n_perm_unique={args.n_perm_unique} n_boot_unique={args.n_boot_unique} | "
        f"fast={args.fast}"
    )

if not any([args.run_all, args.run_univariate, args.run_regression, args.run_nn_mapping]):
    # --run-type preset already set run_univariate=True for all three types,
    # so this fallback only fires if the user explicitly cleared all step flags.
    print("No specific steps requested. Defaulting to --run-univariate.")
    args.run_univariate = True

# --- Source suite runner convenience ---
if getattr(args, "run_source_suite", False):
    import subprocess

    # Recreate the current CLI minus suite/output/source controls, then launch one run per source.
    passthrough = []
    skip_next = False
    for i, tok in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if tok == "--run-source-suite":
            continue
        if tok in {"--source", "--outdir", "--output-tag", "--run-type"}:
            skip_next = True
            continue
        passthrough.append(tok)

    suite_sources = ["human", "ai", "mixed"]
    failed = []
    print(f"[INFO] Running default source suite: {suite_sources}")
    for src in suite_sources:
        cmd = [
            sys.executable,
            __file__,
            *passthrough,
            "--source", src,
            "--output-tag", f"source_{src}",
        ]
        print("[INFO] Launching:", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            failed.append((src, rc))

    if failed:
        print(f"[ERROR] Source suite had failures: {failed}")
        sys.exit(1)
    print("[INFO] Source suite complete.")
    sys.exit(0)

# --- Paper suite runner convenience ---
if getattr(args, "run_paper_suite", False):
    import subprocess

    passthrough = []
    skip_next = False
    for tok in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if tok == "--run-paper-suite":
            continue
        if tok == "--run-source-suite":
            continue
        if tok in {"--source", "--outdir", "--output-tag", "--run-type"}:
            skip_next = True
            continue
        if tok == "--no-visual":
            # Paper suite always keeps visual models in
            continue
        if tok == "--skip-unique":
            continue
        passthrough.append(tok)

    suite_specs = [
        {
            "source": "mixed",
            "tag": "paper_mixed_univariate",
            "extra": ["--run-type", "univariate_all"],
        },
        {
            "source": "human",
            "tag": "paper_human_unique",
            "extra": ["--run-type", "human_unique"],
        },
        {
            "source": "ai",
            "tag": "paper_ai_unique",
            "extra": ["--run-type", "ai_unique"],
        },
    ]

    failed = []
    print("[INFO] Running paper suite:")
    
    if (args.use_completed_nn_seed is not None or args.empirical_rdm):
        suite_rdm_mode = "filled"
    elif getattr(args, "fixed_dense_only", False):
        suite_rdm_mode = "fixed_dense"
    else:
        suite_rdm_mode = "empirical"

    for spec in suite_specs:
        cmd = [
            sys.executable,
            __file__,
            *passthrough,
            *spec["extra"],
            # --source and --output-tag are set by --run-type; only add them if not
            # already covered by the preset (i.e., for any future non-run-type specs).
            *(["--source", spec["source"]] if "--run-type" not in spec["extra"] else []),
            *(["--output-tag", spec["tag"]] if "--run-type" not in spec["extra"] else []),
        ]
        print("[INFO] Launching (Stage 2):", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            failed.append((spec["source"], rc))
        
        # New: Trigger Stage 4 (Variance Attribution) for Human/AI runs
        if spec["source"] in ["human", "ai"]:
            print(f"[INFO] Launching (Stage 4) for {spec['source']}...")
            va_outdir = str(policy_get_compute_dir("stage4_variance_attribution", suite_rdm_mode, spec["source"]))
            va_cmd = [
                sys.executable,
                os.path.join(HERE, "compute_variance_attribution.py"),
                "--pairwise-csv", str(policy_get_compute_dir("stage2_model_evaluation", suite_rdm_mode, spec["source"]) / "pairwise_vectors.csv"),
                "--source", spec["source"],
                "--method", "suite",
                "--bootstrap", "200",
                "--rank-transform",  # use Spearman ρ² to match Stages 3/5 metric
                "--outdir", va_outdir,
            ]
            print("[INFO] Launching:", " ".join(va_cmd))
            rc_va = subprocess.call(va_cmd)
            if rc_va != 0:
                failed.append((f"{spec['source']}_variance", rc_va))

    if failed:
        print(f"[ERROR] Paper suite had failures: {failed}")
        sys.exit(1)

    print("[INFO] Paper suite complete.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Load noise ceiling from compute_data_qc.py output (always-on, graceful NaN fallback)
# ---------------------------------------------------------------------------
_nc_json_path = Path(getattr(args, "noise_ceiling_json", _DEFAULT_NC_JSON))
# When --fixed-dense-only is active and the user hasn't explicitly provided a
# noise-ceiling path, redirect to the fixed-dense-specific QC output so the
# reported ceiling matches the 209-item subset (run compute_data_qc.py
# --fixed-dense-only first to generate this file).
if getattr(args, "fixed_dense_only", False) and "--noise-ceiling-json" not in _explicit:
    _fd_nc_dir = Path(_DEFAULT_NC_JSON).parent.parent / "stage0_data_qc_fixed_dense"
    _nc_json_path = _fd_nc_dir / "data_quality_summary.json"
    if VERBOSE:
        print(f"[INFO] --fixed-dense-only: noise ceiling JSON -> {_nc_json_path}")
_NC_CEILING = float("nan")
_NC_CEILING_CI = (float("nan"), float("nan"))

if _nc_json_path.exists():
    try:
        with open(_nc_json_path, encoding="utf-8") as _ncf:
            _nc_data = json.load(_ncf)
        _nc_blob = _nc_data.get("noise_ceiling", {})
        import math
        _NC_CEILING    = math.sqrt(float(_nc_blob["mean_sb"]))
        _NC_CEILING_CI = (math.sqrt(float(_nc_blob["ci95_lo_sb"])), math.sqrt(float(_nc_blob["ci95_hi_sb"])))
        if VERBOSE:
            print(
                f"[INFO] Noise ceiling loaded from {_nc_json_path}: "
                f"sqrt(mean_sb)={_NC_CEILING:.4f}  "
                f"CI95=({_NC_CEILING_CI[0]:.4f},{_NC_CEILING_CI[1]:.4f})"
            )
    except Exception as _nc_err:
        print(f"[WARN] Could not read noise ceiling JSON ({_nc_json_path}): {_nc_err}. "
              "r_over_ceiling will be NaN.")
else:
    print(f"[WARN] Noise ceiling JSON not found: {_nc_json_path}\n"
          "       Run compute_data_qc.py first. r_over_ceiling will be NaN.")

SCAN_MODELS = args.run_all or args.run_univariate
# Auto-enable regression when doing a unique-variance run so pairwise_vectors.csv
# (required for Shapley) is always written without needing --run-regression explicitly.
_unique_variance_run = not getattr(args, "skip_unique", False)
RUN_REGRESSION = args.run_all or args.run_regression or (
    _unique_variance_run and not getattr(args, "no_shapley", False)
)
RUN_NN_MAPPING = args.run_all or args.run_nn_mapping

# --- Empirical RDM selection (observed-only vs NN-completed) ---
EMPIRICAL_RDM_NPZ = TRIPLET_RDM_NPZ
if args.empirical_rdm:
    EMPIRICAL_RDM_NPZ = os.path.abspath(os.path.expanduser(args.empirical_rdm))
if args.use_completed_nn_seed is not None:
    EMPIRICAL_RDM_NPZ = os.path.join(DERIVATIVES_DIR, f"triplet_rdm_completed_nn_seed{int(args.use_completed_nn_seed)}.npz")

if not os.path.exists(EMPIRICAL_RDM_NPZ):
    raise FileNotFoundError(f"Empirical RDM NPZ not found: {EMPIRICAL_RDM_NPZ}")

if VERBOSE:
    if EMPIRICAL_RDM_NPZ == TRIPLET_RDM_NPZ:
        print(f"[INFO] Using observed-only empirical RDM: {EMPIRICAL_RDM_NPZ}")
    else:
        print(f"[INFO] Using OVERRIDDEN empirical RDM (e.g., NN-completed): {EMPIRICAL_RDM_NPZ}")

# If the empirical RDM is overridden (e.g., NN-completed), we should use *all* pairs.
USING_OVERRIDDEN_EMPIRICAL_RDM = (os.path.abspath(EMPIRICAL_RDM_NPZ) != os.path.abspath(TRIPLET_RDM_NPZ))

# --- Run-tagged output routing ---
# Empirical (observed-only), fixed_dense, and filled (NN-completed / custom empirical) runs should not overwrite each other.
if args.use_completed_nn_seed is not None or args.empirical_rdm:
    rdm_mode = "filled"
elif getattr(args, "fixed_dense_only", False):
    rdm_mode = "fixed_dense"
else:
    rdm_mode = "empirical"

source_tag = getattr(args, "source", "human")

if policy_get_compute_dir is not None:
    RUN_RESULTS_DIR = str(policy_get_compute_dir("stage2_model_evaluation", rdm_mode, source_tag))
else:
    RUN_RESULTS_DIR = os.path.join(HERE, "compute_outputs", "stage2_model_evaluation", rdm_mode, source_tag)

os.makedirs(RUN_RESULTS_DIR, exist_ok=True)

if VERBOSE:
    print(f"[INFO] Compute outdir: {RUN_RESULTS_DIR}")

# --- Model scan controls ---
# SCAN_MODELS is now controlled by argparse
FDR_Q       = 0.05       # BH-FDR level
REDUND_THR  = 0.60       # absolute model↔model Spearman threshold to consider "redundant cluster"

_summary = {
    "coverage": None,
    "fits_df": None,
    "ceiling": None,
    "reg_df": None,
    "reg_meta": {},
    "compmap_df": None,
    "selected_nn_seed": int(args.nn_seed) if getattr(args, "nn_seed", None) is not None else None,
}

N_COMP = 5

# Load empirical RDM using resolved path
D_emp, labels = load_triplet_rdm(EMPIRICAL_RDM_NPZ)

info = load_info(INFO_CSV)

len(labels), info.shape

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# --- Fixed+dense override: rebuild empirical RDM on the 209-item subset (Issue 3 fix) ---
# When --fixed-dense-only is set and no custom --empirical-rdm path was given,
# ignore the pre-built NPZ and reconstruct the RDM from summary.csv filtered to
# is_fixed | is_dense trials.  This produces the same item subset as Stage 5
# (compute_family_generalization.py), enabling a direct apples-to-apples comparison
# of univariate (Stage 3) and multivariate (Stage 5) performance.
if getattr(args, "fixed_dense_only", False) and not USING_OVERRIDDEN_EMPIRICAL_RDM:
    _fd_summary_path = Path(SUMMARY_CSV)
    if not _fd_summary_path.exists():
        raise FileNotFoundError(
            f"--fixed-dense-only requires summary.csv at {_fd_summary_path}"
        )
    print(f"[INFO] --fixed-dense-only: rebuilding empirical RDM from {_fd_summary_path} ...")
    _fd_df_raw = pd.read_csv(str(_fd_summary_path), low_memory=False)
    _fd_df_raw.columns = [str(c).strip() for c in _fd_df_raw.columns]
    _fd_has_fixed = "is_fixed" in _fd_df_raw.columns
    _fd_has_dense = "is_dense" in _fd_df_raw.columns
    if not (_fd_has_fixed or _fd_has_dense):
        raise ValueError(
            "--fixed-dense-only: summary.csv has neither is_fixed nor is_dense columns"
        )
    def _fd_parse_bool(s):
        def _c(v):
            if v is None: return False
            try:
                if pd.isna(v): return False
            except Exception: pass
            if isinstance(v, bool): return bool(v)
            if isinstance(v, (int, np.integer)): return int(v) != 0
            if isinstance(v, (float, np.floating)): return float(v) != 0.0
            return str(v).strip().lower() in {"1", "1.0", "true", "t", "yes", "y", "on"}
        return s.fillna(False).apply(_c).astype(bool)
    _fd_is_fixed = _fd_parse_bool(_fd_df_raw["is_fixed"]) if _fd_has_fixed \
                   else pd.Series(False, index=_fd_df_raw.index)
    _fd_is_dense = _fd_parse_bool(_fd_df_raw["is_dense"]) if _fd_has_dense \
                   else pd.Series(False, index=_fd_df_raw.index)
    _fd_df = _fd_df_raw[(_fd_is_fixed | _fd_is_dense)].copy()
    # Build labels exactly as Stage 5 does (from all stim + choice columns)
    _fd_labs_set: set = set()
    for _col in ("StimA_file", "StimB_file", "StimC_file", "choice_file"):
        if _col in _fd_df.columns:
            _fd_labs_set.update(
                os.path.basename(str(v))
                for v in _fd_df[_col].dropna().astype(str)
                if str(v).strip()
            )
    _fd_labs = sorted(_fd_labs_set)
    _fd_n = len(_fd_labs)
    _fd_idx: dict = {fn: i for i, fn in enumerate(_fd_labs)}
    _fd_sim = np.zeros((_fd_n, _fd_n), dtype=float)
    _fd_cnt = np.zeros((_fd_n, _fd_n), dtype=float)
    for _, _row in _fd_df.iterrows():
        if pd.isna(_row.get("choice_file")): continue
        _a  = os.path.basename(str(_row["StimA_file"]))
        _b  = os.path.basename(str(_row["StimB_file"]))
        _c_ = os.path.basename(str(_row["StimC_file"]))
        _ch = os.path.basename(str(_row["choice_file"]))
        if _a not in _fd_idx or _b not in _fd_idx or _c_ not in _fd_idx: continue
        _ia, _ib, _ic = _fd_idx[_a], _fd_idx[_b], _fd_idx[_c_]
        if   _ch == _a:  _pi, _pj = sorted((_ib, _ic))
        elif _ch == _b:  _pi, _pj = sorted((_ia, _ic))
        elif _ch == _c_: _pi, _pj = sorted((_ia, _ib))
        else: continue
        _fd_sim[_pi, _pj] += 1.0
        for _p1, _p2 in ((_ia, _ib), (_ia, _ic), (_ib, _ic)):
            if _p1 != _p2:
                _u, _v = sorted((_p1, _p2))
                _fd_cnt[_u, _v] += 1.0
    _fd_sim += _fd_sim.T; _fd_cnt += _fd_cnt.T
    with np.errstate(invalid="ignore", divide="ignore"):
        _fd_psim = np.where(_fd_cnt > 0, _fd_sim / _fd_cnt, np.nan)
    _fd_D = 1.0 - _fd_psim
    np.fill_diagonal(_fd_D, np.nan)
    _prev_n = len(labels) if labels else "?"
    D_emp  = _fd_D
    labels = _fd_labs
    print(f"[INFO] --fixed-dense-only: empirical RDM = {len(labels)} items "
          f"(replaced {_prev_n}-item pre-built NPZ)")

# === Utilities for additional model RDMs ===
def build_continuous_rdm(labels, info_df, column):
    """
    Build an n x n RDM from a continuous column using absolute differences.
    Returns an ndarray with NaN on the diagonal.
    """
    if column not in info_df.columns:
        raise KeyError(f"Column not in info: {column}")
    s = pd.to_numeric(info_df.set_index("filename").reindex(labels)[column], errors="coerce")
    v = s.to_numpy(dtype=float)
    D = np.abs(v[:, None] - v[None, :])
    np.fill_diagonal(D, np.nan)
    return D


# --- Robust label getter for ceiling/builders ---
def _ensure_labels(labels_candidate, triplet_npz_path, info_df):
    """
    Return a robust labels list (filenames, basename) for ceiling/builders.
    Priority:
      1) existing labels variable if non-empty,
      2) labels saved in TRIPLET_RDM_NPZ,
      3) info['filename'] (basename) if present,
      4) raise if none available.
    """
    if labels_candidate is not None and len(labels_candidate) > 0:
        return list(labels_candidate)

    # Try to read from the NPZ
    try:
        _npz = np.load(triplet_npz_path, allow_pickle=True)
        if "labels" in _npz:
            vals = _npz["labels"]
            vals = vals.tolist() if hasattr(vals, "tolist") else vals
            return list(vals)
    except Exception:
        pass

    # Fall back to info.csv
    if "filename" in info_df.columns:
        return list(info_df["filename"].astype(str).apply(os.path.basename))

    raise ValueError("Could not determine labels for ceiling; please check NPZ or info file.")

# --- Model scan helpers ---
def _bh_fdr(pvals, q=0.05):
    """Benjamini–Hochberg FDR.

    Returns:
      rejected: boolean array (same shape as pvals)
      qvals: BH-adjusted p-values (same shape as pvals)

    Notes:
      - `rejected` uses the standard BH step-up procedure.
      - `qvals` are monotone non-decreasing with rank.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return np.asarray([], dtype=bool), np.asarray([], dtype=float)

    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, n + 1)

    # Step-up rejection threshold
    thresh = (ranks / n) * float(q)
    passed = p_sorted <= thresh
    if np.any(passed):
        kmax = np.max(np.where(passed)[0])
        rejected_sorted = np.zeros(n, dtype=bool)
        rejected_sorted[: kmax + 1] = True
    else:
        rejected_sorted = np.zeros(n, dtype=bool)

    # BH adjusted p-values (q-values)
    q_sorted = p_sorted * n / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    # Unsort
    rejected = np.zeros(n, dtype=bool)
    qvals = np.zeros(n, dtype=float)
    rejected[order] = rejected_sorted
    qvals[order] = q_sorted
    return rejected, qvals

def _spearman_vec(x, y):
    """Spearman rho between finite entries of two equal-length vectors (rank -> Pearson)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return np.nan
    xr = rankdata(x[m], method="average")
    yr = rankdata(y[m], method="average")
    sx = xr.std(ddof=0)
    sy = yr.std(ddof=0)
    if sx == 0 or sy == 0:
        return np.nan
    xr = (xr - xr.mean()) / sx
    yr = (yr - yr.mean()) / sy
    return float(np.dot(xr, yr) / xr.size)

def _bootstrap_ci_vec_corr(x, y, B=2000, seed=1337, ci=0.95):
    rng = np.random.default_rng(seed)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = len(x)
    if n < 3:
        return (np.nan, np.nan)
    vals = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        vals[b] = _spearman_vec(x[idx], y[idx])
    lo = np.nanpercentile(vals, (1-ci)/2*100)
    hi = np.nanpercentile(vals, (1+(ci))/2*100)
    return (float(lo), float(hi))

def _partial_mantel_rank_resid(y_vec, x_vec, Z_mat=None, n_perm=2000, n_boot=2000, seed=4242):    
    """
    Partial Mantel analogue on vectors:
     - Rank-transform y and all predictors.
     - Regress y_rank on Z_rank to get residuals (if Z provided).
     - Regress x_rank on Z_rank to get residuals (if Z provided).
     - Spearman between residuals; permute x residuals for p.
    y_vec: target empirical vector (upper triangle)
    x_vec: predictor vector of interest
    Z_mat: 2D array [n_pairs x k] of other covariate vectors (or None)
    Returns dict with rho_partial, p_perm, ci95 (bootstrap on residual pairs).
    """
    rng = np.random.default_rng(seed)
    m = np.isfinite(y_vec) & np.isfinite(x_vec)
    if Z_mat is not None:
        mZ = np.isfinite(Z_mat).all(axis=1)
        m = m & mZ
    y = rankdata(np.asarray(y_vec[m], dtype=float), method="average").astype(float)
    x = rankdata(np.asarray(x_vec[m], dtype=float), method="average").astype(float)
    if Z_mat is not None and Z_mat.size > 0:
        Z = np.column_stack([
            rankdata(np.asarray(Z_mat[m, j], dtype=float), method="average").astype(float)
            for j in range(Z_mat.shape[1])
        ])
        # center+scale
        def _zs(v): 
            s = v.std(ddof=0)
            return (v - v.mean()) / (s if s != 0 else 1.0)
        y = _zs(y); x = _zs(x); Z = np.column_stack([_zs(Z[:, j]) for j in range(Z.shape[1])])
        # residualize
        betaZ_y, *_ = np.linalg.lstsq(Z, y, rcond=None)
        betaZ_x, *_ = np.linalg.lstsq(Z, x, rcond=None)
        y_res = y - Z @ betaZ_y
        x_res = x - Z @ betaZ_x
    else:
        y_res, x_res = y, x
    # observed rho
    obs_rho = _spearman_vec(y_res, x_res)
    # permutation on x_res
    n = len(x_res)
    null = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        idx = rng.permutation(n)
        null[b] = _spearman_vec(y_res, x_res[idx])
    p_perm = float((1.0 + np.sum(np.abs(null) >= abs(obs_rho))) / (1.0 + n_perm))
    # bootstrap CI on residual pairs
    ci_lo, ci_hi = _bootstrap_ci_vec_corr(y_res, x_res, B=int(n_boot), seed=seed+7, ci=0.95)
    return {"rho_partial": float(obs_rho), "p_perm": p_perm, "ci95_lo": ci_lo, "ci95_hi": ci_hi, "n_perm": int(n_perm)}

# --- Human-friendly display names for figures (expandable) ---
MODEL_DISPLAY_OVERRIDES = {
    "R": "Red (channel mean)",
    "G": "Green (channel mean)",
    "B": "Blue (channel mean)",
    # Add more overrides here when confirmed, e.g.:
    # "P": "P (feature, clarified)",
    # "E": "E (feature, clarified)",
    # "AN": "AN (feature, clarified)",
    # "S": "S (feature, clarified)",
    # "C": "C (feature, clarified)",
    # "H": "H (feature, clarified)",
    # "SS": "SS (feature, clarified)",
    # "FPC": "FPC (feature, clarified)",
}

def _display_name(name: str) -> str:
    return MODEL_DISPLAY_OVERRIDES.get(name, name)


iu, ju = np.triu_indices_from(D_emp, k=1)
v_emp = D_emp[iu, ju]
mask_finite_emp = np.isfinite(v_emp)

# Try to compute "true" coverage from the raw summary trials (not the padded matrix).
# If we are using an overridden empirical RDM (e.g., NN-completed), we deliberately use *all* pairs,
# so summary-based coverage/masking is not applied.
coverage_counts = None
cnt_vec = None
try:
    if (not USING_OVERRIDDEN_EMPIRICAL_RDM) and os.path.exists(SUMMARY_CSV):
        cnt_mat = _pair_counts_from_summary(SUMMARY_CSV, labels)
        cnt_vec = cnt_mat[iu, ju]
        mask_observed = cnt_vec > 0
        coverage_counts = (int(mask_observed.sum()), int(len(mask_observed)), float(mask_observed.mean()))
except Exception as e:
    print(f"[WARN] Could not compute coverage from summary: {e}")
    coverage_counts = None
    cnt_vec = None

# Fallback to finite entries in D_emp if summary-based coverage is unavailable
coverage_finite = (int(mask_finite_emp.sum()), int(len(mask_finite_emp)), float(mask_finite_emp.mean()))

# Report coverage
if USING_OVERRIDDEN_EMPIRICAL_RDM:
    # Completed matrix: treat all pairs as usable coverage
    n_all = int(len(mask_finite_emp))
    print(f"Observed pairs (completed empirical RDM): {n_all:,}/{n_all:,}  (~100.0%)")
    _summary["coverage_completed"] = (n_all, n_all, 1.0)
else:
    if coverage_counts is not None:
        n_obs, n_all, cov = coverage_counts
        print(f"Observed pairs (from summary counts): {n_obs:,}/{n_all:,}  (~{cov:.1%})")
        # Diagnostic on the RAW empirical matrix (before masking)
        if cnt_vec is not None:
            mid_pad_emp = np.isfinite(v_emp) & (np.abs(v_emp - 0.5) < 1e-12) & ~(cnt_vec > 0)
            n_mid_pad_emp = int(mid_pad_emp.sum())
            if n_mid_pad_emp > 0:
                print(f"[NOTE] Raw matrix contains {n_mid_pad_emp:,} entries exactly 0.5 with zero observations in summary (padding artifacts in D_emp).")
        # Keep both summaries
        _summary["coverage_counts"] = coverage_counts
        _summary["coverage_finite"] = coverage_finite
    else:
        n_obs, n_all, cov = coverage_finite
        print(f"Observed pairs (finite in D_emp): {n_obs:,}/{n_all:,}  (~{cov:.1%})")
        _summary["coverage"] = coverage_finite

# --- Helper to build a symmetric boolean matrix from an upper-tri vector ---
def _uppervec_to_sym_mask(n, iu, ju, vec_bool):
    m = np.zeros((n, n), dtype=bool)
    m[iu, ju] = vec_bool
    m = m | m.T
    np.fill_diagonal(m, False)
    return m

# Use a masked empirical matrix that excludes padded 0.5 entries (unobserved pairs)
# NOTE: If we are using an overridden empirical RDM (e.g., NN-completed), we intentionally keep *all* pairs.
D_TARGET = D_emp.copy()
iu_full, ju_full = np.triu_indices_from(D_emp, k=1)

if USING_OVERRIDDEN_EMPIRICAL_RDM:
    # Completed RDM: do not mask; use all upper-triangle pairs.
    np.fill_diagonal(D_TARGET, np.nan)
else:
    if coverage_counts is not None and cnt_vec is not None:
        # Build symmetric mask of observed pairs from summary counts
        obs_mask_sym = _uppervec_to_sym_mask(D_emp.shape[0], iu_full, ju_full, (cnt_vec > 0))
        # Set unobserved pairs to NaN (remove any 0.5 padding artifacts)
        D_TARGET[~obs_mask_sym] = np.nan
    else:
        # Fall back to using only finite entries of the matrix as observed
        fin_mask_sym = _uppervec_to_sym_mask(D_emp.shape[0], iu_full, ju_full, np.isfinite(D_emp[iu_full, ju_full]))
        D_TARGET[~fin_mask_sym] = np.nan

# --- Sanity check: confirm masking is applied to the target matrix ---
used_pairs = int(np.isfinite(D_TARGET[iu_full, ju_full]).sum())
total_pairs = int(len(iu_full))
if USING_OVERRIDDEN_EMPIRICAL_RDM:
    print(f"[INFO] Using completed empirical RDM: analyses will use {used_pairs:,}/{total_pairs:,} pairs.")
else:
    print(f"[INFO] Mask applied: analyses will use {used_pairs:,}/{total_pairs:,} pairs.")
# Check for any lingering 0.5 entries.
# Important: when we have summary-derived counts, we only want to warn about *unobserved* pairs.
# Observed pairs can legitimately take values near 0.5.
tri_vals = D_TARGET[iu_full, ju_full]
near_half = np.isfinite(tri_vals) & (np.abs(tri_vals - 0.5) < 1e-12)

if (coverage_counts is not None) and (cnt_vec is not None):
    # cnt_vec is aligned to the same upper-triangle indexing as (iu_full, ju_full)
    unobserved = (cnt_vec <= 0)
    n_half_unobs = int((near_half & unobserved).sum())
    if n_half_unobs > 0:
        print(
            f"[WARN] Found {n_half_unobs:,} entries equal to 0.5 on UNOBSERVED pairs in D_TARGET; "
            "these should have been set to NaN (padding artifacts)."
        )
    else:
        print("[INFO] No 0.5 padding remains on unobserved pairs in the masked target matrix (D_TARGET).")

    # Optional diagnostic: how many observed pairs are exactly 0.5 (not necessarily a problem)
    n_half_obs = int((near_half & ~unobserved).sum())
    if VERBOSE and (n_half_obs > 0):
        print(f"[INFO] Note: {n_half_obs:,} observed pairs are exactly 0.5 in D_TARGET (can be valid).")
else:
    # Without summary counts, we can't distinguish observed vs unobserved pairs here.
    n_half_any = int(near_half.sum())
    if n_half_any > 0:
        print(
            f"[INFO] Found {n_half_any:,} entries equal to 0.5 in D_TARGET, but summary-derived counts are unavailable, "
            "so observed vs unobserved cannot be distinguished in this check."
        )
    else:
        print("[INFO] No 0.5 entries remain in the masked target matrix (D_TARGET).")

pd.DataFrame({"filename": labels}).head()

# --- Policy: exclude conceptual contrast models by default ---
# We still *support* them for supplementary runs via --include-conceptual-contrasts.
D_nat = None
if getattr(args, "include_conceptual_contrasts", False):
    required_cols = ["natural_vs_transformed"]
    for c in required_cols:
        assert c in info.columns, f"Missing column in info: {c}"
    D_nat = build_model_rdm(labels, info, "natural_vs_transformed")
    if VERBOSE:
        print("[INFO] Including conceptual contrast: natural_vs_transformed")
else:
    if VERBOSE:
        print("[INFO] Excluding conceptual contrasts by default (use --include-conceptual-contrasts to include).")

# === Additional candidate models from CROCUFID info ===
# No additional categorical models in this dataset (kept empty to avoid noisy [skip] logs)
CATEGORICAL_MODELS = []

# Continuous models from our stimulus QC + AI judgements (canonical headers)
CONTINUOUS_MODELS = [
    "blind_ai_calorie_density",
    "blind_ai_healthiness",
    "blind_ai_sweetness",
    "blind_ai_saltiness",
    "blind_ai_sourness",
    "blind_ai_bitterness",
    "blind_ai_savoriness",
    "blind_ai_fattiness",
    "blind_ai_spiciness",
    "human_calorie_density",
    "human_healthiness",
    "human_appeal",
    "human_sweetness",
    "human_saltiness",
    "human_sourness",
    "human_bitterness",
    "human_savoriness",
    "human_fattiness",
    "human_spiciness",
]

# --- Correlation matrix of food-related factors from info sheet ---
# This helps inspect redundancy/overlap between conceptual and low-level variables.
try:
    FOOD_FACTOR_COLS = sorted(set(CATEGORICAL_MODELS + CONTINUOUS_MODELS))
    factor_cols_present = [c for c in FOOD_FACTOR_COLS if c in info.columns]

    if factor_cols_present:
        # Align to image order and encode factors numerically
        info_aligned = info.set_index("filename").reindex(labels)
        numeric_data = {}
        for col in factor_cols_present:
            s = info_aligned[col]
            # Use numeric values directly where possible; otherwise factorize categories
            if pd.api.types.is_numeric_dtype(s):
                numeric_data[col] = pd.to_numeric(s, errors="coerce")
            else:
                codes, _uniques = pd.factorize(s)
                codes = codes.astype(float)
                # Factorize returns -1 for missing values; map to NaN
                codes[codes < 0] = np.nan
                numeric_data[col] = codes

        factors_num = pd.DataFrame(numeric_data, index=info_aligned.index)
        # Pearson correlations between factors (item-level)
        factor_corr = factors_num.corr(method="pearson", min_periods=10)

        factor_corr_csv = os.path.join(RUN_RESULTS_DIR, "info_food_factor_correlations.csv")
        factor_corr.to_csv(factor_corr_csv)
        if VERBOSE:
            print(f"Food-factor correlation matrix written to: {factor_corr_csv}")
    else:
        if VERBOSE:
            print("[INFO] No factor columns present for info_food_factor_correlations.csv; skipping.")
except Exception as _e:
    print(f"[WARN] Could not compute food-factor correlation matrix: {_e}")

# Optional: skip recomputation of heavy analyses if core inputs have not changed since last run.
# Note: we place this after computing the info-sheet correlation matrix so that the
# factor correlations are always updated, even when the main RSA/NN analyses are skipped.

if SKIP_IF_UP_TO_DATE:
    try:
        manifest_path = os.path.join(RUN_RESULTS_DIR, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as _mf:
                _mani_prev = json.load(_mf)
            prev_sig = _mani_prev.get("inputs_mtime", None)
            # Current signature is based on mtimes of the core inputs
            curr_sig = {}
            for label, path in [
                ("triplet_rdm_npz", TRIPLET_RDM_NPZ),
                ("info_csv", INFO_CSV),
                ("summary_csv", SUMMARY_CSV),
            ]:
                try:
                    curr_sig[label] = os.path.getmtime(path)
                except OSError:
                    curr_sig[label] = None
            if prev_sig is not None and prev_sig == curr_sig:
                print("[INFO] Core inputs unchanged since last run; skipping heavy analyses (SKIP_IF_UP_TO_DATE=True).")
                sys.exit(0)
            else:
                print("[INFO] Inputs changed or no previous mtime signature; recomputing analyses.")
    except Exception as _e:
        print(f"[WARN] Could not check up-to-date status for skipping analyses: {_e}")

# Storage for built models
EXTRA_MODELS = {}  # name -> (matrix, kind)
MODEL_RDMS_DIR = os.path.join(DERIVATIVES_DIR, "model_rdms_canon")
os.makedirs(MODEL_RDMS_DIR, exist_ok=True)

# Dynamically load all pre-computed models from derivatives/model_rdms
# IMPORTANT: model RDMs may be saved with their own label order. If labels are present
# in the .npz, reorder to match the empirical `labels`.
import glob

def _load_any_rdm_from_npz(npz_path: str):
    """Load a square RDM from an .npz with flexible key support.

    Supports keys: rdm, data, D, matrix.
    Returns (D, saved_labels, kind).
    """
    z = np.load(npz_path, allow_pickle=True)
    if "rdm" in z:
        Dm = z["rdm"]
    elif "data" in z:
        Dm = z["data"]
    elif "D" in z:
        Dm = z["D"]
    elif "matrix" in z:
        Dm = z["matrix"]
    else:
        raise KeyError(f"No known RDM key in {npz_path}. Keys={list(z.keys())}")

    Dm = np.asarray(Dm, dtype=float)
    saved_labels = None
    if "labels" in z:
        saved_labels = z["labels"]
        saved_labels = saved_labels.tolist() if hasattr(saved_labels, "tolist") else list(saved_labels)
        saved_labels = [os.path.basename(str(x)) for x in saved_labels]

    kind = str(z["kind"]) if "kind" in z else "continuous"
    return Dm, saved_labels, kind

def _reorder_square_to_labels(Dm: np.ndarray, saved_labels: list[str], target_labels: list[str], npz_path: str) -> np.ndarray:
    """Reorder a square matrix saved in `saved_labels` order to `target_labels` order."""
    if saved_labels is None:
        return Dm
    if len(saved_labels) != Dm.shape[0] or Dm.shape[0] != Dm.shape[1]:
        raise ValueError(f"Shape/labels mismatch in {npz_path}: D={Dm.shape} labels={len(saved_labels)}")

    t = [os.path.basename(str(x)) for x in target_labels]
    idx_of = {lab: i for i, lab in enumerate(saved_labels)}
    missing = [lab for lab in t if lab not in idx_of]
    if missing:
        raise ValueError(
            f"Model RDM labels do not match empirical labels for {os.path.basename(npz_path)}. "
            f"Missing {len(missing)} label(s), e.g. {missing[:5]}"
        )
    order = [idx_of[lab] for lab in t]
    return Dm[np.ix_(order, order)]

EXTRA_MODELS = {}  # name -> (matrix, kind)
for mpath in sorted(glob.glob(os.path.join(MODEL_RDMS_DIR, "rdm_*.npz"))):
    filename = os.path.basename(mpath)
    # derive a nice model name
    name = filename.replace("rdm_", "").replace(".npz", "")

    # Optional: keep only the requested NN seed's precomputed models
    if getattr(args, "nn_seed", None) is not None and name.startswith("nn_seed"):
        wanted_prefix = f"nn_seed{int(args.nn_seed)}_"
        if not name.startswith(wanted_prefix):
            continue

    # Skip the core models that are built natively earlier
    if name in ["natural_vs_transformed", "Natural_vs_transformed", "Natural vs transformed"]:
        continue

    # Skip permanently excluded models (see policy_defaults.EXCLUDE_MODELS)
    if name in EXCLUDE_MODELS:
        if VERBOSE:
            print(f"[skip] {name}: in EXCLUDE_MODELS (permanently removed)")
        continue

    try:
        Dm, saved_labels, kind = _load_any_rdm_from_npz(mpath)
        Dm = _reorder_square_to_labels(Dm, saved_labels, labels, mpath)
        EXTRA_MODELS[name] = (Dm, kind)
        if VERBOSE:
            print(f"[OK] Loaded pre-computed model: {name} (kind={kind})")
    except Exception as e:
        print(f"[ERROR] Could not load {filename}: {e}")

        # --- Apply model-selection policy to loaded precomputed models ---
# NOTE: conceptual contrasts are handled separately via D_nat.
def _policy_keep_key(k: str) -> bool:
    src = getattr(args, "source", "human")
    include_visual = not getattr(args, "no_visual", False)

    # Categorical / transformation reference models — always include regardless of source.
    # (They are neither human_* nor ai_* so they would otherwise be dropped by the
    # rating-source filter below.)
    #
    # REVISION 2026-09-10 (revision/transformation-models):
    #   natural_vs_transformed, Transformation_score — ADDED. Previously excluded on the
    #     grounds that they duplicated NOVA_ordinal. They do not: rho(NOVA, NvT) = 0.54,
    #     rho(NOVA, Transformation_score) = 0.66 on the fixed+dense subset. See
    #     reference/policy_defaults.py and manuscript/reviews/ for the full rationale.
    #   Category_WHO_10, Category_Culinary_9 — these now enter the regression and variance
    #     partitioning as well as the univariate scan, so all three taxonomies are reported
    #     symmetrically (reviewer point RB-4a). They are added to model_rdms explicitly
    #     further down, alongside food_classification.
    if k in POLICY_ALWAYS_INCLUDE_MODELS:
        return True

    # visual (ll_* low-level models)
    if k.startswith("ll_"):
        if k in EXCLUDE_VISUAL_MODELS:
            return False
        return include_visual

    # NN seed models — excluded from paper analyses by policy.
    # They are used for the MDS space visualisation (Stage 2) and the NN
    # interpretation figure (Stage 7) but should not appear in the univariate
    # RSA table or unique-variance / Shapley results.
    # Pass --include-nn-models to override (e.g. for exploratory runs).
    if k.startswith("nn_seed"):
        return getattr(args, "include_nn_models", False)

    # rating source separation
    if src == "human":
        return k.startswith("human_")
    if src == "ai":
        return k.startswith("ai_")
    if src == "mixed":
        return k.startswith("human_") or k.startswith("ai_")

    return False  # fallback (shouldn't happen)

_before = len(EXTRA_MODELS)
_dropped_by_policy = [k for k in EXTRA_MODELS if not _policy_keep_key(k)]
EXTRA_MODELS = {k: v for k, v in EXTRA_MODELS.items() if _policy_keep_key(k)}
_after = len(EXTRA_MODELS)

if VERBOSE:
    print(f"[INFO] Policy-filtered precomputed models: kept={_after}/{_before} (source={getattr(args,'source','human')}, visual={'off' if getattr(args,'no_visual',False) else 'on'})")
    if _dropped_by_policy:
        print(f"[INFO] Policy-excluded (not used in any analysis): {sorted(_dropped_by_policy)}")

# Quick sanity: warn if low-level visual models appear to be missing
_ll_names = [n for n in EXTRA_MODELS.keys() if n.startswith("ll_") or n.startswith("lowlevel_")]
if VERBOSE:
    print(f"[INFO] Pre-computed models loaded: {len(EXTRA_MODELS)} total | low-level candidates: {len(_ll_names)}")
    if len(_ll_names) == 0:
        print("[WARN] No low-level visual models detected (expected names like 'll_*'). Check derivatives/model_rdms.")

# --- Model scan pipeline: univariate, FDR, redundancy, unique variance ---
# Inserted before test_model
# --- Assemble full model set (extras + optional conceptual contrasts) ---
ALL_MODELS: dict[str, tuple[np.ndarray, str]] = {}

# Precomputed models (AI ratings, human ratings, low-level visual, etc.)
ALL_MODELS.update(EXTRA_MODELS)

# Optional conceptual contrast (natural_vs_transformed only)
if D_nat is not None:
    ALL_MODELS["natural_vs_transformed"] = (D_nat, "conceptual_contrast")

if VERBOSE:
    n_ai = sum(1 for k in ALL_MODELS if k.startswith("ai_") or k.startswith("rdm_ai_"))
    n_hu = sum(1 for k in ALL_MODELS if k.startswith("human_") or k.startswith("rdm_human_"))
    n_ll = sum(1 for k in ALL_MODELS if k.startswith("ll_") or k.startswith("rdm_ll_"))
    n_con = sum(1 for _k, (_m, kind) in ALL_MODELS.items() if str(kind) == "conceptual_contrast")
    print(f"[INFO] Model set: total={len(ALL_MODELS)} | ai~{n_ai} | human~{n_hu} | visual~{n_ll} | conceptual_contrasts={n_con}")

# --- Optional: include NN embeddings as additional models for scan/regression ---
INCLUDE_NN_AS_MODELS = False
NN_DISTANCE = "cosine"     # or "euclidean"
NN_NCOMP_LIMIT = N_COMP     # reuse cap for per-component analysis

def _dist_matrix_from_components_df(df_scores, comp_cols, metric="cosine"):
    X = df_scores[comp_cols].to_numpy(dtype=float)
    X = X - np.nanmean(X, axis=0, keepdims=True)  # center columns
    if metric == "cosine":
        eps = 1e-12
        norms = np.sqrt((X * X).sum(axis=1, keepdims=True)) + eps
        U = X / norms
        S = U @ U.T
        D = 1.0 - S
    else:
        xx = (X**2).sum(axis=1, keepdims=True)
        D2 = xx + xx.T - 2.0 * (X @ X.T)
        D2[D2 < 0] = 0.0
        D = np.sqrt(D2, dtype=float)
    np.fill_diagonal(D, np.nan)
    return D

def _add_nn_block_to_models(df_scores, tag_prefix="NN", seed_label=None):
    if "filename" not in df_scores.columns:
        return 0
    comp_cols = [c for c in df_scores.columns if c.startswith("nn_comp")]
    if not comp_cols:
        return 0
    def _key(c):
        try:
            return int(c.replace("nn_comp",""))
        except Exception:
            return 10**9
    comp_cols.sort(key=_key)
    comp_cols_use = comp_cols[:NN_NCOMP_LIMIT] if (NN_NCOMP_LIMIT and len(comp_cols) > NN_NCOMP_LIMIT) else comp_cols
    S = df_scores.set_index("filename").reindex(labels)
    if S[comp_cols_use].isna().any().any():
        return 0
    # multivariate
    D_nn = _dist_matrix_from_components_df(S, comp_cols_use, metric=NN_DISTANCE)
    tag = f"{tag_prefix} (seed {seed_label})" if seed_label is not None else tag_prefix
    ALL_MODELS[tag] = (D_nn, f"nn_{NN_DISTANCE}")
    # per-component
    for c in comp_cols_use:
        v = S[c].to_numpy(dtype=float)
        D_c = np.abs(v[:, None] - v[None, :])
        np.fill_diagonal(D_c, np.nan)
        ALL_MODELS[f"{tag} / {c}"] = (D_c, "nn_component")
    return 1

def _try_add_nn_models_from_folder(root_dir, tag_prefix="NN"):
    added = 0
    # Accept both NN and VICE-style outputs
    agg_files = ["nn_scores_all_seeds.csv", "vice_scores_all_seeds.csv"]
    # aggregated
    for agg_fn in agg_files:
        agg = os.path.join(root_dir, agg_fn)
        if os.path.exists(agg):
            df = pd.read_csv(agg)
            if "seed" in df.columns:
                for seed_val, g in df.groupby("seed"):
                    added += _add_nn_block_to_models(g.copy(), tag_prefix=tag_prefix, seed_label=int(seed_val))
            else:
                added += _add_nn_block_to_models(df.copy(), tag_prefix=tag_prefix, seed_label=None)
    # per-seed
    if os.path.isdir(root_dir):
        for name in sorted(os.listdir(root_dir)):
            sdir = os.path.join(root_dir, name)
            if not (os.path.isdir(sdir) and name.startswith("seed_")):
                continue
            per_seed_files = ["nn_scores.csv", "vice_scores.csv"]
            for sp_fn in per_seed_files:
                sp = os.path.join(sdir, sp_fn)
                if os.path.exists(sp):
                    try:
                        seed_val = int(name.split("_",1)[-1])
                    except Exception:
                        seed_val = None
                    added += _add_nn_block_to_models(pd.read_csv(sp), tag_prefix=tag_prefix, seed_label=seed_val)
    # legacy root-level file
    legacy_files = ["nn_scores.csv", "vice_scores.csv"]
    for legacy_fn in legacy_files:
        legacy_path = os.path.join(root_dir, legacy_fn)
        if os.path.exists(legacy_path):
            added += _add_nn_block_to_models(pd.read_csv(legacy_path), tag_prefix=tag_prefix+"-legacy", seed_label=None)
    return added

if INCLUDE_NN_AS_MODELS:
    _ = _try_add_nn_models_from_folder(os.path.join(DERIVATIVES_DIR, "nn_runs"), tag_prefix="NN")
    _ = _try_add_nn_models_from_folder(os.path.join(DERIVATIVES_DIR, "nn_vice_runs"), tag_prefix="VICE")
    _ = _try_add_nn_models_from_folder(os.path.join(DERIVATIVES_DIR, "nn_runs_test"), tag_prefix="NN-test")
    legacy_scores = os.path.join(DERIVATIVES_DIR, "nn_scores.csv")
    if os.path.exists(legacy_scores):
        _add_nn_block_to_models(pd.read_csv(legacy_scores), tag_prefix="NN-legacy", seed_label=None)

if SCAN_MODELS:
    # Vectorize empirical upper triangle once
    iu_all, ju_all = np.triu_indices_from(D_TARGET, k=1)
    v_emp_all = D_TARGET[iu_all, ju_all].astype(float)

    # 1) Univariate Mantel for each model
    scan_rows = []
    for mname, (Dm, mkind) in ALL_MODELS.items():
        try:
            t = mantel_spearman(D_TARGET, Dm, n_perm=int(args.n_perm), seed=int(args.seed) + 101)
            ci = bootstrap_ci_spearman(D_TARGET, Dm, n_boot=int(args.n_boot), seed=int(args.seed) + 102, ci=0.95)
            scan_rows.append({"model": mname, "kind": mkind, "rho": t["rho"], "p_perm": t["p_perm"], "n_perm": t["n_perm"], "ci95_lo": ci[0], "ci95_hi": ci[1]})
        except Exception as e:
            print(f"[scan skip] {mname}: {e}")
    scan_df = pd.DataFrame(scan_rows)

    # 2) Attach ceiling ratio if available later; temp set NaN (will fill after ceiling computed)
    scan_df["r_over_ceiling"] = np.nan

    # 3) Model↔model redundancy: Spearman on vectorized RDMs
    names = list(ALL_MODELS.keys())
    V = []
    for m in names:
        Dm = ALL_MODELS[m][0]
        V.append(Dm[iu_all, ju_all].astype(float))
    V = np.column_stack(V)  # [n_pairs x n_models]
    M = np.full((len(names), len(names)), np.nan, dtype=float)
    # Use only empirically observed pairs when computing model↔model redundancy
    obs_mask = np.isfinite(v_emp_all)
    for i in range(len(names)):
        for j in range(i, len(names)):
            r = _spearman_vec(V[obs_mask, i], V[obs_mask, j])
            M[i, j] = M[j, i] = r
    model_corr_df = pd.DataFrame(M, index=names, columns=names)

    # 4) Save the model↔model correlation matrix now; save univariate fits later after ceiling/FDR updates
    model_scan_csv = os.path.join(RUN_RESULTS_DIR, "model_fits_univariate.csv")
    model_corr_csv = os.path.join(RUN_RESULTS_DIR, "model_model_corr.csv")
    model_corr_df.to_csv(model_corr_csv, index_label="model_name")

    # 5) Apply precomputed noise ceiling to scan results
    # ceiling value loaded from compute_data_qc.py JSON at startup (see _NC_CEILING below)
    ceiling_scan = _NC_CEILING
    _summary["ceiling_scan"] = float(ceiling_scan)
    _summary["ceiling_scan_ci"] = _NC_CEILING_CI
    _summary["ceiling_scan_kind"] = "split_half_sb_precomputed" if np.isfinite(ceiling_scan) else None
    if VERBOSE and np.isfinite(ceiling_scan):
        print(
            f"[INFO] Noise ceiling (precomputed, SB-corrected): mean={ceiling_scan:.3f}  "
            f"CI95=({_NC_CEILING_CI[0]:.3f},{_NC_CEILING_CI[1]:.3f})"
        )
    elif VERBOSE:
        print("[WARN] Noise ceiling not available — run compute_data_qc.py first. "
              "r_over_ceiling will be NaN.")

    # fill r_over_ceiling
    if np.isfinite(ceiling_scan):
        scan_df["r_over_ceiling"] = scan_df["rho"] / ceiling_scan
    _summary["scan_df"] = scan_df.copy()

    # 6) FDR across all models (use permutation p-values)
    rej, qvals = _bh_fdr(scan_df["p_perm"].to_numpy(), q=FDR_Q)
    scan_df["q_fdr"] = qvals
    scan_df["rejected_fdr"] = rej
    # Now that ceiling ratios and FDR fields have been added, write the final univariate table
    scan_df.to_csv(model_scan_csv, index=False)

    # 7) Redundancy pruning by clustering threshold
    # Build adjacency for |rho| >= REDUND_THR
    nmod = len(names)
    visited = [False]*nmod
    clusters = []
    for i in range(nmod):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        # simple BFS using threshold
        queue = [i]
        while queue:
            k = queue.pop(0)
            for j in range(nmod):
                if not visited[j] and abs(M[k, j]) >= REDUND_THR:
                    visited[j] = True
                    cluster.append(j)
                    queue.append(j)
        clusters.append(cluster)


    # Select a representative from each cluster among FDR-significant models by highest r_over_ceiling (then by rho)
    # Policy: conceptual contrast models are allowed in *univariate* scans, but excluded from *unique variance* by default.
    selected = []
    for cl in clusters:
        cl_names = [names[idx] for idx in cl]
        sub = scan_df[scan_df["model"].isin(cl_names)].copy()

        # Drop conceptual contrasts from unique-variance selection unless explicitly allowed
        if not getattr(args, "allow_conceptual_in_unique", False):
            sub = sub[sub["kind"].astype(str) != "conceptual_contrast"]

        # If nothing remains after filtering, skip this cluster
        if sub.empty:
            continue

        # prefer FDR significant; if none, take best rho anyway
        if sub["rejected_fdr"].any():
            sub = sub[sub["rejected_fdr"]]

        # break ties by r_over_ceiling then rho
        sub = sub.sort_values(["r_over_ceiling", "rho"], ascending=[False, False])
        if not sub.empty:
            selected.append(sub.iloc[0]["model"])

    selected = list(dict.fromkeys(selected))  # de-dup preserve order

    selected_df = scan_df[scan_df["model"].isin(selected)].sort_values(["r_over_ceiling","rho"], ascending=[False, False])
    selected_csv = os.path.join(RUN_RESULTS_DIR, "model_selected.csv")
    selected_df.to_csv(selected_csv, index=False)

def test_model(name, D_model, n_perm=None, n_boot=None, seed_perm=None, seed_boot=None):
    if n_perm is None:
        n_perm = int(args.n_perm)
    if n_boot is None:
        n_boot = int(args.n_boot)
    if seed_perm is None:
        seed_perm = int(args.seed) + 201
    if seed_boot is None:
        seed_boot = int(args.seed) + 202
    test = mantel_spearman(D_TARGET, D_model, n_perm=int(n_perm), seed=int(seed_perm))
    ci   = bootstrap_ci_spearman(D_TARGET, D_model, n_boot=int(n_boot), seed=int(seed_boot), ci=0.95)
    print(f"{name:28s} rho={test['rho']:.3f}  p_perm={test['p_perm']:.4g}  n_perm={test['n_perm']}  CI95=({ci[0]:.3f},{ci[1]:.3f})")
    return {"model": name, "spearman_r": test["rho"], "p_perm": test["p_perm"], "n_perm": test["n_perm"], "ci95_lo": ci[0], "ci95_hi": ci[1]}

if SCAN_MODELS:
    results = []
    # Optional conceptual contrast: natural_vs_transformed only
    if D_nat is not None:
        results.append(test_model("natural_vs_transformed", D_nat))

    # Evaluate additional models and append to results
    for mname, (Dm, mkind) in EXTRA_MODELS.items():
        try:
            results.append(test_model(mname, Dm))
        except Exception as e:
            print(f"[skip] testing model '{mname}': {e}")

    fits = pd.DataFrame(results)
    _summary["fits_df"] = fits.copy()

try:
    fig, axes = plot_heatmaps(D_TARGET, D_nat)
    fig_path = os.path.join(RUN_RESULTS_DIR, "heatmap_models.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    if VERBOSE:
        print(f"Saved heatmap to: {fig_path}")
except Exception as _heatmap_err:
    print(f"[WARN] Could not save heatmap: {_heatmap_err}")


def try_info_columns(columns, n_perm=2000, n_boot=1000):
    rows = []
    for col in columns:
        if col not in info.columns:
            print(f"[skip] {col} (not found in info)")
            continue
        # Use the right RDM builder depending on column type.
        # build_model_rdm is a 0/1 categorical function — for continuous columns
        # (e.g. human_calorie_density, averaged float ratings with no exact ties)
        # every pair gets 1.0 → constant RDM → Spearman ρ = 0 always.
        # Use build_continuous_rdm (absolute differences) for continuous columns.
        col_vals = pd.to_numeric(info[col], errors="coerce")
        n_unique = col_vals.nunique(dropna=True)
        if n_unique <= 2:
            D = build_model_rdm(labels, info, col)
        else:
            D = build_continuous_rdm(labels, info, col)
        rows.append(test_model(col, D, n_perm=n_perm, n_boot=n_boot))
    return pd.DataFrame(rows).sort_values("spearman_r", ascending=False)

# Canonical candidate model columns for quick screening:
candidate_cols = [
    "blind_ai_calorie_density",
    "blind_ai_healthiness",
    "blind_ai_sweetness",
    "blind_ai_saltiness",
    "blind_ai_sourness",
    "blind_ai_bitterness",
    "blind_ai_savoriness",
    "blind_ai_fattiness",
    "blind_ai_spiciness",
    "human_calorie_density",
    "human_healthiness",
    "human_appeal",
    "human_sweetness",
    "human_saltiness",
    "human_sourness",
    "human_bitterness",
    "human_savoriness",
    "human_fattiness",
    "human_spiciness",
]
if getattr(args, "include_conceptual_contrasts", False):
    candidate_cols = ["natural_vs_transformed"] + candidate_cols
if SCAN_MODELS:
    try_info_columns(candidate_cols)

results_csv = os.path.join(RUN_RESULTS_DIR, "model_fits.csv")
if SCAN_MODELS:
    fits.to_csv(results_csv, index=False)

    if VERBOSE:
        print(f"Model fits written to: {results_csv}")

# === k-model regression over all binary columns in the info sheet ===
if RUN_REGRESSION:
    import numpy as np
    import pandas as pd

    def _binary_info_cols(df: pd.DataFrame) -> list[str]:
        # Pick columns with exactly two non-null unique values (binary factors).
        # Skip administrative / QC columns (CONTROL_COLS) unless --controls was passed.
        # Skip conceptual contrast columns unless --include-conceptual-contrasts was passed.
        _skip_ids = {"filename", "filenames", "image", "images", "sweet_vs_savory"}
        _skip_controls = not POLICY.get("include_controls", False)
        _skip_contrasts = not POLICY.get("include_conceptual_contrasts", False)
        _CONTRAST_COLS = {"natural_vs_transformed"}
        cols = []
        for c in df.columns:
            if str(c).strip().lower() in _skip_ids:
                continue
            if _skip_controls and c in POLICY_CONTROL_COLS:
                continue
            if _skip_contrasts and c in _CONTRAST_COLS:
                continue
            try:
                nunq = df[c].dropna().nunique()
            except Exception:
                continue
            if nunq == 2:
                cols.append(c)
        return cols
    
    def _vec_upper(mat: np.ndarray) -> np.ndarray:
        iu, ju = np.triu_indices_from(mat, k=1)
        return mat[iu, ju]
    
    # 1) Discover binary columns and build model RDMs
    bin_cols = _binary_info_cols(info)
    model_rdms = {}
    for col in bin_cols:
        try:
            model_rdms[col] = build_model_rdm(labels, info, col)
        except Exception as e:
            print(f"[skip] could not build model RDM for '{col}': {e}")
    
    # --- Add continuous canonical rating regressors to the regression ---
    # Source-filtered: include only the rating source(s) appropriate for this run.
    #   source=human → human ratings only (paper Shapley baseline; manuscript line 180)
    #   source=ai    → AI ratings only
    #   source=mixed → both (model-space MDS, mixed pairwise vectors)
    _reg_source = getattr(args, "source", "human")
    _include_human = _reg_source in ("human", "mixed")
    _include_ai    = _reg_source in ("ai", "mixed")

    CONTINUOUS_REGRESSORS = []
    if _include_ai:
        CONTINUOUS_REGRESSORS += [
            "blind_ai_calorie_density",
            "blind_ai_healthiness",
            "blind_ai_sweetness",
            "blind_ai_saltiness",
            "blind_ai_sourness",
            "blind_ai_bitterness",
            "blind_ai_savoriness",
            "blind_ai_fattiness",
            "blind_ai_spiciness",
        ]
    if _include_human:
        CONTINUOUS_REGRESSORS += [
            "human_calorie_density",
            "human_healthiness",
            "human_appeal",
            "human_sweetness",
            "human_saltiness",
            "human_sourness",
            "human_bitterness",
            "human_savoriness",
            "human_fattiness",
            "human_spiciness",
        ]

    if VERBOSE:
        print(f"[INFO] Regression regressors: source={_reg_source}, "
              f"human={'on' if _include_human else 'off'}, "
              f"ai={'on' if _include_ai else 'off'} "
              f"({len(CONTINUOUS_REGRESSORS)} rating columns)")

    for col in CONTINUOUS_REGRESSORS:
        if col in info.columns:
            try:
                model_rdms[col] = build_continuous_rdm(labels, info, col)
            except Exception as e:
                print(f"[skip] continuous model '{col}': {e}")
        else:
            print(f"[skip] {col} (continuous regressor not found in info)")

    # Always include food_classification in pairwise_vectors.csv so it enters
    # partial correlations and Shapley analyses.  Pull from ALL_MODELS (already
    # loaded and reordered to match `labels`).
    if "food_classification" in ALL_MODELS:
        model_rdms["food_classification"] = ALL_MODELS["food_classification"][0]
        if VERBOSE:
            print("[INFO] Added food_classification to pairwise regression vectors.")
    else:
        print("[WARN] food_classification not found in ALL_MODELS; skipping from regression.")

    # REVISION 2026-09-10 (revision/transformation-models).
    # Transformation models: all three operationalisations now enter pairwise_vectors.csv
    # and the Shapley / partial-Mantel analyses, rather than NOVA_ordinal alone.
    #   NOVA_ordinal           — industrial processing level (Monteiro et al. 2016)
    #   natural_vs_transformed — whole food vs composite dish (binary; picked up
    #                            automatically by _binary_info_cols once
    #                            include_conceptual_contrasts is on, added here defensively)
    #   Transformation_score   — graded culinary transformation (continuous, 0-100)
    # Transformation_score is retained despite being redundant with natural_vs_transformed
    # (rho = 0.85, VIF 5.37 with all three in) because it has the HIGHEST univariate fit of
    # the three and a NEGATIVE cross-validated increment. Report the VIFs and lead the
    # attribution with Shapley; its individual partial rho is a redundancy artefact.
    #
    # Taxonomies: all three now enter the regression/VP for symmetric reporting (RB-4a).
    # Membership comes from reference/policy_defaults so this stage cannot drift from
    # the univariate filter. food_classification is added just above and is skipped by
    # the `already present` guard below.
    #
    # DELIBERATELY the transformation models ONLY, not CONCEPTUAL_SUPPLEMENTARY in full.
    # Category_WHO_10 and Category_Culinary_9 are alternative PARTITIONS OF THE SAME ITEMS
    # as food_classification, not additional constructs. Because the variance attribution
    # treats every numeric column in pairwise_vectors.csv as a predictor, adding them here
    # would enter all three taxonomies into one Shapley and split the category variance
    # three ways across near-duplicate predictors, destroying the category result rather
    # than testing it.
    #
    # Symmetric reporting of the three taxonomies (reviewer point RB-4a) is done by
    # (a) the univariate scan, which includes all three, and (b) the family-wise blocks
    # in compute_family_generalization.py, which substitute one taxonomy at a time.
    for _m in POLICY_TRANSFORMATION_MODELS:
        if _m in model_rdms:
            continue
        if _m in ALL_MODELS:
            model_rdms[_m] = ALL_MODELS[_m][0]
            if VERBOSE:
                print(f"[INFO] Added {_m} to pairwise regression vectors.")
        else:
            print(f"[WARN] {_m} not found in ALL_MODELS; skipping from regression.")

    if not model_rdms:
        print("[WARN] No binary columns found in info; skipping k-model regression.")
    else:
        # 2) Vectorize upper triangles
        v_emp = _vec_upper(D_TARGET)
        model_vecs = {name: _vec_upper(D) for name, D in model_rdms.items()}
    
        # 3) Finite mask across all vectors (empirical + all models)
        mask = np.isfinite(v_emp)
        for v in model_vecs.values():
            mask &= np.isfinite(v)
    
        n_pairs = int(mask.sum())
        _summary["reg_meta"]["n_pairs_used"] = int(n_pairs)
        if n_pairs < 10:
            print(f"[WARN] Too few valid pairs for regression (n={n_pairs}). Skipping.")
        else:
            # 4) Dump pairwise vectors (indices + values)
            iu, ju = np.triu_indices_from(D_TARGET, k=1)
            df_pairs = pd.DataFrame({
                "i": iu[mask],
                "j": ju[mask],
                "emp": v_emp[mask],
            })
            Xcols = sorted(model_vecs.keys())
            for name in Xcols:
                df_pairs[name] = model_vecs[name][mask]

            # Optionally: also include the precomputed model RDM vectors (e.g., ll_* low-level visual models)
            if getattr(args, "pairwise_include_precomputed", False):
                # Build an optional prefix filter list
                pref_raw = str(getattr(args, "pairwise_precomputed_filter", "") or "").strip()
                prefixes = [p.strip() for p in pref_raw.split(",") if p.strip()] if pref_raw else []

                def _keep_name(nm: str) -> bool:
                    if not prefixes:
                        return True
                    return any(str(nm).startswith(p) for p in prefixes)

                # Add base + EXTRA (precomputed) models that are already in ALL_MODELS
                # Use the same mask as the regression (mask over v_emp and the existing regression predictors)
                added_cols = 0
                for mname, (Dm, _kind) in ALL_MODELS.items():
                    if not _keep_name(mname):
                        continue
                    try:
                        v = _vec_upper(Dm)
                        # Ensure compatibility with the existing mask
                        if v.shape[0] != v_emp.shape[0]:
                            continue
                        col = "rdm_" + _safe_name(mname)
                        # Avoid collisions with existing column names
                        if col in df_pairs.columns:
                            continue
                        df_pairs[col] = v[mask]
                        added_cols += 1
                    except Exception:
                        continue
                if VERBOSE:
                    print(f"[INFO] pairwise_vectors: added {added_cols} precomputed model vector column(s) (prefix filter={prefixes or 'NONE'})")

            df_pairs["used_in_regression"] = True

            pairwise_csv = os.path.join(RUN_RESULTS_DIR, "pairwise_vectors.csv")
            df_pairs.to_csv(pairwise_csv, index=False)
            if VERBOSE:
                print(f"Pairwise vectors written to: {pairwise_csv}")

    
    # === Map NN embedding components to models (relational + item-level), including multi-seed support ===

# === Map NN embedding components to models (relational + item-level), including multi-seed support ===
if RUN_NN_MAPPING:
    def _load_nn_scores_frames(deriv_dir):
        """
        Load NN scores with priority:
          1) derivatives/nn_runs/nn_scores_all_seeds.csv (aggregated)
          2) derivatives/nn_runs/seed_*/nn_scores.csv (one or more seeds)
          3) derivatives/nn_scores.csv (legacy single-run)
        Returns a list of (df, seed_label) where seed_label can be an int or None.
        """
        frames = []
        nn_runs = os.path.join(deriv_dir, "nn_runs")
        # 1) Aggregated
        agg_path = os.path.join(nn_runs, "nn_scores_all_seeds.csv")
        if os.path.exists(agg_path):
            df = pd.read_csv(agg_path)
            if "seed" in df.columns:
                for seed_val, g in df.groupby("seed", as_index=False):
                    frames.append((g.copy(), int(seed_val)))
                return frames
            else:
                frames.append((df.copy(), None))
                return frames
        # 2) Per-seed folders
        if os.path.isdir(nn_runs):
            try:
                for name in sorted(os.listdir(nn_runs)):
                    sdir = os.path.join(nn_runs, name)
                    if not (os.path.isdir(sdir) and name.startswith("seed_")):
                        continue
                    sp = os.path.join(sdir, "nn_scores.csv")
                    if os.path.exists(sp):
                        try:
                            seed_val = int(name.split("_", 1)[-1])
                        except Exception:
                            seed_val = None
                        frames.append((pd.read_csv(sp), seed_val))
                if frames:
                    return frames
            except Exception:
                pass
        # 3) Legacy single-run
        legacy = os.path.join(deriv_dir, "nn_scores.csv")
        if os.path.exists(legacy):
            frames.append((pd.read_csv(legacy), None))
        return frames
    
    def _nn_component_mapping_for_frame(df_scores: pd.DataFrame, labels, info_df: pd.DataFrame, D_models: dict, n_comp_limit: int | None = None):
        # Identify nn component columns like nn_comp1, nn_comp2, ...
        comp_cols = [c for c in df_scores.columns if c.startswith("nn_comp")]
        if len(comp_cols) == 0:
            return pd.DataFrame()
        # Sort by numeric suffix
        def _key(c):
            try:
                return int(c.replace("nn_comp", ""))
            except Exception:
                return 10**9
        comp_cols = sorted(comp_cols, key=_key)
        if n_comp_limit is not None:
            comp_cols = comp_cols[:min(n_comp_limit, len(comp_cols))]
    
        # Align to label order
        if "filename" not in df_scores.columns:
            print("[WARN] nn_scores missing 'filename' column; skipping.")
            return pd.DataFrame()
        scores = df_scores.set_index("filename").reindex(labels)
        if scores[comp_cols].isna().any().any():
            missing = int(scores[comp_cols].isna().sum().sum())
            print(f"[WARN] {missing} NN score entries missing after alignment; skipping.")
            return pd.DataFrame()
    
        rows = []
        # Relational Mantel tests (build 1D RDM from each component)
        for cname in comp_cols:
            vec = scores[cname].to_numpy(dtype=float)
            D_c = np.abs(vec[:, None] - vec[None, :])
            for mname, Dmod in D_models.items():
                t = mantel_spearman(D_c, Dmod, n_perm=int(args.n_perm), seed=int(args.seed) + 701)
                ci = bootstrap_ci_spearman(D_c, Dmod, n_boot=int(args.n_boot), seed=int(args.seed) + 702, ci=0.95)
                rows.append({
                    "kind": "relational_nn",
                    "component": cname,
                    "predictor": mname,
                    "rho": t["rho"],
                    "p_perm": t["p_perm"],
                    "ci95_lo": ci[0],
                    "ci95_hi": ci[1],
                    "n_perm": t["n_perm"],
                })
    
        # Item-level correlations vs binary info columns
        def _binary_cols(df: pd.DataFrame):
            keep = []
            for c in df.columns:
                cl = str(c).strip().lower()
                if cl in {"filename", "filenames", "image", "images", "sweet_vs_savory"}:
                    continue
                try:
                    if df[c].dropna().nunique() == 2:
                        keep.append(c)
                except Exception:
                    continue
            return keep
    
        bin_cols = _binary_cols(info_df)
        info_aligned = info_df.set_index("filename").reindex(labels)
        for cname in comp_cols:
            vec = scores[cname].to_numpy(dtype=float)
            for col in bin_cols:
                if col not in info_aligned.columns:
                    continue
                try:
                    x = pd.to_numeric(info_aligned[col], errors="coerce")
                    if not set(x.dropna().unique()).issubset({0, 1}):
                        x = pd.Series(pd.factorize(info_aligned[col])[0], index=info_aligned.index)
                    msk = np.isfinite(vec) & np.isfinite(x.to_numpy(dtype=float))
                    if msk.sum() < 10:
                        continue
                    r = np.corrcoef(vec[msk], x.to_numpy(dtype=float)[msk])[0, 1]
                    rows.append({
                        "kind": "item_level_nn",
                        "component": cname,
                        "predictor": col,
                        "rho": float(r),
                        "p_perm": np.nan,
                        "ci95_lo": np.nan,
                        "ci95_hi": np.nan,
                        "n_perm": 0,
                    })
                except Exception as e:
                    print(f"[skip] item-level NN {cname} ~ {col}: {e}")
    
        return pd.DataFrame(rows)
    
    # Build dict of model RDMs used above
    _D_models = {"Natural vs transformed": D_nat}
    
    nn_frames = _load_nn_scores_frames(DERIVATIVES_DIR)
    if not nn_frames:
        print("[INFO] No NN scores found in derivatives; skipping NN component↔model mapping.")
    else:
        nn_rows = []
        for df_scores, seed_val in nn_frames:
            df_map = _nn_component_mapping_for_frame(df_scores, labels, info, _D_models, n_comp_limit=N_COMP)
            if not df_map.empty:
                if seed_val is not None:
                    df_map.insert(0, "seed", int(seed_val))
                nn_rows.append(df_map)
        if nn_rows:
            compmap_nn_df = pd.concat(nn_rows, ignore_index=True)
            compmap_nn_csv = os.path.join(RUN_RESULTS_DIR, "nn_component_model_map.csv")
            compmap_nn_df.to_csv(compmap_nn_csv, index=False)
            _summary["compmap_nn_df"] = compmap_nn_df.copy()
            if VERBOSE:
                print(f"NN Component↔model mapping written to: {compmap_nn_csv}")
        else:
            print("[INFO] NN scores present but no valid component columns found; skipping NN mapping.")
    
    
    # Update or create manifest with new artifacts
# Update or create manifest with new artifacts
try:
    manifest_path = os.path.join(RUN_RESULTS_DIR, "manifest.json")

    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as mf:
            mani = json.load(mf)
    else:
        mani = {"run_id": RUN_ID, "inputs": {}, "outputs": {}}

    mani.setdefault("inputs", {})
    mani.setdefault("outputs", {})
    mani.setdefault("metrics", {})
    mani["selected_nn_seed"] = int(args.nn_seed) if getattr(args, "nn_seed", None) is not None else None

    # record coverage stats
    if "_summary" in globals():
        cov_counts = _summary.get("coverage_counts")
        cov_finite = _summary.get("coverage_finite") or _summary.get("coverage")
        if cov_counts is not None:
            mani["metrics"]["coverage_from_summary"] = {
                "observed": int(cov_counts[0]),
                "total": int(cov_counts[1]),
                "proportion": float(cov_counts[2]),
            }
        if cov_finite is not None:
            mani["metrics"]["coverage_from_matrix_finite"] = {
                "observed": int(cov_finite[0]),
                "total": int(cov_finite[1]),
                "proportion": float(cov_finite[2]),
            }

    # --- Policy records ---
    mani.setdefault("policy", {})
    mani["policy"]["exclude_conceptual_contrasts_default"] = True
    mani["policy"]["include_conceptual_contrasts"] = bool(getattr(args, "include_conceptual_contrasts", False))
    mani["policy"]["human_vs_ai_separate"] = True
    mani["policy"]["keep_visual_models"] = True

    # record inputs
    mani["inputs"]["triplet_rdm_npz"] = EMPIRICAL_RDM_NPZ
    mani["inputs"]["triplet_rdm_npz_default"] = TRIPLET_RDM_NPZ
    mani["inputs"]["info_csv"] = INFO_CSV
    mani["inputs"]["summary_csv"] = SUMMARY_CSV

    # record input mtimes for future up-to-date checks
    inputs_mtime = {}
    for label, path in [
        ("triplet_rdm_npz", EMPIRICAL_RDM_NPZ),
        ("info_csv", INFO_CSV),
        ("summary_csv", SUMMARY_CSV),
    ]:
        try:
            inputs_mtime[label] = os.path.getmtime(path)
        except OSError:
            inputs_mtime[label] = None
    mani["inputs_mtime"] = inputs_mtime
    mani["last_run_iso"] = datetime.now().isoformat(timespec="seconds")

    # record outputs (only if defined)
    if "results_csv" in locals():
        mani["outputs"]["model_fits_csv"] = results_csv
    if "fig_path" in locals():
        mani["outputs"]["heatmap_png"] = fig_path

    # Univariate scan artifacts
    scan_uni_csv = os.path.join(RUN_RESULTS_DIR, "model_fits_univariate.csv")
    if os.path.exists(scan_uni_csv):
        mani["outputs"]["model_fits_univariate_csv"] = scan_uni_csv
    model_corr_csv = os.path.join(RUN_RESULTS_DIR, "model_model_corr.csv")
    if os.path.exists(model_corr_csv):
        mani["outputs"]["model_model_corr_csv"] = model_corr_csv
    selected_csv = os.path.join(RUN_RESULTS_DIR, "model_selected.csv")
    if os.path.exists(selected_csv):
        mani["outputs"]["model_selected_csv"] = selected_csv


    # Regression artifacts
    if "pairwise_csv" in locals() and os.path.exists(pairwise_csv):
        mani["outputs"]["pairwise_vectors_csv"] = pairwise_csv
    if "reg_csv" in locals() and os.path.exists(reg_csv):
        mani["outputs"]["rdm_regression_csv"] = reg_csv


    # NN mapping
    nn_compmap_csv_path = os.path.join(RUN_RESULTS_DIR, "nn_component_model_map.csv")
    if os.path.exists(nn_compmap_csv_path):
        mani["outputs"]["nn_component_model_map_csv"] = nn_compmap_csv_path


    # Always record where model RDMs were read from
    mani["outputs"]["model_rdms_dir"] = MODEL_RDMS_DIR

    # noise ceiling metrics (loaded from compute_data_qc.py JSON at startup)
    if np.isfinite(_NC_CEILING):
        mani["metrics"]["noise_ceiling_mean"] = float(_NC_CEILING)
        lo, hi = _NC_CEILING_CI
        if np.isfinite(lo) and np.isfinite(hi):
            mani["metrics"]["noise_ceiling_ci95"] = {"lo": float(lo), "hi": float(hi)}

    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(mani, mf, indent=2)

    if VERBOSE:
        print(f"Manifest updated: {manifest_path}")
except Exception as e:
    print(f"[WARN] Could not update manifest: {e}")