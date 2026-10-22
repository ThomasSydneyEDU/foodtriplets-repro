

#!/usr/bin/env python3
"""policy_defaults.py

Single source of truth for *default* analysis settings.

Goal
----
Keep key analytic choices explicit, stable, and easy to reference across scripts.

This file is intentionally simple: it contains only constants and small helpers.
It should be imported by compute/results scripts (and those scripts can still
override defaults via CLI flags).

Current policy (as agreed)
--------------------------
- Keep Human vs AI ratings separate for unique-variance analyses.
- Keep low-level visual models included (treated as explanatory factors).
- Conceptual contrast models (Natural vs transformed) are
  supplementary: allowed in univariate scans, excluded from unique variance
  unless explicitly requested.
- Prefer pooled participant data (SONA + Prolific) for primary analyses,
  after justifying pool merge in Stage 0.
- Prefer stable stage output folders (no timestamped run folders) by default.
- For NN-completed RDMs/embeddings: use a single representative seed going
  forward (default seed = 20; best-by-centrality after the 2026-09 retrain at
  lambda = 0.014), after reporting stability across seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# -----------------------------------------------------------------------------
# Repository / folders
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve()
ANALYSIS_DIR = HERE.parents[1]              # .../analysis_code
REPO_ROOT = ANALYSIS_DIR.parent             # repo root

DERIVATIVES_DIR = ANALYSIS_DIR / "derivatives"
REFERENCE_DIR = ANALYSIS_DIR / "reference"
RESULTS_DIR = ANALYSIS_DIR / "results"
OUTPUTS_DIR = ANALYSIS_DIR / "outputs"
COMPUTE_DIR = ANALYSIS_DIR / "compute_outputs"


# -----------------------------------------------------------------------------
# Canonical inputs (primary reference files)
# -----------------------------------------------------------------------------

INFO_CSV = REFERENCE_DIR / "Foodpictures_information_dynamic.csv"

# Dynamic reference CSV — source of truth for human_* ratings, blind_ai_*
# ratings, dynamic category models, and visual features.
DYNAMIC_INFO_CSV = REFERENCE_DIR / "Foodpictures_information_dynamic.csv"

SUMMARY_CSV = DERIVATIVES_DIR / "summary.csv"

# Observed-only (empirical) RDM produced from summary trials
EMPIRICAL_RDM_NPZ = DERIVATIVES_DIR / "triplet_rdm_by_filename.npz"

# NN-completed empirical RDM written by NN embedding script (seed is formatted)
NN_COMPLETED_RDM_FMT = DERIVATIVES_DIR / "triplet_rdm_completed_nn_seed{seed}.npz"


# -----------------------------------------------------------------------------
# Default policy switches
# -----------------------------------------------------------------------------

# Participant pooling
DEFAULT_POOL: Literal["all", "prolific", "sona"] = "all"  # primary analyses use pooled data

# Model-selection defaults
DEFAULT_SOURCE: Literal["human", "ai", "mixed"] = "human"  # for unique variance (run human-only and ai-only separately)
DEFAULT_INCLUDE_VISUAL: bool = True
DEFAULT_INCLUDE_CONTROLS: bool = False

# Columns in the information CSV that are administrative / QC and should NEVER be used as model
# predictors unless the user explicitly passes --controls.  These are distinct from the conceptual
# contrast columns (natural_vs_transformed) which are legitimate models but
# gated by a separate policy flag.
CONTROL_COLS: frozenset[str] = frozenset({
    "portion_size_ok",     # QC flag: was the portion size acceptable?
    "quality",             # Image quality tier (standard vs high) — admin, not a food dimension
    "plate_rim_visible",   # QC flag
    "file_exists",         # Admin flag
    "label_match",         # QC: does the AI label match the intended food?
    "label_confidence",    # QC score
    "qc_issues",           # QC notes
    "qc_reasons",          # QC notes
    "qc_at",               # QC timestamp
    "qc_model",            # Which model ran QC — admin
})

# Conceptual contrasts: allowed in univariate scans, excluded from unique variance by default
# NOTE (revision/transformation-models branch, 2026-09-10): re-enabled in response to
# peer review. The previous branch set these False on the grounds that binary NvT was
# "replaced by" NOVA_ordinal and competed for the same processing variance. On the
# fixed+dense subset (703 pairs) rho(NOVA_ordinal, natural_vs_transformed) = 0.54, so
# they are not the same variance and the two measure different constructs: NOVA codes
# industrial processing level, natural_vs_transformed codes whole food vs composite dish
# (baked salmon and prepared asparagus are both coded Natural). Reviewer B raised exactly
# this point. Both now enter all analyses. See manuscript/reviews/ for the diagnostics.
DEFAULT_INCLUDE_CONCEPTUAL_IN_UNIVARIATE: bool = True
DEFAULT_INCLUDE_CONCEPTUAL_IN_UNIQUE: bool = True

# Exclusion lists for specific model features
# NOTE: EXCLUDE_VISUAL_MODELS was previously used to drop 7 scalar visual models
# from ALL_MODELS for compact visualisation (MDS model-space, family-wise plots).
# It is now empty so that all 11 visual feature RDMs enter ALL_MODELS and are
# therefore available for the full-feature Shapley attribution (via
# --pairwise-include-precomputed), consistent with the paper methods (line 106/182).
# Compact visualisation scripts handle their own 4-model subset independently.
EXCLUDE_VISUAL_MODELS: frozenset[str] = frozenset()

# Models permanently removed from all analyses.
# Listed here so the exclusion is explicit and discoverable rather than relying
# on the implicit policy-key filter in compute_model_evaluation.py.
# sweet_vs_savory: removed — redundant with continuous sweetness/savoriness dims.
#
# REVISION 2026-09-10 (revision/transformation-models): natural_vs_transformed,
# concept_natural_vs_transformed and Transformation_score are NO LONGER excluded.
# The previous rationale was that they competed with NOVA_ordinal for the same
# processing variance. That does not hold: on the fixed+dense subset (703 pairs)
# rho(NOVA_ordinal, natural_vs_transformed) = 0.54 and rho(NOVA_ordinal,
# Transformation_score) = 0.66, and the three code different constructs
# (industrial processing level; whole food vs composite dish; graded culinary
# transformation, the only one that separates raw from prepared versions of the
# same item). Reviewers 2 and B both challenged the reliance on NOVA alone.
#
# All three now enter every analysis. Transformation_score is retained despite
# being redundant with natural_vs_transformed (rho = 0.85, VIF 5.37 with all
# three in) BECAUSE it has the highest univariate fit of the three (rho = 0.357,
# above every flavour dimension) and a NEGATIVE cross-validated increment over
# Semantic + Category. Excluding the best univariate performer would look like
# suppression; letting it fail publicly is the result. Report the VIFs, note that
# its individual partial rho is a redundancy artefact, and lead the attribution
# with Shapley, which distributes shared variance across correlated predictors.
# AI RATING MODELS — REMOVED FROM THE PAPER (2026-09-11).
# The nine ai_* rating RDMs are Gemini-generated food ratings. They are not
# reported anywhere in foodtriplets_manuscript_v4.1.docx: the manuscript
# contains one instance of "AI", and it is the name of the stimulus database
# (PAFID). They were still being computed by every run, which cost ~20 min per
# run in the source=ai regression alone and left AI rows in output tables and
# summaries that the paper does not use.
#
# They never contaminated a reported number. The human pairwise_vectors.csv
# carries 25 predictors with no ai_ column, so the Shapley never saw them; the
# family-wise CV fits each block independently; univariate rho is per-model.
# The one shared quantity was the Benjamini-Hochberg correction in the
# univariate scan, which corrected across all 36 models. Dropping the nine
# moves 26 q-values in the fourth decimal and flips no model's significance at
# q <= 0.05, and the manuscript reports no FDR values at all.
AI_RATING_MODELS: tuple[str, ...] = (
    "ai_bitterness",
    "ai_calorie_density",
    "ai_fattiness",
    "ai_healthiness",
    "ai_saltiness",
    "ai_savoriness",
    "ai_sourness",
    "ai_spiciness",
    "ai_sweetness",
)

EXCLUDE_MODELS: frozenset[str] = frozenset({
    *AI_RATING_MODELS,
    # the blind/aware variants carry the same ratings under prefixed keys
    *(f"blind_{m}" for m in AI_RATING_MODELS),
    *(f"aware_{m}" for m in AI_RATING_MODELS),
    "sweet_vs_savory",
    "Sweet_vs_savory",
    "sweet vs savory",
    "Sweet vs savory",
    # Category_NOVA_4: categorical NOVA — still excluded. rho = 0.89 with
    # NOVA_ordinal, so it is pure redundancy rather than a distinct construct.
    "Category_NOVA_4",
    # Category_WHO_10 and Category_Culinary_9 are NOT excluded here. As of this
    # branch they enter the regression/VP as well as the univariate scan, so the
    # three taxonomies are reported symmetrically (reviewer point RB-4a).
})

# ---------------------------------------------------------------------------
# Canonical model groups — SINGLE SOURCE OF TRUTH
# ---------------------------------------------------------------------------
# REVISION 2026-09-10 (revision/transformation-models).
#
# These membership lists were previously duplicated, and had drifted apart, in:
#   analysis_helpers.CONCEPTUAL_SUPPLEMENTARY        (gated Fig 3 MDS + variance attribution)
#   compute_model_evaluation._policy_keep_key        (gated the univariate scan)
#   compute_model_evaluation model_rdms additions    (gated pairwise_vectors.csv)
#   compute_family_generalization.FAMILIES / BLOCKS  (gated Fig 5)
#   results3_model_evaluation_plots.conceptual_keys  (gated Fig 4 grouping)
#
# The practical consequence was that adding a model in one place silently failed
# to reach the others: a predictor could appear in pairwise_vectors.csv and then
# be filtered back out of the Shapley. Every module now imports from here.
#
# TAXONOMY_MODELS      — competing partitions of the same items. Reported
#                        symmetrically as of this branch (reviewer point RB-4a).
# TRANSFORMATION_MODELS — three operationalisations of food transformation. NOT
#                        interchangeable: rho(NOVA, NvT) = 0.54,
#                        rho(NOVA, Transformation_score) = 0.66 on the fixed+dense
#                        subset. They code industrial processing level, whole food
#                        vs composite dish, and graded culinary transformation
#                        respectively.
#
# NOTE: "food_classification" is the runtime key for Category_Intuitive_7; the
# alias is applied in analysis_helpers.load_info(). Use the runtime key here.

TAXONOMY_MODELS: tuple[str, ...] = (
    "food_classification",   # = Category_Intuitive_7 (primary; aliased at load time)
    "Category_WHO_10",
    "Category_Culinary_9",
)

TRANSFORMATION_MODELS: tuple[str, ...] = (
    "NOVA_ordinal",           # industrial processing level (Monteiro et al. 2016)
    "natural_vs_transformed", # whole food vs composite dish (binary)
    "Transformation_score",   # graded culinary transformation (continuous 0-100)
)

# Models that are neither human_* nor ai_* and must therefore bypass the
# rating-source filter in every module that applies one.
CONCEPTUAL_SUPPLEMENTARY: tuple[str, ...] = TAXONOMY_MODELS + TRANSFORMATION_MODELS
ALWAYS_INCLUDE_MODELS: frozenset[str] = frozenset(CONCEPTUAL_SUPPLEMENTARY)

# Sanity: a model must not be both always-included and permanently excluded.
_conflict = ALWAYS_INCLUDE_MODELS & EXCLUDE_MODELS
if _conflict:
    raise ValueError(
        f"policy_defaults: models both always-included and excluded: {sorted(_conflict)}"
    )


# Output conventions
DEFAULT_TIMESTAMPED_OUTPUTS: bool = False


# -----------------------------------------------------------------------------
# NN defaults
# -----------------------------------------------------------------------------

# After NN stability checks, we use a single seed going forward.
# 2026-09-14 retrain (lambda = 0.014, 290-participant sample): best seed by median
# centrality across the 25-seed Mantel similarity matrix is 20 (0.9793, 23 kept dims).
# Seed 19 was the June 2026 choice under the same rule and now ranks 10th (0.9744).
# The field spans 0.9656-0.9793 and rho(19,20) = 0.9747, so the seeds are near
# equivalent; 20 is used because the selection rule, not the narrative, decides.
DEFAULT_NN_SEED: int = 20

# For scripts that need to decide between empirical vs filled.
DEFAULT_RDM_MODE: Literal["empirical", "filled"] = "empirical"


# -----------------------------------------------------------------------------
# Stage folder helpers
# -----------------------------------------------------------------------------

STAGE_DIRS = {
    # Results scripts (stable stage folders)
    "stage0_data_qc_reliability_pooling": RESULTS_DIR / "stage0_data_qc_reliability_pooling",
    "stage0_nn_seed_stability": RESULTS_DIR / "stage0_nn_seed_stability",

    "stage1_behavioural_space_2d": RESULTS_DIR / "stage1_behavioural_space_2d",
    "stage2_model_space_2d": RESULTS_DIR / "stage2_model_space_2d",

    "stage3_model_evaluation_plots": RESULTS_DIR / "stage3_model_evaluation_plots",
    "stage4_model_generalization_plots": RESULTS_DIR / "stage4_model_generalization_plots",
    "stage5_variance_attribution_plots": RESULTS_DIR / "stage5_variance_attribution_plots",
}


def stage_outdir(stage: str) -> Path:
    """Return the stable output dir for a stage (creates nothing)."""
    return STAGE_DIRS.get(stage, RESULTS_DIR / stage)


def get_compute_dir(stage_name: str, rdm_mode: str, source: str) -> Path:
    """Standardized compute output path builder: compute_outputs / stage_name / rdm_mode / source"""
    return COMPUTE_DIR / stage_name / rdm_mode / source


def get_results_dir(stage_name: str, rdm_mode: str, source: str) -> Path:
    """Standardized results output path builder: results / stage_name / rdm_mode / source"""
    return RESULTS_DIR / stage_name / rdm_mode / source


def nn_completed_rdm_path(seed: int | None = None) -> Path:
    """Resolve the NN-completed RDM path for a given seed (defaults to DEFAULT_NN_SEED)."""
    if seed is None:
        seed = DEFAULT_NN_SEED
    return Path(str(NN_COMPLETED_RDM_FMT).format(seed=int(seed)))


@dataclass(frozen=True)
class ModelPolicy:
    """A small structured bundle for logging policy settings in manifests."""

    source: Literal["human", "ai", "mixed"] = DEFAULT_SOURCE
    include_visual: bool = DEFAULT_INCLUDE_VISUAL
    include_controls: bool = DEFAULT_INCLUDE_CONTROLS
    include_contrasts_in_univariate: bool = DEFAULT_INCLUDE_CONCEPTUAL_IN_UNIVARIATE
    include_contrasts_in_unique: bool = DEFAULT_INCLUDE_CONCEPTUAL_IN_UNIQUE


DEFAULT_MODEL_POLICY = ModelPolicy()
