#!/usr/bin/env bash
# =============================================================================
# run_all.sh  --  reproduce the FoodTriplets analyses from the public data
#
# Input : data/foodtriplets_trial_data.csv  (anonymised trials, Zenodo 10.5281/zenodo.23498576)
#         analysis_code/reference/*.csv      (PAFID item information, visual features,
#                                             cooking coding; PAFID v1.0, Zenodo 10.5281/zenodo.23226251)
# Output: analysis_code/compute_outputs/   tables (CSV) for every analysis
#         analysis_code/results/           figures
#         sensitivity/outputs/             review-round sensitivity analyses (CSV)
#
# Usage : bash run_all.sh [quick|full]            (default: quick)
#   quick  data -> RDMs -> univariate RSA, regressions, family CV, figures,
#          sensitivity analyses with reduced resampling.  Roughly an hour on a laptop.
#   full   adds data-quality / noise ceilings and the Shapley variance attribution
#          with 2,000 orderings and 200 bootstrap resamples (many hours).
#
# Notes on exactness
#   * Participant IDs and row order in the public file are shuffled. Analyses that are
#     order-invariant (RSA, regressions, family cross-validation, Shapley, item
#     bootstraps) reproduce the paper to numerical precision. SPoSE training and the
#     split-half noise ceilings depend on participant order, so they are approximate.
#     To keep the paper's embedding, the SPoSE seed-20 solution is shipped in
#     analysis_code/derivatives/nn_runs and re-used; retraining is optional (RUN_NN=1).
#   * The 82 participants excluded before analysis are not in the public file, so the
#     exclusion-sensitivity analyses are not part of this script (available on request).
#   * No API calls are made. The language-model annotations used for the component
#     interpretation figure are cached in analysis_code/results/stage7_nn_interpretation.
# =============================================================================
set -euo pipefail

MODE="${1:-quick}"
case "$MODE" in quick|full) ;; *) echo "usage: bash run_all.sh [quick|full]"; exit 2;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
AC="analysis_code"
PYTHON="${PYTHON:-python3}"
NN_SEED="${NN_SEED:-20}"
export TRIPLETS_AI_MODE=blind
export PYTHONHASHSEED=0

if [ "$MODE" = "full" ]; then
    N_ORDERINGS="${N_ORDERINGS:-2000}"
else
    N_ORDERINGS="${N_ORDERINGS:-100}"
fi
export N_ORDERINGS

T0=$(date +%s)
step() {                     # step "label" cmd...   (fail fast)
    local label="$1"; shift
    echo; echo "── $label"; local t=$(date +%s)
    "$@" || { echo "[FAIL] $label" >&2; exit 1; }
    echo "[OK] $label ($(( $(date +%s) - t ))s)"
}

# ---------------------------------------------------------------- 0. data -> RDMs
mkdir -p "$AC/derivatives/model_rdms_canon"
step "Convert public trial file to analysis schema" \
    $PYTHON scripts/make_summary_from_release.py data/foodtriplets_trial_data.csv "$AC/derivatives/_summary_input.csv"
step "Empirical triplet RDM + pair counts" \
    $PYTHON "$AC/compute_prelim_component.py" --from-summary "$AC/derivatives/_summary_input.csv"
# compute_prelim_component writes summary.csv in the pipeline's own format
step "Model RDMs (ratings, contrasts, food classification)" \
    $PYTHON "$AC/prelim_build_model_rdms.py" --source mixed --include-contrasts --include-food-classification
step "Category, NOVA and transformation RDMs" \
    $PYTHON "$AC/prelim_build_dynamic_category_rdms.py"
step "Visual model RDMs (features read from PAFID reference table)" \
    $PYTHON "$AC/prelim_make_visual_models.py" --build-rdms --deepnet-resnet50 --deepnet-resnet50-layer3
step "Restore shipped SPoSE seed-${NN_SEED} component RDMs" \
    cp "$AC"/cached/rdm_nn_seed${NN_SEED}_*.npz "$AC/derivatives/model_rdms_canon/"

if [ "${RUN_NN:-0}" = "1" ]; then
    step "SPoSE training, 25 seeds (optional, hours; approximate)" \
        $PYTHON "$AC/compute_nn_embedding_decomp.py" --seeds $(seq 0 24) --lmbda 0.014 --write_completed_rdm --aggregate
fi

# ---------------------------------------------------------------- 1. data quality (full only)
if [ "$MODE" = "full" ]; then
    step "Data quality + noise ceiling [pooled]"      $PYTHON "$AC/compute_data_qc.py"
    step "Data quality + noise ceiling [fixed+dense]" $PYTHON "$AC/compute_data_qc.py" --fixed-dense-only
fi

# ---------------------------------------------------------------- 2. RSA + regression
step "Univariate RSA [fixed+dense]" \
    $PYTHON "$AC/compute_model_evaluation.py" --run-univariate --source mixed --fixed-dense-only --skip-unique
step "Univariate RSA [filled, NN seed ${NN_SEED}]" \
    $PYTHON "$AC/compute_model_evaluation.py" --run-univariate --source mixed --use-completed-nn-seed "$NN_SEED" --skip-unique
step "Regression + pairwise vectors [empirical]" \
    $PYTHON "$AC/compute_model_evaluation.py" --run-regression --source human --pairwise-include-precomputed --pairwise-precomputed-filter ll_
step "Regression + pairwise vectors [fixed+dense]" \
    $PYTHON "$AC/compute_model_evaluation.py" --run-regression --source human --fixed-dense-only --pairwise-include-precomputed --pairwise-precomputed-filter ll_
step "Regression + pairwise vectors [filled]" \
    $PYTHON "$AC/compute_model_evaluation.py" --run-regression --source human --use-completed-nn-seed "$NN_SEED" --pairwise-include-precomputed --pairwise-precomputed-filter ll_
step "Family-level cross-validated regression [fixed+dense]" \
    $PYTHON "$AC/compute_family_generalization.py" --fixed-dense-only
step "Transformation beyond semantic + visual predictors [fixed+dense]" \
    $PYTHON "$AC/compute_family_generalization.py" --fixed-dense-only --transformation-beyond-semantic-visual-only
step "Family-level cross-validated regression [filled]" \
    $PYTHON "$AC/compute_family_generalization.py" --no-fixed-dense-only --empirical-rdm "$AC/derivatives/triplet_rdm_completed_nn_seed${NN_SEED}.npz"

# ---------------------------------------------------------------- 3. Shapley (full only)
if [ "$MODE" = "full" ]; then
    for v in fixed_dense filled empirical; do
        step "Shapley variance attribution [$v] ($N_ORDERINGS orderings)" \
            $PYTHON "$AC/compute_variance_attribution.py" \
                --pairwise-csv "$AC/compute_outputs/stage2_model_evaluation/$v/human/pairwise_vectors.csv" \
                --source human --method suite --n-orderings "$N_ORDERINGS" --bootstrap 200 --rank-transform \
                --outdir "$AC/compute_outputs/stage4_variance_attribution/$v/human/"
    done
fi

# ---------------------------------------------------------------- 4. figures
[ "$MODE" = "full" ] && step "Fig: data quality / reliability" $PYTHON "$AC/results0_data_qc_reliability_pooling.py"
step "Fig: behavioural similarity space"      $PYTHON "$AC/results1_behavioural_space_2d.py"
step "Fig: model space"                       $PYTHON "$AC/results2_model_space_2d.py" --source human
step "Fig: univariate RSA [fixed+dense]"      $PYTHON "$AC/results3_model_evaluation_plots.py" --rdm-mode fixed_dense --source mixed --no-ai
step "Fig: family regression [fixed+dense]"   $PYTHON "$AC/results5_family_generalization_plots.py" --rdm-mode fixed_dense --compact
step "Fig: univariate RSA [filled]"           $PYTHON "$AC/results3_model_evaluation_plots.py" --rdm-mode filled --source mixed --no-ai
step "Fig: family regression [filled]"        $PYTHON "$AC/results5_family_generalization_plots.py" --rdm-mode filled --compact
if [ "$MODE" = "full" ]; then
    for v in fixed_dense filled empirical; do
        step "Fig: variance attribution [$v]" $PYTHON "$AC/results6_variance_attribution_plots.py" --rdm-mode "$v" --source human
    done
    step "Full-feature attribution [fixed+dense]" $PYTHON "$AC/run_full_feature_attribution.py" --n-orderings "$N_ORDERINGS" --n-boot 200 --rdm-mode fixed_dense
    step "Full-feature attribution [filled]"      $PYTHON "$AC/run_full_feature_attribution.py" --n-orderings "$N_ORDERINGS" --n-boot 200 --rdm-mode filled
fi
step "Fig: SPoSE component interpretation (cached annotations)" $PYTHON "$AC/results7_nn_interpretation_codified.py"

# ---------------------------------------------------------------- 5. supplementary + review-round analyses
if [ -x sensitivity/run_sensitivity.sh ] || [ -f sensitivity/run_sensitivity.sh ]; then
    step "Sensitivity and supplementary analyses ($MODE)" bash sensitivity/run_sensitivity.sh "$MODE"
fi

echo; echo "Done ($MODE) in $(( ($(date +%s) - T0) / 60 )) min."
echo "Tables : $AC/compute_outputs/    Figures : $AC/results/    Sensitivity : sensitivity/outputs/"
