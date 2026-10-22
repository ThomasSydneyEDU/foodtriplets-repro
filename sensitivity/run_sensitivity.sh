#!/usr/bin/env bash
# Review-round sensitivity analyses on the public data. Called by run_all.sh after the main
# pipeline (needs derivatives/, compute_outputs/stage2 and, for partial_3x3, stage4 Shapley).
# quick = reduced resampling; full = the settings used for the revision.
set -euo pipefail
MODE="${1:-quick}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"
S=sensitivity
mkdir -p $S/sensitivity_outputs
if [ "$MODE" = "full" ]; then NB=10000; REPS=200; B=1000; else NB=1000; REPS=20; B=100; fi
run() { local l="$1"; shift; echo; echo "── $l"; "$@" || { echo "[FAIL] $l" >&2; exit 1; }; }

run "Trial composition (random vs fixed+dense)"          $PYTHON $S/sensitivity_trial_composition.py
run "Random-trial bootstrap of taxonomy contrasts"       $PYTHON $S/sensitivity_random_trials_bootstrap.py --n-boot $NB
run "Taxonomy difference bootstrap (fixed+dense)"        $PYTHON $S/sensitivity_taxonomy_difference.py --n-boot $NB
run "Item-wise cross-validation"                         $PYTHON $S/sensitivity_itemwise_cv.py --reps $REPS
run "Item-wise cross-validation incl. cooking measures"  $PYTHON $S/sensitivity_cooked_itemwise_cv.py --reps $REPS
if [ "$MODE" = "full" ]; then
  # These read the Shapley manifest/outputs written by the full run (stage4_variance_attribution).
  run "Cooking-based transformation measures"            $PYTHON $S/sensitivity_cooked_measure.py --n-boot 200
  run "Kind versus cooking head-to-head"                 $PYTHON $S/sensitivity_kind_vs_cooking.py --B $B
  run "Partial 3x3 + Shapley 3x3"                        $PYTHON $S/sensitivity_partial_3x3.py --shapley
  run "Shapley item bootstrap"                           $PYTHON $S/sensitivity_shapley_item_bootstrap.py --B 200 --n-orderings 300 --models all15
else
  echo "[quick] skipped: cooking measures, kind-vs-cooking, partial 3x3, Shapley item bootstrap (need the Shapley outputs; use full)"
fi
