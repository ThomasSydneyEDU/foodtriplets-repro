# Food triplets: analysis code and reproduction

[![DOI](https://zenodo.org/badge/1429124943.svg)](https://doi.org/10.5281/zenodo.23579454)

Code to reproduce the analyses, tables and figures in Carlson, Avery, Hebart et al. (Cognition, under revision) from the public data.

## What you need

| Item | Where | In this repo |
|---|---|---|
| Anonymised trial data (140,160 trials, 290 participants) | Zenodo, doi:10.5281/zenodo.23498576 | `data/foodtriplets_trial_data.csv` (see `data/README.md`) |
| PAFID item information and cached visual features (350 images) | Zenodo, doi:10.5281/zenodo.23226251 | `analysis_code/reference/` |
| Cooking-based transformation coding | this repo | `analysis_code/reference/cooked_transformation_coding.csv` |
| SPoSE seed-20 embedding used in the paper | this repo | `analysis_code/derivatives/nn_runs/`, `analysis_code/cached/` |

Python 3.10 to 3.12: `pip install -r requirements.txt` (flexible ranges). `requirements-lock.txt` lists the exact package versions used during development, and `requirements-nn.txt` adds torch for optional SPoSE retraining. On macOS use scipy 1.16 or later.

## Run

```
bash run_all.sh quick     # about an hour: RDMs, RSA, regressions, family CV, figures, sensitivity analyses (reduced resampling)
bash run_all.sh full      # adds noise ceilings and the Shapley variance attribution (2,000 orderings; many hours)
```

Tables are written as CSV to `analysis_code/compute_outputs/` and `sensitivity/sensitivity_outputs/`; figures to `analysis_code/results/`. The script stops at the first failing step.

## What reproduces exactly and what does not

Participant IDs and row order in the public file are shuffled to protect participants. Analyses that do not depend on that order (empirical RDM, all RSA fits, regressions, family cross-validation, Shapley attribution, item bootstraps) reproduce the manuscript values. SPoSE training and the split-half noise ceilings depend on participant order, so they are approximate if rerun. By default the paper's SPoSE seed-20 solution is reused; `RUN_NN=1 bash run_all.sh full` retrains it.

Not included: the 82 participants excluded before analysis, so the exclusion-sensitivity and catch-trial analyses cannot be rerun (available from the corresponding author on request). No API calls are made; the language-model annotations behind the SPoSE component labels are cached in `analysis_code/results/stage7_nn_interpretation/`.

## Layout

- `scripts/make_summary_from_release.py` converts the public trial file to the schema used by the pipeline.
- `analysis_code/` pipeline scripts (`compute_*`, `prelim_*`, `results*`), `reference/` inputs, `derivatives/` generated RDMs.
- `sensitivity/` analyses added during peer review; run by `sensitivity/run_sensitivity.sh`.

## Citation

Please cite the paper and the data records above.

## Runtime and checks

`quick` took about two hours on a laptop (it includes the slow univariate permutation scans) and `full` takes many more hours (the Shapley attribution at 2,000 orderings dominates). On the public data, the empirical RDM and pair counts, all univariate RSA fits, the regressions and the family cross-validation matched the original pipeline outputs to numerical precision. Resampling-based intervals (bootstraps, split-half ceilings) match up to Monte Carlo error. `analysis_code/prelim_preprocess_data.py` documents the raw-data preprocessing and is not run, because raw sessions are not public.

## Licence and citation

MIT licence (see `LICENSE`). Citation metadata is in `CITATION.cff`; the released version is archived on Zenodo.
