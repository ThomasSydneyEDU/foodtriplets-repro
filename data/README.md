# FoodTriplets trial-level data

`foodtriplets_trial_data.csv` holds every retained triplet judgement from the analysed sample (290 participants, 140,160 trials). One row per trial.

| Column | Meaning |
|---|---|
| participant | Anonymous ID (P001 to P290, randomly assigned; not linked to Prolific or SONA identifiers) |
| pool | Recruitment pool: `prolific` or `sona` |
| session | Session within participant (almost always 1; a few participants resumed after an interruption, and one SONA code appears in three sessions, giving 1,440 trials) |
| trial | Trial order within participant and session (catch trials are not included) |
| sampling | Triplet sampling scheme: `fixed` (100 triplets per participant, the same for everyone), `dense` (150 triplets per participant, drawn from a 30-item subset) or `random` (230 triplets per participant, drawn uniformly from all 350 items). The fixed and dense trials together cover the 209 items and 703 item pairs used for the primary model evaluation |
| item_A, item_B, item_C | The three foods shown, named by PAFID image file name without `.png` |
| odd_one_out | The food the participant chose as the odd one out (always one of the three) |
| rt_ms | Response time in milliseconds |

Item names match the image files in PAFID (Katsoulis & Carlson, 2026; https://doi.org/10.5281/zenodo.23226251), which also holds the item metadata and category classifications.

Excluded: catch trials, participants removed by the attention-check criterion, pilot sessions, and all identifying fields.

Source: derived from `analysis_code/derivatives/summary.csv` (pipeline output of `prelim_preprocess_data.py`).
