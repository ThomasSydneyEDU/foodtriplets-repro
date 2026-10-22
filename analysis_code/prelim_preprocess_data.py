"""prelim_preprocess_data.py

Stage 0 (Prelim): Preprocess raw behavioral data.

Reads raw participant session CSVs from the `data/` folder, cleans them, and outputs
processed trial-level summaries to `analysis_code/processed/` and `analysis_code/derivatives/`.
"""

import os
import glob
import sys
import csv
import json
from datetime import datetime
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

# Paths
SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
PROCESSED_DIR = os.path.join(SCRIPT_DIR, 'processed')  # new folder next to this script
os.makedirs(PROCESSED_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, 'outputs')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Timestamped run directory under results
RUN_ID = datetime.now().strftime('%Y-%m-%d_%H%M%S')
RUN_RESULTS_DIR = os.path.join(RESULTS_DIR, RUN_ID)
os.makedirs(RUN_RESULTS_DIR, exist_ok=True)

# Global accumulators for catch-trial performance across participants
# - ERROR_COUNTS: total incorrect per catch_id
# - SCORED_COUNTS: total scored (non-NA) per catch_id
GLOBAL_CATCH_ERROR_COUNTS = Counter()
GLOBAL_CATCH_SCORED_COUNTS = Counter()

# Global accumulator for catch-trial explanations across participants
GLOBAL_CATCH_EXPLANATIONS = []


# Configuration
EXPECTED_TRIPLET_COUNT = 500
# Require near-complete data; 500 total includes ~20 catch trials (saved file will have ~480)
MIN_TRIPLET_SAVE_THRESHOLD = 450  # if below this, do not save processed file
TOTAL_TARGET_TRIALS = 35000  # overall project target, across all (saved) participants

# Catch IDs to EXCLUDE from the *attention-check threshold* (catch accuracy cutoff).
# We still remove these trials from saved data and still report their errors.
EXCLUDE_CATCH_IDS_FOR_THRESHOLD = {"catch_5", "catch_8"}


# --- Catch-trial helpers ---

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_category_map() -> dict:
    """Return a mapping from image stem (e.g., 'banana') -> Category.

    Tries (in order):
      1) stim_generation/plated_stimuli/stimuli_master.json (most robust)
      2) stim_generation/food_list_labeled_chatGPTlist_CORE.csv

    Returns an empty dict if nothing can be loaded.
    """
    root = _repo_root()

    # 1) stimuli_master.json
    master_path = root / "stim_generation" / "plated_stimuli" / "stimuli_master.json"
    if master_path.exists():
        try:
            data = json.loads(master_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                m = {}
                for entry in data:
                    if not isinstance(entry, dict):
                        continue
                    img = entry.get("image_file")
                    cat = entry.get("category") or entry.get("Category")
                    if img and cat:
                        stem = os.path.splitext(os.path.basename(str(img)))[0].strip().lower()
                        if stem:
                            m[stem] = str(cat)
                if m:
                    return m
        except Exception:
            pass

    # 2) CORE csv
    csv_path = root / "stim_generation" / "food_list_labeled_chatGPTlist_CORE.csv"
    if csv_path.exists():
        try:
            cdf = pd.read_csv(csv_path)
            if {"Food", "Category"}.issubset(set(cdf.columns)):
                m = {}
                for _, r in cdf.iterrows():
                    food = str(r.get("Food", "")).strip()
                    cat = str(r.get("Category", "")).strip()
                    if not food or not cat:
                        continue
                    # mimic stim slugify: lowercase, non-alnum -> '-', trim
                    import re
                    stem = re.sub(r"[^a-z0-9]+", "-", food.lower()).strip("-")[:80]
                    if stem:
                        m[stem] = cat
                if m:
                    return m
        except Exception:
            pass

    return {}


# --- Catch answer key helper ---
def _load_catch_answer_key() -> dict:
    """
    Load Catch_Trials.js and return a mapping:
      triplet_id -> odd_index
    """
    root = _repo_root()
    js_path = root / "expt_stimuli" / "Catch_Trials.js"
    if not js_path.exists():
        return {}

    text = js_path.read_text(encoding="utf-8")

    import re, json
    m = re.search(r"CATCH_TRIALS\s*=\s*(\[[\s\S]*?\]);", text)
    if not m:
        return {}

    try:
        data = json.loads(m.group(1))
    except Exception:
        return {}

    key = {}
    for d in data:
        tid = d.get("triplet_id")
        odd = d.get("odd_index")
        if tid is not None and odd is not None:
            key[str(tid)] = int(odd)
    return key


def _infer_correct_index_by_category(images: list, category_map: dict) -> 'Optional[int]':
    """Infer the odd-one-out index using category majority rule.

    If exactly one image has a category that differs from the other two (2 vs 1),
    return the index of the singleton category. Otherwise return None.
    """
    if not images or len(images) != 3 or not category_map:
        return None

    stems = [os.path.splitext(os.path.basename(str(p)))[0].strip().lower() for p in images]
    cats = [category_map.get(s) for s in stems]

    # Require all 3 categories known
    if any(c is None or str(c).strip() == "" for c in cats):
        return None

    # Majority vote (2 vs 1)
    counts = {}
    for c in cats:
        counts[c] = counts.get(c, 0) + 1
    if sorted(counts.values()) == [1, 2]:
        singleton_cat = [k for k, v in counts.items() if v == 1][0]
        return cats.index(singleton_cat)

    return None


def _infer_catch_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask for catch trials.

    Preferred rules (in order):
      1) Explicit is_catch-like column if present (True/False/1/0/yes/no).
      2) triplet_id prefix heuristic (case-insensitive): triplet_id starting with 'catch'.
      3) Legacy heuristic: presence of a non-null 'correct' value (older datasets).

    If none apply, returns all-False.
    """
    n = len(df)
    if n == 0:
        return pd.Series([], dtype=bool)

    lower_cols = {c.lower(): c for c in df.columns}

    # --- 1) Explicit catch flag columns (preferred) ---
    for key in ['is_catch', 'catch', 'catch_trial', 'iscatch']:
        if key in lower_cols:
            col = lower_cols[key]
            s = df[col]

            # Handle actual booleans
            if s.dtype == bool:
                mask = s.fillna(False)
            else:
                # Handle numeric encodings
                num = pd.to_numeric(s, errors='coerce')
                if num.notna().any():
                    mask = (num.fillna(0) > 0.5)
                else:
                    # Handle strings
                    lc = s.astype(str).str.strip().str.lower()
                    truthy = {'1', '1.0', 'true', 't', 'yes', 'y'}
                    mask = lc.isin(truthy).fillna(False)

            # IMPORTANT: only trust this column if it actually marks at least one row
            if mask.any():
                return mask
            # otherwise fall through to other detection rules

    # --- 2) triplet_id naming convention (FoodTriplets2) ---
    # Allow for different casings: triplet_id, tripletId, TRIPLET_ID, etc.
    tid_col = None
    for key in ['triplet_id', 'tripletid']:
        if key in lower_cols:
            tid_col = lower_cols[key]
            break

    if tid_col is not None:
        tid = df[tid_col].astype(str).str.strip().str.lower()
        return tid.str.startswith('catch') | tid.str.contains('catch')

    # --- 3) Legacy: non-null correctness marker ---
    # Also treat the presence of a correctness value as implying these rows are special/catch-like.
    correct_col = None
    for key in ['correct', 'is_correct']:
        if key in lower_cols:
            correct_col = lower_cols[key]
            break
    if correct_col is not None:
        return df[correct_col].notna()

    return pd.Series([False] * n)


def _infer_pool(df: pd.DataFrame) -> str:
    """Infer participant pool from columns/values.

    Returns one of: 'sona', 'prolific', or 'unknown'.
    """
    if df is None or len(df) == 0:
        return "unknown"

    lower_cols = {c.lower(): c for c in df.columns}

    # 1) Explicit pool column if present
    if "pool" in lower_cols:
        col = lower_cols["pool"]
        s = df[col].dropna().astype(str).str.strip().str.lower()
        if not s.empty:
            v = s.iloc[-1]
            if "prolific" in v:
                return "prolific"
            if "sona" in v:
                return "sona"

    # 2) Prolific PID present
    for k in ["prolificpid", "prolific_pid", "participantid"]:
        if k in lower_cols:
            col = lower_cols[k]
            s = df[col].dropna().astype(str).str.strip()
            if s.notna().any() and (s.astype(str).str.len() > 0).any():
                return "prolific"

    # 3) SONA code present
    if "sonacode" in lower_cols:
        col = lower_cols["sonacode"]
        s = df[col].dropna().astype(str).str.strip()
        if s.notna().any() and (s.astype(str).str.len() > 0).any():
            return "sona"

    # 4) Platform label (sometimes stored)
    if "platform" in lower_cols:
        col = lower_cols["platform"]
        s = df[col].dropna().astype(str).str.strip().str.lower()
        if not s.empty:
            v = s.iloc[-1]
            if "prolific" in v:
                return "prolific"
            if "sona" in v:
                return "sona"

    return "unknown"


def process_file(filepath: str) -> tuple:  # (filename, pool, n_catch, catch_acc_str, n_saved, did_save, sona_code, n_triplets)
    filename = os.path.basename(filepath)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"[ERROR] Failed to read {filename}: {e}")
        return (filename, "unknown", 0, "Failed to read file", 0, False, '-', 0)

    pool = _infer_pool(df)

    # --- Extract catch-trial explanations (from all rows, not just triplet trials) ---
    # Identify rows where task == "catch_explanation" (case-insensitive)
    lower_cols_df = {c.lower(): c for c in df.columns}
    task_col = None
    for k in ['task']:
        if k in lower_cols_df:
            task_col = lower_cols_df[k]
            break
    explain_rows = pd.Series([False]*len(df))
    if task_col is not None:
        explain_rows = df[task_col].astype(str).str.strip().str.lower() == "catch_explanation"
    # Find the explanation text column: response, catch_explanation, or text (in order)
    explanation_col = None
    for cand in ['response', 'catch_explanation', 'text']:
        if cand in lower_cols_df:
            explanation_col = lower_cols_df[cand]
            break
    tripletid_col = None
    for cand in ['triplet_id', 'tripletid']:
        if cand in lower_cols_df:
            tripletid_col = lower_cols_df[cand]
            break
    explain_df = pd.DataFrame()
    if explain_rows.any() and (explanation_col is not None) and (tripletid_col is not None):
        explain_df = df.loc[explain_rows, [tripletid_col, explanation_col]].copy()
        explain_df = explain_df.rename(columns={tripletid_col: "triplet_id", explanation_col: "catch_explanation"})
        # Drop rows with missing triplet_id or empty explanation text
        explain_df["triplet_id"] = explain_df["triplet_id"].astype(str).str.strip()
        explain_df["catch_explanation"] = explain_df["catch_explanation"].astype(str).str.strip()
        explain_df = explain_df[(explain_df["triplet_id"] != "") & (explain_df["catch_explanation"] != "")]
        # If multiple explanation rows exist for the same catch triplet, keep the last one
        explain_df = explain_df.drop_duplicates(subset=["triplet_id"], keep="last")

    # Load catch answer key from Catch_Trials.js
    catch_answer_key = _load_catch_answer_key()

    # Extract sonaCode (case-insensitive) if available
    sona_code = None
    lower_cols = {c.lower(): c for c in df.columns}
    if 'sonacode' in lower_cols:
        sc_col = lower_cols['sonacode']
        sc_series = df[sc_col]
        # Prefer the last non-null value; fallback to first non-null
        if sc_series.notna().any():
            sona_code = str(sc_series.dropna().iloc[-1])
    if not sona_code:
        sona_code = '-'

    # Filter triplet trials (jsPsych odd-one-out implementation)
    is_triplet = (
        (df.get('trial_type', pd.Series([None]*len(df))) == 'html-button-response') &
        (df.get('task', pd.Series([None]*len(df))) == 'odd_one_out')
    )
    triplet_df = df.loc[is_triplet].copy()

    # Keep only relevant columns if present (case-insensitive), and normalize key column names.
    wanted_lc = {
        # Core trial info
        'trial_index', 'trial_type', 'plugin_version',
        'rt', 'time_elapsed', 'stimulus_source',
        'task',

        # Participant / run context (lightweight, useful for QC)
        'pool', 'prolificpid', 'participantid', 'sonacode',
        'studyid', 'sessionid', 'run_env',

        # Quality / termination markers
        'total_triplets', 'success', 'timeout',
        'failed_images', 'failed_audio', 'failed_video',

        # Triplet identifiers / structure
        'triplet_id',
        'is_catch', 'is_fixed', 'is_dense', 'is_random',

        # Stimulus + response mappings
        'images_original', 'images_onscreen', 'position_map',
        'choice_index_onscreen', 'choice_index_original',
        'choice_image',
        'correct_choice_original', 'correct_choice_onscreen',

        # Catch correctness marker
        'correct',
    }

    # 1) Select wanted columns case-insensitively
    keep_cols = [c for c in triplet_df.columns if c.lower() in wanted_lc]
    if keep_cols:
        triplet_df = triplet_df[keep_cols].copy()

    # 2) Normalize common variant spellings/casing to canonical names used downstream
    rename_map = {}
    for c in triplet_df.columns:
        cl = c.lower()
        if cl in ('tripletid',):
            rename_map[c] = 'triplet_id'
        elif cl in ('iscatch', 'catch', 'catch_trial', 'catchtrial'):
            rename_map[c] = 'is_catch'
        elif cl in ('isfixed', 'fixed', 'is_fixed'):
            rename_map[c] = 'is_fixed'
        elif cl in ('isdense', 'dense', 'is_dense'):
            rename_map[c] = 'is_dense'
        elif cl in ('israndom', 'random', 'is_random'):
            rename_map[c] = 'is_random'
        elif cl in ('participant_id', 'participantid'):
            rename_map[c] = 'participantID'
        elif cl in ('prolific_pid', 'prolificpid'):
            rename_map[c] = 'prolificPID'
        elif cl in ('sonacode', 'sona_code'):
            rename_map[c] = 'sonaCode'
        elif cl in ('studyid', 'study_id'):
            rename_map[c] = 'studyID'
        elif cl in ('sessionid', 'session_id'):
            rename_map[c] = 'sessionID'
    if rename_map:
        triplet_df = triplet_df.rename(columns=rename_map)

    # Drop rows missing triplet_id if column exists
    if 'triplet_id' in triplet_df.columns:
        triplet_df = triplet_df.dropna(subset=['triplet_id']).copy()
    # Triplet trial count (used in reporting even if we later skip the file)
    n_triplets = len(triplet_df)

    # --- Version gate: require non-null `correct` values on catch trials ---
    if 'correct' not in triplet_df.columns:
        print(f"\n=== {filename} ===")
        print("[SKIP] Old experiment version detected: missing 'correct' column on triplet trials. Not saving.")
        return (filename, pool, 0, "Skipped (old version: missing correct column)", 0, False, sona_code, n_triplets)

    if triplet_df['correct'].notna().sum() == 0:
        print(f"\n=== {filename} ===")
        print("[SKIP] Old experiment version detected: 'correct' column is all null on triplet trials. Not saving.")
        return (filename, pool, 0, "Skipped (old version: correct all null)", 0, False, sona_code, n_triplets)

    # Report triplet trial counts
    print(f"\n=== {filename} ===")
    print(f"Triplet trials: {n_triplets} (expected {EXPECTED_TRIPLET_COUNT})")
    print(f"Minimum required to save: {MIN_TRIPLET_SAVE_THRESHOLD}")
    if n_triplets != EXPECTED_TRIPLET_COUNT:
        print("[WARN] Triplet count deviates from expected.")

    # Catch trials are now explicitly indicated by non-null values in `correct`
    catch_mask = triplet_df['correct'].notna()
    n_catch = int(catch_mask.sum())

    # Exclude some catch IDs from the *threshold* computation (but keep them for reporting/removal)
    catch_mask_for_threshold = catch_mask.copy()
    n_catch_excluded = 0
    if n_catch > 0 and 'triplet_id' in triplet_df.columns and EXCLUDE_CATCH_IDS_FOR_THRESHOLD:
        tids = triplet_df.loc[catch_mask, 'triplet_id'].astype(str)
        catch_labels = tids.str.extract(r'(catch_\d+)', expand=False).fillna(tids).astype(str).str.strip().str.lower()
        exclude_mask = catch_labels.isin({c.lower() for c in EXCLUDE_CATCH_IDS_FOR_THRESHOLD})
        n_catch_excluded = int(exclude_mask.sum())
        # Align back to triplet_df index
        exclude_idx = exclude_mask[exclude_mask].index
        catch_mask_for_threshold.loc[exclude_idx] = False

    # --- Prepare for catch explanation merging ---
    explain_catch_df = pd.DataFrame()

    catch_acc_str = "No catch trials detected"
    catch_acc_value = None
    if n_catch > 0:
        # Compute catch correctness primarily from the explicit `correct` column.
        correct_bool = None
        s = triplet_df.loc[catch_mask_for_threshold, 'correct']
        if s.notna().any():
            num = pd.to_numeric(s, errors='coerce')
            if num.notna().any():
                correct_bool = (num.fillna(0) > 0.5)
            else:
                lc = s.astype(str).str.strip().str.lower()
                truthy = {'1', '1.0', 'true', 't', 'yes', 'y'}
                falsy = {'0', '0.0', 'false', 'f', 'no', 'n'}
                correct_bool = lc.isin(truthy)
                unknown = ~(lc.isin(truthy | falsy))
                if unknown.any():
                    correct_bool = correct_bool.mask(unknown)

        # Fallback: if we couldn't parse `correct`, score using Catch_Trials.js answer key.
        if correct_bool is None and (
            'triplet_id' in triplet_df.columns and
            'choice_index_onscreen' in triplet_df.columns and
            catch_answer_key
        ):
            scored = []
            for _, r in triplet_df.loc[catch_mask_for_threshold].iterrows():
                tid = str(r.get('triplet_id', '')).strip()
                chosen = pd.to_numeric(r.get('choice_index_onscreen'), errors='coerce')
                odd_original = catch_answer_key.get(tid)

                if odd_original is None or pd.isna(chosen):
                    scored.append(pd.NA)
                    continue

                # Map original index -> onscreen index using position_map if available
                pos_map = r.get('position_map')
                try:
                    if isinstance(pos_map, str):
                        pos_map = json.loads(pos_map)
                except Exception:
                    pos_map = None

                correct_onscreen = None
                if isinstance(pos_map, list) and odd_original in pos_map:
                    correct_onscreen = pos_map.index(odd_original)

                if correct_onscreen is not None:
                    scored.append(int(chosen) == int(correct_onscreen))
                else:
                    # Fallback: compare directly (legacy / defensive)
                    scored.append(int(chosen) == int(odd_original))

            correct_bool = pd.Series(scored, index=triplet_df.loc[catch_mask_for_threshold].index)
            if correct_bool.notna().any():
                print("[INFO] Catch correctness scored using Catch_Trials.js answer key (fallback).")
            else:
                correct_bool = None

        # --- Catch-trial error reporting + histogram by catch ID ---
        # Run for any file with catch trials where we could compute correctness.
        # Produces:
        #  - console summary of incorrect counts by catch_* id
        #  - accumulates errors for group-level histogram across all participants
        if n_catch > 0 and correct_bool is not None:
            # Extract triplet_id values for catch trials (all catch trials for error reporting)
            catch_triplet_ids = triplet_df.loc[catch_mask, 'triplet_id'].astype(str)

            # Normalize to just catch label (e.g., catch_0, catch_1, ...)
            catch_labels = catch_triplet_ids.str.extract(r'(catch_\d+)', expand=False).fillna(catch_triplet_ids)

            # Identify incorrect catch trials (ignore NAs, using correctness on scored subset)
            # For error reporting, use the scored subset, but report for all catch trials present
            incorrect_mask = pd.Series(False, index=catch_mask.index)
            if correct_bool is not None:
                incorrect_mask.loc[correct_bool.index] = correct_bool.eq(False)
            catch_labels_incorrect = catch_labels[incorrect_mask]
            error_counts = Counter(catch_labels_incorrect)

            # Denominators: how many catch trials were scored (non-NA) for each catch id
            scored_mask = pd.Series(False, index=catch_mask.index)
            scored_mask.loc[correct_bool.index] = correct_bool.notna()
            catch_labels_scored = catch_labels[scored_mask]
            scored_counts = Counter(catch_labels_scored)

            # Accumulate into global counters (group-level)
            for cid, c in scored_counts.items():
                GLOBAL_CATCH_SCORED_COUNTS[cid] += int(c)
            for cid, c in error_counts.items():
                GLOBAL_CATCH_ERROR_COUNTS[cid] += int(c)

            # Always include the expected IDs if present in the dataset (and keep a sensible order)
            # Prefer catch_0..catch_9 ordering; then any other labels.
            observed_labels = sorted(set(catch_labels.dropna().unique().tolist()))

            def _catch_sort_key(x: str):
                try:
                    if isinstance(x, str) and x.startswith('catch_'):
                        tail = x.split('_', 1)[1]
                        return (0, int(tail))
                except Exception:
                    pass
                return (1, str(x))

            sorted_ids = sorted(observed_labels, key=_catch_sort_key)
            counts = [int(error_counts.get(cid, 0)) for cid in sorted_ids]

            # Console reporting
            total_incorrect = int(sum(counts))
            total_scored = int(correct_bool.notna().sum())
            total_correct = int(correct_bool.eq(True).sum())
            total_na = int(correct_bool.isna().sum())

            if total_incorrect > 0:
                print(f"[INFO] Catch errors: {total_incorrect} incorrect out of {total_scored} scored (correct={total_correct}, NA={total_na}).")
                print("[INFO] Incorrect counts by catch ID:")
                for cid, c in zip(sorted_ids, counts):
                    if c > 0:
                        print(f"  - {cid}: {c}")
            else:
                print(f"[INFO] Catch errors: 0 incorrect out of {total_scored} scored (correct={total_correct}, NA={total_na}).")

        # --- Merge explanations into catch trials (if any) ---
        if explain_df is not None and not explain_df.empty:
            catch_trials = triplet_df.loc[catch_mask].copy()

            # Attach scored correctness aligned to catch trial rows BEFORE merging.
            # This avoids length-mismatch errors if the explanation merge duplicates rows.
            if correct_bool is not None:
                catch_trials["correct"] = pd.Series(correct_bool.values, index=catch_trials.index.intersection(correct_bool.index))
            else:
                catch_trials["correct"] = pd.NA

            explain_catch_df = pd.merge(
                catch_trials,
                explain_df,
                on="triplet_id",
                how="left",
                suffixes=("", "_explain")
            )
            # Only keep rows with a non-null, non-empty explanation
            explain_catch_df["catch_explanation"] = explain_catch_df["catch_explanation"].astype(str).str.strip()
            explain_catch_df = explain_catch_df[explain_catch_df["catch_explanation"] != ""]
            explain_catch_df = explain_catch_df[explain_catch_df["catch_explanation"].notna()]
            # Only keep relevant columns for output
            keep_output_cols = ["triplet_id", "catch_explanation", "correct", "choice_image", "images_original"]
            explain_catch_df = explain_catch_df[[c for c in keep_output_cols if c in explain_catch_df.columns]].copy()
        else:
            explain_catch_df = pd.DataFrame()

        if correct_bool is not None:
            catch_acc = correct_bool.mean()
            # If everything is NA, mean() will be NA; treat as not-computable
            catch_acc_value = catch_acc
            n_scored = int(catch_mask_for_threshold.sum())
            if pd.notna(catch_acc):
                if n_catch_excluded > 0:
                    catch_acc_str = f"{catch_acc*100:.1f}% ({n_scored} catch trials; excluded {n_catch_excluded} from threshold)"
                else:
                    catch_acc_str = f"{catch_acc*100:.1f}% ({n_scored} catch trials)"
            else:
                n_scored = int(catch_mask_for_threshold.sum())
                catch_acc_str = f"N/A (could not compute correctness); {n_scored} catch trials scored"
        else:
            n_scored = int(catch_mask_for_threshold.sum())
            catch_acc_str = f"N/A (no correctness fields); {n_scored} catch trials scored"

    print(f"Catch trial accuracy: {catch_acc_str}")

    # Combined skip rule: skip if (low count) OR (low catch accuracy)
    reasons = []
    if n_triplets < MIN_TRIPLET_SAVE_THRESHOLD:
        reasons.append(f"low trial count ({n_triplets}<{MIN_TRIPLET_SAVE_THRESHOLD})")
    if catch_acc_value is not None and catch_acc_value < 0.90:
        reasons.append(f"low catch accuracy ({catch_acc_value*100:.1f}%<90%)")
    if reasons:
        reason_str = " and ".join(reasons)
        print(f"[SKIP] {reason_str}. Not saving.")
        return (filename, pool, n_catch, f"Skipped ({reason_str})", 0, False, sona_code, n_triplets)

    # Remove catch trials directly from the triplet trials
    removed_count = int(catch_mask.sum())
    triplet_df = triplet_df.loc[~catch_mask].copy()
    n_saved = len(triplet_df)
    print(f"Removed {removed_count} catch trials from saved data. Saving {n_saved} triplet rows.")

    # Save to processed folder with the same filename
    out_path = os.path.join(PROCESSED_DIR, filename)
    try:
        triplet_df.to_csv(out_path, index=False)
        print(f"[OK] Saved processed triplet data to: {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save processed file for {filename}: {e}")

    # --- Save explanations for this file (if any) ---
    if explain_catch_df is not None and not explain_catch_df.empty:
        explain_out_path = os.path.join(RUN_RESULTS_DIR, f"catch_explanations_{filename}")
        try:
            explain_catch_df.to_csv(explain_out_path, index=False)
            print(f"[OK] Saved catch-trial explanations to: {explain_out_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save catch explanations for {filename}: {e}")
        # Append to global accumulator for group-level output
        GLOBAL_CATCH_EXPLANATIONS.extend(explain_catch_df.to_dict(orient="records"))

    return (filename, pool, n_catch, catch_acc_str, n_saved, True, sona_code, n_triplets)


if __name__ == "__main__":
    # Process all matching CSV files in the data directory
    pattern = os.path.join(DATA_DIR, 'FoodTriplets*.csv')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching {pattern}")
    else:
        print(f"Found {len(files)} file(s) in {DATA_DIR}")

        # --- Clear processed/ before writing (added 2026-09-14) ---------------
        # This script only ever WROTE into processed/; it never removed anything.
        # Sessions saved by an earlier run therefore survived every later run,
        # even after a criterion change made them ineligible. Three sessions
        # written on 29 May 2026 persisted that way and were still being carried
        # into derivatives/summary.csv in September, inflating the analysed
        # sample from 290 participants to 293. Clearing first makes this script
        # idempotent: processed/ always reflects exactly one run.
        # Pass --keep-processed to suppress (e.g. to resume a partial run).
        if '--keep-processed' in sys.argv:
            print("[INFO] --keep-processed: leaving existing processed/ files in place")
        else:
            stale = sorted(glob.glob(os.path.join(PROCESSED_DIR, '*.csv')))
            for fp_old in stale:
                os.remove(fp_old)
            print(f"[INFO] Cleared {len(stale)} existing file(s) from {PROCESSED_DIR}")
        summary = []
        total_saved_trials = 0
        participants_saved = 0
        for fp in files:
            result = process_file(fp)
            if result is not None:
                summary.append(result)
                _, _, _, _, n_saved, did_save, _, _ = result
                if did_save:
                    total_saved_trials += int(n_saved)
                    participants_saved += 1

        # Print summary
        print("\n=== Summary of Catch Trial Performance ===")
        print(f"{'File':<40} | {'Pool':<9} | {'Catch Trials':<12} | {'Accuracy':<28} | {'Saved Triplets':<14}")
        print("-" * 115)
        for fname, pool, n_catch, acc_str, n_saved, did_save, sona_code, n_triplets in summary:
            saved_str = str(n_saved) if did_save else "-"
            print(f"{fname:<40} | {pool:<9} | {str(n_catch):<12} | {acc_str:<28} | {saved_str:<14}")

        # Grand totals
        print("\n=== Totals ===")
        print(f"Participants saved: {participants_saved}")
        print(f"Total saved triplet trials (sum across saved participants): {total_saved_trials}")

        if TOTAL_TARGET_TRIALS > 0:
            pct = 100.0 * total_saved_trials / TOTAL_TARGET_TRIALS
            print(f"Percent of overall target (vs {TOTAL_TARGET_TRIALS:,} trials): {pct:.1f}%")
        else:
            print("Percent of target: N/A (TOTAL_TARGET_TRIALS not set)")

        # Treat "old version" files separately from genuine attention-check failures
        failed_participants = [
            row for row in summary
            if (not row[5]) and ("Skipped" in row[3]) and ("old version" not in str(row[3]).lower())
        ]
        old_version_files = [
            row for row in summary
            if (not row[5]) and ("old version" in str(row[3]).lower())
        ]
        n_failed = len(failed_participants)
        print(f"\n=== Attention Check Failures ===")
        print(f"Participants who failed attention checks: {n_failed}")
        if n_failed > 0:
            for fname, pool, _, _, _, _, sona_code, n_triplets in failed_participants:
                print(f" - {fname} | pool: {pool} | sonaCode: {sona_code} | triplet_trials_completed: {n_triplets}")

        print(f"\n=== Old Experiment Version Files Skipped ===")
        print(f"Files skipped because 'correct' was missing/all-null: {len(old_version_files)}")
        if old_version_files:
            for fname, pool, _, acc_str, _, _, sona_code, n_triplets in old_version_files:
                print(f" - {fname} | pool: {pool} | sonaCode: {sona_code} | triplet_trials_completed: {n_triplets} | {acc_str}")

        # --- Pool-level acceptance summary ---
        def _is_rejected_attention(row) -> bool:
            # New tuple: (fname, pool, n_catch, acc_str, n_saved, did_save, sona_code, n_triplets)
            did_save = bool(row[5])
            acc_str = str(row[3])
            return (not did_save) and ("Skipped" in acc_str) and ("old version" not in acc_str.lower())

        def _is_accepted(row) -> bool:
            return bool(row[5])

        sona_accepted = sum(1 for r in summary if r[1] == "sona" and _is_accepted(r))
        sona_rejected = sum(1 for r in summary if r[1] == "sona" and _is_rejected_attention(r))
        prol_accepted = sum(1 for r in summary if r[1] == "prolific" and _is_accepted(r))
        prol_rejected = sum(1 for r in summary if r[1] == "prolific" and _is_rejected_attention(r))
        unk_accepted = sum(1 for r in summary if r[1] == "unknown" and _is_accepted(r))
        unk_rejected = sum(1 for r in summary if r[1] == "unknown" and _is_rejected_attention(r))

        print("\n=== Pool Summary (Accepted / Rejected by Attention Checks) ===")
        print(f"SONA     accepted: {sona_accepted:>4}   rejected: {sona_rejected:>4}")
        print(f"Prolific accepted: {prol_accepted:>4}   rejected: {prol_rejected:>4}")
        if unk_accepted or unk_rejected:
            print(f"Unknown  accepted: {unk_accepted:>4}   rejected: {unk_rejected:>4}")

        # Write CSV summary file for all processed participants
        summary_path = os.path.join(RUN_RESULTS_DIR, "processing_summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["sonaCode", "triplet_trials_completed", "catch_accuracy", "pass"])  # header
            for fname, pool, n_catch, acc_str, n_saved, did_save, sona_code, n_triplets in summary:
                # Parse catch_accuracy as float percent if possible
                catch_acc_pct = None
                if isinstance(acc_str, str) and "%" in acc_str:
                    try:
                        catch_acc_pct = float(acc_str.split("%")[0])
                    except Exception:
                        catch_acc_pct = None
                # Determine pass/fail (aligned with save threshold and min trials)
                pass_criteria = False
                if (catch_acc_pct is not None) and (n_triplets >= MIN_TRIPLET_SAVE_THRESHOLD) and (catch_acc_pct >= 90.0):
                    pass_criteria = True
                # For the CSV, use a clean accuracy string; if not parseable, keep `acc_str`
                if catch_acc_pct is not None:
                    catch_acc_out = f"{catch_acc_pct:.1f}%"
                else:
                    catch_acc_out = acc_str
                writer.writerow([sona_code, n_triplets, catch_acc_out, pass_criteria])
        print(f"\n[INFO] Processing summary written to: {summary_path}")

        # Write attention check log
        log_path = os.path.join(RUN_RESULTS_DIR, "log.txt")
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write("=== Preprocessing Run Log ===\n")
            logf.write(f"Run ID: {RUN_ID}\n")
            logf.write(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n\n")
            logf.write(f"Expected triplet count: {EXPECTED_TRIPLET_COUNT}\n")
            logf.write(f"Min trials to save: {MIN_TRIPLET_SAVE_THRESHOLD}\n")
            logf.write("Catch accuracy threshold: 90%\n")
            logf.write(f"Total target trials: {TOTAL_TARGET_TRIALS}\n\n")
            logf.write(f"Files scanned: {len(summary)}\n")
            logf.write(f"Participants saved: {participants_saved}\n")
            logf.write(f"Total saved triplet trials: {total_saved_trials}\n")
            logf.write(f"Attention check failures: {n_failed}\n\n")
            if n_failed > 0:
                logf.write("-- Failed participants --\n")
                for fname, pool, _, _, _, _, sona_code, n_triplets in failed_participants:
                    logf.write(f"{fname} | pool: {pool} | sonaCode: {sona_code} | triplet_trials_completed: {n_triplets}\n")
        print(f"[INFO] Log written to: {log_path}")

        # === Group-level catch error histogram across all participants ===
        if GLOBAL_CATCH_ERROR_COUNTS or GLOBAL_CATCH_SCORED_COUNTS:
            # Ensure stable ordering: catch_0, catch_1, ...
            def _catch_sort_key(x: str):
                try:
                    if isinstance(x, str) and x.startswith('catch_'):
                        return int(x.split('_', 1)[1])
                except Exception:
                    pass
                return float('inf')

            all_ids = set(GLOBAL_CATCH_ERROR_COUNTS.keys()) | set(GLOBAL_CATCH_SCORED_COUNTS.keys())
            group_ids = sorted(all_ids, key=_catch_sort_key)
            group_counts = [int(GLOBAL_CATCH_ERROR_COUNTS.get(cid, 0)) for cid in group_ids]
            group_scored = [int(GLOBAL_CATCH_SCORED_COUNTS.get(cid, 0)) for cid in group_ids]

            print("\n=== Group-level Catch Error Rates ===")
            for cid, n_err, n_sc in zip(group_ids, group_counts, group_scored):
                rate = (100.0 * n_err / n_sc) if n_sc > 0 else float('nan')
                if n_sc > 0:
                    print(f"{cid}: {n_err}/{n_sc} incorrect  ({rate:.1f}%)")
                else:
                    print(f"{cid}: {n_err}/0 incorrect  (n/a)")

            # Keep the histogram as counts (useful for visual)
            plt.figure(figsize=(6, 4))
            plt.bar(group_ids, group_counts, color='tomato')
            plt.xlabel('Catch ID')
            plt.ylabel('Total Incorrect Responses')
            plt.title('Catch Trial Errors (All Participants)')
            plt.tight_layout()

            group_fig_path = os.path.join(RUN_RESULTS_DIR, 'catch_error_hist_GROUP.png')
            plt.savefig(group_fig_path)
            plt.close()
            print(f"[INFO] Group-level catch error histogram saved: {group_fig_path}")
        else:
            print("\n[INFO] No catch-trial errors detected across participants; group histogram not created.")

        # === Group-level catch explanations output across all participants ===
        if GLOBAL_CATCH_EXPLANATIONS:
            group_explain_path = os.path.join(RUN_RESULTS_DIR, "catch_explanations_GROUP.csv")
            group_explain_df = pd.DataFrame(GLOBAL_CATCH_EXPLANATIONS)
            # Ensure columns: triplet_id, catch_explanation, correct, choice_image, images_original
            cols = ["triplet_id", "catch_explanation", "correct", "choice_image", "images_original"]
            group_explain_df = group_explain_df[[c for c in cols if c in group_explain_df.columns]]
            try:
                group_explain_df.to_csv(group_explain_path, index=False)
                print(f"[OK] Group-level catch explanations saved: {group_explain_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save group catch explanations: {e}")
            # Print a short console summary: counts per catch_* id
            print("\n=== Catch Explanation Counts by Catch ID ===")
            catch_id_counts = group_explain_df["triplet_id"].astype(str).value_counts().sort_index()
            for cid, count in catch_id_counts.items():
                print(f"{cid}: {count} explanation(s)")

            # --- Write one text file per catch ID with all explanations ---
            # Create a run-specific folder so repeated runs don't overwrite prior outputs
            outputs_run_dir = os.path.join(OUTPUTS_DIR, RUN_ID)
            os.makedirs(outputs_run_dir, exist_ok=True)

            # Group explanations by catch triplet_id
            for cid in sorted(group_explain_df["triplet_id"].dropna().astype(str).unique().tolist()):
                cid_str = str(cid).strip()
                if cid_str == "":
                    continue

                sub = group_explain_df[group_explain_df["triplet_id"].astype(str) == cid_str].copy()

                # Prefer a stable ordering: incorrect first, then correct, then unknown
                if "correct" in sub.columns:
                    # Map to sort keys (False -> 0, True -> 1, NA/other -> 2)
                    def _ck(v):
                        if pd.isna(v):
                            return 2
                        if isinstance(v, (bool,)):
                            return 1 if v else 0
                        s = str(v).strip().lower()
                        if s in {"false", "0", "0.0", "no", "n"}:
                            return 0
                        if s in {"true", "1", "1.0", "yes", "y"}:
                            return 1
                        return 2
                    sub["__correct_sort__"] = sub["correct"].apply(_ck)
                    sub = sub.sort_values(by=["__correct_sort__"]).drop(columns=["__correct_sort__"])

                out_txt = os.path.join(outputs_run_dir, f"{cid_str}_explanations.txt")

                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(f"Catch explanations for {cid_str}\n")
                    f.write(f"Run ID: {RUN_ID}\n")
                    f.write(f"N explanations: {len(sub)}\n")
                    f.write("=" * 60 + "\n\n")

                    for i, row in enumerate(sub.itertuples(index=False), start=1):
                        d = row._asdict() if hasattr(row, "_asdict") else dict(zip(sub.columns, row))
                        expl = str(d.get("catch_explanation", "")).strip()
                        corr = d.get("correct", pd.NA)
                        choice = str(d.get("choice_image", "")).strip()
                        imgs = d.get("images_original", "")

                        f.write(f"[{i}] correct={corr}  choice_image={choice}\n")
                        if imgs is not None and str(imgs).strip() != "":
                            f.write(f"    images_original: {imgs}\n")
                        f.write(f"    explanation: {expl}\n\n")

                print(f"[OK] Wrote explanations text file: {out_txt}")

        # Write run manifest (machine-readable)
        manifest = {
            "run_id": RUN_ID,
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "paths": {
                "data_dir": DATA_DIR,
                "processed_dir": PROCESSED_DIR,
                "results_dir": RESULTS_DIR,
                "run_results_dir": RUN_RESULTS_DIR,
            },
            "thresholds": {
                "expected_triplets": EXPECTED_TRIPLET_COUNT,
                "min_trials_to_save": MIN_TRIPLET_SAVE_THRESHOLD,
                "catch_accuracy_min": 0.90,
                "total_target_trials": TOTAL_TARGET_TRIALS,
            },
            "counts": {
                "files_scanned": len(summary),
                "participants_saved": participants_saved,
                "total_saved_triplets": total_saved_trials,
                "attention_failures": n_failed,
                "sona_accepted": sona_accepted,
                "sona_rejected": sona_rejected,
                "prolific_accepted": prol_accepted,
                "prolific_rejected": prol_rejected,
            }
        }
        manifest_path = os.path.join(RUN_RESULTS_DIR, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)
        print(f"[INFO] Manifest written to: {manifest_path}")

        # === Extract Sona survey IDs from uncredited participants AND match to processed files ===
        uncredited_path = os.path.join(SCRIPT_DIR, "SonaUncredited.csv")
        if os.path.exists(uncredited_path):
            try:
                # ---- Load SonaUncredited.csv robustly ----
                try:
                    uncredited_df = pd.read_csv(uncredited_path, encoding="utf-8-sig")
                except Exception:
                    uncredited_df = pd.read_csv(uncredited_path, encoding="utf-8-sig", sep=';')

                if uncredited_df.shape[1] == 1:
                    uncredited_df.columns = ["raw"]

                import re
                PARENS_NORMALIZE = str.maketrans({'（': '(', '）': ')', '［': '[', '］': ']', '【': '(', '】': ')'})
                P_PARENS = re.compile(r"\(([^)]*)\)")
                P_DIGITS = re.compile(r"\b(\d{5,})\b")  # 5+ digits

                def normalize_text(x):
                    if pd.isna(x):
                        return ''
                    s = str(x).strip().translate(PARENS_NORMALIZE)
                    s = re.sub(r"\s+", " ", s)
                    return s

                def extract_id_from_string(s: str):
                    # Prefer digits inside parentheses
                    for content in P_PARENS.findall(s):
                        m = P_DIGITS.search(content)
                        if m:
                            return m.group(1)
                    # Fallback: any 5+ digits anywhere
                    m = P_DIGITS.search(s)
                    if m:
                        return m.group(1)
                    return None

                # Normalize text and extract IDs per column, then flatten unique IDs
                uncredited_df = uncredited_df.applymap(normalize_text)
                id_cols = []
                for col in uncredited_df.columns:
                    newc = f"{col}_ID"
                    uncredited_df[newc] = uncredited_df[col].apply(extract_id_from_string)
                    id_cols.append(newc)
                all_ids = pd.unique(uncredited_df[id_cols].values.ravel('K'))
                all_ids = [x for x in all_ids if pd.notna(x) and str(x).strip() != '']

                # Diagnostics if none found
                if len(all_ids) == 0:
                    print("\n[WARN] No numeric IDs extracted from SonaUncredited.csv. Showing first few raw rows for debugging:")
                    for i, row in list(uncredited_df.head(5).iterrows()):
                        print(" - ", {c: row[c] for c in uncredited_df.columns if not c.endswith('_ID')})
                    print("[HINT] Ensure IDs appear as digits (e.g., '(123456)') or elsewhere in the row.")
                else:
                    print("\n=== Uncredited participant survey IDs (extracted) ===")
                    for sid in all_ids:
                        print(sid)
                    print(f"Total uncredited IDs: {len(all_ids)}")

                # ---- Match extracted IDs to processed summary (sonaCode) ----
                def normalize_id(s):
                    if s is None:
                        return None
                    s = str(s)
                    m = P_DIGITS.search(s)
                    return m.group(1) if m else None

                # Build an index from normalized sonaCode -> rows from summary
                summary_index = {}
                for fname, pool, n_catch, acc_str, n_saved, did_save, sona_code, n_triplets in summary:
                    nid = normalize_id(sona_code)
                    if nid:
                        # parse catch_acc as a percent if possible
                        catch_acc_pct = None
                        if isinstance(acc_str, str) and "%" in acc_str:
                            try:
                                catch_acc_pct = float(acc_str.split("%")[0])
                            except Exception:
                                catch_acc_pct = None
                        entry = {
                            "filename": fname,
                            "sonaCode": sona_code,
                            "trials_completed": int(n_triplets),
                            "catch_accuracy_str": acc_str,
                            "catch_accuracy_pct": catch_acc_pct,
                            "saved": bool(did_save),
                        }
                        summary_index.setdefault(nid, []).append(entry)

                # Collect matches for each uncredited ID
                matches = []
                unmatched = []
                for raw_id in all_ids:
                    nid = normalize_id(raw_id)
                    if nid and (nid in summary_index):
                        for entry in summary_index[nid]:
                            matches.append({
                                "survey_id": nid,
                                "trials_completed": entry["trials_completed"],
                                "catch_accuracy": entry["catch_accuracy_str"],
                                "saved": entry["saved"],
                                "filename": entry["filename"],
                            })
                    else:
                        unmatched.append(raw_id)

                print("\n=== Matches for Uncredited IDs (by sonaCode) ===")
                if matches:
                    print(f"Found {len(matches)} match(es):")
                    print(f"{'survey_id':<12} | {'trials_completed':<18} | {'catch_accuracy':<18} | {'saved':<5} | filename")
                    print("-" * 90)
                    for m in matches:
                        print(f"{m['survey_id']:<12} | {m['trials_completed']:<18} | {m['catch_accuracy']:<18} | {str(m['saved']):<5} | {m['filename']}")
                else:
                    print("No matches found in processed summary.")

                if unmatched:
                    print("\n[INFO] Uncredited IDs with no match in processed files:")
                    print(", ".join(unmatched))

            except Exception as e:
                print(f"[WARN] Could not process SonaUncredited.csv: {e}")
        else:
            print("[INFO] No SonaUncredited.csv file found in script directory.")