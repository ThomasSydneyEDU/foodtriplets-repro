"""compute_prelim_component.py

Stage 1 (Compute): Generate preliminary behavioral space components.

Takes the preprocessed summary tables and computes the empirical triplet RDM,
then optionally runs preliminary decompositions (PCA/MDS). Used internally
during data QC and before model evaluation.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import argparse
# --- Triplet -> Similarity/RDM (Hebart-style) ---------------------------------

PCA_VAR_CUTOFF = 0.80  # retain enough components to reach this cumulative explained variance

def build_pairwise_similarity_from_summary(labels: list[str], summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute empirical pairwise similarity p(i,j) and RDM (1-p) from triplets,
    plus a symmetric pairwise trial counts matrix.
    Uses filename columns: StimA_file, StimB_file, StimC_file, choice_file.
    For each triplet (A,B,C) with chosen odd-one-out X, we count the other two as the selected pair.
    p(i,j) is the proportion of times (i,j) was the selected pair among all trials where i and j co-occurred.
    Returns:
        sim_df: (n x n) similarity matrix with NaN for unobserved pairs
        rdm_df: (n x n) dissimilarity = 1 - sim (0 on diagonal)
        trials_df: (n x n) symmetric counts of observations per pair
    """
    req = ['StimA_file','StimB_file','StimC_file','choice_file']
    missing = [c for c in req if c not in summary_df.columns]
    if missing:
        raise KeyError(f"Summary is missing required columns: {missing}")

    n = len(labels)
    index_map = {lab: i for i, lab in enumerate(labels)}
    wins   = np.zeros((n, n), dtype=np.int32)
    trials = np.zeros((n, n), dtype=np.int32)

    for a, b, c, choice in summary_df[req].itertuples(index=False, name=None):
        a = basename_only(a); b = basename_only(b); c = basename_only(c); choice = basename_only(choice)
        ia = index_map.get(str(a)); ib = index_map.get(str(b)); ic = index_map.get(str(c))
        if ia is None or ib is None or ic is None:
            continue
        # count trials for unordered pairs
        for (i,j) in ((ia,ib),(ia,ic),(ib,ic)):
            if i == j:
                continue
            i0, j0 = (i,j) if i < j else (j,i)
            trials[i0, j0] += 1
        # selected pair = the two that are NOT the choice
        if str(choice) == str(a):
            i, j = ib, ic
        elif str(choice) == str(b):
            i, j = ia, ic
        elif str(choice) == str(c):
            i, j = ia, ib
        else:
            continue
        if i != j:
            i0, j0 = (i,j) if i < j else (j,i)
            wins[i0, j0] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        p_upper = np.where(trials > 0, wins / trials, np.nan)

    # mirror to full matrices; diag handling
    p = np.zeros((n, n), dtype=np.float32); p[:] = np.nan
    iu, ju = np.triu_indices(n, k=1)
    p[iu, ju] = p_upper[iu, ju]
    p[ju, iu] = p_upper[iu, ju]
    np.fill_diagonal(p, 1.0)

    rdm = 1.0 - p
    np.fill_diagonal(rdm, 0.0)

    # symmetric trials (count both directions)
    trials_sym = np.zeros((n, n), dtype=np.int32)
    trials_sym[iu, ju] = trials[iu, ju]
    trials_sym[ju, iu] = trials[iu, ju]
    np.fill_diagonal(trials_sym, 0)

    sim_df = pd.DataFrame(p, index=labels, columns=labels)
    rdm_df = pd.DataFrame(rdm, index=labels, columns=labels)
    trials_df = pd.DataFrame(trials_sym, index=labels, columns=labels)
    return sim_df, rdm_df, trials_df
import os
from datetime import datetime
import glob
import json
from pathlib import Path
from analysis_helpers import basename_only, save_dsm

try:
    from reference.policy_defaults import (
        INFO_CSV as POLICY_INFO_CSV,
        DERIVATIVES_DIR as POLICY_DERIVATIVES_DIR,
        RESULTS_DIR as POLICY_RESULTS_DIR,
        OUTPUTS_DIR as POLICY_OUTPUTS_DIR,
    )
except Exception:
    POLICY_INFO_CSV = None
    POLICY_DERIVATIVES_DIR = None
    POLICY_RESULTS_DIR = None
    POLICY_OUTPUTS_DIR = None

# Project directories
SCRIPT_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(SCRIPT_DIR, 'processed')  # preprocessed data (inputs)
DERIVATIVES_DIR = str(POLICY_DERIVATIVES_DIR) if POLICY_DERIVATIVES_DIR is not None else os.path.join(SCRIPT_DIR, 'derivatives')
RESULTS_DIR = str(POLICY_RESULTS_DIR) if POLICY_RESULTS_DIR is not None else os.path.join(SCRIPT_DIR, 'results')
OUTPUTS_DIR = str(POLICY_OUTPUTS_DIR) if POLICY_OUTPUTS_DIR is not None else os.path.join(SCRIPT_DIR, 'outputs')
for _p in (PROCESSED_DIR, DERIVATIVES_DIR, RESULTS_DIR, OUTPUTS_DIR):
    os.makedirs(_p, exist_ok=True)

# Timestamped run directories for results/outputs
RUN_ID = datetime.now().strftime('%Y-%m-%d_%H%M%S')
RUN_RESULTS_DIR = os.path.join(RESULTS_DIR, RUN_ID)
RUN_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, RUN_ID)
os.makedirs(RUN_RESULTS_DIR, exist_ok=True)
os.makedirs(RUN_OUTPUTS_DIR, exist_ok=True)



def parse_triplet(cell):
    if pd.isna(cell):
        return [None, None, None]
    s = str(cell).strip()
    # Fix doubled quotes from CSV quoting
    s = s.replace('""', '"')
    # If it looks like a JSON-like list, try to load; otherwise split on common delimiters
    trip = None
    if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
        s_json = s
        try:
            trip = json.loads(s_json)
        except Exception:
            # try to coerce single quotes to double quotes and retry
            try:
                trip = json.loads(s_json.replace("'", '"'))
            except Exception:
                trip = None
    if trip is None:
        # fallback: split by | ; or ,
        for delim in ['|', ';', ',']:
            if delim in s:
                trip = [p.strip() for p in s.strip('[]()').split(delim)]
                break
    if trip is None:
        trip = [s]
    trip = [basename_only(x) for x in trip]
    if len(trip) != 3:
        trip = (trip + [None, None, None])[:3]
    return trip

# Canonical stimulus info CSV (from policy when available)
reference_dir = os.path.join(os.path.dirname(__file__), 'reference')
food_info_path = str(POLICY_INFO_CSV) if POLICY_INFO_CSV is not None else os.path.join(reference_dir, 'Foodpictures_information_dynamic.csv')

# Load the canonical stimulus info table.
# We keep values as strings and avoid interpreting tokens like "NA" as missing.
food_info_df = pd.read_csv(
    food_info_path,
    keep_default_na=False,
    na_values=[],
    dtype=str,
)
food_info_df.columns = food_info_df.columns.str.strip()
for _c in food_info_df.columns:
    if food_info_df[_c].dtype == object:
        food_info_df[_c] = food_info_df[_c].str.strip()

# Canonical schema expected from Foodpictures_information_canonical.csv:
#   filename, food, natural_vs_transformed
# For backward compatibility, allow the old names if they are still present.
rename_map = {}
if 'Description' not in food_info_df.columns and 'food' in food_info_df.columns:
    rename_map['food'] = 'Description'
if 'Natural vs transformed' not in food_info_df.columns and 'natural_vs_transformed' in food_info_df.columns:
    rename_map['natural_vs_transformed'] = 'Natural vs transformed'
if rename_map:
    food_info_df = food_info_df.rename(columns=rename_map)

# Minimal legacy fallback for older files.
if 'filename' not in food_info_df.columns and 'image_file' in food_info_df.columns:
    food_info_df = food_info_df.rename(columns={'image_file': 'filename'})
if 'Description' not in food_info_df.columns and 'base_food' in food_info_df.columns:
    food_info_df = food_info_df.rename(columns={'base_food': 'Description'})
if 'Natural vs transformed' not in food_info_df.columns and 'Natural_vs_transformed' in food_info_df.columns:
    food_info_df = food_info_df.rename(columns={'Natural_vs_transformed': 'Natural vs transformed'})

# Required columns for this script after normalization.
required_info_cols = ['filename', 'Description']
missing_info = [c for c in required_info_cols if c not in food_info_df.columns]
if missing_info:
    raise KeyError(
        f"Information table is missing required columns {missing_info}. "
        f"Available columns: {list(food_info_df.columns)}"
    )

# Normalize filenames to basenames for matching against processed data.
food_info_df['filename'] = food_info_df['filename'].astype(str).apply(os.path.basename).str.strip()
food_info_df['Description'] = food_info_df['Description'].astype(str).str.strip()

print(f"Loaded food information data with shape: {food_info_df.shape}")
print(f"Using food info file: {food_info_path}")

def _to_list_from_images_onscreen(images_onscreen_str):
    """
    Converts the images_onscreen string representation to a list of image IDs.
    Assumes images_onscreen_str is a string like "['img1', 'img2', 'img3']".
    """
    # Strip brackets and split by comma
    images_onscreen_str = images_onscreen_str.strip()
    if images_onscreen_str.startswith('[') and images_onscreen_str.endswith(']'):
        images_onscreen_str = images_onscreen_str[1:-1]
    # Split by comma and strip quotes and whitespace
    images = [img.strip().strip("'").strip('"') for img in images_onscreen_str.split(',')]
    return images

def _choice_to_index(choice_image, images_onscreen):
    """
    Given a choice_image and the list of images_onscreen, returns the index of the chosen image.
    Returns None if choice_image is not found.
    """
    try:
        return images_onscreen.index(choice_image)
    except ValueError:
        return None

def build_triplet_dataframe(df):
    """
    Given a DataFrame with columns 'images_onscreen' and 'choice_image',
    returns a new DataFrame with columns ['anchor', 'positive', 'negative'] representing triplets.
    The positive is the chosen image, and the other two images are negatives.
    """
    triplets = []
    for _, row in df.iterrows():
        images = _to_list_from_images_onscreen(row['images_onscreen'])
        choice = row['choice_image']
        choice_idx = _choice_to_index(choice, images)
        if choice_idx is None:
            # Skip rows where choice_image not in images_onscreen
            continue
        # Positive image is the chosen one
        positive = choice
        # Anchor and negative are the other two images
        others = [img for idx, img in enumerate(images) if idx != choice_idx]
        if len(others) != 2:
            # Skip if not exactly two other images (should be triplets)
            continue
        anchor, negative = others
        triplets.append({'anchor': anchor, 'positive': positive, 'negative': negative})
    triplet_df = pd.DataFrame(triplets)
    return triplet_df


def concatenate_processed_rt_images_choices(processed_dir: str) -> pd.DataFrame:
    """
    Concatenate all .csv files in processed_dir, keeping a clean minimal set of triplet columns + metadata, including is_fixed/is_dense/is_random, stimulus_source, time_elapsed.
    """
    import logging
    logging.basicConfig(level=logging.WARNING)
    csv_files = sorted(glob.glob(os.path.join(processed_dir, "*.csv")))
    required_cols = [
        'rt', 'images_onscreen', 'choice_image',
        'is_fixed', 'is_dense', 'is_random',
        'stimulus_source', 'time_elapsed',
        'pool', 'prolificPID', 'sonaCode'
    ]
    dfs = []
    for idx, fpath in enumerate(csv_files):
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            logging.warning(f"Failed to read {fpath}: {e}")
            continue
        # Require at least the core triplet fields; metadata may be missing in older processed files
        core_required = ['rt', 'images_onscreen', 'choice_image']
        if not all(c in df.columns for c in core_required):
            logging.warning(f"Skipping {fpath}: missing one of core columns {core_required}.")
            continue

        # Keep the required columns that exist
        df_sub = df[[c for c in required_cols if c in df.columns]].copy()
        # Add missing optional columns as empty
        for c in required_cols:
            if c not in df_sub.columns:
                df_sub[c] = pd.NA
        # Reorder columns (ordered)
        df_sub = df_sub[required_cols]
        # Assign a stable SubID (prefer Prolific PID, then SONA code; fallback to file-based ID)
        subid = None
        if 'prolificPID' in df.columns:
            s = df['prolificPID'].dropna().astype(str).str.strip()
            if not s.empty and (s.iloc[-1] not in {'', 'nan', 'None'}):
                subid = s.iloc[-1]
        if subid is None and 'sonaCode' in df.columns:
            s = df['sonaCode'].dropna().astype(str).str.strip()
            if not s.empty and (s.iloc[-1] not in {'', 'nan', 'None'}):
                subid = f"SONA_{s.iloc[-1]}"
        if subid is None:
            subid = os.path.splitext(os.path.basename(fpath))[0]
        df_sub['SubID'] = subid
        dfs.append(df_sub)
    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
    else:
        all_df = pd.DataFrame(columns=required_cols + ['SubID'])
    return all_df

def clean_and_structure_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Create StimA/StimB/StimC from images_onscreen (basename only)
    stim_cols = df['images_onscreen'].apply(parse_triplet).apply(pd.Series)
    stim_cols.columns = ['StimA_file', 'StimB_file', 'StimC_file']
    # Ensure basenames
    for c in ['StimA_file','StimB_file','StimC_file']:
        stim_cols[c] = stim_cols[c].apply(basename_only)
    # choice_image -> basename only (as filename column)
    if 'choice_image' in df.columns:
        df['choice_file'] = df['choice_image'].apply(basename_only)
    else:
        df['choice_file'] = pd.NA
    # Build initial output with filename columns
    out = pd.concat([df, stim_cols], axis=1)

    # Also provide description-mapped columns (human readable) using the global food_info_df
    if {'filename','Description'}.issubset(set(food_info_df.columns)):
        desc_map = dict(zip(food_info_df['filename'].astype(str), food_info_df['Description'].astype(str)))
        out['StimA'] = out['StimA_file'].apply(lambda x: desc_map.get(str(x), x))
        out['StimB'] = out['StimB_file'].apply(lambda x: desc_map.get(str(x), x))
        out['StimC'] = out['StimC_file'].apply(lambda x: desc_map.get(str(x), x))
        out['choice_image'] = out['choice_file'].apply(lambda x: desc_map.get(str(x), x))
    else:
        out['StimA'] = out['StimA_file']
        out['StimB'] = out['StimB_file']
        out['StimC'] = out['StimC_file']
        out['choice_image'] = out['choice_file']

    # Column order
    cols_keep = []
    # Core behavioural fields
    if 'rt' in out.columns: cols_keep.append('rt')
    if 'time_elapsed' in out.columns: cols_keep.append('time_elapsed')
    if 'stimulus_source' in out.columns: cols_keep.append('stimulus_source')

    # Trial-type flags
    for c in ['is_fixed', 'is_dense', 'is_random']:
        if c in out.columns:
            cols_keep.append(c)

    # Optional context
    for c in ['pool', 'prolificPID', 'sonaCode']:
        if c in out.columns:
            cols_keep.append(c)

    # Triplet filename columns + description-mapped columns
    cols_keep += ['StimA_file','StimB_file','StimC_file','choice_file', 'StimA','StimB','StimC','choice_image']

    # Participant ID
    if 'SubID' in out.columns:
        cols_keep.append('SubID')

    out = out[cols_keep]
    return out


# Helper function to build filename→Description mapping robust to paths and missing extensions
def _build_foodname_mapping(info_df: pd.DataFrame):
    # Normalize filenames to basenames
    fnames = info_df['filename'].astype(str).apply(lambda p: os.path.basename(p))
    descs = info_df['Description'].astype(str)
    mapping_ext = dict(zip(fnames, descs))
    # Also build a stem-based mapping (lowercased) to catch values without extensions
    stems = fnames.apply(lambda x: os.path.splitext(x)[0].lower())
    mapping_stem = dict(zip(stems, descs))
    return mapping_ext, mapping_stem

# Helper function to replace filenames in the summary with food names
def replace_filenames_with_foodnames(df: pd.DataFrame, info_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping_ext, mapping_stem = _build_foodname_mapping(info_df)
    target_cols = ['StimA', 'StimB', 'StimC', 'choice_image']
    def _convert(val):
        if pd.isna(val):
            return val
        base = os.path.basename(str(val))
        if base in mapping_ext:
            return mapping_ext[base]
        stem = os.path.splitext(base)[0].lower()
        return mapping_stem.get(stem, val)
    for c in target_cols:
        if c in df.columns:
            df[c] = df[c].apply(_convert)
    return df

def crosscheck_stimA_vs_info(food_info_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    # Compare labels used in the triplet summary (human-readable columns) against the info table Description
    cols = [c for c in ['StimA', 'StimB', 'StimC', 'choice_image'] if c in summary_df.columns]
    if not cols:
        print("=== Cross-check: Summary vs Information Description ===")
        print("[WARN] No stimulus name columns (StimA/StimB/StimC/choice_image) found in summary.")
        return

    used_vals = pd.unique(pd.concat([summary_df[c] for c in cols], ignore_index=True).dropna())
    used_set = set(map(str, used_vals))
    info_desc_vals = set(map(str, food_info_df['Description'].dropna().unique()))

    missing_in_info = sorted(used_set - info_desc_vals)
    extras_in_info = sorted(info_desc_vals - used_set)

    print("=== Cross-check: Summary stimulus labels vs Information Description ===")
    print(f"Unique labels in summary (across {', '.join(cols)}): {len(used_set)}")
    print(f"Unique Description (info): {len(info_desc_vals)}")
    print(f"Summary labels NOT in info Description: {len(missing_in_info)}")
    print(f"Info Descriptions NOT used in summary: {len(extras_in_info)}")

    if missing_in_info:
        print("Examples (summary labels not in info, up to 20):")
        for item in missing_in_info[:20]:
            print(f" - {item}")

    if extras_in_info:
        print("Examples (info descriptions not used in summary, up to 20):")
        for item in extras_in_info[:20]:
            print(f" - {item}")


def report_info_not_in_triplets(food_info_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: str) -> None:
    """Report rows in the information dataframe that are not present in the triplet data.
    Uses Description-based matching (since summary has been mapped to food names).
    Saves a CSV of unused info rows in out_dir.
    """
    # All labels used anywhere in the summary (after mapping to food names)
    used_labels = pd.unique(pd.concat([
        summary_df.get('StimA', pd.Series(dtype=object)),
        summary_df.get('StimB', pd.Series(dtype=object)),
        summary_df.get('StimC', pd.Series(dtype=object)),
        summary_df.get('choice_image', pd.Series(dtype=object)),
    ], ignore_index=True).dropna())
    used_set = set(used_labels)

    # Info descriptions
    info_desc = food_info_df['Description'].astype(str)
    in_triplets_mask = info_desc.isin(used_set)

    # Rows not present in any triplet column (by Description)
    unused_info_rows = food_info_df.loc[~in_triplets_mask].copy()

    print("=== Info rows NOT used in triplet data (by Description) ===")
    print(f"Total info rows: {len(food_info_df)}")
    print(f"Used in triplets: {in_triplets_mask.sum()}")
    print(f"Not used in triplets: {len(unused_info_rows)}")
    if not unused_info_rows.empty:
        # Show a few examples
        preview_cols = [c for c in ['filename','Description','Natural vs transformed'] if c in unused_info_rows.columns]
        print("Examples (up to 20):")
        for _, r in unused_info_rows.head(20)[preview_cols].iterrows():
            print(" - ", {k: r[k] for k in preview_cols})
        # Save CSV
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, 'info_rows_not_in_triplets.csv')
        unused_info_rows.to_csv(out_csv, index=False)
        print(f"Saved unused info rows to: {out_csv}")
    else:
        print("All info descriptions are represented in the triplet data.")


# === Duplicate Description report ===
def report_duplicate_descriptions(food_info_df: pd.DataFrame, out_dir: str) -> None:
    """Find duplicated Description entries in the information dataframe and save details.
    Outputs a CSV with per-Description counts, filenames, and category values to help resolve duplicates.
    """
    def _unique_join(series: pd.Series) -> str:
        vals = [str(x) for x in series.dropna().unique()]
        # stable but compact ordering
        return "; ".join(sorted(vals))

    cols = list(food_info_df.columns)
    agg_spec = {
        'Description': ('Description', 'size'),
        'filename': ('filename', _unique_join),
    }
    # Include category columns if present
    if 'Natural vs transformed' in cols:
        agg_spec['Natural vs transformed'] = ('Natural vs transformed', _unique_join)

    grouped = food_info_df.groupby('Description', dropna=False).agg(**agg_spec)
    grouped = grouped.rename(columns={'Description': 'count', 'filename': 'filenames'})
    dups = grouped[grouped['count'] > 1].reset_index().sort_values(['count', 'Description'], ascending=[False, True])

    print("=== Duplicate Descriptions in information dataframe ===")
    print(f"Total rows: {len(food_info_df)}  |  Unique descriptions: {food_info_df['Description'].nunique()}  |  Duplicates: {len(dups)}")
    if dups.empty:
        print("No duplicate Description entries found.")
        return

    # Show a brief preview in console
    preview_cols = [c for c in ['Description', 'count', 'filenames', 'Natural vs transformed'] if c in dups.columns]
    print("Examples (up to 20):")
    for _, row in dups.head(20)[preview_cols].iterrows():
        row_dict = {k: row[k] for k in preview_cols}
        print(" - ", row_dict)

    # Save full details
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'info_duplicate_descriptions.csv')
    dups.to_csv(out_csv, index=False)
    print(f"Saved duplicate description report to: {out_csv}")

import numpy as np
from scipy.spatial.distance import squareform

def _get_col_exact(df: pd.DataFrame, target: str):
    # Exact match helper with a clear error if missing
    if target not in df.columns:
        raise KeyError(f"Column '{target}' not found in information dataframe. Available: {list(df.columns)}")
    return target

def _category_order(series: pd.Series) -> list:
    # Preserve the order of first appearance, dropna
    seen = []
    for v in series.dropna().tolist():
        if v not in seen:
            seen.append(v)
    return seen


def make_sorted_label_order(info_df: pd.DataFrame, used_labels: list[str]) -> list[str]:
    # Columns (exact names as requested)
    col_nat = _get_col_exact(info_df, 'Natural vs transformed')

    # Work only with rows used in the current summary
    sub = info_df[info_df['Description'].isin(used_labels)].copy()

    # Warn about labels present in summary but not in info
    missing_desc = sorted(set(used_labels) - set(sub['Description']))
    if missing_desc:
        print(f"[WARN] {len(missing_desc)} stimuli in summary not found in information table. They will be appended alphabetically at the end.")
        for m in missing_desc[:20]:
            print(f"  - {m}")

    # Warnings for stimuli missing category values
    missing_nat = sub[sub[col_nat].isna()]['Description'].tolist()
    if missing_nat:
        print(f"[WARN] {len(missing_nat)} stimuli have no value in '{col_nat}'. Examples:")
        for m in missing_nat[:20]:
            print(f"  - {m}")

    # Category orders as they appear in the info table
    nat_order = _category_order(sub[col_nat])
    if not nat_order:
        print(f"[WARN] No non-null values found in '{col_nat}'. Falling back to alphabetical only.")

    # Build categoricals for sorting
    if nat_order:
        sub['__nat'] = pd.Categorical(sub[col_nat], categories=nat_order, ordered=True)
    else:
        sub['__nat'] = sub[col_nat]

    sub = sub.sort_values(['__nat', 'Description'], kind='mergesort')
    ordered = sub['Description'].tolist()

    # Append any labels missing from info at the end (alphabetical)
    if missing_desc:
        ordered += sorted(missing_desc)

    return ordered


# --- Filename-based label builder ------------------------------------------
def build_filename_label_list(summary_df: pd.DataFrame, food_info_df: pd.DataFrame) -> list[str]:
    """Create an ordered list of unique filename labels for the DSM.
    Order: Natural vs transformed (first-seen), then alphabetical by filename.
    Warn if a filename lacks either category in the info table.
    """
    # Unique filenames used anywhere in triplets
    used_files = pd.unique(pd.concat([
        summary_df.get('StimA_file', pd.Series(dtype=object)),
        summary_df.get('StimB_file', pd.Series(dtype=object)),
        summary_df.get('StimC_file', pd.Series(dtype=object))
    ], ignore_index=True).dropna())

    used_df = pd.DataFrame({'filename': used_files})

    info_cols = ['filename', 'Description']
    if 'Natural vs transformed' in food_info_df.columns:
        info_cols.append('Natural vs transformed')

    merged = used_df.merge(food_info_df[info_cols], on='filename', how='left')

    # Warnings for missing category values
    for col in ['Natural vs transformed']:
        if col in merged.columns:
            miss = merged[col].isna()
            if miss.any():
                print(f"[WARN] {miss.sum()} stimuli (by filename) have no value in '{col}'. Examples:")
                ex = merged.loc[miss, ['filename','Description']].head(5)
                for _, r in ex.iterrows():
                    print(f"  - {r['filename']}  ({r.get('Description','')})")

    # First-seen orders
    def first_seen(series: pd.Series) -> list:
        seen = []
        for v in series.dropna():
            if v not in seen:
                seen.append(v)
        return seen

    nat_order   = first_seen(merged.get('Natural vs transformed', pd.Series(dtype=object)))

    # Build helper index columns for deterministic sorting
    def idx_or_last(val, order_list):
        try:
            return order_list.index(val)
        except ValueError:
            return len(order_list)
        except Exception:
            return len(order_list)

    merged['__nat_idx'] = merged.get('Natural vs transformed', pd.Series([None]*len(merged))).apply(lambda v: idx_or_last(v, nat_order))

    # Now sort explicitly by the helper indices then filename (stable mergesort)
    merged_sorted = merged.sort_values(by=['__nat_idx','filename'], kind='mergesort')

    return merged_sorted['filename'].tolist()




#
# --- Triplet extraction helpers -------------------------------------------------

def extract_triplets(summary_df: pd.DataFrame, mode: str = 'filename') -> list[tuple[str,str,str,str]]:
    """Return list of (a, b, c, choice) using either filename or description columns.
    mode='filename' uses StimA_file/StimB_file/StimC_file/choice_file
    mode='description' uses StimA/StimB/StimC/choice_image
    """
    if mode == 'filename':
        cols = ['StimA_file','StimB_file','StimC_file','choice_file']
    elif mode == 'description':
        cols = ['StimA','StimB','StimC','choice_image']
    else:
        raise ValueError("mode must be 'filename' or 'description'")

    missing = [c for c in cols if c not in summary_df.columns]
    if missing:
        raise KeyError(f"extract_triplets: missing columns {missing}")

    df = summary_df[cols].dropna()
    df = df[(df[cols[0]] != '') & (df[cols[1]] != '') & (df[cols[2]] != '') & (df[cols[3]] != '')]
    return list(df.itertuples(index=False, name=None))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build summary.csv + empirical triplet RDM and supporting artifacts")
    p.add_argument(
        "--full",
        action="store_true",
        help="Run full legacy outputs (DSM placeholders + PCA component orderings). Default is lightweight.",
    )
    p.add_argument(
        "--from-summary",
        default=None,
        help="Path to a ready-made summary-schema CSV (see scripts/make_summary_from_release.py). "
             "Skips the per-participant processed/ concatenation step.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.from_summary:
        summary_df = pd.read_csv(args.from_summary, low_memory=False)
    else:
        all_df = concatenate_processed_rt_images_choices(PROCESSED_DIR)
        summary_df = clean_and_structure_summary(all_df)
    summary_df = replace_filenames_with_foodnames(summary_df, food_info_df)

    # Enforce canonical column order for summary
    summary_cols = [
        'rt','time_elapsed','stimulus_source',
        'is_fixed','is_dense','is_random',
        'pool','prolificPID','sonaCode',
        'StimA_file','StimB_file','StimC_file','choice_file',
        'StimA','StimB','StimC','choice_image',
        'SubID'
    ]
    summary_df = summary_df.reindex(columns=[c for c in summary_cols if c in summary_df.columns])

    save_path = os.path.join(DERIVATIVES_DIR, 'summary.csv')
    summary_df.to_csv(save_path, index=False)
    print(f"Summary saved to: {save_path}")
    print(f"Shape: {summary_df.shape}")
    crosscheck_stimA_vs_info(food_info_df, summary_df)

    # Reports directory for info/triplet reports
    reports_dir = os.path.join(DERIVATIVES_DIR, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    report_info_not_in_triplets(food_info_df, summary_df, reports_dir)
    report_duplicate_descriptions(food_info_df, reports_dir)

    # === Build ordered filename label list (canonical ordering) ===
    labels = build_filename_label_list(summary_df, food_info_df)

    saved_paths: list[str] = []
    if args.full:
        # Legacy: Create an empty DSM placeholder (float32) with that order; fill zeros for now
        n = len(labels)
        dsm_df = pd.DataFrame(np.zeros((n, n), dtype=np.float32), index=labels, columns=labels)

        # Save DSM (filename labels). Parquet disabled by default to avoid engine dependency.
        dsm_prefix = os.path.join(DERIVATIVES_DIR, 'food_dsm_filenames')
        saved_paths = save_dsm(dsm_df, dsm_prefix, enable_parquet=False)
        print(f"DSM placeholder built with filename labels: {n}×{n}")
        print("DSM placeholders saved:", ", ".join(saved_paths))

    # Save a label lookup for plots/reports (filename -> Description), robust to extension/case
    lookup_csv = os.path.join(DERIVATIVES_DIR, 'labels_lookup.csv')
    labels_lookup = pd.DataFrame({'filename': labels})
    if {'filename','Description'}.issubset(food_info_df.columns):
        # Build robust maps
        _fn_series = food_info_df['filename'].astype(str).apply(lambda p: os.path.basename(p))
        _desc_series = food_info_df['Description'].astype(str)
        _map_ext = dict(zip(_fn_series, _desc_series))
        _map_stem = dict(zip(_fn_series.str.replace(r"\.[^.]*$","", regex=True).str.lower(), _desc_series))
        def _lookup_desc(fn):
            base = os.path.basename(str(fn))
            if base in _map_ext:
                return _map_ext[base]
            stem = base.rsplit('.', 1)[0].lower()
            return _map_stem.get(stem, '')
        labels_lookup['Description'] = [ _lookup_desc(fn) for fn in labels ]
    labels_lookup.to_csv(lookup_csv, index=False)
    print(f"Saved label lookup to: {lookup_csv}")

    # === Build triplet-based similarity and RDM (Hebart-style) ===
    sim_df, rdm_df, trials_df = build_pairwise_similarity_from_summary(labels, summary_df)

    # Report observed pair coverage from raw counts
    iu, ju = np.triu_indices(len(labels), k=1)
    cnt_vec = trials_df.to_numpy()[iu, ju]
    seen_pairs = int(np.count_nonzero(cnt_vec > 0))
    total_pairs = int(len(iu))
    print(f"Observed pairs (from summary counts): {seen_pairs} / {total_pairs} (~{seen_pairs/total_pairs:0.1%})")

    # Save CSV (NaNs preserved) and NPZ (fill NaNs with 0.5 in arrays)
    sim_path_csv = os.path.join(DERIVATIVES_DIR, 'triplet_similarity_by_filename.csv.gz')
    rdm_path_csv = os.path.join(DERIVATIVES_DIR, 'triplet_rdm_by_filename.csv.gz')
    sim_df.to_csv(sim_path_csv, index=True)
    rdm_df.to_csv(rdm_path_csv, index=True)

    # Save pairwise counts and observed mask
    counts_csv = os.path.join(DERIVATIVES_DIR, 'triplet_pair_counts_by_filename.csv.gz')
    trials_df.to_csv(counts_csv, index=True)
    observed_mask = (trials_df.to_numpy(dtype=np.int32) > 0)

    np.savez_compressed(os.path.join(DERIVATIVES_DIR, 'triplet_pair_counts_by_filename.npz'),
                        counts=trials_df.to_numpy(dtype=np.int32), labels=np.array(labels, dtype=object))
    np.savez_compressed(os.path.join(DERIVATIVES_DIR, 'triplet_observed_mask_by_filename.npz'),
                        observed=observed_mask.astype(np.uint8), labels=np.array(labels, dtype=object))

    # Force writable arrays (some pandas/numpy builds return read-only views)
    sim_np = sim_df.to_numpy(dtype=np.float32, copy=True)
    rdm_np = rdm_df.to_numpy(dtype=np.float32, copy=True)
    if not sim_np.flags.writeable:
        sim_np = np.array(sim_np, copy=True)
    if not rdm_np.flags.writeable:
        rdm_np = np.array(rdm_np, copy=True)
    # Midpoint padding only for truly unobserved pairs
    unobserved = ~observed_mask
    # keep diagonal intact
    np.fill_diagonal(unobserved, False)
    n_midpad = int(unobserved.sum())
    if n_midpad > 0:
        sim_np[unobserved] = 0.5
        rdm_np = 1.0 - sim_np
        np.fill_diagonal(rdm_np, 0.0)
        n_midpad_unique = int(n_midpad // 2)
        print(f"[NOTE] Applied midpoint padding (0.5) to {n_midpad} unobserved matrix cells (both triangles) ~= {n_midpad_unique} unique pairs.")

    np.savez_compressed(os.path.join(DERIVATIVES_DIR, 'triplet_similarity_by_filename.npz'),
                        data=sim_np, labels=np.array(labels, dtype=object))
    np.savez_compressed(os.path.join(DERIVATIVES_DIR, 'triplet_rdm_by_filename.npz'),
                        data=rdm_np, labels=np.array(labels, dtype=object))

    print("Saved triplet similarity/RDM to derivatives/:",
          "triplet_similarity_by_filename.csv.gz, .npz;",
          "triplet_rdm_by_filename.csv.gz, .npz")

    # === PCA summarization of behavioural similarity (legacy; optional) ===
    if args.full:
        try:
            X = sim_np.astype(np.float64)
            # Center columns (features)
            Xc = X - X.mean(axis=0, keepdims=True)
            # SVD: Xc = U S V^T; rows of U*S give component scores per stimulus
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

            # Explained variance ratios from singular values
            S2 = (S ** 2)
            total_var = S2.sum()
            evr = S2 / (total_var if total_var > 0 else 1.0)
            cum_evr = np.cumsum(evr)

            # How many components to retain based on cutoff
            K = int(np.searchsorted(cum_evr, PCA_VAR_CUTOFF) + 1)
            K = max(1, min(K, U.shape[1]))

            # Component scores (n x K)
            scores = (U * S)[:, :K]

            # Save explained variance table
            pca_var_df = pd.DataFrame({
                'component': np.arange(1, len(evr) + 1, dtype=int),
                'explained_variance_ratio': evr,
                'cumulative_explained_variance': cum_evr
            })
            pca_var_csv = os.path.join(DERIVATIVES_DIR, 'pca_variance.csv')
            pca_var_df.to_csv(pca_var_csv, index=False)
            print(f"Saved PCA variance table to: {pca_var_csv} (K={K} components retained for {PCA_VAR_CUTOFF:.0%} cumulative EV)")

            # Create ordered name lists for each retained component
            comp_order_paths = []
            if 'Description' in labels_lookup.columns:
                _desc_map_direct = dict(zip(labels_lookup['filename'].astype(str), labels_lookup['Description'].astype(str)))
            else:
                _desc_map_direct = {}
            if {'filename','Description'}.issubset(food_info_df.columns):
                _fn_series = food_info_df['filename'].astype(str).apply(lambda p: os.path.basename(p))
                _desc_series = food_info_df['Description'].astype(str)
                _map_ext = dict(zip(_fn_series, _desc_series))
                _map_stem = dict(zip(_fn_series.str.replace(r"\.[^.]*$","", regex=True).str.lower(), _desc_series))
            else:
                _map_ext, _map_stem = {}, {}
            def _pretty_name(fn):
                base = os.path.basename(str(fn))
                name = _desc_map_direct.get(base, None)
                if name is None or str(name).strip() == '' or str(name).lower() == 'nan':
                    if base in _map_ext:
                        name = _map_ext[base]
                    else:
                        stem = base.rsplit('.', 1)[0].lower()
                        name = _map_stem.get(stem, base)
                return name if (name is not None and str(name).strip() != '' and str(name).lower() != 'nan') else base

            for k in range(K):
                comp = scores[:, k]
                order_idx = np.argsort(comp)
                ordered = [labels[i] for i in order_idx]
                ordered_names = [_pretty_name(fn) for fn in ordered]
                comp_path = os.path.join(RUN_RESULTS_DIR, f'component{k+1}.txt')
                with open(comp_path, 'w', encoding='utf-8') as f:
                    for name in ordered_names:
                        f.write(f"{name}\n")
                comp_order_paths.append(comp_path)
            print("Saved PCA component orderings to:", ", ".join(comp_order_paths))

            # Save PCA scores for all retained components
            cols = {f'comp{k+1}': scores[:, k].astype(float) for k in range(K)}
            cols['filename'] = labels
            pca_scores_df = pd.DataFrame(cols)
            # put filename first
            pca_scores_df = pca_scores_df[['filename'] + [c for c in pca_scores_df.columns if c != 'filename']]
            pca_scores_csv = os.path.join(DERIVATIVES_DIR, 'pca_scores.csv')
            pca_scores_df.to_csv(pca_scores_csv, index=False)
            print(f"Saved PCA scores (K={K}) to: {pca_scores_csv}")
        except Exception as e:
            print(f"[WARN] PCA summarization failed: {e}")
    else:
        print("[INFO] Lightweight mode: skipping PCA summarization (use --full to enable).")

    # Write manifest for this setup step
    manifest = {
        'run_id': RUN_ID,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'inputs': {
            'processed_dir': PROCESSED_DIR
        },
        'derivatives_written': ([os.path.basename(p) for p in saved_paths] if saved_paths else []) + [
            os.path.basename(sim_path_csv),
            os.path.basename(rdm_path_csv),
            'triplet_similarity_by_filename.npz',
            'triplet_rdm_by_filename.npz',
            os.path.basename(save_path),
            'triplet_pair_counts_by_filename.csv.gz',
            'triplet_pair_counts_by_filename.npz',
            'triplet_observed_mask_by_filename.npz'
        ]
    }
    with open(os.path.join(RUN_RESULTS_DIR, 'setup_manifest.json'), 'w', encoding='utf-8') as mf:
        json.dump(manifest, mf, indent=2)
    print(f"Setup manifest written to: {os.path.join(RUN_RESULTS_DIR, 'setup_manifest.json')}")
