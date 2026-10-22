#!/usr/bin/env python3
"""Convert the anonymised Zenodo trial file into the summary.csv schema used by analysis_code.

Usage: make_summary_from_release.py RELEASE_CSV OUT_SUMMARY_CSV
Participant IDs and row order in the release are shuffled, so analyses that depend on
participant order (SPoSE epochs, noise-ceiling splits) reproduce approximately.
"""
import sys
import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src)
need = {"participant","pool","session","trial","sampling","item_A","item_B","item_C","odd_one_out","rt_ms"}
assert need <= set(d.columns), f"missing columns: {need - set(d.columns)}"
s = pd.DataFrame({
    "rt": d.rt_ms.astype(float),
    "time_elapsed": np.nan,
    "stimulus_source": "Filtered_ImageList.json",
    "is_fixed": d.sampling.eq("fixed"),
    "is_dense": d.sampling.eq("dense"),
    "is_random": d.sampling.eq("random"),
    "pool": d.pool,
    "prolificPID": np.nan,
    "sonaCode": np.nan,
    "StimA_file": d.item_A + ".png",
    "StimB_file": d.item_B + ".png",
    "StimC_file": d.item_C + ".png",
    "choice_file": d.odd_one_out + ".png",
    "StimA": d.item_A, "StimB": d.item_B, "StimC": d.item_C,
    "choice_image": d.odd_one_out,
    "SubID": d.participant,
})
s.to_csv(out, index=False)
print(f"wrote {out}: {len(s)} trials, {s.SubID.nunique()} participants")
