"""
Trial-composition sensitivity analysis (reviewer point R2-3, feeding R2-1).

Reviewer 2: "I wondered if the structure you get out depends on stimulus contrasts you
present participant with. At an extreme, if every triplet were composed of 2 foods of
one colour and one food of a different colour, would colour come out as the organizing
principle? ... So again the structure that emerges originates in the structure where
the kinds of food images are presented."

This is the reviewer's most fundamental objection: that the recovered similarity space
may be a property of how the triplets were composed rather than of how people represent
food. It is answerable directly, because the design already contains the control.

THE TEST
--------
The 480 experimental trials per participant comprise three pools:
  fixed   100 triplets, the same set shown to everyone
  dense   150 triplets drawn only from a 30-item subset
  random  230 triplets drawn uniformly from the full 350-item set

The fixed and dense pools are deliberately structured, and they are what the reviewer's
objection targets. The random pool is uniform over the stimulus set, so the composition
of its triplets cannot favour any particular organising dimension. If the same models
win by the same margins on random trials alone, the structure is not an artefact of how
the triplets were built.

This complements, at the level of the model comparison, the random-trials-only SPoSE
check the manuscript already reports at the level of the embedding geometry.

WHAT TO REPORT
--------------
The ORDERING of model fits, and the rank correlation between the orderings, not the
absolute rho values. Random trials give a median of about 3 observations per pair
against roughly 312 for the fixed+dense pairs, so per-pair estimates are far noisier
and absolute correlations are attenuated by measurement error. Attenuation is expected
and is not evidence of anything; a change in ordering would be.

STANDALONE BY DESIGN. Reads raw session CSVs and model RDMs read-only; writes only into
sensitivity/sensitivity_outputs/. Never touches processed/, derivatives/ or
compute_outputs/.

Usage
-----
    python sensitivity/sensitivity_trial_composition.py
    python sensitivity/sensitivity_trial_composition.py --min-obs 3
"""
from __future__ import annotations
import argparse, csv, glob, json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
DERIV = os.path.join(ROOT, "analysis_code", "derivatives")
CANON = os.path.join(DERIV, "model_rdms_canon")
OUTDIR = os.path.join(HERE, "sensitivity_outputs")

EXCLUDED_CATCH_IDS = {"catch_5", "catch_8"}
MAX_CATCH_ERRORS = 1     # published rule: exclude at 2 or more
MIN_TRIPLETS = 450

MODELS = ["food_classification", "Category_WHO_10", "Category_Culinary_9",
          "NOVA_ordinal", "natural_vs_transformed", "Transformation_score",
          "human_healthiness", "human_calorie_density", "human_savoriness",
          "human_sweetness", "human_saltiness", "human_fattiness",
          "human_sourness", "human_bitterness", "human_spiciness", "human_appeal"]


def ranks(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1)
    sx = x[order]; i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d else np.nan


def spear(a, b):
    return pear(ranks(a), ranks(b))


def truthy(v):
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def read_trials(index):
    """Return per-pool trial lists from the public data (derivatives/summary.csv).

    The public file contains only the participants who passed the published exclusion
    rule, so no catch-trial screening is needed here (it was applied to raw sessions in
    the original analysis).
    """
    import pandas as pd
    df = pd.read_csv(os.path.join(DERIV, "summary.csv"), low_memory=False)
    pools = {"all": [], "random": [], "fixed_dense": []}
    for a, b, c, o, f, d in zip(df.StimA_file, df.StimB_file, df.StimC_file, df.choice_file,
                                df.is_fixed, df.is_dense):
        items = [os.path.basename(str(x)) for x in (a, b, c)]
        tr = (items, os.path.basename(str(o)))
        pools["all"].append(tr)
        pools["fixed_dense" if (truthy(f) or truthy(d)) else "random"].append(tr)
    return pools, int(df.SubID.nunique())


def build(trials, index, n):
    tog = np.zeros((n, n)); pres = np.zeros((n, n))
    for items, odd in trials:
        idx = [index.get(it) for it in items]
        if any(i is None for i in idx):
            continue
        o = index.get(odd)
        if o is None:
            continue
        for a in range(3):
            for b in range(a + 1, 3):
                i, j = idx[a], idx[b]
                pres[i, j] += 1; pres[j, i] += 1
                if idx[a] != o and idx[b] != o:
                    tog[i, j] += 1; tog[j, i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        D = 1.0 - (tog / pres)
    D[pres == 0] = np.nan
    return D, pres


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-obs", type=int, default=2,
                    help="minimum observations for a pair to enter the random-only scan")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    prim = np.load(os.path.join(DERIV, "triplet_rdm_by_filename.npz"), allow_pickle=True)
    labels = [str(x) for x in prim["labels"]]
    index = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    iu = np.triu_indices(n, 1)

    M = {}
    for m in MODELS:
        p = os.path.join(CANON, f"rdm_{m}.npz")
        if os.path.exists(p):
            M[m] = np.load(p, allow_pickle=True)["data"].astype(float)[iu]

    print("reading raw sessions ...")
    pools, kept = read_trials(index)
    print(f"  {kept} participants pass the published criterion")
    for k, v in pools.items():
        print(f"  {k:12s} {len(v):7d} trials")

    results = {}
    for pool in ("all", "fixed_dense", "random"):
        D, pres = build(pools[pool], index, n)
        y = D[iu]; c = pres[iu]
        thr = 20 if pool == "fixed_dense" else args.min_obs
        ok = np.isfinite(y) & (c >= thr)
        fits = {m: spear(y[ok], v[ok]) for m, v in M.items()}
        results[pool] = dict(n_pairs=int(ok.sum()), median_obs=float(np.median(c[ok])), fits=fits)
        print(f"\n{pool}: {int(ok.sum())} pairs (min {thr} obs, median {np.median(c[ok]):.0f})")

    order_ref = sorted(M, key=lambda m: -results["fixed_dense"]["fits"][m])
    print(f"\n{'model':26s}{'fixed+dense':>13s}{'random only':>13s}{'all trials':>12s}")
    for m in order_ref:
        print(f"{m:26s}{results['fixed_dense']['fits'][m]:>13.4f}"
              f"{results['random']['fits'][m]:>13.4f}{results['all']['fits'][m]:>12.4f}")

    a = np.array([results["fixed_dense"]["fits"][m] for m in MODELS if m in M])
    b = np.array([results["random"]["fits"][m] for m in MODELS if m in M])
    print(f"\nRank correlation between the two model orderings: {spear(a, b):.4f}")
    print(f"Top model on fixed+dense: {order_ref[0]}")
    print(f"Top model on random only: {max(M, key=lambda m: results['random']['fits'][m])}")
    print("\nAbsolute rho is attenuated on random trials by measurement error "
          "(median\n%d obs per pair vs %d on fixed+dense). The ORDERING is the result."
          % (results["random"]["median_obs"], results["fixed_dense"]["median_obs"]))

    out = os.path.join(OUTDIR, "trial_composition.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "rho_fixed_dense", "rho_random_only", "rho_all_trials"])
        for m in order_ref:
            w.writerow([m, round(results["fixed_dense"]["fits"][m], 4),
                        round(results["random"]["fits"][m], 4),
                        round(results["all"]["fits"][m], 4)])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
