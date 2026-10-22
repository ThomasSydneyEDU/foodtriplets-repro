"""
Category-scheme x transformation-measure sensitivity of the partial Spearman RSA
(reviewer point R1-4a, "circular selection of the category taxonomy").

WHY THIS EXISTS
---------------
The published 25-predictor variance-attribution model (manuscript "Variance attribution",
Figure 6) enters only the Intuitive category scheme but all three transformation measures.
Reviewer 1 (4a) objected that Intuitive was chosen partly for its fit to the behavioural
data. Two further problems surfaced during the 24 September 2026 audit:
  * the three taxonomies are LESS redundant with each other (model-RDM Spearman
    rho = 0.72-0.76 over the 703 fixed+dense pairs) than natural-vs-transformed is with
    transformation score (rho = 0.85), so "taxonomies redundant, transformation measures
    distinct" cannot be argued statistically;
  * entering the correlated transformation measures together lets each partial out the
    others, which can understate transformation's unique association.
This script therefore refits the partial Spearman RSA for every combination of ONE category
scheme (Intuitive 7, WHO 10, Culinary 9) and ONE transformation measure (NOVA level, natural
vs transformed, transformation score): a 3 x 3 grid of 23-predictor models (10 human-rated
dimensions + 11 visual RDMs + 1 category + 1 transformation).
Question asked: irrespective of the category and transformation model used, does category
remain the strongest unique predictor, and how large is transformation's unique association?

This is a ROBUSTNESS CHECK. The published 25-predictor model remains the headline analysis.
Shapley attribution is deliberately NOT rerun here (decision, 24 Sept 2026: the question is
about unique, not shared, contribution).

METHOD (identical to the pipeline, not a reimplementation)
---------------------------------------------------------
The partial correlations, bootstrap CIs and permutation p-values are computed by importing
`bootstrap_partial_correlation` from analysis_code/compute_variance_attribution.py, the
function that produced the published values: rank-transform y, x and every control; regress
the control ranks out of y and x by OLS; Pearson-correlate the residuals. The same call draws
n_boot bootstrap resamples of item pairs (95% percentile CI) and n_boot permutations of the
target predictor (two-sided p), with the pipeline's default seed (123). Defaults here:
n_boot = 200, seed = 123, matching the published run.

DATA
----
Rows: the 703 fixed+dense item pairs in
  analysis_code/compute_outputs/stage2_model_evaluation/fixed_dense/human/pairwise_vectors.csv
(the exact input to the published variance attribution). That file carries Intuitive
(`food_classification`) but not WHO/Culinary, so those two vectors are rebuilt from
analysis_code/derivatives/model_rdms_canon/rdm_Category_{WHO_10,Culinary_9}.npz.
ROW-ORDER TRAP: the pairwise file's i, j index the 209 fixed+dense items in the order the
pipeline builds them under --fixed-dense-only (sorted basenames of every file named in the
fixed/dense rows of derivatives/summary.csv; compute_model_evaluation.py ~l.938-981), NOT the
order of triplet_rdm_by_filename.npz or of the model RDM files. (Mixing these orders is what
broke the original A6 raw/prepared diagnostic.) The script rebuilds that order and ASSERTS
that it reproduces existing pairwise columns exactly before using it.

VALIDATION (the script stops if either fails)
---------------------------------------------
1. Index mapping: rebuilt Intuitive, natural-vs-transformed, transformation-score and appeal
   vectors must equal the pairwise-file columns.
2. Published reproduction: the published 25-predictor model is refitted and every
   predictor's partial rho must match
   analysis_code/compute_outputs/stage4_variance_attribution/fixed_dense/human/model_unique.csv
   to within 5e-4, i.e. identical at the 3 decimals the manuscript reports. Tolerance note:
   run on Linux (24 Sept 2026, numpy 2.2.6 / pandas 2.3.3 / scipy 1.15.3, the pinned
   requirements) the max |diff| was 2.7e-5, largest for the human-rating predictors. It is
   not CSV float parsing (checked); it is attributed to the BLAS/LAPACK backend (published
   run on macOS). On the machine that produced the published run it should be ~0. CIs and
   p-values are compared and reported, not asserted.

SHAPLEY OPTION (added 24 Sept 2026, --shapley)
----------------------------------------------
Also computes Shapley variance attribution for the same nine models with the pipeline's own
`shapley_attribution` (rank-transformed OLS R^2, 2,000 random orderings, seed 123, no controls:
the published settings). POINT ESTIMATES ONLY: the published run's bootstrap CIs would cost
~200x more and are not needed for the question (does category's share stay large whichever
scheme and transformation measure is used?). Validation: the published 25-predictor Shapley
values (variance_attribution_shapley.csv) must be reproduced to within 5e-4.
Output: sensitivity_outputs/shapley_3x3_long.csv and shapley_3x3_summary.csv.

USAGE (from the repository root, with the analysis environment active)
---------------------------------------------------------------------
    python sensitivity/sensitivity_partial_3x3.py
    python sensitivity/sensitivity_partial_3x3.py --n-boot 1000   # optional, slower
    python sensitivity/sensitivity_partial_3x3.py --shapley-only  # Shapley 3x3 only (~11 min, timed 24 Sept 2026)
    python sensitivity/sensitivity_partial_3x3.py --shapley       # partial and Shapley

OUTPUTS (sensitivity/sensitivity_outputs/)
-------------------------------------------------
partial_3x3_long.csv     every predictor in every model (model, category, transformation,
                         predictor, rho_partial, ci95_lo, ci95_hi, p_perm)
partial_3x3_summary.csv  one row per model: category and transformation partials with CIs/p,
                         category's rank among all 23 predictors, and whether category is the
                         strongest positive predictor
Also printed to the console.

STANDALONE: reads pipeline outputs read-only; writes only to sensitivity_outputs/.
Written 24 September 2026 during the manual response-letter audit; see
reviewer_comments_and_actions.md, "Reviewer 1, Item 4a".
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AC = os.path.join(ROOT, "analysis_code")
PAIRWISE = os.path.join(AC, "compute_outputs", "stage2_model_evaluation", "fixed_dense", "human", "pairwise_vectors.csv")
SUMMARY = os.path.join(AC, "derivatives", "summary.csv")
RDMS = os.path.join(AC, "derivatives", "model_rdms_canon")
PUB_DIR = os.path.join(AC, "compute_outputs", "stage4_variance_attribution", "fixed_dense", "human")
OUT = os.path.join(HERE, "sensitivity_outputs")

sys.path.insert(0, AC)
from compute_variance_attribution import bootstrap_partial_correlation, shapley_attribution  # noqa: E402  (the published functions)

CATEGORY = {"Intuitive": "food_classification", "WHO": "Category_WHO_10", "Culinary": "Category_Culinary_9"}
TRANSFORMATION = {"NOVA": "NOVA_ordinal", "Natural vs transformed": "natural_vs_transformed",
                  "Transformation score": "Transformation_score"}


def fixed_dense_labels():
    """209 fixed+dense item filenames, in the order the pipeline indexes pairwise_vectors (i, j)."""
    s = pd.read_csv(SUMMARY, low_memory=False)
    truthy = lambda c: s[c].astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])
    fd = s[truthy("is_fixed") | truthy("is_dense")]
    labs = set()
    for c in ("StimA_file", "StimB_file", "StimC_file", "choice_file"):
        labs.update(os.path.basename(str(x)) for x in fd[c].dropna().astype(str) if str(x).strip())
    return sorted(labs)


def rdm_vector(name, labels, v):
    d = np.load(os.path.join(RDMS, f"rdm_{name}.npz"), allow_pickle=True)
    idx = {os.path.basename(str(l)): n for n, l in enumerate(d["labels"])}
    M = d["data"]
    return np.array([M[idx[labels[i]], idx[labels[j]]] for i, j in zip(v["i"], v["j"])], dtype=float)


def fit(v, predictors, n_boot, seed, targets=None):
    """Partial Spearman for each target, controlling for all other predictors (pipeline method)."""
    y = v["emp"].to_numpy(dtype=float)
    rows = []
    for name in (targets or predictors):
        others = [c for c in predictors if c != name]
        Z = np.column_stack([v[c].to_numpy(dtype=float) for c in others])
        rho, lo, hi, p = bootstrap_partial_correlation(y, v[name].to_numpy(dtype=float), Z=Z,
                                                       n_boot=n_boot, seed=seed)
        rows.append({"predictor": name, "rho_partial": rho, "ci95_lo": lo, "ci95_hi": hi, "p_perm": p})
    return pd.DataFrame(rows)


def shapley_one(v, preds, a):
    y = v["emp"].to_numpy(dtype=float)
    X = {c: v[c].to_numpy(dtype=float) for c in preds}   # insertion order = predictor order, as in the pipeline
    out = shapley_attribution(y, X, controls=[], n_orderings=a.n_orderings, rank_transform=True, seed=a.seed)
    total = out.pop("__total_R2_with_all__"); out.pop("__baseline_R2__", None)
    return pd.Series(out), total


def run_shapley(v, published, a):
    t0 = time.time()
    pub = pd.read_csv(os.path.join(PUB_DIR, "variance_attribution_shapley.csv")).set_index("predictor")["mean"]
    rep, tot = shapley_one(v, published, a)
    d = (rep - pub.loc[rep.index]).abs().max()
    assert d < 5e-4 and abs(tot - pub["__total_R2_with_all__"]) < 5e-4, f"published Shapley not reproduced (max diff {d:.2e})"
    print(f"[ok] published 25-predictor Shapley reproduced (max |diff| {d:.1e}; total R2 {tot:.3f}) ({time.time()-t0:.0f}s)")
    controls = [p for p in published if p.startswith("human_") or p.startswith("rdm_ll_")]
    long_rows, summ = [], []
    for cname, ccol in CATEGORY.items():
        for tname, tcol in TRANSFORMATION.items():
            sh, tot = shapley_one(v, controls + [ccol, tcol], a)
            order = sh.sort_values(ascending=False)
            long_rows.append(pd.DataFrame({"model": f"{cname} + {tname}", "category": cname, "transformation": tname,
                                           "predictor": sh.index, "shapley_delta_r2": sh.values, "total_r2": tot}))
            summ.append({"model": f"{cname} + {tname}", "category": cname, "transformation": tname, "total_r2": tot,
                         "top_predictor": order.index[0], "top_share": order.iloc[0],
                         "cat_share": sh[ccol], "cat_rank_of_23": list(order.index).index(ccol) + 1,
                         "trans_share": sh[tcol], "trans_rank_of_23": list(order.index).index(tcol) + 1})
            print(f"  {cname:9s} + {tname:22s} total {tot:.3f} | category {sh[ccol]:.4f} (rank {summ[-1]['cat_rank_of_23']})"
                  f" | transformation {sh[tcol]:.4f} (rank {summ[-1]['trans_rank_of_23']}) | top {order.index[0]} ({time.time()-t0:.0f}s)", flush=True)
    pd.concat(long_rows).to_csv(os.path.join(OUT, "shapley_3x3_long.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "shapley_3x3_summary.csv"), index=False)
    print(f"Wrote shapley_3x3_long.csv and shapley_3x3_summary.csv ({time.time()-t0:.0f}s, {a.n_orderings} orderings, seed {a.seed})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples AND permutations (default 200, as published)")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed (default 123, as published)")
    ap.add_argument("--skip-reproduction", action="store_true", help="skip refitting the published model (not recommended)")
    ap.add_argument("--shapley", action="store_true", help="also run the Shapley 3x3 (point estimates)")
    ap.add_argument("--shapley-only", action="store_true", help="run only the Shapley 3x3")
    ap.add_argument("--n-orderings", type=int, default=2000, help="Shapley orderings (default 2000, as published)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    v = pd.read_csv(PAIRWISE)
    if "used_in_regression" in v.columns:
        used = v["used_in_regression"].astype(str).str.lower().isin(["true", "1", "1.0"])
        if not used.all():
            v = v[used].reset_index(drop=True)
    assert len(v) == 703, f"expected 703 fixed+dense pairs, found {len(v)}"

    # --- 1. index mapping ---------------------------------------------------------------
    labels = fixed_dense_labels()
    assert len(labels) == 209 and v[["i", "j"]].to_numpy().max() == 208, "fixed+dense label order not recovered"
    for col in ("food_classification", "natural_vs_transformed", "Transformation_score", "human_appeal"):
        assert np.allclose(rdm_vector(col, labels, v), v[col].to_numpy(dtype=float)), f"index mapping fails for {col}"
    for key in ("Category_WHO_10", "Category_Culinary_9"):
        v[key] = rdm_vector(key, labels, v)
    print("[ok] index mapping verified against the pipeline's pairwise vectors")

    # --- 2. reproduce the published 25-predictor model ----------------------------------
    published = json.load(open(os.path.join(PUB_DIR, "variance_attribution_manifest.json")))["predictors"]
    assert len(published) == 25
    if a.shapley or a.shapley_only:
        run_shapley(v, published, a)
        if a.shapley_only:
            return
    if not a.skip_reproduction:
        pub = pd.read_csv(os.path.join(PUB_DIR, "model_unique.csv")).set_index("model")
        rep = fit(v, published, a.n_boot, a.seed).set_index("predictor")
        d = (rep["rho_partial"] - pub.loc[rep.index, "rho_partial"]).abs().max()
        assert d < 5e-4, f"published partial rho not reproduced to 3 dp (max diff {d:.2e})"
        dci = (rep[["ci95_lo", "ci95_hi", "p_perm"]] - pub.loc[rep.index, ["ci95_lo", "ci95_hi", "p_perm"]]).abs().max().max()
        print(f"[ok] published 25-predictor partials reproduced (max |diff| rho {d:.1e}; CI/p {dci:.1e})")

    # --- 3. the 3 x 3 grid ----------------------------------------------------------------
    controls = [p for p in published if p.startswith("human_") or p.startswith("rdm_ll_")]
    assert len(controls) == 21
    long_rows, summ = [], []
    for cname, ccol in CATEGORY.items():
        for tname, tcol in TRANSFORMATION.items():
            preds = controls + [ccol, tcol]
            res = fit(v, preds, a.n_boot, a.seed)
            res.insert(0, "transformation", tname); res.insert(0, "category", cname)
            res.insert(0, "model", f"{cname} + {tname}")
            long_rows.append(res)
            r = res.set_index("predictor")
            order = r["rho_partial"].sort_values(ascending=False)
            summ.append({
                "model": f"{cname} + {tname}", "category": cname, "transformation": tname,
                "cat_rho": r.at[ccol, "rho_partial"], "cat_lo": r.at[ccol, "ci95_lo"], "cat_hi": r.at[ccol, "ci95_hi"], "cat_p": r.at[ccol, "p_perm"],
                "cat_rank_of_23": int(list(order.index).index(ccol)) + 1,
                "cat_strongest": bool(order.index[0] == ccol),
                "trans_rho": r.at[tcol, "rho_partial"], "trans_lo": r.at[tcol, "ci95_lo"], "trans_hi": r.at[tcol, "ci95_hi"], "trans_p": r.at[tcol, "p_perm"],
                "trans_ci_excludes_zero": bool(r.at[tcol, "ci95_lo"] > 0 or r.at[tcol, "ci95_hi"] < 0),
            })
            print(f"  {cname:9s} + {tname:22s} category {r.at[ccol,'rho_partial']:+.3f} (rank {summ[-1]['cat_rank_of_23']})"
                  f" | transformation {r.at[tcol,'rho_partial']:+.3f} [{r.at[tcol,'ci95_lo']:+.3f}, {r.at[tcol,'ci95_hi']:+.3f}] p={r.at[tcol,'p_perm']:.3f}")

    pd.concat(long_rows).to_csv(os.path.join(OUT, "partial_3x3_long.csv"), index=False)
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(OUT, "partial_3x3_summary.csv"), index=False)  # full precision: round only when reporting
    print(f"\nCategory strongest positive predictor in {s['cat_strongest'].sum()}/9 models; "
          f"transformation partial {s['trans_rho'].min():+.3f} to {s['trans_rho'].max():+.3f}, "
          f"CI excludes zero in {s['trans_ci_excludes_zero'].sum()}/9.")
    print(f"Wrote partial_3x3_long.csv and partial_3x3_summary.csv to {OUT}  ({time.time()-t0:.0f}s, n_boot={a.n_boot}, seed={a.seed})")


if __name__ == "__main__":
    main()
