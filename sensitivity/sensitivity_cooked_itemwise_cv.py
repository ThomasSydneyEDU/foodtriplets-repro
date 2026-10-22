"""
Cooked-measure extension of the item-wise cross-validation (RB-4c) in sensitivity_itemwise_cv.py.

Adds the two cooked transformation measures (liberal, strict) from
analysis_code/reference/cooked_transformation_coding.csv to the model sets, under the same three
schemes (pair-wise 5-fold, leave-one-item-out, strict item holdout), using the functions and
settings of sensitivity_itemwise_cv.py (imported, not copied): exact Lawson-Hanson NNLS,
MIN_OBS = 20 pairs, dense block >= 25 pairs, strict holdout fraction 0.30, 200 repeats.
The cooked RDMs are built IN MEMORY as binary class-difference matrices over the same item order
as triplet_rdm_by_filename.npz; nothing is written to model_rdms_canon and the pipeline is untouched.

Columns of cooked_itemwise_cv.csv
---------------------------------
pairwise, loio, strict_median/p5/p95 : same definitions as itemwise_cv.csv. The first six rows
    (the original sets) are run first, in the original order with the original shared RNG, so
    they reproduce the committed itemwise_cv.csv (checked if the file is found).
strict_paired_median : strict median when EVERY set uses an identical fresh RNG (seed 0), i.e.
    identical item splits across sets. Use this column for increments between sets, because the
    shared-RNG column gives each set different random splits.

Usage: python sensitivity/sensitivity_cooked_itemwise_cv.py [--reps 200] [--existing PATH]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, csv, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sensitivity_itemwise_cv as CV   # noqa: E402

CODING = os.path.join(CV.ROOT, "analysis_code", "reference", "cooked_transformation_coding.csv")
LIB, STR = "cooked_liberal", "cooked_strict"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--frac", type=float, default=0.30)
    ap.add_argument("--existing", default=os.path.join(CV.OUTDIR, "itemwise_cv.csv"))
    a = ap.parse_args()
    os.makedirs(CV.OUTDIR, exist_ok=True)

    emp = np.load(os.path.join(CV.DERIV, "triplet_rdm_by_filename.npz"), allow_pickle=True)
    labels = [str(x) for x in emp["labels"]]
    E = emp["data"].astype(float)
    counts = np.load(os.path.join(CV.DERIV, "triplet_pair_counts_by_filename.npz"), allow_pickle=True)["counts"].astype(float)
    iu = np.triu_indices(len(labels), 1)
    mask = np.isfinite(E[iu]) & (counts[iu] >= CV.MIN_OBS)
    y = E[iu][mask]; I, J = iu[0][mask], iu[1][mask]
    items = sorted(set(I.tolist()) | set(J.tolist()))
    deg = {}
    for p, q in zip(I, J):
        deg[p] = deg.get(p, 0) + 1; deg[q] = deg.get(q, 0) + 1
    dense = {k for k, n in deg.items() if n >= CV.DENSE_DEG}
    print(f"{len(items)} items, {len(y)} pairs; dense block {len(dense)} items")

    coding = pd.read_csv(CODING).set_index("filename")
    base = [os.path.basename(l) for l in labels]
    missing = [b for b in base if b not in coding.index]
    assert not missing, f"{len(missing)} labels missing from coding file, e.g. {missing[:3]}"

    def cooked(col):
        cls = np.array([1.0 if coding.at[b, col] == "Transformed" else 0.0 for b in base])
        return np.abs(cls[:, None] - cls[None, :])[iu][mask]

    def load(n):
        return np.load(os.path.join(CV.CANON, f"rdm_{n}.npz"), allow_pickle=True)["data"].astype(float)[iu][mask]
    M = {m: load(m) for m in CV.SEM + [CV.CAT] + CV.TRANSFORM}
    M[LIB] = cooked("natural_vs_transformed_cooked"); M[STR] = cooked("natural_vs_transformed_cooked_strict")
    NVT = "natural_vs_transformed"
    SEM, CAT = CV.SEM, CV.CAT
    SETS = {   # first six = original sets, original order
        "Semantic": SEM,
        "Semantic + Category": SEM + [CAT],
        "Semantic + Category + NOVA": SEM + [CAT, "NOVA_ordinal"],
        "Semantic + Category + NvT": SEM + [CAT, NVT],
        "Semantic + Category + Transf": SEM + [CAT, "Transformation_score"],
        "Semantic + Category + all three": SEM + [CAT] + CV.TRANSFORM,
        "Semantic + Category + Cooked (liberal)": SEM + [CAT, LIB],
        "Semantic + Category + Cooked (strict)": SEM + [CAT, STR],
        "Semantic + Category + NvT + Cooked (liberal)": SEM + [CAT, NVT, LIB],
        "Semantic + Cooked (liberal)": SEM + [LIB],
        "Semantic + NvT": SEM + [NVT],
        "Semantic + Cooked (strict)": SEM + [STR],
    }
    rng = np.random.default_rng(0)
    rows = []
    print(f"\n{'model set':46s}{'pairwise':>10s}{'LOIO':>9s}{'strict med [5-95]':>24s}{'paired':>9s}")
    for name, cols in SETS.items():
        X = np.column_stack([M[c] for c in cols])
        pw = CV.cv_pairwise(X, y, 10, rng)
        lo, n_lo = CV.cv_loio(X, y, I, J, items)
        st, nt = CV.cv_strict(X, y, I, J, items, dense, a.frac, a.reps, rng)
        sp, _ = CV.cv_strict(X, y, I, J, items, dense, a.frac, a.reps, np.random.default_rng(0))
        rows.append(dict(model_set=name, pairwise=round(float(pw.mean()), 4), loio=round(float(lo), 4),
                         strict_median=round(float(np.median(st)), 4), strict_p5=round(float(np.percentile(st, 5)), 4),
                         strict_p95=round(float(np.percentile(st, 95)), 4), strict_paired_median=round(float(np.median(sp)), 4),
                         strict_n_splits=len(st), strict_median_test_pairs=int(np.median(nt)), loio_test_predictions=n_lo))
        print(f"{name:46s}{pw.mean():>10.4f}{lo:>9.4f}{np.median(st):>13.4f} [{np.percentile(st,5):.3f},{np.percentile(st,95):.3f}]{np.median(sp):>9.4f}", flush=True)

    df = pd.DataFrame(rows).set_index("model_set")
    if os.path.exists(a.existing):
        ex = pd.read_csv(a.existing).set_index("model_set")
        d = (df.loc[ex.index, ex.columns.intersection(df.columns)] - ex[ex.columns.intersection(df.columns)]).abs().max().max()
        print(f"\nreproduction of committed itemwise_cv.csv rows: max |diff| {d:.1e}")
    ref = df.loc["Semantic + Category"]
    print("\nINCREMENTS OVER SEMANTIC + CATEGORY (strict uses the paired column)")
    print(f"{'added':46s}{'pairwise':>10s}{'LOIO':>9s}{'strict (paired)':>17s}")
    inc = []
    for nme in ["Semantic + Category + NOVA", "Semantic + Category + NvT", "Semantic + Category + Transf",
                "Semantic + Category + Cooked (liberal)", "Semantic + Category + Cooked (strict)",
                "Semantic + Category + NvT + Cooked (liberal)", "Semantic + Category + all three"]:
        r = df.loc[nme]
        inc.append(dict(model_set=nme, d_pairwise=r.pairwise - ref.pairwise, d_loio=r.loio - ref.loio,
                        d_strict_paired=r.strict_paired_median - ref.strict_paired_median))
        print(f"{nme.replace('Semantic + Category + ',''):46s}{inc[-1]['d_pairwise']:>+10.4f}{inc[-1]['d_loio']:>+9.4f}{inc[-1]['d_strict_paired']:>+17.4f}")
    sem = df.loc["Semantic"]
    print(f"\nCategory increment over Semantic: pairwise {ref.pairwise-sem.pairwise:+.4f}  LOIO {ref.loio-sem.loio:+.4f}  strict(paired) {ref.strict_paired_median-sem.strict_paired_median:+.4f}")
    for nme in ["Semantic + NvT", "Semantic + Cooked (liberal)", "Semantic + Cooked (strict)"]:
        r = df.loc[nme]
        print(f"{nme} over Semantic (no category): pairwise {r.pairwise-sem.pairwise:+.4f}  LOIO {r.loio-sem.loio:+.4f}  strict(paired) {r.strict_paired_median-sem.strict_paired_median:+.4f}")
    out = df.reset_index()
    out.to_csv(os.path.join(CV.OUTDIR, "cooked_itemwise_cv.csv"), index=False)
    pd.DataFrame(inc).to_csv(os.path.join(CV.OUTDIR, "cooked_itemwise_cv_increments.csv"), index=False)
    print("\nwrote cooked_itemwise_cv.csv, cooked_itemwise_cv_increments.csv")


if __name__ == "__main__":
    main()
