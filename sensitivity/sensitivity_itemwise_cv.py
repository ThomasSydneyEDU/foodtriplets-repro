"""
Item-wise cross-validation sensitivity analysis (reviewer point RB-4c).

Reviewer B: "The five folds are constructed over stimulus pairs, allowing the same
food items to occur in both training and test sets in different pairings. The current
procedure therefore assesses generalisation to unseen pairings of represented items,
rather than to unseen food stimuli. The authors should clarify this point and, if
feasible, test whether the main findings hold under an item-wise cross-validation
scheme in which test items are absent from the training set."

STANDALONE BY DESIGN. Does not import or modify the main pipeline. Reads the empirical
triplet RDM and the model RDMs read-only and writes only into
sensitivity/sensitivity_outputs/.

THREE SCHEMES
-------------
pairwise      The published scheme. Folds over pairs; items recur between train and
              test in different pairings. Included as the reference point.

leave-one-item-out (LOIO)
              209 folds. For item i: train on every pair not involving i, test on every
              pair that does. The held-out item is genuinely absent from training, which
              is what the reviewer asked for, and every pair is used as test data (each
              appears in exactly two test sets). This is the well-powered item-wise test
              and is the one to report.

strict        Both items of a test pair absent from training. The conservative reading.
              Repeated random item holdout rather than k-fold, because k-fold here is a
              lottery: the fixed+dense subset is dominated by a 30-item dense block
              (30 items carry >=25 pairs each and account for 435 of the 703 pairs,
              while the median item has 2), so how many dense items land in a fold
              swamps everything else. Splits are STRATIFIED on dense-block membership to
              remove that lottery, and the spread across splits is reported rather than
              a point estimate.

WHAT TO EXPECT
--------------
The category increment (+0.044 pair-wise) should be resolvable under every scheme. The
transformation increments (+0.005 to +0.009) are below the resolution of the strict
scheme, which has a median of ~58 test pairs per split. Reporting that limit with the
fold sizes attached is a complete answer to a request hedged with "if feasible".

Note on why leakage risk is structurally low here: the NNLS fits ~14 weights, one per
predictor family, and every predictor is an RDM derived from item properties. There is
no item-level free parameter. The residual route is noise correlated within item, not
parameter memorisation. Make this point second, after the numbers, not instead of them.

Usage
-----
    python sensitivity/sensitivity_itemwise_cv.py
    python sensitivity/sensitivity_itemwise_cv.py --reps 50    # faster
"""
from __future__ import annotations
import argparse, csv, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DERIV = os.path.join(ROOT, "analysis_code", "derivatives")
CANON = os.path.join(DERIV, "model_rdms_canon")
OUTDIR = os.path.join(HERE, "sensitivity_outputs")
MIN_OBS = 20          # yields the 703 fixed+dense pairs
DENSE_DEG = 25        # items with >= this many pairs form the dense block


SEM = ["human_healthiness", "human_calorie_density", "human_savoriness", "human_sweetness",
       "human_saltiness", "human_fattiness", "human_sourness", "human_bitterness",
       "human_spiciness", "human_appeal"]
CAT = "food_classification"
TRANSFORM = ["NOVA_ordinal", "natural_vs_transformed", "Transformation_score"]


# ------------------------------------------------------------------ small stats
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


def nnls(A, b, tol=1e-10, max_iter=200):
    """Exact non-negative least squares (Lawson-Hanson active set).

    Replaces the projected-gradient solver used in the earlier diagnostics. That one
    needed ~8000 iterations to converge and still depressed absolute rho by ~0.015 at
    4000, which is the same order as the effects being measured. With ~14 predictors
    the active-set method converges in a handful of passes and is exact, so the
    solver can no longer contribute to the numbers.
    """
    m, n = A.shape
    P = np.zeros(n, bool)          # passive (free) set
    x = np.zeros(n)
    w = A.T @ (b - A @ x)
    for _ in range(max_iter):
        if P.all() or (w[~P] <= tol).all():
            break
        j = np.flatnonzero(~P)[np.argmax(w[~P])]
        P[j] = True
        for _ in range(max_iter):
            Ap = A[:, P]
            s_p, *_ = np.linalg.lstsq(Ap, b, rcond=None)
            if (s_p > tol).all():
                x = np.zeros(n); x[P] = s_p
                break
            # backtrack toward feasibility
            bad = s_p <= tol
            idx = np.flatnonzero(P)
            xp = x[P]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = np.where(bad, xp / np.maximum(xp - s_p, 1e-30), np.inf)
            alpha = np.min(ratios)
            x[P] = xp + alpha * (s_p - xp)
            P[idx[x[idx] <= tol]] = False
            x[~P] = 0.0
        else:
            break
        w = A.T @ (b - A @ x)
    return x


def fit_predict(Xtr, ytr, Xte):
    w = nnls(np.column_stack([np.ones(len(ytr)), Xtr]), ytr)
    return np.column_stack([np.ones(len(Xte)), Xte]) @ w


# --------------------------------------------------------------------- schemes
def cv_pairwise(X, y, reps, rng):
    out = []
    for r in range(reps):
        f = np.random.default_rng(r).permutation(len(y)) % 5
        pred = np.zeros(len(y))
        for k in range(5):
            tr, te = f != k, f == k
            pred[te] = fit_predict(X[tr], y[tr], X[te])
        out.append(spear(pred, y))
    return np.array(out)


def cv_loio(X, y, I, J, items):
    """Leave-one-item-out. Returns one pooled rho over all held-out predictions."""
    preds, obs = [], []
    for it in items:
        te = (I == it) | (J == it)
        tr = ~te
        if te.sum() == 0 or tr.sum() < 30:
            continue
        preds.append(fit_predict(X[tr], y[tr], X[te]))
        obs.append(y[te])
    return spear(np.concatenate(preds), np.concatenate(obs)), int(sum(len(p) for p in preds))


def cv_strict(X, y, I, J, items, dense, frac, reps, rng):
    """Both items of a test pair held out. Stratified on dense-block membership."""
    dense = np.array(sorted(dense)); sparse = np.array(sorted(set(items) - set(dense.tolist())))
    rhos, ntest = [], []
    for _ in range(reps):
        hd = rng.choice(dense, max(1, int(round(len(dense) * frac))), replace=False)
        hs = rng.choice(sparse, max(1, int(round(len(sparse) * frac))), replace=False)
        held = set(hd.tolist()) | set(hs.tolist())
        inh_i = np.isin(I, list(held)); inh_j = np.isin(J, list(held))
        te = inh_i & inh_j
        tr = (~inh_i) & (~inh_j)
        if te.sum() < 20 or tr.sum() < 60:
            continue
        p = fit_predict(X[tr], y[tr], X[te])
        rhos.append(spear(p, y[te])); ntest.append(int(te.sum()))
    return np.array(rhos), np.array(ntest)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200, help="repeats for the strict scheme")
    ap.add_argument("--frac", type=float, default=0.30, help="item holdout fraction, strict scheme")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(0)

    emp = np.load(os.path.join(DERIV, "triplet_rdm_by_filename.npz"), allow_pickle=True)
    labels = [str(x) for x in emp["labels"]]
    E = emp["data"].astype(float)
    counts = np.load(os.path.join(DERIV, "triplet_pair_counts_by_filename.npz"),
                     allow_pickle=True)["counts"].astype(float)
    iu = np.triu_indices(len(labels), 1)
    mask = np.isfinite(E[iu]) & (counts[iu] >= MIN_OBS)
    y = E[iu][mask]
    I, J = iu[0][mask], iu[1][mask]
    items = sorted(set(I.tolist()) | set(J.tolist()))

    deg = {}
    for a, b in zip(I, J):
        deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
    dense = {k for k, v in deg.items() if v >= DENSE_DEG}
    print(f"{len(items)} items, {len(y)} pairs; dense block = {len(dense)} items "
          f"carrying {int(np.sum(np.isin(I, list(dense)) & np.isin(J, list(dense))))} pairs")

    def load(n):
        return np.load(os.path.join(CANON, f"rdm_{n}.npz"), allow_pickle=True)["data"].astype(float)[iu][mask]
    M = {m: load(m) for m in SEM + [CAT] + TRANSFORM}

    SETS = {
        "Semantic": SEM,
        "Semantic + Category": SEM + [CAT],
        "Semantic + Category + NOVA": SEM + [CAT, "NOVA_ordinal"],
        "Semantic + Category + NvT": SEM + [CAT, "natural_vs_transformed"],
        "Semantic + Category + Transf": SEM + [CAT, "Transformation_score"],
        "Semantic + Category + all three": SEM + [CAT] + TRANSFORM,
    }

    rows = []
    print(f"\n{'model set':34s}{'pairwise':>12s}{'LOIO':>10s}{'strict (median [5-95])':>28s}")
    for name, cols in SETS.items():
        X = np.column_stack([M[c] for c in cols])
        pw = cv_pairwise(X, y, 10, rng)
        lo, n_lo = cv_loio(X, y, I, J, items)
        st, nt = cv_strict(X, y, I, J, items, dense, args.frac, args.reps, rng)
        rows.append(dict(model_set=name,
                         pairwise=round(float(pw.mean()), 4),
                         loio=round(float(lo), 4),
                         strict_median=round(float(np.median(st)), 4),
                         strict_p5=round(float(np.percentile(st, 5)), 4),
                         strict_p95=round(float(np.percentile(st, 95)), 4),
                         strict_n_splits=len(st),
                         strict_median_test_pairs=int(np.median(nt)),
                         loio_test_predictions=n_lo))
        print(f"{name:34s}{pw.mean():>12.4f}{lo:>10.4f}"
              f"{np.median(st):>16.4f} [{np.percentile(st,5):.3f},{np.percentile(st,95):.3f}]")

    base = {r["model_set"]: r for r in rows}["Semantic + Category"]
    print(f"\nINCREMENTS OVER SEMANTIC + CATEGORY")
    print(f"{'added predictor':34s}{'pairwise':>12s}{'LOIO':>10s}{'strict median':>16s}")
    for r in rows:
        if not r["model_set"].startswith("Semantic + Category +"):
            continue
        print(f"{r['model_set'].replace('Semantic + Category + ',''):34s}"
              f"{r['pairwise']-base['pairwise']:>+12.4f}"
              f"{r['loio']-base['loio']:>+10.4f}"
              f"{r['strict_median']-base['strict_median']:>+16.4f}")

    cat = {r["model_set"]: r for r in rows}["Semantic + Category"]
    sem = {r["model_set"]: r for r in rows}["Semantic"]
    print(f"\nCategory increment over Semantic alone:"
          f"  pairwise {cat['pairwise']-sem['pairwise']:+.4f}"
          f"   LOIO {cat['loio']-sem['loio']:+.4f}"
          f"   strict {cat['strict_median']-sem['strict_median']:+.4f}")
    print(f"\nResolution note: strict scheme median {rows[0]['strict_median_test_pairs']} test pairs "
          f"per split across {rows[0]['strict_n_splits']} splits; LOIO pools "
          f"{rows[0]['loio_test_predictions']} held-out predictions.")

    out = os.path.join(OUTDIR, "itemwise_cv.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
