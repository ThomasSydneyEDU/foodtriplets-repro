"""
Item-level (cluster) bootstrap of the Shapley variance attribution, for the question
"does category stay the largest share, and by how much, when item dependence is respected?"
(Jason's point on the bootstrap; reviewer R1-4a follow-up).

WHY THIS EXISTS
---------------
The pipeline's `bootstrap_shapley` (analysis_code/compute_variance_attribution.py) resamples
ROWS, i.e. the 703 item pairs, as if they were independent. They are not: the pairs share only
209 items (median item has 2 pairs, a 30-item dense block carries 435 pairs). Row resampling
therefore understates uncertainty. This script resamples the 209 ITEMS with replacement and
keeps the pairs they induce. It is a sensitivity analysis; the published run is untouched.

RESAMPLING SCHEME (documented choices)
--------------------------------------
For each resample b:
  1. Draw 209 items with replacement from the 209 fixed+dense items; k_a = times item a is drawn.
  2. Every observed pair (a, b) enters with multiplicity k_a * k_b (an item drawn twice
     contributes its pairs twice; an item not drawn contributes none). Rows are replicated, so
     rank transformation sees tied copies exactly as a pair-level resample would.
  3. Pairs between two copies of the SAME item would be self-pairs (distance 0, never observed
     in the data) and are dropped. No such rows exist in the 703 observed pairs, so this is
     implicit: only observed pairs (a != b) can be selected.
  4. The empirical dissimilarity (`emp`) of each pair is held fixed; only which pairs are
     present, and how often, changes. (Re-deriving `emp` from resampled triplets is not done.)
  5. Shapley values are recomputed on the resample: rank-transformed OLS R^2, no controls,
     random orderings (pipeline settings except the number of orderings, default 300).
The same item resamples are used for every model within a run (paired comparison). Resample b
uses item seed [seed, b] and ordering seed seed + 10 + b (as in the pipeline's bootstrap).

QUANTITIES RECORDED PER MODEL
-----------------------------
Per-predictor Shapley mean and percentile 95% CI; lead = category share minus the largest
share among the other predictors (and which predictor that is); 95% CI of the lead; and
P(category first) = share of resamples with lead > 0.

TWO IMPLEMENTATIONS (identical numbers)
---------------------------------------
impl="pipeline": calls the published `shapley_attribution` (slow: re-ranks and refits OLS
                 for every subset).
impl="fast":     ranks and z-scores once per resample, then evaluates each subset R^2 from the
                 predictor correlation matrix (R^2 = b' G^-1 b). Same RNG orderings, same
                 result to ~1e-12; checked against the pipeline function by the
                 pilot. Typically >30x faster, which is what makes B=200 feasible.

USAGE (repository root, pinned analysis environment)
----------------------------------------------------
    python sensitivity/sensitivity_shapley_item_bootstrap.py --pilot
    python sensitivity/sensitivity_shapley_item_bootstrap.py --B 200 --n-orderings 300 --models all15

Outputs (new names only): sensitivity_outputs/bootstrap_shapley_item_*.csv.
Standalone: reads pipeline outputs read-only. Does not write to model_rdms_canon.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, sys, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sensitivity_partial_3x3 as P3   # noqa: E402  (published helpers, no side effects on import)
from sensitivity_partial_3x3 import shapley_attribution  # noqa: E402

OUT = P3.OUT
CODING = os.path.join(P3.AC, "reference", "cooked_transformation_coding.csv")
COOKED = {"Cooked (liberal)": "natural_vs_transformed_cooked",
          "Cooked (strict)": "natural_vs_transformed_cooked_strict"}


# ------------------------------------------------------------------------------ data
def cooked_rdm_vector(col, labels, v):
    """Binary RDM (0 same class, 1 different) built IN MEMORY from the coding file, same
    convention as rdm_natural_vs_transformed (checked in main script). Never written to disk."""
    c = pd.read_csv(CODING).set_index("filename")[col]
    cls = np.array([1.0 if c[l] == "Transformed" else 0.0 for l in labels])
    return np.abs(cls[v["i"].to_numpy()] - cls[v["j"].to_numpy()])


def load_data():
    """Returns (v, published, controls, model_defs). v has all predictor columns incl. cooked."""
    v = pd.read_csv(P3.PAIRWISE)
    if "used_in_regression" in v.columns:
        used = v["used_in_regression"].astype(str).str.lower().isin(["true", "1", "1.0"])
        if not used.all():
            v = v[used].reset_index(drop=True)
    assert len(v) == 703
    labels = P3.fixed_dense_labels()
    assert len(labels) == 209
    for col in ("food_classification", "natural_vs_transformed", "Transformation_score", "human_appeal"):
        assert np.allclose(P3.rdm_vector(col, labels, v), v[col].to_numpy(dtype=float)), col
    for key in ("Category_WHO_10", "Category_Culinary_9"):
        v[key] = P3.rdm_vector(key, labels, v)
    for name, col in COOKED.items():
        v[col] = cooked_rdm_vector(col, labels, v)
    published = json.load(open(os.path.join(P3.PUB_DIR, "variance_attribution_manifest.json")))["predictors"]
    controls = [p for p in published if p.startswith("human_") or p.startswith("rdm_ll_")]
    assert len(controls) == 21
    trans = dict(P3.TRANSFORMATION); trans.update(COOKED)
    defs = {}
    for cn, cc in P3.CATEGORY.items():
        for tn, tc in trans.items():
            defs[f"{cn} + {tn}"] = dict(category=cn, transformation=tn, ccol=cc, tcol=tc,
                                         preds=controls + [cc, tc])
    return v, published, controls, defs


MODEL_SETS = {
    "cooked3": lambda d: [m for m in d if m.endswith("Cooked (liberal)")],
    "all12": lambda d: [m for m in d if not m.endswith("Cooked (strict)")],
    "all15": lambda d: list(d),
}


# ------------------------------------------------------------------------- fast Shapley
def _z(M):
    M = M - M.mean(0)
    sd = M.std(0)
    sd[sd == 0] = 1.0
    return M / sd


def fast_shapley(y, X, n_orderings, seed):
    """Shapley deltas of rank-transformed OLS R^2 (no controls); identical orderings to
    `shapley_attribution`. y: (n,), X: (n, p). Returns (phi (p,), total_R2)."""
    n, p = X.shape
    yz = _z(rankdata(y)[:, None])[:, 0]
    Xz = _z(np.column_stack([rankdata(X[:, j]) for j in range(p)]))
    G = Xz.T @ Xz / n
    b = Xz.T @ yz / n

    def r2(idx):
        Gs, bs = G[np.ix_(idx, idx)], b[idx]
        try:
            w = np.linalg.solve(Gs, bs)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(Gs, bs, rcond=None)[0]
        return min(max(float(bs @ w), 0.0), 1.0)

    rng = np.random.default_rng(seed)
    phi = np.zeros(p)
    for _ in range(n_orderings):
        order = list(range(p)); rng.shuffle(order)
        prev = 0.0
        for k in range(p):
            cur = r2(order[:k + 1])
            phi[order[k]] += cur - prev
            prev = cur
    return phi / n_orderings, r2(list(range(p)))


def shapley_vec(y, Xmat, names, n_orderings, seed, impl):
    if impl == "fast":
        return fast_shapley(y, Xmat, n_orderings, seed)
    out = shapley_attribution(y, {k: Xmat[:, j] for j, k in enumerate(names)}, controls=[],
                              n_orderings=n_orderings, rank_transform=True, seed=seed)
    tot = out.pop("__total_R2_with_all__"); out.pop("__baseline_R2__", None)
    return np.array([out[k] for k in names]), tot


# --------------------------------------------------------------------------- bootstrap
_G = {}


def _init(v, defs, n_orderings, seed, impl):
    _G.update(v=v, defs=defs, n_orderings=n_orderings, seed=seed, impl=impl)


def item_resample_rows(i, j, b, seed, n_items=209):
    rng = np.random.default_rng([seed, b])
    k = np.bincount(rng.integers(0, n_items, n_items), minlength=n_items)
    w = k[i] * k[j]                                   # multiplicity; a self-pair cannot occur (i != j)
    return np.repeat(np.arange(len(i)), w)


def one_resample(args):
    b, models = args
    v, defs = _G["v"], _G["defs"]
    rows = item_resample_rows(v["i"].to_numpy(), v["j"].to_numpy(), b, _G["seed"])
    y = v["emp"].to_numpy(dtype=float)[rows]
    res = []
    for m in models:
        d = defs[m]
        Xmat = v[d["preds"]].to_numpy(dtype=float)[rows]
        phi, tot = shapley_vec(y, Xmat, d["preds"], _G["n_orderings"], _G["seed"] + 10 + b, _G["impl"])
        res.append((m, b, phi, tot, len(rows)))
    return res


def summarise(raw, defs, models):
    """raw: list of (model, b, phi, tot, nrows). Returns (per-predictor df, per-model lead df)."""
    pred_rows, lead_rows = [], []
    for m in models:
        d = defs[m]; names = d["preds"]
        recs = [r for r in raw if r[0] == m]
        Phi = np.vstack([r[2] for r in recs]); tot = np.array([r[3] for r in recs])
        ci = names.index(d["ccol"])
        others = [k for k in range(len(names)) if k != ci]
        comp = Phi[:, others]
        lead = Phi[:, ci] - comp.max(1)
        rival = np.array(names, dtype=object)[np.array(others)[comp.argmax(1)]]
        for j, nme in enumerate(names):
            pred_rows.append(dict(model=m, predictor=nme, mean=Phi[:, j].mean(),
                                  ci95_lo=np.percentile(Phi[:, j], 2.5), ci95_hi=np.percentile(Phi[:, j], 97.5)))
        top_rival = pd.Series(rival).value_counts()
        lead_rows.append(dict(model=m, category=d["category"], transformation=d["transformation"],
                              B=len(recs), cat_mean=Phi[:, ci].mean(),
                              cat_lo=np.percentile(Phi[:, ci], 2.5), cat_hi=np.percentile(Phi[:, ci], 97.5),
                              lead_mean=lead.mean(), lead_lo=np.percentile(lead, 2.5), lead_hi=np.percentile(lead, 97.5),
                              p_category_first=float((lead > 0).mean()),
                              modal_rival=top_rival.index[0], modal_rival_share=float(top_rival.iloc[0] / len(rival)),
                              total_r2_mean=tot.mean(), rows_mean=float(np.mean([r[4] for r in recs]))))
    return pd.DataFrame(pred_rows), pd.DataFrame(lead_rows)


def run_bootstrap(v, defs, models, B, n_orderings, seed, impl, workers):
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(v, defs, n_orderings, seed, impl)) as pool:
        out = pool.map(one_resample, [(b, models) for b in range(B)], chunksize=1)
    raw = [r for chunk in out for r in chunk]
    return raw, time.time() - t0


# -------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--n-orderings", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--models", default="cooked3", help="cooked3 | all12 | all15 | comma-separated model names")
    ap.add_argument("--impl", choices=["fast", "pipeline"], default="fast")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--pilot", action="store_true", help="timing + B=20 on one model only; no full run")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    v, published, controls, defs = load_data()

    if a.pilot:
        pilot_model = "WHO + Cooked (liberal)"
        d = defs[pilot_model]
        y = v["emp"].to_numpy(dtype=float); Xm = v[d["preds"]].to_numpy(dtype=float)
        # 1. agreement + seconds per fit, original data, 300 orderings
        t = time.time(); ph_p, tot_p = shapley_vec(y, Xm, d["preds"], 300, 123, "pipeline"); t_pipe = time.time() - t
        ts = []
        for _ in range(3):
            t = time.time(); ph_f, tot_f = shapley_vec(y, Xm, d["preds"], 300, 123, "fast"); ts.append(time.time() - t)
        print(f"fit @300 orderings: pipeline {t_pipe:.2f}s | fast {np.mean(ts):.3f}s; "
              f"max|diff| {np.abs(ph_p-ph_f).max():.1e}; cores {os.cpu_count()}")
        # 2. agreement on item-resampled data
        rows = item_resample_rows(v["i"].to_numpy(), v["j"].to_numpy(), 0, a.seed)
        pp, _ = shapley_vec(y[rows], Xm[rows], d["preds"], 300, 133, "pipeline")
        pf, _ = shapley_vec(y[rows], Xm[rows], d["preds"], 300, 133, "fast")
        print(f"resample agreement max|diff| {np.abs(pp-pf).max():.1e} ({len(rows)} rows)")
        # 3. B=20 on one model
        for impl, w in (("fast", a.workers), ("fast", 1)):
            raw, el = run_bootstrap(v, defs, [pilot_model], 20, a.n_orderings, a.seed, impl, w)
            print(f"pilot B=20, {pilot_model}, {a.n_orderings} orderings, impl={impl}, workers={w}: {el:.1f}s "
                  f"({el/20:.2f}s per resample)")
        pr, ld = summarise(raw, defs, [pilot_model])
        pr.to_csv(os.path.join(OUT, f"bootstrap_shapley_item_pilot_predictors{a.tag}.csv"), index=False)
        ld.to_csv(os.path.join(OUT, f"bootstrap_shapley_item_pilot_lead{a.tag}.csv"), index=False)
        print(ld.T.to_string())
        return

    models = MODEL_SETS[a.models](defs) if a.models in MODEL_SETS else a.models.split(",")
    raw, el = run_bootstrap(v, defs, models, a.B, a.n_orderings, a.seed, a.impl, a.workers)
    pr, ld = summarise(raw, defs, models)
    pr.to_csv(os.path.join(OUT, f"bootstrap_shapley_item_predictors{a.tag}.csv"), index=False)
    ld.to_csv(os.path.join(OUT, f"bootstrap_shapley_item_lead{a.tag}.csv"), index=False)
    print(ld.round(4).to_string()); print(f"{el:.0f}s")


if __name__ == "__main__":
    main()
