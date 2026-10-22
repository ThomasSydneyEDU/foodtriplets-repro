"""
Cooked-measure extension of the category x transformation sensitivity grid
(Jason's point: "transformed" should mean cooked or processed, FRIDa-style).

WHAT THIS DOES
--------------
Adds two new transformation measures to the 3 x 3 grid of sensitivity_partial_3x3.py, giving
3 category schemes x 5 transformation measures = 15 models of 23 predictors each (21 controls =
10 human-rated dimensions + 11 visual RDMs, one category scheme, one transformation measure):
  * Cooked (liberal): analysis_code/reference/cooked_transformation_coding.csv,
                      column natural_vs_transformed_cooked (Transformed = heat-cooked or
                      otherwise processed; visually ambiguous items coded by usual preparation)
  * Cooked (strict) : column natural_vs_transformed_cooked_strict (the 14 items coded only by
                      that knowledge-based default are recoded Natural)
Both RDMs are built IN MEMORY (binary: 0 same class, 1 different class). Nothing is merged
into load_info(), the dynamic CSV or analysis_helpers, and nothing is written to
analysis_code/derivatives/model_rdms_canon. The pipeline is untouched.

METHOD: identical to sensitivity_partial_3x3.py (it imports that module's helpers and the
published `bootstrap_partial_correlation` / `shapley_attribution`). Partial Spearman: n_boot
200, seed 123. Shapley: 2000 random orderings, seed 123, rank-transformed OLS R^2, via the
fast equivalent in sensitivity_shapley_item_bootstrap.py (identical to the pipeline function
to ~1e-15; verified here on the published model and one cooked model).

VALIDATION (stops on failure)
-----------------------------
  1. index mapping (as in the 3 x 3 script);
  2. binary-RDM convention: the existing rdm_natural_vs_transformed is a 0/1 class-difference
     matrix, so the cooked RDMs are built the same way;
  3. published 25-predictor partial rho reproduced to 5e-4;
  4. published 25-predictor Shapley (pipeline function, 2000 orderings) reproduced to 5e-4.
The nine pre-existing grid cells are also refitted and compared with the committed
partial_3x3_summary.csv / shapley_3x3_summary.csv (reported, not asserted).

OUTPUTS (new names; existing sensitivity_outputs files are never overwritten)
-------------------------------------------------------------------------------
cooked_partial_grid_long.csv, cooked_partial_grid_summary.csv
cooked_shapley_grid_long.csv, cooked_shapley_grid_summary.csv
cooked_univariate.csv   univariate Spearman of each transformation RDM with the behavioural RDM
                        on the 703 pairs (pair-bootstrap CI, permutation p)

USAGE (repository root, pinned environment)
    python sensitivity/sensitivity_cooked_measure.py [--existing-dir PATH]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, sys, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sensitivity_shapley_item_bootstrap as BS   # noqa: E402
import sensitivity_partial_3x3 as P3               # noqa: E402

OUT = P3.OUT
_V = {}


def _partial_job(args):
    m, preds, n_boot, seed = args
    return m, P3.fit(_V["v"], preds, n_boot, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-orderings", type=int, default=2000)
    ap.add_argument("--existing-dir", default=OUT, help="folder with committed partial_3x3_summary.csv / shapley_3x3_summary.csv")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    v, published, controls, defs = BS.load_data()
    y = v["emp"].to_numpy(dtype=float)
    print("[ok] index mapping verified")

    # 2. convention check: existing NvT RDM is a binary class-difference matrix
    d = np.load(os.path.join(P3.RDMS, "rdm_natural_vs_transformed.npz"), allow_pickle=True)
    M = d["data"]; s = M[0].copy(); s[0] = 0.0
    assert set(np.unique(M)) <= {0.0, 1.0} and np.array_equal(M, np.abs(s[:, None] - s[None, :])), "NvT RDM is not a 0/1 class-difference matrix"
    print("[ok] rdm_natural_vs_transformed is a 0/1 class-difference matrix (cooked RDMs use the same convention)")

    # 3. + 4. reproduction
    pub = pd.read_csv(os.path.join(P3.PUB_DIR, "model_unique.csv")).set_index("model")
    rep = P3.fit(v, published, a.n_boot, a.seed).set_index("predictor")
    dr = (rep["rho_partial"] - pub.loc[rep.index, "rho_partial"]).abs().max()
    assert dr < 5e-4, dr
    print(f"[ok] published 25-predictor partial rho reproduced (max |diff| {dr:.1e}) ({time.time()-t0:.0f}s)")
    pubsh = pd.read_csv(os.path.join(P3.PUB_DIR, "variance_attribution_shapley.csv")).set_index("predictor")["mean"]
    t1 = time.time()
    rsh, tot = P3.shapley_one(v, published, argparse.Namespace(n_orderings=a.n_orderings, seed=a.seed))
    dsh = (rsh - pubsh.loc[rsh.index]).abs().max()
    assert dsh < 5e-4 and abs(tot - pubsh["__total_R2_with_all__"]) < 5e-4, dsh
    fph, ftot = BS.fast_shapley(y, v[published].to_numpy(dtype=float), a.n_orderings, a.seed)
    print(f"[ok] published 25-predictor Shapley reproduced with the pipeline function (max |diff| {dsh:.1e}; "
          f"{time.time()-t1:.0f}s); fast implementation agrees to {np.abs(fph - rsh.reindex(published).to_numpy()).max():.1e}")

    # ---------------------------------------------------------------- partial grid (parallel over models)
    models = list(defs)
    _V["v"] = v
    with Pool(2, initializer=lambda vv: _V.update(v=vv), initargs=(v,)) as pool:
        fits = dict(pool.map(_partial_job, [(m, defs[m]["preds"], a.n_boot, a.seed) for m in models]))
    long_rows, summ = [], []
    for m in models:
        dd = defs[m]; res = fits[m].copy()
        res.insert(0, "transformation", dd["transformation"]); res.insert(0, "category", dd["category"]); res.insert(0, "model", m)
        long_rows.append(res)
        r = res.set_index("predictor"); order = r["rho_partial"].sort_values(ascending=False)
        c, t = dd["ccol"], dd["tcol"]
        summ.append(dict(model=m, category=dd["category"], transformation=dd["transformation"],
                         cat_rho=r.at[c, "rho_partial"], cat_lo=r.at[c, "ci95_lo"], cat_hi=r.at[c, "ci95_hi"], cat_p=r.at[c, "p_perm"],
                         cat_rank_of_23=list(order.index).index(c) + 1, cat_strongest=bool(order.index[0] == c),
                         trans_rho=r.at[t, "rho_partial"], trans_lo=r.at[t, "ci95_lo"], trans_hi=r.at[t, "ci95_hi"], trans_p=r.at[t, "p_perm"],
                         trans_ci_excludes_zero=bool(r.at[t, "ci95_lo"] > 0 or r.at[t, "ci95_hi"] < 0)))
    pd.concat(long_rows).to_csv(os.path.join(OUT, "cooked_partial_grid_long.csv"), index=False)
    ps = pd.DataFrame(summ); ps.to_csv(os.path.join(OUT, "cooked_partial_grid_summary.csv"), index=False)
    print(f"\nPARTIAL SPEARMAN (n_boot={a.n_boot}, seed={a.seed}) ({time.time()-t0:.0f}s)")
    for _, r in ps.iterrows():
        print(f"  {r.model:42s} category {r.cat_rho:+.3f} [{r.cat_lo:+.3f},{r.cat_hi:+.3f}] rank {r.cat_rank_of_23:2d} | "
              f"transf {r.trans_rho:+.3f} [{r.trans_lo:+.3f},{r.trans_hi:+.3f}] p={r.trans_p:.3f}")
    ex = os.path.join(a.existing_dir, "partial_3x3_summary.csv")
    if os.path.exists(ex):
        e = pd.read_csv(ex).set_index("model"); n = ps.set_index("model").loc[e.index]
        print(f"  refit vs committed 3x3 partial summary: max |diff| cat_rho {np.abs(n.cat_rho-e.cat_rho).max():.1e}, "
              f"trans_rho {np.abs(n.trans_rho-e.trans_rho).max():.1e}, CI {np.abs(n[['trans_lo','trans_hi']].to_numpy()-e[['trans_lo','trans_hi']].to_numpy()).max():.1e}")

    # ---------------------------------------------------------------- Shapley grid (point estimates)
    long_rows, summ = [], []
    for m in models:
        dd = defs[m]; names = dd["preds"]
        ph, tot = BS.fast_shapley(y, v[names].to_numpy(dtype=float), a.n_orderings, a.seed)
        sh = pd.Series(ph, index=names); order = sh.sort_values(ascending=False)
        c, t = dd["ccol"], dd["tcol"]
        rival = sh.drop(c).idxmax()
        long_rows.append(pd.DataFrame({"model": m, "category": dd["category"], "transformation": dd["transformation"],
                                       "predictor": names, "shapley_delta_r2": ph, "total_r2": tot}))
        summ.append(dict(model=m, category=dd["category"], transformation=dd["transformation"], total_r2=tot,
                         top_predictor=order.index[0], cat_share=sh[c], cat_rank_of_23=list(order.index).index(c) + 1,
                         strongest_rival=rival, rival_share=sh[rival], cat_lead=sh[c] - sh[rival],
                         trans_share=sh[t], trans_rank_of_23=list(order.index).index(t) + 1))
    pd.concat(long_rows).to_csv(os.path.join(OUT, "cooked_shapley_grid_long.csv"), index=False)
    ss = pd.DataFrame(summ); ss.to_csv(os.path.join(OUT, "cooked_shapley_grid_summary.csv"), index=False)
    print(f"\nSHAPLEY ({a.n_orderings} orderings, seed {a.seed}) ({time.time()-t0:.0f}s)")
    for _, r in ss.iterrows():
        print(f"  {r.model:42s} total R2 {r.total_r2:.3f} | category {r.cat_share:.4f} (rank {r.cat_rank_of_23}) lead {r.cat_lead:+.4f} over {r.strongest_rival} "
              f"| transf {r.trans_share:.4f} (rank {r.trans_rank_of_23})")
    ex = os.path.join(a.existing_dir, "shapley_3x3_summary.csv")
    if os.path.exists(ex):
        e = pd.read_csv(ex).set_index("model"); n = ss.set_index("model").loc[e.index]
        print(f"  refit vs committed 3x3 Shapley summary: max |diff| cat_share {np.abs(n.cat_share-e.cat_share).max():.1e}, trans_share {np.abs(n.trans_share-e.trans_share).max():.1e}")
    # pipeline-function spot check on one cooked model
    m = "WHO + Cooked (liberal)"; dd = defs[m]
    t1 = time.time()
    sp, _ = P3.shapley_one(v, dd["preds"], argparse.Namespace(n_orderings=a.n_orderings, seed=a.seed))
    fp = pd.concat(long_rows).query("model == @m").set_index("predictor")["shapley_delta_r2"]
    print(f"  pipeline-function spot check on {m}: max |diff| {np.abs(sp - fp.reindex(sp.index)).max():.1e} ({time.time()-t1:.0f}s)")

    # ---------------------------------------------------------------- univariate fits
    rng = np.random.default_rng(a.seed); n = len(y); rows = []
    for name, col in (("NOVA", "NOVA_ordinal"), ("Natural vs transformed", "natural_vs_transformed"),
                      ("Transformation score", "Transformation_score"),
                      ("Cooked (liberal)", "natural_vs_transformed_cooked"), ("Cooked (strict)", "natural_vs_transformed_cooked_strict")):
        x = v[col].to_numpy(dtype=float)
        rho = spearmanr(x, y)[0]
        bs = np.array([spearmanr(x[i], y[i])[0] for i in (rng.integers(0, n, n) for _ in range(1000))])
        perm = np.array([spearmanr(rng.permutation(x), y)[0] for _ in range(1000)])
        rows.append(dict(measure=name, rho=rho, ci95_lo=np.nanpercentile(bs, 2.5), ci95_hi=np.nanpercentile(bs, 97.5),
                         p_perm=float((np.abs(perm) >= abs(rho)).mean()), prop_pairs_different=float((x > 0).mean()) if col.startswith("natural") else np.nan))
    uni = pd.DataFrame(rows); uni.to_csv(os.path.join(OUT, "cooked_univariate.csv"), index=False)
    print("\nUNIVARIATE SPEARMAN with behavioural RDM (703 pairs)")
    print(uni.round(4).to_string(index=False))
    a_, b_ = v["natural_vs_transformed"].to_numpy(), v["natural_vs_transformed_cooked"].to_numpy()
    print(f"cooked (liberal) vs old NvT RDM on the 703 pairs: Spearman {spearmanr(a_, b_)[0]:.3f}, differ on {np.mean(a_ != b_):.1%} of pairs")
    print(f"\ndone ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
