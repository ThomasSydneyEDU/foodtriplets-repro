"""Does 'becoming a different kind of food' (NvT / transformation score) predict similarity beyond cooking, and vice versa?
Sensitivity analysis only. Reads the pipeline data read-only; nothing is written outside sensitivity_outputs/.
(1) Head-to-head: partial Spearman and delta R2 (rank OLS) of each measure beyond the other, plus paired univariate
    differences, all with item-level bootstrap (resampling the 209 items; pair multiplicity k_i*k_j).
(2) Within-category pairs (same Intuitive category): do pairs differing in cooked status differ more in behaviour?
    Compared against NvT and score differences; item bootstrap; permutation of cooked labels within category.
"""
import os, sys, argparse, time
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sensitivity_shapley_item_bootstrap as BS
import sensitivity_partial_3x3 as P3
OUT = P3.OUT

def rk(M): return np.apply_along_axis(rankdata, 0, M)
def resid(y, C, icpt=True):
    C1 = np.column_stack([np.ones(len(y)), C]) if icpt else C; b = np.linalg.lstsq(C1, y, rcond=None)[0]; return y - C1 @ b
def partial(y, t, C):
    Y, T, CC = rk(y[:, None])[:, 0], rk(t[:, None])[:, 0], rk(C)
    a, b = resid(Y, CC, False), resid(T, CC, False)  # no intercept, as in the pipeline's _partial_correlation
    if a.std() == 0 or b.std() == 0: return np.nan
    return float(np.corrcoef(a, b)[0, 1])
def r2(y, X):
    Y, XX = rk(y[:, None])[:, 0], rk(X); e = resid(Y, XX); return 1 - (e @ e) / (((Y - Y.mean()) ** 2).sum())
def pct(x): x = np.asarray(x, float); return np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=1000); ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--perm", type=int, default=2000); a = ap.parse_args(); t0 = time.time()
    v, published, controls, defs = BS.load_data()
    y = v["emp"].to_numpy(float); i = v["i"].to_numpy(); j = v["j"].to_numpy()
    base = controls + ["food_classification"]
    # reproduction: partial of liberal cooked given controls + Intuitive should equal the grid value
    g = pd.read_csv(os.path.join(OUT, "cooked_partial_grid_summary.csv")).set_index("model")
    chk = partial(y, v["natural_vs_transformed_cooked"].to_numpy(float), v[base].to_numpy(float))
    assert abs(chk - g.at["Intuitive + Cooked (liberal)", "trans_rho"]) < 5e-4, chk
    print(f"[ok] reproduces grid value ({chk:+.4f})")
    cols = {"NvT": "natural_vs_transformed", "Score": "Transformation_score", "NOVA": "NOVA_ordinal"}
    cook = {"Cooked (liberal)": "natural_vs_transformed_cooked", "Cooked (strict)": "natural_vs_transformed_cooked_strict"}
    rows_by_b = [BS.item_resample_rows(i, j, b, a.seed) for b in range(a.B)]
    out = []
    for bname, bcols in (("controls + Intuitive", base), ("controls only (no category)", controls)):
        for ck, cc in cook.items():
            for mk, mc in cols.items():
                def stats(rows):
                    yy = y[rows]; B_ = v[bcols].to_numpy(float)[rows]; m = v[mc].to_numpy(float)[rows]; c = v[cc].to_numpy(float)[rows]
                    Bm = np.column_stack([B_, m]); Bc = np.column_stack([B_, c])
                    return dict(
                        m_beyond_c=partial(yy, m, Bc), c_beyond_m=partial(yy, c, Bm),
                        dR2_m_beyond_c=r2(yy, np.column_stack([Bc, m])) - r2(yy, Bc),
                        dR2_c_beyond_m=r2(yy, np.column_stack([Bm, c])) - r2(yy, Bm),
                        uni_diff=spearmanr(m, yy)[0] - spearmanr(c, yy)[0])
                pt = stats(np.arange(len(y)))
                bs = pd.DataFrame([stats(r) for r in rows_by_b])
                for k, val in pt.items():
                    lo, hi = pct(bs[k]); out.append(dict(base=bname, cooked=ck, measure=mk, stat=k, point=val, lo=lo, hi=hi))
        print(f"  head-to-head base '{bname}' done ({time.time()-t0:.0f}s)")
    H = pd.DataFrame(out); H.to_csv(os.path.join(OUT, "kind_vs_cooking_headtohead.csv"), index=False)

    # ---------------- within-category pairs
    labels = P3.fixed_dense_labels(); code = pd.read_csv(BS.CODING).set_index("filename")
    same = (v["food_classification"].to_numpy(float) == 0)
    def cls(col): return np.array([1.0 if code[col][l] == "Transformed" else 0.0 for l in labels])
    cat_item = np.full(len(labels), -1)
    # item categories from the pair structure: use the original info table
    info = pd.read_csv(os.path.join(P3.AC, "reference", "Foodpictures_information_dynamic.csv")).set_index("filename" if "filename" in pd.read_csv(os.path.join(P3.AC, "reference", "Foodpictures_information_dynamic.csv"), nrows=1).columns else 0)
    catcol = "Category_Intuitive_7"
    cats = np.array([info.at[l, catcol] if l in info.index else None for l in labels])
    assert all(c is not None for c in cats), "label missing in info table"
    assert np.array_equal(same, cats[i] == cats[j]), "category mismatch with pair RDM"
    measures = {"Cooked (liberal)": cls("natural_vs_transformed_cooked"), "Cooked (strict)": cls("natural_vs_transformed_cooked_strict"),
                "NvT": np.array([1.0 if info.at[l, "natural_vs_transformed"] == "Transformed" else 0.0 for l in labels])}
    score = np.array([float(info.at[l, "Transformation_score"]) for l in labels])
    print(f"within-category pairs: {same.sum()} of {len(y)}; categories present: {pd.Series(cats[i][same]).value_counts().to_dict()}")
    def within(rows, lab):
        s = same[rows]; yy = y[rows][s]; ii, jj = i[rows][s], j[rows][s]
        res = {}
        for k, x in lab.items():
            d = np.abs(x[ii] - x[jj]); res[k + " mean diff (different - same)"] = yy[d == 1].mean() - yy[d == 0].mean() if (d == 1).any() and (d == 0).any() else np.nan
            res[k + " rho"] = spearmanr(d, yy)[0]
        d = np.abs(score[ii] - score[jj]); res["Score rho"] = spearmanr(d, yy)[0]
        for k in ("Cooked (liberal)", "Cooked (strict)"):
            res[f"Score rho - {k} rho"] = res["Score rho"] - res[k + " rho"]; res[f"NvT rho - {k} rho"] = res["NvT rho"] - res[k + " rho"]
        return res
    pt = within(np.arange(len(y)), measures); bs = pd.DataFrame([within(r, measures) for r in rows_by_b])
    W = pd.DataFrame([dict(stat=k, point=pt[k], lo=pct(bs[k])[0], hi=pct(bs[k])[1]) for k in pt]); W["n_within_pairs"] = int(same.sum())
    # permutation: shuffle cooked labels among items within category
    rng = np.random.default_rng(a.seed); obs = pt["Cooked (liberal) mean diff (different - same)"]; cnt = 0
    for _ in range(a.perm):
        x = measures["Cooked (liberal)"].copy()
        for c in np.unique(cats): idx = np.where(cats == c)[0]; x[idx] = rng.permutation(x[idx])
        d = np.abs(x[i][same] - x[j][same]); yy = y[same]
        cnt += abs(yy[d == 1].mean() - yy[d == 0].mean()) >= abs(obs)
    W.loc[len(W)] = dict(stat="perm p (cooked liberal mean diff, labels shuffled within category)", point=(cnt + 1) / (a.perm + 1), lo=np.nan, hi=np.nan, n_within_pairs=int(same.sum()))
    W.to_csv(os.path.join(OUT, "kind_vs_cooking_within_category.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nHEAD-TO-HEAD (item bootstrap)\n", H.round(3).to_string(index=False)); print("\nWITHIN-CATEGORY\n", W.round(3).to_string(index=False)); print(f"done {time.time()-t0:.0f}s")
if __name__ == "__main__": main()
