"""
Is the Intuitive taxonomy's univariate advantage over WHO and Culinary reliable?
(reviewer point R1-4a, "circular selection of the category taxonomy")

WHY THIS EXISTS
---------------
The manuscript reports univariate fits to the behavioural RDM (fixed+dense, 703 pairs) of
Intuitive rho = 0.417, WHO rho = 0.388, Culinary rho = 0.385 (Results, "Individual models").
Choosing the best of several taxonomies and then comparing it with single continuous
dimensions tilts the comparison (winner's curse). If Intuitive's lead is itself reliable,
"lay categories fit better than expert WHO or culinary schemes" can be stated as a finding.
If not, the three schemes should be treated as equivalent operationalisations of category.
Companion to sensitivity_partial_3x3.py (24 Sept 2026 audit).

METHOD
------
Paired bootstrap of the difference in univariate Spearman fit, rho(Intuitive, emp) minus
rho(WHO or Culinary, emp), both taxonomies evaluated on the SAME resample each time:
  * pair bootstrap: resample the 703 item pairs with replacement (matches how the pipeline
    builds its CIs; anti-conservative because pairs share items);
  * item bootstrap: resample the 209 items with replacement and keep every observed pair
    between two distinct resampled items (Nili et al., 2014 style stimulus bootstrap;
    accounts for the dependence among pairs that share an item, so it is the one to report).
Reported: observed difference, 95% percentile CI, two-sided bootstrap p (twice the smaller
tail proportion of the bootstrap distribution on either side of zero). Also reported for the
WHO minus Culinary difference, and for each taxonomy versus the best continuous predictor
(healthiness) for context. Default 10,000 resamples, seed 2026.

DATA
----
Same inputs and item-order reconstruction as sensitivity_partial_3x3.py (imported from it),
including its assertion that the rebuilt vectors match the pipeline's pairwise columns.
Validation: observed univariate fits must reproduce model_fits.csv (fixed_dense/human) to
1e-6 for the three taxonomies, and to 5e-4 (i.e. at the 3 decimals reported) for healthiness,
whose difference-of-ratings vector carries near-ties that resolve differently across platforms
(24 Sept 2026 Linux run: 0.3889 either way, diff < 1e-4). See sensitivity_partial_3x3.py.

USAGE (from the repository root)
--------------------------------
    python sensitivity/sensitivity_taxonomy_difference.py [--n-boot 10000] [--seed 2026]

OUTPUT
------
sensitivity/sensitivity_outputs/taxonomy_difference.csv, also printed.
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sensitivity_partial_3x3 as base  # noqa: E402  (paths, item order, RDM rebuild)

FITS = os.path.join(base.AC, "compute_outputs", "stage2_model_evaluation", "fixed_dense", "human", "model_fits.csv")


def rank(a):
    return pd.Series(a).rank(method="average").to_numpy()


def spearman_cols(y, X):
    """Spearman rho of y with each column of X."""
    ry = rank(y); ry = (ry - ry.mean()) / ry.std()
    out = []
    for k in range(X.shape[1]):
        rx = rank(X[:, k]); rx = (rx - rx.mean()) / rx.std()
        out.append(float((ry * rx).mean()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    t0 = time.time()

    v = pd.read_csv(base.PAIRWISE)
    assert len(v) == 703
    labels = base.fixed_dense_labels()
    assert np.allclose(base.rdm_vector("food_classification", labels, v), v["food_classification"].to_numpy(float))
    v["Category_WHO_10"] = base.rdm_vector("Category_WHO_10", labels, v)
    v["Category_Culinary_9"] = base.rdm_vector("Category_Culinary_9", labels, v)

    models = {"Intuitive": "food_classification", "WHO": "Category_WHO_10",
              "Culinary": "Category_Culinary_9", "Healthiness": "human_healthiness"}
    names = list(models)
    y = v["emp"].to_numpy(float)
    X = np.column_stack([v[c].to_numpy(float) for c in models.values()])
    obs = dict(zip(names, spearman_cols(y, X)))

    pub = pd.read_csv(FITS).set_index("model")["spearman_r"]
    for n in ("Intuitive", "WHO", "Culinary", "Healthiness"):
        key = {"Intuitive": "food_classification", "WHO": "Category_WHO_10",
               "Culinary": "Category_Culinary_9", "Healthiness": "human_healthiness"}[n]
        tol = 5e-4 if n == "Healthiness" else 1e-6
        assert abs(obs[n] - pub[key]) < tol, f"{n} univariate fit not reproduced ({obs[n]:.4f} vs {pub[key]:.4f})"
    print("[ok] univariate fits reproduced: " + ", ".join(f"{n} {obs[n]:.3f}" for n in names))

    contrasts = [("Intuitive", "WHO"), ("Intuitive", "Culinary"), ("WHO", "Culinary"),
                 ("Intuitive", "Healthiness"), ("WHO", "Healthiness"), ("Culinary", "Healthiness")]
    idx = {n: k for k, n in enumerate(names)}

    rng = np.random.default_rng(a.seed)
    n_pairs = len(v)
    boot_pair = np.empty((a.n_boot, len(names)))
    for b in range(a.n_boot):
        s = rng.integers(0, n_pairs, n_pairs)
        boot_pair[b] = spearman_cols(y[s], X[s])

    # item bootstrap: lookup matrix of pair-row index among the 209 items
    n_items = 209
    look = -np.ones((n_items, n_items), dtype=int)
    look[v["i"].to_numpy(), v["j"].to_numpy()] = np.arange(n_pairs)
    look[v["j"].to_numpy(), v["i"].to_numpy()] = np.arange(n_pairs)
    iu, ju = np.triu_indices(n_items, k=1)
    boot_item = np.empty((a.n_boot, len(names)))
    n_used = np.empty(a.n_boot, dtype=int)
    for b in range(a.n_boot):
        s = rng.integers(0, n_items, n_items)
        si, sj = s[iu], s[ju]
        rows = look[si, sj]
        rows = rows[(si != sj) & (rows >= 0)]
        n_used[b] = len(rows)
        boot_item[b] = spearman_cols(y[rows], X[rows])

    out = []
    for m1, m2 in contrasts:
        d_obs = obs[m1] - obs[m2]
        for scheme, B in (("pair", boot_pair), ("item", boot_item)):
            d = B[:, idx[m1]] - B[:, idx[m2]]
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            out.append({"contrast": f"{m1} - {m2}", "bootstrap": scheme, "diff_obs": d_obs,
                        "ci95_lo": np.percentile(d, 2.5), "ci95_hi": np.percentile(d, 97.5), "p_boot": p,
                        "prop_diff_gt0": (d > 0).mean()})
    res = pd.DataFrame(out)
    os.makedirs(base.OUT, exist_ok=True)
    res.to_csv(os.path.join(base.OUT, "taxonomy_difference.csv"), index=False, float_format="%.4f")
    print(res.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print(f"\nitem bootstrap used a median of {int(np.median(n_used))} pairs per resample; "
          f"n_boot={a.n_boot}, seed={a.seed}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
