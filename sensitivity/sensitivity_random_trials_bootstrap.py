"""
Under uniform random triplet sampling, do the category schemes reliably outperform the
best continuous dimension? (reviewer points R1-4a and R2-3)

WHY THIS EXISTS
---------------
In the primary fixed+dense subset (703 pairs) the three category schemes and healthiness fit
the behavioural RDM equivalently (item bootstrap, sensitivity_taxonomy_difference.py). In the
random-trials-only check (Supplementary Table S3 after the 24 Sept 2026 renumbering; formerly S1; sensitivity_trial_composition.py) the
taxonomies fit at rho = 0.391-0.399 and healthiness at 0.336, i.e. a gap about twice the size of
the between-taxonomy gaps. Random triplets are drawn uniformly from all 350 items, so their
composition cannot favour any organising dimension (the R2-3 argument). This script tests
whether that gap, and the between-taxonomy gaps, are reliable. Written 24 Sept 2026 audit.

METHOD
------
* Behavioural RDM: random-pool trials only, participants passing the published exclusion rule,
  pairs with >= 2 observations: exactly the random-only RDM of sensitivity_trial_composition.py
  (its read_trials() and build() are imported, not reimplemented).
* Model vectors are looked up BY ITEM NAME. (Caution recorded 24 Sept 2026:
  sensitivity_trial_composition.py indexes model RDMs by row position, assuming they share
  triplet_rdm_by_filename.npz's order. That holds for every model it uses, verified, but 18
  files in model_rdms_canon, including the visual RDMs and rdm_concept_natural_vs_transformed,
  are stored in a different order. Do not add those models to that script without fixing it.)
* Validation: univariate fits must reproduce the random-only column of
  sensitivity_outputs/trial_composition.csv to 1e-4 (that file is rounded to 4 dp).
* Paired item bootstrap (default 10,000 resamples, seed 2026): resample the 350 items with
  replacement, keep every qualifying pair between two distinct resampled items, recompute each
  model's Spearman fit, take differences. 95% percentile CI and two-sided bootstrap p.
* Contrasts: each taxonomy minus the best continuous (human-rated) predictor on random trials
  (identified from the data, not assumed), each taxonomy minus healthiness, and the three
  between-taxonomy differences.

USAGE (from the repository root)
--------------------------------
    python sensitivity/sensitivity_random_trials_bootstrap.py [--n-boot 10000] [--seed 2026]

OUTPUT
------
sensitivity_outputs/random_trials_bootstrap.csv (contrasts) and
sensitivity_outputs/random_trials_fits.csv (all univariate fits on random trials), also printed.
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sensitivity_trial_composition as tc  # noqa: E402  read_trials(), build(), paths, MODELS

TAX = {"Intuitive": "food_classification", "WHO": "Category_WHO_10", "Culinary": "Category_Culinary_9"}


def rank(a):
    return pd.Series(a).rank(method="average").to_numpy()


def zr(a):
    r = rank(a); return (r - r.mean()) / r.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-obs", type=int, default=2, help="as in sensitivity_trial_composition.py")
    a = ap.parse_args()
    t0 = time.time()

    prim = np.load(os.path.join(tc.DERIV, "triplet_rdm_by_filename.npz"), allow_pickle=True)
    labels = [os.path.basename(str(x)) for x in prim["labels"]]
    index = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    pools, kept = tc.read_trials(index)
    D, pres = tc.build(pools["random"], index, n)
    iu, ju = np.triu_indices(n, 1)
    y_all = D[iu, ju]; c_all = pres[iu, ju]
    ok = np.isfinite(y_all) & (c_all >= a.min_obs)
    pi, pj = iu[ok], ju[ok]
    y = y_all[ok]
    print(f"{kept} participants; {len(pools['random'])} random trials; {ok.sum()} pairs with >= {a.min_obs} obs")

    X = {}
    for m in tc.MODELS:
        p = os.path.join(tc.CANON, f"rdm_{m}.npz")
        d = np.load(p, allow_pickle=True)
        li = {os.path.basename(str(l)): k for k, l in enumerate(d["labels"])}
        perm = np.array([li[l] for l in labels])
        M = d["data"].astype(float)[np.ix_(perm, perm)]
        X[m] = M[pi, pj]
    names = list(X)
    zy = zr(y)
    fits = {m: float((zy * zr(X[m])).mean()) for m in names}

    ref = pd.read_csv(os.path.join(tc.OUTDIR, "trial_composition.csv")).set_index("model")["rho_random_only"]
    for m in names:
        assert abs(fits[m] - ref[m]) < 1e-4, f"{m}: {fits[m]:.4f} vs published {ref[m]:.4f}"
    print("[ok] random-only fits reproduce trial_composition.csv")

    cont = [m for m in names if m.startswith("human_")]
    best = max(cont, key=lambda m: fits[m])
    print("fits: " + ", ".join(f"{m} {fits[m]:.3f}" for m in sorted(names, key=lambda m: -fits[m])[:8]))
    print(f"best continuous predictor on random trials: {best} ({fits[best]:.3f})")

    contrasts = [(t, best) for t in TAX] + ([(t, "human_healthiness") for t in TAX] if best != "human_healthiness" else [])
    contrasts += [("Intuitive", "WHO"), ("Intuitive", "Culinary"), ("WHO", "Culinary")]
    col = lambda k: TAX.get(k, k)
    need = sorted({col(x) for c in contrasts for x in c})

    look = -np.ones((n, n), dtype=np.int64)
    look[pi, pj] = np.arange(len(y)); look[pj, pi] = np.arange(len(y))
    rng = np.random.default_rng(a.seed)
    B = {m: np.empty(a.n_boot) for m in need}
    for b in range(a.n_boot):
        s = rng.integers(0, n, n)
        si, sj = s[iu], s[ju]
        rows = look[si, sj]
        rows = rows[(si != sj) & (rows >= 0)]
        zyb = zr(y[rows])
        for m in need:
            B[m][b] = (zyb * zr(X[m][rows])).mean()
        if b and b % 1000 == 0:
            print(f"  {b} resamples ({time.time()-t0:.0f}s)", flush=True)

    out = []
    for m1, m2 in contrasts:
        d = B[col(m1)] - B[col(m2)]
        out.append({"contrast": f"{m1} - {m2}", "diff_obs": fits[col(m1)] - fits[col(m2)],
                    "ci95_lo": np.percentile(d, 2.5), "ci95_hi": np.percentile(d, 97.5),
                    "p_boot": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())),
                    "prop_diff_gt0": (d > 0).mean()})
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(tc.OUTDIR, "random_trials_bootstrap.csv"), index=False)
    pd.Series(fits, name="rho_random_only").sort_values(ascending=False).to_csv(os.path.join(tc.OUTDIR, "random_trials_fits.csv"))
    print(res.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print(f"n_boot={a.n_boot}, seed={a.seed}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
