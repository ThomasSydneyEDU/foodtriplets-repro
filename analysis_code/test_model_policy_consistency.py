"""
Policy consistency check for the model set.

Added 2026-09-10 on revision/transformation-models.

WHY THIS EXISTS
---------------
Model membership used to be declared independently in five places, and they had
drifted. The failure mode was silent: a predictor could be written into
pairwise_vectors.csv by compute_model_evaluation and then filtered straight back
out of the Shapley by compute_variance_attribution, producing a complete-looking
attribution table with a predictor quietly missing. Nothing errored.

Membership now comes from reference/policy_defaults.py. This script asserts that
every consumer still agrees with it, so the next person to add a model finds out
immediately if they have missed a filter.

Deliberately dependency-light (no scipy, no matplotlib) so it can run anywhere.

PORTABILITY NOTE. When checking predictor counts by hand, use extended regex:
    head -1 <pairwise_vectors.csv> | tr ',' '\n' | grep -Ev '^(i|j|emp|used_in)' | wc -l
BSD grep (macOS) does not treat \| as alternation in basic regex, so a GNU-style
pattern silently filters nothing and inflates the count by 3. Expected count on this
branch is 25: 10 human + 11 visual + food_classification + 3 transformation measures.

    python analysis_code/test_model_policy_consistency.py
"""
from __future__ import annotations
import os, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from reference.policy_defaults import (  # noqa: E402
    TAXONOMY_MODELS, TRANSFORMATION_MODELS, CONCEPTUAL_SUPPLEMENTARY,
    ALWAYS_INCLUDE_MODELS, EXCLUDE_MODELS,
)

failures: list[str] = []
def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)

print("\n1. policy_defaults internal consistency")
check(set(CONCEPTUAL_SUPPLEMENTARY) == set(TAXONOMY_MODELS) | set(TRANSFORMATION_MODELS),
      "CONCEPTUAL_SUPPLEMENTARY == TAXONOMY_MODELS + TRANSFORMATION_MODELS")
check(set(ALWAYS_INCLUDE_MODELS) == set(CONCEPTUAL_SUPPLEMENTARY),
      "ALWAYS_INCLUDE_MODELS == CONCEPTUAL_SUPPLEMENTARY")
check(not (set(ALWAYS_INCLUDE_MODELS) & set(EXCLUDE_MODELS)),
      "no model is both always-included and excluded")

print("\n2. RDM files exist on disk for every policy model")
canon = os.path.join(HERE, "derivatives", "model_rdms_canon")
have = {os.path.basename(f)[4:-4] for f in glob.glob(os.path.join(canon, "rdm_*.npz"))}
# food_classification is the runtime alias for Category_Intuitive_7; either file satisfies it.
for m in CONCEPTUAL_SUPPLEMENTARY:
    ok = m in have or (m == "food_classification" and "Category_Intuitive_7" in have)
    check(ok, f"rdm_{m}.npz present")

print("\n3. analysis_helpers.select_models() returns every policy model")
import analysis_helpers as H  # noqa: E402
cat = H.build_model_catalog()
for m in CONCEPTUAL_SUPPLEMENTARY:
    spec = cat.get(m)
    check(spec is not None and spec.family == "conceptual",
          f"{m} in catalog with family='conceptual' (drives Fig 3 colour/marker)")
sel = {s.key for s in H.select_models(source_mode="human", include_supplementary_contrasts=True)}
for m in CONCEPTUAL_SUPPLEMENTARY:
    check(m in sel, f"{m} survives select_models() (gates Fig 3 and variance attribution)")

print("\n4. compute_family_generalization families and blocks")
try:
    import compute_family_generalization as G  # noqa: E402
    for m in TRANSFORMATION_MODELS:
        check(m in G.FAMILIES, f"{m} has a FAMILIES entry")
    check("transformation" in G.FAMILIES, "combined 'transformation' family exists")
    check(set(G.FAMILIES.get("transformation", [])) ==
          {f"rdm_{m}" for m in TRANSFORMATION_MODELS},
          "'transformation' family holds exactly the three transformation RDMs")
    block_keys = {b[0] for b in G.BLOCKS}
    for need in ("human_full+transformation",
                 "human_full+food_classification+transformation",
                 "human_full+food_classification+natural_vs_transformed",
                 "human_full+food_classification+Transformation_score"):
        check(need in block_keys, f"BLOCKS contains '{need}'")
except ImportError as e:
    print(f"  SKIP  compute_family_generalization not importable here ({e})")

print("\n5. Fig 5 plotting script renders every computed block")
# results5 filters with `order = [k for k in _BLOCK_ORDER if k in present]`, so a
# block computed by compute_family_generalization but missing from _BLOCK_ORDER is
# silently dropped from the figure. This is the check that catches that.
try:
    import compute_family_generalization as G5  # noqa: E402
    import results5_family_generalization_plots as R5  # noqa: E402
    computed = [b[0] for b in G5.BLOCKS]
    order = set(R5._BLOCK_ORDER)
    skip = getattr(R5, "_INTENTIONALLY_NOT_PLOTTED", set())
    missing = [k for k in computed if k not in order and k not in skip]
    check(not missing,
          "every computed block is either plotted or explicitly declared "
          f"not-plotted (unaccounted for: {missing})")
    stale = [k for k in skip if k not in set(computed)]
    check(not stale,
          f"_INTENTIONALLY_NOT_PLOTTED has no stale entries (stale: {stale})")
    ungrouped = [k for k in R5._BLOCK_ORDER if k not in R5._BLOCK_GROUP]
    check(not ungrouped,
          f"every ordered block has a colour group (missing: {ungrouped})")
    unknown = [g for g in R5._BLOCK_GROUP.values() if g not in R5._GROUP_COLORS]
    check(not unknown,
          f"every colour group has a palette entry (missing: {sorted(set(unknown))})")
except ImportError as e:
    print(f"  SKIP  results5/compute_family_generalization not importable here ({e})")

print("\n6. figure grouping (results3) resolves every policy model")
try:
    import results3_model_evaluation_plots as R3  # noqa: E402
    for m in CONCEPTUAL_SUPPLEMENTARY:
        g = R3._model_group_with_policy(m, {"dummy": True})
        check(g == "Conceptual (supp)", f"{m} groups as 'Conceptual (supp)', got '{g}'")
except ImportError as e:
    print(f"  SKIP  results3 not importable here ({e})")

print("\n" + ("ALL CHECKS PASSED" if not failures else
              f"{len(failures)} CHECK(S) FAILED:\n  - " + "\n  - ".join(failures)))
sys.exit(1 if failures else 0)
