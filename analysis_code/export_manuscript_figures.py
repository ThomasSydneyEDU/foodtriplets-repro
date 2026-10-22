#!/usr/bin/env python3
"""Copy the figures that appear in the manuscript out of analysis_code/results/.

Why this exists
---------------
Until September 2026 there was no record of which results file fed which figure
in the paper. Figures were copied across by hand, the mapping lived in someone's
head, and it went wrong: the four Appendix A classification tables turned out to
be rendered from an archived draft, and Figures 3, 4, 6, B1 and B3 sat at June
vintage in the docx while the text quoted September numbers. This script makes
the mapping executable, so it cannot drift again.

Layout
------
manuscript/figures/                figures in the body of the Word document
manuscript/figures_supplementary/  Appendix B and supplementary figures
manuscript/figures/_source/        editable source for Figure 1 (not generated)

Two figures are deliberately absent from the mapping because no analysis
produces them, so a rerun cannot replace them:

  Figure 1   the paradigm diagram; source artwork is in figures/_source/
  Figure S1  the 350-item stimulus montage; the stimulus set does not change

Note that these folders record what the last run produced, not what is currently
embedded in the .docx. The two diverge whenever a figure has been regenerated
but not yet swapped into the document.

Usage
-----
    python export_manuscript_figures.py            copy, overwriting
    python export_manuscript_figures.py --check    report only, copy nothing
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

AC = Path(__file__).resolve().parent
REPO = AC.parent
RESULTS = AC / "results"
BODY = REPO / "manuscript" / "figures"
SUPP = REPO / "manuscript" / "figures_supplementary"

# (destination folder, published name, source path under analysis_code/results/)
#
# The source for each row is the output of the Phase 4 step whose run_step label
# names that figure; see paper_master_analysis_run.sh. Keep the two in step.
MAPPING: list[tuple[Path, str, str]] = [
    # --- body of the Word document -------------------------------------------
    (BODY, "Figure2.png",
     "stage1_behavioural_space/stimuli_mds_combined_empirical.png"),
    (BODY, "Figure3.png",
     "stage2_model_visualisation/human_visual_on_controls_off_contrasts_on/model_mds_2d.png"),
    (BODY, "Figure4.png",
     "stage3_model_evaluation_plots/fixed_dense/mixed/fig_model_scan_univariate.png"),
    (BODY, "Figure5.png",
     "stage5_family_generalization_plots/fixed_dense/fig_family_cv_rho_compact.png"),
    (BODY, "Figure6.png",
     "stage6_variance_attribution_plots/fixed_dense/human/fig_shapley_butterfly.png"),
    # --- Appendix B and supplementary ----------------------------------------
    (SUPP, "FigureB1.png",
     "stage3_model_evaluation_plots/filled/mixed/fig_model_scan_univariate.png"),
    (SUPP, "FigureB2.png",
     "stage5_family_generalization_plots/filled/fig_family_cv_rho_compact.png"),
    (SUPP, "FigureB3.png",
     "stage6_variance_attribution_plots/filled/human/fig_shapley_butterfly.png"),
]

NOT_GENERATED = [
    (BODY, "Figure1.png", "paradigm diagram; edit figures/_source/Figure1.ai"),
    (SUPP, "figure_s1_stimulus_set_montage.png", "stimulus montage; the set does not change"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="Report what would be copied and exit without writing.")
    args = ap.parse_args()

    BODY.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(parents=True, exist_ok=True)

    copied, missing = 0, []
    print(f"{'figure':16s} {'->':2s} {'destination':24s} source")
    for dest_dir, name, rel in MAPPING:
        src = RESULTS / rel
        where = dest_dir.relative_to(REPO)
        if not src.exists():
            missing.append((name, rel))
            print(f"  {name:14s} !! MISSING          {rel}")
            continue
        if not args.check:
            shutil.copy2(src, dest_dir / name)
            copied += 1
        print(f"  {name:14s} -> {str(where):22s} {rel}")

    for dest_dir, name, why in NOT_GENERATED:
        present = (dest_dir / name).exists()
        mark = "present" if present else "ABSENT"
        print(f"  {name:14s} == {str(dest_dir.relative_to(REPO)):22s} not generated ({why}) [{mark}]")

    if missing:
        print(f"\n[WARN] {len(missing)} figure(s) had no source. Did Phase 4 run to completion?")
        for name, rel in missing:
            print(f"        {name}: expected results/{rel}")
        return 1
    if args.check:
        print("\n[OK] --check: every mapped source exists. Nothing written.")
    else:
        print(f"\n[OK] Copied {copied} figure(s) into manuscript/figures and manuscript/figures_supplementary.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
