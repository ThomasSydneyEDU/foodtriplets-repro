import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

# 1. Build primary empirical target RDM (fixed+dense)
summary_path = "analysis_code/derivatives/summary.csv"
print(f"Loading summary from {summary_path}...")
df_raw = pd.read_csv(summary_path, low_memory=False)

# Filter to fixed or dense
def parse_bool(s):
    def _c(v):
        if pd.isna(v): return False
        if isinstance(v, bool): return v
        if isinstance(v, (int, np.integer)): return int(v) != 0
        if isinstance(v, (float, np.floating)): return float(v) != 0.0
        return str(v).strip().lower() in {"1", "1.0", "true", "t", "yes", "y", "on"}
    return s.fillna(False).apply(_c).astype(bool)

is_fixed = parse_bool(df_raw["is_fixed"])
is_dense = parse_bool(df_raw["is_dense"])
df = df_raw[(is_fixed | is_dense)].copy()

# Get unique labels
labs_set = set()
for col in ("StimA_file", "StimB_file", "StimC_file", "choice_file"):
    if col in df.columns:
        labs_set.update(os.path.basename(str(v)) for v in df[col].dropna().astype(str) if str(v).strip())
labs = sorted(labs_set)
n = len(labs)
idx = {fn: i for i, fn in enumerate(labs)}

sim = np.zeros((n, n), dtype=float)
cnt = np.zeros((n, n), dtype=float)

for _, row in df.iterrows():
    if pd.isna(row.get("choice_file")): continue
    a  = os.path.basename(str(row["StimA_file"]))
    b  = os.path.basename(str(row["StimB_file"]))
    c  = os.path.basename(str(row["StimC_file"]))
    ch = os.path.basename(str(row["choice_file"]))
    if a not in idx or b not in idx or c not in idx: continue
    ia, ib, ic = idx[a], idx[b], idx[c]
    if   ch == a:  pi, pj = sorted((ib, ic))
    elif ch == b:  pi, pj = sorted((ia, ic))
    elif ch == c:  pi, pj = sorted((ia, ib))
    else: continue
    sim[pi, pj] += 1.0
    for p1, p2 in ((ia, ib), (ia, ic), (ib, ic)):
        if p1 != p2:
            u, v = sorted((p1, p2))
            cnt[u, v] += 1.0

sim += sim.T
cnt += cnt.T
with np.errstate(invalid="ignore", divide="ignore"):
    psim = np.where(cnt > 0, sim / cnt, np.nan)
D_emp = 1.0 - psim

# Mask where counts > 0
has_obs = cnt > 0
iu_obs = np.where(has_obs & np.triu(np.ones((n, n)), k=1).astype(bool))

# Extract empirical dissimilarities
vec_emp = D_emp[iu_obs]

# 2. Load Completed RDMs
print("Loading SPoSE completed RDMs...")
full_data = np.load("analysis_code/derivatives/triplet_rdm_completed_nn_seed20.npz", allow_pickle=True)
rand_data = np.load("analysis_code/derivatives/triplet_rdm_completed_nn_seed20_random_only.npz", allow_pickle=True)

D_full_350 = full_data["D"]
D_rand_350 = rand_data["D"]

labels_350 = list(full_data["labels"])
labels_rand_350 = list(rand_data["labels"])

assert labels_350 == labels_rand_350, "Full and random-only RDM label orderings are not identical!"

# Sub-index the 350-item RDMs to the 209 fixed+dense items in 'labs'
idx_350 = {fn: i for i, fn in enumerate(labels_350)}
fd_to_350 = [idx_350[fn] for fn in labs]

D_full_fd = D_full_350[np.ix_(fd_to_350, fd_to_350)]
D_rand_fd = D_rand_350[np.ix_(fd_to_350, fd_to_350)]

# Extract dissimilarities for observed pairs
vec_full = D_full_fd[iu_obs]
vec_rand = D_rand_fd[iu_obs]

# Compute correlations against empirical target
rho_full, _ = spearmanr(vec_full, vec_emp)
rho_rand, _ = spearmanr(vec_rand, vec_emp)

# Cross-correlate full vs random RDMs on full 350x350
iu_350 = np.triu_indices(350, k=1)
rho_cross, _ = spearmanr(D_full_350[iu_350], D_rand_350[iu_350])

print("\n=== SPoSE ROBUSTNESS COMPARISON REPORT ===")
print(f"SPoSE Full-Dataset Embedding Fit on Fixed+Dense target: Spearman rho = {rho_full:.4f}")
print(f"SPoSE Random-Only Embedding Fit on Fixed+Dense target:  Spearman rho = {rho_rand:.4f}")
print(f"Fit difference (Full - Random-Only) on evaluation subset:           {rho_full - rho_rand:.4f}")
print(f"\nSPoSE RDM Cross-correlation (Full vs Random-Only on all 350 items):  Spearman rho = {rho_cross:.4f}")

# 3. Component-level stability check
print("\n=== COMPONENT-LEVEL STABILITY CHECK ===")
full_scores_path = "analysis_code/derivatives/nn_runs/seed_20/nn_scores.csv"
rand_scores_path = "analysis_code/derivatives/nn_runs/seed_20_random_only/nn_scores.csv"

# Check if paths exist
if os.path.exists(full_scores_path) and os.path.exists(rand_scores_path):
    print("Loading component scores for stability check...")
    df_full_scores = pd.read_csv(full_scores_path)
    df_rand_scores = pd.read_csv(rand_scores_path)
    
    # Ensure they are sorted by filename identically
    df_full_scores = df_full_scores.sort_values("filename").reset_index(drop=True)
    df_rand_scores = df_rand_scores.sort_values("filename").reset_index(drop=True)
    
    assert list(df_full_scores["filename"]) == list(df_rand_scores["filename"]), "Filename mismatch between score files!"
    
    # Get component columns
    full_cols = [c for c in df_full_scores.columns if c.startswith("nn_comp")]
    rand_cols = [c for c in df_rand_scores.columns if c.startswith("nn_comp")]
    
    # Compute cross-model correlation matrix
    W_full = df_full_scores[full_cols].values
    W_rand = df_rand_scores[rand_cols].values
    
    # Compute column-wise pearson correlation
    stability_scores = {}
    for i, c_full in enumerate(full_cols):
        max_r = -1.0
        vec_full = W_full[:, i]
        if np.std(vec_full) == 0:
            stability_scores[c_full] = 0.0
            continue
            
        for j, c_rand in enumerate(rand_cols):
            vec_rand = W_rand[:, j]
            if np.std(vec_rand) == 0:
                continue
            r, _ = pearsonr(vec_full, vec_rand)
            if r > max_r:
                max_r = r
        stability_scores[c_full] = max_r
        
    print("Stability scores computed.")
    
    # Merge with spose_interpretation_codified.csv
    interp_path = "analysis_code/results/stage7_nn_interpretation/spose_interpretation_codified.csv"
    if os.path.exists(interp_path):
        print(f"Loading {interp_path}...")
        df_interp = pd.read_csv(interp_path)
        
        # Add Stability (r) column
        df_interp["Stability (r)"] = df_interp["Component"].map(stability_scores).round(3)
        
        # full_components.csv - Sorted by Rank
        full_out_path = "analysis_code/results/stage7_nn_interpretation/full_components.csv"
        df_full_out = df_interp.sort_values("Rank")
        df_full_out.to_csv(full_out_path, index=False)
        print(f"Wrote full components to {full_out_path}")
        
        # core_components.csv - Filter Stability >= 0.70, Sort by Type then Rank
        core_out_path = "analysis_code/results/stage7_nn_interpretation/core_components.csv"
        df_core = df_interp[df_interp["Stability (r)"] >= 0.70].copy()
        
        df_core = df_core.sort_values(["Type", "Rank"], ascending=[True, True])
        df_core.to_csv(core_out_path, index=False)
        print(f"Wrote core components ({len(df_core)} dimensions) to {core_out_path}")
    else:
        print(f"Warning: {interp_path} not found. Cannot generate merged CSVs.")
else:
    print("Warning: Component score CSVs not found. Cannot perform component-level stability check.")

