"""
compute_nn_embedding_decomp.py
-----------------------------
Learn a non-negative sparse embedding from triplet (odd-one-out) choices and save
embedding component scores and diagnostics.

This script supports BOTH summary formats produced by preprocessing:

1) Split-column format (current pipeline; preferred):
   - derivatives/summary.csv must contain:
       StimA_file, StimB_file, StimC_file, choice_file

2) List format (legacy / jsPsych-style):
   - derivatives/summary.csv must contain:
       images_onscreen : a Python-like list string of 3 file names
       choice_image    : the selected (odd) image file name

Soft targets (optional):
- If you pass --soft_targets AND summary.csv includes p_ab, p_ac, p_bc, the model
  will use those pair-choice probabilities as a soft target distribution.

Optional inputs:
- derivatives/labels_lookup.csv
    If present, can be used for stable filename ordering / mapping.
- Canonical info CSV from `reference.policy_defaults.INFO_CSV`
    (fallback: analysis_code/reference/Foodpictures_information_canonical.csv)
    Used only for nicer labels in the per-component sorted lists.

Outputs
-------
Per seed (written to derivatives/nn_runs/seed_<SEED>/):
- nn_scores.csv                Component scores (one row per stimulus)
- nn_results.npz               Scores + kept dims (npz)
- nn_training.json             Run configuration + metadata
- training_history.json        Loss curve (and val loss if early stopping)
- final_model.pt               Final model weights
- sorted/nn_compXX.csv + .txt  Items sorted by component score
- completed_rdm_from_embedding.npz (if --write_completed_rdm)

Global completed RDM (if --write_completed_rdm):
- derivatives/triplet_rdm_completed_nn_seed<SEED>.npz

Aggregation (when multiple seeds are run and/or --aggregate is set):
- derivatives/nn_runs/nn_scores_all_seeds.csv
- derivatives/nn_runs/nn_training_runs_summary.csv

Common usage
------------
Single seed (official SPoSE defaults):
  python analysis_code/compute_nn_embedding_decomp.py \
    --seed 0 --dim 90 --epochs 500 --bs 100 --lr 0.001 --lmbda 0.007 \
    --prune_threshold 0.1 --write_completed_rdm

Publication-grade preset (recommended; runs seeds 0..4 and aggregates):
  python analysis_code/compute_nn_embedding_decomp.py --pubgrade

Notes
-----
- Requires: numpy, pandas, torch, scipy
- Uses the official SPoSE model (nn.Linear, no bias) with soft positivity penalty;
  sparsity encouraged via L1 regularisation scaled as (lmbda / n_items).
- Pruning uses the official SPoSE max-weight criterion (keep if max_weight > threshold).
- Completed RDM is computed from the learned embedding using cosine distance by
  default (see embedding_to_completed_rdm).
"""

import os
import json
import ast
import argparse
import random
from typing import List, Tuple, Dict

from pathlib import Path

try:
    from reference.policy_defaults import (
        INFO_CSV as POLICY_INFO_CSV,
        DERIVATIVES_DIR as POLICY_DERIVATIVES_DIR,
        SUMMARY_CSV as POLICY_SUMMARY_CSV,
        DEFAULT_NN_SEED as POLICY_DEFAULT_NN_SEED,
    )
except Exception:
    POLICY_INFO_CSV = None
    POLICY_DERIVATIVES_DIR = None
    POLICY_SUMMARY_CSV = None
    POLICY_DEFAULT_NN_SEED = 0

import numpy as np
import pandas as pd

# Torch is optional at import-time to allow printing a helpful message if missing
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except Exception as e:
    raise RuntimeError(
        "PyTorch is required. Install it in your virtual env:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from e

# Official SPoSE model and loss (vendored from https://github.com/ViCCo-Group/SPoSE)
# Always insert the absolute path of this file's directory first so the spose/
# sub-package is importable regardless of the working directory.
import sys as _sys
_here_abs = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
if _here_abs not in _sys.path:
    _sys.path.insert(0, _here_abs)
try:
    from spose import SPoSE, l1_regularization, trinomial_loss, encode_as_onehot
    from spose.model import get_embedding, get_nneg_dims
except ImportError as _e:
    raise ImportError(
        f"Could not import the 'spose' package from {_here_abs}/spose/. "
        "Ensure analysis_code/spose/__init__.py exists and PyTorch is installed."
    ) from _e

HERE = os.path.dirname(__file__)
DERIV = str(POLICY_DERIVATIVES_DIR) if POLICY_DERIVATIVES_DIR is not None else os.path.join(HERE, "derivatives")
os.makedirs(DERIV, exist_ok=True)

RUNS_ROOT = os.path.join(DERIV, "nn_runs")
os.makedirs(RUNS_ROOT, exist_ok=True)

# Helper import for aggregation
from glob import glob
from analysis_helpers import parse_images_list, basename_only, load_summary, build_filename_index

# ---------------------- Data utilities ---------------------- #

def embedding_to_completed_rdm(
    E: np.ndarray,
    eps: float = 1e-12,
    distance: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an embedding E [N,K] into a full similarity matrix and an RDM.

    We keep the dot-product similarity matrix S = E @ E^T for inspection (it matches
    the training objective), but **the completed RDM** should behave like a distance
    on the learned representation.

    By default we use cosine distance, which removes norm effects:
      - S_cos = cosine similarity between rows
      - D = 1 - S_cos

    This tends to preserve interpretable structure better than min-max scaling of
    raw dot-products.

    Args:
      E: [N,K] embedding (non-negative).
      eps: numerical stability.
      distance: "cosine" (default) or "dot_minmax" (legacy behavior).

    Returns:
      D [N,N] float32 with zeros on diagonal
      S [N,N] float32 (dot-product similarity)
    """
    E = np.asarray(E, dtype=np.float32)
    if E.ndim != 2:
        raise ValueError(f"E must be 2D [N,K], got shape {E.shape}")

    # Dot-product similarity (kept for reference / diagnostics)
    S = (E @ E.T).astype(np.float32)

    n = S.shape[0]
    if n < 2:
        D = np.zeros_like(S, dtype=np.float32)
        return D, S

    distance = str(distance).strip().lower()

    if distance == "dot_minmax":
        # Legacy behavior: min-max rescale dot-products into [0,1] then invert.
        mask_off = ~np.eye(n, dtype=bool)
        off = S[mask_off]
        smin = float(np.nanmin(off))
        smax = float(np.nanmax(off))
        denom = (smax - smin)
        if not np.isfinite(denom) or denom < eps:
            D = np.zeros_like(S, dtype=np.float32)
            np.fill_diagonal(D, 0.0)
            return D, S
        S01 = (S - smin) / denom
        S01 = np.clip(S01, 0.0, 1.0)
        D = (1.0 - S01).astype(np.float32)
        np.fill_diagonal(D, 0.0)
        return D, S

    if distance != "cosine":
        raise ValueError(f"Unknown distance='{distance}'. Use 'cosine' or 'dot_minmax'.")

    # Cosine distance completion (recommended)
    norms = np.sqrt((E * E).sum(axis=1, keepdims=True))
    norms = np.maximum(norms, eps)
    U = E / norms
    S_cos = (U @ U.T).astype(np.float32)
    S_cos = np.clip(S_cos, -1.0, 1.0)

    D = (1.0 - S_cos).astype(np.float32)
    np.fill_diagonal(D, 0.0)
    return D, S


def save_completed_rdm(run_dir: str, seed_value: int, filenames: List[str],
                       E_pruned: np.ndarray, E_full: np.ndarray, keep_idx: np.ndarray) -> tuple[str, str]:
    """Save completed RDM artifacts for a given seed.

    Writes into:
      - <run_dir>/completed_rdm_from_embedding.npz
      - <DERIV>/triplet_rdm_completed_nn_seed<seed>.npz
    """
    D_comp, S_comp = embedding_to_completed_rdm(E_pruned, distance="cosine")
    labels = np.array(list(filenames), dtype=object)

    out_local = os.path.join(run_dir, "completed_rdm_from_embedding.npz")
    np.savez_compressed(
        out_local,
        D=D_comp.astype(np.float32),
        data=D_comp.astype(np.float32),
        rdm=D_comp.astype(np.float32),
        S=S_comp.astype(np.float32),
        labels=labels,
        seed=int(seed_value),
        kept_dims=keep_idx.astype(np.int32),
        dim_pruned=int(E_pruned.shape[1]),
        dim_full=int(E_full.shape[1]),
    )

    suffix = "_random_only" if "random_only" in run_dir else ""
    out_global = os.path.join(DERIV, f"triplet_rdm_completed_nn_seed{int(seed_value)}{suffix}.npz")
    np.savez_compressed(
        out_global,
        D=D_comp.astype(np.float32),
        data=D_comp.astype(np.float32),
        rdm=D_comp.astype(np.float32),
        S=S_comp.astype(np.float32),
        labels=labels,
        seed=int(seed_value),
        kept_dims=keep_idx.astype(np.int32),
        dim_pruned=int(E_pruned.shape[1]),
        dim_full=int(E_full.shape[1]),
    )

    print(f"Saved completed RDM (seed {seed_value}) -> {out_local}")
    print(f"Saved completed RDM (seed {seed_value}) -> {out_global}")
    return out_local, out_global

def triplets_from_summary(summary_df: pd.DataFrame, idx_of: Dict[str, int], soft_targets: bool):
    """
    Convert each trial into (ia, ib, ic, y) for hard targets, where y ∈ {0,1,2}
      0 -> (a,b) chosen; 1 -> (a,c); 2 -> (b,c)
    Or into (ia, ib, ic, p_ab, p_ac, p_bc) for soft targets.
    """
    T = []
    # Soft targets only supported with images_onscreen path (p_ab/p_ac/p_bc columns)
    use_soft = soft_targets and {"p_ab", "p_ac", "p_bc"}.issubset(summary_df.columns)

    use_split = {"StimA_file", "StimB_file", "StimC_file", "choice_file"}.issubset(summary_df.columns)

    for _, r in summary_df.iterrows():
        if use_split:
            a_raw, b_raw, c_raw = str(r["StimA_file"]), str(r["StimB_file"]), str(r["StimC_file"])
            choice_raw = str(r["choice_file"]) if "choice_file" in r else None
            imgs = [basename_only(a_raw), basename_only(b_raw), basename_only(c_raw)]
        else:
            imgs = parse_images_list(r["images_onscreen"])
            choice_raw = basename_only(str(r["choice_image"])) if "choice_image" in r and not pd.isna(r["choice_image"]) else None
            imgs = [ basename_only(x) for x in imgs ]

        if len(imgs) != 3:
            continue
        a, b, c = imgs
        if not all(x in idx_of for x in (a, b, c)):
            continue
        ia, ib, ic = idx_of[a], idx_of[b], idx_of[c]

        if use_soft:
            pab, pac, pbc = float(r["p_ab"]), float(r["p_ac"]), float(r["p_bc"])
            s = pab + pac + pbc
            if s <= 0:
                continue
            T.append((ia, ib, ic, pab / s, pac / s, pbc / s))
        else:
            if choice_raw is None or choice_raw not in (a, b, c):
                continue
            if choice_raw == a:
                y = 2  # (b,c) is the similar pair
            elif choice_raw == b:
                y = 1  # (a,c)
            else:  # choice_raw == c
                y = 0  # (a,b)
            T.append((ia, ib, ic, y))
    return T, use_soft

# ---------------------- Model & Training (official SPoSE) ---------------------- #
# Uses SPoSE from spose/ (vendored from https://github.com/ViCCo-Group/SPoSE).
#
# Key differences from the previous custom implementation:
#   - Model:  nn.Linear(n_items, dim, bias=False)  [official SPoSE architecture]
#             Non-negativity encouraged via soft positivity penalty (0.01 * relu(-W).sum())
#             rather than Softplus activation.
#   - Loss:   c_entropy + 0.01 * pos_pen + (lmbda / n_items) * l1_pen
#             where c_entropy uses the manual softmax+CE from the official repo.
#   - Input:  one-hot encoded via identity matrix (official BatchGenerator pattern).
#   - Early stopping: linear-regression convergence criterion on training losses
#             (matches official SPoSE train.py).

try:
    from scipy.stats import linregress as _linregress
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def _triplets_to_spose_format(triplets_hard) -> np.ndarray:
    """Convert (ia, ib, ic, y) tuples to SPoSE integer triplets [pair1, pair2, odd].

    In SPoSE format the first two indices are the *similar* pair and the third
    is the odd-one-out, matching how the official BatchGenerator / encode_as_onehot
    constructs training batches.

    Our coding convention:
        y = 0  →  (a, b) are similar, c is odd  →  [ia, ib, ic]
        y = 1  →  (a, c) are similar, b is odd  →  [ia, ic, ib]
        y = 2  →  (b, c) are similar, a is odd  →  [ib, ic, ia]
    """
    out = []
    for ia, ib, ic, y in triplets_hard:
        if y == 0:
            out.append((ia, ib, ic))
        elif y == 1:
            out.append((ia, ic, ib))
        else:  # y == 2
            out.append((ib, ic, ia))
    return np.array(out, dtype=np.int64)  # shape (N, 3)


def train_embedding(
    triplets,
    n_items,
    dim: int = 90,
    lmbda: float = 0.007,
    epochs: int = 500,
    bs: int = 100,
    lr: float = 0.001,
    soft_targets: bool = False,
    seed: int = 0,
    device=None,
    early_stop: bool = True,
    window_size: int = 50,
    temperature: float = 1.0,
    distance_metric: str = "dot",
):
    """Train a SPoSE embedding using the official training procedure.

    Parameters match the official SPoSE train.py where applicable.
    Soft-target mode is retained for optional use with empirical choice probabilities.

    Returns
    -------
    model : SPoSE
    history : list[dict]
    best_val : None  (SPoSE uses train-loss convergence, not val loss)
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Build integer triplet tensor in SPoSE format ---
    if soft_targets:
        # Soft-target path: keep (ia, ib, ic) ordering and use probability targets
        # Note: soft targets are incompatible with the standard SPoSE loss;
        # we fall back to KL divergence as before.
        triplets_np = np.array([(ia, ib, ic) for ia, ib, ic, *_ in triplets], dtype=np.int64)
        probs_np    = np.array([(p0, p1, p2) for *_, p0, p1, p2 in triplets], dtype=np.float32)
        trip_t  = torch.from_numpy(triplets_np)   # (N, 3)
        probs_t = torch.from_numpy(probs_np)       # (N, 3)
    else:
        triplets_spose = _triplets_to_spose_format(triplets)  # (N, 3) ints
        trip_t = torch.from_numpy(triplets_spose)             # (N, 3)

    n_total = len(trip_t)

    # Shuffle
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    trip_t = trip_t[perm]
    if soft_targets:
        probs_t = probs_t[perm]

    # Identity matrix for one-hot encoding (official SPoSE pattern)
    I = torch.eye(n_items, device=device)

    # Batch indices
    n_batches = max(1, n_total // bs)

    # --- Model & optimizer (official SPoSE) ---
    model = SPoSE(in_size=n_items, out_size=dim, init_weights=True).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    t_tensor = torch.tensor(temperature, device=device)

    train_losses: list[float] = []
    history: list[dict] = []

    print(f"[SPoSE] n_items={n_items}  dim={dim}  lmbda={lmbda}  lr={lr}  bs={bs}  "
          f"epochs={epochs}  early_stop={early_stop}  window={window_size}  seed={seed}")

    for ep in range(1, epochs + 1):
        model.train()
        # Reshuffle each epoch (mimics official SPoSE BatchGenerator)
        perm_ep = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed + ep))
        trip_ep = trip_t[perm_ep]
        if soft_targets:
            prob_ep = probs_t[perm_ep]

        total_loss, n_seen = 0.0, 0
        for b in range(n_batches):
            batch_trips = trip_ep[b * bs : (b + 1) * bs].to(device)  # (B, 3)

            # One-hot encode — official SPoSE encode_as_onehot pattern
            batch_onehot = encode_as_onehot(I, batch_trips)           # (B*3, n_items)
            logits = model(batch_onehot)                               # (B*3, dim)
            anchor, positive, negative = torch.unbind(
                logits.view(-1, 3, dim), dim=1
            )                                                          # each (B, dim)

            # Loss: official SPoSE = c_entropy + 0.01*pos_pen + (lmbda/n_items)*l1_pen
            if soft_targets:
                target = prob_ep[b * bs : (b + 1) * bs].to(device)   # (B, 3)
                logp = torch.log_softmax(
                    torch.stack([
                        (anchor * positive).sum(1),
                        (anchor * negative).sum(1),
                        (positive * negative).sum(1),
                    ], dim=1), dim=1
                )
                c_entropy = nn.KLDivLoss(reduction="batchmean")(logp, target)
            else:
                c_entropy = trinomial_loss(anchor, positive, negative, t_tensor, distance_metric)

            W = model.fc.weight
            pos_pen        = torch.sum(F.relu(-W))           # official positivity penalty
            l1_pen         = l1_regularization(model)        # official L1
            complexity     = (lmbda / n_items) * l1_pen      # official scaling
            loss           = c_entropy + 0.01 * pos_pen + complexity

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * batch_trips.size(0)
            n_seen     += batch_trips.size(0)

        train_loss = total_loss / max(n_seen, 1)
        train_losses.append(train_loss)
        history.append({"epoch": ep, "train_loss": float(train_loss)})
        print(f"epoch {ep:03d}/{epochs} | loss {train_loss:.6f} | "
              f"nneg_dims={get_nneg_dims(model).numel()}")

        # --- Official SPoSE early stopping: linear regression on training window ---
        if early_stop and _HAVE_SCIPY and ep > window_size:
            window = train_losses[-window_size:]
            lmres  = _linregress(range(window_size), window)
            if lmres.slope > 0 or lmres.pvalue > 0.1:
                print(f"[SPoSE] Convergence criterion met at epoch {ep} "
                      f"(slope={lmres.slope:.2e}, p={lmres.pvalue:.3f}). Stopping.")
                break

    return model, history, None  # no val loss in official SPoSE


def _eval_val_loss(
    model: "SPoSE",
    val_trips_t: "torch.Tensor",
    n_items: int,
    temperature: float = 1.0,
    device: str = "cpu",
) -> float:
    """Compute trinomial loss on held-out validation triplets (no gradient).

    Only the cross-entropy term is used (no regularisation), so this measures
    pure generalisation — how well the embedding predicts unseen triplet choices.
    """
    model.eval()
    I = torch.eye(n_items, device=device)
    t_tensor = torch.tensor(temperature, device=device)
    dim = model.fc.weight.shape[0]
    with torch.no_grad():
        batch_onehot = encode_as_onehot(I, val_trips_t.to(device))
        logits = model(batch_onehot)
        anchor, positive, negative = torch.unbind(logits.view(-1, 3, dim), dim=1)
        val_loss = trinomial_loss(anchor, positive, negative, t_tensor).item()
    return val_loss


def search_lambda(
    triplets,
    n_items: int,
    lambda_grid,
    n_val_splits: int = 10,
    val_frac: float = 0.1,
    dim: int = 90,
    epochs: int = 500,
    bs: int = 100,
    lr: float = 0.001,
    early_stop: bool = True,
    window_size: int = 50,
    temperature: float = 1.0,
    seed_offset: int = 1000,
    device=None,
) -> tuple:
    """Select lambda via repeated hold-out cross-validation.

    For each value in *lambda_grid*, trains *n_val_splits* models on random
    train/val splits of the triplet data and averages the final validation loss
    (trinomial cross-entropy on held-out triplets, no regularisation).  The
    lambda with the lowest mean validation loss is returned.

    This matches the procedure described by Muttenthaler / Martin:
      - 10 random splits per lambda value
      - Hone in on the best order-of-magnitude first, then fine-grain search
        (e.g. 0.0038, 0.00385, …).

    Parameters
    ----------
    triplets : list of (ia, ib, ic, y) tuples
        Hard-target triplets (output of triplets_from_summary).
    n_items : int
        Number of unique stimuli.
    lambda_grid : list[float]
        Lambda values to evaluate, ordered from coarse to fine.
    n_val_splits : int
        Number of random train/val splits per lambda (Martin: 10).
    val_frac : float
        Fraction of triplets held out for validation (default 10 %).
    seed_offset : int
        Base random seed for the search splits.  Use a value that doesn't
        overlap with the final-model seed range (0–24) — default 1000.

    Returns
    -------
    best_lmbda : float
    results_df : pd.DataFrame
        Columns: lmbda, split, val_loss, n_nneg_dims, stopped_epoch
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert triplets once
    triplets_spose = _triplets_to_spose_format(triplets)   # (N, 3) int64 array
    n_total = len(triplets_spose)
    n_val = max(1, int(n_total * val_frac))

    print(f"\n[lambda-search] grid={lambda_grid}  n_splits={n_val_splits}  "
          f"val_frac={val_frac}  n_total={n_total}  n_val={n_val}")

    rows = []
    for lmbda in lambda_grid:
        split_losses = []
        for split_idx in range(n_val_splits):
            split_seed = seed_offset + split_idx

            # Deterministic random split
            rng = np.random.default_rng(split_seed)
            perm = rng.permutation(n_total)
            val_idx   = perm[:n_val]
            train_idx = perm[n_val:]

            train_t = torch.from_numpy(triplets_spose[train_idx])  # (N_train, 3)
            val_t   = torch.from_numpy(triplets_spose[val_idx])    # (N_val, 3)

            # Train on train split
            torch.manual_seed(split_seed)
            random.seed(split_seed)
            np.random.seed(split_seed)

            I = torch.eye(n_items, device=device)
            t_tensor = torch.tensor(temperature, device=device)
            model = SPoSE(in_size=n_items, out_size=dim, init_weights=True).to(device)
            opt   = torch.optim.Adam(model.parameters(), lr=lr)

            n_train = len(train_t)
            n_batches = max(1, n_train // bs)
            train_losses_inner: list[float] = []
            stopped_epoch = epochs

            for ep in range(1, epochs + 1):
                model.train()
                perm_ep = torch.randperm(
                    n_train, generator=torch.Generator().manual_seed(split_seed + ep)
                )
                trip_ep = train_t[perm_ep]

                total_loss, n_seen = 0.0, 0
                for b in range(n_batches):
                    batch = trip_ep[b * bs : (b + 1) * bs].to(device)
                    oh    = encode_as_onehot(I, batch)
                    out   = model(oh)
                    anc, pos, neg = torch.unbind(out.view(-1, 3, dim), dim=1)

                    c_ent  = trinomial_loss(anc, pos, neg, t_tensor)
                    W      = model.fc.weight
                    pos_p  = torch.sum(F.relu(-W))
                    l1_p   = l1_regularization(model)
                    loss   = c_ent + 0.01 * pos_p + (lmbda / n_items) * l1_p

                    opt.zero_grad(); loss.backward(); opt.step()
                    total_loss += loss.item() * batch.size(0)
                    n_seen     += batch.size(0)

                train_losses_inner.append(total_loss / max(n_seen, 1))

                if early_stop and _HAVE_SCIPY and ep > window_size:
                    w   = train_losses_inner[-window_size:]
                    res = _linregress(range(window_size), w)
                    if res.slope > 0 or res.pvalue > 0.1:
                        stopped_epoch = ep
                        break

            val_loss   = _eval_val_loss(model, val_t, n_items, temperature, device)
            n_nneg     = get_nneg_dims(model).numel()
            split_losses.append(val_loss)
            rows.append({
                "lmbda": lmbda, "split": split_idx,
                "val_loss": val_loss, "n_nneg_dims": n_nneg,
                "stopped_epoch": stopped_epoch,
            })
            print(f"  lmbda={lmbda:.6f}  split={split_idx:02d}  "
                  f"val_loss={val_loss:.5f}  nneg={n_nneg}  ep={stopped_epoch}")

        mean_val = float(np.mean(split_losses))
        print(f"  >>> lmbda={lmbda:.6f}  mean_val_loss={mean_val:.5f}")

    results_df = pd.DataFrame(rows)
    summary = (results_df.groupby("lmbda")["val_loss"]
                         .mean()
                         .reset_index()
                         .rename(columns={"val_loss": "mean_val_loss"}))
    best_lmbda = float(summary.loc[summary["mean_val_loss"].idxmin(), "lmbda"])
    print(f"\n[lambda-search] Best lambda: {best_lmbda}  "
          f"(mean val loss = {summary.loc[summary['mean_val_loss'].idxmin(), 'mean_val_loss']:.5f})")
    return best_lmbda, results_df


# ---------------------- Pruning & Saving ---------------------- #

def prune_dimensions(E: np.ndarray, threshold: float = 0.1, max_keep: int = None):
    """Keep dimensions where the maximum weight across items exceeds *threshold*.

    Mirrors the official SPoSE pruning criterion (``get_nneg_dims`` /
    ``remove_zeros`` in the ViCCo-Group/SPoSE repo): a dimension is retained
    when at least one item has a non-negligible activation.

    Parameters
    ----------
    E : ndarray of shape (n_items, embed_dim)
        Non-negative embedding (relu of weight matrix transpose).
    threshold : float
        Minimum max-weight for a dimension to be kept.  Official SPoSE default
        is 0.1.
    max_keep : int or None
        If set, further limit to the top *max_keep* dimensions by max-weight.
    """
    dim_max = E.max(axis=0)           # max activation per dimension
    keep = np.where(dim_max > threshold)[0]
    if max_keep is not None and keep.size > max_keep:
        order = np.argsort(-dim_max[keep])
        keep = keep[order[:max_keep]]
    return keep

def save_outputs(filenames: List[str], E_full: np.ndarray, keep_idx: np.ndarray):
    E_pruned = E_full[:, keep_idx] if keep_idx.size else E_full[:, :0]
    comp_cols = [f"nn_comp{i+1}" for i in range(E_pruned.shape[1])]
    out = pd.DataFrame(E_pruned, columns=comp_cols)
    out.insert(0, "filename", filenames)
    out_path_csv = os.path.join(DERIV, "nn_scores.csv")
    out.to_csv(out_path_csv, index=False)

    npz_path = os.path.join(DERIV, "nn_results.npz")
    np.savez_compressed(npz_path,
                        scores=E_pruned.astype(np.float32),
                        scores_full=E_full.astype(np.float32),
                        kept_dims=keep_idx.astype(np.int32),
                        filenames=np.array(filenames, dtype=object))

    print(f"Saved: {out_path_csv}")
    print(f"Saved: {npz_path}")
    return out_path_csv, npz_path


# ---------------------- Aggregation ---------------------- #
def aggregate_runs(out_root: str):
    """
    Scan out_root (e.g., derivatives/nn_runs) for seed_* folders and aggregate:
      - nn_scores.csv across seeds -> nn_scores_all_seeds.csv (adds 'seed' column)
      - nn_training.json across seeds -> nn_training_runs_summary.csv
    Returns paths to the two outputs.
    """
    seeds = sorted([d for d in os.listdir(out_root) if os.path.isdir(os.path.join(out_root, d)) and d.startswith("seed_")])
    if not seeds:
        raise RuntimeError(f"No seed folders found in {out_root}")

    rows_scores = []
    rows_meta = []
    for sd in seeds:
        sdir = os.path.join(out_root, sd)
        seed_str = sd.split("_", 1)[-1]
        # scores
        score_path = os.path.join(sdir, "nn_scores.csv")
        if os.path.exists(score_path):
            df0 = pd.read_csv(score_path)
            seed_int = int(seed_str)

            # Build a new, de-fragmented frame in one concat (avoids insert/assign warnings)
            if "filename" in df0.columns:
                rest_cols = [c for c in df0.columns if c != "filename"]
                df = pd.concat(
                    [
                        df0[["filename"]],
                        pd.Series(seed_int, index=df0.index, name="seed"),
                        df0[rest_cols],
                    ],
                    axis=1,
                )
            else:
                # Fallback: just prepend seed
                df = pd.concat(
                    [pd.Series(seed_int, index=df0.index, name="seed"), df0],
                    axis=1,
                )

            # Explicitly defragment
            df = df.copy()
            rows_scores.append(df)
        # meta
        meta_path = os.path.join(sdir, "nn_training.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["seed"] = int(seed_str)
                rows_meta.append(meta)
            except Exception:
                pass

    if not rows_scores:
        raise RuntimeError("Found seed folders but no nn_scores.csv files.")

    # Concatenate scores
    scores_all = pd.concat(rows_scores, ignore_index=True)
    out_scores = os.path.join(out_root, "nn_scores_all_seeds.csv")
    scores_all.to_csv(out_scores, index=False)

    # Flatten meta rows into a DataFrame
    meta_df = pd.json_normalize(rows_meta, sep=".")
    out_meta = os.path.join(out_root, "nn_training_runs_summary.csv")
    meta_df.to_csv(out_meta, index=False)

    print(f"Aggregated scores -> {out_scores}")
    print(f"Aggregated training manifests -> {out_meta}")
    return out_scores, out_meta


# ---------------------- Per-component sorted lists utility ---------------------- #

def save_sorted_lists(run_dir: str, filenames: List[str], E_pruned: np.ndarray,
                      labels_csv: str = None, info_csv: str = None):
    """
    For each NN component, save items sorted by descending score.
    Writes:
      - sorted/nn_compXX.csv  (columns: rank, filename, score, label)
      - sorted/nn_compXX.txt  (filename per line, descending)

    Name mapping mirrors PCA:
      1) Use labels_lookup.csv 'Description' if available (direct map).
      2) Fall back to Filtered_Foodpictures_information_protein_split.csv by basename or stem (case-insensitive).
    """
    sorted_dir = os.path.join(run_dir, "sorted")
    os.makedirs(sorted_dir, exist_ok=True)

    # Direct map from labels_lookup.csv if present
    desc_map_direct = {}
    if labels_csv and os.path.exists(labels_csv):
        try:
            labdf = pd.read_csv(labels_csv)
            if "filename" in labdf.columns:
                if "Description" in labdf.columns:
                    desc_map_direct = dict(zip(labdf["filename"].astype(str), labdf["Description"].astype(str)))
                else:
                    desc_map_direct = dict(zip(labdf["filename"].astype(str), labdf["filename"].astype(str)))
        except Exception:
            desc_map_direct = {}

    # Basename/stem maps from full info CSV
    map_ext, map_stem = {}, {}
    if info_csv and os.path.exists(info_csv):
        try:
            inf = pd.read_csv(info_csv, dtype=str, keep_default_na=False, na_values=[])

            # Prefer (filename, Description) if present
            if {"filename", "Description"}.issubset(inf.columns):
                fn_series = inf["filename"].astype(str).apply(lambda p: os.path.basename(p))
                desc_series = inf["Description"].astype(str)

            # Otherwise accept (image_file, Food) used by the current pipeline
            elif {"image_file", "Food"}.issubset(inf.columns):
                fn_series = inf["image_file"].astype(str).apply(lambda p: os.path.basename(p))
                desc_series = inf["Food"].astype(str)

            # Or (image_file, food) if lower-case
            elif {"image_file", "food"}.issubset(inf.columns):
                fn_series = inf["image_file"].astype(str).apply(lambda p: os.path.basename(p))
                desc_series = inf["food"].astype(str)

            else:
                fn_series, desc_series = None, None

            if fn_series is not None and desc_series is not None:
                map_ext = dict(zip(fn_series, desc_series))
                stems = fn_series.str.replace(r"\.[^.]*$", "", regex=True).str.lower()
                map_stem = dict(zip(stems, desc_series))

        except Exception:
            map_ext, map_stem = {}, {}

    def pretty_name(fn):
        base = basename_only(str(fn))
        name = desc_map_direct.get(base)
        if not name or str(name).strip().lower() == "nan":
            if base in map_ext:
                name = map_ext[base]
            else:
                stem = base.rsplit(".", 1)[0].lower()
                name = map_stem.get(stem, base)
        if not name or str(name).strip().lower() == "nan":
            name = base
        return name

    n_items, n_comp = E_pruned.shape
    for j in range(n_comp):
        scores = E_pruned[:, j]
        order = np.argsort(-scores)  # descending
        ordered_files = [filenames[i] for i in order]
        df = pd.DataFrame({
            "rank": np.arange(1, n_items + 1, dtype=int),
            "filename": ordered_files,
            "score": scores[order],
        })
        df["label"] = [pretty_name(fn) for fn in ordered_files]

        comp_tag = f"{j+1:02d}"
        csv_path = os.path.join(sorted_dir, f"nn_comp{comp_tag}.csv")
        txt_path = os.path.join(sorted_dir, f"nn_comp{comp_tag}.txt")
        df.to_csv(csv_path, index=False)
        with open(txt_path, "w", encoding="utf-8") as f:
            for fn in df["filename"]:
                f.write(f"{fn}\n")

# ---------------------- Main ---------------------- #


# Helper function for per-seed training and saving
def train_and_save_for_seed(seed_value, args, filenames, triplets, used_soft, summary_df):
    # Per-seed run directory
    suffix = "_random_only" if getattr(args, "random_only", False) else ""
    run_dir = os.path.join(args.out_root, f"seed_{seed_value}{suffix}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Writing outputs to: {run_dir}")

    print(f"Items: {len(filenames)} | Triplets: {len(triplets)} | Soft targets: {used_soft}")

    if len(triplets) == 0:
        raise RuntimeError("No triplets parsed from summary.csv. Check the columns and formatting.")

    # Train model (official SPoSE procedure)
    model, history, best_val = train_embedding(
        triplets,
        n_items=len(filenames),
        dim=args.dim,
        lmbda=args.lmbda,
        epochs=args.epochs,
        bs=args.bs,
        lr=args.lr,
        soft_targets=used_soft,
        seed=seed_value,
        early_stop=args.early_stop,
        window_size=args.window_size,
    )

    # Extract non-negative embedding: relu(W.T), shape (n_items, embed_dim)
    E_full = get_embedding(model).numpy()  # official SPoSE get_embedding

    # Prune & save to the per-seed directory
    keep_idx = prune_dimensions(E_full, threshold=args.prune_threshold, max_keep=args.max_keep)
    # Save scores and npz under run_dir
    out_csv = os.path.join(run_dir, "nn_scores.csv")
    out_npz = os.path.join(run_dir, "nn_results.npz")
    E_pruned = E_full[:, keep_idx] if keep_idx.size else E_full[:, :0]
    comp_cols = [f"nn_comp{i+1}" for i in range(E_pruned.shape[1])]
    pd.DataFrame({"filename": filenames}).join(pd.DataFrame(E_pruned, columns=comp_cols)).to_csv(out_csv, index=False)
    np.savez_compressed(out_npz,
                        scores=E_pruned.astype(np.float32),
                        scores_full=E_full.astype(np.float32),
                        kept_dims=keep_idx.astype(np.int32),
                        filenames=np.array(filenames, dtype=object))
    # Save per-component sorted lists (descending by score)
    save_sorted_lists(run_dir, filenames, E_pruned, labels_csv=args.labels, info_csv=args.info)

    # Save a completed (hole-free) RDM implied by the learned embedding
    if bool(getattr(args, "write_completed_rdm", False)):
        try:
            save_completed_rdm(run_dir, seed_value, filenames, E_pruned, E_full, keep_idx)
        except Exception as e:
            print(f"[WARN] Could not save completed RDM for seed {seed_value}: {e}")

    # Also save the final model weights for reproducibility
    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pt"))

    # Save training curve
    hist_path = os.path.join(run_dir, "training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved: {out_csv}\nSaved: {out_npz}\nSaved: {hist_path}")

    manifest = {
        "items": len(filenames),
        "triplets": len(triplets),
        "dim_requested": args.dim,
        "epochs": args.epochs,
        "batch_size": args.bs,
        "lr": args.lr,
        "lmbda": args.lmbda,
        "prune_threshold": args.prune_threshold,
        "max_keep": args.max_keep,
        "kept_dims": int(len(keep_idx)),
        "soft_targets": bool(used_soft),
        "early_stop": bool(args.early_stop),
        "window_size": args.window_size if args.early_stop else None,
        "best_val_loss": None,  # SPoSE uses train-loss convergence, not val loss
        "outputs": {
            "scores_csv": "nn_scores.csv",
            "npz": "nn_results.npz",
            "final_model": "final_model.pt",
            "training_history": "training_history.json",
            "sorted_lists_dir": "sorted",
            "completed_rdm_local": "completed_rdm_from_embedding.npz" if bool(getattr(args, "write_completed_rdm", False)) else None,
            "completed_rdm_global": f"triplet_rdm_completed_nn_seed{int(seed_value)}.npz" if bool(getattr(args, "write_completed_rdm", False)) else None,
        }
    }
    with open(os.path.join(run_dir, "nn_training.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved: {os.path.join(run_dir, 'nn_training.json')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default=(str(POLICY_SUMMARY_CSV) if POLICY_SUMMARY_CSV is not None else os.path.join(DERIV, "summary.csv")))
    p.add_argument("--labels", default=os.path.join(DERIV, "labels_lookup.csv"))
    p.add_argument("--dim", type=int, default=90,
                   help="Embedding dimensionality (official SPoSE default: 90).")
    p.add_argument("--epochs", type=int, default=500,
                   help="Maximum training epochs (official SPoSE default: 500).")

    # Hyperparameters matching official SPoSE train.py defaults
    p.add_argument("--bs", "--batch-size", dest="bs", type=int, default=100,
                   help="Batch size (official SPoSE default: 100).")
    p.add_argument("--lr", type=float, default=0.001,
                   help="Adam learning rate (official SPoSE default: 0.001).")
    p.add_argument("--lmbda", "--l1", "--l1-lambda", dest="lmbda", type=float, default=None,
                   help="L1 regularisation strength λ (official SPoSE default: 0.007). "
                        "Scaled internally as (λ / n_items).")
    p.add_argument("--prune_threshold", "--prune-threshold", dest="prune_threshold",
                   type=float, default=0.1,
                   help="Keep dimensions whose max weight > threshold (official SPoSE default: 0.1).")
    p.add_argument("--max_keep", type=int, default=None)

    p.add_argument("--seed", type=int, default=int(POLICY_DEFAULT_NN_SEED))
    p.add_argument(
        "--soft_targets",
        action="store_true",
        help="Use p_ab, p_ac, p_bc columns in summary.csv if present.",
    )
    p.add_argument(
        "--random-only",
        action="store_true",
        help="Filter summary.csv to train SPoSE only on random trials (is_fixed == False & is_dense == False).",
    )
    p.add_argument(
        "--out_root",
        default=RUNS_ROOT,
        help="Root folder for per-seed outputs (default: derivatives/nn_runs)",
    )
    p.add_argument(
        "--early_stop",
        action="store_true",
        default=True,
        help="Enable official SPoSE early stopping (linear-regression convergence criterion "
             "on training loss; default: on).",
    )
    p.add_argument(
        "--no-early-stop",
        dest="early_stop",
        action="store_false",
        help="Disable early stopping and run for the full --epochs budget.",
    )
    p.add_argument(
        "--window-size", "--window_size",
        dest="window_size",
        type=int,
        default=50,
        help="Number of recent loss values used in early-stopping regression (official SPoSE default: 50).",
    )
    # Kept for backwards-compatibility; ignored by the SPoSE training loop
    p.add_argument("--patience", type=int, default=10,
                   help="(Deprecated) Ignored. Use --window-size instead.")
    # --val_frac is now used by --search-lambda; the old deprecated default (0.08) is superseded.
    p.add_argument(
        "--aggregate",
        action="store_true",
        help="After training, aggregate all seed_* runs under out_root.",
    )
    p.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Only aggregate existing runs under out_root and exit.",
    )

    # Seeds can be provided as:
    #   --seeds 0 1 2 3 4
    # or legacy:
    #   --seeds 0,1,2,3,4
    p.add_argument(
        "--seeds",
        nargs="*",
        default=None,
        help=(
            "Seeds to run sequentially. Accepts either space-separated ints "
            "(e.g., '--seeds 0 1 2 3 4') or a single comma-separated token "
            "(e.g., '--seeds 0,1,2,3,4'). Overrides --seed."
        ),
    )

    p.add_argument(
        "--pubgrade",
        action="store_true",
        help="Use publication-grade stable settings (overrides most training args).",
    )

    # Backwards-compatible alias for older CLI
    p.add_argument(
        "--write_completed_rdm",
        "--write-completed-rdm",
        dest="write_completed_rdm",
        action="store_true",
        help="Also save a completed (hole-free) RDM implied by the learned embedding.",
    )

    # Backwards-compatible no-op flag (older commands used --write-manifest)
    p.add_argument(
        "--write-manifest",
        dest="write_manifest",
        action="store_true",
        help="(Deprecated) Accepted for backwards compatibility. Per-seed manifests are always written.",
    )

    p.add_argument(
        "--select",
        action="store_true",
        help=(
            "Mark the trained seed(s) as the selected seed for downstream scripts. "
            "Writes derivatives/nn_runs/selected_seed.json so that compute_model_evaluation.py "
            "can auto-detect which NN seed to use without requiring --nn-seed. "
            "When --seeds is used with --select, the last seed in the list is marked selected "
            "unless exactly one seed is given."
        ),
    )

    # ── Lambda cross-validation search (Martin's procedure) ─────────────────────
    p.add_argument(
        "--search-lambda", "--search_lambda",
        dest="search_lambda",
        action="store_true",
        help=(
            "Run a repeated hold-out cross-validation to select the best lambda "
            "before training the final model(s).  For each value in --lambda-grid, "
            "trains n_val_splits models on random train/val splits and averages "
            "validation loss.  The best lambda is then used for all --seeds runs."
        ),
    )
    p.add_argument(
        "--lambda-grid", "--lambda_grid",
        dest="lambda_grid",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Space-separated lambda values to search.  If omitted, a built-in "
            "two-phase grid is used: coarse (0.001–0.05) followed by fine steps "
            "around the coarse best (e.g. 0.0035, 0.0038, 0.00385, 0.0039, 0.004). "
            "Example: --lambda-grid 0.001 0.003 0.005 0.007 0.01"
        ),
    )
    p.add_argument(
        "--n-val-splits", "--n_val_splits",
        dest="n_val_splits",
        type=int,
        default=10,
        help="Number of random train/val splits per lambda candidate (Martin: 10).  Default: 10.",
    )
    p.add_argument(
        "--val-frac",
        dest="val_frac",
        type=float,
        default=0.1,
        help="Fraction of triplets held out for validation during lambda search.  Default: 0.1.",
    )
    p.add_argument(
        "--lambda-search-seed-offset",
        dest="lambda_search_seed_offset",
        type=int,
        default=1000,
        help=(
            "Base random seed for lambda-search splits (default: 1000).  "
            "Kept well above the final-model seed range (0–24) to avoid overlap."
        ),
    )
    p.add_argument(
        "--lambda-search-out",
        dest="lambda_search_out",
        default=None,
        help=(
            "Path to write the lambda-search results CSV.  "
            "Default: <out_root>/lambda_search_results.csv"
        ),
    )

    # Info CSV used for robust label mapping (matches PCA code)
    default_info = str(POLICY_INFO_CSV) if POLICY_INFO_CSV is not None else os.path.join(HERE, "reference", "Foodpictures_information_dynamic.csv")
    p.add_argument(
        "--info",
        default=default_info,
        help="Path to the canonical info CSV for filename→label mapping.",
    )

    args = p.parse_args()

    # Apply publication-grade preset (overrides most settings)
    if args.pubgrade:
        # Official SPoSE hyperparameters (Muttenthaler et al., 2022)
        args.dim = 90
        args.epochs = 500
        args.bs = 100
        args.lr = 0.001
        if args.lmbda is None:          # only apply default if user didn't pass --lmbda
            args.lmbda = 0.007
        args.prune_threshold = 0.1
        args.early_stop = True
        args.window_size = 50
        args.write_completed_rdm = True
        # Run multiple seeds and aggregate by default.
        # IMPORTANT: do not override a user-provided --seeds list.
        if args.seeds is None or (isinstance(args.seeds, (list, tuple)) and len(args.seeds) == 0):
            args.seeds = [0, 1, 2, 3, 4]
            seeds_msg = "seeds=0..4 (default)"
        else:
            seeds_msg = f"seeds={args.seeds} (user)"
        args.aggregate = True
        print(
            "[pubgrade] Applied: dim=90, epochs=500, bs=100, lr=0.001, lmbda=0.007, prune=0.1, "
            "early_stop=on (window=50), "
            + seeds_msg + ", aggregate=on"
        )

    # Apply default lmbda if neither --pubgrade nor --lmbda was given
    if args.lmbda is None:
        args.lmbda = 0.007

    # Fast path: aggregation only
    if args.aggregate_only:
        aggregate_runs(args.out_root)
        return

    # Load data
    summary_df = load_summary(args.summary)
    if getattr(args, "random_only", False):
        summary_df = summary_df[(summary_df['is_fixed'] == False) & (summary_df['is_dense'] == False)].copy()
        print(f"[INFO] --random-only: filtered summary to only include random trials (kept {len(summary_df)}/{len(summary_df) + 43800 + 29200} trials).")
    filenames, idx_of = build_filename_index(summary_df, labels_lookup_path=args.labels)
    triplets, used_soft = triplets_from_summary(summary_df, idx_of, soft_targets=args.soft_targets)

    fmt = "split-cols (StimA/B/C_file)" if {"StimA_file","StimB_file","StimC_file","choice_file"}.issubset(summary_df.columns) else "images_onscreen"
    print(f"Detected summary format: {fmt}")

    # ── Lambda cross-validation search (optional) ────────────────────────────────
    if getattr(args, "search_lambda", False) and not used_soft:
        # Build the search grid
        if args.lambda_grid:
            grid = args.lambda_grid
        else:
            # Built-in two-phase grid:
            #   Phase 1 — coarse order-of-magnitude sweep
            #   Phase 2 — fine steps around a likely optimum for food triplet data
            #   (based on Martin's experience: best lambda often near 0.0038–0.004)
            grid = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.05,
                    0.0035, 0.0036, 0.0037, 0.0038, 0.00385, 0.0039, 0.004,
                    0.0041, 0.0042, 0.0045]
            print("[lambda-search] Using built-in two-phase grid (coarse + fine).")

        best_lmbda, search_df = search_lambda(
            triplets=triplets,
            n_items=len(filenames),
            lambda_grid=grid,
            n_val_splits=args.n_val_splits,
            val_frac=args.val_frac,
            dim=args.dim,
            epochs=args.epochs,
            bs=args.bs,
            lr=args.lr,
            early_stop=args.early_stop,
            window_size=args.window_size,
            seed_offset=args.lambda_search_seed_offset,
        )

        # Save search results
        search_out = args.lambda_search_out or os.path.join(args.out_root, "lambda_search_results.csv")
        os.makedirs(os.path.dirname(search_out), exist_ok=True)
        search_df.to_csv(search_out, index=False)
        # Also write a one-line summary
        summary_out = search_out.replace(".csv", "_summary.csv")
        (search_df.groupby("lmbda")["val_loss"]
                  .agg(["mean", "std", "min", "max", "count"])
                  .reset_index()
                  .sort_values("mean")
                  .to_csv(summary_out, index=False))
        print(f"[lambda-search] Results -> {search_out}")
        print(f"[lambda-search] Summary -> {summary_out}")
        print(f"[lambda-search] *** Overriding --lmbda {args.lmbda} -> {best_lmbda} ***")
        args.lmbda = best_lmbda

    elif getattr(args, "search_lambda", False) and used_soft:
        print("[lambda-search] WARNING: --search-lambda is not supported with soft targets. "
              "Using fixed --lmbda.")

    # Run one or many seeds
    if args.seeds:
        # args.seeds is a list when passed as --seeds 0 1 2, but could be a single
        # comma-separated token (legacy). Normalize to a list[int].
        if isinstance(args.seeds, (list, tuple)):
            raw = list(args.seeds)
        else:
            raw = [args.seeds]

        # If provided as a single comma-separated token, split it.
        if len(raw) == 1 and isinstance(raw[0], str) and ("," in raw[0]):
            raw = [x.strip() for x in raw[0].split(",") if x.strip()]

        # Strip commas and whitespace (handles tokens like "9,")
        raw_clean = []
        for x in raw:
            if x is None:
                continue
            xs = str(x).strip().strip(",")
            if xs != "":
                raw_clean.append(xs)

        try:
            seeds_list = [int(x) for x in raw_clean]
        except Exception:
            raise ValueError("Could not parse --seeds. Use e.g. '--seeds 0 1 2 3 4' or '--seeds 0,1,2,3,4'.")

        print(f"[INFO] Running seeds: {seeds_list}")
        for sv in seeds_list:
            print(f"\n=== Running seed {sv} ===")
            train_and_save_for_seed(sv, args, filenames, triplets, used_soft, summary_df)

        # Aggregate automatically if multiple seeds were run or --aggregate is set
        if args.aggregate or len(seeds_list) > 1:
            try:
                aggregate_runs(args.out_root)
            except Exception as e:
                print(f"[WARN] Aggregation skipped: {e}")
    else:
        # Single seed path (existing behaviour)
        train_and_save_for_seed(args.seed, args, filenames, triplets, used_soft, summary_df)
        if args.aggregate:
            try:
                aggregate_runs(args.out_root)
            except Exception as e:
                print(f"[WARN] Aggregation skipped: {e}")

    # --select: write selected_seed.json so downstream scripts know which seed to use
    if getattr(args, "select", False):
        if args.seeds:
            # With multiple seeds, mark whichever single seed was specified (warn if >1)
            if len(seeds_list) == 1:
                _selected = seeds_list[0]
            else:
                _selected = seeds_list[-1]
                print(f"[WARN] --select with multiple seeds: marking the last seed ({_selected}) as selected. "
                      "Re-run with a single --seed to select a specific one.")
        else:
            _selected = args.seed
        _sel_path = os.path.join(args.out_root, "selected_seed.json")
        with open(_sel_path, "w", encoding="utf-8") as _sf:
            json.dump({
                "selected_seed": int(_selected),
                "note": (
                    "Written by compute_nn_embedding_decomp.py --select. "
                    "Read by compute_model_evaluation.py to auto-set --nn-seed default."
                ),
            }, _sf, indent=2)
        print(f"[OK] Selected seed {_selected} written to: {_sel_path}")

if __name__ == "__main__":
    main()