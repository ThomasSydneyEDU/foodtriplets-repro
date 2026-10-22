#!/usr/bin/env python3
"""make_visual_models.py

Compute low-level visual feature models for plated food images (384x384) and append
per-image feature columns to the reference info CSV.

Feature sources (contingent computation)
----------------------------------------
- ll_* features: reused from `Foodpictures_information_dynamic.csv` (PAFID
  pipeline) when it fully covers the stimulus set; otherwise recomputed from
  pixels (full recompute, keeping the HOG PCA basis consistent).
- dl_* (ResNet-50) features: NOT part of PAFID; reused from the reference CSV
  when complete, otherwise recomputed from pixels (requires torch/torchvision).
- Pass --recompute-features to force pixel computation for everything.

Assumptions
-----------
- Images are centered on plates with a smooth grey background.
- Images are 384x384 (script will still run if not, but models are tuned for that).

Inputs
------
- CSV (default): canonical reference file from `reference.policy_defaults.INFO_CSV`
  (fallback: analysis_code/reference/Foodpictures_information_canonical.csv)
  Must contain an image filename column, preferably `filename`.

Image lookup
------------
By default the script searches, in order:
- <repo>/expt_stimuli/images
- <repo>/expt_stimuli_current/images
- <repo>/expt_stimuli/images (redundant safety)

Outputs
-------
- Overwrites the input CSV with new/updated columns:
  ll_mean_luminance
  ll_rms_contrast
  ll_lab_L_mean, ll_lab_L_std
  ll_lab_a_mean, ll_lab_a_std
  ll_lab_b_mean, ll_lab_b_std
  ll_hsv_s_mean
  ll_edge_energy
  ll_power_radial_30_json
  ll_hog_pc01..ll_hog_pc10 (configurable)

- Writes: analysis_code/reference/visual_models_manifest.csv

Usage
-----
  python analysis_code/make_visual_models.py
  python analysis_code/make_visual_models.py --hog-pcs 12
  python analysis_code/make_visual_models.py --images-dir /path/to/images

Dependencies
------------
  pip install numpy pandas pillow scikit-image scikit-learn torch torchvision

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:

    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    try:
        from reference.policy_defaults import INFO_CSV as _INFO_CSV, DERIVATIVES_DIR as _DERIVATIVES_DIR
    except Exception:
        _INFO_CSV = repo_root / "analysis_code" / "reference" / "Foodpictures_information_dynamic.csv"
        _DERIVATIVES_DIR = repo_root / "analysis_code" / "derivatives"

    default_csv = Path(_INFO_CSV)

    p = argparse.ArgumentParser(description="Compute low-level visual models and append to reference CSV")
    p.add_argument("--csv", type=str, default=str(default_csv), help="Path to reference CSV to update")
    p.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Optional override: directory containing images (filenames must match CSV image_file basenames)",
    )
    p.add_argument("--hog-pcs", type=int, default=10, help="Number of PCA components to keep for HOG features")
    p.add_argument("--build-rdms", action="store_true", help="Also build and save model RDMs to --rdm-outdir")
    default_rdm_dir = Path(_DERIVATIVES_DIR) / "model_rdms_canon"
    
    p.add_argument(
        "--rdm-outdir",
        type=str,
        default=str(default_rdm_dir),
        help="Output directory for saved RDM .npz files",
    )
    p.add_argument(
        "--deepnet-resnet50",
        action="store_true",
        help="Add ResNet-50 conv1 features (ImageNet pretrained)",
    )
    p.add_argument(
        "--deepnet-resnet50-layer3",
        action="store_true",
        help="Add ResNet-50 layer3 features (ImageNet pretrained) for mid-level texture",
    )
    p.add_argument(
        "--lbp-bins",
        type=int,
        default=10,
        help="Number of bins for LBP (Local Binary Pattern) histogram",
    )
    p.add_argument(
        "--deep-pcs",
        type=int,
        default=10,
        help="Number of PCA components to keep for deepnet conv1 features",
    )
    p.add_argument(
        "--recompute-features",
        action="store_true",
        help="Force recomputation of ALL features from pixels, ignoring stored "
             "ll_ values in Foodpictures_information_dynamic.csv and stored dl_ "
             "columns in the reference CSV.",
    )
    return p.parse_args()

def compute_lbp(rgb: np.ndarray, n_bins: int = 10) -> np.ndarray:
    from skimage.color import rgb2gray
    from skimage.feature import local_binary_pattern
    gray = rgb2gray(rgb)
    # Convert to uint8 to avoid floating-point comparison warnings and errors
    gray_uint8 = (gray * 255).astype(np.uint8)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_uint8, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_points + 2), density=True)
    return hist.astype(np.float32)

def compute_resnet50_layer3_vec(rgb: np.ndarray) -> np.ndarray:
    try:
        import torch
        import torch.nn.functional as F
        from torchvision import models
    except Exception as e:
        raise RuntimeError("torch/torchvision required") from e

    if not hasattr(compute_resnet50_layer3_vec, "_model"):
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.eval()
        compute_resnet50_layer3_vec._model = m
        compute_resnet50_layer3_vec._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        compute_resnet50_layer3_vec._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    model = compute_resnet50_layer3_vec._model
    mean = compute_resnet50_layer3_vec._mean
    std = compute_resnet50_layer3_vec._std

    x = np.asarray(rgb, dtype=np.float32)
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    t = (t - mean) / std

    with torch.no_grad():
        y = model.conv1(t)
        y = model.bn1(y)
        y = F.relu(y)
        y = model.maxpool(y)
        y = model.layer1(y)
        y = model.layer2(y)
        y = model.layer3(y)
        v = y.mean(dim=(2, 3)).squeeze(0)

    return v.detach().cpu().numpy().astype(np.float32)

def compute_resnet50_conv1_vec(rgb: np.ndarray) -> np.ndarray:
    """Compute ResNet-50 conv1 channel vector (global-average pooled).
    Returns a 64-d float32 vector (conv1 has 64 output channels).
    """
    try:
        import torch
        import torch.nn.functional as F
        from torchvision import models
    except Exception as e:
        raise RuntimeError(
            "torch/torchvision are required for --deepnet-resnet50. "
            "Install with: pip install torch torchvision"
        ) from e

    # Build model lazily and cache it on the function
    if not hasattr(compute_resnet50_conv1_vec, "_model"):
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.eval()
        compute_resnet50_conv1_vec._model = m  # type: ignore

        compute_resnet50_conv1_vec._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # type: ignore
        compute_resnet50_conv1_vec._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)  # type: ignore

    model = compute_resnet50_conv1_vec._model  # type: ignore
    mean = compute_resnet50_conv1_vec._mean  # type: ignore
    std = compute_resnet50_conv1_vec._std  # type: ignore

    x = np.asarray(rgb, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB array, got shape {x.shape}")

    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    t = (t - mean) / std

    with torch.no_grad():
        y = model.conv1(t)
        y = model.bn1(y)
        y = F.relu(y)
        v = y.mean(dim=(2, 3)).squeeze(0)  # 64

    return v.detach().cpu().numpy().astype(np.float32)

def _basename(x: str) -> str:
    return os.path.basename(str(x).strip())


def find_image_dir(repo_root: Path, override: Optional[str]) -> Path:
    if override:
        d = Path(override).expanduser().resolve()
        if not d.exists():
            raise FileNotFoundError(f"--images-dir does not exist: {d}")
        return d

    candidates = [
        repo_root / "expt_stimuli" / "images",
        repo_root / "expt_stimuli_current" / "images",
        repo_root / "expt_stimuli" / "images",
    ]
    for d in candidates:
        if d.exists():
            return d
    raise FileNotFoundError("Could not find an images directory. Use --images-dir to specify one.")


def load_rgb_image(path: Path) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def compute_basic_features(rgb: np.ndarray) -> Dict[str, float]:
    from skimage.color import rgb2gray, rgb2lab, rgb2hsv
    from skimage.filters import sobel

    gray = rgb2gray(rgb).astype(np.float32)

    mean_lum = float(np.mean(gray))
    rms_contrast = float(np.std(gray))

    lab = rgb2lab(rgb).astype(np.float32)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    hsv = rgb2hsv(rgb).astype(np.float32)
    s = hsv[..., 1]

    edge_energy = float(np.mean(np.abs(sobel(gray))))

    feats = {
        "ll_mean_luminance": mean_lum,
        "ll_rms_contrast": rms_contrast,
        "ll_lab_L_mean": float(np.mean(L)),
        "ll_lab_L_std": float(np.std(L)),
        "ll_lab_a_mean": float(np.mean(a)),
        "ll_lab_a_std": float(np.std(a)),
        "ll_lab_b_mean": float(np.mean(b)),
        "ll_lab_b_std": float(np.std(b)),
        "ll_hsv_s_mean": float(np.mean(s)),
        "ll_edge_energy": edge_energy,
    }
    return feats


def radial_power_spectrum(gray: np.ndarray, n_bins: int = 30) -> np.ndarray:
    """Compute radially-averaged power spectrum with n_bins.

    gray should be float32 in [0,1].
    """
    g = gray.astype(np.float32)
    g = g - np.mean(g)
    F = np.fft.fft2(g)
    F = np.fft.fftshift(F)
    P = (np.abs(F) ** 2).astype(np.float32)

    h, w = P.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r = r / np.max(r)  # normalize 0..1

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = np.zeros(n_bins, dtype=np.float32)

    for i in range(n_bins):
        m = (r >= bins[i]) & (r < bins[i + 1])
        if np.any(m):
            out[i] = float(np.mean(P[m]))
        else:
            out[i] = 0.0

    # log-compress for stability
    out = np.log1p(out)
    # normalize per-image to unit mean to reduce overall brightness scaling
    mu = float(np.mean(out))
    if mu > 0:
        out = out / mu
    return out


def compute_hog(rgb: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2gray
    from skimage.feature import hog

    gray = rgb2gray(rgb).astype(np.float32)
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.astype(np.float32)


def _nan_spearman(a: np.ndarray, b: np.ndarray, min_overlap: int = 10) -> float:
    """Spearman rho with NaN masking (no scipy)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n < min_overlap:
        return np.nan
    ar = pd.Series(a[m]).rank(method="average").to_numpy(dtype=float)
    br = pd.Series(b[m]).rank(method="average").to_numpy(dtype=float)
    # Pearson on ranks
    if np.std(ar) == 0 or np.std(br) == 0:
        return np.nan
    return float(np.corrcoef(ar, br)[0, 1])


def rdm_from_scalar(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    D = np.full((n, n), np.nan, dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                D[i, j] = 0.0
                continue
            if not (np.isfinite(x[i]) and np.isfinite(x[j])):
                continue
            D[i, j] = D[j, i] = float(abs(x[i] - x[j]))

    # normalize by median non-zero distance (robust scale)
    vals = D[np.isfinite(D) & (D > 0)]
    if vals.size > 0:
        med = float(np.median(vals))
        if med > 0:
            D = (D / med).astype(np.float32)
            np.fill_diagonal(D, 0.0)
    return D


def rdm_from_vectors(X: np.ndarray, min_overlap: int = 10) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    # Fast path: no missing values — vectorized Spearman. Mathematically
    # identical to the per-pair loop below when all entries are finite
    # (rank each row once, then 1 - Pearson correlation of the rank rows).
    if np.isfinite(X).all() and X.shape[1] >= min_overlap:
        R = np.vstack([
            pd.Series(X[i]).rank(method="average").to_numpy(dtype=float)
            for i in range(n)
        ])
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(R)
        D = (1.0 - C).astype(np.float32)
        np.fill_diagonal(D, 0.0)
        return D

    D = np.full((n, n), np.nan, dtype=np.float32)
    for i in range(n):
        D[i, i] = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rho = _nan_spearman(X[i], X[j], min_overlap=min_overlap)
            if np.isfinite(rho):
                d = float(1.0 - rho)
                D[i, j] = D[j, i] = np.float32(d)
    return D


import json
from typing import Optional

def save_rdm_npz(
    out_path: Path,
    D: np.ndarray,
    labels: List[str],
    kind: str = "low_level",
    meta: Optional[dict] = None,
) -> None:
    """Save an RDM NPZ in a schema compatible across this repo.

    We write multiple equivalent matrix keys for backwards/forwards compatibility:
      - rdm: preferred key for model RDMs
      - data: used by some older loaders
      - D: used by some scripts

    Also stores:
      - labels: basename labels
      - kind: short descriptor string
      - meta_json: optional JSON metadata
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    D32 = np.asarray(D, dtype=np.float32)
    lab = np.asarray(labels, dtype=object)
    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    np.savez_compressed(
        out_path,
        rdm=D32,
        data=D32,
        D=D32,
        labels=lab,
        kind=np.asarray(kind, dtype=object),
        meta_json=np.asarray(meta_json, dtype=object),
    )


def main() -> int:
    args = parse_args()

    here = Path(__file__).resolve()
    repo_root = here.parent.parent

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # The image folder is only needed when features must be recomputed from pixels.
    # The released reference CSV already holds every ll_/dl_ feature, so a missing
    # folder is tolerated here and only fails if pixels are actually required.
    try:
        images_dir = find_image_dir(repo_root, args.images_dir)
    except FileNotFoundError:
        images_dir = None

    df = pd.read_csv(csv_path)

    # Determine image filename column
    img_col = None
    for c in ["filename", "image_file", "file"]:
        if c in df.columns:
            img_col = c
            break
    if img_col is None:
        raise KeyError("CSV must contain an image filename column (preferably 'filename')")

    df["_img_base"] = df[img_col].astype(str).apply(_basename)

    # ------------------------------------------------------------------
    # Feature reuse (contingent computation)
    #
    # ll_* features: the dynamic reference CSV (PAFID pipeline) is the source
    # of truth. If it covers ALL rows for ALL ll_ columns, those values are
    # used and pixel computation of ll_ features is skipped. If any row is
    # missing (e.g. new stimuli), everything is recomputed from pixels so the
    # HOG PCA basis stays consistent across the full image set.
    #
    # dl_* features: not part of PAFID; reused from the reference CSV itself
    # when complete, otherwise recomputed from pixels (requires torch).
    #
    # --recompute-features forces pixel computation for everything.
    # ------------------------------------------------------------------
    LL_SCALAR_COLS = [
        "ll_mean_luminance", "ll_rms_contrast",
        "ll_lab_L_mean", "ll_lab_L_std", "ll_lab_a_mean", "ll_lab_a_std",
        "ll_lab_b_mean", "ll_lab_b_std", "ll_hsv_s_mean", "ll_edge_energy",
    ]
    hog_cols = [f"ll_hog_pc{str(k).zfill(2)}" for k in range(1, int(args.hog_pcs) + 1)]
    LL_ALL_COLS = LL_SCALAR_COLS + ["ll_power_radial_30_json", "ll_lbp_hist_json"] + hog_cols

    def _complete_numeric(frame: pd.DataFrame, cols: List[str]) -> bool:
        if not all(c in frame.columns for c in cols):
            return False
        sub = frame[[c for c in cols if not c.endswith("_json")]].apply(pd.to_numeric, errors="coerce")
        if sub.isna().any().any():
            return False
        for c in cols:
            if c.endswith("_json"):
                s = frame[c].astype(str).str.strip()
                if (s.isin(["", "nan", "None"])).any():
                    return False
        return True

    reuse_ll = False
    if not args.recompute_features:
        try:
            from reference.policy_defaults import DYNAMIC_INFO_CSV as _DYN_CSV
            dyn_path = Path(_DYN_CSV)
        except Exception:
            dyn_path = repo_root / "analysis_code" / "reference" / "Foodpictures_information_dynamic.csv"
        if dyn_path.exists():
            dyn = pd.read_csv(dyn_path, encoding="utf-8-sig")
            dyn.columns = [str(c).strip() for c in dyn.columns]
            if "filename" in dyn.columns:
                dyn["_img_base"] = dyn["filename"].astype(str).apply(_basename)
                dyn = dyn.drop_duplicates(subset="_img_base", keep="first").set_index("_img_base")
                if all(c in dyn.columns for c in LL_ALL_COLS):
                    mapped = pd.DataFrame(
                        {c: df["_img_base"].map(dyn[c]) for c in LL_ALL_COLS}
                    )
                    if _complete_numeric(mapped, LL_ALL_COLS):
                        for c in LL_ALL_COLS:
                            df[c] = mapped[c].values
                        reuse_ll = True
                        print(f"[INFO] ll_ features reused from dynamic CSV ({dyn_path.name}); "
                              "pixel computation of ll_ features skipped.")
                    else:
                        print("[INFO] ll_ columns in dynamic CSV are incomplete for this stimulus set; "
                              "recomputing ALL ll_ features from pixels (keeps HOG PCA basis consistent).")
                else:
                    print("[INFO] Dynamic CSV lacks some ll_ columns; computing ll_ features from pixels.")
        else:
            print(f"[INFO] Dynamic CSV not found at {dyn_path}; computing ll_ features from pixels.")

    conv1_cols = [f"dl_resnet50_conv1_pc{str(k).zfill(2)}" for k in range(1, int(args.deep_pcs) + 1)]
    layer3_cols = [f"dl_resnet50_layer3_pc{str(k).zfill(2)}" for k in range(1, int(args.deep_pcs) + 1)]
    reuse_conv1 = (not args.recompute_features) and bool(args.deepnet_resnet50) \
        and _complete_numeric(df, conv1_cols)
    reuse_layer3 = (not args.recompute_features) and bool(getattr(args, "deepnet_resnet50_layer3", False)) \
        and _complete_numeric(df, layer3_cols)
    if reuse_conv1:
        print("[INFO] dl_resnet50_conv1 PCs reused from reference CSV; ResNet conv1 pass skipped.")
    if reuse_layer3:
        print("[INFO] dl_resnet50_layer3 PCs reused from reference CSV; ResNet layer3 pass skipped.")

    compute_ll = not reuse_ll
    compute_conv1 = bool(args.deepnet_resnet50) and not reuse_conv1
    compute_layer3 = bool(getattr(args, "deepnet_resnet50_layer3", False)) and not reuse_layer3

    # Load all images and compute per-image features
    basic_rows: List[Dict[str, float]] = []
    power_rows: List[str] = []
    hog_feats: List[np.ndarray] = []
    deep_feats: List[np.ndarray] = []
    lbp_rows: List[str] = []
    deep_layer3_feats: List[np.ndarray] = []
    missing: List[str] = []

    # Lazy imports used in the loop
    if compute_ll:
        from skimage.color import rgb2gray

    if not (compute_ll or compute_conv1 or compute_layer3):
        print("[INFO] All features reused from stored columns; image processing skipped entirely.")

    if (compute_ll or compute_conv1 or compute_layer3) and images_dir is None:
        raise FileNotFoundError("Pixel features must be recomputed but no images directory was found. Use --images-dir.")

    if compute_ll or compute_conv1 or compute_layer3:
        for i, row in df.iterrows():
            fn = row["_img_base"]
            img_path = images_dir / fn
            if not img_path.exists():
                missing.append(fn)
                basic_rows.append({
                    "ll_mean_luminance": np.nan,
                    "ll_rms_contrast": np.nan,
                    "ll_lab_L_mean": np.nan,
                    "ll_lab_L_std": np.nan,
                    "ll_lab_a_mean": np.nan,
                    "ll_lab_a_std": np.nan,
                    "ll_lab_b_mean": np.nan,
                    "ll_lab_b_std": np.nan,
                    "ll_hsv_s_mean": np.nan,
                    "ll_edge_energy": np.nan,
                })
                power_rows.append(json.dumps([np.nan] * 30))
                hog_feats.append(np.full(1, np.nan, dtype=np.float32))
                deep_feats.append(np.full(1, np.nan, dtype=np.float32))
                lbp_rows.append(json.dumps([np.nan] * args.lbp_bins))
                deep_layer3_feats.append(np.full(1, np.nan, dtype=np.float32))
                continue

            rgb = load_rgb_image(img_path)

            if compute_ll:
                basic = compute_basic_features(rgb)
                basic_rows.append(basic)

                gray = rgb2gray(rgb).astype(np.float32)
                rad = radial_power_spectrum(gray, n_bins=30)
                power_rows.append(json.dumps([float(x) for x in rad.tolist()]))

                hog_vec = compute_hog(rgb)
                hog_feats.append(hog_vec)

                # LBP features
                lbp_hist = compute_lbp(rgb, n_bins=args.lbp_bins)
                lbp_rows.append(json.dumps([float(x) for x in lbp_hist.tolist()]))

            # Deepnet features (optional)
            if compute_conv1:
                try:
                    deep_vec = compute_resnet50_conv1_vec(rgb)
                except Exception:
                    deep_vec = np.full(1, np.nan, dtype=np.float32)
                deep_feats.append(deep_vec)
            else:
                deep_feats.append(np.full(1, np.nan, dtype=np.float32))

            if compute_layer3:
                try:
                    deep_l3 = compute_resnet50_layer3_vec(rgb)
                except Exception:
                    deep_l3 = np.full(1, np.nan, dtype=np.float32)
                deep_layer3_feats.append(deep_l3)
            else:
                deep_layer3_feats.append(np.full(1, np.nan, dtype=np.float32))

    # Attach basic features (skipped when ll_ values were reused from the dynamic CSV)
    if compute_ll:
        basic_df = pd.DataFrame(basic_rows)
        for c in basic_df.columns:
            df[c] = basic_df[c].values

        # Attach power spectrum JSON
        df["ll_power_radial_30_json"] = power_rows
        df["ll_lbp_hist_json"] = lbp_rows

    # Build HOG PCA across valid rows
    # Stack HOG vectors; handle missing by masking
    valid = [v for v in hog_feats if v.size > 1 and np.isfinite(v).all()] if compute_ll else []
    if not compute_ll:
        pass  # HOG PCs already present from the dynamic CSV
    elif len(valid) < 5:
        print("[WARN] Not enough valid HOG vectors to compute PCA; skipping HOG PCs")
        for k in range(1, int(args.hog_pcs) + 1):
            df[f"ll_hog_pc{str(k).zfill(2)}"] = np.nan
    else:
        # Ensure consistent length
        L = int(valid[0].shape[0])
        X = np.zeros((len(df), L), dtype=np.float32)
        mask = np.zeros(len(df), dtype=bool)
        for i, v in enumerate(hog_feats):
            if v.size == L and np.isfinite(v).all():
                X[i] = v
                mask[i] = True
            else:
                # leave zeros; will be marked missing
                pass

        # PCA on z-scored features (fit on valid only)
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler(with_mean=True, with_std=True)
        Xv = scaler.fit_transform(X[mask])

        k = int(args.hog_pcs)
        k = max(1, min(k, Xv.shape[1], Xv.shape[0]))
        pca = PCA(n_components=k, random_state=0)
        Zv = pca.fit_transform(Xv)

        # Project all rows (missing rows -> NaN)
        Zall = np.full((len(df), k), np.nan, dtype=np.float32)
        Zall[mask] = Zv

        for j in range(k):
            df[f"ll_hog_pc{str(j+1).zfill(2)}"] = Zall[:, j]

        # If user asked for more than k, create remaining columns as NaN
        for j in range(k, int(args.hog_pcs)):
            df[f"ll_hog_pc{str(j+1).zfill(2)}"] = np.nan

    # --- Deepnet (ResNet-50 conv1) PCA features ---
    if compute_conv1:
        valid_d = [v for v in deep_feats if v.size > 1 and np.isfinite(v).all()]
        if len(valid_d) < 5:
            print("[WARN] Not enough valid deepnet vectors to compute PCA; skipping deepnet PCs")
            for j in range(1, int(args.deep_pcs) + 1):
                df[f"dl_resnet50_conv1_pc{str(j).zfill(2)}"] = np.nan
        else:
            Ld = int(valid_d[0].shape[0])  # 64
            Xd = np.zeros((len(df), Ld), dtype=np.float32)
            maskd = np.zeros(len(df), dtype=bool)
            for i, v in enumerate(deep_feats):
                if v.size == Ld and np.isfinite(v).all():
                    Xd[i] = v
                    maskd[i] = True

            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            scaler_d = StandardScaler(with_mean=True, with_std=True)
            Xdv = scaler_d.fit_transform(Xd[maskd])

            kd = int(args.deep_pcs)
            kd = max(1, min(kd, Xdv.shape[1], Xdv.shape[0]))
            pca_d = PCA(n_components=kd, random_state=0)
            Zdv = pca_d.fit_transform(Xdv)

            Zall_d = np.full((len(df), kd), np.nan, dtype=np.float32)
            Zall_d[maskd] = Zdv

            for j in range(kd):
                df[f"dl_resnet50_conv1_pc{str(j+1).zfill(2)}"] = Zall_d[:, j]

            for j in range(kd, int(args.deep_pcs)):
                df[f"dl_resnet50_conv1_pc{str(j+1).zfill(2)}"] = np.nan

    # --- Deepnet (ResNet-50 layer3) PCA features ---
    if compute_layer3:
        valid_d3 = [v for v in deep_layer3_feats if v.size > 1 and np.isfinite(v).all()]
        if len(valid_d3) < 5:
            print("[WARN] Not enough valid deepnet layer3 vectors; skipping")
            for j in range(1, int(args.deep_pcs) + 1):
                df[f"dl_resnet50_layer3_pc{str(j).zfill(2)}"] = np.nan
        else:
            Ld3 = int(valid_d3[0].shape[0])  # 1024
            Xd3 = np.zeros((len(df), Ld3), dtype=np.float32)
            maskd3 = np.zeros(len(df), dtype=bool)
            for i, v in enumerate(deep_layer3_feats):
                if v.size == Ld3 and np.isfinite(v).all():
                    Xd3[i] = v
                    maskd3[i] = True

            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            scaler_d3 = StandardScaler(with_mean=True, with_std=True)
            Xdv3 = scaler_d3.fit_transform(Xd3[maskd3])

            kd3 = int(args.deep_pcs)
            kd3 = max(1, min(kd3, Xdv3.shape[1], Xdv3.shape[0]))
            pca_d3 = PCA(n_components=kd3, random_state=0)
            Zdv3 = pca_d3.fit_transform(Xdv3)

            Zall_d3 = np.full((len(df), kd3), np.nan, dtype=np.float32)
            Zall_d3[maskd3] = Zdv3

            for j in range(kd3):
                df[f"dl_resnet50_layer3_pc{str(j+1).zfill(2)}"] = Zall_d3[:, j]

            for j in range(kd3, int(args.deep_pcs)):
                df[f"dl_resnet50_layer3_pc{str(j+1).zfill(2)}"] = np.nan

    # Drop helper
    df = df.drop(columns=["_img_base"], errors="ignore")

    # Write updated CSV (overwrite)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Updated CSV written: {csv_path}")

    if missing:
        print(f"[WARN] Missing images: {len(missing)}")
        for fn in missing[:25]:
            print(f"  - {fn}")
        if len(missing) > 25:
            print(f"  ... and {len(missing)-25} more")

    # Write manifest
    manifest_rows = [
        ("ll_lbp_hist_json", "Local Binary Pattern histogram for texture, JSON-encoded list"),
        ("ll_mean_luminance", "Mean grayscale intensity"),
        ("ll_rms_contrast", "Std of grayscale intensity"),
        ("ll_lab_L_mean", "Mean Lab L* (lightness)"),
        ("ll_lab_L_std", "Std Lab L*"),
        ("ll_lab_a_mean", "Mean Lab a* (green-red)"),
        ("ll_lab_a_std", "Std Lab a*"),
        ("ll_lab_b_mean", "Mean Lab b* (blue-yellow)"),
        ("ll_lab_b_std", "Std Lab b*"),
        ("ll_hsv_s_mean", "Mean HSV saturation"),
        ("ll_edge_energy", "Mean Sobel edge magnitude"),
        ("ll_power_radial_30_json", "Radially-averaged log power spectrum (30 bins), JSON-encoded list"),
    ]
    for k in range(1, int(args.hog_pcs) + 1):
        manifest_rows.append((f"ll_hog_pc{str(k).zfill(2)}", f"HOG descriptor PCA component {k}"))

    if bool(args.deepnet_resnet50):
        for k in range(1, int(args.deep_pcs) + 1):
            manifest_rows.append(
                (f"dl_resnet50_conv1_pc{str(k).zfill(2)}", f"ResNet-50 conv1 pooled channels PCA component {k}")
            )
    if bool(getattr(args, "deepnet_resnet50_layer3", False)):
        for k in range(1, int(args.deep_pcs) + 1):
            manifest_rows.append(
                (f"dl_resnet50_layer3_pc{str(k).zfill(2)}", f"ResNet-50 layer3 pooled channels PCA component {k}")
            )

    manifest = pd.DataFrame(manifest_rows, columns=["column", "description"])
    manifest_path = repo_root / "analysis_code" / "reference" / "visual_models_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"[OK] Manifest written: {manifest_path}")

    # --- Build model RDMs (optional) ---
    if getattr(args, "build_rdms", False):
        rdm_outdir = Path(args.rdm_outdir).expanduser().resolve()
        rdm_outdir.mkdir(parents=True, exist_ok=True)

        # Labels in the same order as the CSV
        labels = df[img_col].astype(str).apply(_basename).tolist()

        index_rows = []

        def _write(name: str, Dm: np.ndarray, cols: List[str], kind: str):
            # Use the same naming convention as other scripts: files starting with `rdm_`
            outp = rdm_outdir / f"rdm_ll_{name}.npz"
            save_rdm_npz(outp, Dm, labels, kind=kind, meta={"feature_columns": cols})
            index_rows.append({"file": outp.name, "model_name": f"ll_{name}", "kind": kind, "feature_columns": "+".join(cols)})
            print(f"[OK] Wrote RDM: {outp}")

        # 1) Scalar models
        scalar_models = [
            ("mean_luminance", "ll_mean_luminance"),
            ("rms_contrast", "ll_rms_contrast"),
            ("hsv_s_mean", "ll_hsv_s_mean"),
            ("edge_energy", "ll_edge_energy"),
        ]
        for name, col in scalar_models:
            if col in df.columns:
                Dm = rdm_from_scalar(pd.to_numeric(df[col], errors="coerce").to_numpy())
                _write(name, Dm, [col], kind="lowlevel_scalar")

        # 2) Lab mean/std vectors
        lab_mean_cols = ["ll_lab_L_mean", "ll_lab_a_mean", "ll_lab_b_mean"]
        if all(c in df.columns for c in lab_mean_cols):
            Xlab = df[lab_mean_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            _write("lab_mean", rdm_from_vectors(Xlab, min_overlap=3), lab_mean_cols, kind="lowlevel_vector")

        lab_std_cols = ["ll_lab_L_std", "ll_lab_a_std", "ll_lab_b_std"]
        if all(c in df.columns for c in lab_std_cols):
            Xlab = df[lab_std_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            _write("lab_std", rdm_from_vectors(Xlab, min_overlap=3), lab_std_cols, kind="lowlevel_vector")

        # 3) Power spectrum vector (30 bins)
        ps_col = "ll_power_radial_30_json"
        if ps_col in df.columns:
            ps = []
            for s in df[ps_col].astype(str).tolist():
                try:
                    v = json.loads(s)
                    if isinstance(v, list):
                        ps.append([float(x) if x is not None else np.nan for x in v])
                    else:
                        ps.append([np.nan] * 30)
                except Exception:
                    ps.append([np.nan] * 30)
            Xps = np.asarray(ps, dtype=float)
            _write("power_radial30", rdm_from_vectors(Xps, min_overlap=10), [ps_col], kind="lowlevel_vector")

        # 4) HOG PCs vector
        hog_cols = [f"ll_hog_pc{str(k).zfill(2)}" for k in range(1, int(args.hog_pcs) + 1)]
        hog_cols_present = [c for c in hog_cols if c in df.columns]
        if len(hog_cols_present) >= 2:
            Xhog = df[hog_cols_present].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            _write(
                f"hog_pcs{len(hog_cols_present)}",
                rdm_from_vectors(Xhog, min_overlap=max(3, min(10, len(hog_cols_present)))),
                hog_cols_present,
                kind="lowlevel_vector"
            )

        # LBP vector
        lbp_col = "ll_lbp_hist_json"
        if lbp_col in df.columns:
            lbps = []
            for s in df[lbp_col].astype(str).tolist():
                try:
                    v = json.loads(s)
                    if isinstance(v, list):
                        lbps.append([float(x) if x is not None else np.nan for x in v])
                    else:
                        lbps.append([np.nan] * getattr(args, "lbp_bins", 10))
                except Exception:
                    lbps.append([np.nan] * getattr(args, "lbp_bins", 10))
            Xlbp = np.asarray(lbps, dtype=float)
            _write("lbp_hist", rdm_from_vectors(Xlbp, min_overlap=3), [lbp_col], kind="lowlevel_vector")

        # 5) Deepnet ResNet-50 conv1 PCs vector
        if bool(getattr(args, "deepnet_resnet50", False)):
            deep_cols = [f"dl_resnet50_conv1_pc{str(k).zfill(2)}" for k in range(1, int(args.deep_pcs) + 1)]
            deep_cols_present = [c for c in deep_cols if c in df.columns]
            if len(deep_cols_present) >= 2:
                Xdeep = df[deep_cols_present].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                _write(
                    f"resnet50_conv1_pcs{len(deep_cols_present)}",
                    rdm_from_vectors(Xdeep, min_overlap=max(3, min(10, len(deep_cols_present)))),
                    deep_cols_present,
                    kind="lowlevel_vector",
                )
        if bool(getattr(args, "deepnet_resnet50_layer3", False)):
            deep_cols3 = [f"dl_resnet50_layer3_pc{str(k).zfill(2)}" for k in range(1, int(args.deep_pcs) + 1)]
            deep_cols_present3 = [c for c in deep_cols3 if c in df.columns]
            if len(deep_cols_present3) >= 2:
                Xdeep3 = df[deep_cols_present3].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                _write(
                    f"resnet50_layer3_pcs{len(deep_cols_present3)}",
                    rdm_from_vectors(Xdeep3, min_overlap=max(3, min(10, len(deep_cols_present3)))),
                    deep_cols_present3,
                    kind="lowlevel_vector",
                )

        # Write index
        index_df = pd.DataFrame(index_rows)
        index_path = rdm_outdir / "low_level_visual_rdms_index.csv"
        index_df.to_csv(index_path, index=False)
        print(f"[OK] Wrote RDM index: {index_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())