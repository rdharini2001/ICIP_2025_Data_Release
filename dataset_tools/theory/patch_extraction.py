"""
Extracts real per-instance RGB and IR patches (bird/drone) from the real
AVID corpus, resized to a canonical size and reduced to a compact,
interpretable feature vector, for the Bayes-risk analysis in
dataset_tools/theory/bayes_bounds.py. No model training involved -- this
is image I/O and simple linear algebra (PCA), which is why it runs in
minutes rather than hours.

Feature representation: each patch is resized to CANONICAL_SIZE, converted
to grayscale intensity (for a modality-comparable representation -- RGB and
IR are otherwise different channel counts), flattened, and projected onto
the top-K principal components fit per (modality, split) -- a classical,
information-preserving-in-the-mean-square-sense dimensionality reduction
that keeps the downstream Gaussian/KDE density estimation tractable
(fitting a full-rank Gaussian in a 256+-dim raw-pixel space from a few
thousand samples is ill-posed; the covariance would be singular).
"""

import json
import os
import sys

import cv2
import numpy as np

CANONICAL_SIZE = 24  # px, chosen larger than the ~8-16px native object size
                       # so downsampling doesn't discard the little real
                       # detail these targets have, but small enough that a
                       # few thousand real instances comfortably estimate a
                       # low-rank covariance after PCA.
N_PCA_COMPONENTS = 8


def load_pairs(split):
    records = []
    with open(rf"E:\ICIP_2025\avid_release\metadata\pairs_{split}.jsonl") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_labels(rgb_path):
    label_path = rgb_path.replace("\\images\\", "\\labels\\").rsplit(".", 1)[0] + ".txt"
    boxes = []
    if os.path.isfile(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
                boxes.append((cls, cx, cy, w, h))
    return boxes


CLASS_NAMES = {0: "bird", 1: "drone"}


def extract_patch(img, cx, cy, w, h, pad_factor=1.5):
    """Crop a slightly padded square patch around the box center (padding
    gives the density estimator a little context beyond the exact box,
    which for 8-16px objects is often just a handful of pixels), resize to
    CANONICAL_SIZE, return as a normalized grayscale float32 array."""
    H, W = img.shape[:2]
    side = max(w * W, h * H) * pad_factor
    x1 = int(np.clip(cx * W - side / 2, 0, W))
    y1 = int(np.clip(cy * H - side / 2, 0, H))
    x2 = int(np.clip(cx * W + side / 2, 0, W))
    y2 = int(np.clip(cy * H + side / 2, 0, H))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = img[y1:y2, x1:x2]
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    patch = cv2.resize(patch, (CANONICAL_SIZE, CANONICAL_SIZE), interpolation=cv2.INTER_AREA)
    return (patch.astype(np.float32) / 255.0)


def extract_all_patches(splits=("train", "val"), max_per_split=None, seed=0):
    """Returns a list of dicts: {video_id, frame_idx, cls, cls_name,
    rgb_patch (24x24 float32), ir_patch (24x24 float32), bbox_side_px}."""
    rng = np.random.default_rng(seed)
    results = []
    for split in splits:
        records = load_pairs(split)
        if max_per_split:
            idx = rng.permutation(len(records))[:max_per_split]
            records = [records[i] for i in idx]
        for i, rec in enumerate(records):
            boxes = load_labels(rec["rgb_path"])
            if not boxes:
                continue
            rgb_img = cv2.cvtColor(cv2.imread(rec["rgb_path"]), cv2.COLOR_BGR2RGB)
            ir_img = cv2.imread(rec["ir_path"], cv2.IMREAD_GRAYSCALE)
            if rgb_img is None or ir_img is None:
                continue
            img_h, img_w = rgb_img.shape[:2]
            for cls, cx, cy, w, h in boxes:
                rgb_patch = extract_patch(rgb_img, cx, cy, w, h)
                ir_patch = extract_patch(ir_img, cx, cy, w, h)
                if rgb_patch is None or ir_patch is None:
                    continue
                side_px = float(np.sqrt(w * img_w * h * img_h))
                results.append({
                    "video_id": rec["video_id"], "frame_idx": rec["frame_idx"], "split": split,
                    "cls": cls, "cls_name": CLASS_NAMES.get(cls, str(cls)),
                    "rgb_patch": rgb_patch, "ir_patch": ir_patch, "bbox_side_px": side_px,
                })
            if i % 2000 == 0:
                print(f"[{split}] {i}/{len(records)} frames processed, {len(results)} patches so far")
    return results


def fit_pca(patches_flat, n_components=N_PCA_COMPONENTS):
    """patches_flat: (N, D). Returns (mean, components) with components of
    shape (n_components, D); project new data via (x - mean) @ components.T"""
    mean = patches_flat.mean(axis=0)
    centered = patches_flat - mean
    # SVD-based PCA, numerically stable for D > N (our case: D=576, N~thousands, fine either way)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    explained_var_ratio = (s[:n_components] ** 2) / np.sum(s ** 2)
    return mean, components, explained_var_ratio


def project(patches_flat, mean, components):
    return (patches_flat - mean) @ components.T


if __name__ == "__main__":
    import pickle

    patches = extract_all_patches(max_per_split=None)
    print(f"Total patches extracted: {len(patches)}")
    with open(r"E:\ICIP_2025\avid_release\metadata\theory_patches.pkl", "wb") as f:
        pickle.dump(patches, f)
    print("wrote theory_patches.pkl")
