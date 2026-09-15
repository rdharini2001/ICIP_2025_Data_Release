"""
Experiment 1: Bayes-error bound as a function of object scale.

Uses the real patches extracted by patch_extraction.py, bucketed by the
same fine-grained pixel-side-length bins as
dataset_tools/hf_release/bbox_histogram.py, so this plugs directly into
the dataset's own real small-object-scale distribution rather than an
arbitrary binning. For each bucket and each modality (RGB, IR, and the
joint RGB+IR fusion feature), fits class-conditional Gaussians to
bird-vs-drone patches and computes the Bhattacharyya Bayes-error bound --
no classifier is trained anywhere in this script.
"""
import pickle
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.theory.bayes_bounds import bayes_error_bound_gaussian, bayes_error_bound_kde
from dataset_tools.theory.patch_extraction import fit_pca, project, CANONICAL_SIZE

BIN_EDGES_PX = [0, 8, 12, 16, 20, 24, 28, 32, 48, 1e9]
BIN_LABELS = [f"{int(BIN_EDGES_PX[i])}-{int(BIN_EDGES_PX[i+1]) if BIN_EDGES_PX[i+1] < 1e9 else 'inf'}" for i in range(len(BIN_EDGES_PX) - 1)]


def bin_index(side_px):
    for i in range(len(BIN_EDGES_PX) - 1):
        if BIN_EDGES_PX[i] <= side_px < BIN_EDGES_PX[i + 1]:
            return i
    return len(BIN_EDGES_PX) - 2


def load_patches():
    with open(r"E:\ICIP_2025\avid_release\metadata\theory_patches.pkl", "rb") as f:
        return pickle.load(f)


def build_feature_space(patches, modality, pca_mean=None, pca_components=None):
    """modality: 'rgb', 'ir', or 'fusion' (concatenated rgb+ir PCA features).
    Fits PCA once on the FULL pooled dataset (both classes, all scales) if
    not provided, so projections are comparable across scale/severity
    buckets rather than each bucket getting its own basis."""
    if modality == "fusion":
        rgb_feats, rgb_mean, rgb_comp = build_feature_space(patches, "rgb")
        ir_feats, ir_mean, ir_comp = build_feature_space(patches, "ir")
        return np.concatenate([rgb_feats, ir_feats], axis=1), (rgb_mean, ir_mean), (rgb_comp, ir_comp)

    key = f"{modality}_patch"
    flat = np.stack([p[key].reshape(-1) for p in patches])
    if pca_mean is None:
        mean, components, _ = fit_pca(flat)
    else:
        mean, components = pca_mean, pca_components
    feats = project(flat, mean, components)
    return feats, mean, components


def run_scale_sweep(modality="rgb", min_n_per_class=15):
    patches = load_patches()
    print(f"loaded {len(patches)} total patches")

    # fit PCA basis once, globally, so every bucket's Bhattacharyya bound is
    # computed in the SAME feature space (otherwise a shrinking bound across
    # buckets could just reflect a changing basis, not a real effect).
    feats, mean, components = build_feature_space(patches, modality)

    buckets = {label: {"bird": [], "drone": []} for label in BIN_LABELS}
    for feat, p in zip(feats, patches):
        b = BIN_LABELS[bin_index(p["bbox_side_px"])]
        buckets[b][p["cls_name"]].append(feat)

    results = []
    for label in BIN_LABELS:
        bird = np.array(buckets[label]["bird"])
        drone = np.array(buckets[label]["drone"])
        if len(bird) < min_n_per_class or len(drone) < min_n_per_class:
            results.append({"bucket": label, "n_bird": len(bird), "n_drone": len(drone), "skipped": True})
            continue
        gauss = bayes_error_bound_gaussian(bird, drone)
        row = {"bucket": label, "n_bird": len(bird), "n_drone": len(drone), "skipped": False, **gauss}
        results.append(row)
        print(f"{label}px: n_bird={len(bird)} n_drone={len(drone)} bayes_bound={row['bayes_error_bound']:.4f}")
    return results


if __name__ == "__main__":
    import json

    all_results = {}
    for modality in ["rgb", "ir", "fusion"]:
        print(f"=== {modality} ===")
        all_results[modality] = run_scale_sweep(modality)

    with open(r"E:\ICIP_2025\avid_release\metadata\theory_scale_sweep.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("wrote theory_scale_sweep.json")
