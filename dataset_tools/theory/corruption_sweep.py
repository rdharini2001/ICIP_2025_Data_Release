"""
Experiment 2: Bayes-error bound as a function of AVID-C corruption
severity, applied directly to the real extracted patches (not full
frames -- corruption at the patch level isolates the effect on the
target's own appearance from any frame-level context, and is drastically
cheaper: a few thousand 24x24 patches vs. tens of thousands of 320x256
frames). No classifier is trained; only the corruption functions
(already implemented and unit-tested in dataset_tools/corruptions/) and
the closed-form Bayes bound from bayes_bounds.py are used.
"""
import pickle
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.corruptions.corruptions import CORRUPTIONS
from dataset_tools.theory.bayes_bounds import bayes_error_bound_gaussian
from dataset_tools.theory.patch_extraction import fit_pca, project


def load_patches():
    with open(r"E:\ICIP_2025\avid_release\metadata\theory_patches.pkl", "rb") as f:
        return pickle.load(f)


def corrupt_patch(patch_float, name, severity, seed):
    """patch_float: (H,W) in [0,1]. AVID-C functions expect uint8 [0,255]."""
    patch_u8 = (patch_float * 255).astype(np.uint8)
    out = CORRUPTIONS[name](patch_u8, severity=severity, seed=seed)
    return out.astype(np.float32) / 255.0


def run_corruption_sweep(modality="rgb", n_sample=1500, seed=0):
    patches = load_patches()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(patches))[:n_sample]
    patches = [patches[i] for i in idx]
    key = f"{modality}_patch"

    # fit ONE PCA basis on the clean data, and reuse it for every corrupted
    # condition -- otherwise a change in the bound could just reflect a
    # changing basis rather than a real loss of class information.
    clean_flat = np.stack([p[key].reshape(-1) for p in patches])
    mean, components, _ = fit_pca(clean_flat)

    results = {}
    for cond_name, severity in [("clean", 0)] + [(c, s) for c in CORRUPTIONS for s in range(1, 6)]:
        feats_bird, feats_drone = [], []
        for i, p in enumerate(patches):
            patch = p[key] if severity == 0 else corrupt_patch(p[key], cond_name, severity, seed=seed + i)
            feat = project(patch.reshape(1, -1), mean, components)[0]
            (feats_bird if p["cls_name"] == "bird" else feats_drone).append(feat)
        bound = bayes_error_bound_gaussian(np.array(feats_bird), np.array(feats_drone))
        key_name = "clean" if severity == 0 else f"{cond_name}_s{severity}"
        results[key_name] = bound
        print(f"{modality} {key_name}: bayes_bound={bound['bayes_error_bound']:.4f}")
    return results


if __name__ == "__main__":
    import json

    all_results = {}
    for modality in ["rgb", "ir"]:
        print(f"=== {modality} ===")
        all_results[modality] = run_corruption_sweep(modality)

    with open(r"E:\ICIP_2025\avid_release\metadata\theory_corruption_sweep.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("wrote theory_corruption_sweep.json")
