"""
Two additional real-data checks on top of the closed-form Gaussian
Bhattacharyya bound reported in scale_sweep.py, both at the corpus's modal
12-16px bucket (17,706 instances, the largest well-populated bucket):

1. Optimized Chernoff bound (chernoff_gaussian, s swept over (0,1)) vs. the
   Bhattacharyya bound (the closed-form special case s=0.5) -- quantifies
   how much tighter the general Chernoff family gets when s is actually
   optimized rather than fixed at 1/2.
2. Nonparametric KDE cross-check (bayes_error_bound_kde) computed directly
   on these real features, not just on synthetic self-test data -- a real,
   not merely synthetic, check on the Gaussian assumption.
"""
import pickle
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.theory.bayes_bounds import (
    bayes_error_bound_gaussian, bayes_error_bound_kde, optimized_chernoff_bound_gaussian,
)
from dataset_tools.theory.scale_sweep import build_feature_space, load_patches


def run(bucket_lo=12, bucket_hi=16, n_kde_sample=2000, seed=0):
    patches = load_patches()
    bucket_mask = np.array([bucket_lo <= p["bbox_side_px"] < bucket_hi for p in patches])
    bird_mask_full = np.array([p["cls_name"] == "bird" for p in patches])
    print(f"bucket [{bucket_lo},{bucket_hi})px: n={bucket_mask.sum()} "
          f"(bird={(bucket_mask & bird_mask_full).sum()}, "
          f"drone={(bucket_mask & ~bird_mask_full).sum()})")

    results = {}
    for modality in ["rgb", "ir", "fusion"]:
        # Same global PCA basis (fit once on the full pooled corpus) as
        # scale_sweep.py, so this bucket's bound is directly comparable to
        # the headline scale-sweep numbers rather than a different number
        # from a bucket-local basis.
        feats, _, _ = build_feature_space(patches, modality)
        feats = feats[bucket_mask]
        bird_mask = bird_mask_full[bucket_mask]
        f0, f1 = feats[bird_mask], feats[~bird_mask]

        gauss = bayes_error_bound_gaussian(f0, f1)
        chernoff = optimized_chernoff_bound_gaussian(f0, f1)

        rng = np.random.default_rng(seed)
        idx0 = rng.permutation(len(f0))[:n_kde_sample]
        idx1 = rng.permutation(len(f1))[:n_kde_sample]
        kde = bayes_error_bound_kde(f0[idx0], f1[idx1], n_mc=5000, seed=seed)

        results[modality] = {
            "bhattacharyya_bound": gauss["bayes_error_bound"],
            "chernoff_bound": chernoff["bayes_error_bound"],
            "chernoff_optimal_s": chernoff["optimal_s"],
            "kde_bound": kde["bayes_error_bound"],
            "kde_n_sample": int(min(len(f0), n_kde_sample) + min(len(f1), n_kde_sample)),
        }
        r = results[modality]
        print(f"{modality:>6}: Bhattacharyya={r['bhattacharyya_bound']:.4f}  "
              f"Chernoff(opt s={r['chernoff_optimal_s']:.2f})={r['chernoff_bound']:.4f}  "
              f"KDE(n={r['kde_n_sample']})={r['kde_bound']:.4f}")
    return results


if __name__ == "__main__":
    import json
    result = run()
    with open(r"E:\ICIP_2025\avid_release\metadata\real_cross_checks.json", "w") as f:
        json.dump(result, f, indent=2)
    print("wrote real_cross_checks.json")
