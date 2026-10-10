"""
Feature-space visualization: real bird/drone patches projected onto (x)
the Fisher/Bhattacharyya discriminant direction w = Sigma_avg^-1(mu1-mu0)
-- the exact direction the Mahalanobis term in Eq. "eq:bhatt-gaussian" is
computed along -- and (y) the top variance direction orthogonal to it, at
the corpus's modal 12-16px bucket, for RGB, IR, and the joint fusion
feature. This is the direct visual counterpart to Table
"tab:bayes-scale-full": it answers *why* the Bayes bound differs so much
across modalities by showing the actual class-conditional point clouds
along the specific axis the bound formula is computed from, rather than
an arbitrary unsupervised PCA view that need not align with class
separation at all (checked: raw PC1/PC2 of an 8-dim PCA space showed much
weaker visual separation than this discriminant projection, since PCA
maximizes variance, not label separability).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.theory.bayes_bounds import fit_gaussian
from dataset_tools.theory.scale_sweep import build_feature_space, load_patches


def discriminant_projection(feats, bird_mask):
    """Returns 2D coordinates: x = projection onto w = Sigma_avg^-1(mu1-mu0)
    (normalized), y = projection onto the top variance direction orthogonal
    to w (for visual spread only; not part of the Bayes-bound computation)."""
    f0, f1 = feats[bird_mask], feats[~bird_mask]
    mean0, cov0 = fit_gaussian(f0)
    mean1, cov1 = fit_gaussian(f1)
    cov_avg = (cov0 + cov1) / 2
    w = np.linalg.solve(cov_avg, mean1 - mean0)
    w = w / np.linalg.norm(w)

    centered = feats - feats.mean(axis=0)
    x = centered @ w
    # remove the w-component, then take the top principal direction of the
    # residual for the y-axis (visual spread only)
    residual = centered - np.outer(x, w)
    _, _, vt = np.linalg.svd(residual, full_matrices=False)
    y = residual @ vt[0]
    return x, y

OUT_DIR = r"C:\Users\dragha4\avid_repo\paper\figures"
COLORS = {"bird": "#CC79A7", "drone": "#0072B2"}

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
})


def panel_label(ax, letter):
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=15,
             fontweight="bold", va="top", ha="left")


def fig_feature_space(bucket_lo=12, bucket_hi=16, seed=0, n_plot_per_class=1200):
    patches = load_patches()
    bucket_mask = np.array([bucket_lo <= p["bbox_side_px"] < bucket_hi for p in patches])
    bird_mask_full = np.array([p["cls_name"] == "bird" for p in patches])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    rng = np.random.default_rng(seed)

    for ax, modality in zip(axes, ["rgb", "ir", "fusion"]):
        feats, _, _ = build_feature_space(patches, modality)
        feats = feats[bucket_mask]
        bird_mask = bird_mask_full[bucket_mask]
        x, y = discriminant_projection(feats, bird_mask)

        for cls_name, mask, color in [("bird", bird_mask, COLORS["bird"]),
                                       ("drone", ~bird_mask, COLORS["drone"])]:
            idx = np.where(mask)[0]
            if len(idx) > n_plot_per_class:
                idx = rng.choice(idx, n_plot_per_class, replace=False)
            ax.scatter(x[idx], y[idx], s=6, alpha=0.35, color=color,
                       label=cls_name, linewidths=0)

        ax.set_xlabel("Discriminant direction $w$")
        ax.set_ylabel("Orthogonal spread" if modality == "rgb" else "")
        title = {"rgb": "RGB", "ir": "IR", "fusion": "Fusion (RGB+IR joint)"}[modality]
        ax.set_title(title)

    axes[0].legend(markerscale=3, fontsize=9, loc="upper right")
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter)
    fig.suptitle("Real bird/drone patches projected onto the Bhattacharyya\n"
                 "discriminant direction (same pipeline behind every Bayes bound), 12-16px bucket", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT_DIR}/fig_feature_space.png", dpi=160)
    plt.close(fig)
    print("wrote fig_feature_space.png")


if __name__ == "__main__":
    fig_feature_space()
