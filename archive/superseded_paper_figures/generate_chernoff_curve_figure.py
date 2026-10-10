"""
Visualizes the Chernoff bound P_e(s) as a function of s in (0,1)
(Eq. "eq:chernoff"), for RGB, IR, and fusion at the modal 12-16px bucket --
the direct visual counterpart to Table "tab:bayes-tightness". Marks the
Bhattacharyya point (s=0.5, the closed-form special case used everywhere
else in the paper) against the true minimum, making concrete what
"optimizing s tightens the bound" (Sec. "Tightening the bound") actually
looks like as a curve rather than two numbers in a table.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.theory.bayes_bounds import chernoff_gaussian, fit_gaussian
from dataset_tools.theory.scale_sweep import build_feature_space, load_patches

OUT_DIR = r"C:\Users\dragha4\avid_repo\paper\figures"
PALETTE = {"rgb": "#0072B2", "ir": "#D55E00", "fusion": "#009E73"}

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
})


def fig_chernoff_curve(bucket_lo=12, bucket_hi=16, n_grid=97):
    patches = load_patches()
    bucket_mask = np.array([bucket_lo <= p["bbox_side_px"] < bucket_hi for p in patches])
    bird_mask_full = np.array([p["cls_name"] == "bird" for p in patches])

    fig, ax = plt.subplots(figsize=(7, 5))
    s_grid = np.linspace(0.02, 0.98, n_grid)

    for modality in ["rgb", "ir", "fusion"]:
        feats, _, _ = build_feature_space(patches, modality)
        feats = feats[bucket_mask]
        bird_mask = bird_mask_full[bucket_mask]
        f0, f1 = feats[bird_mask], feats[~bird_mask]
        mean0, cov0 = fit_gaussian(f0)
        mean1, cov1 = fit_gaussian(f1)

        bounds = []
        for s in s_grid:
            mu_s = chernoff_gaussian(mean0, cov0, mean1, cov1, s)
            bound = (0.5 ** (1 - s)) * (0.5 ** s) * np.exp(-mu_s)
            bounds.append(bound)
        bounds = np.array(bounds)

        ax.plot(s_grid, bounds, color=PALETTE[modality], linewidth=2, label=modality.upper())
        i_min = np.argmin(bounds)
        ax.scatter([s_grid[i_min]], [bounds[i_min]], color=PALETTE[modality], s=60,
                   zorder=5, marker="*")
        i_half = np.argmin(np.abs(s_grid - 0.5))
        ax.scatter([s_grid[i_half]], [bounds[i_half]], color=PALETTE[modality], s=45,
                   zorder=5, marker="o", facecolors="none", linewidths=1.5)

    ax.axvline(0.5, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.text(0.51, 0.005, "Bhattacharyya\n(s=0.5)", fontsize=8, color="gray", va="bottom")
    ax.set_xlabel("Chernoff parameter $s$")
    ax.set_ylabel("Bayes error bound $P_e(s)$")
    ax.set_title("The Bhattacharyya bound is one point on a curve:\n"
                 "optimizing $s$ finds a strictly tighter bound")
    ax.legend(title="filled star = optimum, open circle = $s{=}0.5$", fontsize=8,
              title_fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_chernoff_curve.png", dpi=160)
    plt.close(fig)
    print("wrote fig_chernoff_curve.png")


if __name__ == "__main__":
    fig_chernoff_curve()
