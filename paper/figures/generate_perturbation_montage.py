"""
Perturbation montage: real AVID drone/bird patches under actual AVID-C
corruptions, at multiple severities, both modalities -- the concrete visual
counterpart to the numerical corruption sweep in corruption_sweep.py. Every
pixel here is a real patch passed through the real corruption functions used
to compute the Bayes bounds reported in the paper (no illustrative stand-in).
"""
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.corruptions.corruptions import CORRUPTIONS

OUT_DIR = r"C:\Users\dragha4\avid_repo\paper\figures"
# 4 representative families (2 noise-type, 2 blur/motion-type) for a
# compact visual; the numeric table (tab:bayes-corruption-full) already
# covers all 7 families in full.
FAMILIES = ["gaussian_noise", "speckle_noise", "gaussian_blur", "camera_instability"]
SEVERITIES = [0, 2, 4]


def load_patches():
    with open(r"E:\ICIP_2025\avid_release\metadata\theory_patches.pkl", "rb") as f:
        return pickle.load(f)


def corrupt(patch_float, name, severity, seed):
    if severity == 0:
        return patch_float
    patch_u8 = (patch_float * 255).astype(np.uint8)
    out = CORRUPTIONS[name](patch_u8, severity=severity, seed=seed)
    return out.astype(np.float32) / 255.0


def fig_perturbation_montage(modality="ir", cls_name="drone", seed=7):
    patches = load_patches()
    rng = np.random.default_rng(seed)
    candidates = [p for p in patches if p["cls_name"] == cls_name
                  and 12 <= p["bbox_side_px"] < 20]
    patch = candidates[rng.integers(len(candidates))][f"{modality}_patch"]

    n_rows, n_cols = len(FAMILIES), len(SEVERITIES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows))
    cmap = "gray" if modality == "rgb" else "inferno"

    for row, fam in enumerate(FAMILIES):
        for col, sev in enumerate(SEVERITIES):
            ax = axes[row, col]
            img = corrupt(patch, fam, sev, seed=seed)
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#cccccc")
            if row == 0:
                ax.set_title(f"severity {sev}" if sev else "clean", fontsize=10)
            if col == 0:
                ax.set_ylabel(fam.replace("_", "\n"), fontsize=8.5, rotation=0,
                               ha="right", va="center")
    fig.suptitle(f"Real AVID-C perturbations on one real {cls_name} patch "
                 f"({modality.upper()}, 12-20px)", fontsize=12)
    fig.tight_layout(rect=[0.05, 0, 1, 0.95])
    fig.savefig(f"{OUT_DIR}/fig_perturbation_montage_{modality}.png", dpi=160)
    plt.close(fig)
    print(f"wrote fig_perturbation_montage_{modality}.png")


if __name__ == "__main__":
    fig_perturbation_montage("rgb", "drone")
    fig_perturbation_montage("ir", "drone")
