"""
Alignment-shift montage: one real RGB-IR patch pair, with the IR patch
subjected to the exact same rigid-shift injection used by
alignment_sweep.py's Bayes Fusion Gain sweep, at several magnitudes up to
the real measured ceiling of this corpus's misalignment (17px). This is
the visual counterpart to Figure "fig:theory-alignment" / Table
"tab:bfg-full": it makes concrete what "injected cross-modal
misalignment" actually looks like on a real target, the one perturbation
type in the paper's three (scale, corruption, alignment) that previously
had no montage.
"""
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.theory.alignment_sweep import shift_patch

OUT_DIR = r"C:\Users\dragha4\avid_repo\paper\figures"
SHIFT_LEVELS = [0.0, 2.0, 5.0, 8.0, 12.0, 17.0]


def load_patches():
    with open(r"E:\ICIP_2025\avid_release\metadata\theory_patches.pkl", "rb") as f:
        return pickle.load(f)


def fig_alignment_montage(cls_name="drone", seed=3):
    patches = load_patches()
    rng = np.random.default_rng(seed)
    candidates = [p for p in patches if p["cls_name"] == cls_name and 12 <= p["bbox_side_px"] < 20]
    patch = candidates[rng.integers(len(candidates))]
    rgb_patch, ir_patch = patch["rgb_patch"], patch["ir_patch"]

    n_cols = len(SHIFT_LEVELS)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.0 * n_cols, 4.4))

    for col, shift in enumerate(SHIFT_LEVELS):
        shift_rng = np.random.default_rng(seed * 1000 + col)
        dx = shift_rng.uniform(-shift, shift) if shift > 0 else 0.0
        dy = shift_rng.uniform(-shift, shift) if shift > 0 else 0.0
        ir_shifted = shift_patch(ir_patch, dx, dy) if shift > 0 else ir_patch

        axes[0, col].imshow(rgb_patch, cmap="gray")
        axes[1, col].imshow(ir_shifted, cmap="inferno")
        for row in [0, 1]:
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#cccccc")
        axes[0, col].set_title(f"{shift:.0f}px" if shift else "0px\n(native)", fontsize=10)

    axes[0, 0].set_ylabel("RGB\n(unchanged)", fontsize=9, rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("IR\n(shifted)", fontsize=9, rotation=0, ha="right", va="center")
    real_note = "shaded range in Fig. 10 = where 32.3% of real AVID pairs actually fall"
    fig.suptitle(f"Injected cross-modal misalignment on one real {cls_name} patch\n"
                 f"(RGB fixed, IR shifted -- {real_note})", fontsize=10.5)
    fig.tight_layout(rect=[0.04, 0, 1, 0.88])
    fig.savefig(f"{OUT_DIR}/fig_alignment_montage.png", dpi=160)
    plt.close(fig)
    print("wrote fig_alignment_montage.png")


if __name__ == "__main__":
    fig_alignment_montage()
