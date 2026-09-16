"""
Supplementary figure: full-resolution (320x256, native) AVID-C corruption
grids for BOTH modalities on the SAME real frame, so RGB and IR can be
compared side by side under identical corruption severities. This is the
full-frame counterpart to fig_perturbation_montage_{rgb,ir}.png (which use
tiny 24x24 canonical patches -- necessarily blurry once enlarged, since
that IS this dataset's real, extreme small-object scale, not a rendering
artifact of the montage). Full frames make the same RGB/IR corruption
asymmetry visible at a scale a reader can actually see texture in.
"""
import glob
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.corruptions.corruptions import CORRUPTIONS, apply_corruption

OUT_DIR = r"C:\Users\dragha4\avid_repo\paper\figures"
RAW_ROOT = r"E:\ICIP_2025\raw_extracted\releasev1\releasev1-detection&tracking\Train"

VIDEO_ID, FRAME_IDX = "DRONE_001111", "001"


def find_files(video_id, frame_idx):
    rgb_glob = os.path.join(RAW_ROOT, "RGB", "images", f"{video_id}_{frame_idx}_png.rf.*.jpg")
    ir_glob = os.path.join(RAW_ROOT, "IR", "images", f"IR_{video_id}_{frame_idx}_png.rf.*.jpg")
    rgb_matches = sorted(glob.glob(rgb_glob), key=os.path.getsize)
    ir_matches = sorted(glob.glob(ir_glob), key=os.path.getsize)
    return rgb_matches[0], ir_matches[0]


def fig_corruption_grid(img, cmap, out_name, title):
    names = list(CORRUPTIONS.keys())
    fig, axes = plt.subplots(len(names), 6, figsize=(9, 1.35 * len(names)))
    for row, name in enumerate(names):
        axes[row, 0].imshow(img, cmap=cmap)
        axes[row, 0].set_ylabel(name.replace("_", "\n"), fontsize=8, rotation=0, ha="right", va="center")
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        if row == 0:
            axes[row, 0].set_title("clean", fontsize=9)
        for sev in range(1, 6):
            out = apply_corruption(img, name, severity=sev, seed=sev)
            axes[row, sev].imshow(out, cmap=cmap)
            axes[row, sev].set_xticks([]); axes[row, sev].set_yticks([])
            if row == 0:
                axes[row, sev].set_title(f"severity {sev}", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT_DIR, out_name), dpi=160)
    plt.close(fig)
    print(f"wrote {out_name}")


def main():
    rgb_path, ir_path = find_files(VIDEO_ID, FRAME_IDX)
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

    fig_corruption_grid(rgb, None, "fig_corruption_grid_rgb_fullframe.png",
                        "AVID-C corruption suite on a real full-resolution RGB frame, severities 1-5")
    fig_corruption_grid(ir, "inferno", "fig_corruption_grid_ir_fullframe.png",
                        "AVID-C corruption suite on a real full-resolution IR frame, severities 1-5")


if __name__ == "__main__":
    main()
