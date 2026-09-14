"""
Real dataset-sample figure: a small grid of actual RGB/IR pairs with their
ground-truth boxes drawn, replacing the synthetic stand-in figure used
before real data was available. Also regenerates the corruption-severity
figure on a REAL frame instead of the synthetic blob image.
"""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.corruptions.corruptions import CORRUPTIONS, apply_corruption

OUT_DIR = r"C:\Users\dragha4\avid_repo\paper\figures"

# a handful of manually-picked, visually-clear real examples (mix of drone/bird)
SAMPLES = [
    ("DRONE_001111", "001", "DRONE"),
    ("BIRD_00160", "001", "BIRD"),
]

RAW_ROOT = r"E:\ICIP_2025\raw_extracted\releasev1\releasev1-detection&tracking\Train"


def find_files(video_id, frame_idx, cls):
    import glob
    rgb_glob = os.path.join(RAW_ROOT, "RGB", "images", f"{video_id}_{frame_idx}_png.rf.*.jpg")
    ir_glob = os.path.join(RAW_ROOT, "IR", "images", f"IR_{video_id}_{frame_idx}_png.rf.*.jpg")
    rgb_matches = sorted(glob.glob(rgb_glob), key=os.path.getsize)
    ir_matches = sorted(glob.glob(ir_glob), key=os.path.getsize)
    return rgb_matches[0], ir_matches[0]


def load_yolo_boxes(image_path):
    label_path = image_path.replace("\\images\\", "\\labels\\").rsplit(".", 1)[0] + ".txt"
    boxes = []
    if os.path.isfile(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
                boxes.append((cls_id, cx, cy, w, h))
    return boxes


CLASS_NAMES = {0: "bird", 1: "drone"}


def draw_pair(ax_rgb, ax_ir, rgb_path, ir_path, title_suffix=""):
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    h, w = rgb.shape[:2]
    boxes = load_yolo_boxes(rgb_path)

    ax_rgb.imshow(rgb)
    ax_ir.imshow(ir, cmap="inferno")
    for ax in (ax_rgb, ax_ir):
        for cls_id, cx, cy, bw, bh in boxes:
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            rect = patches.Rectangle((x1, y1), bw * w, bh * h, linewidth=1.2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, max(0, y1 - 3), CLASS_NAMES.get(cls_id, str(cls_id)), color="lime", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    ax_rgb.set_title(f"RGB{title_suffix}", fontsize=9)
    ax_ir.set_title(f"IR{title_suffix}", fontsize=9)
    return rgb


def fig_real_samples():
    fig, axes = plt.subplots(len(SAMPLES), 2, figsize=(6, 3 * len(SAMPLES)))
    if len(SAMPLES) == 1:
        axes = [axes]
    for row, (video_id, frame_idx, cls) in enumerate(SAMPLES):
        rgb_path, ir_path = find_files(video_id, frame_idx, cls)
        draw_pair(axes[row][0], axes[row][1], rgb_path, ir_path, title_suffix=f" ({cls.lower()})")
    fig.suptitle("Real AVID RGB-IR pairs with ground-truth boxes (green)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "fig_real_samples.png"), dpi=160)
    plt.close(fig)
    return rgb_path  # return last for reuse


def fig_corruption_grid_real(rgb_path):
    img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    names = list(CORRUPTIONS.keys())
    fig, axes = plt.subplots(len(names), 6, figsize=(9, 1.35 * len(names)))
    for row, name in enumerate(names):
        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(name.replace("_", "\n"), fontsize=8, rotation=0, ha="right", va="center")
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        if row == 0:
            axes[row, 0].set_title("clean", fontsize=9)
        for sev in range(1, 6):
            out = apply_corruption(img, name, severity=sev, seed=sev)
            axes[row, sev].imshow(out)
            axes[row, sev].set_xticks([]); axes[row, sev].set_yticks([])
            if row == 0:
                axes[row, sev].set_title(f"severity {sev}", fontsize=9)
    fig.suptitle("AVID-C corruption suite, severities 1-5 (real AVID frame)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT_DIR, "fig_corruption_grid.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    rgb_path = fig_real_samples()
    fig_corruption_grid_real(rgb_path)
    print("wrote fig_real_samples.png and updated fig_corruption_grid.png with real imagery")
