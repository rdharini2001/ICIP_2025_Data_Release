"""
Clean hero/teaser banner for the Hugging Face dataset card: real RGB-IR
pairs only, no per-tile text clutter, a thin bottom caption bar with the
headline numbers. Reuses the same real-example selection logic as
generate_diversity_montage.py but renders a tighter, wider strip suited to
a social-card / page-header image rather than a dense supplementary grid.
"""
import glob
import os
import random
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")

OUT_DIR = r"C:\Users\dragha4\avid_repo\paper\figures"
RAW_ROOT = r"E:\ICIP_2025\raw_extracted\releasev1\releasev1-detection&tracking\Train"


def list_videos():
    img_dir = os.path.join(RAW_ROOT, "RGB", "images")
    vids = {}
    for fname in os.listdir(img_dir):
        parts = fname.split("_")
        if parts[0] not in ("DRONE", "BIRD"):
            continue
        vid = f"{parts[0]}_{parts[1]}"
        frame = parts[2]
        vids.setdefault(vid, set()).add(frame)
    return vids


def find_files(video_id, frame_idx):
    rgb_glob = os.path.join(RAW_ROOT, "RGB", "images", f"{video_id}_{frame_idx}_png.rf.*.jpg")
    ir_glob = os.path.join(RAW_ROOT, "IR", "images", f"IR_{video_id}_{frame_idx}_png.rf.*.jpg")
    rgb_matches = sorted(glob.glob(rgb_glob), key=os.path.getsize)
    ir_matches = sorted(glob.glob(ir_glob), key=os.path.getsize)
    if not rgb_matches or not ir_matches:
        return None, None
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


def pick_examples(n=6, seed=11):
    vids = list_videos()
    rng = random.Random(seed)
    all_vids = sorted(vids.keys())
    rng.shuffle(all_vids)
    examples = []
    for vid in all_vids:
        if len(examples) >= n:
            break
        frames = list(vids[vid])
        rng.shuffle(frames)
        for frame in frames[:15]:
            rgb_path, ir_path = find_files(vid, frame)
            if rgb_path is None:
                continue
            boxes = load_yolo_boxes(rgb_path)
            if len(boxes) == 1:
                examples.append((rgb_path, ir_path))
                break
    return examples


def fig_hf_teaser(n=6, seed=11):
    examples = pick_examples(n=n, seed=seed)
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.3), gridspec_kw={"wspace": 0.03, "hspace": 0.03})

    for col, (rgb_path, ir_path) in enumerate(examples):
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        h, w = rgb.shape[:2]
        boxes = load_yolo_boxes(rgb_path)

        for ax, img, cmap in [(axes[0, col], rgb, None), (axes[1, col], ir, "inferno")]:
            ax.imshow(img, cmap=cmap)
            for cls_id, cx, cy, bw, bh in boxes:
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                pad = 6
                rect = mpatches.Rectangle((x1 - pad, y1 - pad), bw * w + 2 * pad, bh * h + 2 * pad,
                                           linewidth=1.6, edgecolor="#39FF14", facecolor="none")
                ax.add_patch(rect)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.patch.set_facecolor("white")
    fig.suptitle("")
    fig.tight_layout(rect=[0, 0.05, 1, 1], pad=0.3)
    fig.text(0.5, 0.015,
             "AVID -- 32,040 real RGB-IR pairs  |  weakly aligned (not pixel-registered)  |  "
             "68.1% of targets 8-16px  |  AVID-C: 7 x 5 corruption benchmark",
             ha="center", va="bottom", fontsize=11, color="#222222")
    fig.savefig(f"{OUT_DIR}/fig_hf_teaser.png", dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote fig_hf_teaser.png")


if __name__ == "__main__":
    fig_hf_teaser()
