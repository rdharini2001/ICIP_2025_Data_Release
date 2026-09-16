"""
Supplementary figure: a large, diverse grid of real AVID RGB-IR pairs with
ground-truth boxes, sampled across many distinct source videos (not the 2
hardcoded examples in fig_real_samples.png) to show the corpus's real
visual variety -- different backgrounds/terrain, both classes, and
single- vs. multi-instance frames. Unlike the patch-level theory figures
(24x24 canonical crops, necessarily blurry once enlarged -- that IS the
real, extreme small-object scale this dataset has, not a rendering
artifact), every image here is a full 320x256 native-resolution frame.

No scenario metadata (hilly/forest/fog/etc.) is available as a queryable
per-video field in this release (see SCHEMA.md) -- the diversity shown
here comes from sampling many distinct real video_ids directly, which
naturally spans the corpus's actual background/terrain variety.
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
CLASS_NAMES = {0: "bird", 1: "drone"}


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


def draw_pair(ax_rgb, ax_ir, rgb_path, ir_path, label):
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    h, w = rgb.shape[:2]
    boxes = load_yolo_boxes(rgb_path)

    ax_rgb.imshow(rgb)
    ax_ir.imshow(ir, cmap="inferno")
    for ax in (ax_rgb, ax_ir):
        for cls_id, cx, cy, bw, bh in boxes:
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            rect = mpatches.Rectangle((x1, y1), bw * w, bh * h, linewidth=1.0,
                                       edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
        ax.set_xticks([])
        ax.set_yticks([])
    ax_rgb.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")


def pick_diverse_examples(n_single_drone=5, n_single_bird=4, n_multi=3, seed=7):
    vids = list_videos()
    rng = random.Random(seed)
    drone_vids = sorted([v for v in vids if v.startswith("DRONE")])
    bird_vids = sorted([v for v in vids if v.startswith("BIRD")])
    rng.shuffle(drone_vids)
    rng.shuffle(bird_vids)

    examples = []  # (rgb_path, ir_path, label)

    def try_add(vid, frames, want_multi, bucket, max_tries=25):
        frames = list(frames)
        rng.shuffle(frames)
        for frame in frames[:max_tries]:
            rgb_path, ir_path = find_files(vid, frame)
            if rgb_path is None:
                continue
            boxes = load_yolo_boxes(rgb_path)
            n = len(boxes)
            if want_multi and n >= 3:
                bucket.append((rgb_path, ir_path, f"{vid}\n(n={n})"))
                return True
            if not want_multi and n == 1:
                bucket.append((rgb_path, ir_path, vid))
                return True
        return False

    single_drone, single_bird, multi = [], [], []
    for vid in drone_vids:
        if len(single_drone) >= n_single_drone:
            break
        try_add(vid, vids[vid], False, single_drone)
    for vid in bird_vids:
        if len(single_bird) >= n_single_bird:
            break
        try_add(vid, vids[vid], False, single_bird)
    for vid in bird_vids:  # multi-instance examples observed only in BIRD-prefixed videos
        if len(multi) >= n_multi:
            break
        try_add(vid, vids[vid], True, multi)

    return single_drone + single_bird + multi


def fig_diversity_montage():
    examples = pick_diverse_examples()
    n = len(examples)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(3.1 * n_cols, 3.0 * n_rows))
    for idx, (rgb_path, ir_path, label) in enumerate(examples):
        row, col = divmod(idx, n_cols)
        draw_pair(axes[row * 2, col], axes[row * 2 + 1, col], rgb_path, ir_path, label)
        axes[row * 2, col].set_title(label if "\n" not in label else label, fontsize=8)

    # blank out unused cells
    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row * 2, col].axis("off")
        axes[row * 2 + 1, col].axis("off")

    fig.suptitle("Real AVID diversity: RGB (top of each pair) / IR (bottom), "
                 f"ground-truth boxes in green, {n} distinct real source videos", fontsize=12)
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    fig.savefig(f"{OUT_DIR}/fig_diversity_montage.png", dpi=150)
    plt.close(fig)
    print(f"wrote fig_diversity_montage.png ({n} examples)")


if __name__ == "__main__":
    fig_diversity_montage()
