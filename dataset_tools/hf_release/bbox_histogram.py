"""
Fine-grained bbox-size histogram (Experiment E1 addendum): standard COCO
small/medium/large buckets put ~99% of AVID's boxes in a single "small"
bucket (see corpus_summary.json), which says almost nothing about the
actual difficulty distribution. This script computes a finer histogram
(pixel-side-length bins) to characterize just how extreme the small-object
problem is, for a dedicated figure/table in the paper.
"""

import json
import os
import re

FNAME_RE = re.compile(r"^(?P<ir_prefix>IR_)?(?P<cls>DRONE|BIRD)_(?P<video>\d+)_(?P<frame>\d+)_png\.rf\.(?P<hash>[0-9a-f]+)\.(?P<ext>jpg|txt)$")

IMG_W, IMG_H = 320, 256
BIN_EDGES_PX = [0, 4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 96, 128, 1e9]
BIN_LABELS = [f"{int(BIN_EDGES_PX[i])}-{int(BIN_EDGES_PX[i+1]) if BIN_EDGES_PX[i+1] < 1e9 else 'inf'}px" for i in range(len(BIN_EDGES_PX) - 1)]


def bin_index(side_px):
    for i in range(len(BIN_EDGES_PX) - 1):
        if BIN_EDGES_PX[i] <= side_px < BIN_EDGES_PX[i + 1]:
            return i
    return len(BIN_EDGES_PX) - 2


def compute_histogram(pairs_jsonl_path):
    hist = {label: 0 for label in BIN_LABELS}
    n_boxes = 0
    with open(pairs_jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            label_path = re.sub(r"([\\/])images([\\/])", r"\1labels\2", rec["rgb_path"]).rsplit(".", 1)[0] + ".txt"
            if not os.path.isfile(label_path):
                continue
            with open(label_path) as lf:
                for l in lf:
                    parts = l.split()
                    if len(parts) < 5:
                        continue
                    w, h = float(parts[3]), float(parts[4])
                    side_px = (w * IMG_W * h * IMG_H) ** 0.5  # geometric-mean side length in px
                    hist[BIN_LABELS[bin_index(side_px)]] += 1
                    n_boxes += 1
    return hist, n_boxes


if __name__ == "__main__":
    out = {}
    for split in ("train", "val"):
        path = rf"E:\ICIP_2025\avid_release\metadata\pairs_{split}.jsonl"
        if os.path.isfile(path):
            hist, n = compute_histogram(path)
            out[split] = {"n_boxes": n, "histogram_px_side": hist}
            print(split, n, hist)
    with open(r"E:\ICIP_2025\avid_release\metadata\bbox_fine_histogram.json", "w") as f:
        json.dump(out, f, indent=2)
