"""
Wires a trained Ultralytics YOLO checkpoint into the ARS harness
(dataset_tools/alignment/alignment_robustness_score.py). This is the
adapter layer between "a real model" and the model-agnostic
predict_fn/score_fn interface the metric expects -- it does not change the
metric itself, only supplies a real predict_fn/score_fn pair for it.

score_fn here is IoU-based: 1.0 for a correct-class prediction with IoU >=
0.5 against the (single, per our small-object-dominated corpus) ground
truth box, scaled down smoothly for partial overlap, 0 for a wrong class or
no detection -- deliberately simple (not full mAP) so the sensitivity sweep
stays fast per pair; report full mAP separately via yolo_corruption_eval.py
style val() runs for the headline accuracy numbers.
"""

import random
import sys
import os

import cv2
import numpy as np

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.alignment.alignment_robustness_score import compute_ars


def _iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def make_rgb_predict_fn(model):
    """predict_fn(rgb, ir) -> best single detection as (x1,y1,x2,y2,cls),
    normalized coords. Ignores ir (RGB-only model); ir is still accepted so
    this has the same call signature the harness expects for any model."""

    def predict_fn(rgb, ir):
        result = model.predict(rgb, verbose=False, conf=0.15)[0]
        if len(result.boxes) == 0:
            return None
        best = result.boxes[result.boxes.conf.argmax()]
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = best.xyxy[0].tolist()
        return (x1 / w, y1 / h, x2 / w, y2 / h, int(best.cls[0]))

    return predict_fn


def iou_score_fn(pred, gt):
    """gt: (x1,y1,x2,y2,cls) normalized, ground-truth box for the pair."""
    if pred is None:
        return 0.0
    if pred[4] != gt[4]:
        return 0.0
    return _iou(pred[:4], gt[:4])


def load_val_pairs_with_gt(n=40, seed=0):
    """Samples n real validation pairs, returning (rgb, ir, gt) triples
    where gt is the first labeled box per frame (fine for this corpus,
    dominated by single-instance frames).

    IMPORTANT: val_rgb.txt and val_ir.txt (written by fast_index.py) are
    paired by LINE INDEX, not by any shared filename pattern -- RGB and IR
    files for "the same" pair have completely different filenames
    (different Roboflow hashes, and IR has an extra 'IR_' prefix), so the
    two lists must be read together, never derived from each other via
    string substitution (an earlier version of this function tried
    rgb_path.replace('RGB','IR') and silently matched zero real pairs)."""
    with open(r"E:\ICIP_2025\avid_release\lists\val_rgb.txt") as f:
        rgb_paths = [l.strip() for l in f if l.strip()]
    with open(r"E:\ICIP_2025\avid_release\lists\val_ir.txt") as f:
        ir_paths = [l.strip() for l in f if l.strip()]
    assert len(rgb_paths) == len(ir_paths), (len(rgb_paths), len(ir_paths))

    indices = list(range(len(rgb_paths)))
    random.Random(seed).shuffle(indices)

    pairs = []
    for i in indices:
        rgb_path, ir_path = rgb_paths[i], ir_paths[i]
        label_path = rgb_path.replace("\\images\\", "\\labels\\").rsplit(".", 1)[0] + ".txt"
        if not os.path.isfile(label_path):
            continue
        with open(label_path) as lf:
            line = lf.readline().split()
        if len(line) < 5:
            continue
        cls, cx, cy, bw, bh = int(line[0]), *map(float, line[1:5])
        gt = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, cls)

        rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if rgb_img is None or ir_img is None:
            continue
        pairs.append((rgb_img, ir_img, gt))
        if len(pairs) >= n:
            break
    return pairs


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-pairs", type=int, default=40)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    pairs = load_val_pairs_with_gt(n=args.n_pairs)
    print(f"Loaded {len(pairs)} real validation pairs with ground truth")

    result = compute_ars(pairs, make_rgb_predict_fn(model), iou_score_fn, seed=0)
    print(json.dumps({k: v for k, v in result.items() if k != "curve"}, indent=2))
    for row in result["curve"]:
        print(f"  shift={row['shift_px']:>5}px  IoU-score={row['mean_score']:.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print("wrote", args.out)
