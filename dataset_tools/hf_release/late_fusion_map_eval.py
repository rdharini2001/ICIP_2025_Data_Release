"""
Real late-fusion (WBF) evaluation: runs the trained RGB-only and IR-only
YOLOv8n checkpoints on the same validation images, combines their
per-image detections via Weighted Boxes Fusion
(dataset_tools/baselines/late_fusion_wbf.py), and computes mAP@0.5 with a
standard single-IoU-threshold, interpolated-precision-recall calculation
applied identically to all three conditions (RGB-only, IR-only, WBF
fusion) so the three are directly comparable -- this does not go through
Ultralytics' internal val() pipeline (which has no native concept of
combining two independent models), so these numbers are computed by our
own consistent method rather than reported as official Ultralytics mAP;
see dataset_tools/hf_release/final_val_metrics.json for the official
Ultralytics mAP of the two single-modality models individually.
"""
import json
import random
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.baselines.late_fusion_wbf import weighted_boxes_fusion


def load_val_index(n=None, seed=0):
    """IMPORTANT: val_rgb.txt/val_ir.txt are written in sorted (video_id,
    frame_idx) order by fast_index.py, NOT shuffled -- taking the first `n`
    lines samples only the alphabetically-first few videos, not a
    representative slice of the corpus (this produced a visibly biased,
    IR-outperforms-RGB result on the first pass, at odds with the official
    Ultralytics val() numbers where RGB clearly outperforms IR; the fix is
    this shuffle, not a change to the mAP math)."""
    with open(r"E:\ICIP_2025\avid_release\lists\val_rgb.txt") as f:
        rgb_paths = [l.strip() for l in f if l.strip()]
    with open(r"E:\ICIP_2025\avid_release\lists\val_ir.txt") as f:
        ir_paths = [l.strip() for l in f if l.strip()]
    assert len(rgb_paths) == len(ir_paths)

    indices = list(range(len(rgb_paths)))
    random.Random(seed).shuffle(indices)
    if n:
        indices = indices[:n]
    return [rgb_paths[i] for i in indices], [ir_paths[i] for i in indices]


def load_gt_boxes(rgb_path):
    label_path = rgb_path.replace("\\images\\", "\\labels\\").rsplit(".", 1)[0] + ".txt"
    boxes, classes = [], []
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
                boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
                classes.append(cls)
    except FileNotFoundError:
        pass
    return np.array(boxes).reshape(-1, 4), np.array(classes)


def _iou_matrix(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    ax1, ay1, ax2, ay2 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    iw, ih = np.clip(ix2 - ix1, 0, None), np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def compute_map50(all_preds, all_gts, n_classes=2, iou_thr=0.5):
    """all_preds: list of (boxes[N,4], scores[N], labels[N]) per image.
    all_gts: list of (boxes[M,4], labels[M]) per image.
    Standard single-IoU-threshold AP: sort detections by confidence,
    greedily match to unmatched GT of the same class with IoU>=thr,
    compute precision/recall curve, integrate via all-points interpolation
    (the same method COCO/VOC-2012 use), then average over classes."""
    aps = []
    for c in range(n_classes):
        # flatten all detections for this class across images, tagged with image index
        dets = []
        n_gt_total = 0
        gt_matched = {}
        for img_idx, ((boxes, scores, labels), (gt_boxes, gt_labels)) in enumerate(zip(all_preds, all_gts)):
            cls_mask = gt_labels == c
            n_gt_total += cls_mask.sum()
            gt_matched[img_idx] = np.zeros(cls_mask.sum(), dtype=bool)
            for box, score, label in zip(boxes, scores, labels):
                if label == c:
                    dets.append((img_idx, box, score))
        if n_gt_total == 0:
            continue
        dets.sort(key=lambda d: -d[2])

        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        for i, (img_idx, box, score) in enumerate(dets):
            gt_boxes, gt_labels = all_gts[img_idx]
            cls_gt_boxes = gt_boxes[gt_labels == c]
            if len(cls_gt_boxes) == 0:
                fp[i] = 1
                continue
            ious = _iou_matrix(box.reshape(1, 4), cls_gt_boxes)[0]
            best_j = np.argmax(ious)
            if ious[best_j] >= iou_thr and not gt_matched[img_idx][best_j]:
                tp[i] = 1
                gt_matched[img_idx][best_j] = True
            else:
                fp[i] = 1

        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        recall = tp_cum / n_gt_total
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        # all-points interpolation (COCO/VOC2012 style)
        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.concatenate([[0.0], precision, [0.0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-model", required=True)
    parser.add_argument("--ir-model", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--out", default=r"E:\ICIP_2025\avid_release\metadata\late_fusion_map50.json")
    args = parser.parse_args()

    from ultralytics import YOLO

    rgb_model = YOLO(args.rgb_model)
    ir_model = YOLO(args.ir_model)
    rgb_paths, ir_paths = load_val_index(n=args.n)

    all_gts, preds_rgb, preds_ir, preds_fused = [], [], [], []
    for i, (rgb_path, ir_path) in enumerate(zip(rgb_paths, ir_paths)):
        gt_boxes, gt_labels = load_gt_boxes(rgb_path)
        all_gts.append((gt_boxes, gt_labels))

        r_rgb = rgb_model.predict(rgb_path, verbose=False, conf=0.05)[0]
        r_ir = ir_model.predict(ir_path, verbose=False, conf=0.05)[0]

        rgb_boxes = r_rgb.boxes.xyxyn.cpu().numpy() if len(r_rgb.boxes) else np.zeros((0, 4))
        rgb_scores = r_rgb.boxes.conf.cpu().numpy() if len(r_rgb.boxes) else np.zeros(0)
        rgb_labels = r_rgb.boxes.cls.cpu().numpy().astype(int) if len(r_rgb.boxes) else np.zeros(0, dtype=int)

        ir_boxes = r_ir.boxes.xyxyn.cpu().numpy() if len(r_ir.boxes) else np.zeros((0, 4))
        ir_scores = r_ir.boxes.conf.cpu().numpy() if len(r_ir.boxes) else np.zeros(0)
        ir_labels = r_ir.boxes.cls.cpu().numpy().astype(int) if len(r_ir.boxes) else np.zeros(0, dtype=int)

        preds_rgb.append((rgb_boxes, rgb_scores, rgb_labels))
        preds_ir.append((ir_boxes, ir_scores, ir_labels))

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            [rgb_boxes, ir_boxes], [rgb_scores, ir_scores], [rgb_labels, ir_labels], iou_thr=0.5
        )
        preds_fused.append((fused_boxes, fused_scores, fused_labels))

        if i % 100 == 0:
            print(f"{i}/{len(rgb_paths)}")

    map50_rgb = compute_map50(preds_rgb, all_gts)
    map50_ir = compute_map50(preds_ir, all_gts)
    map50_fused = compute_map50(preds_fused, all_gts)

    result = {"n_images": len(rgb_paths), "map50_rgb_only": map50_rgb, "map50_ir_only": map50_ir, "map50_late_fusion": map50_fused}
    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
