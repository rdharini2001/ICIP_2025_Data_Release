"""
Real ARS evaluation for late-fusion (WBF), completing the RGB-only vs.
late-fusion comparison in the paper's Alignment Robustness Score results
table. predict_fn here runs both checkpoints and combines via WBF, exactly
as late_fusion_map_eval.py does for the headline mAP number -- the only
difference is the IR input is the (possibly misalignment-injected) frame
the ARS sweep passes in, not the clean IR file.
"""
import random
import sys
import os

import cv2
import numpy as np

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.alignment.alignment_robustness_score import compute_ars
from dataset_tools.baselines.late_fusion_wbf import weighted_boxes_fusion
from dataset_tools.alignment.real_ars_yolo import load_val_pairs_with_gt, iou_score_fn


def make_late_fusion_predict_fn(rgb_model, ir_model):
    def predict_fn(rgb, ir):
        r_rgb = rgb_model.predict(rgb, verbose=False, conf=0.05)[0]
        r_ir = ir_model.predict(ir, verbose=False, conf=0.05)[0]

        rgb_boxes = r_rgb.boxes.xyxyn.cpu().numpy() if len(r_rgb.boxes) else np.zeros((0, 4))
        rgb_scores = r_rgb.boxes.conf.cpu().numpy() if len(r_rgb.boxes) else np.zeros(0)
        rgb_labels = r_rgb.boxes.cls.cpu().numpy().astype(int) if len(r_rgb.boxes) else np.zeros(0, dtype=int)

        ir_boxes = r_ir.boxes.xyxyn.cpu().numpy() if len(r_ir.boxes) else np.zeros((0, 4))
        ir_scores = r_ir.boxes.conf.cpu().numpy() if len(r_ir.boxes) else np.zeros(0)
        ir_labels = r_ir.boxes.cls.cpu().numpy().astype(int) if len(r_ir.boxes) else np.zeros(0, dtype=int)

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            [rgb_boxes, ir_boxes], [rgb_scores, ir_scores], [rgb_labels, ir_labels], iou_thr=0.5
        )
        if len(fused_boxes) == 0:
            return None
        best = np.argmax(fused_scores)
        x1, y1, x2, y2 = fused_boxes[best]
        return (x1, y1, x2, y2, int(fused_labels[best]))

    return predict_fn


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-model", required=True)
    parser.add_argument("--ir-model", required=True)
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--out", default=r"E:\ICIP_2025\avid_release\metadata\ars_late_fusion.json")
    args = parser.parse_args()

    from ultralytics import YOLO

    rgb_model = YOLO(args.rgb_model)
    ir_model = YOLO(args.ir_model)
    pairs = load_val_pairs_with_gt(n=args.n_pairs)
    print(f"Loaded {len(pairs)} real validation pairs with ground truth")

    result = compute_ars(pairs, make_late_fusion_predict_fn(rgb_model, ir_model), iou_score_fn, seed=0)
    print(json.dumps({k: v for k, v in result.items() if k != "curve"}, indent=2))
    for row in result["curve"]:
        print(f"  shift={row['shift_px']:>5}px  IoU-score={row['mean_score']:.3f}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", args.out)
