"""
Object-relative misalignment ratio: rho = d_align / s_target, where
d_align is the per-pair estimated RGB-IR global translation
(real_alignment_sample.json, from alignment_metrics.py's NMI search) and
s_target is the median geometric-mean box side length of the annotated
targets in that same frame. Raw pixel shift is hard to interpret without
knowing the target's own scale (Sec. "An extreme tiny-object
distribution"); this ratio makes that comparison explicit: rho > 1 means
the estimated inter-sensor shift is larger than the object it is
supposed to be describing.

Joins the existing real 300-pair alignment sample with the real YOLO
label files for the same (video_id, frame_idx) keys -- no new geometric
estimation, just a real-data join and a ratio.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")

RAW_ROOT = r"E:\ICIP_2025\raw_extracted\releasev1\releasev1-detection&tracking\Train"
FRAME_W, FRAME_H = 320, 256


def find_rgb_label(video_id, frame_idx):
    frame_str = f"{int(frame_idx):03d}"
    pattern = os.path.join(RAW_ROOT, "RGB", "labels", f"{video_id}_{frame_str}_png.rf.*.txt")
    matches = glob.glob(pattern)
    if not matches:
        return None
    # if the Roboflow-duplicate pair exists, both label files are identical
    # (labels are shared across the noise-augmented duplicate), so any match works
    return matches[0]


def median_target_side_px(label_path):
    sides = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            _, _, _, w, h = parts[:5]
            w_px, h_px = float(w) * FRAME_W, float(h) * FRAME_H
            sides.append((w_px * h_px) ** 0.5)
    if not sides:
        return None
    return float(np.median(sides))


def compute_ratios():
    with open(r"E:\ICIP_2025\avid_release\metadata\real_alignment_sample.json") as f:
        data = json.load(f)

    rows = []
    missing = 0
    for s in data["samples"]:
        label_path = find_rgb_label(s["video_id"], s["frame_idx"])
        if label_path is None:
            missing += 1
            continue
        s_target = median_target_side_px(label_path)
        if s_target is None or s_target == 0:
            missing += 1
            continue
        rho = s["alignment_shift_px"] / s_target
        rows.append({
            "video_id": s["video_id"], "frame_idx": s["frame_idx"],
            "alignment_shift_px": s["alignment_shift_px"],
            "median_target_side_px": s_target, "rho": rho,
        })

    rhos = np.array([r["rho"] for r in rows])
    summary = {
        "n_matched": len(rows), "n_missing_label": missing,
        "mean_rho": float(np.mean(rhos)), "median_rho": float(np.median(rhos)),
        "frac_rho_gt_0.25": float(np.mean(rhos > 0.25)),
        "frac_rho_gt_0.5": float(np.mean(rhos > 0.5)),
        "frac_rho_gt_1.0": float(np.mean(rhos > 1.0)),
        "median_target_side_px_overall": float(np.median([r["median_target_side_px"] for r in rows])),
    }
    return rows, summary


if __name__ == "__main__":
    rows, summary = compute_ratios()
    print(json.dumps(summary, indent=2))
    out = {"rows": rows, "summary": summary}
    with open(r"E:\ICIP_2025\avid_release\metadata\object_relative_misalignment.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote object_relative_misalignment.json")
