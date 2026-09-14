"""
Runs real alignment quantification on a random sample of real AVID pairs
(both splits combined), producing genuine summary statistics for the paper
-- replacing the earlier inaccurate claim that this had already run.
"""
import json
import random
import sys

import cv2

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.alignment.alignment_metrics import pair_alignment_quality


def load_all_pairs():
    pairs = []
    for split in ("train", "val"):
        with open(rf"E:\ICIP_2025\avid_release\metadata\pairs_{split}.jsonl") as f:
            for line in f:
                pairs.append(json.loads(line))
    return pairs


if __name__ == "__main__":
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    all_pairs = load_all_pairs()
    random.Random(0).shuffle(all_pairs)
    sample = all_pairs[:n_sample]

    results = []
    for i, rec in enumerate(sample):
        rgb = cv2.cvtColor(cv2.imread(rec["rgb_path"]), cv2.COLOR_BGR2RGB)
        ir = cv2.imread(rec["ir_path"], cv2.IMREAD_GRAYSCALE)
        try:
            r = pair_alignment_quality(rgb, ir, max_shift=12)
            results.append({
                "video_id": rec["video_id"], "frame_idx": rec["frame_idx"],
                "alignment_shift_px": r["alignment_shift_px"],
                "confidence": r["mutual_information"]["confidence"],
                "disagreement_px": r["estimator_disagreement_px"],
            })
        except Exception as exc:
            pass
        if i % 50 == 0:
            print(f"{i}/{n_sample}")

    shifts = [r["alignment_shift_px"] for r in results]
    import statistics
    summary = {
        "n_sampled": len(results),
        "mean_shift_px": statistics.mean(shifts),
        "median_shift_px": statistics.median(shifts),
        "stdev_shift_px": statistics.stdev(shifts) if len(shifts) > 1 else 0,
        "max_shift_px": max(shifts),
        "min_shift_px": min(shifts),
        "frac_zero_shift": sum(1 for s in shifts if s == 0) / len(shifts),
        "frac_over_2px": sum(1 for s in shifts if s > 2) / len(shifts),
        "frac_over_5px": sum(1 for s in shifts if s > 5) / len(shifts),
    }
    print(json.dumps(summary, indent=2))
    with open(r"E:\ICIP_2025\avid_release\metadata\real_alignment_sample.json", "w") as f:
        json.dump({"summary": summary, "samples": results}, f, indent=2)
