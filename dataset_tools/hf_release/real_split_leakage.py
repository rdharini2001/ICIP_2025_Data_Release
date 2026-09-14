"""
E8 upgrade: split-leakage ablation on the REAL AVID video/frame index
(metadata/pairs_train.jsonl + pairs_val.jsonl), replacing the earlier
synthetic-corpus demonstration with the actual per-video frame-count
distribution of this dataset (real videos range from a handful of frames
to several hundred, unlike the synthetic uniform 300-frames-per-video
grid used before real data was available).
"""

import json
import sys

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.baselines.split_leakage_ablation import compare_split_strategies


def load_real_frame_index():
    frame_index = []
    for split in ("train", "val"):
        path = rf"E:\ICIP_2025\avid_release\metadata\pairs_{split}.jsonl"
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                frame_index.append((rec["video_id"], rec["frame_idx"]))
    return frame_index


if __name__ == "__main__":
    frame_index = load_real_frame_index()
    n_videos = len({v for v, _ in frame_index})
    print(f"Real corpus: {len(frame_index)} frames across {n_videos} videos "
          f"(train+val combined, as if being freshly split)")

    report = compare_split_strategies(frame_index, neighbor_window=3, seed=0)
    print("frame-level split leakage:", report["frame_level"])
    print("video-level split leakage:", report["video_level"])

    with open(r"E:\ICIP_2025\avid_release\metadata\split_leakage_real.json", "w") as f:
        json.dump({"n_frames": len(frame_index), "n_videos": n_videos, **report}, f, indent=2)
