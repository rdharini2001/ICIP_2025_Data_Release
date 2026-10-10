"""
Builds the real AVID v1 dataset (Train + Validation) from the extracted
releasev1 Roboflow export, deduplicating the systematic noise-augmented
copy Roboflow baked into every frame (confirmed: 100% of the 28,790 Train
and 3,250 Validation frame keys have exactly 2 file variants per modality;
smaller JPEG file size reliably identifies the clean original -- verified
against 8 random samples before trusting it at scale).

Output layout matches dataset_tools/hf_release/SCHEMA.md.
"""

import os
import re
import json
import shutil
import sys
import time

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.alignment.alignment_metrics import pair_alignment_quality
import cv2

FNAME_RE = re.compile(r"^(?P<ir_prefix>IR_)?(?P<cls>DRONE|BIRD)_(?P<video>\d+)_(?P<frame>\d+)_png\.rf\.(?P<hash>[0-9a-f]+)\.(?P<ext>jpg|txt)$")
CLASS_ID_TO_NAME = {0: "bird", 1: "drone"}

RAW_ROOT = r"E:\ICIP_2025\raw_extracted\releasev1\releasev1-detection&tracking"
OUT_ROOT = r"E:\ICIP_2025\avid_release"
SPLIT_MAP = {"Train": "train", "Validation": "val"}


def index_modality(images_dir):
    """Returns dict: (video_id, frame_idx) -> list of (filename, filesize), sorted by size ascending.
    Uses os.scandir() rather than os.listdir()+os.path.getsize() -- the latter issues a
    separate stat() syscall per file and is drastically slower under Windows Defender
    real-time scanning at this corpus's scale (tens of thousands of files per directory);
    scandir's DirEntry caches stat info from the directory enumeration itself."""
    by_key = {}
    with os.scandir(images_dir) as it:
        for entry in it:
            fname = entry.name
            m = FNAME_RE.match(fname)
            if not m:
                continue
            key = (f"{m.group('cls')}_{m.group('video')}", int(m.group("frame")))
            size = entry.stat().st_size
            by_key.setdefault(key, []).append((fname, size))
    for key in by_key:
        by_key[key].sort(key=lambda x: x[1])
    return by_key


def build_split(raw_split_dir, split_name, out_root, alignment_sample_rate=1.0, seed=0):
    import random

    rng = random.Random(seed)
    ir_images_dir = os.path.join(raw_split_dir, "IR", "images")
    ir_labels_dir = os.path.join(raw_split_dir, "IR", "labels")
    rgb_images_dir = os.path.join(raw_split_dir, "RGB", "images")
    rgb_labels_dir = os.path.join(raw_split_dir, "RGB", "labels")

    ir_index = index_modality(ir_images_dir)
    rgb_index = index_modality(rgb_images_dir)
    keys = sorted(set(ir_index) & set(rgb_index))
    print(f"[{split_name}] {len(keys)} paired keys (IR-only keys: {len(set(ir_index)-set(rgb_index))}, RGB-only: {len(set(rgb_index)-set(ir_index))})", flush=True)

    pairs_path = os.path.join(out_root, "metadata", f"pairs_{split_name}.jsonl")
    os.makedirs(os.path.dirname(pairs_path), exist_ok=True)

    video_seen = {}
    t0 = time.time()
    n_aligned = 0
    with open(pairs_path, "w") as pf:
        for i, key in enumerate(keys):
            video_id, frame_idx = key
            ir_fname = ir_index[key][0][0]  # smallest = clean
            rgb_fname = rgb_index[key][0][0]

            rgb_dst_dir = os.path.join(out_root, "images", "rgb", split_name, video_id)
            ir_dst_dir = os.path.join(out_root, "images", "ir", split_name, video_id)
            label_dst_dir = os.path.join(out_root, "labels", split_name, video_id)
            os.makedirs(rgb_dst_dir, exist_ok=True)
            os.makedirs(ir_dst_dir, exist_ok=True)
            os.makedirs(label_dst_dir, exist_ok=True)

            rgb_dst = os.path.join(rgb_dst_dir, f"{frame_idx:06d}.jpg")
            ir_dst = os.path.join(ir_dst_dir, f"{frame_idx:06d}.jpg")
            shutil.copyfile(os.path.join(rgb_images_dir, rgb_fname), rgb_dst)
            shutil.copyfile(os.path.join(ir_images_dir, ir_fname), ir_dst)

            rgb_label_src = os.path.join(rgb_labels_dir, rgb_fname.rsplit(".", 1)[0] + ".txt")
            label_dst = os.path.join(label_dst_dir, f"{frame_idx:06d}.txt")
            class_ids = []
            if os.path.isfile(rgb_label_src):
                shutil.copyfile(rgb_label_src, label_dst)
                with open(rgb_label_src) as lf:
                    class_ids = [int(line.split()[0]) for line in lf if line.strip()]
            else:
                open(label_dst, "w").close()

            alignment = {"alignment_shift_px": None, "alignment_confidence": None}
            if rng.random() < alignment_sample_rate:
                try:
                    rgb_img = cv2.cvtColor(cv2.imread(rgb_dst), cv2.COLOR_BGR2RGB)
                    ir_img = cv2.imread(ir_dst, cv2.IMREAD_GRAYSCALE)
                    result = pair_alignment_quality(rgb_img, ir_img, max_shift=12)
                    alignment = {
                        "alignment_shift_px": result["alignment_shift_px"],
                        "alignment_confidence": result["mutual_information"]["confidence"],
                    }
                    n_aligned += 1
                except Exception as exc:
                    alignment["alignment_error"] = str(exc)

            classes_present = sorted({CLASS_ID_TO_NAME.get(c, str(c)) for c in class_ids})
            rec = {
                "video_id": video_id,
                "frame_idx": frame_idx,
                "split": split_name,
                "classes_present": classes_present,
                **alignment,
            }
            pf.write(json.dumps(rec) + "\n")
            video_seen.setdefault(video_id, {"n_frames": 0, "classes_present": set()})
            video_seen[video_id]["n_frames"] += 1
            video_seen[video_id]["classes_present"].update(classes_present)

            if i % 500 == 0:
                elapsed = time.time() - t0
                print(f"[{split_name}] {i}/{len(keys)} ({elapsed:.0f}s elapsed, {n_aligned} aligned)", flush=True)

    videos_path = os.path.join(out_root, "metadata", f"videos_{split_name}.jsonl")
    with open(videos_path, "w") as vf:
        for video_id, info in video_seen.items():
            vf.write(json.dumps({"video_id": video_id, "split": split_name, "n_frames": info["n_frames"], "classes_present": sorted(info["classes_present"])}) + "\n")

    print(f"[{split_name}] DONE: {len(keys)} pairs, {len(video_seen)} videos, {n_aligned} alignment-scored", flush=True)
    return {"n_pairs": len(keys), "n_videos": len(video_seen), "n_aligned": n_aligned}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment-sample-rate", type=float, default=1.0)
    args = parser.parse_args()

    summary = {}
    for raw_name, split_name in SPLIT_MAP.items():
        raw_dir = os.path.join(RAW_ROOT, raw_name)
        summary[split_name] = build_split(raw_dir, split_name, OUT_ROOT, alignment_sample_rate=args.alignment_sample_rate)

    with open(os.path.join(OUT_ROOT, "metadata", "build_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
