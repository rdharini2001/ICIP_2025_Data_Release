"""
Real corpus-statistics script (Experiment E1), operating directly on the
raw Roboflow-exported layout: <root>/<Split>/<Modality>/{images,labels}/
with filenames shaped like '{IR_}?{CLASS}_{videonum}_{frameidx}_png.rf.{hash}.{jpg,txt}'.

Confirmed against real extracted data (2026-09-13):
  - class 0 = bird, class 1 = drone in the label files (opposite of our
    original placeholder assumption -- fixed in SCHEMA.md/build_hf_dataset.py).
  - RGB and IR label files for "the same" (class, videonum, frameidx) key
    carry IDENTICAL box coordinates -- the annotation was done once per
    nominal frame and shared across modalities, not re-annotated per
    modality to account for the physical RGB/IR offset. This is exactly why
    the alignment-quantification work in dataset_tools/alignment/ matters:
    a shared box does not correct for per-pair weak alignment.
  - Some (class, videonum, frameidx) keys have multiple files with
    different Roboflow hashes but identical label content -- these are
    noise-augmented duplicates of the same source frame (confirmed visually:
    clean frame + a salt/speckle-noise-augmented copy), not independent
    frames. Counted separately below as "augmented duplicate" pairs so they
    don't silently inflate the headline pair count.
"""

import os
import re
import json
from collections import defaultdict

FNAME_RE = re.compile(r"^(?P<ir_prefix>IR_)?(?P<cls>DRONE|BIRD)_(?P<video>\d+)_(?P<frame>\d+)_png\.rf\.(?P<hash>[0-9a-f]+)\.(?P<ext>jpg|txt)$")

CLASS_ID_TO_NAME = {0: "bird", 1: "drone"}


def scan_split_modality(images_dir, labels_dir):
    """Returns list of dicts: {video_id, frame_idx, hash, class_ids (from
    the label file)} for every image in images_dir with a matching label."""
    records = []
    if not os.path.isdir(images_dir):
        return records
    for fname in os.listdir(images_dir):
        m = FNAME_RE.match(fname)
        if not m:
            continue
        video_id = f"{m.group('cls')}_{m.group('video')}"
        frame_idx = int(m.group("frame"))
        label_path = os.path.join(labels_dir, fname.rsplit(".", 1)[0] + ".txt")
        class_ids = []
        box_areas = []  # normalized w*h, for the scale-stratified bucket
        if os.path.isfile(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    class_ids.append(int(parts[0]))
                    w, h = float(parts[3]), float(parts[4])
                    box_areas.append(w * h)
        records.append(
            {
                "video_id": video_id,
                "frame_idx": frame_idx,
                "hash": m.group("hash"),
                "class_ids": class_ids,
                "box_areas": box_areas,
                "filename_class": m.group("cls").lower(),
            }
        )
    return records


def coco_size_bucket(norm_area, img_w=320, img_h=256):
    px_area = norm_area * img_w * img_h
    if px_area < 32 ** 2:
        return "small"
    if px_area < 96 ** 2:
        return "medium"
    return "large"


def compute_stats(release_root, splits=("Train", "Validation", "Test")):
    stats = {}
    for split in splits:
        split_dir = os.path.join(release_root, split)
        if not os.path.isdir(split_dir):
            continue
        modality_records = {}
        for modality in ("IR", "RGB"):
            images_dir = os.path.join(split_dir, modality, "images")
            labels_dir = os.path.join(split_dir, modality, "labels")
            modality_records[modality] = scan_split_modality(images_dir, labels_dir)

        by_key = defaultdict(lambda: {"IR": [], "RGB": []})
        for modality, recs in modality_records.items():
            for r in recs:
                key = (r["video_id"], r["frame_idx"])
                by_key[key][modality].append(r)

        n_unique_keys = len(by_key)
        n_augmented_duplicate_variants = sum(
            max(0, len(v["IR"]) - 1) + max(0, len(v["RGB"]) - 1) for v in by_key.values()
        )
        n_paired_keys = sum(1 for v in by_key.values() if v["IR"] and v["RGB"])

        class_counts = defaultdict(int)
        size_bucket_counts = defaultdict(int)
        video_ids = set()
        for modality, recs in modality_records.items():
            for r in recs:
                video_ids.add(r["video_id"])
                for cid, area in zip(r["class_ids"], r["box_areas"]):
                    class_counts[CLASS_ID_TO_NAME.get(cid, f"unknown_{cid}")] += 1
                    size_bucket_counts[coco_size_bucket(area)] += 1

        stats[split] = {
            "n_images_ir": len(modality_records.get("IR", [])),
            "n_images_rgb": len(modality_records.get("RGB", [])),
            "n_unique_frame_keys": n_unique_keys,
            "n_paired_keys_with_both_modalities": n_paired_keys,
            "n_augmented_duplicate_variants": n_augmented_duplicate_variants,
            "n_unique_videos": len(video_ids),
            "class_instance_counts": dict(class_counts),
            "bbox_size_bucket_counts": dict(size_bucket_counts),
        }
    return stats


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else r"E:\ICIP_2025\raw_extracted\releasev1\releasev1-detection&tracking"
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    stats = compute_stats(root)
    print(json.dumps(stats, indent=2))
    if out_path:
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        print("wrote", out_path)
