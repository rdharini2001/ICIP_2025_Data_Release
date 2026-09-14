"""
Fast, scandir-based indexing + dedup for the real releasev1 corpus, and
generation of Ultralytics-compatible image-list .txt files -- no physical
file copying needed for single-modality training, since Ultralytics YOLO
resolves each image's label by replacing '/images/' with '/labels/' and
swapping the extension to .txt, which the raw Roboflow export layout
already satisfies directly (IR/images/X.jpg <-> IR/labels/X.txt).

Dedup: every (class, video, frame) key has exactly one clean + one
noise-augmented variant (confirmed on the full corpus, not just a sample --
see build_avid_v1.py's docstring for the verification). We keep the
smaller-filesize variant (the clean original) per key, using stat info
already returned by os.scandir() so no extra per-file syscalls are needed.
"""

import os
import re
import json
import time

FNAME_RE = re.compile(r"^(?P<ir_prefix>IR_)?(?P<cls>DRONE|BIRD)_(?P<video>\d+)_(?P<frame>\d+)_png\.rf\.(?P<hash>[0-9a-f]+)\.(?P<ext>jpg|txt)$")
CLASS_ID_TO_NAME = {0: "bird", 1: "drone"}

RAW_ROOT = r"E:\ICIP_2025\raw_extracted\releasev1\releasev1-detection&tracking"
OUT_ROOT = r"E:\ICIP_2025\avid_release"
SPLIT_MAP = {"Train": "train", "Validation": "val"}


def index_modality_fast(images_dir):
    """scandir-based: returns dict (video_id, frame_idx) -> [(full_path, size), ...] sorted by size asc."""
    by_key = {}
    with os.scandir(images_dir) as it:
        for entry in it:
            m = FNAME_RE.match(entry.name)
            if not m:
                continue
            key = (f"{m.group('cls')}_{m.group('video')}", int(m.group("frame")))
            by_key.setdefault(key, []).append((entry.path, entry.stat().st_size))
    for key in by_key:
        by_key[key].sort(key=lambda x: x[1])
    return by_key


def label_path_for(image_path):
    return re.sub(r"([\\/])images([\\/])", r"\1labels\2", image_path).rsplit(".", 1)[0] + ".txt"


def build_index(raw_split_dir, split_name):
    t0 = time.time()
    ir_index = index_modality_fast(os.path.join(raw_split_dir, "IR", "images"))
    rgb_index = index_modality_fast(os.path.join(raw_split_dir, "RGB", "images"))
    keys = sorted(set(ir_index) & set(rgb_index))
    print(f"[{split_name}] indexed {len(ir_index)} IR keys, {len(rgb_index)} RGB keys, {len(keys)} paired in {time.time()-t0:.2f}s")
    return ir_index, rgb_index, keys


def write_image_lists_and_metadata(raw_split_dir, split_name, out_root):
    ir_index, rgb_index, keys = build_index(raw_split_dir, split_name)

    os.makedirs(os.path.join(out_root, "lists"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "metadata"), exist_ok=True)

    ir_list_path = os.path.join(out_root, "lists", f"{split_name}_ir.txt")
    rgb_list_path = os.path.join(out_root, "lists", f"{split_name}_rgb.txt")
    pairs_path = os.path.join(out_root, "metadata", f"pairs_{split_name}.jsonl")

    video_seen = {}
    class_instance_counts = {"bird": 0, "drone": 0}
    bbox_areas = []

    with open(ir_list_path, "w") as irf, open(rgb_list_path, "w") as rgbf, open(pairs_path, "w") as pf:
        for video_id, frame_idx in keys:
            key = (video_id, frame_idx)
            ir_path, _ = ir_index[key][0]  # smallest = clean original
            rgb_path, _ = rgb_index[key][0]
            irf.write(ir_path + "\n")
            rgbf.write(rgb_path + "\n")

            label_path = label_path_for(rgb_path)
            class_ids = []
            areas = []
            if os.path.isfile(label_path):
                with open(label_path) as lf:
                    for line in lf:
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        cid = int(parts[0])
                        class_ids.append(cid)
                        w, h = float(parts[3]), float(parts[4])
                        areas.append(w * h)
                        class_instance_counts[CLASS_ID_TO_NAME.get(cid, str(cid))] = class_instance_counts.get(CLASS_ID_TO_NAME.get(cid, str(cid)), 0) + 1
                        bbox_areas.append(w * h)

            classes_present = sorted({CLASS_ID_TO_NAME.get(c, str(c)) for c in class_ids})
            pf.write(json.dumps({
                "video_id": video_id, "frame_idx": frame_idx, "split": split_name,
                "classes_present": classes_present, "ir_path": ir_path, "rgb_path": rgb_path,
            }) + "\n")

            video_seen.setdefault(video_id, {"n_frames": 0, "classes_present": set()})
            video_seen[video_id]["n_frames"] += 1
            video_seen[video_id]["classes_present"].update(classes_present)

    videos_path = os.path.join(out_root, "metadata", f"videos_{split_name}.jsonl")
    with open(videos_path, "w") as vf:
        for video_id, info in video_seen.items():
            vf.write(json.dumps({"video_id": video_id, "split": split_name, "n_frames": info["n_frames"], "classes_present": sorted(info["classes_present"])}) + "\n")

    def bucket(a, img_w=320, img_h=256):
        px = a * img_w * img_h
        return "small" if px < 32**2 else ("medium" if px < 96**2 else "large")

    bucket_counts = {"small": 0, "medium": 0, "large": 0}
    for a in bbox_areas:
        bucket_counts[bucket(a)] += 1

    summary = {
        "n_pairs": len(keys), "n_videos": len(video_seen),
        "class_instance_counts": class_instance_counts,
        "bbox_size_bucket_counts": bucket_counts,
    }
    print(f"[{split_name}] {json.dumps(summary)}")
    return summary


if __name__ == "__main__":
    overall = {}
    for raw_name, split_name in SPLIT_MAP.items():
        overall[split_name] = write_image_lists_and_metadata(os.path.join(RAW_ROOT, raw_name), split_name, OUT_ROOT)
    with open(os.path.join(OUT_ROOT, "metadata", "corpus_summary.json"), "w") as f:
        json.dump(overall, f, indent=2)
    print(json.dumps(overall, indent=2))
