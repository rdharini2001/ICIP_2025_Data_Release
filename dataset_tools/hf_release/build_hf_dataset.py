"""
Restructures raw extracted VIP Cup zip contents into the AVID HF schema
(see SCHEMA.md), computes alignment metadata, and writes metadata/*.jsonl.

IMPORTANT: the raw Drive zips span several release passes over the
challenge timeline (releasev1, pre-releasev2, Test_detection&tracking(_with_labels),
Validation_Videos, consolidated_IR_RGB_Pairs, detecttrack) and are not
guaranteed to share one internal layout. Rather than guess and silently
mis-map files, this script takes a *raw layout adapter* -- a generator
function you point at your extracted zip(s) that yields one record per
frame pair it finds. A `generic_frame_folder_adapter` is provided for the
common "RGB/<video_id>/<frame>.jpg + IR/<video_id>/<frame>.jpg (+
labels/<video_id>/<frame>.txt)" shape; inspect your actual extracted zip
first (`find <extracted_dir> -maxdepth 4 | head -50`) and write a small
adapter of your own if it doesn't match -- the adapter contract is the only
thing that needs to change, `build_dataset()` itself doesn't care where the
records came from.

Payload frames/labels must never be yielded by an adapter for this release
-- this script does not filter classes for you.

NOT run against the real corpus in this sandbox (no access to the raw zips
here — see PLAN.md). The orchestration logic (copy/rename/metadata-write) is
tested against a synthetic mock raw layout in test_build_hf_dataset.py, so
that logic is verified even though it's never touched a real zip.
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Iterator, Optional

import cv2

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dataset_tools.alignment.alignment_metrics import pair_alignment_quality  # noqa: E402


@dataclass
class RawRecord:
    video_id: str
    frame_idx: int
    split: str  # "train" / "val" / "test"
    rgb_path: str
    ir_path: str
    label_path: Optional[str] = None  # YOLO .txt, may be None if unlabeled
    scenario_tags: list = field(default_factory=list)
    builtin_distortion: Optional[str] = None
    builtin_distortion_severity: Optional[int] = None


def generic_frame_folder_adapter(raw_root, split) -> Iterator[RawRecord]:
    """Assumes raw_root/RGB/<video_id>/<frame_idx>.jpg,
    raw_root/IR/<video_id>/<frame_idx>.jpg, and optionally
    raw_root/labels/<video_id>/<frame_idx>.txt. Adjust or replace this
    function for whichever real layout your extracted zip actually has."""
    rgb_root = os.path.join(raw_root, "RGB")
    ir_root = os.path.join(raw_root, "IR")
    label_root = os.path.join(raw_root, "labels")

    if not os.path.isdir(rgb_root):
        raise FileNotFoundError(f"expected {rgb_root} to exist -- write a custom adapter for this layout")

    for video_id in sorted(os.listdir(rgb_root)):
        rgb_video_dir = os.path.join(rgb_root, video_id)
        ir_video_dir = os.path.join(ir_root, video_id)
        if not os.path.isdir(rgb_video_dir):
            continue
        for fname in sorted(os.listdir(rgb_video_dir)):
            frame_idx = int(os.path.splitext(fname)[0])
            rgb_path = os.path.join(rgb_video_dir, fname)
            ir_path = os.path.join(ir_video_dir, fname)
            if not os.path.isfile(ir_path):
                continue
            label_path = os.path.join(label_root, video_id, f"{os.path.splitext(fname)[0]}.txt")
            yield RawRecord(
                video_id=video_id,
                frame_idx=frame_idx,
                split=split,
                rgb_path=rgb_path,
                ir_path=ir_path,
                label_path=label_path if os.path.isfile(label_path) else None,
            )


def build_dataset(records: Iterator[RawRecord], out_root, compute_alignment=True):
    """Consumes an adapter's records, writes the AVID schema to out_root,
    and returns summary counts. Idempotent per-record (re-running overwrites
    the same destination paths)."""
    os.makedirs(out_root, exist_ok=True)
    metadata_dir = os.path.join(out_root, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    video_seen = {}
    n_pairs = 0

    with open(os.path.join(metadata_dir, "pairs.jsonl"), "w") as pairs_f:
        for rec in records:
            rgb_dst_dir = os.path.join(out_root, "images", "rgb", rec.split, rec.video_id)
            ir_dst_dir = os.path.join(out_root, "images", "ir", rec.split, rec.video_id)
            label_dst_dir = os.path.join(out_root, "labels", rec.split, rec.video_id)
            os.makedirs(rgb_dst_dir, exist_ok=True)
            os.makedirs(ir_dst_dir, exist_ok=True)
            os.makedirs(label_dst_dir, exist_ok=True)

            rgb_dst = os.path.join(rgb_dst_dir, f"{rec.frame_idx:06d}.jpg")
            ir_dst = os.path.join(ir_dst_dir, f"{rec.frame_idx:06d}.jpg")
            shutil.copyfile(rec.rgb_path, rgb_dst)
            shutil.copyfile(rec.ir_path, ir_dst)
            if rec.label_path:
                shutil.copyfile(rec.label_path, os.path.join(label_dst_dir, f"{rec.frame_idx:06d}.txt"))

            alignment = {"alignment_shift_px": None, "alignment_confidence": None}
            if compute_alignment:
                rgb_img = cv2.cvtColor(cv2.imread(rgb_dst), cv2.COLOR_BGR2RGB)
                ir_img = cv2.imread(ir_dst, cv2.IMREAD_GRAYSCALE)
                try:
                    result = pair_alignment_quality(rgb_img, ir_img)
                    alignment = {
                        "alignment_shift_px": result["alignment_shift_px"],
                        "alignment_confidence": result["mutual_information"]["confidence"],
                    }
                except Exception as exc:  # keep going; flag the pair rather than aborting the whole build
                    alignment = {"alignment_shift_px": None, "alignment_confidence": None, "alignment_error": str(exc)}

            classes_present = []
            if rec.label_path:
                with open(rec.label_path) as lf:
                    class_ids = {int(line.split()[0]) for line in lf if line.strip()}
                # Confirmed against the real Roboflow export labels (not the
                # class order we originally assumed): class 0 = bird, class 1 = drone.
                classes_present = [{0: "bird", 1: "drone"}.get(c, str(c)) for c in class_ids]

            record_out = {
                "video_id": rec.video_id,
                "frame_idx": rec.frame_idx,
                "split": rec.split,
                "classes_present": classes_present,
                "scenario_tags": rec.scenario_tags,
                "builtin_distortion": rec.builtin_distortion,
                "builtin_distortion_severity": rec.builtin_distortion_severity,
                **alignment,
            }
            pairs_f.write(json.dumps(record_out) + "\n")
            n_pairs += 1
            video_seen.setdefault(rec.video_id, {"split": rec.split, "n_frames": 0, "classes_present": set()})
            video_seen[rec.video_id]["n_frames"] += 1
            video_seen[rec.video_id]["classes_present"].update(classes_present)

    with open(os.path.join(metadata_dir, "videos.jsonl"), "w") as videos_f:
        for video_id, info in video_seen.items():
            videos_f.write(
                json.dumps(
                    {
                        "video_id": video_id,
                        "split": info["split"],
                        "n_frames": info["n_frames"],
                        "classes_present": sorted(info["classes_present"]),
                    }
                )
                + "\n"
            )

    return {"n_pairs": n_pairs, "n_videos": len(video_seen)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, help="path to one extracted zip's contents")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--no-alignment", action="store_true", help="skip the (slower) per-pair alignment pass")
    args = parser.parse_args()

    records = generic_frame_folder_adapter(args.raw_root, args.split)
    summary = build_dataset(records, args.out_root, compute_alignment=not args.no_alignment)
    print(summary)
