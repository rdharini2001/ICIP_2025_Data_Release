"""
End-to-end test of build_dataset() against a synthetic mock "raw zip"
layout (real files on disk, generic_frame_folder_adapter's expected shape),
so the copy/rename/metadata-write/alignment-computation pipeline is verified
without needing any real VIP Cup data.
"""

import json
import os
import shutil
import tempfile

import cv2
import numpy as np

from .build_hf_dataset import build_dataset, generic_frame_folder_adapter


def _write_mock_raw_zip(raw_root, video_id="video_000", n_frames=3):
    rgb_dir = os.path.join(raw_root, "RGB", video_id)
    ir_dir = os.path.join(raw_root, "IR", video_id)
    label_dir = os.path.join(raw_root, "labels", video_id)
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(ir_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for i in range(n_frames):
        rgb = np.full((256, 320, 3), 100, dtype=np.uint8)
        rgb[100:120, 140:160] = [30, 30, 30]
        ir = np.full((256, 320), 60, dtype=np.uint8)
        ir[100:120, 145:165] = 220  # shifted a few px, simulating weak alignment
        cv2.imwrite(os.path.join(rgb_dir, f"{i:06d}.jpg"), rgb)
        cv2.imwrite(os.path.join(ir_dir, f"{i:06d}.jpg"), ir)
        with open(os.path.join(label_dir, f"{i:06d}.txt"), "w") as f:
            # class 0 = bird, class 1 = drone (confirmed against real Roboflow
            # label files -- see SCHEMA.md; this test used to say "class 0 =
            # drone" before that correction, which this file's own assertion
            # below would now catch if it regressed).
            f.write("1 0.47 0.43 0.06 0.08\n")


def test_build_dataset_end_to_end():
    tmp = tempfile.mkdtemp(prefix="avid_build_test_")
    try:
        raw_root = os.path.join(tmp, "raw")
        out_root = os.path.join(tmp, "out")
        _write_mock_raw_zip(raw_root, video_id="video_000", n_frames=3)

        records = generic_frame_folder_adapter(raw_root, split="train")
        summary = build_dataset(records, out_root, compute_alignment=True)

        assert summary["n_pairs"] == 3
        assert summary["n_videos"] == 1

        for i in range(3):
            assert os.path.isfile(os.path.join(out_root, "images", "rgb", "train", "video_000", f"{i:06d}.jpg"))
            assert os.path.isfile(os.path.join(out_root, "images", "ir", "train", "video_000", f"{i:06d}.jpg"))
            assert os.path.isfile(os.path.join(out_root, "labels", "train", "video_000", f"{i:06d}.txt"))

        with open(os.path.join(out_root, "metadata", "pairs.jsonl")) as f:
            pair_records = [json.loads(line) for line in f]
        assert len(pair_records) == 3
        for rec in pair_records:
            assert rec["classes_present"] == ["drone"]
            assert rec["alignment_shift_px"] is not None
            assert rec["alignment_shift_px"] > 0  # the mock injects a real few-px offset

        with open(os.path.join(out_root, "metadata", "videos.jsonl")) as f:
            video_records = [json.loads(line) for line in f]
        assert len(video_records) == 1
        assert video_records[0]["n_frames"] == 3
        assert video_records[0]["classes_present"] == ["drone"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_generic_adapter_raises_on_unexpected_layout():
    tmp = tempfile.mkdtemp(prefix="avid_build_test_bad_")
    try:
        try:
            list(generic_frame_folder_adapter(tmp, split="train"))
            raise AssertionError("expected FileNotFoundError for a layout with no RGB/ dir")
        except FileNotFoundError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_build_dataset_end_to_end()
    test_generic_adapter_raises_on_unexpected_layout()
    print("All build_hf_dataset tests passed.")
