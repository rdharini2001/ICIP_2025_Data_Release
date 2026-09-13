"""
AVIDPairDataset: a torch Dataset over the AVID RGB-IR pairs, following the
directory/metadata schema in dataset_tools/hf_release/SCHEMA.md.

NOTE ON TEST STATUS: this module needs `torch` and the real, restructured
corpus, neither of which is available in the sandbox this was written in (see
PLAN.md and docs/RELEASE_RUNBOOK.md for why). It has been reviewed carefully
against that schema and against the fusion_models.py it feeds, but it has not
been executed. Before relying on it, run:

    python -m dataset_tools.baselines.dataset --root <path> --split val --fusion-mode early --smoke-test

which loads a handful of real pairs and prints tensor shapes/label counts.
"""

import json
import os

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # allow --smoke-test's own error message to be the first thing a user without torch sees
    torch = None
    Dataset = object


FUSION_MODES = ("rgb", "ir", "early", "dual")


class AVIDPairDataset(Dataset):
    def __init__(self, root, split, fusion_mode="dual", img_size=(256, 320), transform=None):
        if torch is None:
            raise ImportError("AVIDPairDataset requires PyTorch: pip install torch")
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}")
        self.root = root
        self.split = split
        self.fusion_mode = fusion_mode
        self.img_size = img_size  # (H, W)
        self.transform = transform

        meta_path = os.path.join(root, "metadata", "pairs.jsonl")
        self.records = []
        with open(meta_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                if rec["split"] == split:
                    self.records.append(rec)
        if not self.records:
            raise RuntimeError(f"No records found for split={split!r} in {meta_path}")

    def __len__(self):
        return len(self.records)

    def _load_image(self, path, as_gray=False):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        if img.shape[:2] != self.img_size:
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        if not as_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_labels(self, path, n_boxes_max=None):
        # YOLO format: one row per object, "class cx cy w h" normalized to [0, 1]
        boxes, labels = [], []
        if os.path.isfile(path):
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls, cx, cy, w, h = parts[:5]
                    labels.append(int(cls))
                    boxes.append([float(cx), float(cy), float(w), float(h)])
        return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(labels, dtype=np.int64)

    def __getitem__(self, idx):
        rec = self.records[idx]
        rgb_path = os.path.join(self.root, "images", "rgb", self.split, rec["video_id"], f"{rec['frame_idx']:06d}.jpg")
        ir_path = os.path.join(self.root, "images", "ir", self.split, rec["video_id"], f"{rec['frame_idx']:06d}.jpg")
        label_path = os.path.join(self.root, "labels", self.split, rec["video_id"], f"{rec['frame_idx']:06d}.txt")

        boxes, labels = self._load_labels(label_path)
        target = {"boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(labels), "meta": rec}

        if self.fusion_mode == "rgb":
            rgb = self._load_image(rgb_path)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        elif self.fusion_mode == "ir":
            ir = self._load_image(ir_path, as_gray=True)
            tensor = torch.from_numpy(ir).unsqueeze(0).float() / 255.0
        elif self.fusion_mode == "early":
            rgb = self._load_image(rgb_path)
            ir = self._load_image(ir_path, as_gray=True)
            stacked = np.concatenate([rgb, ir[:, :, None]], axis=2)  # (H, W, 4)
            tensor = torch.from_numpy(stacked).permute(2, 0, 1).float() / 255.0
        else:  # "dual": mid/late fusion models want the two streams separate
            rgb = self._load_image(rgb_path)
            ir = self._load_image(ir_path, as_gray=True)
            rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            ir_t = torch.from_numpy(ir).unsqueeze(0).float() / 255.0
            tensor = (rgb_t, ir_t)

        if self.transform is not None:
            tensor, target = self.transform(tensor, target)
        return tensor, target


def _smoke_test(root, split, fusion_mode):
    ds = AVIDPairDataset(root, split, fusion_mode=fusion_mode)
    print(f"{len(ds)} records in split={split!r}")
    sample, target = ds[0]
    if isinstance(sample, tuple):
        print("rgb tensor:", sample[0].shape, "ir tensor:", sample[1].shape)
    else:
        print("tensor:", sample.shape)
    print("boxes:", target["boxes"].shape, "labels:", target["labels"].shape)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--fusion-mode", default="dual", choices=FUSION_MODES)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        _smoke_test(args.root, args.split, args.fusion_mode)
