"""
Real robustness/corruption evaluation harness for a trained AVID YOLO
checkpoint, using Ultralytics' own val() pipeline (proper mAP50/mAP50-95/
precision/recall) rather than a hand-rolled metric.

Scope note: only the 6 *photometric* AVID-C corruptions are evaluated here
(gaussian_noise, speckle_noise, salt_and_pepper, gaussian_blur,
uneven_illumination, motion_blur) -- these don't move object locations, so
the existing YOLO-format label files stay valid under them. camera_instability
is a *geometric* corruption (it warps pixel content), which would silently
invalidate the label boxes if applied the same way; that one is correctly
handled instead by dataset_tools/alignment/sensitivity_study.py, which
tracks the injected transform explicitly rather than assuming labels stay
fixed.

To keep this tractable without materializing 100K+ new files, evaluation
runs on a fixed random subsample of the validation set (default 400 images)
-- standard practice for a corruption-robustness sweep (ImageNet-C itself
validates on subsets for several of its own follow-up analyses).
"""

import json
import os
import random
import shutil
import sys

import cv2

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.corruptions.corruptions import CORRUPTIONS

PHOTOMETRIC_CORRUPTIONS = [
    "gaussian_noise", "speckle_noise", "salt_and_pepper",
    "gaussian_blur", "uneven_illumination", "motion_blur",
]

VAL_LIST = r"E:\ICIP_2025\avid_release\lists\val_rgb.txt"
WORK_DIR = r"E:\ICIP_2025\avid_release\corruption_eval"


def sample_val_subset(n=400, seed=0):
    with open(VAL_LIST) as f:
        paths = [l.strip() for l in f if l.strip()]
    random.Random(seed).shuffle(paths)
    return paths[:n]


def materialize_corrupted_subset(image_paths, corruption, severity, out_dir):
    """Writes corrupted copies of image_paths (+ symlinked/copied labels) to
    out_dir/images and out_dir/labels, and returns a data.yaml-ready dict."""
    img_out = os.path.join(out_dir, "images")
    lbl_out = os.path.join(out_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    for i, path in enumerate(image_paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if corruption != "clean":
            img_rgb = CORRUPTIONS[corruption](img_rgb, severity=severity, seed=i)
        out_img_path = os.path.join(img_out, f"{i:05d}.jpg")
        cv2.imwrite(out_img_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        label_src = path.replace("\\images\\", "\\labels\\").rsplit(".", 1)[0] + ".txt"
        label_dst = os.path.join(lbl_out, f"{i:05d}.txt")
        if os.path.isfile(label_src):
            shutil.copyfile(label_src, label_dst)
        else:
            open(label_dst, "w").close()


def write_yaml(out_dir, yaml_path):
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir}\nval: images\nnc: 2\nnames: [\"bird\", \"drone\"]\n")


def run_sweep(model_path, n_images=400, severities=(1, 2, 3, 4, 5)):
    from ultralytics import YOLO

    results = {}
    subset = sample_val_subset(n=n_images)

    # clean baseline
    clean_dir = os.path.join(WORK_DIR, "clean")
    materialize_corrupted_subset(subset, "clean", 0, clean_dir)
    yaml_path = os.path.join(WORK_DIR, "clean.yaml")
    write_yaml(clean_dir, yaml_path)
    model = YOLO(model_path)
    r = model.val(data=yaml_path, imgsz=320, split="val", verbose=False)
    results["clean"] = {"map50": float(r.box.map50), "map5095": float(r.box.map), "precision": float(r.box.mp), "recall": float(r.box.mr)}
    print("clean:", results["clean"])

    for corruption in PHOTOMETRIC_CORRUPTIONS:
        results[corruption] = {}
        for severity in severities:
            out_dir = os.path.join(WORK_DIR, f"{corruption}_s{severity}")
            materialize_corrupted_subset(subset, corruption, severity, out_dir)
            yaml_path = os.path.join(WORK_DIR, f"{corruption}_s{severity}.yaml")
            write_yaml(out_dir, yaml_path)
            model = YOLO(model_path)  # reload to avoid any dataloader caching cross-talk
            r = model.val(data=yaml_path, imgsz=320, split="val", verbose=False)
            row = {"map50": float(r.box.map50), "map5095": float(r.box.map), "precision": float(r.box.mp), "recall": float(r.box.mr)}
            results[corruption][severity] = row
            print(f"{corruption} s{severity}:", row)
            shutil.rmtree(out_dir, ignore_errors=True)  # reclaim disk as we go

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-images", type=int, default=400)
    parser.add_argument("--out", default=r"E:\ICIP_2025\avid_release\metadata\corruption_sweep_rgb.json")
    args = parser.parse_args()

    results = run_sweep(args.model, n_images=args.n_images)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)
