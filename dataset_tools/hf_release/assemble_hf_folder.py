"""Assemble the AVID Hugging Face release from local data and repository assets.

Usage::
    python -m dataset_tools.hf_release.assemble_hf_folder \
        --source-dir /path/to/curated_AVID --out /path/to/avid_hf_release

This is a local preparation step; it does not upload anything. When only
updating the dataset card and display GIFs on an existing HF dataset, prefer
release_notes/HF_CARD_ONLY_UPDATE.md and its allowlisted upload script.
"""
from pathlib import Path
import argparse
import os
import shutil

REPO_ROOT = Path(__file__).resolve().parents[2]


def copy_tree(src: Path, dst: Path):
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"  copied {src} -> {dst}")
    else:
        print(f"  SKIP (not found): {src}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True,
                        help="local curated AVID release folder containing images/, labels/, metadata/")
    parser.add_argument("--out", required=True, help="output directory for HF dataset")
    args = parser.parse_args()
    source, out = Path(args.source_dir).expanduser().resolve(), Path(args.out).expanduser().resolve()
    if source == out:
        parser.error("--source-dir and --out must not be the same directory")
    missing = [p for p in ("images", "labels", "metadata") if not (source / p).is_dir()]
    if missing:
        parser.error("source release is missing required folders: " + ", ".join(missing))
    out.mkdir(parents=True, exist_ok=True)
    for dirname in ("images", "labels", "metadata", "videos"):
        copy_tree(source / dirname, out / dirname)
    (out / "paper").mkdir(exist_ok=True)
    (out / "media").mkdir(exist_ok=True)
    (out / "assets").mkdir(exist_ok=True)
    shutil.copyfile(REPO_ROOT / "dataset_tools/hf_release/README_HF.md", out / "README.md")
    shutil.copyfile(REPO_ROOT / "dataset_tools/hf_release/LICENSE", out / "LICENSE")
    shutil.copyfile(REPO_ROOT / "paper/main.pdf", out / "paper/main.pdf")
    shutil.copyfile(REPO_ROOT / "paper/figures/fig_hf_teaser.png", out / "assets/avid_teaser.png")
    copy_tree(REPO_ROOT / "reproducibility_data", out / "reproducibility_data")
    src_gifs = REPO_ROOT / "assets/showcase"
    for name in ("DRONE_082192_RGB_IR.gif", "DRONE_083193_RGB_IR.gif",
                 "DRONE_085195_RGB_IR.gif", "BIRD_03897_RGB_IR.gif"):
        shutil.copyfile(src_gifs / name, out / "media" / name)
    for name in ("fig_diversity_montage.png", "fig_corruption_analysis.png",
                 "fig_dataset_statistics.png"):
        shutil.copyfile(REPO_ROOT / "paper/figures" / name, out / "media" / name)
    print(f"Prepared local release in {out}. It has NOT been uploaded to Hugging Face.")


if __name__ == "__main__":
    main()
