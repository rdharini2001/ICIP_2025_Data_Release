"""
Assembles the final, upload-ready Hugging Face dataset folder from the
pieces built/maintained elsewhere in this repo: the real images/labels/
metadata from build_avid_v1.py's output, the polished dataset card
(README_HF.md), the dataset LICENSE, the compiled paper PDF, the
reproducibility CSVs, and the teaser image. Pure file assembly -- no new
numbers are computed here.

Usage:
    python -m dataset_tools.hf_release.assemble_hf_folder --out E:\\ICIP_2025\\avid_hf_release
"""
import argparse
import os
import shutil

REPO_ROOT = r"C:\Users\dragha4\avid_repo"
AVID_RELEASE = r"E:\ICIP_2025\avid_release"


def copy_tree(src, dst):
    if not os.path.isdir(src):
        print(f"  SKIP (not found): {src}")
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"  copied {src} -> {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="destination folder, e.g. E:\\ICIP_2025\\avid_hf_release")
    args = parser.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "assets"), exist_ok=True)
    os.makedirs(os.path.join(out, "paper"), exist_ok=True)

    print("Copying real data (images, labels, metadata) ...")
    copy_tree(os.path.join(AVID_RELEASE, "images"), os.path.join(out, "images"))
    copy_tree(os.path.join(AVID_RELEASE, "labels"), os.path.join(out, "labels"))
    copy_tree(os.path.join(AVID_RELEASE, "metadata"), os.path.join(out, "metadata"))

    print("Copying dataset card, license, paper, reproducibility data ...")
    shutil.copyfile(os.path.join(REPO_ROOT, "dataset_tools", "hf_release", "README_HF.md"),
                     os.path.join(out, "README.md"))
    shutil.copyfile(os.path.join(REPO_ROOT, "dataset_tools", "hf_release", "LICENSE"),
                     os.path.join(out, "LICENSE"))
    shutil.copyfile(os.path.join(REPO_ROOT, "paper", "main.pdf"),
                     os.path.join(out, "paper", "main.pdf"))
    shutil.copyfile(os.path.join(REPO_ROOT, "paper", "figures", "fig_hf_teaser.png"),
                     os.path.join(out, "assets", "avid_teaser.png"))
    copy_tree(os.path.join(REPO_ROOT, "reproducibility_data"), os.path.join(out, "reproducibility_data"))

    n_images = sum(len(files) for _, _, files in os.walk(os.path.join(out, "images")))
    n_labels = sum(len(files) for _, _, files in os.walk(os.path.join(out, "labels")))
    print(f"\nDone. {out}:")
    print(f"  {n_images} image files, {n_labels} label files")
    print("  Sanity-check README.md's YAML frontmatter before uploading.")


if __name__ == "__main__":
    main()
