"""
Pushes a built AVID dataset directory (see SCHEMA.md) to a Hugging Face
Hub dataset repo. Requires `huggingface_hub` and an authenticated HF token
with write access (`huggingface-cli login`, or pass --token).

Not run in this sandbox (no dataset built here, and pushing to a real HF
repo is exactly the kind of externally-visible action that should happen
from a machine you control, with a token you control — see
docs/RELEASE_RUNBOOK.md). This script is a thin, reviewed wrapper around
the standard huggingface_hub upload API; there is no dataset-specific logic
here to unit-test independent of a real HF repo and real data.

Usage:
    python -m dataset_tools.hf_release.upload_to_hf \\
        --local-dir /path/to/avid \\
        --repo-id <your-hf-username>/avid \\
        --dataset-card /path/to/DATASET_CARD.md

The dataset card is copied to <local-dir>/README.md before upload (HF reads
that as the repo's card) -- it does not overwrite your working copy.
"""

import argparse
import shutil
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", required=True, help="path built by build_hf_dataset.py")
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/avid")
    parser.add_argument("--dataset-card", default=os.path.join(os.path.dirname(__file__), "DATASET_CARD.md"))
    parser.add_argument("--token", default=None, help="defaults to the token from `huggingface-cli login`")
    parser.add_argument("--private", action="store_true", help="create/push as a private repo")
    parser.add_argument(
        "--commit-message", default="Add AVID: aerial visible-infrared drone dataset (detection + tracking)"
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        raise SystemExit("pip install huggingface_hub, then re-run this script") from exc

    readme_dst = os.path.join(args.local_dir, "README.md")
    shutil.copyfile(args.dataset_card, readme_dst)
    print(f"Copied dataset card to {readme_dst}")

    api = HfApi(token=args.token)
    create_repo(args.repo_id, repo_type="dataset", private=args.private, token=args.token, exist_ok=True)
    print(f"Repo {args.repo_id} ready (private={args.private}). Uploading {args.local_dir} ...")

    api.upload_folder(
        folder_path=args.local_dir,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
    )
    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
