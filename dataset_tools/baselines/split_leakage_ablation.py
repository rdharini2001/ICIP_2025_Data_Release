"""
Experiment E8: quantify the data-leakage risk of frame-level random
train/val/test splitting on a 10s/30fps video corpus, versus splitting by
whole video. Consecutive frames of the same 10s clip are near-duplicates
(the same object, same background, a few pixels of motion apart); a
frame-level random split puts near-duplicate frames on both sides of the
split, which inflates validation/test metrics relative to genuine
generalization. This module makes that leakage measurable rather than
anecdotal, using only frame/video indices -- no pixels or trained model
needed to demonstrate the effect exists in a given split.
"""

import numpy as np


def frame_level_split(frame_index, val_frac=0.15, test_frac=0.15, seed=0):
    """frame_index: list of (video_id, frame_idx). Splits individual frames
    uniformly at random, ignoring which video they came from."""
    rng = np.random.default_rng(seed)
    n = len(frame_index)
    perm = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = set(perm[:n_test].tolist())
    val_idx = set(perm[n_test:n_test + n_val].tolist())
    splits = []
    for i in range(n):
        if i in test_idx:
            splits.append("test")
        elif i in val_idx:
            splits.append("val")
        else:
            splits.append("train")
    return splits


def video_level_split(frame_index, val_frac=0.15, test_frac=0.15, seed=0):
    """frame_index: list of (video_id, frame_idx). Splits whole videos, so
    every frame of a given video lands in exactly one split -- the correct
    way to split a temporally-correlated video corpus."""
    rng = np.random.default_rng(seed)
    video_ids = sorted({v for v, _ in frame_index})
    perm = rng.permutation(len(video_ids))
    n_test = int(len(video_ids) * test_frac)
    n_val = int(len(video_ids) * val_frac)
    video_split = {}
    for rank, vid_pos in enumerate(perm):
        vid = video_ids[vid_pos]
        if rank < n_test:
            video_split[vid] = "test"
        elif rank < n_test + n_val:
            video_split[vid] = "val"
        else:
            video_split[vid] = "train"
    return [video_split[v] for v, _ in frame_index]


def leakage_report(frame_index, splits, neighbor_window=3):
    """
    For a frame-level split, count how many train-split frames are within
    `neighbor_window` frames (same video) of a val/test-split frame --
    i.e. near-duplicate leakage across the split boundary. Returns the raw
    count and the fraction of train frames affected.
    """
    by_video = {}
    for (vid, fidx), split in zip(frame_index, splits):
        by_video.setdefault(vid, []).append((fidx, split))

    leaked = 0
    total_train = 0
    for vid, entries in by_video.items():
        entries.sort()
        for i, (fidx, split) in enumerate(entries):
            if split != "train":
                continue
            total_train += 1
            for j in range(max(0, i - neighbor_window), min(len(entries), i + neighbor_window + 1)):
                if j == i:
                    continue
                other_fidx, other_split = entries[j]
                if other_split != "train" and abs(other_fidx - fidx) <= neighbor_window:
                    leaked += 1
                    break
    frac = leaked / total_train if total_train else 0.0
    return {"leaked_train_frames": leaked, "total_train_frames": total_train, "leak_fraction": frac}


def compare_split_strategies(frame_index, neighbor_window=3, seed=0):
    """Convenience wrapper: runs both split strategies and reports the
    leakage delta, for a corpus described only by (video_id, frame_idx)
    pairs (e.g. 300 videos x 300 frames each at 30fps/10s)."""
    fl_splits = frame_level_split(frame_index, seed=seed)
    vl_splits = video_level_split(frame_index, seed=seed)
    return {
        "frame_level": leakage_report(frame_index, fl_splits, neighbor_window),
        "video_level": leakage_report(frame_index, vl_splits, neighbor_window),
    }


def _synthetic_corpus(n_videos=50, frames_per_video=300):
    return [(f"video_{v:03d}", f) for v in range(n_videos) for f in range(frames_per_video)]


if __name__ == "__main__":
    corpus = _synthetic_corpus()
    report = compare_split_strategies(corpus)
    print("frame-level split leakage:", report["frame_level"])
    print("video-level split leakage:", report["video_level"])
    assert report["video_level"]["leak_fraction"] == 0.0
    assert report["frame_level"]["leak_fraction"] > report["video_level"]["leak_fraction"]
    print("Split-leakage ablation self-check passed: frame-level splitting leaks, video-level does not.")
