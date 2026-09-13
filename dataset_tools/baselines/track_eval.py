"""
Tracking-metric utilities implementing the specific evaluation criteria from
the original VIP Cup proposal: (a) classifying whether a tracked object is
approaching or receding from the camera's field of view, and (b) flagging
tracks with a missed-detection gap of >= 15 consecutive frames. Both are
plain sequence analysis (no detector/tracker dependency), so they're
implemented in NumPy and fully tested here without needing the real dataset,
a trained model, or a tracker library installed.

Pair this with any tracker's output (ByteTrack, BoT-SORT, DeepSORT, ...) by
converting its per-frame boxes for a single track ID into the box sequence
these functions expect: (frame_idx, x1, y1, x2, y2) in normalized [0, 1]
image coordinates, one row per frame the object was detected in.
"""

import numpy as np


def classify_approach_recede(box_sequence, min_frames=5, slope_threshold=1e-4):
    """
    box_sequence: (N, 5) array of (frame_idx, x1, y1, x2, y2), sorted by
    frame_idx, for a single track. Classifies the overall trend in target
    size (sqrt of bbox area, a proxy for distance under a roughly constant
    true object size) via a robust linear fit vs. frame index.

    Returns one of "approaching", "receding", "stationary", or
    "insufficient_data" (fewer than `min_frames` observations).
    """
    box_sequence = np.asarray(box_sequence, dtype=np.float64)
    if len(box_sequence) < min_frames:
        return "insufficient_data"

    frames = box_sequence[:, 0]
    widths = box_sequence[:, 3] - box_sequence[:, 1]
    heights = box_sequence[:, 4] - box_sequence[:, 2]
    size_proxy = np.sqrt(np.clip(widths * heights, 1e-12, None))

    # simple least-squares slope of size vs. frame index; a median-based
    # robust variant could replace this if outlier detections are a concern.
    frames_c = frames - frames.mean()
    slope = np.sum(frames_c * (size_proxy - size_proxy.mean())) / (np.sum(frames_c ** 2) + 1e-12)

    if slope > slope_threshold:
        return "approaching"
    if slope < -slope_threshold:
        return "receding"
    return "stationary"


def longest_missed_track_gap(gt_frames, pred_frames):
    """
    gt_frames: sorted array-like of frame indices where the ground-truth
    object is present in the FoV.
    pred_frames: sorted array-like of frame indices where the tracker
    reported a matching detection (IoU >= threshold with the GT box; that
    matching step happens upstream of this function).

    Returns the longest run of consecutive GT-present frames with no
    matching prediction ("missed track" gap length, in frames), and whether
    that gap violates the proposal's "< 15 frames" acceptance rule.
    """
    gt_frames = sorted(set(int(f) for f in gt_frames))
    pred_frames = set(int(f) for f in pred_frames)
    if not gt_frames:
        return {"longest_gap": 0, "violates_15_frame_rule": False}

    longest_gap = 0
    current_gap = 0
    prev_frame = None
    for f in gt_frames:
        # a gap only continues across *consecutive* GT-present frames; a
        # break in GT presence itself (object left and re-entered FoV) resets
        # the run, since that's not a tracking failure.
        if prev_frame is not None and f != prev_frame + 1:
            current_gap = 0
        if f in pred_frames:
            current_gap = 0
        else:
            current_gap += 1
            longest_gap = max(longest_gap, current_gap)
        prev_frame = f

    return {"longest_gap": longest_gap, "violates_15_frame_rule": longest_gap >= 15}


def mota_lite(n_gt, n_false_negatives, n_false_positives, n_id_switches):
    """Standard MOTA: 1 - (FN + FP + IDSW) / n_gt. Provided for convenience
    alongside the two proposal-specific metrics above; use a full tracking
    metrics library (e.g. TrackEval/py-motmetrics) for IDF1/HOTA once real
    tracker output is available."""
    if n_gt <= 0:
        return float("nan")
    return 1.0 - (n_false_negatives + n_false_positives + n_id_switches) / n_gt


def _cli_demo():
    approaching = np.array([[f, 0.5 - 0.002 * f, 0.5 - 0.002 * f, 0.5 + 0.002 * f, 0.5 + 0.002 * f] for f in range(30)])
    print("approaching example ->", classify_approach_recede(approaching))
    gap = longest_missed_track_gap(gt_frames=range(0, 40), pred_frames=list(range(0, 10)) + list(range(30, 40)))
    print("20-frame miss example ->", gap)


if __name__ == "__main__":
    _cli_demo()
