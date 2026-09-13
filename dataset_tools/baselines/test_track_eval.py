import numpy as np

from .track_eval import classify_approach_recede, longest_missed_track_gap, mota_lite


def test_approaching_track_grows_in_size():
    frames = np.arange(20)
    half_size = 0.02 + 0.01 * frames  # box grows over time -> approaching
    boxes = np.stack([frames, 0.5 - half_size, 0.5 - half_size, 0.5 + half_size, 0.5 + half_size], axis=1)
    assert classify_approach_recede(boxes) == "approaching"


def test_receding_track_shrinks_in_size():
    frames = np.arange(20)
    half_size = 0.20 - 0.01 * frames  # box shrinks over time -> receding
    boxes = np.stack([frames, 0.5 - half_size, 0.5 - half_size, 0.5 + half_size, 0.5 + half_size], axis=1)
    assert classify_approach_recede(boxes) == "receding"


def test_stationary_track_is_stationary():
    frames = np.arange(20)
    half_size = np.full(20, 0.05)
    boxes = np.stack([frames, 0.5 - half_size, 0.5 - half_size, 0.5 + half_size, 0.5 + half_size], axis=1)
    assert classify_approach_recede(boxes) == "stationary"


def test_insufficient_data_flag():
    boxes = np.array([[0, 0.1, 0.1, 0.2, 0.2], [1, 0.1, 0.1, 0.2, 0.2]])
    assert classify_approach_recede(boxes, min_frames=5) == "insufficient_data"


def test_missed_track_gap_under_threshold_does_not_violate_rule():
    gt = range(0, 30)
    pred = [f for f in gt if f % 20 != 0]  # sparse single-frame misses, never 15 in a row
    result = longest_missed_track_gap(gt, pred)
    assert result["longest_gap"] < 15
    assert result["violates_15_frame_rule"] is False


def test_missed_track_gap_at_threshold_violates_rule():
    gt = list(range(0, 40))
    pred = list(range(0, 10)) + list(range(25, 40))  # frames 10..24 missed = 15-frame gap
    result = longest_missed_track_gap(gt, pred)
    assert result["longest_gap"] == 15
    assert result["violates_15_frame_rule"] is True


def test_missed_track_gap_resets_when_object_leaves_fov():
    # object present 0-9, absent 10-19 (left FoV, not a tracking failure),
    # present again 20-39 but never detected in that second stretch (20
    # consecutive true misses starting fresh).
    gt = list(range(0, 10)) + list(range(20, 40))
    pred = list(range(0, 10))
    result = longest_missed_track_gap(gt, pred)
    assert result["longest_gap"] == 20
    assert result["violates_15_frame_rule"] is True


def test_mota_lite_perfect_tracking():
    assert mota_lite(n_gt=100, n_false_negatives=0, n_false_positives=0, n_id_switches=0) == 1.0


def test_mota_lite_penalizes_errors():
    score = mota_lite(n_gt=100, n_false_negatives=10, n_false_positives=5, n_id_switches=2)
    assert abs(score - 0.83) < 1e-9


if __name__ == "__main__":
    test_approaching_track_grows_in_size()
    test_receding_track_shrinks_in_size()
    test_stationary_track_is_stationary()
    test_insufficient_data_flag()
    test_missed_track_gap_under_threshold_does_not_violate_rule()
    test_missed_track_gap_at_threshold_violates_rule()
    test_missed_track_gap_resets_when_object_leaves_fov()
    test_mota_lite_perfect_tracking()
    test_mota_lite_penalizes_errors()
    print("All track_eval tests passed.")
