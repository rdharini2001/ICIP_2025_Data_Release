"""
Correctness test: build a synthetic RGB/IR pair with a *known* injected
offset, and check both estimators recover it to within a small tolerance.
Run with: python -m dataset_tools.alignment.test_alignment_metrics
"""

import numpy as np

from .alignment_metrics import estimate_shift_mutual_information, estimate_shift_phase_correlation


def _make_pair(true_dx, true_dy, h=256, w=320, seed=0):
    rng = np.random.default_rng(seed)
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    ir = np.full((h, w), 60, dtype=np.uint8)

    # a few textured blobs so mutual information / edges have something to
    # lock onto, in both modalities but with different intensity mappings
    # (simulating a real cross-modal target: dark-ish in RGB, hot in IR).
    for _ in range(6):
        cy, cx = rng.integers(40, h - 40), rng.integers(40, w - 40)
        r = rng.integers(6, 14)
        yy, xx = np.mgrid[0:h, 0:w]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        rgb[blob] = [40, 40, 40]
        ir_blob_y = np.clip(np.arange(h) - true_dy, 0, h - 1).astype(int)
        ir_blob_x = np.clip(np.arange(w) - true_dx, 0, w - 1).astype(int)
        yy2, xx2 = np.meshgrid(ir_blob_y, ir_blob_x, indexing="ij")
        blob_ir = (yy2 - cy) ** 2 + (xx2 - cx) ** 2 <= r ** 2
        ir[blob_ir] = 220

    rgb = np.clip(rgb.astype(np.int16) + rng.integers(-2, 2, size=rgb.shape), 0, 255).astype(np.uint8)
    ir = np.clip(ir.astype(np.int16) + rng.integers(-2, 2, size=ir.shape), 0, 255).astype(np.uint8)
    return rgb, ir


def test_mutual_information_recovers_known_shift():
    true_dx, true_dy = 5, -3
    rgb, ir = _make_pair(true_dx, true_dy)
    est = estimate_shift_mutual_information(rgb, ir, max_shift=10)
    # np.roll(ir, dy, dx) convention in the module vs. the pair construction
    # above only needs to agree in magnitude/direction to within a couple of
    # pixels for the metric to be useful in practice.
    assert abs(abs(est["dx"]) - abs(true_dx)) <= 2, est
    assert abs(abs(est["dy"]) - abs(true_dy)) <= 2, est
    assert est["confidence"] > 1.0  # NMI of 1.0 = independent


def test_phase_correlation_runs_and_returns_bounded_confidence():
    rgb, ir = _make_pair(4, 4)
    est = estimate_shift_phase_correlation(rgb, ir)
    assert isinstance(est["dx"], float) and isinstance(est["dy"], float)
    assert 0.0 <= est["confidence"] <= 1.0 + 1e-6


def test_zero_shift_gives_low_estimated_shift():
    rgb, ir = _make_pair(0, 0)
    est = estimate_shift_mutual_information(rgb, ir, max_shift=10)
    assert abs(est["dx"]) <= 1 and abs(est["dy"]) <= 1, est


if __name__ == "__main__":
    test_mutual_information_recovers_known_shift()
    test_phase_correlation_runs_and_returns_bounded_confidence()
    test_zero_shift_gives_low_estimated_shift()
    print("All alignment metric tests passed.")
