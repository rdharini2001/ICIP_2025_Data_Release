"""
Alignment-sensitivity study harness (Experiment E7 in PLAN.md).

Open empirical question this answers: given the corpus is already weakly
aligned, how much does *additional* misalignment actually cost a fusion
model? This harness injects controlled, increasing synthetic misalignment
(shift / rotation / scale) into the IR frame of a pair on top of whatever
native weak alignment already exists, and records how a caller-supplied
model's performance degrades as a function of injected misalignment
magnitude.

The harness is deliberately model-agnostic: pass in a `predict_fn(rgb, ir) ->
prediction` and a `score_fn(prediction, ground_truth) -> float` and it works
with any detector/tracker you've trained on the real corpus. This lets the
harness itself be tested and shipped now, without needing the real dataset or
a trained model in this environment -- see the toy template-matching
self-test at the bottom, which exercises the exact same code path a real
YOLO-fusion model would use.
"""

from dataclasses import dataclass, field

import numpy as np
import cv2


@dataclass
class MisalignmentLevel:
    shift_px: float
    rotation_deg: float
    scale_delta: float


# Default sweep: 0 px (native weak alignment only) up to 20 px / 8 degrees,
# mirroring the camera_instability severities in dataset_tools.corruptions
# but applied as a *static* per-pair misalignment rather than per-frame
# jitter, since the question here is "how far off can the two sensors be
# before fusion breaks", not "how shaky is the video".
DEFAULT_SWEEP = [
    MisalignmentLevel(0, 0, 0.0),
    MisalignmentLevel(2, 1, 0.005),
    MisalignmentLevel(5, 2, 0.01),
    MisalignmentLevel(10, 4, 0.02),
    MisalignmentLevel(15, 6, 0.03),
    MisalignmentLevel(20, 8, 0.04),
]


def inject_misalignment(ir, level: MisalignmentLevel, seed=None):
    """Apply a fixed rigid transform to `ir` simulating additional sensor
    misalignment beyond whatever weak alignment the pair already has."""
    rng = np.random.default_rng(seed)
    h, w = ir.shape[:2]
    angle = rng.uniform(-level.rotation_deg, level.rotation_deg) if level.rotation_deg else 0.0
    dx = rng.uniform(-level.shift_px, level.shift_px) if level.shift_px else 0.0
    dy = rng.uniform(-level.shift_px, level.shift_px) if level.shift_px else 0.0
    scale = 1.0 + (rng.uniform(-level.scale_delta, level.scale_delta) if level.scale_delta else 0.0)
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    mat[0, 2] += dx
    mat[1, 2] += dy
    return cv2.warpAffine(ir, mat, (w, h), borderMode=cv2.BORDER_REFLECT101)


def run_sensitivity_study(pairs, predict_fn, score_fn, sweep=None, seed=0):
    """
    pairs: iterable of (rgb, ir, ground_truth) triples.
    predict_fn: callable(rgb, ir) -> prediction (whatever score_fn expects).
    score_fn: callable(prediction, ground_truth) -> float (higher is better).
    sweep: list[MisalignmentLevel], defaults to DEFAULT_SWEEP.

    Returns a list of dicts: {shift_px, rotation_deg, scale_delta, mean_score, n}
    -- one row per sweep level, averaged over all pairs.
    """
    sweep = sweep or DEFAULT_SWEEP
    results = []
    for level in sweep:
        scores = []
        for i, (rgb, ir, gt) in enumerate(pairs):
            ir_perturbed = inject_misalignment(ir, level, seed=seed + i)
            pred = predict_fn(rgb, ir_perturbed)
            scores.append(score_fn(pred, gt))
        results.append(
            {
                "shift_px": level.shift_px,
                "rotation_deg": level.rotation_deg,
                "scale_delta": level.scale_delta,
                "mean_score": float(np.mean(scores)) if scores else float("nan"),
                "n": len(scores),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Self-test: a toy "model" and scorer that don't need the real dataset, to
# prove the harness mechanics (injection -> predict -> score -> aggregate)
# are correct end-to-end.
# ---------------------------------------------------------------------------


def _toy_pairs(n=8, h=128, w=160, seed=0):
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(n):
        cy, cx = rng.integers(30, h - 30), rng.integers(30, w - 30)
        rgb = np.full((h, w, 3), 100, dtype=np.uint8)
        ir = np.full((h, w), 60, dtype=np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= 8 ** 2
        rgb[blob] = [30, 30, 30]
        ir[blob] = 220
        pairs.append((rgb, ir, (cy, cx)))
    return pairs


def _toy_predict_fn(rgb, ir):
    """'Detector': the brightest blob centroid in the IR channel."""
    ys, xs = np.where(ir > 200)
    if len(ys) == 0:
        return None
    return (float(ys.mean()), float(xs.mean()))


def _toy_score_fn(pred, gt):
    if pred is None:
        return 0.0
    dist = np.hypot(pred[0] - gt[0], pred[1] - gt[1])
    return float(max(0.0, 1.0 - dist / 20.0))  # 1.0 = perfect, 0 at >=20px error


def test_sensitivity_study_degrades_monotonically():
    pairs = _toy_pairs()
    results = run_sensitivity_study(pairs, _toy_predict_fn, _toy_score_fn, seed=0)
    scores = [r["mean_score"] for r in results]
    assert scores[0] >= scores[-1], scores
    # allow a little non-monotonic noise in the middle, but the overall trend
    # across the full sweep must be a real drop, since this is a "does the
    # harness work" check rather than a claim about any real model.
    assert scores[0] - scores[-1] > 0.05, scores


if __name__ == "__main__":
    test_sensitivity_study_degrades_monotonically()
    for row in run_sensitivity_study(_toy_pairs(), _toy_predict_fn, _toy_score_fn, seed=0):
        print(row)
    print("Sensitivity-study harness self-test passed.")
