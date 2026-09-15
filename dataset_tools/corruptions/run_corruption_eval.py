"""
Experiment E5/E6 harness: sweep a trained model over the full AVID-C grid
(7 corruptions x 5 severities), optionally with one modality dropped
entirely (Experiment E6, missing-modality robustness), and report a
per-corruption mean score plus an ImageNet-C-style aggregate "mean
corruption error/score" summary.

Model-agnostic by design: pass in `predict_fn(rgb, ir)` and
`score_fn(prediction, ground_truth)` for whatever model you've trained
yourself, and this sweeps corruptions around it. The toy model/self-test
below proves the sweep mechanics (corrupt -> predict -> score -> aggregate)
are correct without needing the real dataset or a trained model in this
sandbox. This paper's own use of AVID-C (see dataset_tools/theory/) is
entirely classifier-agnostic and does not use this harness; this file is
released so a future user with a trained detector of their own can.
"""

import numpy as np

from .corruptions import CORRUPTIONS, apply_corruption


def run_corruption_sweep(pairs, predict_fn, score_fn, modalities=("rgb", "ir"), seed=0):
    """
    pairs: iterable of (rgb, ir, ground_truth) triples.
    predict_fn: callable(rgb, ir) -> prediction (rgb/ir may be None if
        dropped via `modalities`).
    score_fn: callable(prediction, ground_truth) -> float, higher is better.
    modalities: which modalities get corrupted; use e.g. ("rgb",) to also
        study the missing/degraded-IR case (Experiment E6) by never
        corrupting IR and instead handling modality dropout in predict_fn.

    Returns a dict: {corruption_name: {severity: mean_score}}, plus a
    "clean" baseline row for reference.
    """
    results = {"clean": {0: float(np.mean([score_fn(predict_fn(rgb, ir), gt) for rgb, ir, gt in pairs]))}}

    for c_idx, name in enumerate(CORRUPTIONS):
        results[name] = {}
        for severity in range(1, 6):
            scores = []
            for i, (rgb, ir, gt) in enumerate(pairs):
                s = seed + c_idx * 1000 + severity * 10 + i
                rgb_c = apply_corruption(rgb, name, severity=severity, seed=s) if "rgb" in modalities else rgb
                ir_c = apply_corruption(ir, name, severity=severity, seed=s + 1) if "ir" in modalities else ir
                pred = predict_fn(rgb_c, ir_c)
                scores.append(score_fn(pred, gt))
            results[name][severity] = float(np.mean(scores))
    return results


def summarize(results):
    """Mean score per corruption (averaged over severities 1-5), and the
    overall mean-corruption-score across all 7 families -- the AVID-C
    analogue of ImageNet-C's mCE, except higher is better here since we
    report task score directly rather than an error ratio."""
    per_corruption = {
        name: float(np.mean(list(sev_scores.values()))) for name, sev_scores in results.items() if name != "clean"
    }
    overall = float(np.mean(list(per_corruption.values())))
    return {"clean": results["clean"][0], "per_corruption_mean": per_corruption, "mean_corruption_score": overall}


# ---------------------------------------------------------------------------
# Self-test with a toy centroid "detector", so this file's mechanics can be
# verified without the real dataset or a trained model.
# ---------------------------------------------------------------------------


def _toy_pairs(n=6, h=128, w=160, seed=0):
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
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
    ys, xs = np.where(ir > 200)
    if len(ys) == 0:
        return None
    return (float(ys.mean()), float(xs.mean()))


def _toy_score_fn(pred, gt):
    if pred is None:
        return 0.0
    dist = np.hypot(pred[0] - gt[0], pred[1] - gt[1])
    return float(max(0.0, 1.0 - dist / 20.0))


def test_corruption_sweep_degrades_relative_to_clean():
    pairs = _toy_pairs()
    results = run_corruption_sweep(pairs, _toy_predict_fn, _toy_score_fn, modalities=("ir",))
    summary = summarize(results)
    assert summary["mean_corruption_score"] < summary["clean"], summary


if __name__ == "__main__":
    test_corruption_sweep_degrades_relative_to_clean()
    pairs = _toy_pairs()
    results = run_corruption_sweep(pairs, _toy_predict_fn, _toy_score_fn, modalities=("ir",))
    summary = summarize(results)
    print("clean score:", summary["clean"])
    for name, score in sorted(summary["per_corruption_mean"].items(), key=lambda kv: kv[1]):
        print(f"  {name:22s} {score:.3f}")
    print("mean corruption score:", summary["mean_corruption_score"])
    print("run_corruption_eval self-test passed.")
