"""
Alignment Robustness Score (ARS): a general, reusable metric for how much
a cross-modal fusion model's performance depends on precise spatial
correspondence between its input modalities.

WHY THIS METRIC EXISTS (see paper Sec. "Alignment Robustness as a
First-Class Metric" for the full argument):

Classic multi-view learning theory (Blum & Mitchell, 1998, co-training)
assumes two views of an instance are (a) conditionally independent given
the label and (b) exact views of the *same* instance -- i.e. perfectly
registered. Every existing RGB-thermal fusion benchmark (LLVIP, M3FD,
KAIST) is built on co-located or heavily calibrated rigs that make (b)
approximately true, so no existing benchmark evaluates what happens when
it isn't. Weakly-paired sensor rigs (the realistic case for most fielded
dual-optic gimbals -- see AVID's own measured misalignment distribution,
dataset_tools/hf_release/real_alignment_sample.py) violate (b) directly:
there is an unknown, per-sample rigid nuisance transform T_i relating the
two modalities' coordinate frames.

ARS answers a question no existing RGB-T benchmark's evaluation protocol
can even ask: *how much does a given fusion architecture's accuracy
degrade as a function of that nuisance transform's magnitude?* This is the
direct geometric analogue of a Lipschitz-sensitivity / certified-radius
curve in the adversarial-robustness literature, except the perturbation is
a rigid 2D misalignment between modalities rather than an adversarial pixel
perturbation to a single input.

DEFINITION. Let s(delta) be a model's task score (mAP, F1, IoU-based
recall, ...) when a controlled misalignment of magnitude `delta` (pixels,
combining shift/rotation/scale via `dataset_tools.alignment.sensitivity_study
.MisalignmentLevel`) is injected into one modality at inference time, on
top of whatever native weak alignment the pair already has. Then:

  ARS_AUC(f)   = (1 / delta_max) * integral_0^delta_max s(delta) d(delta)
                  ~ normalized area under the degradation curve (this
                    module's trapezoidal estimate over the sweep points),
                    in [0, s(0)]. Higher = more robust to misalignment.

  ARS_RADIUS(f, tau) = the largest delta at which s(delta) >= tau * s(0)
                  ~ direct analogue of "certified radius": the misalignment
                    budget the model tolerates before falling below a
                    tau-fraction of its clean-alignment performance.
                    Reported as None if s(delta) never drops below tau*s(0)
                    within the tested sweep (robust throughout the tested
                    range, not "infinitely" robust).

Both are computed from the same sweep this module runs via
`dataset_tools.alignment.sensitivity_study.run_sensitivity_study`, so no
new data collection is needed beyond what that harness already does --
this module only adds the two standardized scalar summaries plus the
theoretical framing above, so results are comparable across models/papers.
"""

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset_tools.alignment.sensitivity_study import (
    DEFAULT_SWEEP,
    MisalignmentLevel,
    run_sensitivity_study,
)


def summarize_curve(shifts, scores, tau=0.5):
    """Pure math half of the metric, factored out so it's testable against
    hand-constructed synthetic curves independent of the sweep-running
    machinery (which needs a real or toy model to call). See `compute_ars`
    for the scalar definitions."""
    shifts = np.asarray(shifts, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(shifts)
    shifts, scores = shifts[order], scores[order]

    clean_score = float(scores[0])  # shift=0 must be the first sweep point
    delta_max = float(shifts[-1])

    # trapezoidal AUC, normalized by the sweep width so ARS_AUC has units of
    # "average score over the tested misalignment range" (directly
    # comparable to clean_score, not an arbitrary integral scale).
    # np.trapz was removed in NumPy 2.0 in favor of np.trapezoid; support both.
    _trapezoid = getattr(np, "trapezoid", None) or np.trapz
    auc = float(_trapezoid(scores, shifts) / delta_max) if delta_max > 0 else clean_score

    radius = None
    threshold = tau * clean_score
    for shift, score in zip(shifts, scores):
        if score < threshold:
            radius = float(shift)
            break

    return {
        "curve": [{"shift_px": float(s), "mean_score": float(v)} for s, v in zip(shifts, scores)],
        "clean_score": clean_score,
        "ars_auc": auc,
        "ars_auc_relative": auc / clean_score if clean_score > 0 else float("nan"),
        "ars_radius_px": radius,
        "tau": tau,
    }


def compute_ars(pairs, predict_fn, score_fn, sweep=None, tau=0.5, seed=0):
    """
    Runs the misalignment sweep and returns both ARS summaries plus the raw
    curve, for one model.

    Returns:
        {
          "curve": [{"shift_px": ..., "mean_score": ...}, ...],  # raw sweep
          "clean_score": s(0),
          "ars_auc": normalized area under the curve (0 to clean_score),
          "ars_auc_relative": ars_auc / clean_score (0 to 1, comparable
              across models/tasks with different absolute score scales),
          "ars_radius_px": critical misalignment radius at threshold tau,
              or None if the model never drops below tau within the sweep,
          "tau": the threshold used,
        }
    """
    sweep = sweep or DEFAULT_SWEEP
    results = run_sensitivity_study(pairs, predict_fn, score_fn, sweep=sweep, seed=seed)
    shifts = [r["shift_px"] for r in results]
    scores = [r["mean_score"] for r in results]
    return summarize_curve(shifts, scores, tau=tau)


def compare_models(pairs, model_predict_fns, score_fn, sweep=None, tau=0.5, seed=0):
    """Convenience wrapper: model_predict_fns is a dict {name: predict_fn}.
    Returns {name: compute_ars(...)} so multiple fusion strategies (e.g.
    'early_fusion', 'mid_fusion', 'late_fusion_wbf') can be compared under
    an identical sweep and identical data, which is the actual scientific
    comparison this metric exists to enable."""
    return {name: compute_ars(pairs, fn, score_fn, sweep=sweep, tau=tau, seed=seed) for name, fn in model_predict_fns.items()}


# ---------------------------------------------------------------------------
# Self-tests.
#
# 1. Pure-math tests on hand-constructed curves: verify summarize_curve
#    recovers a KNOWN robustness ordering (a "fragile" curve that decays
#    fast must score lower ars_auc_relative and a smaller/no critical
#    radius than a "robust" curve that decays slowly), independent of any
#    model or the sweep-running machinery.
# 2. An integration smoke test using the same simple toy centroid-matching
#    "model" already validated in sensitivity_study.py, confirming
#    compute_ars's plumbing (run the sweep, then summarize) works
#    end-to-end and produces internally consistent output.
# ---------------------------------------------------------------------------


def test_summarize_curve_ranks_fragile_below_robust():
    shifts = [0, 5, 10, 15, 20]
    robust_scores = [1.0, 0.95, 0.90, 0.85, 0.80]  # slow, graceful decay
    fragile_scores = [1.0, 0.60, 0.30, 0.10, 0.02]  # fast collapse

    robust = summarize_curve(shifts, robust_scores, tau=0.5)
    fragile = summarize_curve(shifts, fragile_scores, tau=0.5)

    assert robust["ars_auc_relative"] > fragile["ars_auc_relative"], (
        robust["ars_auc_relative"], fragile["ars_auc_relative"],
    )
    # robust never drops below tau*clean_score=0.5 within the sweep -> None;
    # fragile drops below 0.5 at shift=10 (score 0.30).
    assert robust["ars_radius_px"] is None
    assert fragile["ars_radius_px"] == 10


def test_summarize_curve_auc_equals_clean_score_at_single_point():
    result = summarize_curve([0], [0.83], tau=0.5)
    assert abs(result["ars_auc"] - 0.83) < 1e-9
    assert result["clean_score"] == 0.83


def test_summarize_curve_handles_unsorted_input():
    # shifts arrive out of order -- summarize_curve must sort before
    # computing AUC/radius, not assume the caller already did.
    shifts = [10, 0, 5]
    scores = [0.5, 1.0, 0.8]
    result = summarize_curve(shifts, scores, tau=0.5)
    assert result["clean_score"] == 1.0
    assert result["curve"][0] == {"shift_px": 0.0, "mean_score": 1.0}


def _toy_pairs(n=8, h=128, w=160, seed=0):
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


def test_compute_ars_integration_smoke():
    pairs = _toy_pairs()
    result = compute_ars(pairs, _toy_predict_fn, _toy_score_fn, seed=0)
    assert result["clean_score"] > 0.9  # the toy detector is near-perfect at zero injected misalignment
    assert result["ars_auc"] <= result["clean_score"] + 1e-9  # AUC can't exceed the clean baseline
    assert 0.0 <= result["ars_auc_relative"] <= 1.0 + 1e-9
    # the toy detector visibly degrades over the default sweep (already
    # shown in sensitivity_study.py's own test), so it should cross a 50%
    # threshold somewhere within the tested range.
    assert result["ars_radius_px"] is not None


def test_compare_models_returns_one_entry_per_model():
    pairs = _toy_pairs()
    comparison = compare_models(pairs, {"toy_a": _toy_predict_fn, "toy_b": _toy_predict_fn}, _toy_score_fn)
    assert set(comparison.keys()) == {"toy_a", "toy_b"}
    assert comparison["toy_a"]["ars_auc"] == comparison["toy_b"]["ars_auc"]  # identical model -> identical score


if __name__ == "__main__":
    test_summarize_curve_ranks_fragile_below_robust()
    test_summarize_curve_auc_equals_clean_score_at_single_point()
    test_summarize_curve_handles_unsorted_input()
    test_compute_ars_integration_smoke()
    test_compare_models_returns_one_entry_per_model()

    pairs = _toy_pairs()
    result = compute_ars(pairs, _toy_predict_fn, _toy_score_fn, seed=0)
    print("toy detector ARS summary:")
    for row in result["curve"]:
        print(f"  shift={row['shift_px']:>5}px  score={row['mean_score']:.3f}")
    print(f"  ars_auc={result['ars_auc']:.3f}  ars_auc_relative={result['ars_auc_relative']:.3f}  "
          f"critical radius (tau={result['tau']})={result['ars_radius_px']}")
    print("All ARS metric tests passed.")
