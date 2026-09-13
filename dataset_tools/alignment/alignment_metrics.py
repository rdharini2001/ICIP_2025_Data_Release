"""
Weak-alignment quantification for the AVID RGB-IR pairs.

The RGB and IR sensors are physically offset (different optics, different
FOVs, boresight-aligned at best) so pairs are *weakly* aligned rather than
pixel-registered like LLVIP or M3FD. Rather than treating that as a defect to
hide, this module estimates, per pair, how large the residual geometric
offset is -- so it can be published as metadata (`alignment_shift_px`,
`alignment_confidence`, `alignment_method`) and downstream users can filter,
weight, or stratify results by alignment quality instead of discovering it
the hard way.

Two independent estimators are provided because they fail in different, and
useful-to-cross-check, ways:

1. `estimate_shift_phase_correlation` -- cheap, subpixel, FFT-based phase
   correlation on Sobel edge-magnitude maps (edges are one of the few
   structures that tend to co-occur across RGB and thermal, since object
   boundaries usually produce both a visible-light and a thermal edge).
   Fast, but can lock onto the wrong edge in low-texture scenes (e.g. plain
   sky) or when the target itself dominates the edge map asymmetrically.

2. `estimate_shift_mutual_information` -- brute-force integer-shift search
   maximizing normalized mutual information between grayscale RGB and IR.
   More robust for genuinely multi-modal statistics (a warm object can be
   dark in one modality and bright in the other), at the cost of being
   integer-resolution and O(search_window^2).

`pair_alignment_quality` runs both and returns a combined record; large
disagreement between the two estimators is itself a useful low-confidence
signal (exposed as `estimator_disagreement_px`).
"""

import numpy as np
import cv2


def _to_gray(img):
    if img.ndim == 2:
        return img
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img[:, :, 0]


def edge_map(img):
    """Sobel gradient-magnitude map, normalized to [0, 1] float32."""
    gray = _to_gray(img).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (mag.max() + 1e-8)
    return mag


def estimate_shift_phase_correlation(rgb, ir):
    """Subpixel (dx, dy) translational offset of `ir` relative to `rgb`,
    estimated via phase correlation on Sobel edge maps. `dx`/`dy` are the
    shift to apply to `ir` to align it onto `rgb` (positive dx = ir needs to
    move right). Returns dict(dx, dy, confidence) where confidence is the
    phase-correlation peak response in [0, 1] (higher = more trustworthy)."""
    h, w = rgb.shape[:2]
    ir_r = cv2.resize(ir, (w, h), interpolation=cv2.INTER_LINEAR) if ir.shape[:2] != (h, w) else ir

    e_rgb = edge_map(rgb)
    e_ir = edge_map(ir_r)

    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(e_rgb * window, e_ir * window)
    return {"dx": float(dx), "dy": float(dy), "confidence": float(response)}


def _normalized_mutual_information(a, b, bins=32):
    a = a.ravel()
    b = b.ravel()
    joint_hist, _, _ = np.histogram2d(a, b, bins=bins)
    joint_p = joint_hist / (joint_hist.sum() + 1e-12)
    a_p = joint_p.sum(axis=1)
    b_p = joint_p.sum(axis=0)

    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    h_a, h_b = entropy(a_p), entropy(b_p)
    h_joint = entropy(joint_p.ravel())
    if h_joint <= 1e-12:
        return 0.0
    return float((h_a + h_b) / h_joint)  # NMI in [1, 2], 2 = perfectly redundant


def estimate_shift_mutual_information(rgb, ir, max_shift=15, bins=32):
    """Brute-force integer-shift search maximizing normalized mutual
    information between grayscale RGB and IR over a `[-max_shift, max_shift]`
    px window in both axes. Returns dict(dx, dy, confidence) where confidence
    is the best NMI value found (roughly in [1, 2])."""
    h, w = rgb.shape[:2]
    ir_r = cv2.resize(ir, (w, h), interpolation=cv2.INTER_LINEAR) if ir.shape[:2] != (h, w) else ir
    gray_rgb = _to_gray(rgb)
    gray_ir = _to_gray(ir_r)

    best = {"dx": 0, "dy": 0, "confidence": -np.inf}
    margin = max_shift
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            ir_shifted = np.roll(np.roll(gray_ir, dy, axis=0), dx, axis=1)
            crop = slice(margin, h - margin), slice(margin, w - margin)
            if crop[0].stop <= crop[0].start or crop[1].stop <= crop[1].start:
                raise ValueError("max_shift too large for image size")
            nmi = _normalized_mutual_information(gray_rgb[crop], ir_shifted[crop], bins=bins)
            if nmi > best["confidence"]:
                best = {"dx": dx, "dy": dy, "confidence": nmi}
    return best


def pair_alignment_quality(rgb, ir, max_shift=15):
    """Run both estimators and return a combined per-pair alignment record."""
    pc = estimate_shift_phase_correlation(rgb, ir)
    mi = estimate_shift_mutual_information(rgb, ir, max_shift=max_shift)
    disagreement = float(np.hypot(pc["dx"] - mi["dx"], pc["dy"] - mi["dy"]))
    shift_px = float(np.hypot(mi["dx"], mi["dy"]))
    return {
        "phase_correlation": pc,
        "mutual_information": mi,
        "alignment_shift_px": shift_px,
        "estimator_disagreement_px": disagreement,
    }


def _cli():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Estimate RGB-IR weak-alignment offset for one pair.")
    parser.add_argument("rgb", help="Path to the RGB image.")
    parser.add_argument("ir", help="Path to the IR/thermal image.")
    parser.add_argument("--max-shift", type=int, default=15)
    args = parser.parse_args()

    rgb = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    ir = cv2.imread(args.ir, cv2.IMREAD_UNCHANGED)
    if rgb is None or ir is None:
        raise FileNotFoundError("could not read one of the input images")
    result = pair_alignment_quality(rgb, ir, max_shift=args.max_shift)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
