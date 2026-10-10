"""
AVID-C: a reproducible, parameterized corruption benchmark for the AVID
RGB-IR drone/bird dataset.

Methodology follows Hendrycks & Dietterich (ICLR 2019), "Benchmarking Neural
Network Robustness to Common Corruptions and Perturbations" (5 severities per
corruption type), applied to the 7 corruption families used in the original
IEEE ICIP 2025 VIP Cup dataset design: speckle noise, salt-and-pepper noise,
Gaussian blur, uneven illumination, motion blur, camera instability/jitter,
and additive white Gaussian noise. The severity parameter tables below are
our own calibration for 320x256 8-bit RGB/thermal drone imagery -- they are
not a copy of the ImageNet-C parameter tables, which were calibrated for a
different image domain and resolution.

The point of formalizing this as code (rather than the 30 fixed pre-rendered
distorted videos in the original release) is that any user can now corrupt
*any* subset of the corpus, at *any* of the 5 severities, deterministically
(seeded) -- turning a one-off feature of the original challenge into a
general, extensible robustness benchmark.

All functions operate on uint8 arrays shaped (H, W, C) or (H, W) in [0, 255]
and return an array of the same shape and dtype.
"""

import os

import cv2
import numpy as np


def _rng(seed):
    return np.random.default_rng(seed)


def _to_float(img):
    return img.astype(np.float64) / 255.0


def _to_uint8(img):
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def gaussian_noise(img, severity=3, seed=None):
    """Additive white Gaussian noise (AWGN)."""
    std_table = [5, 10, 18, 28, 40]
    std = std_table[severity - 1] / 255.0
    rng = _rng(seed)
    x = _to_float(img)
    noise = rng.normal(0, std, size=x.shape)
    return _to_uint8(x + noise)


def speckle_noise(img, severity=3, seed=None):
    """Multiplicative (speckle) noise: x' = x + x * n, n ~ N(0, sigma^2)."""
    sigma_table = [0.05, 0.10, 0.18, 0.28, 0.40]
    sigma = sigma_table[severity - 1]
    rng = _rng(seed)
    x = _to_float(img)
    noise = rng.normal(0, sigma, size=x.shape)
    return _to_uint8(x + x * noise)


def salt_and_pepper(img, severity=3, seed=None, salt_vs_pepper=0.5):
    """Impulse noise: a fraction of pixels are driven to 0 or 255."""
    amount_table = [0.005, 0.01, 0.02, 0.04, 0.08]
    amount = amount_table[severity - 1]
    rng = _rng(seed)
    out = img.copy()
    h, w = img.shape[:2]
    n_pixels = int(amount * h * w)
    ys = rng.integers(0, h, size=n_pixels)
    xs = rng.integers(0, w, size=n_pixels)
    is_salt = rng.random(n_pixels) < salt_vs_pepper
    if out.ndim == 3:
        out[ys[is_salt], xs[is_salt], :] = 255
        out[ys[~is_salt], xs[~is_salt], :] = 0
    else:
        out[ys[is_salt], xs[is_salt]] = 255
        out[ys[~is_salt], xs[~is_salt]] = 0
    return out


def gaussian_blur(img, severity=3, seed=None):
    """Isotropic Gaussian blur."""
    sigma_table = [0.5, 1.0, 2.0, 3.5, 5.5]
    sigma = sigma_table[severity - 1]
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def uneven_illumination(img, severity=3, seed=None):
    """Smooth, low-frequency multiplicative illumination field (simulates
    uneven sun / thermal-background lighting across the frame)."""
    strength_table = [0.10, 0.20, 0.35, 0.50, 0.70]
    strength = strength_table[severity - 1]
    rng = _rng(seed)
    h, w = img.shape[:2]
    low_h, low_w = max(2, h // 32), max(2, w // 32)
    field = rng.normal(0, 1, size=(low_h, low_w)).astype(np.float64)
    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
    field = field / (np.abs(field).max() + 1e-8)
    illum = 1.0 + strength * field
    x = _to_float(img)
    if x.ndim == 3:
        illum = illum[:, :, None]
    return _to_uint8(x * illum)


def motion_blur(img, severity=3, seed=None):
    """Linear directional motion blur at a random angle (simulates
    fast-moving small targets / rolling-shutter smear)."""
    length_table = [3, 7, 11, 17, 25]
    length = length_table[severity - 1]
    rng = _rng(seed)
    angle = rng.uniform(0, 180)
    kernel = np.zeros((length, length), dtype=np.float64)
    kernel[length // 2, :] = 1.0
    rot_mat = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (length, length))
    kernel /= kernel.sum() + 1e-8
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT101)


def camera_instability(img, severity=3, seed=None):
    """Random small rigid jitter (translation + rotation + slight scale),
    simulating an unstable gimbal/mount. Border-reflected to avoid black
    edges."""
    shift_table = [1, 3, 6, 10, 15]
    rot_table = [0.5, 1.5, 3.0, 5.0, 8.0]
    max_shift = shift_table[severity - 1]
    max_rot = rot_table[severity - 1]
    rng = _rng(seed)
    h, w = img.shape[:2]
    dx = rng.uniform(-max_shift, max_shift)
    dy = rng.uniform(-max_shift, max_shift)
    angle = rng.uniform(-max_rot, max_rot)
    scale = 1.0 + rng.uniform(-0.01, 0.01) * severity
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, scale)
    mat[0, 2] += dx
    mat[1, 2] += dy
    return cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REFLECT101)


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "speckle_noise": speckle_noise,
    "salt_and_pepper": salt_and_pepper,
    "gaussian_blur": gaussian_blur,
    "uneven_illumination": uneven_illumination,
    "motion_blur": motion_blur,
    "camera_instability": camera_instability,
}

# Nominal severities used for the 30 originally-distorted videos in the 2025
# VIP Cup release, per the challenge dataset-description table. Use these to
# reproduce the *original* fixed-severity distortion set; use the full 1-5
# sweep in CORRUPTIONS for the extended AVID-C robustness benchmark.
PROPOSAL_DEFAULT_SEVERITIES = {
    "speckle_noise": 2,
    "salt_and_pepper": 2,
    "gaussian_blur": 4,
    "uneven_illumination": 4,
    "motion_blur": 4,
    "camera_instability": 5,
    "gaussian_noise": 3,
}

_MODALITY_SEED_OFFSET = {"rgb": 0, "ir": 1}


def apply_corruption(img, name, severity=3, seed=None):
    if name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption '{name}'. Choices: {sorted(CORRUPTIONS)}")
    if not (1 <= severity <= 5):
        raise ValueError("severity must be an integer in [1, 5]")
    return CORRUPTIONS[name](img, severity=severity, seed=seed)


def generate_benchmark_suite(rgb, ir, out_dir, modalities=("rgb", "ir"), seed=0):
    """Write the full 7 x 5 = 35 corrupted variants for an RGB-IR pair to
    out_dir, named '<corruption>_s<severity>_<modality>.png'. `modalities`
    controls which of the pair get corrupted independently, so callers can
    also study asymmetric corruption (e.g. fog only degrades RGB, thermal
    background drift only degrades IR).

    Returns a list of (path, corruption_name, severity, modality) tuples.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    sources = {}
    if "rgb" in modalities:
        sources["rgb"] = rgb
    if "ir" in modalities:
        sources["ir"] = ir

    for c_idx, (name, fn) in enumerate(CORRUPTIONS.items()):
        for severity in range(1, 6):
            for modality, src in sources.items():
                s = seed + c_idx * 100 + severity * 10 + _MODALITY_SEED_OFFSET[modality]
                out = fn(src, severity=severity, seed=s)
                fname = f"{name}_s{severity}_{modality}.png"
                path = os.path.join(out_dir, fname)
                to_write = out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path, to_write)
                manifest.append((path, name, severity, modality))
    return manifest


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Apply a single AVID-C corruption to an image.")
    parser.add_argument("input", help="Path to input image (RGB or single-channel IR).")
    parser.add_argument("output", help="Path to write the corrupted output image.")
    parser.add_argument("--corruption", required=True, choices=sorted(CORRUPTIONS))
    parser.add_argument("--severity", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(args.input)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = apply_corruption(img, args.corruption, severity=args.severity, seed=args.seed)
    to_write = out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, to_write)
    print(f"Wrote {args.output} ({args.corruption}, severity={args.severity})")


if __name__ == "__main__":
    _cli()
