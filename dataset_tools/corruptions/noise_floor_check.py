"""
Validates a methodological question about AVID-C (see paper Sec. "AVID-C: A
Reproducible Corruption Benchmark"): is the "clean" baseline the corruption
severities are injected on top of actually low-noise, or could real,
already-present sensor/compression noise be large enough to swamp the
smallest injected severity? Every real capture pipeline has non-zero noise
-- there is no such thing as a theoretically noise-free surveillance
frame -- so this checks a concrete, falsifiable claim: is the corpus's
real intrinsic noise floor small relative to AVID-C's smallest severity,
or not?

Uses the classical fast no-reference noise-sigma estimator (Immerkaer
1996): convolve with a discrete Laplacian, and estimate sigma from the
expected response of that operator to pure Gaussian noise. Validated on
synthetic data with KNOWN injected noise before trusting it on real
images (this file's self-tests), same standard as every other estimator
in this project.
"""
import glob
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.corruptions.corruptions import gaussian_noise

RAW_ROOT = r"E:\ICIP_2025\raw_extracted\releasev1\releasev1-detection&tracking\Train"
LAPLACIAN_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)

# AVID-C's own severity table (dataset_tools/corruptions/corruptions.py),
# std in [0,255] units, for direct comparison against the measured noise floor
AVID_C_GAUSSIAN_SEVERITY_STD = [5, 10, 18, 28, 40]


def immerkaer_noise_sigma(img_gray):
    """Fast no-reference noise-sigma estimate (Immerkaer 1996), in [0,255]
    units. img_gray: 2D uint8 or float array."""
    img_gray = img_gray.astype(np.float64)
    H, W = img_gray.shape
    conv = cv2.filter2D(img_gray, -1, LAPLACIAN_KERNEL)
    return float(np.sqrt(np.pi / 2) * np.sum(np.abs(conv)) / (6 * (W - 2) * (H - 2)))


def _list_clean_and_noisy_variants(modality="RGB", n_pairs=200, seed=1):
    """Returns (clean_paths, noisy_paths) for frames with exactly the 2
    Roboflow file variants described in the paper's "pitfalls" section
    (smaller-filesize = clean, larger = the noise-augmented duplicate)."""
    img_dir = os.path.join(RAW_ROOT, modality, "images")
    groups = {}
    for fname in os.listdir(img_dir):
        if not fname.endswith(".jpg"):
            continue
        key = fname.split("_png.rf.")[0]
        groups.setdefault(key, []).append(fname)
    pairs = [(k, v) for k, v in groups.items() if len(v) == 2]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]

    clean_paths, noisy_paths = [], []
    for key, files in pairs:
        full_paths = [os.path.join(img_dir, f) for f in files]
        sized = sorted(full_paths, key=os.path.getsize)
        clean_paths.append(sized[0])
        noisy_paths.append(sized[1])
    return clean_paths, noisy_paths


def measure_real_noise_floor(n_pairs=200, seed=1):
    results = {}
    for modality in ["RGB", "IR"]:
        clean_paths, noisy_paths = _list_clean_and_noisy_variants(modality, n_pairs, seed)
        clean_full, clean_sky, noisy_full = [], [], []
        for cp, npth in zip(clean_paths, noisy_paths):
            c_img = cv2.imread(cp, cv2.IMREAD_GRAYSCALE)
            n_img = cv2.imread(npth, cv2.IMREAD_GRAYSCALE)
            if c_img is None or n_img is None:
                continue
            h, _ = c_img.shape
            clean_full.append(immerkaer_noise_sigma(c_img))
            clean_sky.append(immerkaer_noise_sigma(c_img[: h // 3, :]))
            noisy_full.append(immerkaer_noise_sigma(n_img))
        results[modality] = {
            "n": len(clean_full),
            "clean_full_frame_mean": float(np.mean(clean_full)),
            "clean_full_frame_median": float(np.median(clean_full)),
            "clean_sky_region_mean": float(np.mean(clean_sky)),
            "clean_sky_region_median": float(np.median(clean_sky)),
            "excluded_noisy_variant_full_frame_mean": float(np.mean(noisy_full)),
            "excluded_noisy_variant_full_frame_median": float(np.median(noisy_full)),
        }
    return results


# ---------------------------------------------------------------------------
# Self-tests on synthetic data with KNOWN injected noise, so the real-data
# numbers above can be trusted once these pass.
# ---------------------------------------------------------------------------


def test_zero_on_smooth_image():
    smooth = np.tile(np.linspace(0, 255, 100), (100, 1))
    sigma = immerkaer_noise_sigma(smooth)
    assert sigma < 0.5, sigma  # a perfectly smooth gradient has ~zero Laplacian response


def test_recovers_known_injected_sigma():
    rng = np.random.default_rng(0)
    base = rng.integers(50, 200, size=(128, 128)).astype(np.uint8)
    # smooth the base so its own texture doesn't dominate the estimate
    base = cv2.GaussianBlur(base, (9, 9), 3)
    for true_sigma in [5, 15, 30]:
        noisy = np.clip(base.astype(np.float64) + rng.normal(0, true_sigma, base.shape), 0, 255)
        est_sigma = immerkaer_noise_sigma(noisy)
        # Immerkaer's estimator is a fast approximation, not exact -- check
        # it's in the right ballpark (within 40% relative) and monotonic
        assert abs(est_sigma - true_sigma) / true_sigma < 0.4, (true_sigma, est_sigma)


def test_monotonic_in_injected_severity():
    rng = np.random.default_rng(1)
    base = (rng.integers(50, 200, size=(128, 128))).astype(np.uint8)
    base = cv2.GaussianBlur(base, (9, 9), 3)
    sigmas = []
    for severity in range(1, 6):
        corrupted = gaussian_noise(base, severity=severity, seed=severity)
        sigmas.append(immerkaer_noise_sigma(corrupted))
    assert all(sigmas[i] < sigmas[i + 1] for i in range(len(sigmas) - 1)), sigmas


if __name__ == "__main__":
    test_zero_on_smooth_image()
    test_recovers_known_injected_sigma()
    test_monotonic_in_injected_severity()
    print("All noise-floor estimator self-tests passed.")

    import json
    result = measure_real_noise_floor()
    result["avid_c_gaussian_severity_std"] = AVID_C_GAUSSIAN_SEVERITY_STD
    print(json.dumps(result, indent=2))
    with open(r"E:\ICIP_2025\avid_release\metadata\noise_floor_check.json", "w") as f:
        json.dump(result, f, indent=2)
    print("wrote noise_floor_check.json")
