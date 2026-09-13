"""
Self-contained correctness tests for AVID-C, using a synthetic checkerboard
+ soft "blob" image (a stand-in for a drone silhouette against sky) so this
runs with no real dataset present. Run with: pytest dataset_tools/corruptions
or: python -m dataset_tools.corruptions.test_corruptions
"""

import numpy as np

from .corruptions import CORRUPTIONS, apply_corruption, generate_benchmark_suite


def _synthetic_rgb(h=256, w=320, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 2] = 180  # sky-blue-ish background
    img[:, :, 1] = 140
    img[:, :, 0] = 90
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx, r = h // 2, w // 2, 8
    blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    img[blob] = [20, 20, 20]
    img = img + rng.integers(-3, 3, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _synthetic_ir(h=256, w=320, seed=1):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 60, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx, r = h // 2, w // 2 + 3, 8  # slight offset -> "weak alignment"
    blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    img[blob] = 230
    img = img.astype(np.int16) + rng.integers(-3, 3, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_shapes_and_dtypes_preserved():
    rgb = _synthetic_rgb()
    ir = _synthetic_ir()
    for name in CORRUPTIONS:
        out_rgb = apply_corruption(rgb, name, severity=3, seed=0)
        assert out_rgb.shape == rgb.shape, name
        assert out_rgb.dtype == np.uint8, name
        out_ir = apply_corruption(ir, name, severity=3, seed=0)
        assert out_ir.shape == ir.shape, name
        assert out_ir.dtype == np.uint8, name


def test_determinism():
    rgb = _synthetic_rgb()
    for name in CORRUPTIONS:
        a = apply_corruption(rgb, name, severity=4, seed=42)
        b = apply_corruption(rgb, name, severity=4, seed=42)
        assert np.array_equal(a, b), f"{name} is not deterministic given a fixed seed"


def test_severity_rejects_out_of_range():
    rgb = _synthetic_rgb()
    for bad in (0, 6, -1):
        try:
            apply_corruption(rgb, "gaussian_noise", severity=bad, seed=0)
            raise AssertionError("expected ValueError for out-of-range severity")
        except ValueError:
            pass


def test_severity_monotonicity_for_noise_and_blur():
    """For corruptions where 'more severe' should mean 'more different from
    the clean image' on average, check that mean absolute pixel difference
    is non-decreasing across severities 1..5 (averaged over a few seeds to
    denoise the comparison)."""
    rgb = _synthetic_rgb()
    monotonic_families = [
        "gaussian_noise",
        "speckle_noise",
        "salt_and_pepper",
        "gaussian_blur",
        "motion_blur",
        "uneven_illumination",
    ]
    for name in monotonic_families:
        diffs = []
        for severity in range(1, 6):
            vals = []
            for seed in range(5):
                out = apply_corruption(rgb, name, severity=severity, seed=seed)
                vals.append(np.mean(np.abs(out.astype(np.float64) - rgb.astype(np.float64))))
            diffs.append(np.mean(vals))
        # allow tiny numerical slack, but the overall trend must be increasing
        assert diffs[-1] > diffs[0], f"{name}: severity 5 not worse than severity 1 ({diffs})"
        non_decreasing_steps = sum(diffs[i + 1] >= diffs[i] - 1e-6 for i in range(4))
        assert non_decreasing_steps >= 3, f"{name}: not monotonic enough ({diffs})"


def test_generate_benchmark_suite(tmp_path=None):
    import shutil
    import tempfile

    rgb = _synthetic_rgb()
    ir = _synthetic_ir()
    out_dir = tempfile.mkdtemp(prefix="avidc_test_")
    try:
        manifest = generate_benchmark_suite(rgb, ir, out_dir)
        assert len(manifest) == 7 * 5 * 2  # 7 corruptions x 5 severities x {rgb, ir}
        for path, _, _, _ in manifest:
            import os

            assert os.path.isfile(path)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    test_shapes_and_dtypes_preserved()
    test_determinism()
    test_severity_rejects_out_of_range()
    test_severity_monotonicity_for_noise_and_blur()
    test_generate_benchmark_suite()
    print("All AVID-C corruption tests passed.")
