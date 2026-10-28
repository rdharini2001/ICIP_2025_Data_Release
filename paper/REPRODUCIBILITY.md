# Paper reproducibility map

`reproducibility_data/` is synchronized with the top-level directory and supports the final paper's tables, figures, and numerical claims.

| Result | Source CSV |
|---|---|
| Dataset composition | `dataset_corpus_stats.csv` |
| Fine target sizes | `bbox_scale_histogram.csv` |
| 300 raw translation estimates | `real_alignment_sample_300pairs.csv` |
| 264 target-matched alignment ratios | `object_relative_misalignment_rows.csv` |
| 264 matched alignment burden | `alignment_burden_curve.csv`, `alignment_burden_summary.csv` |
| Scale-conditioned fitted bounds | `scale_sweep.csv` |
| Optimized Chernoff and KDE | `bayes_bound_tightness_and_kde_crosscheck.csv` |
| PCA dimensions and controls | `dimension_control_12_16.csv`, `representation_sensitivity_12_16.csv` |
| Perturbation response curves | `corruption_sweep.csv`, `corruption_severity5_summary.csv` |
| Original-image noise estimate | `noise_floor_check.csv`, `avid_c_severity_std_table.csv` |
| Controlled added shifts | `alignment_bfg_sweep.csv`, `alignment_sensitivity_normalized.csv` |

The included values can be audited without downloading the images. Recomputing them from the image pixels requires the public AVID data and the scripts under `dataset_tools/` at repository root. The current package does not contain the full image corpus or the precomputed patch matrix.

For the quantitative figures, run `python analysis/make_paper_artifacts.py` with Python, NumPy, pandas, Matplotlib, Pillow, and OpenCV installed.
