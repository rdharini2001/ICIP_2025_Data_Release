# AVID numerical data: authoritative final-paper tables

These CSV files are the numeric inputs behind the final submitted preprint. They describe the *already-computed* measurements and are intended to be checked against the paper, rather than claiming that the images were recomputed during this repository packaging.

**Two different sample sizes must not be mixed:** `real_alignment_sample_300pairs.csv` covers **300** global translation estimates; `object_relative_misalignment_rows.csv` covers the **264** pairs whose targets could be matched. All target-relative alignment percentages and `alignment_burden_{summary,curve}.csv` use the 264 rows. The correct 264-pair unit burden is **0.2647566**, rounded to **0.265**. Historical 300-pair *target-relative* CSVs are archived separately and are not part of the published results.

| Claim | Data file |
|---|---|
| Train and Validation pairs, videos, class instances, COCO sizes | `dataset_corpus_stats.csv` |
| Fine geometric-mean bounding box size distribution | `bbox_scale_histogram.csv` |
| Pixel shift statistics for 300 sampled pairs | `real_alignment_sample_300pairs.csv`, `real_alignment_sample_summary.csv` |
| Target-normalized ratios on 264 matched pairs | `object_relative_misalignment_rows.csv`, `object_relative_misalignment_summary.csv` |
| Burden curve and unit summary on 264 pairs | `alignment_burden_curve.csv`, `alignment_burden_summary.csv` |
| Bhattacharyya bounds across sizes | `scale_sweep.csv` |
| Chernoff optimization and KDE checks | `bayes_bound_tightness_and_kde_crosscheck.csv` |
| PCA dimension comparisons | `representation_sensitivity_12_16.csv`, `dimension_control_12_16.csv` |
| Perturbation curves and severity-5 averages | `corruption_sweep.csv`, `corruption_severity5_summary.csv` |
| Clean-frame noise estimate and AWGN calibration | `noise_floor_check.csv`, `avid_c_severity_std_table.csv` |
| Added-shift sweep and retention | `alignment_bfg_sweep.csv`, `alignment_sensitivity_normalized.csv` |

## Reproduction limits

The included scripts in `dataset_tools/` can be run on a downloaded AVID dataset to regenerate the raw-data analyses. Some scripts require a local configuration of dataset paths and an extracted patch matrix. Historical scripts and paths under `archive/` are not part of the validated workflow. See `VERIFICATION_REPORT.md` for exactly what was checked in this release.

Use `python paper/audit/verify_final_csvs.py` to validate the numerical CSVs without the original images.
