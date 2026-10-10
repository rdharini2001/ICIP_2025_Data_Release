# Reproducibility map

The paper is designed so that every quantitative figure can be traced to a small table.

| Paper component | Primary files |
|---|---|
| Corpus counts | `reproducibility_data/dataset_corpus_stats.csv` |
| Fine object-scale histogram | `reproducibility_data/bbox_scale_histogram.csv` |
| Real 300-pair alignment sample | `reproducibility_data/real_alignment_sample_300pairs.csv` |
| Object-relative alignment ratios | `reproducibility_data/alignment_relative_ratio_300pairs.csv` |
| Alignment burden curve | `reproducibility_data/alignment_burden_curve.csv` |
| Alignment burden summary | `reproducibility_data/alignment_burden_summary.csv` |
| Scale-conditioned overlap bounds | `reproducibility_data/scale_sweep.csv` |
| Chernoff/KDE cross-check | `reproducibility_data/bayes_bound_tightness_and_kde_crosscheck.csv` |
| Representation sensitivity | `reproducibility_data/representation_sensitivity_12_16.csv` |
| Dimension controls | `reproducibility_data/dimension_control_12_16.csv` |
| AVID-C response sweep | `reproducibility_data/corruption_sweep.csv` |
| Clean noise-floor check | `reproducibility_data/noise_floor_check.csv` |
| Controlled shift sweep | `reproducibility_data/alignment_bfg_sweep.csv` |
| Normalized shift sensitivity | `reproducibility_data/alignment_sensitivity_normalized.csv` |

Run `python analysis/make_paper_artifacts.py` from any location to regenerate the plotted PDF/PNG figures. The script resolves package paths relative to itself.
