# Reproducibility data

Every CSV in this folder is a direct re-serialization of a JSON file produced
by a real-data script in this repository (no numbers are recomputed or
altered here). Together they cover every table, figure, and in-text number
reported in `paper/main.tex`. Regenerate all of them at once with:

```
python dataset_tools/theory/export_reproducibility_csvs.py
```

which reads from the JSON files under `E:\ICIP_2025\avid_release\metadata\`
(produced by the scripts named below) and writes the CSVs here.

| CSV file | Produced by | Paper location |
|---|---|---|
| `scale_sweep.csv` | `dataset_tools/theory/scale_sweep.py` | Table "Full scale-sweep results" (`tab:bayes-scale-full`), Figure "theory-scale" (Sec. 6.3) |
| `corruption_sweep.csv` | `dataset_tools/theory/corruption_sweep.py` | Table "Full corruption-sweep summary" (`tab:bayes-corruption-full`), Figure "theory-corruption" and the perturbation montage (Sec. 6.4) |
| `alignment_bfg_sweep.csv` | `dataset_tools/theory/alignment_sweep.py` | Table "Full Bayes Fusion Gain / DPI sweep" (`tab:bfg-full`), Figure "theory-alignment" (Sec. 6.6) |
| `real_measured_shift_distribution_summary.csv` | `dataset_tools/theory/alignment_sweep.py` (from `real_alignment_sample.json`) | The "32.3% of real pairs" shaded region marked on Figure "theory-alignment" |
| `bayes_bound_tightness_and_kde_crosscheck.csv` | `dataset_tools/theory/real_cross_checks.py` | Table "Real-data validation of the Bayes-risk estimator" (`tab:bayes-tightness`), Sec. 6.2 |
| `dataset_corpus_stats.csv` | `dataset_tools/hf_release/corpus_stats.py` | Table "AVID detection/tracking scale" (`tab:dataset-stats`), Sec. 3.2, and the corpus-size numbers in the abstract/intro |
| `bbox_scale_histogram.csv` | `dataset_tools/hf_release/bbox_histogram.py` | Figure "bbox-histogram" (Sec. 3.4), and the "67.6% of boxes are 8-16px" claim |
| `real_alignment_sample_300pairs.csv` | `dataset_tools/hf_release/real_alignment_sample.py` | Sec. 5 (per-pair alignment quantification): the raw 300-pair sample underlying the "32.3% exceed 5px" claim, and the direct input to `alignment_sweep.py`'s real shift distribution |
| `real_alignment_sample_summary.csv` | same as above | Summary statistics (mean/median/stdev/fractions) quoted in Sec. 5 |
| `split_leakage_ablation.csv` | `dataset_tools/hf_release/real_split_leakage.py` | Figure "leakage" (Sec. 3.6): the 87.5% (frame-level) vs. 0.0% (video-level) leakage comparison |

## Notes on reproducing from scratch

- `scale_sweep.csv`, `corruption_sweep.csv`, `alignment_bfg_sweep.csv`, and
  `bayes_bound_tightness_and_kde_crosscheck.csv` are all computed from the
  same 35,952-patch extraction (`dataset_tools/theory/patch_extraction.py`,
  writing `theory_patches.pkl`, ~170MB, not included in this folder or the
  repo due to size — regenerate it by running `patch_extraction.py` against
  a local copy of the AVID dataset). All four scripts share the same PCA
  basis methodology (fit once on the full pooled corpus, reused everywhere)
  so that numbers are directly comparable across files — see
  `scale_sweep.build_feature_space()`.
- `bayes_bounds.py`'s own self-tests (synthetic data with known separation)
  are not re-exported here since they aren't numbers reported in the paper
  — they're the correctness checks that make the real-data numbers above
  trustworthy. Run `python dataset_tools/theory/bayes_bounds.py` to
  reproduce them (6 tests, all pass).
- `dataset_corpus_stats.csv`, `bbox_scale_histogram.csv`,
  `real_alignment_sample_300pairs.csv`, and `split_leakage_ablation.csv` are
  computed directly from the released dataset's frame/video index and do
  not depend on `theory_patches.pkl`.
- No detector is trained anywhere in the scripts that produced any file in
  this folder.
