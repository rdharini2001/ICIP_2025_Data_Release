# Real result evidence

Raw output files from the real experiments reported in `paper/main.tex`,
kept here for reproducibility/auditing rather than only quoted as numbers
in the paper text.

| File | Produced by | Reported in |
|---|---|---|
| `corpus_summary.json` | `dataset_tools/hf_release/fast_index.py` | Table 2 (dataset scale) |
| `bbox_fine_histogram.json` | `dataset_tools/hf_release/bbox_histogram.py` | §"An extreme small-object distribution", Figure (bbox histogram) |
| `split_leakage_real.json` | `dataset_tools/hf_release/real_split_leakage.py` | §"Splits" (87.5% vs 0.0%) |
| `real_alignment_sample.json` | `dataset_tools/hf_release/real_alignment_sample.py` | §"Per-pair alignment quantification" (300-pair sample) |
| `rgb_training_results.csv`, `ir_training_results.csv` | Ultralytics YOLOv8n training (`yolo.train(...)`, see `docs/RELEASE_RUNBOOK.md` §5 for the exact command) | §"Real baseline results" (per-epoch curves; `best.pt` selection) |
| `final_val_metrics.json` | `model.val()` on both checkpoints | Table 3, official-Ultralytics columns |
| `late_fusion_map50.json` | `dataset_tools/hf_release/late_fusion_map_eval.py` | Table 3, matched-methodology mAP50 column |
| `ars_late_fusion.json` | `dataset_tools/alignment/real_ars_late_fusion.py` | Table 4 (ARS results), late-fusion row |

The RGB-only ARS reference row (Table 4) was produced by
`dataset_tools/alignment/real_ars_yolo.py`; its raw output wasn't saved to
a file in this pass (printed to stdout only) -- rerunning it against
`runs/detect/runs/rgb_only_v4/weights/best.pt` reproduces it in under a
minute.

Trained model checkpoints (`best.pt`/`last.pt` for both runs) are not
included here (binary, several MB each, and not needed to verify the
reported numbers) -- they live at
`E:\ICIP_2025\avid_release\runs\detect\runs\{rgb_only_v4,ir_only_v1}\weights\`
on the machine this was trained on.
