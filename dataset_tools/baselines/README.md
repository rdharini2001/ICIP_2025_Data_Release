# Baseline reference code

Implements Experiments E2-E4 and E8 from `PLAN.md`.

| File | What it does | Tested in this repo's CI-less sandbox? |
|---|---|---|
| `dataset.py` | `AVIDPairDataset`, loads RGB/IR/labels per the schema in `dataset_tools/hf_release/SCHEMA.md`, four fusion modes | No — needs `torch` + real data. Syntax-checked; run `--smoke-test` on real data first. |
| `fusion_models.py` | `SingleModalityDetector` (RGB-only / IR-only / early-fusion), `MidFusionDetector` (CBAM-style dual-stream fusion), `patch_yolo_early_fusion()` recipe for Ultralytics YOLO | No — needs `torch`. Run `python -m dataset_tools.baselines.fusion_models --self-test` first; it does a forward+backward pass on random tensors of the real input shape. |
| `train.py` | Reference training loop (center-cell target assignment, obj+box+class loss) for all non-late-fusion variants | No — needs `torch` + real data. Run `python -m dataset_tools.baselines.train --self-test` first (trains on one synthetic sample, checks loss goes down). |
| `late_fusion_wbf.py` | Weighted Boxes Fusion for combining independently-trained RGB/IR detectors at inference | **Yes** — pure NumPy, `test_late_fusion_wbf.py` passes with no dependencies beyond numpy. |
| `track_eval.py` | Approach/recede classification, missed-track-<15-frames rule, MOTA-lite, from the original proposal's evaluation criteria | **Yes** — pure NumPy, `test_track_eval.py` passes. |
| `split_leakage_ablation.py` | Frame-level vs. video-level split leakage measurement (Experiment E8) | **Yes** — `python -m dataset_tools.baselines.split_leakage_ablation` runs on a synthetic corpus shaped like the real one (50 videos x 300 frames) and shows ~88% train-frame leakage under naive frame-level splitting vs. 0% for video-level splitting. Re-run on the real video index once available. |

## Why some of this is untested here

This was built in a sandboxed session with no GPU and no way to pull the
multi-GB source zips from Drive (would cost millions of tokens through the
available connector). Everything that only needs NumPy/OpenCV logic (WBF,
tracking metrics, split-leakage, the corruption suite, the alignment
estimators) is written, tested, and passing in this repo. Everything that
needs `torch` and the real corpus (the detector/fusion models and the
training loop) is written and reviewed carefully, but has to be validated by
whoever runs it next, on a machine with both. Each such file has a
`--self-test` or `--smoke-test` entry point specifically so that validation
takes one command, not a debugging session — run those before trusting any
numbers that come out of a real training run.

## Recommended order of operations once you have the real data

1. `python -m dataset_tools.baselines.dataset --root <path> --split val --fusion-mode dual --smoke-test`
2. `python -m dataset_tools.baselines.fusion_models --self-test`
3. `python -m dataset_tools.baselines.train --self-test`
4. `python -m dataset_tools.baselines.train --root <path> --variant rgb` (repeat for `ir`, `early`, `mid_fusion`)
5. Combine the `rgb`/`ir` checkpoints at inference with `late_fusion_wbf.py` for the late-fusion baseline.
6. Feed any trained model into `dataset_tools/corruptions/run_corruption_eval.py` and
   `dataset_tools/alignment/sensitivity_study.py` for the robustness numbers.
