# Release runbook

Status as of 2026-09-14: steps 1-4 below are **done** (real data extracted,
real corpus built, real stats computed, real split-leakage measured). RGB
and IR YOLOv8n training (step 5) is running on a local GPU as this is
written. This runbook now covers what's left, not a from-scratch plan.

## 0. Before you start

- Rename `AVID` if you don't keep it (appears in `PLAN.md`, `paper/main.tex`
  as `\datasetname{}`/`\corruptname{}` macros, and
  `dataset_tools/hf_release/DATASET_CARD.md`).
- Decide on the arXiv category (`cs.CV`, possibly cross-listed `eess.IV`).
- Create a Hugging Face account/org and a write-scoped access token.

## 1-4. Done — data, schema, stats, leakage

- Real corpus: `E:\ICIP_2025\avid_release\` (built via
  `dataset_tools/hf_release/fast_index.py` against the extracted
  `releasev1` export at `E:\ICIP_2025\raw_extracted\releasev1\...`).
  28,790 train / 3,250 val pairs, deduplicated (see `PLAN.md` §0 for why
  the raw export had 2x that many files).
- Corpus stats: `metadata/corpus_summary.json`,
  `metadata/bbox_fine_histogram.json` (the extreme small-object finding).
- Real alignment sample (300 pairs): `metadata/real_alignment_sample.json`.
  **Not yet done**: extending this to the full 32,040-pair corpus
  (~9.8 CPU-hours at the measured ~1.1s/pair; `real_alignment_sample.py`
  takes an `n` argument, so this is a size bump + a longer wait, not new
  code).
- Real split-leakage on the real video index:
  `metadata/split_leakage_real.json` (87.5% vs 0.0%).
- **Still open**: the Test split's RGB/IR files cannot be paired (see
  `PLAN.md` §0 and `paper/main.tex` §"pitfalls"). Extracting
  `pre_releasev2-detection&tracking.zip` (not yet done, 3.24GB, likely a
  corrected/expanded release) is the natural next step to check whether it
  fixes this before a camera-ready revision.

## 5. Baselines and the Alignment Robustness Score (the paper's spine)

Training (Ultralytics, not the custom `dataset_tools/baselines/train.py`
scaffold from the pre-data-access phase of this project):

```bash
cd E:\ICIP_2025\avid_release
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').train(data='configs/rgb.yaml', epochs=60, imgsz=320, batch=64, device=0, project='runs', name='rgb_only', workers=2, cache='ram', patience=15)"
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').train(data='configs/ir.yaml', epochs=60, imgsz=320, batch=64, device=0, project='runs', name='ir_only', workers=2, cache='ram', patience=15)"
```

`workers=2, cache='ram'` matters: `workers=0` fully serializes CPU
augmentation with GPU compute (0% GPU utilization, confirmed via
`nvidia-smi` during this build); `cache='ram'` avoids re-reading images
from disk every epoch, but Ultralytics silently skips it (with a printed
warning, easy to miss) if free RAM is under its safety margin — check for
that warning in the log before trusting subsequent epochs are actually
using the cache.

Once both checkpoints exist:

```bash
python -m dataset_tools.alignment.real_ars_yolo --model runs/detect/runs/rgb_only/weights/best.pt --n-pairs 100 --out metadata/ars_rgb.json
```

For the late-fusion (WBF) ARS number, write a `predict_fn` that runs both
checkpoints and combines their outputs via
`dataset_tools/baselines/late_fusion_wbf.py::weighted_boxes_fusion`, then
pass it to `dataset_tools.alignment.alignment_robustness_score.compute_ars`
the same way `real_ars_yolo.py`'s `make_rgb_predict_fn` does for the
single-modality case — this is the comparison that actually tests the
paper's hypothesis (\S"Alignment robustness as a first-class metric").

Early/mid fusion (channel-concat / CBAM dual-stream,
`dataset_tools/baselines/fusion_models.py`) are released as reviewed,
self-tested reference code, not trained this pass — training them (they
need a custom loop, not Ultralytics, since they're not standard YOLO) and
running ARS on the result is the single highest-value remaining experiment,
since it's what actually tests the early-vs-late robustness hypothesis
rather than just late fusion in isolation.

Fill in `paper/main.tex` Table `tab:baseline-results` (mAP-based) and Table
`tab:ars-results` (ARS) with whatever's real, and update
`tab:experiment-status` row-by-row as each queued item completes — every
`TODO*` placeholder in `main.tex` marks exactly what's left.

## 6. Corruption sweep, missing-modality, tracking (E5/E6/E4)

```bash
python -m dataset_tools.hf_release.yolo_corruption_eval --model runs/detect/runs/rgb_only/weights/best.pt --n-images 400
```

Missing-modality: same harness, drop or corrupt only one modality (see
`PLAN.md` §4, E6). Tracking (E4): needs a tracker (ByteTrack/BoT-SORT) run
on top of detections from either checkpoint, scored with
`dataset_tools/baselines/track_eval.py` (unit-tested, not yet run on real
tracks).

## 7. Regenerate figures and recompile

```bash
python paper/figures/generate_figures.py
python paper/figures/generate_real_figures.py
python paper/figures/generate_bbox_histogram_figure.py
python paper/figures/generate_real_leakage_figure.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

(No LaTeX toolchain was available on the machine this was drafted on --
verify compilation once you have `pdflatex`/`bibtex`, e.g. via a TeX Live
or MiKTeX install, before submitting.)

## 8. Push the dataset to Hugging Face

```bash
pip install huggingface_hub
huggingface-cli login
python -m dataset_tools.hf_release.upload_to_hf --local-dir E:\ICIP_2025\avid_release --repo-id <your-username-or-org>/avid
```

Update `DATASET_CARD.md`'s citation block and placeholder counts first if
steps 5-6 changed anything since the last edit.

## 9. Submit to arXiv

1. Finalize author list/affiliations in `\maketitle` and the citation block
   in `DATASET_CARD.md`.
2. Package and submit:
   ```bash
   cd paper
   tar czf avid_arxiv_submission.tar.gz main.tex references.bib main.bbl figures/
   ```
3. arxiv.org → Submit → category `cs.CV` (cross-list `eess.IV` if
   relevant) → upload → check the preview PDF against your local compile.
4. Once you have the arXiv ID, update `DATASET_CARD.md`'s citation and
   re-push (step 8); cross-link the HF card and arXiv page to each other.

## 10. After both are live

- Update this repo's top-level `README.md` links and commit.
- If step 5/6 numbers differ meaningfully from whatever was in the version
  you first posted, ship an arXiv v2 rather than silently editing --
  reviewers and citers benefit from the version history.
