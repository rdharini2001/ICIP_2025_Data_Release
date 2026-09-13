# Release runbook

Everything in this repo that needed the real pixel data, a GPU, or write
access to arXiv/Hugging Face was prepared but not executed, because none of
those were available in the sandbox this was built in (see `PLAN.md` §1 and
`dataset_tools/baselines/README.md` for exactly what was/wasn't testable).
This is the checklist for turning the prepared repo into an actual release.
Run the steps in order; each one names the script it depends on.

## 0. Before you start

- Rename `AVID` to your final dataset name if you don't keep it (it appears
  in `PLAN.md`, `paper/main.tex`, `dataset_tools/hf_release/DATASET_CARD.md`,
  and as `\datasetname{}`/`\corruptname{}` macros in the paper — the LaTeX
  macros mean you only have to edit two lines there).
- Decide on the arXiv category (likely `cs.CV`, possibly cross-listed to
  `eess.IV`) and get an arXiv account if you don't have one.
- Create a Hugging Face account/organization for the dataset repo if you
  don't have one, and generate a write-scoped access token
  (huggingface.co/settings/tokens).

## 1. Get the raw data local and inspect its real layout

Pull the source zips from the `ICIP_2025` Drive folder to a machine with
enough disk (the zips alone total roughly 8 GB: `releasev1`,
`pre-releasev2`, `Test_detection&tracking(_with_labels)`,
`Validation_Videos`, `consolidated_IR_RGB_Pairs`, `detecttrack`). Extract
each and run:

```bash
find <extracted_dir> -maxdepth 4 | head -80
```

**Do not assume they share one layout** — they're from different passes
over the challenge timeline. `dataset_tools/hf_release/build_hf_dataset.py`
ships a `generic_frame_folder_adapter()` for the common
`RGB/<video_id>/<frame>.jpg` + `IR/<video_id>/<frame>.jpg` (+
`labels/<video_id>/<frame>.txt`) shape; if a zip's real layout differs,
write a small adapter function following that same contract (yield
`RawRecord` objects) rather than fighting the existing one into shape.

**Exclude payload data explicitly** — if any raw zip mixes payload frames/
labels in with detection/tracking data, your adapter must not yield them.
This release is drone/bird detection+tracking only.

## 2. Build the HF-schema dataset

For each split/zip:

```bash
python -m dataset_tools.hf_release.build_hf_dataset \
    --raw-root <extracted_dir> --split train --out-root <avid_out_dir>
```

Repeat for `val` and `test` (they'll merge into the same `--out-root`, since
`build_dataset()` appends). This also runs the per-pair alignment estimator
(`dataset_tools/alignment/`) by default — drop `--no-alignment` only for a
quick structural dry run, since `alignment_shift_px`/`alignment_confidence`
are meant to ship as real metadata, not placeholders, in the final release.

Sanity-check the output against `dataset_tools/hf_release/SCHEMA.md` before
moving on — `python -m dataset_tools.hf_release.test_build_hf_dataset` is
the synthetic-data version of the same check and should still pass.

## 3. Regenerate the corpus-statistics table (Experiment E1)

Not yet built (see `PLAN.md` — flagged as pending schema confirmation).
Once step 2 has run, this is a short script over `metadata/pairs.jsonl` /
`metadata/videos.jsonl`: class balance, bounding-box size distribution
(feeds the scale-stratified evaluation protocol), scenario-tag coverage,
and per-split video/frame counts. Use its output to replace the
placeholder numbers in `paper/main.tex` Table 2 (`tab:dataset-stats`).

## 4. Confirm the split-leakage ablation on the real video index

`dataset_tools/baselines/split_leakage_ablation.py` currently reports a
result on a **synthetic** corpus shaped like AVID (50 videos x 300 frames)
— 87.8% vs. 0.0% leakage. Re-run `compare_split_strategies()` against the
real `(video_id, frame_idx)` list from `metadata/pairs.jsonl` and update
`paper/main.tex` §"Splits" (`sec:leakage`) and Figure 2 with the real
number before treating this as a final result.

## 5. Run and validate the baselines (Experiments E2-E4)

In order, per `dataset_tools/baselines/README.md`:

```bash
pip install torch  # not installed in the prep sandbox; see that file for why
python -m dataset_tools.baselines.dataset --root <avid_out_dir> --split val --fusion-mode dual --smoke-test
python -m dataset_tools.baselines.fusion_models --self-test
python -m dataset_tools.baselines.train --self-test
python -m dataset_tools.baselines.train --root <avid_out_dir> --variant rgb
python -m dataset_tools.baselines.train --root <avid_out_dir> --variant ir
python -m dataset_tools.baselines.train --root <avid_out_dir> --variant early
python -m dataset_tools.baselines.train --root <avid_out_dir> --variant mid_fusion
```

For a competitive (not just reference) number, adapt
`patch_yolo_early_fusion()` onto a real Ultralytics YOLO checkpoint instead
of the built-in reference detector. Combine the independently-trained
`rgb`/`ir` checkpoints at inference with `late_fusion_wbf.py` for the
late-fusion baseline. Run tracking with your tracker of choice
(ByteTrack/BoT-SORT) on top of any of the above, then score with
`track_eval.py`.

## 6. Run the robustness and sensitivity sweeps (E5-E7)

```python
from dataset_tools.corruptions.run_corruption_eval import run_corruption_sweep, summarize
from dataset_tools.alignment.sensitivity_study import run_sensitivity_study
# wrap your trained model as predict_fn(rgb, ir) -> prediction and plug in
# a score_fn matching your task metric (mAP, IoU, whatever you're reporting)
```

Feed the results into `paper/main.tex` Table 4 (`tab:experiment-status`),
flipping each row's "Result status" column from "not yet run" once real.

## 7. Regenerate paper figures and recompile

`paper/figures/generate_figures.py` regenerates every figure from code; if
you added a corpus-stats script in step 3, add a real-data figure there too
(e.g. class/scenario distribution) rather than only the synthetic
illustrations currently in the paper. Recompile:

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## 8. Push the dataset to Hugging Face

```bash
pip install huggingface_hub
huggingface-cli login   # paste your write-scoped token
python -m dataset_tools.hf_release.upload_to_hf \
    --local-dir <avid_out_dir> \
    --repo-id <your-username-or-org>/avid
```

This copies `dataset_tools/hf_release/DATASET_CARD.md` to
`<avid_out_dir>/README.md` before pushing (that's what Hugging Face renders
as the dataset card) — update the placeholder counts/citation block in that
card first if steps 3-6 changed them. Add `--private` if you want to review
the repo before making it public; drop it (or flip visibility in the HF UI)
when ready.

## 9. Submit to arXiv

1. Update the citation block in `DATASET_CARD.md` and the `\maketitle`
   author/affiliation block in `paper/main.tex` if anything changed.
2. Package `main.tex`, `references.bib`, `main.bbl` (already generated —
   arXiv's own bibtex pass can be flaky across TeX Live versions, so
   submitting the pre-built `.bbl` is the safer default), and the
   `figures/` directory into a single archive:
   ```bash
   cd paper
   tar czf avid_arxiv_submission.tar.gz main.tex references.bib main.bbl figures/
   ```
3. Go to arxiv.org → Submit → new submission, category `cs.CV` (cross-list
   `eess.IV` if relevant), upload the archive, and let arXiv's compiler
   produce the preview PDF. Check the preview against your local
   `paper/main.pdf` before finalizing — arXiv's TeX Live version can differ
   slightly from whatever compiled it locally.
4. Once you have the arXiv ID, update the citation block in
   `DATASET_CARD.md` (and re-push per step 8) and cross-link the HF dataset
   card and the arXiv abstract page to each other.

## 10. After both are live

- Update this repo's top-level `README.md` links (arXiv ID, HF dataset URL)
  and commit.
- Consider watermarking/versioning: if step 5's real baseline numbers
  differ meaningfully from the "not yet run" placeholders removed from the
  paper, post an arXiv v2 rather than silently editing — reviewers and
  citers benefit from the version history showing when real numbers landed.
