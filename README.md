# AVID: Aerial Visible-Infrared Drone Dataset — release prep

This repo prepares the release of **AVID**, a large-scale, weakly-aligned
RGB-Infrared drone/bird detection and tracking dataset, building on data
originally constructed for the IEEE ICIP 2025 VIP Cup ("Infrared-Visual
Fusion for Enhanced Drone Detection, Tracking and Payload Identification in
Surveillance Videos"). **Payload data and the payload task are excluded
from this release entirely** — everything here is the drone/bird
detection+tracking dataset, a paper about it, and the tooling to publish
both.

Start here, in order:

1. **[`PLAN.md`](PLAN.md)** — what's actually novel about this release,
   the full list of contributions, and the experiment plan (what's done vs.
   what still needs to run on real hardware).
2. **[`paper/main.tex`](paper/main.tex)** (or the compiled
   [`paper/main.pdf`](paper/main.pdf)) — the arXiv-ready dataset paper.
   Compiles clean (`pdflatex && bibtex && pdflatex && pdflatex`), no
   undefined references or citations, as of the last commit.
3. **[`dataset_tools/`](dataset_tools/)** — all the released code:
   - `corruptions/` — AVID-C, the reproducible 7×5 corruption benchmark.
   - `alignment/` — weak-alignment quantification + sensitivity-study harness.
   - `baselines/` — single-modality/fusion detector reference code, late-fusion
     (WBF), tracking metrics, and the split-leakage ablation.
   - `hf_release/` — the Hugging Face dataset card, schema, and build/upload scripts.
4. **[`docs/RELEASE_RUNBOOK.md`](docs/RELEASE_RUNBOOK.md)** — the exact
   step-by-step checklist for turning this prepared repo into a live
   Hugging Face dataset and arXiv preprint.

## Status: what's real vs. what's still reference code

This started in a sandboxed environment with no GPU and no way to pull the
multi-gigabyte source zips out of Google Drive, then moved to a machine
with the real `releasev1` export and a local GPU. As a result:

- **Everything that only needs NumPy/OpenCV-level logic is written,
  unit-tested, and independently confirmed on the real corpus**: the
  corruption suite, the alignment estimators (including a real 300-pair
  alignment sample — see `metadata/real_alignment_sample.json` after
  running the build), the weighted-boxes-fusion combiner, the tracking
  metrics, and the split-leakage ablation (87.5% vs. 0.0% on the real
  32,040-frame/115-video index).
- **Real baselines**: YOLOv8n single-modality (RGB-only, IR-only) trained
  on the real, deduplicated corpus, combined at inference via late fusion
  (WBF) for a real fusion number — see the paper's Results section and
  `PLAN.md` §4 for current status of each experiment.
- **Real findings from release preparation**: the source export had a
  100%-of-frames noise-augmentation duplicate we identified and removed;
  the true class-index mapping is `0=bird, 1=drone`, not the reverse; the
  Test split's RGB and IR files cannot currently be paired by any
  available key; and the corpus has an extreme small-object distribution
  (67.6% of boxes are 8-16px). All of this is written up in `PLAN.md` §0
  and in the paper.
- **Still reference code, not yet trained/run**: early- and mid-level
  fusion architectures, the tracking pipeline, the corruption-robustness
  sweep, and the alignment-sensitivity study. `docs/RELEASE_RUNBOOK.md`
  and `PLAN.md` §4 are the checklist for closing that gap.

## Authors

Dharini Raghavan, Vishnu Rao. Built on the dataset construction effort of
the full IEEE ICIP 2025 VIP Cup organizing team (see the paper's
acknowledgements and `dataset_tools/hf_release/DATASET_CARD.md`).

## License

Code in this repo: no license file has been added yet — add one matching
your intent before making the repo public (MIT/Apache-2.0 are the common
defaults for benchmark tooling like this). The dataset itself is intended
for release under **CC-BY-NC-4.0** (see `dataset_tools/hf_release/DATASET_CARD.md`).
