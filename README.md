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

## Why some things here are "prepared" rather than "done"

This was built in a sandboxed environment with no GPU and no way to pull
the multi-gigabyte source zips out of Google Drive (doing so through the
available connector would have cost millions of tokens per file — see
`PLAN.md` §1). Two consequences, stated plainly rather than hidden:

- **Everything that only needs NumPy/OpenCV-level logic is written,
  unit-tested, and passing right now**: the corruption suite, the
  alignment estimators, the weighted-boxes-fusion combiner, the tracking
  metrics, and the split-leakage ablation. Run any of their `test_*.py`
  files directly to see them pass.
- **Everything that needs the real corpus and/or a GPU (the detector/fusion
  training code, and every baseline number in the paper) is written and
  reviewed carefully, but not yet run.** The paper says this explicitly
  (see its Table 4) rather than presenting placeholder numbers as results.
  `docs/RELEASE_RUNBOOK.md` is the checklist for closing that gap.

## Authors

Dharini Raghavan, Vishnu Rao. Built on the dataset construction effort of
the full IEEE ICIP 2025 VIP Cup organizing team (see the paper's
acknowledgements and `dataset_tools/hf_release/DATASET_CARD.md`).

## License

Code in this repo: no license file has been added yet — add one matching
your intent before making the repo public (MIT/Apache-2.0 are the common
defaults for benchmark tooling like this). The dataset itself is intended
for release under **CC-BY-NC-4.0** (see `dataset_tools/hf_release/DATASET_CARD.md`).
