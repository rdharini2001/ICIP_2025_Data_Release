# AVID: Aerial Visible-Infrared Drone Dataset

This repo prepares the release of **AVID**, a large-scale, weakly-aligned
RGB-Infrared drone/bird detection and tracking dataset, and a
classifier-agnostic Bayes-risk benchmark built on top of it. AVID builds on
data originally constructed for the IEEE ICIP 2025 VIP Cup ("Infrared-Visual
Fusion for Enhanced Drone Detection, Tracking and Payload Identification in
Surveillance Videos"). **Payload data and the payload task are excluded from
this release entirely** — everything here is the drone/bird
detection+tracking dataset, the paper about it, and the tooling to publish
both.

No detector is trained anywhere in this release. Every number reported in
the paper is a property of the data distribution itself (closed-form
Bhattacharyya/Chernoff bounds, a nonparametric KDE cross-check, mutual
information), computed directly on real patches.

Start here, in order:

1. **[`paper/main.tex`](paper/main.tex)** (or the compiled
   [`paper/main.pdf`](paper/main.pdf)) — the arXiv-ready paper. Compiles
   clean (`pdflatex && bibtex && pdflatex && pdflatex`), no undefined
   references/citations, as of the last commit.
2. **[`reproducibility_data/`](reproducibility_data/)** — a plain-CSV
   export of every number in the paper, with a README mapping each file to
   the table/figure it reproduces. Regenerate all of them with
   `python dataset_tools/theory/export_reproducibility_csvs.py`.
3. **[`dataset_tools/`](dataset_tools/)** — the released code:
   - `theory/` — the Bayes-risk benchmark itself: patch extraction, the
     scale/corruption/alignment sweeps, the Chernoff/KDE tightness
     cross-checks, and the CSV exporter.
   - `corruptions/` — AVID-C, the reproducible 7×5 corruption benchmark,
     plus the real-data noise-floor validation for its "clean" baseline.
   - `alignment/` — the per-pair weak-alignment estimators (phase
     correlation + mutual-information registration search).
   - `hf_release/` — corpus statistics, the object-relative misalignment
     ratio, the split-leakage ablation, and the Hugging Face dataset
     card/schema/build/upload scripts.
   - `baselines/` — a single real-data script (`split_leakage_ablation.py`)
     providing the synthetic cross-check used alongside the real
     split-leakage measurement; kept separate because `hf_release/`'s real
     leakage script imports it directly.
4. **[`docs/RELEASE_RUNBOOK.md`](docs/RELEASE_RUNBOOK.md)** — the
   step-by-step checklist for turning this prepared repo into a live
   Hugging Face dataset and arXiv preprint.
5. **[`archive/`](archive/)** — historical planning notes (`PLAN.md`,
   the full session-by-session record of how this release's scope
   evolved) kept for provenance, not needed to reproduce the paper.

## What's real

Every number in the paper is computed directly on the real, deduplicated
corpus (28,790 train / 3,250 validation RGB-IR pairs, 35,952 annotated
instances) — there is no untrained-placeholder or synthetic-data result
presented as final anywhere in the current paper. Real findings from
release preparation, all disclosed in the paper rather than papered over:
the source export had a 100%-of-frames noise-augmentation duplicate we
identified and removed; the true class-index mapping is `0=bird,
1=drone`, not the reverse; the Test split's RGB and IR files cannot
currently be paired by any available key; the corpus has an extreme
small-object distribution (68.1% of boxes are 8-16px); and naive
frame-level splitting leaks 87.5% of nominally held-out frames (vs. 0.0%
for the video-level splitting this release uses).

## Authors

Dharini Raghavan, Vishnu Rao. Built on the dataset construction effort of
the full IEEE ICIP 2025 VIP Cup organizing team (see the paper's
acknowledgements and `dataset_tools/hf_release/DATASET_CARD.md`).

## License

Code in this repo: no license file has been added yet — add one matching
your intent before making the repo public (MIT/Apache-2.0 are the common
defaults for benchmark tooling like this). The dataset itself is intended
for release under **CC-BY-NC-4.0** (see `dataset_tools/hf_release/DATASET_CARD.md`).
