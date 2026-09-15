# Release runbook

Status as of 2026-09-15: the paper is finalized around a classifier-agnostic
Bayes-risk benchmark and the AVID dataset. No detector is trained anywhere
in the paper or in the scripts that produced any number in it. This runbook
covers what's left to actually publish: the arXiv submission and the
Hugging Face dataset upload.

## 0. Before you start

- Rename `AVID` if you don't keep it (appears in `PLAN.md`, `paper/main.tex`
  as `\datasetname{}`/`\corruptname{}` macros, and
  `dataset_tools/hf_release/DATASET_CARD.md`).
- Decide on the arXiv category (`cs.CV`, possibly cross-listed `eess.IV`).
- Create a Hugging Face account/org and a write-scoped access token.

## 1. Data, schema, stats, leakage, theory sweeps — all done

- Real corpus: `E:\ICIP_2025\avid_release\` (built via
  `dataset_tools/hf_release/build_avid_v1.py` from the extracted
  `releasev1` Roboflow export). 28,790 train / 3,250 val pairs,
  deduplicated (see `PLAN.md` for why the raw export had 2x that many
  files).
- Corpus stats: `metadata/corpus_summary.json`,
  `metadata/bbox_fine_histogram.json` (the extreme small-object finding) --
  produced by `dataset_tools/hf_release/corpus_stats.py` and
  `bbox_histogram.py`.
- Real alignment sample (300 pairs): `metadata/real_alignment_sample.json`,
  produced by `dataset_tools/hf_release/real_alignment_sample.py` (which
  uses the estimators in `dataset_tools/alignment/alignment_metrics.py`).
- Real split-leakage on the real video index:
  `metadata/split_leakage_real.json` (87.5% vs 0.0%), produced by
  `dataset_tools/hf_release/real_split_leakage.py`.
- The full Bayes-risk theory pipeline (`dataset_tools/theory/`):
  `patch_extraction.py` extracts 35,952 real patches
  (`metadata/theory_patches.pkl`, ~170MB, not committed to git --
  regenerate locally before rerunning any sweep below); `scale_sweep.py`,
  `corruption_sweep.py`, and `alignment_sweep.py` produce the three real
  results in the paper's Bayes-risk section; `real_cross_checks.py`
  produces the Chernoff/KDE tightness cross-check. Regenerate all of it:
  ```bash
  python -m dataset_tools.theory.patch_extraction
  python -m dataset_tools.theory.scale_sweep
  python -m dataset_tools.theory.corruption_sweep
  python -m dataset_tools.theory.alignment_sweep
  python -m dataset_tools.theory.real_cross_checks
  python dataset_tools/theory/export_reproducibility_csvs.py
  ```
- `reproducibility_data/` holds a plain-CSV export of every number in the
  paper (see its `README.md` for the exact table/figure each file maps to).
- **Still open**: the Test split's RGB/IR files cannot be paired (see
  `PLAN.md` and `paper/main.tex`'s "pitfalls" section, disclosed as a
  limitation rather than worked around). Extracting
  `pre_releasev2-detection&tracking.zip` (not yet done, 3.24GB, likely a
  corrected/expanded release) is the natural next step to check whether it
  fixes this before a camera-ready revision.

## 2. Regenerate figures and recompile

```bash
python paper/figures/generate_figures.py
python paper/figures/generate_theory_figures.py
python paper/figures/generate_perturbation_montage.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Verify zero `Undefined` references/citations and zero `Overfull`/`Underfull`
warnings in the final `pdflatex` log before treating a compile as final.

## 3. Push the dataset to Hugging Face

```bash
pip install huggingface_hub
huggingface-cli login
python -m dataset_tools.hf_release.upload_to_hf --local-dir E:\ICIP_2025\avid_release --repo-id <your-username-or-org>/avid
```

Update `DATASET_CARD.md`'s citation block and placeholder counts first if
anything in the dataset itself has changed since the last edit.

## 4. Submit to arXiv

1. Finalize author list/affiliations in `\maketitle` and the citation block
   in `DATASET_CARD.md`.
2. Package and submit:
   ```bash
   cd paper
   tar czf avid_arxiv_submission.tar.gz main.tex references.bib main.bbl figures/
   ```
3. arxiv.org -> Submit -> category `cs.CV` (cross-list `eess.IV` if
   relevant) -> upload -> check the preview PDF against your local compile.
4. Once you have the arXiv ID, update `DATASET_CARD.md`'s citation and
   re-push (step 3); cross-link the HF card and arXiv page to each other.

## 5. After both are live

- Update this repo's top-level `README.md` links and commit.
- If any number in the paper changes after the first posted version, ship
  an arXiv v2 rather than silently editing -- reviewers and citers benefit
  from the version history.
