# Release runbook

Status as of 2026-10-10: the paper package (`paper/`) is the finished,
submission-ready version — 24 pages, compiles clean (zero undefined
refs/citations, zero overfull/underfull warnings), `main.bbl` included.
This runbook covers what's left: materializing the real dataset into the
Hugging Face upload layout, pushing it, and submitting to arXiv.

## 0. Before you start

- Rename `AVID` if you don't keep it (appears in `archive/PLAN.md`,
  `paper/main.tex`, and `dataset_tools/hf_release/README_HF.md`).
- Decide on the arXiv category (`cs.CV`, possibly cross-listed `eess.IV`).
- Hugging Face: `hf auth login` once; verify with `hf auth whoami`.

## 1. Materialize the real dataset into HF layout

`avid_release/` previously held only metadata (JSONL indexes, the theory
patch pickle, sweep results) — the real images had never been copied into
the clean `images/rgb/{split}/{video}/…` layout. Build it:

```bash
python -m dataset_tools.hf_release.build_avid_v1 --alignment-sample-rate 0.0
```

`--alignment-sample-rate 0.0` skips the per-pair alignment estimator during
the bulk copy (it costs ~1.1s/pair; at full rate this is ~9.8 CPU-hours
across all 32,040 pairs) — the real 300-pair alignment sample already lives
in `metadata/real_alignment_sample.json` and does not need regenerating
here. This writes `avid_release/images/`, `avid_release/labels/`, and
refreshes `avid_release/metadata/{pairs,videos}_{train,val}.jsonl`.

The build script indexes each modality's image directory with
`os.scandir()` rather than `os.listdir()` + a separate `os.path.getsize()`
call per file — the latter is dramatically slower under Windows Defender
real-time scanning at this corpus's scale (tens of thousands of files per
directory); fixed once, in the script itself, not just worked around here.
Even so, copying ~96,000 small files (two JPEGs + one label per pair) is
I/O-bound and takes on the order of an hour or two depending on the
machine — this is a real wait, not a hang; progress prints every 500 pairs.

## 2. Assemble the Hugging Face upload folder

```bash
python -m dataset_tools.hf_release.assemble_hf_folder --out E:\ICIP_2025\avid_hf_release
```

(Or do it by hand: copy `avid_release/{images,labels,metadata}/`,
`dataset_tools/hf_release/README_HF.md` → `README.md`,
`dataset_tools/hf_release/LICENSE`, `paper/main.pdf` → `paper/main.pdf`,
`reproducibility_data/`, and `paper/figures/fig_hf_teaser.png` →
`assets/avid_teaser.png` into one folder.)

Sanity-check before uploading: open `README.md` and confirm the YAML
frontmatter parses (no tab characters, correct list indentation) and that
the teaser image path resolves once pushed.

## 3. Push the dataset to Hugging Face

```bash
hf repo create Dharini24/AVID --repo-type dataset
hf upload Dharini24/AVID E:\ICIP_2025\avid_hf_release . --repo-type dataset
```

or via the Python wrapper (handles repo creation + card copy in one call):

```bash
python -m dataset_tools.hf_release.upload_to_hf \
    --local-dir E:\ICIP_2025\avid_hf_release \
    --repo-id Dharini24/AVID
```

This is an externally-visible, hard-to-fully-reverse action (the repo URL
becomes public and citable) — run it yourself from a machine with your own
token, after reviewing the assembled folder, not unattended.

## 4. Submit to arXiv

1. Finalize author list/affiliations in `paper/main.tex`'s `\author{}`
   block and the citation block in `dataset_tools/hf_release/README_HF.md`.
2. Package and submit:
   ```bash
   cd paper
   tar czf avid_arxiv_submission.tar.gz main.tex references.bib main.bbl figures/
   ```
3. arxiv.org -> Submit -> category `cs.CV` (cross-list `eess.IV` if
   relevant) -> upload -> check the preview PDF against your local compile.
4. Once you have the arXiv ID: update the `Paper` badge/link in both
   `README.md` (GitHub) and `dataset_tools/hf_release/README_HF.md` (HF),
   and the citation block in both places.

## 5. After both are live

- Cross-link: HF card's badges point at the GitHub repo and the paper PDF;
  GitHub README's badges point at the HF dataset and the paper PDF. Both
  already do this as of this commit — just verify the links resolve once
  the HF repo exists.
- If any number in the paper changes after the first posted version, ship
  an arXiv v2 rather than silently editing -- reviewers and citers benefit
  from the version history.
