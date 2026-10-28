# AVID manuscript: final pre-arXiv source package

**AVID: A Weakly Aligned RGB-Infrared Dataset and Perturbation Benchmark for Tiny Aerial Targets**

Authors: **Dharini Raghavan\***, **Vishnu Rao\***, **Sethu Selvi S**.  \* Equal contributions by the first two authors.

This directory contains the final 28-page manuscript supplied for the release audit, including all figure assets and its 30-entry bibliography. It is a preprint prepared for arXiv submission; no arXiv identifier, published DOI, or acceptance is claimed here.

## Build

Run from `paper/` using a standard TeX Live installation:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

`main.bbl` is also included. The compiled text was checked against the supplied final PDF; all citations and cross-references resolved.

## Reproduce figures and values

The numerical CSVs in `reproducibility_data/` match the files at the repository root. Run `python analysis/make_paper_artifacts.py` from this folder to regenerate the quantitative figures and montages, using the bundled CSVs and prepared example-image sources.

**Important alignment distinction:** the raw estimated shift distribution uses **300** sampled RGB-infrared frame pairs. The object-relative ratio, burden curve, and burden score require matching target annotations and use **264** pairs. Their correct unit-scale burden is **0.2647566**, rounded to **0.265**. An earlier all-300 diagnostic is explicitly archived and is not used in the paper.

Bound values are for a fixed PCA/density representation and are not detector results. The full 32,040-pair image dataset is on Hugging Face, not embedded in the arXiv source package.

See `../VERIFICATION_REPORT.md`, `../reproducibility_data/README.md`, and `../DATA_PROVENANCE_AND_LICENSING.md`.
