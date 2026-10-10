# AVID arXiv paper package

This directory is the submission-ready source package for:

**AVID: A Weakly Aligned RGB--Infrared Dataset and Perturbation Benchmark for Tiny Aerial Targets**

The manuscript is intentionally organized as a dataset and benchmark paper. It does not use task-training results to establish the contribution. The paper focuses on the paired corpus, target scale, weak cross-modal alignment, the AVID-C perturbation suite, and reproducible data-intrinsic diagnostics.

## Build

Standard TeX Live / Overleaf:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

`main.bbl` is included so arXiv can also compile the package without rerunning BibTeX.

## Main files

- `main.tex` - complete manuscript and appendices
- `references.bib` - bibliography
- `main.bbl` - generated bibliography
- `main.pdf` - verified compiled paper
- `figures/` - vector main plots, raster inspection copies, representative real pairs, and full-frame AVID-C grids
- `reproducibility_data/` - machine-readable values behind the reported quantitative results
- `metadata/` - release metadata snapshots used by the paper
- `analysis/make_paper_artifacts.py` - regenerates the plotted paper figures from the included tables and prepared raster sources
- `analysis/avid_c.py` - exact seven-family, five-severity AVID-C transform implementation

## Alignment quantities

The package uses the all-300-pair Train+Validation join for the scale-aware alignment analysis. The central definitions are:

- object-relative alignment ratio: `rho = d_align / s_target`
- alignment burden curve: `A(tau) = Pr(rho > tau)`
- truncated burden score: `B_T = E[min(rho,T)] / T`

For `T=1`, the released sample gives `B_1 = 0.303575...`.

## Important interpretation notes

- The NMI translation search is limited to +/-12 pixels per axis. The upper displacement tail is therefore search-censored.
- The separability analysis is conditioned on the released low-dimensional representation. The package includes dimension controls and does not present the fusion-gap magnitude as a representation-independent Bayes risk.
- The source Test RGB and infrared exports are not included in the paired benchmark because authoritative correspondence is unresolved.
- The full annotation/adjudication workflow is not documented in the source materials available for this release, so the paper does not invent it.
- The full image corpus is not embedded in the arXiv source archive because of size. Release tooling and metadata are maintained at `https://github.com/rdharini2001/ICIP_2025_Data_Release`.

## Figure style

The quantitative figures were regenerated with the visual language requested for the paper: a white background, dark navy primary series, medium blue and muted cyan secondary series, warm gold for infrared comparisons, green for scale-aware alignment summaries, sparse red annotation boxes, lowercase panel labels, and compact multi-panel composition.
