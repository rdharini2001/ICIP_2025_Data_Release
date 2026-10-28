<h1 align="center">AVID: A Weakly Aligned RGB-Infrared Dataset and Perturbation Benchmark for Tiny Aerial Targets</h1>

<p align="center"><a href="https://rdharini2001.github.io/ICIP_2025_Data_Release/"><img src="docs/assets/panorama.webp" width="1000" alt="AVID RGB and infrared research website"/></a></p>

<p align="center"><a href="paper/main.pdf"><b>Paper PDF</b></a> · <a href="paper/main.tex"><b>LaTeX source</b></a> · <a href="https://huggingface.co/datasets/Dharini24/AVID"><b>Hugging Face dataset</b></a> · <a href="https://rdharini2001.github.io/ICIP_2025_Data_Release/"><b>Research website</b></a></p>

**Dharini Raghavan\*** (Georgia Institute of Technology), **Vishnu Rao\*** (ARTPARK, Indian Institute of Science), and **Sethu Selvi S** (Ramaiah Institute of Technology).  
\* Dharini Raghavan and Vishnu Rao contributed equally and share first authorship.

**AVID** contains real ground-based surveillance imagery in which birds and drones occupy only a few pixels. **AVID-C** uses simulated perturbations to study changes in image quality. The paired observations are *not* guaranteed to be pixel-aligned.

This repository accompanies the final **pre-arXiv** paper. No arXiv ID, paper DOI, or conference acceptance is claimed in this package; citation identifiers will be added after publication.

## Dataset

| | Train | Validation | Total |
|---|---:|---:|---:|
| RGB-infrared frame pairs | 28,790 | 3,250 | **32,040** |
| Source videos | 102 | 13 | **115** |
| Bird annotations | 14,113 | 1,358 | **15,471** |
| Drone annotations | 18,335 | 2,146 | **20,481** |
| Total annotations | 32,448 | 3,504 | **35,952** |

Native resolution: **320 × 256**. **24,489 / 35,952 (68.1%)** of the boxes have geometric-mean sides between **8 and 16 px**. **99.0%** are COCO-small. The original Test split is **not included** because reliable RGB-infrared matching was not established from the available records.

Download the released images and labels from [Dharini24/AVID on Hugging Face](https://huggingface.co/datasets/Dharini24/AVID):

```bash
python -m pip install --upgrade "huggingface_hub[cli]"
hf download Dharini24/AVID --repo-type dataset --local-dir ./AVID
```

## What the paper measures

- **Object-relative alignment:** on a 300-pair global-shift sample, the median estimated displacement is **1 px**, and **32.3%** exceed 5 px. For the **264** pairs with matched target annotations, **16.3%** exceed one median target side and the unit-scale alignment burden is **0.265**. The shift search is limited to ±12 px per axis; the values are estimated global shifts, not ground-truth registration.
- **AVID-C:** seven families of simulated noise, blur, illumination change, motion blur, and camera instability at five severity levels. RGB, infrared, or both can be perturbed with reproducible parameters.
- **Fitted class separation:** for targets with geometric-mean sides of **12–16 px**, Bhattacharyya upper bounds are **0.148 RGB**, **0.198 infrared**, and **0.046 paired** under an 8+8 PCA feature representation. Equal-dimensional comparisons reduce the gap; these are **not detector error rates**.
- **Controlled added shifts:** at a **12-px per-axis shift budget**, the fitted fusion-gap retention is **0.844**, measured relative to the unshifted baseline in the specified representation.

The published numerical values are checked against the included CSVs in [`VERIFICATION_REPORT.md`](VERIFICATION_REPORT.md). The full image corpus is on Hugging Face, not in this GitHub source archive.

## Files and reproducibility

```text
paper/                     final 28-page PDF, main.tex, references.bib, main.bbl, figures,
                           analysis code, metadata, synchronized reproducibility CSVs
dataset_tools/             dataset preparation, alignment estimators, AVID-C, PCA/overlap experiments
reproducibility_data/      authoritative CSV tables and their provenance notes
docs/                      GitHub Pages website, current paper.pdf and paper.tex, interactive plots
website/                   alternate deployable website mirror with current paper files
archive/                   historical scripts and alternate preliminary diagnostics
paper/audit/               machine-checkable numerical validations
VERIFICATION_REPORT.md     final audit scope, numeric checks, and remaining limitations
DATA_PROVENANCE_AND_LICENSING.md   source attribution and licensing layers
release_notes/             draft Hugging Face dataset card and post-arXiv citation checklist
```

Recheck the final CSV metrics:

```bash
python paper/audit/verify_final_csvs.py
```

Check implementation tests:

```bash
python -m pytest -q dataset_tools/alignment/test_alignment_metrics.py dataset_tools/corruptions/test_corruptions.py
```

Rebuild the manuscript from `paper/` using `pdflatex`, `bibtex`, and two further `pdflatex` runs. See [`paper/README.md`](paper/README.md) and [`reproducibility_data/README.md`](reproducibility_data/README.md) for the exact data and figures.

## Citation (before an arXiv identifier or DOI is assigned)

```bibtex
@misc{raghavan2026avid,
  title  = {AVID: A Weakly Aligned RGB-Infrared Dataset and Perturbation Benchmark for Tiny Aerial Targets},
  author = {Raghavan, Dharini and Rao, Vishnu and S, Sethu Selvi},
  year   = {2026},
  note   = {Manuscript and dataset release; arXiv identifier and DOI pending},
  url    = {https://github.com/rdharini2001/ICIP_2025_Data_Release}
}
```

Please also cite the **original** Svanström et al. Data in Brief dataset publication. See [`DATA_PROVENANCE_AND_LICENSING.md`](DATA_PROVENANCE_AND_LICENSING.md).

## Licensing and use

Code: **MIT** (`LICENSE`). The Hugging Face dataset listing specifies **CC BY-NC 4.0** for the AVID package, but the underlying source recordings were released under **CC0 1.0 Universal** and retain their original licensing status. The paper includes limitations and responsible-use considerations.
