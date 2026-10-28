---
license: cc-by-nc-4.0
pretty_name: AVID
size_categories:
- 10K<n<100K
task_categories:
- object-detection
tags:
- aerial-surveillance
- drone-detection
- infrared
- thermal
- rgb-infrared
- small-object-detection
- weak-alignment
- corruption-benchmark
---

# AVID: A Weakly Aligned RGB-Infrared Dataset and Perturbation Benchmark for Tiny Aerial Targets

**Dharini Raghavan\***, **Vishnu Rao\***, **Sethu Selvi S**  
\* The first two authors contributed equally and share first authorship.

**Paper:** [Final preprint PDF](https://github.com/rdharini2001/ICIP_2025_Data_Release/blob/claude/rgb-ir-drone-dataset-ppc03f/paper/main.pdf) · [Source LaTeX](https://github.com/rdharini2001/ICIP_2025_Data_Release/blob/claude/rgb-ir-drone-dataset-ppc03f/paper/main.tex) · [Code and reproducibility CSVs](https://github.com/rdharini2001/ICIP_2025_Data_Release) · [Research website](https://rdharini2001.github.io/ICIP_2025_Data_Release/)

**Publication status:** arXiv submission and a paper DOI are **pending**. Please use the temporary three-author citation below until identifiers have been assigned. No arXiv ID or paper DOI is claimed here.

## Dataset overview

AVID contains **32,040 corresponding RGB-infrared frame pairs** from **115** ground-based surveillance videos, with **35,952** existing bird/drone bounding-box annotations. It was prepared from the bird-drone data previously curated for the IEEE ICIP 2025 Video and Image Processing Cup, using original footage from the **Drone Detection Dataset** collected by Svanström, Alonso-Fernandez, and Englund.

The RGB and infrared streams correspond to the same scene, but the cameras are **not necessarily pixel-aligned**. The original videos and existing annotations were not newly captured or labeled for this paper.

| Quantity | Train | Validation | Total |
|---|---:|---:|---:|
| RGB-IR pairs | 28,790 | 3,250 | 32,040 |
| Videos | 102 | 13 | 115 |
| Bird instances | 14,113 | 1,358 | 15,471 |
| Drone instances | 18,335 | 2,146 | 20,481 |
| All instances | 32,448 | 3,504 | 35,952 |

Frames are **320 × 256 pixels**; **68.1%** of annotated boxes have geometric-mean sides of 8–16 pixels and **99.0%** are COCO-small. Classes are `0=bird`, `1=drone` in the provided YOLO-style labels.

### Download

```bash
python -m pip install --upgrade "huggingface_hub[cli]"
hf download Dharini24/AVID --repo-type dataset --local-dir ./AVID
```

The current paired release covers **Train and Validation only**. The source Test exports are excluded because reliable RGB-infrared frame correspondence could not be established from the available records.

## AVID-C and alignment

AVID-C applies **seven types of simulated image degradation** at **five severity levels** to RGB, infrared, or both modalities. The released implementation and parameters reproduce the transformations without publishing separate corrupted copies of the images.

Global RGB-infrared translations were estimated on **300 sampled pairs** (median shift 1 px; 32.3% exceed 5 px). Target-relative alignment analysis uses the **264 pairs with matched annotations**, of which 16.3% exceed one median target side. The unit-scale alignment burden for the 264 pairs is **0.265**. These are estimated global translations, **not** ground-truth pixel correspondence for all frames.

The paper studies Bhattacharyya and Chernoff upper bounds and a KDE comparison using a fixed PCA representation. These values represent fitted class separation, **not detector error rates**. Evaluation on other cameras, environments and models is needed for deployment claims.

## Source attribution and licensing

The original **Drone Detection Dataset** was collected and publicly shared by **Fredrik Svanström, Fernando Alonso-Fernandez and Cristofer Englund**. Source repository: https://github.com/DroneDetectionThesis/Drone-detection-dataset. Please cite Svanström et al., *A Dataset for Multi-Sensor Drone Detection*, Data in Brief 39 (2021), 107521, https://doi.org/10.1016/j.dib.2021.107521.

**License clarification:** the source imagery was released under **CC0 1.0 Universal**. The AVID dataset-package listing uses **CC BY-NC 4.0**, which does not change the original CC0 status of the underlying source media. Code in the GitHub repository uses **MIT**.

## Responsible use and limitations

The dataset supports research on tiny-target detection, weak alignment and image degradation. It does not provide person-identification or personal-attribute labels. UAV surveillance can have privacy and dual-use implications. The label review history is incomplete, alignment was measured on a sample, and simulated degradation does not cover every real-world failure mode. These limitations are described in the paper.

## Temporary citation (update after arXiv/DOI assignment)

```bibtex
@misc{raghavan2026avid,
  title = {AVID: A Weakly Aligned RGB-Infrared Dataset and Perturbation Benchmark for Tiny Aerial Targets},
  author = {Raghavan, Dharini and Rao, Vishnu and S, Sethu Selvi},
  year = {2026},
  note = {Manuscript and dataset release; arXiv ID and DOI pending},
  url = {https://github.com/rdharini2001/ICIP_2025_Data_Release}
}
```
