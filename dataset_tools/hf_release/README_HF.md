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

<h1 align="center">AVID</h1>
<h3 align="center">A Weakly Aligned RGB-Infrared Dataset and Perturbation Benchmark<br/>for Tiny Aerial Targets</h3>
<p align="center">
<a href="https://github.com/rdharini2001/ICIP_2025_Data_Release/blob/claude/rgb-ir-drone-dataset-ppc03f/paper/main.pdf"><img alt="Paper" src="https://img.shields.io/badge/Paper-PDF-6141ac?style=for-the-badge"></a>
<a href="https://github.com/rdharini2001/ICIP_2025_Data_Release"><img alt="GitHub code" src="https://img.shields.io/badge/GitHub-Code-243447?style=for-the-badge"></a>
<a href="https://rdharini2001.github.io/ICIP_2025_Data_Release/"><img alt="Website" src="https://img.shields.io/badge/Research-Website-168a96?style=for-the-badge"></a>
</p>

<p align="center"><img src="https://huggingface.co/datasets/Dharini24/AVID/resolve/main/media/DRONE_082192_RGB_IR.gif" width="780" alt="Animated corresponding RGB and infrared observations of a distant drone"></p>
<p align="center"><sub>Actual paired surveillance video: RGB on the left, infrared on the right. The modalities are not guaranteed to be pixel-aligned.</sub></p>

> **How much can a few pixels change?** AVID contains real surveillance observations of distant birds and drones, with tiny annotated targets and weakly aligned RGB and infrared camera views. AVID-C adds simulated noise, blur, illumination and camera-motion changes so that the effects of image quality can be studied under controlled conditions.

**Dharini Raghavan\*** · **Vishnu Rao\*** · **Sethu Selvi S**  
*Georgia Institute of Technology · ARTPARK, Indian Institute of Science · Ramaiah Institute of Technology*  
\* Equal contribution and shared first authorship.

**Paper:** [PDF](https://github.com/rdharini2001/ICIP_2025_Data_Release/blob/claude/rgb-ir-drone-dataset-ppc03f/paper/main.pdf) · [LaTeX](https://github.com/rdharini2001/ICIP_2025_Data_Release/blob/claude/rgb-ir-drone-dataset-ppc03f/paper/main.tex) · [Code and reproducibility CSVs](https://github.com/rdharini2001/ICIP_2025_Data_Release) · [Interactive research website](https://rdharini2001.github.io/ICIP_2025_Data_Release/)

> [!NOTE]
> **Publication status:** The final three-author manuscript is available in the GitHub repository. An arXiv identifier and paper DOI are **pending**; neither is claimed in this card. Please update the temporary AVID citation once official identifiers are assigned.

## Dataset overview

| Measure | Train | Validation | Total |
|:---|---:|---:|---:|
| RGB/infrared frame pairs | 28,790 | 3,250 | **32,040** |
| Videos | 102 | 13 | **115** |
| Bird annotations | 14,113 | 1,358 | **15,471** |
| Drone annotations | 18,335 | 2,146 | **20,481** |
| All annotations | 32,448 | 3,504 | **35,952** |

- **Capture:** ground-based surveillance, native **320 × 256** RGB and infrared imagery.
- **Targets:** **68.1%** of boxes have a geometric-mean side of **8–16 px**; **99.0%** are COCO-small.
- **Classes:** YOLO label IDs are `0 = bird` and `1 = drone`.
- **Pairing:** the two frames correspond to the same scene but are **not guaranteed pixel-aligned**.
- **Scope:** only the original VIP Cup **Train and Validation** paired splits. The original Test exports are excluded because reliable matching between their RGB and infrared file identifiers has not been established.

AVID was prepared from the **bird-drone subset curated for the IEEE ICIP 2025 Video and Image Processing Cup**. The original footage comes from the **Drone Detection Dataset** published by Svanström, Alonso-Fernandez, and Englund. The videos and existing labels were not newly captured or annotated for this manuscript.

## See the RGB and infrared observations

| Drone sequence | Drone sequence |
|:---:|:---:|
| <img src="https://huggingface.co/datasets/Dharini24/AVID/resolve/main/media/DRONE_083193_RGB_IR.gif" width="430" alt="Real RGB and IR drone sequence"> | <img src="https://huggingface.co/datasets/Dharini24/AVID/resolve/main/media/DRONE_085195_RGB_IR.gif" width="430" alt="Real RGB and IR drone sequence"> |

| Bird sequence | Additional sampled scenes |
|:---:|:---:|
| <img src="https://huggingface.co/datasets/Dharini24/AVID/resolve/main/media/BIRD_03897_RGB_IR.gif" width="430" alt="Real bird RGB and IR sequence"> | <img src="https://huggingface.co/datasets/Dharini24/AVID/resolve/main/media/fig_diversity_montage.png" width="430" alt="Multiple paired RGB and IR scenes"> |

The recordings above are **real imagery**. The image changes used by AVID-C are **simulated perturbations** of that imagery.

## Download

```bash
pip install --upgrade "huggingface_hub[cli]"
hf download Dharini24/AVID --repo-type dataset --local-dir ./AVID
```

The images, labels, and accompanying sequence identifiers are intended to support **video-level evaluation**. Keep frames from the same source video in the same split. Do not randomly split adjacent frames between training and evaluation. For the available file conventions and preparation details, see the [GitHub dataset schema](https://github.com/rdharini2001/ICIP_2025_Data_Release/blob/claude/rgb-ir-drone-dataset-ppc03f/dataset_tools/hf_release/SCHEMA.md).

## AVID-C and benchmark measurements

**AVID-C** simulates **seven perturbation families** at **five severity levels**: Gaussian noise, speckle noise, salt-and-pepper noise, Gaussian blur, uneven illumination, motion blur and camera instability. Apply each to RGB alone, infrared alone, or both, and report the affected modality along with the perturbation type and severity.

<p align="center"><img src="https://huggingface.co/datasets/Dharini24/AVID/resolve/main/media/fig_corruption_analysis.png" width="820" alt="AVID-C corruption results across seven perturbation families"></p>

For **300 sampled pairs**, the estimated global RGB–infrared shift has a **1 px median**, and **32.3%** exceed 5 px. Target-relative alignment uses the **264 pairs with matched annotations**: **16.3%** exceed one median target side, and the unit-scale alignment burden is **0.265**. These are global translation estimates, not pixel-level correspondence or registration ground truth for the entire dataset.

<p align="center"><img src="https://huggingface.co/datasets/Dharini24/AVID/resolve/main/media/fig_dataset_statistics.png" width="820" alt="Target scale and alignment distributions"></p>

The paper's fixed-feature analysis reports fitted Bhattacharyya upper bounds of **0.148 RGB**, **0.198 infrared**, and **0.046 for 8+8 combined features** for objects of 12–16 px. With the same total feature dimension (4+4 components), the combined bound is **0.140**. Such values describe **class separation in a specified feature representation**, **not detector accuracy**. They should not be compared across different model dimensions without appropriate controls.

## Source attribution and licenses

The original surveillance recordings were released as the [Drone Detection Dataset](https://github.com/DroneDetectionThesis/Drone-detection-dataset) by **Fredrik Svanström, Fernando Alonso-Fernandez and Cristofer Englund**. Please cite *A Dataset for Multi-Sensor Drone Detection*, *Data in Brief* 39 (2021), 107521, [DOI: 10.1016/j.dib.2021.107521](https://doi.org/10.1016/j.dib.2021.107521).

The source imagery is covered by **CC0 1.0 Universal**. The AVID dataset package is listed here under **CC BY-NC 4.0**, but this label does **not** replace or restrict the original CC0 status of the source media. The code distributed through GitHub is **MIT-licensed**. Please check the applicable terms for the specific material you reuse.

## Responsible use and limitations

AVID supports research on tiny aerial targets, sensor alignment, tracking and robustness. It does not contain identity or personal-attribute labels. The annotation history is not fully documented; alignment is estimated on a sample, and the simulated image transformations do not cover all real-world failure modes. Real-world surveillance deployment requires separate performance, safety and legal evaluation.

## Citation

Until the AVID paper has an official arXiv ID/DOI:

```bibtex
@misc{raghavan2026avid,
  title  = {AVID: A Weakly Aligned RGB-Infrared Dataset and Perturbation Benchmark for Tiny Aerial Targets},
  author = {Raghavan, Dharini and Rao, Vishnu and S, Sethu Selvi},
  year   = {2026},
  note   = {Manuscript and dataset release; arXiv ID and DOI pending},
  url    = {https://github.com/rdharini2001/ICIP_2025_Data_Release}
}
```

Please also cite the original Svanström et al. *Data in Brief* dataset publication above.
