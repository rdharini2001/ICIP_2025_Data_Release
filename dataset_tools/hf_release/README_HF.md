---
license: cc-by-nc-4.0
pretty_name: "AVID"
size_categories:
- 10K<n<100K
tags:
- aerial-surveillance
- drone-detection
- anti-uav
- infrared
- thermal
- rgb-infrared
- multimodal
- object-detection
- small-object-detection
- weak-alignment
- corruption-benchmark
- counter-uav
- video
task_categories:
- object-detection
---

[![AVID](https://huggingface.co/datasets/Dharini24/AVID/resolve/main/assets/avid_teaser.png)](https://huggingface.co/datasets/Dharini24/AVID/resolve/main/assets/avid_teaser.png)

# AVID: A Weakly Aligned RGB–Infrared Dataset and Perturbation Benchmark for Tiny Aerial Targets

[![AVID Dataset](https://img.shields.io/badge/AVID-Dataset-0B3D91.svg)](#avid-dataset) [![AVID-C Benchmark](https://img.shields.io/badge/AVID--C-Perturbation%20Benchmark-D55E00.svg)](#perturbation-and-alignment-benchmark) [![Paper PDF](https://img.shields.io/badge/Paper-PDF-7A3FF2.svg)](https://huggingface.co/datasets/Dharini24/AVID/resolve/main/paper/main.pdf) [![GitHub](https://img.shields.io/badge/GitHub-Code-181717.svg?logo=github)](https://github.com/rdharini2001/ICIP_2025_Data_Release) [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

We present **AVID**, a paired RGB–infrared dataset and perturbation benchmark for ground-based counter-drone surveillance, where the target being detected is a drone or bird rather than the camera platform. AVID contains **32,040** real RGB–IR frame pairs from **115** short surveillance videos and **35,952** annotated bird/drone instances. Targets are unusually small — **68.1%** of boxes have a geometric-mean side between 8 and 16 pixels in a 320×256 frame — and the two modalities are **weakly aligned rather than pixel-registered**: on a reproducible 300-pair sample, 32.3% of pairs exceed 5px of measured cross-modal shift, large relative to the targets themselves.

AVID ships with **AVID-C**, a deterministic 7-family × 5-severity perturbation suite, and a fully reproducible set of data-intrinsic diagnostics (Bhattacharyya/Chernoff separability bounds, a nonparametric cross-check, a Bayes fusion gap, and a mutual-information sensitivity sweep) that characterize the dataset's own scale, corruption, and alignment properties directly from data — without training a detector. It is built and maintained by [Dharini Raghavan](https://github.com/rdharini2001) and [Vishnu Rao](https://github.com/vrao-ai).

## Paper

**AVID: A Weakly Aligned RGB–Infrared Dataset and Perturbation Benchmark for Tiny Aerial Targets**
Dharini Raghavan¹, Vishnu Rao²
¹Georgia Institute of Technology  ²ARTPARK, Indian Institute of Science

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-purple)](https://huggingface.co/datasets/Dharini24/AVID/resolve/main/paper/main.pdf) [![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github)](https://github.com/rdharini2001/ICIP_2025_Data_Release)

## AVID Dataset

<details>
<summary>Set up the environment (optional)</summary>

```bash
conda create -n huggingface python=3.11 -y
conda activate huggingface
python -m pip install --upgrade pip
python -m pip install --upgrade "huggingface_hub[cli]"
hf --version
hf auth login
```

Logging in is optional; it gives higher download rate limits.

</details>

### Download the dataset

```bash
hf download Dharini24/AVID --repo-type dataset --local-dir ./AVID
```

```
AVID/
├── images/
│   ├── rgb/{train,val}/{video_id}/{frame_idx:06d}.jpg
│   └── ir/{train,val}/{video_id}/{frame_idx:06d}.jpg
├── labels/{train,val}/{video_id}/{frame_idx:06d}.txt      # YOLO format: class cx cy w h, normalized
├── metadata/
│   ├── pairs_{train,val}.jsonl       # per-frame record: video_id, frame_idx, split, classes_present, alignment
│   ├── videos_{train,val}.jsonl      # per-video record: n_frames, classes_present
│   ├── corpus_summary.json
│   ├── bbox_fine_histogram.json
│   ├── real_alignment_sample.json    # 300-pair weak-alignment sample
│   └── object_relative_misalignment.json
├── paper/main.pdf                    # the dataset paper, included for convenience
└── reproducibility_data/             # CSV export of every number reported in the paper
```

Classes: `0 = bird`, `1 = drone` — a fine-grained, thermally-confusable negative class most drone-detection datasets omit.

### Updates

- **AVID 1.0** (2026): First public release — Train/Validation RGB-IR pairs, AVID-C perturbation suite, and the full reproducibility package.

### Dataset at a glance

| | Train | Validation | Total |
|---|---:|---:|---:|
| RGB-IR pairs | 28,790 | 3,250 | **32,040** |
| videos | 102 | 13 | **115** |
| bird instances | 14,113 | 1,358 | **15,471** |
| drone instances | 18,335 | 2,146 | **20,481** |
| all instances | 32,448 | 3,504 | **35,952** |

| | |
|---|---:|
| resolution | **320 × 256** |
| frame rate (measured) | **24 fps** |
| clip duration (measured) | **10.4–13.7 s** |
| targets 8–16px (geometric-mean side) | **68.1%** |
| targets exceeding 48px | **0** |
| pairs with >5px measured cross-modal shift (300-pair sample) | **32.3%** |
| AVID-C corruption families × severities | **7 × 5** |

A Test split exists in the source export but is **not included** in the paired release: its RGB and IR files use incompatible numbering schemes and no authoritative correspondence manifest was available — see "Known issues" below rather than a silently guessed pairing.

### Weak alignment, quantified

Unlike co-located or beam-splitter RGB-T rigs (LLVIP, M3FD, KAIST), AVID's RGB and IR optics are physically separate and only boresight-aligned — the realistic configuration for most fielded dual-sensor counter-drone systems. We measure this rather than assume it away: on a reproducible 300-pair sample (mutual-information registration search, ±12px per axis), the shift distribution is genuinely bimodal (46% at zero measurable offset, a long tail to the search limit), and an **object-relative misalignment ratio** ρ = shift / target-size shows 16.3% of sampled pairs have an estimated shift *larger than the object it describes*. Full per-pair values ship in `metadata/real_alignment_sample.json` and `reproducibility_data/`.

### Perturbation and alignment benchmark

**AVID-C** (released as code, not fixed files) implements 7 corruption families — Gaussian noise, speckle noise, salt-and-pepper noise, Gaussian blur, uneven illumination, motion blur, and camera instability — at 5 reproducible, seeded severities each, calibrated for native 320×256 RGB/thermal imagery. The paper additionally validates that AVID-C's smallest severity is a dominant perturbation relative to the corpus's own real intrinsic noise floor (not swamped by pre-existing sensor/compression noise), and reports a classifier-agnostic separability analysis across scale, corruption, and injected misalignment — see the paper and `reproducibility_data/` for every reported number with its source.

### Known issues

- **Test-split RGB/IR pairing is unresolved** — disclosed rather than guessed; Test is usable for single-modality evaluation only.
- **Alignment metadata is sample-based** (300 pairs), not exhaustive over all 32,040 pairs (~9.8 CPU-hours at the measured rate) — the full-corpus field is documented in the schema for anyone who wants to extend it.
- **Annotation workflow is not fully documented** in the source materials available for this release; we disclose this gap rather than invent a protocol.

### Considerations for using the data

**Weak alignment is real, not a bug.** Don't assume RGB and IR boxes are pixel-identical; use the shipped `alignment_shift_px` / object-relative ratio to filter or weight pairs. **Bird/drone confusability is deliberate** — a feature for benchmarking fine-grained discrimination, not labeling noise; evaluate with scale-stratified buckets rather than one aggregate number. **AVID-C is not a claim of coverage** over all real-world corruptions (no adversarial perturbations, compression artifacts, or thermal-blooming beyond the modeled jitter).

**Ethics / dual use.** This is a counter-drone security dataset. Content is aerial imagery of drones and birds against sky/terrain backgrounds — no people, faces, or personally identifiable information are in frame by construction. Drone-detection technology is dual-use; this release is licensed non-commercially specifically to require a separate conversation before commercial productization. The payload-identification subset and task from the original source challenge are **excluded entirely** from this release.

### License

AVID is released under **CC BY-NC 4.0**. See [`LICENSE`](LICENSE). For commercial licensing inquiries, please open an issue on the [GitHub repository](https://github.com/rdharini2001/ICIP_2025_Data_Release).

---

### Citation

```bibtex
@misc{avid2026,
  title  = {AVID: A Weakly Aligned RGB-Infrared Dataset and Perturbation Benchmark for Tiny Aerial Targets},
  author = {Raghavan, Dharini and Rao, Vishnu},
  year   = {2026},
  note   = {Dataset and preprint; see https://github.com/rdharini2001/ICIP_2025_Data_Release}
}
```

### Acknowledgements

AVID builds on the dataset construction effort for the IEEE ICIP 2025 VIP Cup, "Infrared-Visual Fusion for Enhanced Drone Detection, Tracking and Payload Identification in Surveillance Videos," organized by S. Sethu Selvi, Raghuram S, Sitaram Ramachandrula, Vishnu Rao, Dharini Raghavan, Shefali Singh, Sangeeta Kar, and Suman Jangid (Ramaiah Institute of Technology / Indian Institute of Science / [24]7.ai).
