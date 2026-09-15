---
license: cc-by-nc-4.0
task_categories:
  - object-detection
  - video-classification
tags:
  - drone-detection
  - anti-uav
  - rgb-thermal
  - multimodal
  - infrared
  - object-tracking
  - robustness
  - corruption-benchmark
  - surveillance
size_categories:
  - 10K<n<100K
pretty_name: "AVID: Aerial Visible-Infrared Drone Dataset"
---

# AVID: Aerial Visible-Infrared Drone Dataset

> Rename this file's H1 and the `pretty_name` above if you pick a different
> dataset name before release — used as a placeholder that's easy to
> find-and-replace. This file is written to double as the arXiv paper's
> dataset-description source of truth; keep the two in sync if you edit one.

## Dataset Description

AVID is a large-scale, **weakly-aligned** RGB-Infrared (thermal) video and
image-pair dataset in which **drones and birds are the target objects being
detected and tracked** — as opposed to drone-*view* datasets (VisDrone,
DroneRGBT, VTUAV) where a drone-mounted camera looks down at other things.
It originates from the dataset built for the IEEE ICIP 2025 VIP Cup,
"Infrared-Visual Fusion for Enhanced Drone Detection, Tracking and Payload
Identification in Surveillance Videos." **This release covers the
detection + tracking data only; the payload-identification subset and task
are explicitly excluded.**

- **Modality:** paired RGB and thermal/IR, both as extracted frame pairs and
  as source video (24 fps measured, 10.4-13.7s clips measured, 320×256).
- **Classes:** `bird` (label id 0), `drone` (label id 1) — a fine-grained,
  thermally-confusable negative class most drone-detection datasets don't
  include.
- **Scale (real, measured, deduplicated):** 28,790 train pairs across 102
  videos; 3,250 validation pairs across 13 videos; Test reported separately
  per modality (6,500 IR frames / 3,250 RGB frames, 13 IR videos / 12 RGB
  videos) because Test's RGB and IR files cannot currently be matched into
  pairs — see "Known issues" below. Note the asymmetry: Test RGB went
  through the same export pipeline as Train/Validation (2 file variants per
  frame, deduplicated to 3,250), but Test IR did not (6,500 raw,
  sequentially-numbered frames, no duplicate to remove) — a further sign
  the two modalities' Test data came from different processing passes.
  Train/Validation's raw source export contained exactly 2x their listed
  counts: every frame had a systematic Roboflow noise-augmentation
  duplicate, which we identified and removed (kept the
  smaller-filesize, clean variant) rather than silently double the
  advertised scale.
- **Extreme small-object distribution:** 67.6% of all ~32K annotated boxes
  are 8-16px wide in a 320x256 frame; zero boxes exceed 48px in either
  split. Standard COCO small/medium/large buckets are not informative here
  (>99% "small"); see `dataset_tools/hf_release/bbox_histogram.py` for the
  fine-grained bucketing we recommend using instead.
- **Alignment:** RGB and IR are captured from physically offset optics
  (different FOVs, boresight-aligned at best) — **weakly aligned, not
  pixel-registered**, unlike LLVIP/M3FD/KAIST, and the two modalities'
  boxes are annotated once and shared (not independently corrected per
  modality). On a real 300-pair sample, measured shift is bimodal: 46% at
  zero measurable offset, but 32.3% exceed 5px (up to 17px; mean 4.66px,
  median 1.0px, $\sigma=6.44$px) — large relative to the 8-16px targets
  themselves. See `dataset_tools/alignment/` and
  `metadata/real_alignment_sample.json`.
- **Built-in corruption suite (AVID-C):** speckle noise, salt-and-pepper
  noise, Gaussian blur, uneven illumination, motion blur, camera
  instability/jitter, and additive white Gaussian noise, each at 5
  reproducible severities via `dataset_tools/corruptions/` (extends the 30
  originally fixed-severity distorted videos from the source challenge into
  a general, regenerate-on-demand robustness benchmark).
- **Scenario coverage:** hilly regions, thick forest cover, cloudy sky,
  fog/mist; single- and multi-instance frames; swarms; targets far from the
  camera's field of view.

## Dataset Structure

See `dataset_tools/hf_release/SCHEMA.md` for the exact directory layout and
the full field list for `metadata/pairs.jsonl` and `metadata/videos.jsonl`.
Splits are defined at the video level — every frame of a given clip is in
exactly one of `train`/`val`/`test` (see "Known pitfall" below).

## Dataset Creation

**Source.** Captured for the IEEE ICIP 2025 VIP Cup by the organizing team
(Ramaiah Institute of Technology, in partnership with IISc and industry
collaborators) as paired RGB and thermal surveillance footage of drones and
birds under varied environmental conditions, with a subset synthetically
degraded under the 7 corruption families described above at fixed, graded
severities as part of the original challenge design.

**Annotation.** Bounding boxes for `drone`/`bird`, provided per the source
challenge's release; verify/re-state the exact annotation protocol
(single-annotator vs. reviewed, tool used) once you have that documentation
in hand — flagging this as a gap in the current card rather than guessing.

**Splits.** Video-level, not frame-level (Experiment E8 in `PLAN.md`: on the
real 32,040-frame/115-video corpus, naive frame-level splitting measured
**87.5%** train-frame leakage into val/test, vs. **0.0%** for video-level
splitting — inflating any metric computed on a frame-level split).

## Known Issues

- **Test-split RGB/IR pairing is unresolved.** The Test export's RGB files
  use a `V_` prefix with independent numbering (`V_BIRD_00564_*`) while IR
  files use a different scheme (`IR_BIRD_068142_*`); this mismatch persists
  at the raw video level (13 IR videos vs. 12 RGB videos, no shared IDs, no
  manifest). Treat Test as single-modality-only evaluation data until a
  corrected export resolves this.
- **Alignment metadata is sample-based, not exhaustive.** Computing
  `alignment_shift_px` for all 32,040 pairs would cost ~9.8 CPU-hours at
  ~1.1s/pair; the released `metadata/real_alignment_sample.json` covers a
  random 300-pair sample. Extending to the full corpus is a straightforward
  but not-yet-run batch job (`dataset_tools/hf_release/real_alignment_sample.py`).

## Considerations for Using the Data

**Weak alignment is real, not a bug.** Don't assume RGB and IR boxes are
pixel-identical; use the shipped `alignment_shift_px` to filter or weight
pairs. See the paper's classifier-agnostic Bayes Fusion Gain analysis
(`dataset_tools/theory/alignment_sweep.py`) for a quantified account of how
much this costs a fusion system in theory before assuming a model's
failure mode is architectural rather than geometric.

**Corruption suite scope.** AVID-C targets the 7 degradation families
observed in the source surveillance footage; it is not a claim of coverage
over all possible real-world corruptions (adversarial perturbations,
sensor-specific artifacts like thermal blooming or rolling-shutter beyond
the modeled jitter, compression artifacts, etc. are out of scope).

**Bird/drone confusability.** The two classes are deliberately confusable in
thermal at range — this is a feature for benchmarking fine-grained
discrimination, not a labeling-noise concern, but expect lower per-class
accuracy at long range than aggregate mAP suggests; use scale-stratified
evaluation (bucket by bounding-box pixel area) rather than a single
aggregate number.

**Ethics / dual use.** This is a counter-drone security dataset. Content is
aerial imagery of drones and birds against sky/terrain backgrounds — no
people, faces, or other personally identifiable information are within the
frame by construction of the capture setup. That said, drone-detection
technology has dual-use potential (legitimate airspace security vs.
surveillance overreach); this release is licensed CC-BY-NC-4.0
(non-commercial, research-use framing) specifically to discourage direct
commercial productization without a separate license conversation with the
dataset owners.

**What's excluded.** Payload-identification images/labels and task are not
part of this release. No challenge-leaderboard or team-attributed results
are included in this card or the accompanying paper, by the dataset owners'
choice.

## Licensing and Citation

Licensed under **CC-BY-NC-4.0**. See `LICENSE`.

If you use this dataset, please cite:

```bibtex
@misc{avid2025,
  title  = {AVID: A Large-Scale Weakly-Aligned RGB-Infrared Drone and Bird
            Detection, Tracking, and Corruption-Robustness Benchmark},
  author = {Raghavan, Dharini and Rao, Vishnu},
  year   = {2025},
  note   = {arXiv preprint},
}
```

(Update with the final arXiv ID once posted — see `docs/RELEASE_RUNBOOK.md`.)

## Acknowledgements

Built on top of the dataset construction effort for the IEEE ICIP 2025 VIP
Cup, "Infrared-Visual Fusion for Enhanced Drone Detection, Tracking and
Payload Identification in Surveillance Videos," organized by S. Sethu Selvi,
Raghuram S, Sitaram Ramachandrula, Vishnu Rao, Dharini Raghavan, Shefali
Singh, Sangeeta Kar, and Suman Jangid (Ramaiah Institute of Technology /
IISc / [24]7.ai).
