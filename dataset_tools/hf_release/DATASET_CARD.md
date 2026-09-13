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
  as source video (30 fps, 10 s clips, 320×256).
- **Classes:** `drone`, `bird` — a fine-grained, thermally-confusable
  negative class most drone-detection datasets don't include.
- **Scale:** ~45K train / 6.5K val / 13K test RGB-IR pairs, plus 40
  validation and 25 test video sequences (see `metadata/videos.jsonl` for the
  exact, post-restructuring counts — fill in once `build_hf_dataset.py` has
  run over the real corpus).
- **Alignment:** RGB and IR are captured from physically offset optics
  (different FOVs, boresight-aligned at best) — **weakly aligned, not
  pixel-registered**, unlike LLVIP/M3FD/KAIST. Every pair ships with an
  estimated `alignment_shift_px` / `alignment_confidence` (see
  `dataset_tools/alignment/`) so this is a documented property, not a
  discovered defect.
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

**Splits.** Video-level, not frame-level (Experiment E8 in `PLAN.md`
demonstrates why: naive frame-level splitting on 30 fps/10 s clips measured
~88% train-frame leakage into val/test on a corpus shaped like this one, vs.
0% for video-level splitting — inflating any metric computed on a
frame-level split).

## Considerations for Using the Data

**Weak alignment is real, not a bug.** Don't assume RGB and IR boxes are
pixel-identical; use the shipped `alignment_shift_px` to filter or weight
pairs, and see the alignment-sensitivity study protocol in
`dataset_tools/alignment/sensitivity_study.py` before assuming a fusion
model's failure mode is architectural rather than geometric.

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
