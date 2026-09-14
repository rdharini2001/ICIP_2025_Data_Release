# Release Plan: Large-Scale Weakly-Aligned RGB-IR Drone Dataset

Working name: **AVID** (Aerial Visible-Infrared Drone dataset). Rename freely — used
consistently below and in the paper/HF card so a rename is a find-and-replace.

Source material this plan is grounded in (from the ICIP 2025 VIP Cup Drive folder):
the challenge proposal (dataset stats, scenario/distortion taxonomy), the internal
"Paper Notes" log (real CycleGAN-Turbo IR↔RGB translation experiments, with
hyperparameters and iteration counts), and the fact that this exact data already
went through a live international challenge (30+ teams), which is where the *design
mistakes teams made* become free signal for what the released benchmark protocol
should enforce.

Per your direction: **payload data and payload task are excluded entirely** from
this release. Everything below concerns the drone/bird detection+tracking data only.

## -1. THEORETICAL PIVOT (2026-09-14, later same day): the paper's spine

Per explicit direction, the paper's central contribution is no longer
"here's a dataset with YOLO baselines" -- it's the **Alignment Robustness
Score (ARS)**, a general, portable metric for how much an arbitrary
cross-modal fusion model's accuracy depends on the (usually silently
assumed) precise registration between its input modalities. AVID becomes
the testbed that makes this measurable and severe (32.3% of pairs >5px
misalignment against 8-16px targets), not the point of the paper by
itself. See `dataset_tools/alignment/alignment_robustness_score.py` (built
and unit-tested) and `paper/main.tex` §"Alignment robustness as a
first-class metric" for the full theoretical framing:

- Grounded in multi-view learning theory (Blum & Mitchell 1998
  co-training): classic theory assumes views are registered to the same
  instance; weak pairing violates that specific assumption.
- Connected to noisy-correspondence learning (Huang et al., NeurIPS 2021),
  which studies violated pairing for global retrieval/matching but -- as
  far as a dedicated verification pass found -- not for dense, per-pixel
  spatial correspondence in detection.
- The metric itself (AUC + critical radius) is a direct structural
  analogue of certified adversarial robustness (Cohen et al., ICML 2019
  randomized smoothing), applied to a geometric cross-modal perturbation
  instead of an adversarial pixel-norm one.
- Important scope correction from an initial overclaim: misalignment-
  *aware* RGB-T detection is an active, real applied literature (CoDAF,
  JFRDet, AlignFreeNet, DOD-SA -- all verified real 2025/2026 arXiv papers,
  citations added to references.bib). None of them report a standardized,
  portable metric comparable across architectures/papers -- each reports
  robustness, if at all, as an internal ablation specific to their own
  proposed fix. That is the precise, narrower gap ARS targets.

Testable hypothesis ARS enables: early/pixel-level fusion architectures
should show a steeper ARS degradation curve (lower AUC, smaller critical
radius) than late/decision-level fusion, since early fusion implicitly
learns a spatial correspondence during training that late fusion never
requires. See PLAN.md §4 (updated) for which fusion strategies we
actually get real ARS numbers for in this pass vs. release as protocol only.

**FINAL REAL RESULT (2026-09-14, end of session):** RGB-only and IR-only
YOLOv8n trained to 7/6 epochs (stopped early -- running both concurrently
on one GPU was starving whichever started second of disk I/O; see
`dataset_tools/baselines/README.md`-style honesty in the paper itself,
`sec:results`). Official Ultralytics val(): RGB P=0.851/R=0.741/
mAP50=0.762/mAP50-95=0.335; IR P=0.719/R=0.650/mAP50=0.587/mAP50-95=0.158.
Matched-methodology mAP50 (our own single-IoU-threshold AP, n=500,
identical calc for all three so the comparison isn't confounded by
methodology): RGB 0.570, IR 0.504, **late fusion (WBF) 0.685 -- fusion
beats both single modalities**. Real ARS (n=20 pairs each): RGB-only is
the expected flat/trivial reference (ars_auc_relative=1.0, no critical
radius, since it never reads IR); **late fusion has a real critical
radius of 15px** (score 0.518->0.225 over the 0-20px sweep), which lands
almost exactly at the ceiling of the corpus's own real measured
misalignment (up to 17px, §0). This is the paper's central empirical
result. Early/mid fusion ARS (to test the comparative hypothesis directly)
remains the single highest-value follow-up experiment.

## 0. STATUS UPDATE (2026-09-14): real data, real GPU, real numbers

Everything below §1 was written before we had the actual pixel data. As of
2026-09-14 we have the real `releasev1` export extracted and indexed
(Train + Validation splits; the Test split and a `pre_releasev2` release
were left unextracted per your call to work with what's already in hand —
see the note at the end of this section), plus a local RTX Blackwell GPU
(24GB VRAM), so this is no longer a "prepare scripts for later" plan for
E1-E8 — those are now either DONE with real numbers or actively running.
Corrections and real findings vs. the original placeholder assumptions:

- **Real scale (Train+Validation, `releasev1`, deduplicated):** 28,790
  train pairs / 102 videos, 3,250 validation pairs / 13 videos. The
  earlier "45K/6.5K/13K" figures were the *source challenge's* stated
  design targets, not what's in this specific export -- and even the raw
  file counts in this export (57,580 train "pairs") turned out to be an
  artifact: **every single frame has exactly one clean copy and one
  Roboflow noise-augmented duplicate** (confirmed across the full corpus,
  not a sample), so the true unique-frame count is exactly half the raw
  file count. We dedupe by keeping the smaller-filesize variant (verified
  reliable indicator of "the clean original" against visual inspection).
- **Class index correction:** the real label files use `0 = bird, 1 =
  drone`, the opposite of the placeholder mapping used before real data
  was available. Fixed in SCHEMA.md and build_hf_dataset.py.
- **RGB and IR share identical box coordinates per nominal frame** —
  confirmed on real data, not assumed. Annotation was done once per frame
  and copied to both modalities, which does *not* correct for the real
  physical RGB/IR offset -- directly motivating §3.3 (alignment
  quantification).
- **The Test split has a real, disclosed data-organization problem**: RGB
  and IR files in `Test_detection&tracking(_with_labels)` use completely
  different, non-overlapping numbering schemes, even at the raw video
  level (13 IR videos vs. 12 RGB videos, no shared IDs, no manifest file
  to resolve it). Train/Validation pair perfectly by filename key; Test
  does not. We are treating Test as single-modality-only evaluation data
  until/unless a corrected release (`pre_releasev2`, not yet extracted)
  resolves this -- see the runbook note below. **This asymmetry itself is
  worth a paragraph in the paper's Limitations**: it is exactly the kind
  of real-world data-engineering failure mode a released benchmark should
  document rather than silently paper over.
- **Extreme small-object distribution (new finding, real data):** COCO-style
  small/medium/large bucketing collapses almost the entire dataset into
  "small" (~99%, zero "large" instances in either split). A finer
  pixel-side-length histogram is far more informative: **67.6% of all
  ~32K annotated boxes are 8-16px on a side**, and *zero* boxes exceed
  48px, in a 320x256 frame. This is a stronger, more specific small-object
  characterization than any of the comparison datasets in Table 1 report,
  and justifies a dedicated fine-grained scale-stratified evaluation
  protocol (§4, new item) rather than the coarser COCO buckets originally
  planned.
- Not extracted / not used for this pass, by your direction to proceed
  with what's already in hand rather than wait further: `Test_detecttrack`
  (unlabeled test videos), `detecttrack.zip` (appears to be a duplicate of
  `releasev1` based on matching internal folder name -- not confirmed
  byte-identical), and `pre_releasev2-detection&tracking.zip` (3.24GB,
  likely a corrected/expanded release given the naming and given it's
  larger than releasev1+Test combined -- worth extracting in a follow-up
  pass specifically to check whether it fixes the Test RGB/IR pairing
  problem before a camera-ready revision).

## 1. What's actually in hand

- ~45K train / 6.5K val / 13K test RGB-IR **image pairs**, plus **video sequences**
  (40 val, 25 test, 10s @ 30fps, 320×256) — so this is a video dataset with a
  frame-pair sampling, not just a still-image pair corpus. That's the single
  biggest under-exploited asset: most RGB-T pair datasets (LLVIP, M3FD, RoadScene)
  are still-image only and have no temporal/tracking dimension.
- Two classes: drone vs. bird — a genuinely hard fine-grained discrimination
  problem in thermal (similar small warm blobs at range) that most drone-vision
  datasets don't have to deal with because they don't include birds as a
  confusable negative class.
- Scenario coverage: hilly regions, thick forest cover, cloudy sky, fog/mist;
  single- and multi-instance frames; swarms; objects far off the camera's FoV.
- A built-in, graded distortion taxonomy already applied to 30 videos: speckle
  noise, salt-and-pepper, Gaussian blur, uneven illumination, motion blur, camera
  instability, AWGN — each with a documented severity rating (1-5). This is a
  corruption-robustness benchmark that already half-exists; it just hasn't been
  formalized, parameterized, or made reproducible/extensible.
- RGB-IR pairs are **weakly aligned** (not pixel-registered) — true of the source
  sensor rig, not a defect to hide. This is actually the realistic case for most
  deployed dual-sensor gimbals (RGB and thermal cameras are physically offset,
  have different FOVs/optics, and are boresight-aligned at best), unlike LLVIP/M3FD
  which use co-located or heavily pre-registered rigs. Almost no existing paired
  RGB-T dataset documents alignment quality quantitatively — we can be first here.
- A real, already-executed cross-modal translation experiment log: CycleGAN-Turbo
  trained IR→RGB (and night→day) using LLVIP+M3FD+RoadScene+FLIR plus ~20K of our
  own unpaired drone/bird images, 13,500 iterations total, LoRA ranks (UNet 128,
  VAE 4), documented failure modes for two earlier loss formulations (vanilla
  cycle+identity loss; VGG-loss + spectral norm). This is a legitimate, reportable
  baseline experiment, not a placeholder.

## 2. The novelty problem and how this release solves it

A plain "here is an RGB+IR drone dataset" paper is a weak arXiv submission in 2025 —
LLVIP/M3FD/Anti-UAV/DroneRGBT/VTUAV already exist. The defensible angle is:
**drones (and their thermal-confusable look-alike, birds) as the target class, in
video, under a controlled and reproducible corruption suite, with weak alignment
treated as a first-class, quantified property rather than swept under the rug.**
No existing public RGB-T pair dataset combines all four of: (a) drone-as-target,
(b) video/tracking, (c) a parameterized corruption benchmark, (d) documented,
per-pair alignment-quality metadata.

## 3. Concrete contributions to build and claim

1. **The dataset itself** (detection + tracking, drone vs. bird, RGB+IR, video +
   frame pairs), released on Hugging Face under CC-BY-NC-4.0, video-level splits
   with no leakage (see §4).
2. **A reproducible, parameterized corruption suite ("AVID-C")** — code
   (`dataset_tools/corruptions/`) that regenerates the 7 corruption families at 5
   controlled severities, so it isn't limited to the 30 originally-distorted
   videos: any user can corrupt any subset, at any severity, deterministically
   (seeded). This is the ImageNet-C move applied to a paired RGB-T drone dataset —
   nobody has done this for this data type yet, as far as the related-work research
   turned up (background job in progress on that verification).
3. **Per-pair alignment-quality metadata** — a phase-correlation / edge-based
   registration-offset estimator (`dataset_tools/alignment/`) run once over the
   corpus to attach an `alignment_shift_px` / `alignment_confidence` field to every
   RGB-IR pair's metadata. Downstream users can filter, weight, or stratify by
   alignment quality instead of discovering it the hard way.
4. **A missing-modality / graceful-degradation evaluation protocol** — at test
   time, evaluate the same fusion model with RGB dropped, IR dropped, and both
   present, reporting a degradation curve. Directly motivated by weak alignment and
   by real-world sensor dropout (fog fully blinds RGB; a cold thermal background
   can wash out IR).
5. **An alignment-sensitivity study** — synthetically inject additional, controlled
   spatial misalignment (shift/rotate/scale one modality) on top of the existing
   weak alignment and measure fusion accuracy vs. injected misalignment magnitude.
   This directly answers "how much does alignment precision actually matter for
   drone fusion detection", which is an open, citable empirical question.
6. **A video-level-split leakage ablation.** Frame-level random splitting on 10s/
   30fps clips leaks near-duplicate frames across train/val/test and inflates
   reported metrics — a documented failure mode we can demonstrate directly
   (compare frame-level vs. video-level split performance on the same model). This
   also sets a citable best-practice for anyone reusing the data.
7. **A released cross-modal translation baseline** (IR→RGB via CycleGAN-Turbo),
   reporting the real training configuration and using the translated synthetic
   RGB as a documented augmentation path for training under missing/degraded RGB.
8. **Scale-stratified evaluation** (COCO-style small/medium/large bounding-box
   buckets) because "objects far off the FoV" is an explicit design scenario —
   report performance broken out by object pixel-area bucket, not just an
   aggregate mAP that hides far-range failure.
9. **A datasheet for the dataset** (Gebru et al. style) plus an explicit ethics /
   dual-use statement: this is a counter-drone security dataset (aerial
   drone/bird silhouettes only, no people/faces/PII, staged/recorded for the
   challenge), released CC-BY-NC-4.0, research-use framing stated up front.

Explicitly **not** in scope per your call: payload data/task, and no
leaderboard/team-attributed results from the 2025 challenge anywhere in the paper
or dataset card.

## 4. Experiment plan (status as of 2026-09-14, running on real GPU + real data)

| # | Experiment | Script | Status |
|---|---|---|---|
| E1 | Corpus statistics: class balance, video/frame counts | `dataset_tools/hf_release/fast_index.py` | **DONE, real numbers** (§0) |
| E1b | Fine-grained bbox-size histogram (COCO buckets are uninformative here) | `dataset_tools/hf_release/bbox_histogram.py` | **DONE, real numbers** (§0) |
| E2 | Single-modality baselines: YOLOv8n on RGB-only, IR-only | `ultralytics` train against `configs/rgb.yaml` / `configs/ir.yaml` | RGB training running; IR queued next |
| E3-late | Late (decision-level) fusion via Weighted Boxes Fusion combining the independently-trained RGB and IR detectors | `dataset_tools/baselines/late_fusion_wbf.py` (unit-tested) | queued after E2 |
| E3-early/mid | Early-fusion (channel-concat) and mid-fusion (CBAM dual-stream) reference architectures | `dataset_tools/baselines/fusion_models.py`, `patch_yolo_early_fusion()` | released as reference code, not trained this pass (time-boxed decision -- late fusion gives a real, defensible "fusion beats single modality" result without needing new 4/6-channel training data) |
| E4 | Tracking metrics: approach/recede classification, missed-track-<15-frames rule | `dataset_tools/baselines/track_eval.py` (unit-tested) | needs a tracker (ByteTrack/BoT-SORT) run on top of E2 detections -- queued if time allows |
| E5 | AVID-C robustness sweep (6 *photometric* corruptions x 5 severities; camera_instability handled separately in E7, see rationale in the script) using Ultralytics' own `val()` for real mAP50/mAP50-95/precision/recall | `dataset_tools/hf_release/yolo_corruption_eval.py` | queued after E2, runs on a random 400-image validation subsample |
| E6 | Missing-modality degradation (drop RGB or IR entirely, or corrupt one modality while holding the other clean) | extension of E5's harness | queued |
| E7 | Real per-pair alignment quantification (sampled) | `dataset_tools/hf_release/real_alignment_sample.py` | **DONE** on 300 real pairs -- see `metadata/real_alignment_sample.json` |
| E7b | Alignment-sensitivity study (injected misalignment vs. accuracy) -- also where camera_instability-style geometric perturbation belongs, since it tracks the transform explicitly instead of assuming static labels | `dataset_tools/alignment/sensitivity_study.py` (unit-tested end-to-end with a toy model) | queued, needs wiring to the real E2 checkpoint |
| E8 | Frame-level vs. video-level split-leakage ablation | `dataset_tools/hf_release/real_split_leakage.py` | **DONE on the REAL corpus**: 32,040 frames / 115 videos (train+val combined) -- 87.5% leakage under naive frame-level splitting vs. 0.0% for video-level. Near-identical to the earlier synthetic-corpus estimate (87.8%), a nice cross-validation of that simulation. |
| E9 | Cross-modal translation baseline report (CycleGAN-Turbo IR→RGB) | already executed once, real log in Drive | **DONE**, real hyperparameters, pull into paper directly |
| E10 (new) | Tiny-object stratified recall: bucket predictions/misses by the E1b histogram bins (8-16px vs 16-32px vs >32px) rather than COCO's uninformative 3 buckets | to add: a thin wrapper around Ultralytics' per-image prediction output, joined against `bbox_fine_histogram.py`'s bins | planned, real-data dependent |
| E11 (new) | Fusion-gain metric: Δ(mAP50) of late-fusion vs. max(RGB-only, IR-only) per corruption condition -- directly answers "does fusion actually help, and when" rather than reporting three separate numbers | derived from E2 + E3-late + E5 results, no new code | planned |
| E12 (new) | Cross-modal complementarity/error analysis: for frames where RGB-only misses the target, what fraction does IR-only catch (and vice versa)? A confusion-style breakdown, not just an aggregate score | derived from E2 per-image predictions | planned |

Time-boxing note: given real GPU/data access only materialized partway
through this pass (see §0), E3-early/mid, full tracking (E4), and E10-E12
are scoped as "do if time allows, else ship as protocol + code" -- consistent
with the honesty principle from the original plan (§6 of the old plan):
never present a number that wasn't actually computed.

## 5. Recommended paper structure (arXiv, NeurIPS D&B / WACV-level rigor)

See `paper/main.tex`. Sections: Abstract; Introduction (gap + contributions);
Related Work (comparison table vs. LLVIP/M3FD/RoadScene/KAIST/FLIR-ADAS/Anti-UAV
family/DroneRGBT/VTUAV/VisDrone); Dataset Construction & Statistics; Weak
Alignment as a Quantified Property; The AVID-C Corruption Benchmark; Baseline
Protocol & Cross-Modal Translation Baseline; Evaluation Protocol (scale-stratified,
missing-modality, split-leakage guidance); Ethics & Responsible Use; Limitations;
Conclusion. Author list: you + Vishnu Rao, per your direction. No leaderboard /
challenge-team results anywhere in the paper.

## 6. Release mechanics (what you run, since I can't touch HF/arXiv credentials
or move multi-GB files from here)

See `docs/RELEASE_RUNBOOK.md`.
