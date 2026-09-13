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

## 4. Experiment plan (what needs to actually run on real hardware)

I can't pull the multi-GB source zips into this sandbox (base64 transfer through
the Drive connector would cost millions of tokens per file), so I've written the
code for all of the below but **none of it has been executed on the real corpus
yet** — that has to happen on a machine with the actual data (yours, or Vishnu's).
Treat the paper's results tables as templates to fill in, not finished numbers.

| # | Experiment | Script | Depends on |
|---|---|---|---|
| E1 | Corpus statistics (class balance, bbox size distribution, env-factor coverage, video durations) | `dataset_tools/stats/` (to add once annotation schema is confirmed) | raw annotations |
| E2 | Single-modality baselines: YOLO on RGB-only, IR-only | `dataset_tools/baselines/train_single_modality.py` | restructured data |
| E3 | Fusion baselines: early (channel-concat), mid (feature-level attention fusion), late (decision fusion) | `dataset_tools/baselines/train_fusion.py` + `fusion_models.py` | restructured data |
| E4 | Tracking: ByteTrack/BoT-SORT on top of E2/E3 detectors; MOTA/IDF1 + approach/recede classification + missed-track-<15-frames rule from the original proposal | `dataset_tools/baselines/track_eval.py` | E2/E3 checkpoints |
| E5 | Corruption robustness sweep (AVID-C, 7×5) → mean corruption error per modality/fusion variant | `dataset_tools/corruptions/run_corruption_eval.py` | E2/E3 checkpoints |
| E6 | Missing-modality degradation curve | same eval harness, modality-drop flag | E3 checkpoint |
| E7 | Alignment-sensitivity study (injected misalignment vs. accuracy) | `dataset_tools/alignment/sensitivity_study.py` | E3 checkpoint |
| E8 | Frame-level vs. video-level split leakage ablation | `dataset_tools/baselines/split_leakage_ablation.py` | raw video index |
| E9 | Cross-modal translation baseline report (CycleGAN-Turbo IR→RGB) | already executed once (real log in Drive); package as `dataset_tools/translation/README.md` with the real hyperparameters | — |

E9 is the only one with real numbers/logs already in hand (via wandb runs
referenced in the Drive doc) — pull those into the paper directly.

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
