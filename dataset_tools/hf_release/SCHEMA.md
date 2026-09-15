# AVID directory & metadata schema

Target layout for the Hugging Face release (payload data/task excluded
entirely, per release scope). This is what `build_avid_v1.py` produces
from the extracted Roboflow export, and what
`dataset_tools/theory/patch_extraction.py` reads to build every real
number in the paper.

```
avid/
  images/
    rgb/{split}/{video_id}/{frame_idx:06d}.jpg
    ir/{split}/{video_id}/{frame_idx:06d}.jpg
  labels/
    {split}/{video_id}/{frame_idx:06d}.txt      # YOLO format, see below
  videos/
    rgb/{split}/{video_id}.mp4
    ir/{split}/{video_id}.mp4
  metadata/
    pairs.jsonl      # one JSON object per frame pair
    videos.jsonl      # one JSON object per video
  avid_c/              # optional: pre-materialized AVID-C corruption suite
    {corruption}/{severity}/{modality}/{split}/{video_id}/{frame_idx:06d}.jpg
  README.md            # the dataset card (this becomes the HF repo's model/dataset card)
  LICENSE               # CC-BY-NC-4.0 text
```

`split` is one of `train`, `val`, `test`. Splits are defined at the
**video** level (every frame of a given `video_id` is in exactly one split)
per PLAN.md Experiment E8 — never re-split at the frame level, it leaks.

## Label format

YOLO-style, one row per object, normalized `[0, 1]` coordinates:

```
<class_id> <cx> <cy> <w> <h>
```

`class_id`: `0` = bird, `1` = drone (confirmed against the real Roboflow-exported
label files — this is the opposite of the placeholder mapping used before real
data was available, so double-check any code you wrote against the old
assumption). Boxes are given once per pair (i.e. the
same box file is the ground truth for both the RGB and IR frame in that
pair) since annotation was performed on the weakly-aligned pair jointly; see
`alignment_shift_px` in `pairs.jsonl` for how far the two modalities can
actually disagree geometrically.

## `metadata/pairs.jsonl` fields

| field | type | description |
|---|---|---|
| `video_id` | str | parent video/clip identifier |
| `frame_idx` | int | 0-indexed frame number within the clip |
| `split` | str | `train` / `val` / `test` |
| `classes_present` | list[str] | subset of `{"drone", "bird"}` present in this frame |
| `scenario_tags` | list[str] | any of `hilly`, `forest`, `cloudy`, `fog_mist`, `swarm`, `far_range`, `multi_instance` |
| `builtin_distortion` | str or null | one of the 7 AVID-C corruption names if this frame is from one of the 30 originally-distorted videos, else null |
| `builtin_distortion_severity` | int or null | 1-5, per `PROPOSAL_DEFAULT_SEVERITIES` in `dataset_tools/corruptions/corruptions.py`, if applicable |
| `alignment_shift_px` | float | estimated RGB-IR geometric offset for this pair, from `dataset_tools/alignment` (Euclidean px) |
| `alignment_confidence` | float | normalized mutual information at the estimated shift (see `alignment_metrics.py`); higher = more trustworthy estimate |

## `metadata/videos.jsonl` fields

| field | type | description |
|---|---|---|
| `video_id` | str | identifier, matches the `videos/{rgb,ir}/{split}/{video_id}.mp4` filenames |
| `split` | str | `train` / `val` / `test` |
| `duration_s` | float | nominal 10.0 for the original clips |
| `fps` | int | nominal 30 |
| `resolution` | [int, int] | `[320, 256]` |
| `n_frames` | int | |
| `scenario_tags` | list[str] | as above, video-level union of its frames' tags |
| `classes_present` | list[str] | as above |

## Notes for whoever runs `build_avid_v1.py`

- Operates on the extracted `releasev1` Roboflow export, deduplicating the
  systematic noise-augmented copy Roboflow baked into every frame (100% of
  the 28,790 Train and 3,250 Validation frame keys have exactly 2 file
  variants per modality; smaller JPEG file size reliably identifies the
  clean original).
- The Test split's RGB and IR files use non-overlapping numbering schemes
  with no manifest to resolve them (see the paper's "pitfalls" section);
  `build_avid_v1.py` does not attempt to force an unverified pairing, and
  reports Test RGB/IR as separate single-modality pools.
- `dataset_tools/hf_release/real_alignment_sample.py` (using
  `dataset_tools/alignment/alignment_metrics.py`'s `pair_alignment_quality`)
  computes `alignment_shift_px` / `alignment_confidence` on a 300-pair
  random sample, not exhaustively over all 32,040 pairs (~9.8 CPU-hours at
  the measured ~1.1s/pair) — extending to the full corpus is a
  straightforward but not-yet-run batch job.
- Payload data/labels, if present in any source zip, must **not** be copied
  into this layout — this release is drone/bird detection+tracking only.
