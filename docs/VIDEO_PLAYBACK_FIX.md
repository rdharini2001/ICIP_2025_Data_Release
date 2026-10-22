# Paired video playback — October 2026

The original website assets used MPEG-4 Part 2 (`mp4v`) video inside `.mp4` files. The videos are now encoded in browser-compatible H.264/AVC Constrained Baseline (`avc1`), 8-bit `yuv420p`, 320×256, 24 fps, with `+faststart` for streaming. Equivalent VP8 `.webm` files are provided as fallback sources. No video frames are invented; these are transcodes of the uploaded RGB and thermal bird/drone clips.

The RGB/IR viewer now detects media errors, reports blocked autoplay, starts both clips from a user Play gesture, updates the displayed progress bar, and allows scene switching and synchronized seeking. Existing looping paired GIFs remain.

**Important:** `index.html` must be hosted **together with** the `assets/` and `media/` directories. Opening an isolated copy of `index.html` without those directories will break the video as well as other elements.

For GitHub Pages, copy the website contents into the repository's `/docs` folder, commit, and ensure Pages is configured to serve that folder. Opening from `file://` may behave differently from GitHub Pages because browsers can restrict autoplay or local-file loading; use Play if necessary.

Validation: ffprobe recognizes H.264 (`avc1`) and VP8 for the four clips; Chromium browser simulation verified playback progresses for RGB and IR bird/drone streams, Pause/Play, scene switch, and seeking, with no page errors. This test used media embedded in a browser page because this execution environment blocks direct local navigation; site deployment remains to be checked separately.
