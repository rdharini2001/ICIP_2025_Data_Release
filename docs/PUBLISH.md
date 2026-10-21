# Publishing the AVID research website

This self-contained site is in `docs/`; open `docs/index.html` directly, or enable GitHub Pages at **Settings → Pages → Deploy from a branch → default branch /docs**. The intended URL is `https://rdharini2001.github.io/ICIP_2025_Data_Release/` after deployment. The site assets include actual paired MP4s, animated RGB/IR GIFs, a text-free visual illustration, source-generated charts, the revised research PDF and `.tex`. A `.nojekyll` file is included.

Regenerate scientific data: `python docs/build_data.py && python paper/audit/check_metrics.py`.

To compile the revised paper, run `pdflatex main.tex` twice in `paper/` using the included `main.bbl` bibliography; then update `docs/paper.pdf` and `docs/paper.tex`. The remote Hugging Face paper copy also needs updating separately if the corrected edition is to replace it.
