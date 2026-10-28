# AVID final-paper verification report (2026-10-28)

## Inputs and scope

- Author-supplied **28-page final PDF** (`drone_bird_dataset_paper (4).pdf`).
- Author-supplied **LaTeX/source archive** (`drone_bird_dataset_paper.zip`).
- Author-supplied **code repository archive** (`ICIP_2025_Data_Release.zip`).
- Public GitHub repository and Hugging Face dataset page were read for metadata and existing descriptions.

This audit validates the supplied **summary and per-sample CSV files** and checks the final PDF, manuscript source, figures and citations. **It does not independently recompute every result from the full original RGB/infrared image corpus**: those image files and the precomputed patch matrix are not included in the uploaded archives, and the complete Hugging Face image collection was not downloaded for this audit. We therefore distinguish *CSV consistency* from *raw-image reproducibility*.

## Paper checks

- PDF: 28 pages, all pages visually inspected using a contact sheet; no missing-page or missing-figure issues observed.
- Final LaTeX source: independently compiled to 28 pages. Extracted text matches the uploaded PDF page by page (minor text-extraction differences only in math glyph rendering).
- Exactly one `\begin{document}`, one `\maketitle`, and one `\end{document}`.
- 30 distinct `\citep{}` / citation keys, all present in the 30-entry bibliography; BibTeX and two LaTeX passes produced no undefined citations.
- 44 LaTeX labels, no duplicates, and no unresolved `\ref{}`/`\eqref{}` values.
- Eight `\includegraphics` references, all assets present. Additional vector/raster figure copies are included for regeneration.
- Authorship: Dharini Raghavan and Vishnu Rao share first authorship; Sethu Selvi S is the third author; all three appear in the final PDF and citation metadata.

## Numerical checks (against bundled CSVs)

| Paper quantity | Independently calculated from provided CSVs | Result |
|---|---:|---|
| Train / Validation frame pairs | 28,790 / 3,250 | MATCH |
| Total frame pairs / videos | 32,040 / 115 | MATCH |
| Bird / drone instances | 15,471 / 20,481 | MATCH |
| Total annotations | 35,952 | MATCH |
| COCO small / medium counts | 35,581 / 371 | MATCH |
| Targets with size 8–16 px | 24,489 = 68.115% | MATCH (68.1%) |
| Targets in 12–16 px | 17,706 | MATCH |
| Raw alignment frame pairs | 300 | MATCH |
| Shift >5 px among 300 | 97/300 = 32.333% | MATCH (32.3%) |
| Target-annotation matches | 264; 36 excluded | MATCH |
| Ratios >0.25, >0.50, >1 among 264 | 72, 69, 43 | MATCH (27.3%, 26.1%, 16.3%) |
| Unit-scale burden among 264 | 0.2647566467 | MATCH (0.265) |
| 12–16 px Bhat: RGB / IR / paired | 0.148394 / 0.197933 / 0.045716 | MATCH |
| 12–16 px Chernoff: RGB / IR / paired | 0.131410 / 0.194409 / 0.039554 | MATCH |
| 12–16 px KDE: RGB / IR / paired | 0.164996 / 0.177372 / 0.047218 | MATCH |
| Equal-total-dimension 4+4 paired bound | 0.139600 | MATCH |
| Corruption clean RGB / IR | 0.147694 / 0.162318 | MATCH |
| Mean severity-5 penalty RGB / IR | 0.132477 / 0.051517 | MATCH |
| Noise-floor ratios at severity 1 | 16.56× RGB / 7.83× IR | MATCH |
| Shift-budget retention at 12 px/axis | 0.844447 | MATCH (0.844) |

A machine-checkable implementation is in `paper/audit/verify_final_csvs.py`. The evaluation code's eight existing automated alignment/corruption tests passed.

## Issues detected and corrected in the packaged repository

1. **Old paper and incorrect author list:** replaced `paper/main.tex`, `paper/main.pdf`, bibliography and figures; updated the website paper copies, README, citation, and HF draft to the final **three-author** version.
2. **Two incompatible alignment calculations:** the uploaded source archive included 300-row target-relative CSVs (`B1≈0.303575`) while the paper and code repository use the correct 264 annotation-matched rows (`B1≈0.264757`). The released root and paper CSVs now consistently use **264** for target-relative metrics, and the old 300-relative diagnostics are archived under `archive/alternative_300_pair_ratio_diagnostic/` with an explicit warning. The 300 original global-shift measurements remain unchanged.
3. **Figure regeneration bug:** the earlier `paper/analysis/make_paper_artifacts.py` plotted the all-300 burden and wrote 0.304 even though the final paper uses 0.265. It now reads the 300 raw translations separately from the **264** target-matched rows and renders the 264 thresholds, eliminating that mismatch.
4. **Stale README claims:** removed assertions of demonstrated detector accuracy, incorrect 17-pixel-per-axis interpretation, blanket claims of co-registration, and stale two-author temporary citations.
5. **Stale website mirrors:** final PDF/LaTeX is now in `docs/` and `website/`; authors and on-page summary data were aligned.
6. **Source and license attribution:** added explicit citation and CC0/CC BY-NC 4.0/MIT separation.
7. **Hugging Face card:** prepared an updated draft containing the third author, correct 264-pair scope, and source-data attribution. The remote dataset card has **not** been modified.

## Remaining checks before public submission

- **Raw-data rerun not performed.** Download the published AVID frames/labels and re-run `dataset_tools/` if independent validation from image pixels is needed, including image-level pairing and annotation counts.
- **Reference bibliographic truth:** citation keys were validated and the paper compiles, but the 30 papers' publisher metadata were not individually verified against external publisher records.
- **Other benchmark dataset descriptions:** summary descriptions are not independently verified against all external datasets.
- **Hugging Face file mirroring:** this ZIP does not change the hosted Hugging Face PDF/card; those need a separate update after the GitHub changes are published.
- **The arXiv preprint/DOI does not yet exist.** Update the real identifiers once available, without inventing placeholders.
- **No remote repositories were pushed or changed.** This is a prepared, complete local repository archive for user review and upload.
