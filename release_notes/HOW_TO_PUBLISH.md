# Publish the AVID visual documentation update

This package is a **complete GitHub working tree** containing the audited final three-author paper, code, CSVs, figures, research website and redesigned README. It contains no `.git` directory; it has **not** been committed or pushed.

## GitHub instructions

1. Clone `https://github.com/rdharini2001/ICIP_2025_Data_Release` on your computer. At the time of review, the default branch was `claude/rgb-ir-drone-dataset-ppc03f`. Confirm your branch before publishing.
2. Copy the **contents** of the `AVID_visual_release/` directory into your working clone; keep its `.git/` directory.
3. Review the changes, particularly `README.md`, `assets/showcase/`, and `dataset_tools/hf_release/README_HF.md`.
4. Run the CSV verification (`python paper/audit/verify_final_csvs.py`) and the alignment/corruption unit tests. Make a commit and push when satisfied.
5. If you change the GitHub default branch, revise branch-specific GitHub links in the Hugging Face dataset card. Also verify GitHub Pages continues serving `docs/`.

## Hugging Face instructions: card and GIF update only

Use the **separate** archive `AVID_HuggingFace_Card_Update.zip`, which contains:

- `README.md`: updated Hugging Face dataset card, including its existing YAML front matter;
- `media/`: the four GIFs and supporting plots;
- `upload_card_only.py`: an allowlisted upload script that modifies **only** `README.md` and new files in `media/`;
- `UPLOAD_INSTRUCTIONS.md`: local publishing directions.

**Do not upload the card-only directory as if it were a replacement for the entire dataset.** The hosted images, labels and metadata are intentionally absent from the update package. The script does not delete them.

Upload the GitHub working tree first (so paper and code links point to final material), then upload the Hugging Face card assets.

## Assets and type corrections

- User-provided `DRONE_082192_RGB_IR.gif` is animated (40 frames).
- User-provided `DRONE_RGB_IR.gif` is a byte-identical duplicate of that animation; it is not stored twice in `assets/showcase/`.
- The files named `DRONE_083193_RGB_IR.gif` and `DRONE_085195_RGB_IR.gif` in the upload are **PNG still images**, not GIF animations. They are stored with `.png` extensions.
- The repository already had the **animated** `DRONE_083193_RGB_IR.gif` and `DRONE_085195_RGB_IR.gif`, so the README uses those genuine animated sequences.
- The existing bird GIF is reused for a contrasting bird example.

For social sharing, `assets/showcase/AVID_cover.png` is a 1400×670 PNG made from real RGB/IR example frames. You can set it as the GitHub repository's social preview in repository settings; adding the PNG alone does not change that GitHub setting.

## arXiv and DOI

The AVID paper has three authors: Dharini Raghavan and Vishnu Rao (equal first authors) and Sethu Selvi S. No arXiv number or DOI has been invented. Update `README.md`, `CITATION.cff`, and both dataset cards once an actual identifier has been assigned. The source-data DOI `10.1016/j.dib.2021.107521` belongs to the *original* Drone Detection Dataset, not the AVID paper.

## Numerical integrity

The visual refresh makes **no changes** to the final manuscript's reported values, tables, or reproducibility CSVs. All key values continue to describe the 32,040 RGB–IR pairs and the 264 annotation-matched alignment subset. See `VERIFICATION_REPORT.md` for the previous 59-check audit and its limitations.
