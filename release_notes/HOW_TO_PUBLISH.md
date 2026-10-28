# Publishing the final AVID repository

This ZIP contains the complete updated **working tree**, without its historical `.git/` directory or local test caches. It has **not** been pushed to GitHub or Hugging Face.

## GitHub

1. Review `VERIFICATION_REPORT.md` and the final PDF. In particular, confirm that the final author list and 264-pair target-relative alignment convention are approved.
2. On your computer, clone `https://github.com/rdharini2001/ICIP_2025_Data_Release` and check out the branch on which you want to publish. The GitHub connector showed the repository contents on `claude/rgb-ir-drone-dataset-ppc03f` at the time of the audit; **do not assume `main` exists or is the release branch**.
3. Copy the **contents** of `AVID_final_release/` into the checked-out repository (preserve `.git/` from your clone, do not copy build caches).
4. Run `python paper/audit/verify_final_csvs.py`, the eight alignment/corruption tests, and compile `paper/main.tex` before publishing.
5. Review the diff with `git diff --stat` and `git diff`, commit the intended changes, and push them to your selected branch. Configure GitHub Pages from `docs/` as appropriate.

## Hugging Face

The remote Hugging Face dataset was **inspected but not modified**. The live card currently lists the outdated two-author paper and needs the third author and updated source/CC0 explanation. Use `release_notes/HF_DATASET_CARD_READY.md` as the proposed replacement for the dataset's `README.md` **after reviewing its file paths and links**. Update the dataset's mirrored `paper/main.pdf` only if you intend the hosted PDF to match the three-author final version.

## After arXiv/DOI

Follow `release_notes/AFTER_ARXIV_AND_DOI.md`. Do not claim a paper DOI or arXiv identifier before assignment. The original upstream Data in Brief DOI belongs only to the original source dataset.
