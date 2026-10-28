# After arXiv submission and DOI assignment

This release intentionally **does not invent or pre-populate** an arXiv ID or DOI.

1. Confirm that the final arXiv upload contains the same three-author manuscript and approved PDF/LaTeX source.
2. Add the actual arXiv URL and identifier to the top-level GitHub `README.md`, `CITATION.cff`, `paper/README.md`, the website, and the Hugging Face dataset card.
3. When a DOI is assigned, insert its *actual* value in the paper and dataset citation on GitHub and Hugging Face. Do not use the original Svanström dataset DOI as though it were the AVID manuscript DOI.
4. Replace the Hugging Face hosted `paper/main.pdf` and its dataset card only if these are intended to mirror the final arXiv version. Update the release version and note any changes.
5. Preserve correct source CC0 imagery attribution, separate package licensing, and the 300-versus-264 alignment distinction.
6. Re-run `python paper/audit/verify_final_csvs.py` and the project tests before tagging a release.

The updated Hugging Face card is prepared at `release_notes/HF_DATASET_CARD_READY.md`. This ZIP only prepares the changes locally; it does **not** publish to any remote service.
