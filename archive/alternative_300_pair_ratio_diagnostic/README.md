# Alternative 300-pair target-size ratio diagnostic (not used in the final paper)

These are historical working CSVs bundled with the earlier paper-source archive. They compute target-relative ratios over all 300 sampled shifts, using a different matching strategy from the final manuscript. That calculation gives `B1=0.303575` and different threshold percentages.

**Do not use these values for Table/Figure 2, the abstract, or the final manuscript.** The final published definitions use the 264 frames with matched target annotations. The authoritative source is `reproducibility_data/object_relative_misalignment_rows.csv`, giving `B1=0.2647566`, rounded to `0.265`.

The raw 300 global shifts ARE still used for displacement statistics (`real_alignment_sample_300pairs.csv`), which is a distinct question from the 264 target-relative ratios.
