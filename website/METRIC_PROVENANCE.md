# AVID paper, repository and Hugging Face numerical audit

The supplied ZIP, repository CSV exports and public release descriptions agree on the principal corpus statistics: **32,040 paired frames**, **115 videos**, **35,952 object instances** (15,471 bird, 20,481 drone), **68.1% of targets with 8–16 px geometric-mean side**, and **seven AVID-C families each with five levels**. The 300-pair global displacement sample has 97 shifts >5 px, yielding 32.3%.

## Important reconciliation

The earlier manuscript used the 300 measured-shift pairs as the denominator for target-relative alignment, although the available label join contains **264 rows** and **36 unmatched annotations**. The corrected paper and website use the released records for target-relative results:

| Quantity | Earlier paper | Released CSV / corrected edition |
| --- | ---: | ---: |
| ρ>0.25 | 33.0% | 72/264 = **27.3%** |
| ρ>0.5 | 29.7% | 69/264 = **26.1%** |
| ρ>1 | 16.7% | 43/264 = **16.3%** |
| Truncated unit burden | 0.304 | **0.265** |
| Median target side | 14.07 px | **14.18 px** |
| Number of global shift estimates | 300 | **300** (unchanged) |

The accompanying paper's alignment figure and appendix have been rebuilt to follow these source rows. The hosted Hugging Face dataset card already uses 16.3% for ρ>1, but does not always clarify that this is the **264 annotation-matched subset**. The public HF card and paper have **not** been remotely overwritten as part of this package.

The 12–16 px Bhattacharyya upper bounds agree to rounding across the original paper and the released `bayes_bound_tightness_and_kde_crosscheck.csv`: RGB **0.148**, IR **0.198**, RGB+IR **0.046**. The optimized Chernoff bounds are **0.131, 0.194, 0.040**, and the KDE cross-checks **0.165, 0.177, 0.047**. The `alignment_bfg_sweep.csv` reports fusion-gap retention **0.844** at an injected 12-pixel-per-axis shift and MI approximately **0.647 nats**. Each corruption-family curve on the site is read from `corruption_sweep.csv`; no chart describes detector accuracy.

## Provenance and limitations

- Local numerical audits: `python paper/audit/check_metrics.py`, `python docs/build_data.py`.
- Source: `reproducibility_data/` in the supplied repository; source paper `paper/main.tex`.
- Public GitHub: https://github.com/rdharini2001/ICIP_2025_Data_Release
- Public Hugging Face: https://huggingface.co/datasets/Dharini24/AVID
- The public HF **dataset description** was inspected. The full remote image corpus was not re-downloaded and rerun in this audit.
- All other paper sections, equations, figures and discussion were retained during light editorial polishing.
