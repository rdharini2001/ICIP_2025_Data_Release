"""
Generates the paper's remaining schematic/illustrative figures -- the ones
that are legitimately synthetic by design (fig_alignment_illustration.png
needs a known ground-truth offset to validate the estimator against, and
fig_landscape.png is a qualitative dataset-positioning schematic, not a
data plot). The real-data equivalents of this file's two former functions
(fig_corruption_severity_grid, fig_split_leakage) have been removed: they
generated synthetic stand-in images that are now superseded by
generate_real_figures.py and generate_real_leakage_figure.py, which write
to the same output filenames using real AVID data -- keeping the old
synthetic versions here would have silently overwritten the real figures
on a re-run. Run with:

    python paper/figures/generate_figures.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

OUT_DIR = os.path.dirname(__file__)

# a colorblind-safe qualitative palette (Okabe-Ito)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]


def fig_alignment_illustration():
    from dataset_tools.alignment.test_alignment_metrics import _make_pair
    from dataset_tools.alignment.alignment_metrics import estimate_shift_mutual_information

    true_dx, true_dy = 6, -4
    rgb, ir = _make_pair(true_dx, true_dy)
    est = estimate_shift_mutual_information(rgb, ir, max_shift=10)
    # est reports the *correction* needed to align IR onto RGB, which is the
    # negative of the injected IR-relative-to-RGB offset by construction;
    # negate it back for the caption so it reads as a direct, same-sign
    # comparison to the injected offset rather than requiring the reader to
    # know that sign convention.
    recovered_dx, recovered_dy = -est["dx"], -est["dy"]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB (synthetic)", fontsize=9)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].imshow(ir, cmap="inferno")
    axes[1].set_title("IR (synthetic, offset injected)", fontsize=9)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.suptitle(
        f"Weak-alignment estimation: injected IR offset ({true_dx}, {true_dy})px, "
        f"recovered ({recovered_dx:.0f}, {recovered_dy:.0f})px via NMI search",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUT_DIR, "fig_alignment_illustration.png"), dpi=160)
    plt.close(fig)


def fig_landscape():
    # qualitative positioning of related datasets on two axes established in
    # the related-work research pass: (x) is the target object drone-vs-not,
    # (y) is RGB-T paired vs single-modality. Purely illustrative/schematic.
    points = {
        "LLVIP": (0.05, 0.95, PALETTE[0]),
        "M3FD": (0.05, 0.90, PALETTE[0]),
        "RoadScene": (0.05, 0.85, PALETTE[0]),
        "KAIST": (0.05, 0.80, PALETTE[0]),
        "FLIR ADAS": (0.05, 0.75, PALETTE[0]),
        "Anti-UAV / 410 / 600": (0.90, 0.10, PALETTE[1]),
        "CST Anti-UAV": (0.90, 0.15, PALETTE[1]),
        "DroneRGBT": (0.15, 0.55, PALETTE[4]),
        "VisDrone": (0.15, 0.05, PALETTE[4]),
        "VTUAV": (0.15, 0.60, PALETTE[4]),
        "AVID (this work)": (0.90, 0.90, PALETTE[2]),
    }
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (x, y, c) in points.items():
        marker = "*" if "this work" in name else "o"
        size = 260 if "this work" in name else 90
        ax.scatter(x, y, s=size, color=c, marker=marker, edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(0.5, color="gray", lw=0.6, ls="--")
    ax.axvline(0.5, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("Target object is a drone  →")
    ax.set_ylabel("RGB + thermal, paired  →")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Where AVID sits relative to prior RGB-T and drone datasets\n(schematic, from Table 1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_landscape.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_alignment_illustration()
    fig_landscape()
    print("Wrote figures to", OUT_DIR)
