"""
Graphical-abstract-style concept diagram contrasting the classical
evaluation paradigm (train one model, measure its error, confounded by
architecture/budget/luck) against this paper's classifier-agnostic
Bayes-risk paradigm (compute a bound on the optimal classifier's error
directly from data, no training). Pure schematic -- no real data plotted
here; the real data results are in the other theory figures. This is
meant to be read in ~5 seconds and communicate the paper's central
methodological move before the reader hits any equation.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT_DIR = r"C:\Users\dragha4\avid_repo\paper\figures"


def box(ax, xy, w, h, text, facecolor, edgecolor="#333333", fontsize=9.5, textcolor="black"):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                            linewidth=1.3, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             color=textcolor, wrap=True)
    return patch


def arrow(ax, start, end, color="#333333", style="-|>", lw=1.6, connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14, linewidth=lw,
                         color=color, connectionstyle=connectionstyle)
    ax.add_patch(a)


def fig_concept_diagram():
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    RED = "#f6d6d6"
    GREEN = "#d6f0e0"
    GRAY = "#eeeeee"
    BLUE = "#dbe9f8"

    # ---- Top track: classical paradigm ----
    ax.text(0.15, 6.05, "Classical evaluation (every prior RGB-T benchmark)", fontsize=11.5,
            fontweight="bold", color="#a83232")
    box(ax, (0.15, 4.9), 2.1, 0.9, "Weakly-paired\nRGB-IR data", GRAY)
    box(ax, (2.7, 4.9), 2.3, 0.9, "Train ONE model\n(architecture $A$,\nbudget $B$)", RED)
    box(ax, (5.5, 4.9), 2.3, 0.9, "Measure its\nempirical error", RED)
    box(ax, (8.3, 4.9), 2.4, 0.9, "Report: confounded\nby $A$, $B$, and\noptimization luck", RED,
        fontsize=8.8)
    arrow(ax, (2.25, 5.35), (2.7, 5.35))
    arrow(ax, (5.0, 5.35), (5.5, 5.35))
    arrow(ax, (7.8, 5.35), (8.3, 5.35))
    ax.text(9.5, 4.55, "A different $A$ or $B$\ncould show a different story\n"
            "-- this number alone can't tell you", fontsize=8, color="#a83232",
            ha="center", style="italic")

    # ---- Divider ----
    ax.plot([0.15, 10.7], [4.15, 4.15], color="#bbbbbb", lw=1, ls="--")
    ax.text(5.4, 3.95, "this paper's move", fontsize=9, color="#555555", ha="center", style="italic")

    # ---- Bottom track: this paper ----
    ax.text(0.15, 3.55, "Classifier-agnostic Bayes-risk benchmark (this paper, "
            r"$\S$6)", fontsize=11.5, fontweight="bold", color="#1a7a4c")
    box(ax, (0.15, 2.35), 2.1, 0.9, "Weakly-paired\nRGB-IR data", GRAY)
    box(ax, (2.7, 2.35), 2.3, 0.9, "Fit class-conditional\ndistributions\n"
        "$p(x\\mid\\mathrm{bird})$, $p(x\\mid\\mathrm{drone})$", BLUE, fontsize=8.6)
    box(ax, (5.5, 2.35), 2.3, 0.9, "Closed-form\nBhattacharyya /\nChernoff bound", GREEN)
    box(ax, (8.3, 2.35), 2.4, 0.9, "Report: a ceiling on\nEVERY possible\nclassifier, no training",
        GREEN, fontsize=8.8)
    arrow(ax, (2.25, 2.8), (2.7, 2.8))
    arrow(ax, (5.0, 2.8), (5.5, 2.8))
    arrow(ax, (7.8, 2.8), (8.3, 2.8))
    ax.text(9.5, 1.95, "Same bound regardless of\nwhich architecture or budget\n"
            "anyone ever uses", fontsize=8, color="#1a7a4c", ha="center", style="italic")

    # ---- Shared input, diverging arrows ----
    arrow(ax, (1.2, 4.9), (1.2, 3.25), color="#888888", lw=1.2, style="-",
          connectionstyle="arc3,rad=0.0")

    # ---- Bottom callouts: the three real perturbations ----
    ax.text(0.15, 1.35, "Applied to three real, controlled perturbations of the same corpus:",
            fontsize=9.5, fontweight="bold")
    labels = ["Object scale\n(Fig. 7)", "AVID-C corruption\nseverity (Fig. 8-9)",
              "Cross-modal\nmisalignment (Fig. 11-12)"]
    xs = [0.3, 3.9, 7.5]
    for x, label in zip(xs, labels):
        box(ax, (x, 0.15), 3.2, 0.85, label, "#fff3d6", edgecolor="#c9a227", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_concept_diagram.png", dpi=170)
    plt.close(fig)
    print("wrote fig_concept_diagram.png")


if __name__ == "__main__":
    fig_concept_diagram()
