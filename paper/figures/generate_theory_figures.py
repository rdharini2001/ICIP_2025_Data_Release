"""
Figures for the Bayes-risk theory section: real data throughout (patch
montages, scale/corruption/alignment sweeps). No synthetic stand-ins here
-- every image is a real extracted AVID patch, every curve is computed
from the real sweep JSONs produced by scale_sweep.py / corruption_sweep.py
/ alignment_sweep.py.
"""
import json
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, r"C:\Users\dragha4\avid_repo")
from dataset_tools.theory.scale_sweep import BIN_LABELS

OUT_DIR = r"C:\Users\dragha4\avid_repo\paper\figures"
PALETTE = {"rgb": "#0072B2", "ir": "#D55E00", "fusion": "#009E73"}


def load_json(name):
    with open(rf"E:\ICIP_2025\avid_release\metadata\{name}.json") as f:
        return json.load(f)


def load_patches():
    with open(r"E:\ICIP_2025\avid_release\metadata\theory_patches.pkl", "rb") as f:
        return pickle.load(f)


def fig_patch_montage():
    """Real bird/drone patches, RGB+IR, across scale buckets -- gives the
    reader a direct, qualitative sense of what the density estimator in
    bayes_bounds.py actually sees at each operating point."""
    patches = load_patches()
    rng = np.random.default_rng(0)

    target_buckets = ["8-12", "12-16", "16-20", "20-24"]
    n_cols = len(target_buckets)
    fig, axes = plt.subplots(4, n_cols, figsize=(2.0 * n_cols, 8.4))

    from dataset_tools.theory.scale_sweep import bin_index, BIN_LABELS

    for col, bucket in enumerate(target_buckets):
        bucket_idx = BIN_LABELS.index(bucket)
        for row, (cls_name, modality) in enumerate([("bird", "rgb"), ("bird", "ir"), ("drone", "rgb"), ("drone", "ir")]):
            candidates = [p for p in patches if p["cls_name"] == cls_name and bin_index(p["bbox_side_px"]) == bucket_idx]
            ax = axes[row, col]
            if candidates:
                p = candidates[rng.integers(len(candidates))]
                cmap = "gray" if modality == "rgb" else "inferno"
                ax.imshow(p[f"{modality}_patch"], cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{bucket}px", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{cls_name}\n({modality.upper()})", fontsize=9, rotation=0, ha="right", va="center")
    fig.suptitle("Real AVID patches by object scale (rows: class x modality, columns: pixel-side bucket)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT_DIR}/fig_theory_patch_montage.png", dpi=160)
    plt.close(fig)
    print("wrote fig_theory_patch_montage.png")


def fig_scale_sweep():
    data = load_json("theory_scale_sweep")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 1]})

    for modality in ["rgb", "ir", "fusion"]:
        rows = [r for r in data[modality] if not r.get("skipped")]
        x = [r["bucket"] for r in rows]
        y = [r["bayes_error_bound"] for r in rows]
        ax1.plot(x, y, "o-", label=modality.upper(), color=PALETTE[modality], linewidth=2)
    ax1.set_xlabel("Object side length (px)")
    ax1.set_ylabel("Bayes error bound (Bhattacharyya)")
    ax1.set_title("Theoretical bird/drone separability vs. object scale")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)
    ax1.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance")

    counts = [r["n_bird"] + r["n_drone"] for r in data["rgb"]]
    labels = [r["bucket"] for r in data["rgb"]]
    ax2.barh(labels, counts, color="#888888")
    ax2.set_xlabel("# real instances")
    ax2.set_title("Corpus scale distribution", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_theory_scale_sweep.png", dpi=160)
    plt.close(fig)
    print("wrote fig_theory_scale_sweep.png")


def fig_corruption_sweep():
    data = load_json("theory_corruption_sweep")
    corruption_names = sorted({k.rsplit("_s", 1)[0] for k in data["rgb"] if k != "clean"})

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey=True)
    axes = axes.flatten()
    for i, cname in enumerate(corruption_names):
        ax = axes[i]
        for modality in ["rgb", "ir"]:
            clean = data[modality]["clean"]["bayes_error_bound"]
            xs = [0] + list(range(1, 6))
            ys = [clean] + [data[modality][f"{cname}_s{s}"]["bayes_error_bound"] for s in range(1, 6)]
            ax.plot(xs, ys, "o-", color=PALETTE[modality], label=modality.upper())
        ax.set_title(cname.replace("_", " "), fontsize=9)
        ax.set_xlabel("severity")
        ax.axhline(0.5, color="gray", ls="--", lw=0.6)
    axes[0].set_ylabel("Bayes error bound")
    axes[0].legend(fontsize=8)
    for j in range(len(corruption_names), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Theoretical corruption-robustness ceiling (real AVID patches, AVID-C severities)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT_DIR}/fig_theory_corruption_sweep.png", dpi=160)
    plt.close(fig)
    print("wrote fig_theory_corruption_sweep.png")


def fig_alignment_sweep():
    data = load_json("theory_alignment_sweep")
    rows = data["sweep"]
    shifts = [r["injected_shift_px"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(shifts, [r["rgb_bayes_bound"] for r in rows], "o-", color=PALETTE["rgb"], label="RGB only")
    ax1.plot(shifts, [r["ir_bayes_bound"] for r in rows], "o-", color=PALETTE["ir"], label="IR only")
    ax1.plot(shifts, [r["fusion_bayes_bound"] for r in rows], "o-", color=PALETTE["fusion"], label="Fusion (joint)", linewidth=2.5)
    real_frac = data["real_shift_distribution_summary"]["frac_over_5px"]
    ax1.axvspan(5, max(shifts), color="red", alpha=0.08, label=f"real: {real_frac*100:.0f}% of pairs here")
    ax1.set_xlabel("Injected cross-modal misalignment (px)")
    ax1.set_ylabel("Bayes error bound")
    ax1.set_title("Fusion's theoretical advantage vanishes as\nreal misalignment increases")
    ax1.legend(fontsize=8)

    ax2.plot(shifts, [r["mutual_information_nats"] for r in rows], "o-", color="#333333", linewidth=2)
    ax2.set_xlabel("Injected cross-modal misalignment (px)")
    ax2.set_ylabel("$I(Y; X_{RGB}, X_{IR})$ (nats)")
    ax2.set_title("Data-processing-inequality degradation\n(real Gaussian-mixture MI estimate)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_theory_alignment_sweep.png", dpi=160)
    plt.close(fig)
    print("wrote fig_theory_alignment_sweep.png")


if __name__ == "__main__":
    fig_patch_montage()
    fig_scale_sweep()
    fig_corruption_sweep()
    fig_alignment_sweep()
