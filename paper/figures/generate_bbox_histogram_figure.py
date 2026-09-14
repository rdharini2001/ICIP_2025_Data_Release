import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open(r"E:\ICIP_2025\avid_release\metadata\bbox_fine_histogram.json") as f:
    data = json.load(f)

labels = list(data["train"]["histogram_px_side"].keys())
train_counts = [data["train"]["histogram_px_side"][l] for l in labels]
val_counts = [data["val"]["histogram_px_side"][l] for l in labels]

fig, ax = plt.subplots(figsize=(8, 4))
x = range(len(labels))
w = 0.38
ax.bar([i - w/2 for i in x], train_counts, width=w, label=f"Train (n={sum(train_counts)})", color="#0072B2")
ax.bar([i + w/2 for i in x], val_counts, width=w, label=f"Validation (n={sum(val_counts)})", color="#D55E00")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Number of annotated boxes")
ax.set_xlabel("Bounding-box side length (px, geometric mean of w,h at 320x256)")
ax.set_title("AVID bounding-box scale distribution: extreme small-object concentration")
ax.legend()
fig.tight_layout()
fig.savefig(r"C:\Users\dragha4\avid_repo\paper\figures\fig_bbox_histogram.png", dpi=160)
print("wrote fig_bbox_histogram.png")
