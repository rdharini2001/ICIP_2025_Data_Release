import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open(r"E:\ICIP_2025\avid_release\metadata\split_leakage_real.json") as f:
    d = json.load(f)

fl = d["frame_level"]["leak_fraction"] * 100
vl = d["video_level"]["leak_fraction"] * 100

fig, ax = plt.subplots(figsize=(4.2, 3.2))
bars = ax.bar(["Frame-level\nsplit (naive)", "Video-level\nsplit (correct)"], [fl, vl], color=["#D55E00", "#009E73"])
for b, v in zip(bars, [fl, vl]):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10)
ax.set_ylabel("Train frames within 3 frames\nof a val/test frame (%)")
ax.set_ylim(0, 100)
ax.set_title(f"Split-leakage ablation (REAL corpus)\n{d['n_frames']} frames, {d['n_videos']} videos", fontsize=10)
fig.tight_layout()
fig.savefig(r"C:\Users\dragha4\avid_repo\paper\figures\fig_split_leakage.png", dpi=160)
print("wrote real fig_split_leakage.png")
