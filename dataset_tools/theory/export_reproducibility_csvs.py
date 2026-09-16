"""
Exports every real number reported in the paper as plain CSV files under
reproducibility_data/, so a reader can check any figure/table without
re-running the (sometimes multi-minute) sweep scripts themselves. Every
CSV here is a direct, non-lossy re-serialization of a JSON file already
produced by a real-data script in this repo (scale_sweep.py,
corruption_sweep.py, alignment_sweep.py, real_cross_checks.py, and the
earlier corpus/alignment/leakage measurement scripts) -- no new numbers
are computed here, only reformatted for spreadsheet-friendliness.
"""
import csv
import json
import os

METADATA_DIR = r"E:\ICIP_2025\avid_release\metadata"
OUT_DIR = r"C:\Users\dragha4\avid_repo\reproducibility_data"


def load_json(name):
    with open(os.path.join(METADATA_DIR, f"{name}.json")) as f:
        return json.load(f)


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {path} ({len(rows)} rows)")


def export_scale_sweep():
    """-> Table 'tab:bayes-scale-full', Figure 'fig:theory-scale' (Sec. 6.3)."""
    data = load_json("theory_scale_sweep")
    rows = []
    for modality in ["rgb", "ir", "fusion"]:
        for r in data[modality]:
            rows.append({
                "modality": modality, "bucket_px": r["bucket"],
                "n_bird": r["n_bird"], "n_drone": r["n_drone"],
                "skipped": r.get("skipped", False),
                "b_distance": r.get("b_distance", ""),
                "bhattacharyya_coefficient": r.get("bhattacharyya_coefficient", ""),
                "bayes_error_bound": r.get("bayes_error_bound", ""),
            })
    write_csv(f"{OUT_DIR}/scale_sweep.csv",
              ["modality", "bucket_px", "n_bird", "n_drone", "skipped",
               "b_distance", "bhattacharyya_coefficient", "bayes_error_bound"], rows)


def export_corruption_sweep():
    """-> Table 'tab:bayes-corruption-full', Figure 'fig:theory-corruption' (Sec. 6.4)."""
    data = load_json("theory_corruption_sweep")
    rows = []
    for modality in ["rgb", "ir"]:
        for key, r in data[modality].items():
            if key == "clean":
                family, severity = "clean", 0
            else:
                family, sev_str = key.rsplit("_s", 1)
                severity = int(sev_str)
            rows.append({
                "modality": modality, "corruption_family": family, "severity": severity,
                "b_distance": r["b_distance"],
                "bhattacharyya_coefficient": r["bhattacharyya_coefficient"],
                "bayes_error_bound": r["bayes_error_bound"],
            })
    rows.sort(key=lambda r: (r["modality"], r["corruption_family"], r["severity"]))
    write_csv(f"{OUT_DIR}/corruption_sweep.csv",
              ["modality", "corruption_family", "severity", "b_distance",
               "bhattacharyya_coefficient", "bayes_error_bound"], rows)


def export_alignment_bfg_sweep():
    """-> Table 'tab:bfg-full', Figure 'fig:theory-alignment', Eq. 'eq:dpi' (Sec. 6.6)."""
    data = load_json("theory_alignment_sweep")
    rows = []
    for r in data["sweep"]:
        rows.append({
            "injected_shift_px": r["injected_shift_px"],
            "rgb_bayes_bound": r["rgb_bayes_bound"],
            "ir_bayes_bound": r["ir_bayes_bound"],
            "fusion_bayes_bound": r["fusion_bayes_bound"],
            "fusion_bayes_bound_std": r["fusion_bayes_bound_std"],
            "bayes_fusion_gain": r["bayes_fusion_gain"],
            "mutual_information_nats": r["mutual_information_nats"],
            "n_trials": r["n_trials"],
        })
    write_csv(f"{OUT_DIR}/alignment_bfg_sweep.csv",
              ["injected_shift_px", "rgb_bayes_bound", "ir_bayes_bound",
               "fusion_bayes_bound", "fusion_bayes_bound_std", "bayes_fusion_gain",
               "mutual_information_nats", "n_trials"], rows)

    s = data["real_shift_distribution_summary"]
    write_csv(f"{OUT_DIR}/real_measured_shift_distribution_summary.csv",
              ["n", "mean", "median", "frac_over_5px"], [s])


def export_bayes_tightness():
    """-> Table 'tab:bayes-tightness', Eq. 'eq:chernoff' (Sec. 6.2)."""
    data = load_json("real_cross_checks")
    rows = []
    for modality, r in data.items():
        rows.append({
            "modality": modality,
            "bhattacharyya_bound_s0.5": r["bhattacharyya_bound"],
            "chernoff_bound_optimized": r["chernoff_bound"],
            "chernoff_optimal_s": r["chernoff_optimal_s"],
            "kde_nonparametric_bound": r["kde_bound"],
            "kde_n_sample": r["kde_n_sample"],
        })
    write_csv(f"{OUT_DIR}/bayes_bound_tightness_and_kde_crosscheck.csv",
              ["modality", "bhattacharyya_bound_s0.5", "chernoff_bound_optimized",
               "chernoff_optimal_s", "kde_nonparametric_bound", "kde_n_sample"], rows)


def export_corpus_stats():
    """-> Table 'tab:dataset-stats' (Sec. 3.2), abstract/intro corpus-size numbers."""
    data = load_json("corpus_summary")
    rows = []
    for split, d in data.items():
        rows.append({
            "split": split, "n_pairs": d["n_pairs"], "n_videos": d["n_videos"],
            "bird_instances": d["class_instance_counts"]["bird"],
            "drone_instances": d["class_instance_counts"]["drone"],
            "bbox_small": d["bbox_size_bucket_counts"]["small"],
            "bbox_medium": d["bbox_size_bucket_counts"]["medium"],
            "bbox_large": d["bbox_size_bucket_counts"]["large"],
        })
    write_csv(f"{OUT_DIR}/dataset_corpus_stats.csv",
              ["split", "n_pairs", "n_videos", "bird_instances", "drone_instances",
               "bbox_small", "bbox_medium", "bbox_large"], rows)


def export_bbox_histogram():
    """-> Figure 'fig:bbox-histogram', Sec. 3.4 (67.6% at 8-16px claim)."""
    data = load_json("bbox_fine_histogram")
    rows = []
    for split, d in data.items():
        for bucket, count in d["histogram_px_side"].items():
            rows.append({"split": split, "bucket_px": bucket, "n_boxes_total": d["n_boxes"], "count": count})
    write_csv(f"{OUT_DIR}/bbox_scale_histogram.csv",
              ["split", "bucket_px", "n_boxes_total", "count"], rows)


def export_alignment_sample():
    """-> Sec. 5 (per-pair alignment quantification): 300-pair real sample,
    32.3% exceed 5px claim, and the input to alignment_bfg_sweep's real
    shift distribution."""
    data = load_json("real_alignment_sample")
    rows = data["samples"]
    write_csv(f"{OUT_DIR}/real_alignment_sample_300pairs.csv",
              ["video_id", "frame_idx", "alignment_shift_px", "confidence", "disagreement_px"], rows)
    write_csv(f"{OUT_DIR}/real_alignment_sample_summary.csv",
              list(data["summary"].keys()), [data["summary"]])


def export_noise_floor_check():
    """-> Sec. 4 ("Is the 'clean' baseline actually clean?"): real intrinsic
    noise-sigma floor vs. AVID-C's injected severities."""
    data = load_json("noise_floor_check")
    rows = []
    for modality in ["RGB", "IR"]:
        d = data[modality]
        rows.append({
            "modality": modality, "n_pairs": d["n"],
            "clean_full_frame_mean_sigma": d["clean_full_frame_mean"],
            "clean_full_frame_median_sigma": d["clean_full_frame_median"],
            "clean_sky_region_mean_sigma": d["clean_sky_region_mean"],
            "clean_sky_region_median_sigma": d["clean_sky_region_median"],
            "excluded_noisy_variant_full_frame_mean_sigma": d["excluded_noisy_variant_full_frame_mean"],
            "excluded_noisy_variant_full_frame_median_sigma": d["excluded_noisy_variant_full_frame_median"],
        })
    write_csv(f"{OUT_DIR}/noise_floor_check.csv",
              ["modality", "n_pairs", "clean_full_frame_mean_sigma", "clean_full_frame_median_sigma",
               "clean_sky_region_mean_sigma", "clean_sky_region_median_sigma",
               "excluded_noisy_variant_full_frame_mean_sigma",
               "excluded_noisy_variant_full_frame_median_sigma"], rows)

    sev = data["avid_c_gaussian_severity_std"]
    write_csv(f"{OUT_DIR}/avid_c_severity_std_table.csv",
              ["severity", "gaussian_noise_std_0_255"],
              [{"severity": i + 1, "gaussian_noise_std_0_255": s} for i, s in enumerate(sev)])


def export_split_leakage():
    """-> Figure 'fig:leakage', Sec. 3.6 (87.5% vs 0.0% claim)."""
    data = load_json("split_leakage_real")
    rows = [
        {"split_method": "frame_level", "n_frames": data["n_frames"], "n_videos": data["n_videos"],
         "leaked_train_frames": data["frame_level"]["leaked_train_frames"],
         "total_train_frames": data["frame_level"]["total_train_frames"],
         "leak_fraction": data["frame_level"]["leak_fraction"]},
        {"split_method": "video_level", "n_frames": data["n_frames"], "n_videos": data["n_videos"],
         "leaked_train_frames": data["video_level"]["leaked_train_frames"],
         "total_train_frames": data["video_level"]["total_train_frames"],
         "leak_fraction": data["video_level"]["leak_fraction"]},
    ]
    write_csv(f"{OUT_DIR}/split_leakage_ablation.csv",
              ["split_method", "n_frames", "n_videos", "leaked_train_frames",
               "total_train_frames", "leak_fraction"], rows)


if __name__ == "__main__":
    export_scale_sweep()
    export_corruption_sweep()
    export_alignment_bfg_sweep()
    export_bayes_tightness()
    export_corpus_stats()
    export_bbox_histogram()
    export_alignment_sample()
    export_split_leakage()
    export_noise_floor_check()
    print("done -- all CSVs written to", OUT_DIR)
