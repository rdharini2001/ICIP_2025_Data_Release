"""
Late/decision-level RGB-IR fusion via Weighted Boxes Fusion (WBF), following
Solovyev, Wang & Gabruseva, "Weighted Boxes Fusion: Ensembling Boxes from
Different Object Detection Models," Image and Vision Computing, 2021.

Use case: train two independent single-modality detectors (RGB-only,
IR-only -- Experiment E2 in PLAN.md), run both on a pair, and combine their
detections here. This is the simplest fusion strategy to get right and the
only one of the three (early/mid/late) that needs no architecture surgery,
which is why it's implemented as pure NumPy and fully unit-tested in this
sandbox even without a GPU or the real dataset.

All boxes are (x1, y1, x2, y2) in normalized [0, 1] image coordinates.
"""

import numpy as np


def _iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=0.55, skip_box_thr=0.0, weights=None):
    """
    boxes_list / scores_list / labels_list: one entry per model (e.g.
    [rgb_detector_outputs, ir_detector_outputs]), each entry being an
    (N_i, 4) / (N_i,) / (N_i,) array-like for that model's detections on one
    image. `weights` optionally reweights each model's contribution (e.g. to
    down-weight IR at long range where thermal blooming hurts localization).

    Returns (fused_boxes, fused_scores, fused_labels) as (M, 4)/(M,)/(M,)
    arrays, M <= sum(N_i).
    """
    n_models = len(boxes_list)
    weights = np.ones(n_models) if weights is None else np.asarray(weights, dtype=np.float64)
    assert len(weights) == n_models

    # flatten all detections, tagging each with its source model index
    flat = []
    for m, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        for box, score, label in zip(boxes, scores, labels):
            if score < skip_box_thr:
                continue
            flat.append({"box": np.asarray(box, dtype=np.float64), "score": float(score), "label": label, "model": m})

    if not flat:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    # sort by score descending, greedily cluster by (label, IoU >= thr)
    flat.sort(key=lambda d: -d["score"])
    clusters = []  # each cluster: list of detections
    for det in flat:
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            if rep["label"] == det["label"] and _iou(rep["box"], det["box"]) >= iou_thr:
                cluster.append(det)
                placed = True
                break
        if not placed:
            clusters.append([det])

    fused_boxes, fused_scores, fused_labels = [], [], []
    for cluster in clusters:
        w = np.array([weights[d["model"]] * d["score"] for d in cluster])
        w_sum = w.sum()
        boxes = np.stack([d["box"] for d in cluster])
        fused_box = (boxes * w[:, None]).sum(axis=0) / w_sum
        # WBF score correction: average score scaled by how many of the
        # available models agreed (more agreement -> more confident fusion).
        avg_score = np.mean([d["score"] for d in cluster])
        n_contributing_models = len({d["model"] for d in cluster})
        fused_score = avg_score * min(1.0, n_contributing_models / n_models)
        fused_boxes.append(fused_box)
        fused_scores.append(fused_score)
        fused_labels.append(cluster[0]["label"])

    order = np.argsort(-np.array(fused_scores))
    fused_boxes = np.array(fused_boxes)[order]
    fused_scores = np.array(fused_scores)[order]
    fused_labels = np.array(fused_labels)[order]
    return fused_boxes, fused_scores, fused_labels
