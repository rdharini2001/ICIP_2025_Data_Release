import numpy as np

from .late_fusion_wbf import weighted_boxes_fusion, _iou


def test_iou_identical_boxes_is_one():
    box = (0.1, 0.1, 0.3, 0.3)
    assert abs(_iou(box, box) - 1.0) < 1e-9


def test_iou_disjoint_boxes_is_zero():
    assert _iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0


def test_agreeing_detections_are_fused_into_one_higher_confidence_box():
    # RGB detector and IR detector both find "the same" drone, slightly
    # offset boxes (as expected from weak alignment), similar confidence.
    rgb_boxes = [(0.40, 0.40, 0.50, 0.50)]
    rgb_scores = [0.7]
    rgb_labels = [0]  # 0 = drone

    ir_boxes = [(0.42, 0.41, 0.52, 0.51)]
    ir_scores = [0.6]
    ir_labels = [0]

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        [rgb_boxes, ir_boxes], [rgb_scores, ir_scores], [rgb_labels, ir_labels], iou_thr=0.5
    )
    assert fused_boxes.shape == (1, 4)
    assert fused_labels[0] == 0
    # fused confidence should reflect agreement from both modalities, i.e.
    # be at least as high as the average of the two contributing scores
    assert fused_scores[0] >= (0.7 + 0.6) / 2 - 1e-6


def test_disagreeing_detections_stay_separate():
    rgb_boxes = [(0.1, 0.1, 0.2, 0.2)]
    rgb_scores = [0.9]
    rgb_labels = [0]

    ir_boxes = [(0.8, 0.8, 0.9, 0.9)]  # a bird elsewhere in frame
    ir_scores = [0.5]
    ir_labels = [1]  # 1 = bird

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        [rgb_boxes, ir_boxes], [rgb_scores, ir_scores], [rgb_labels, ir_labels], iou_thr=0.5
    )
    assert fused_boxes.shape == (2, 4)
    assert set(fused_labels.tolist()) == {0, 1}


def test_empty_inputs_return_empty_arrays():
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion([[], []], [[], []], [[], []])
    assert fused_boxes.shape == (0, 4)
    assert fused_scores.shape == (0,)


def test_modality_weighting_pulls_fused_box_toward_higher_weight_model():
    rgb_boxes = [(0.0, 0.0, 0.2, 0.2)]
    ir_boxes = [(0.10, 0.10, 0.30, 0.30)]
    fused_boxes, _, _ = weighted_boxes_fusion(
        [rgb_boxes, ir_boxes], [[0.9], [0.9]], [[0], [0]], iou_thr=0.1, weights=[5.0, 1.0]
    )
    # heavily up-weighting RGB should pull the fused box much closer to the
    # RGB box than a naive unweighted average would.
    unweighted_center = 0.15  # midpoint of the two box origins if equally weighted
    assert fused_boxes[0][0] < unweighted_center


if __name__ == "__main__":
    test_iou_identical_boxes_is_one()
    test_iou_disjoint_boxes_is_zero()
    test_agreeing_detections_are_fused_into_one_higher_confidence_box()
    test_disagreeing_detections_stay_separate()
    test_empty_inputs_return_empty_arrays()
    test_modality_weighting_pulls_fused_box_toward_higher_weight_model()
    print("All late-fusion WBF tests passed.")
