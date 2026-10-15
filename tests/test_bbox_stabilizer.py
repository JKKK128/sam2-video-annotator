from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from sam2_annotator.anomaly_detector import evaluate_tracks, observation_from_record, AnomalyParameters
from sam2_annotator.bbox_stabilizer import BBoxStabilizer, StabilizerParameters, clean_mask_components, parameters_from_mapping
from sam2_annotator.mask_utils import mask_to_bbox

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from track_box import _load_confirmed_keys, _write_yolo_label  # noqa: E402


def test_isolated_pixel_does_not_expand_cleaned_bbox() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:8, 5:8] = 1
    mask[18, 18] = 1

    cleaned, info = clean_mask_components(mask, min_component_area=3)

    assert mask_to_bbox(cleaned) == (5, 5, 7, 7)
    assert info.component_count == 2
    assert info.removed_component_count == 1


def test_multiple_components_selects_nearest_previous_center() -> None:
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    mask[28:34, 28:34] = 1

    cleaned, info = clean_mask_components(mask, previous_center=(31.0, 31.0), recent_area=36)

    assert mask_to_bbox(cleaned) == (28, 28, 33, 33)
    assert info.selected_component_area == 36
    assert info.valid_component_count == 2
    assert info.selected_component_center == (30.5, 30.5)


def test_without_history_selects_largest_component() -> None:
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    mask[20:28, 20:28] = 255

    cleaned, info = clean_mask_components(mask, min_component_area=3)

    assert mask_to_bbox(cleaned) == (20, 20, 27, 27)
    assert info.selected_component_area == 64


def test_first_frame_initializes_stable_bbox_and_preserves_raw() -> None:
    stabilizer = BBoxStabilizer()

    result = stabilizer.update((1, 2, 4, 5), (20, 20))

    assert result.bbox_status == "initialized"
    assert result.raw_bbox_xyxy == (1, 2, 4, 5)
    assert result.stable_bbox_xyxy == (1, 2, 4, 5)


def test_normal_motion_center_follows() -> None:
    stabilizer = BBoxStabilizer(StabilizerParameters(center_alpha=0.5, minimum_center_jump_pixels=50.0))
    stabilizer.update((10, 10, 19, 19), (100, 100))

    result = stabilizer.update((12, 10, 21, 19), (100, 100))

    assert result.bbox_status == "smoothed"
    assert result.stable_center_xy == (15.5, 14.5)


def test_width_height_jitter_is_reduced() -> None:
    stabilizer = BBoxStabilizer(StabilizerParameters(size_alpha=0.25, minimum_center_jump_pixels=50.0))
    stabilizer.update((10, 10, 19, 19), (100, 100))

    result = stabilizer.update((10, 10, 21, 19), (100, 100))

    assert result.raw_width == 12
    assert result.stable_width is not None
    assert result.stable_width < result.raw_width


def test_true_center_jump_is_unstable() -> None:
    stabilizer = BBoxStabilizer(StabilizerParameters(minimum_center_jump_pixels=8.0))
    for i in range(5):
        stabilizer.update((10 + i, 10, 19 + i, 19), (200, 200))

    result = stabilizer.update((100, 100, 109, 109), (200, 200))

    assert result.bbox_status == "unstable"
    assert "center_outlier" in result.bbox_stability_reasons


def test_true_size_jump_is_unstable() -> None:
    stabilizer = BBoxStabilizer(StabilizerParameters(minimum_center_jump_pixels=50.0))
    for _ in range(5):
        stabilizer.update((10, 10, 19, 19), (200, 200))

    result = stabilizer.update((5, 5, 34, 34), (200, 200))

    assert result.bbox_status == "unstable"
    assert "size_outlier" in result.bbox_stability_reasons


def test_empty_mask_does_not_crash() -> None:
    stabilizer = BBoxStabilizer()

    result = stabilizer.update(None, (20, 20))

    assert result.bbox_status == "missing"
    assert result.stable_bbox_xyxy is None


def test_clean_empty_mask_returns_diagnostics() -> None:
    cleaned, info = clean_mask_components(np.zeros((5, 5), dtype=np.uint8), min_component_area=3)

    assert cleaned.sum() == 0
    assert info.component_count == 0
    assert info.valid_component_count == 0
    assert info.selected_component_area == 0
    assert info.selected_component_center is None


def test_obj_histories_can_be_independent() -> None:
    obj1 = BBoxStabilizer(StabilizerParameters(minimum_center_jump_pixels=8.0))
    obj2 = BBoxStabilizer(StabilizerParameters(minimum_center_jump_pixels=8.0))
    obj1.update((10, 10, 19, 19), (200, 200))
    obj2.update((100, 100, 109, 109), (200, 200))

    result1 = obj1.update((12, 10, 21, 19), (200, 200))
    result2 = obj2.update((102, 100, 111, 109), (200, 200))

    assert result1.bbox_status != "unstable"
    assert result2.bbox_status != "unstable"
    assert result1.stable_center_xy != result2.stable_center_xy


def test_bbox_is_clipped_to_image_bounds() -> None:
    stabilizer = BBoxStabilizer()

    result = stabilizer.update((-5, -5, 30, 30), (20, 10))

    assert result.raw_bbox_xyxy == (0, 0, 19, 9)
    assert result.stable_bbox_xyxy == (0, 0, 19, 9)


def test_raw_bbox_is_preserved_when_candidate_is_cleaned() -> None:
    stabilizer = BBoxStabilizer()

    result = stabilizer.update((1, 1, 18, 18), (20, 20), candidate_bbox_xyxy=(5, 5, 7, 7))

    assert result.raw_bbox_xyxy == (1, 1, 18, 18)
    assert result.stable_bbox_xyxy == (5, 5, 7, 7)


def test_invalid_config_reports_clear_error() -> None:
    import pytest

    with pytest.raises(ValueError, match="center_alpha"):
        parameters_from_mapping({"center_alpha": 1.5})
    with pytest.raises(ValueError, match="history_size"):
        parameters_from_mapping({"history_size": 0})
    with pytest.raises(ValueError, match="min_component_area"):
        parameters_from_mapping({"min_component_area": 0})
    with pytest.raises(ValueError, match="max_size_ratio"):
        parameters_from_mapping({"max_size_ratio": 0})


def test_yolo_export_uses_stable_bbox(tmp_path: Path) -> None:
    label_path = tmp_path / "000000.txt"

    _write_yolo_label(label_path, (10, 10, 19, 19), 100, 100)

    assert label_path.read_text(encoding="utf-8").startswith("0 0.15000000 0.15000000")


def test_confirmed_keys_only_accept_confirmed_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "confirmed.jsonl"
    path.write_text(
        '{"frame_idx": 1, "obj_id": 1, "label_status": "confirmed"}\n'
        '{"frame_idx": 2, "obj_id": 1, "label_status": "working_unconfirmed"}\n',
        encoding="utf-8",
    )

    assert _load_confirmed_keys(path) == {(1, 1)}


def test_anomaly_detection_uses_raw_bbox_when_stable_exists() -> None:
    records = []
    for idx in range(5):
        records.append(
            {
                "frame_idx": idx,
                "obj_id": 1,
                "mask_area": 80,
                "bbox_area": 100,
                "raw_bbox_xyxy": [10 + idx, 10, 19 + idx, 19],
                "stable_bbox_xyxy": [10 + idx, 10, 19 + idx, 19],
            }
        )
    records.append(
        {
            "frame_idx": 5,
            "obj_id": 1,
            "mask_area": 80,
            "bbox_area": 100,
            "raw_bbox_xyxy": [100, 10, 109, 19],
            "stable_bbox_xyxy": [15, 10, 24, 19],
        }
    )
    observations = [observation_from_record(record) for record in records]

    result = evaluate_tracks(observations, AnomalyParameters(history_size=5), image_size=(200, 100))[-1]

    assert "center_jump" in result.reasons
    assert result.raw_bbox_xyxy == [100, 10, 109, 19]
    assert result.stable_bbox_xyxy == [15, 10, 24, 19]
