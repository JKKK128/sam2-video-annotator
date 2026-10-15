from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sam2_annotator.correction_utils import (  # noqa: E402
    merge_track_records,
    render_range,
    validate_box_xyxy,
    validate_correction_obj_id,
    write_label_for_record,
)
from sam2_annotator.mask_utils import mask_to_bbox  # noqa: E402
from track_box import _tensor_mask_to_numpy  # noqa: E402


def test_correction_prompt_box_validation_accepts_in_bounds() -> None:
    assert validate_box_xyxy((1, 2, 5, 6), 10, 12) == (1, 2, 5, 6)


@pytest.mark.parametrize("box", [(-1, 0, 2, 2), (0, 0, 10, 2), (0, 0, 0, 2), (0, 2, 3, 2)])
def test_correction_prompt_box_validation_rejects_invalid(box: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError):
        validate_box_xyxy(box, 10, 10)


def test_correction_obj_id_must_match_initial_obj_id() -> None:
    with pytest.raises(ValueError):
        validate_correction_obj_id({"obj_id": 1}, {"obj_id": 2})


def test_old_single_object_prompt_is_compatible() -> None:
    assert validate_correction_obj_id({"obj_id": 1, "prompt_type": "box"}, {"obj_id": 1, "prompt_type": "box"}) == 1


def test_merge_keeps_baseline_before_correction() -> None:
    baseline = [_record(0, "base0"), _record(1, "base1"), _record(2, "base2")]
    corrected = [_record(1, "corr1"), _record(2, "corr2")]
    merged = merge_track_records(baseline, corrected, correction_frame=1)
    assert [record["source"] for record in merged] == ["base0", "corr1", "corr2"]


def test_merge_replaces_correction_frame_and_after() -> None:
    baseline = [_record(0, "base0"), _record(1, "base1"), _record(2, "base2")]
    corrected = [_record(1, "corr1")]
    merged = merge_track_records(baseline, corrected, correction_frame=1)
    assert [record["frame_idx"] for record in merged] == [0, 1]
    assert merged[1]["source"] == "corr1"


def test_merge_has_no_duplicate_keys_and_is_sorted() -> None:
    baseline = [_record(2, "base2"), _record(0, "base0")]
    corrected = [_record(1, "corr1"), _record(2, "corr2")]
    merged = merge_track_records(baseline, corrected, correction_frame=1)
    keys = [(record["frame_idx"], record["obj_id"]) for record in merged]
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))


def test_merge_correction_frame_zero_uses_corrected_only() -> None:
    baseline = [_record(0, "base0")]
    corrected = [_record(0, "corr0"), _record(1, "corr1")]
    merged = merge_track_records(baseline, corrected, correction_frame=0)
    assert [record["source"] for record in merged] == ["corr0", "corr1"]


def test_corrected_labels_are_generated_from_raw_bbox(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    record = {"frame_idx": 3, "obj_id": 1, "bbox_xyxy": [2, 1, 5, 4]}
    path = write_label_for_record(record, labels_dir, image_width=10, image_height=8)
    assert path.read_text(encoding="utf-8").startswith("0 0.40000000 0.37500000")


def test_empty_mask_label_generation_does_not_crash(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    record = {"frame_idx": 3, "obj_id": 1, "bbox_xyxy": None}
    path = write_label_for_record(record, labels_dir, image_width=10, image_height=8)
    assert path.read_text(encoding="utf-8") == ""


def test_render_compare_range_default_is_clipped() -> None:
    assert render_range(list(range(0, 20)), correction_frame=3) == (0, 19)
    assert render_range(list(range(10, 100)), correction_frame=20, end_frame=25) == (15, 25)


def test_bbox_directly_comes_from_binary_mask_without_stable_bbox() -> None:
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    assert mask_to_bbox(mask) == (3, 2, 6, 4)


def test_track_box_tensor_mask_uses_raw_logits_only() -> None:
    logits = np.array([[[-1.0, 0.5], [2.0, -0.1]]], dtype=np.float32)
    mask = _tensor_mask_to_numpy(logits, mask_threshold=0.0)
    assert mask.tolist() == [[False, True], [True, False]]


def test_merge_script_label_sync_uses_mask_size(tmp_path: Path) -> None:
    from merge_corrected_tracks import merge_corrected_tracks

    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), np.zeros((8, 10), dtype=np.uint8))
    baseline_path = tmp_path / "baseline.jsonl"
    corrected_path = tmp_path / "corrected.jsonl"
    baseline_path.write_text(json.dumps({"frame_idx": 0, "obj_id": 1, "mask_path": str(mask_path), "bbox_xyxy": [0, 0, 1, 1]}) + "\n", encoding="utf-8")
    corrected_path.write_text(json.dumps({"frame_idx": 1, "obj_id": 1, "mask_path": str(mask_path), "bbox_xyxy": [2, 1, 5, 4]}) + "\n", encoding="utf-8")
    output = tmp_path / "merged_tracks.jsonl"
    merge_corrected_tracks(baseline_path, corrected_path, 1, output)
    assert output.is_file()
    assert (tmp_path / "labels" / "000001.txt").is_file()


def _record(frame_idx: int, source: str) -> dict[str, object]:
    return {
        "frame_idx": frame_idx,
        "obj_id": 1,
        "source": source,
        "bbox_xyxy": [1, 1, 2, 2],
        "mask_area": 4,
        "bbox_area": 4,
    }
