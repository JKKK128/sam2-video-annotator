from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sam2_annotator.anomaly_detector import AnomalyParameters, TrackObservation, evaluate_tracks  # noqa: E402
from sam2_annotator.correction_utils import merge_track_records  # noqa: E402
from sam2_annotator.mask_utils import mask_to_bbox  # noqa: E402
from sam2_annotator.object_registry import validate_registry  # noqa: E402
from sam2_annotator.tracking_io import (  # noqa: E402
    collect_mask_links,
    convert_prompt_to_segment,
    ensure_unique_track_keys,
    local_to_original_map,
    write_multi_object_labels,
)
from detect_anomalies import apply_duplicate_target_warnings  # noqa: E402
from render_multi_overlay import group_tracks  # noqa: E402


def test_registry_obj_id_must_be_unique() -> None:
    registry = _registry(
        [
            {"obj_id": 1, "name": "drone_1", "first_prompt_frame": 0, "status": "active"},
            {"obj_id": 1, "name": "drone_2", "first_prompt_frame": 1, "status": "active"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate obj_id"):
        validate_registry(registry)


def test_registry_name_must_be_unique() -> None:
    registry = _registry(
        [
            {"obj_id": 1, "name": "drone", "first_prompt_frame": 0, "status": "active"},
            {"obj_id": 2, "name": "drone", "first_prompt_frame": 1, "status": "active"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate object name"):
        validate_registry(registry)


def test_global_prompt_converts_to_local_prompt() -> None:
    manifest = _manifest(203, 205)
    prompt = {"obj_id": 1, "object_name": "drone_1", "frame_idx": 204, "original_frame_idx": 204, "prompt_type": "box"}
    converted = convert_prompt_to_segment(prompt, manifest)
    assert converted["frame_idx"] == 1
    assert converted["local_frame_idx"] == 1
    assert converted["original_frame_idx"] == 204


def test_local_global_frame_mapping() -> None:
    assert local_to_original_map(_manifest(10, 12)) == {0: 10, 1: 11, 2: 12}


def test_single_object_track_record_uses_global_mask_name() -> None:
    record = _record(203, 1)
    assert record["frame_idx"] == 203
    assert record["original_frame_idx"] == 203
    assert record["local_frame_idx"] == 0
    assert Path(str(record["mask_path"])).name == "000203_obj1.png"


def test_three_active_objects_are_rejected() -> None:
    records = [_record(418, 1), _record(418, 2), _record(418, 3)]
    registry = _registry([_object(1), _object(2), _object(3)])
    with pytest.raises(ValueError, match="at most 2"):
        validate_registry(registry)


def test_merge_rejects_duplicate_frame_obj_key() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ensure_unique_track_keys([_record(1, 1), _record(1, 1)])


def test_same_frame_two_yolo_labels_are_not_overwritten(tmp_path: Path) -> None:
    records = [_record(5, 1, bbox=[0, 0, 1, 1]), _record(5, 2, bbox=[2, 0, 3, 1])]
    write_multi_object_labels(records, tmp_path, 10, 10)
    assert len((tmp_path / "000005.txt").read_text(encoding="utf-8").strip().splitlines()) == 2


def test_one_object_frame_writes_one_label(tmp_path: Path) -> None:
    write_multi_object_labels([_record(6, 2, bbox=[0, 0, 1, 1])], tmp_path, 10, 10)
    assert len((tmp_path / "000006.txt").read_text(encoding="utf-8").strip().splitlines()) == 1


def test_full_frame_range_creates_empty_txt_before_object_entry(tmp_path: Path) -> None:
    write_multi_object_labels([_record(10, 1, bbox=[0, 0, 1, 1])], tmp_path, 10, 10, full_frame_range=(8, 10))
    assert (tmp_path / "000008.txt").read_text(encoding="utf-8") == ""
    assert (tmp_path / "000009.txt").read_text(encoding="utf-8") == ""


def test_overlay_grouping_keeps_missing_objects_optional() -> None:
    grouped = group_tracks([_record(1, 1), _record(2, 1), _record(2, 3)])
    assert [record["obj_id"] for record in grouped[1]] == [1]
    assert sorted(record["obj_id"] for record in grouped[2]) == [1, 3]


def test_mask_file_conflict_detection(tmp_path: Path) -> None:
    mask_a = tmp_path / "a.png"
    mask_b = tmp_path / "b.png"
    cv2.imwrite(str(mask_a), np.ones((4, 4), dtype=np.uint8) * 255)
    cv2.imwrite(str(mask_b), np.ones((4, 4), dtype=np.uint8) * 255)
    tracks_a = tmp_path / "a.jsonl"
    tracks_b = tmp_path / "b.jsonl"
    tracks_a.write_text(json.dumps({**_record(1, 1), "mask_path": str(mask_a)}) + "\n", encoding="utf-8")
    tracks_b.write_text(json.dumps({**_record(1, 1), "mask_path": str(mask_b)}) + "\n", encoding="utf-8")
    collect_mask_links([tracks_a], tmp_path / "merged")
    with pytest.raises(FileExistsError, match="conflict"):
        collect_mask_links([tracks_b], tmp_path / "merged")


def test_multi_object_anomaly_history_is_independent() -> None:
    observations = [_obs(i, obj_id=1, mask_area=80) for i in range(5)]
    observations.extend(_obs(i + 20, obj_id=2, bbox=(100, 10, 119, 29), mask_area=300) for i in range(5))
    observations.append(_obs(5, obj_id=1, mask_area=400, bbox=(5, 5, 24, 24)))
    observations.append(_obs(25, obj_id=2, mask_area=300, bbox=(100, 10, 119, 29)))
    by_key = {(result.frame_idx, result.obj_id): result for result in evaluate_tracks(observations, AnomalyParameters(history_size=5), (200, 100))}
    assert "area_increase" in by_key[(5, 1)].reasons
    assert "area_increase" not in by_key[(25, 2)].reasons


def test_periodic_check_uses_each_object_first_frame() -> None:
    observations = [_obs(i, obj_id=1) for i in range(0, 7)]
    observations.extend(_obs(i, obj_id=2) for i in range(20, 27))
    by_key = {(result.frame_idx, result.obj_id): result for result in evaluate_tracks(observations, AnomalyParameters(periodic_interval=5), (200, 100))}
    assert "periodic_check" in by_key[(5, 1)].reasons
    assert "periodic_check" in by_key[(25, 2)].reasons
    assert "periodic_check" not in by_key[(20, 2)].reasons


def test_duplicate_target_warning(tmp_path: Path) -> None:
    mask1 = np.zeros((8, 8), dtype=np.uint8)
    mask1[1:6, 1:6] = 255
    mask2 = np.zeros((8, 8), dtype=np.uint8)
    mask2[2:7, 2:7] = 255
    path1 = tmp_path / "m1.png"
    path2 = tmp_path / "m2.png"
    cv2.imwrite(str(path1), mask1)
    cv2.imwrite(str(path2), mask2)
    records = [{**_record(3, 1), "mask_path": str(path1)}, {**_record(3, 2), "mask_path": str(path2)}]
    results = evaluate_tracks([_obs(3, obj_id=1), _obs(3, obj_id=2)], AnomalyParameters(), (8, 8))
    warned = apply_duplicate_target_warnings(results, records, tmp_path / "tracks.jsonl", iou_threshold=0.3)
    assert all("duplicate_target_warning" in result.reasons for result in warned)


def test_hard_restart_replaces_only_selected_obj_id() -> None:
    baseline = [_record(9, 1, source="base1"), _record(10, 1, source="base1"), _record(10, 2, source="base2"), _record(11, 2, source="base2")]
    corrected = [_record(10, 1, source="corr1"), _record(11, 1, source="corr1")]
    merged = merge_track_records(baseline, corrected, correction_frame=10, obj_id=1)
    by_key = {(record["frame_idx"], record["obj_id"]): record for record in merged}
    assert by_key[(10, 1)]["source"] == "corr1"
    assert by_key[(10, 2)]["source"] == "base2"
    assert by_key[(11, 2)]["source"] == "base2"


def test_bbox_directly_from_raw_mask() -> None:
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    assert mask_to_bbox(mask) == (3, 2, 6, 4)


def test_registered_tracking_mainline_does_not_call_bbox_stabilizer() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "track_registered_object.py").read_text(encoding="utf-8")
    assert "BBoxStabilizer" not in source
    assert "tracks_stabilized" not in source


def _registry(objects: list[dict[str, object]]) -> dict[str, object]:
    normalized = []
    for index, obj in enumerate(objects):
        normalized.append({"global_drone_id": f"drone_{index + 1}", "class_id": index, "class_name": f"drone_{index + 1}", "first_visible_frame": int(obj.get("first_prompt_frame", 0)), "validated": True, **obj})
    return {"video": "data/videos/input.mp4", "frames_dir": "data/frames/input", "objects": normalized, "invalid_objects": []}


def _object(obj_id: int) -> dict[str, object]:
    return {"obj_id": obj_id, "name": f"drone_{obj_id}", "first_prompt_frame": 0, "status": "active"}


def _manifest(start: int, end: int) -> dict[str, object]:
    return {
        "start_frame": start,
        "end_frame": end,
        "frames": [{"local_frame_idx": local, "original_frame_idx": original} for local, original in enumerate(range(start, end + 1))],
    }


def _record(frame_idx: int, obj_id: int, bbox: list[int] | None = [0, 0, 1, 1], source: str = "track") -> dict[str, object]:
    return {
        "frame_idx": frame_idx,
        "original_frame_idx": frame_idx,
        "local_frame_idx": max(0, frame_idx - 203),
        "obj_id": obj_id,
        "object_name": f"drone_{obj_id}",
        "class_id": {1: 1, 2: 2, 3: 0}[obj_id],
        "mask_path": f"masks/{frame_idx:06d}_obj{obj_id}.png",
        "bbox_xyxy": bbox,
        "mask_area": 4 if bbox else 0,
        "bbox_area": 4 if bbox else 0,
        "center_xy": [0.5, 0.5] if bbox else None,
        "empty_mask": bbox is None,
        "source": source,
    }


def _obs(
    frame_idx: int,
    *,
    obj_id: int,
    bbox: tuple[int, int, int, int] | None = (10, 10, 19, 19),
    mask_area: int = 80,
) -> TrackObservation:
    area = 0 if bbox is None else (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    center = None if bbox is None else ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
    return TrackObservation(frame_idx=frame_idx, obj_id=obj_id, mask_area=mask_area, bbox_area=area, bbox_xyxy=bbox, center_xy=center)
