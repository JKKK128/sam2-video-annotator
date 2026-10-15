"""Regression tests for fixed global drone identities and YOLO classes."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from merge_object_tracks import merge_object_tracks  # noqa: E402
from sam2_annotator.object_registry import validate_registry  # noqa: E402


def _catalog(path: Path) -> Path:
    path.write_text(json.dumps({"drones": [
        {"global_drone_id": "drone_1", "class_id": 1, "class_name": "drone_1"},
        {"global_drone_id": "drone_2", "class_id": 2, "class_name": "drone_2"},
        {"global_drone_id": "drone_3", "class_id": 0, "class_name": "drone_3"},
    ]}), encoding="utf-8")
    return path


def _object(obj_id: int, drone: str, class_id: int) -> dict[str, object]:
    return {"obj_id": obj_id, "name": f"local_{obj_id}", "global_drone_id": drone, "class_id": class_id, "class_name": drone, "first_visible_frame": 0, "status": "active", "validated": True}


def _registry(path: Path, objects: list[dict[str, object]], invalid: list[dict[str, object]] | None = None) -> Path:
    path.write_text(json.dumps({"video": "videos/example.mp4", "frames_dir": "frames/example", "objects": objects, "invalid_objects": invalid or []}), encoding="utf-8")
    return path


def _track(path: Path, frame: int, obj_id: int) -> Path:
    path.write_text(json.dumps({"frame_idx": frame, "obj_id": obj_id, "bbox_xyxy": [0, 0, 9, 9], "frame_width": 100, "frame_height": 100}) + "\n", encoding="utf-8")
    return path


def _merge(tmp_path: Path, registry: Path, catalog: Path, tracks: list[Path]) -> list[dict[str, object]]:
    return merge_object_tracks(tracks, tmp_path / "merged.jsonl", tmp_path / "labels", registry, catalog, True, None, None, False, None)


def test_obj1_and_obj2_map_to_fixed_classes_without_class0(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path / "catalog.json")
    registry = _registry(tmp_path / "registry.json", [_object(1, "drone_1", 1), _object(2, "drone_2", 2)])
    records = _merge(tmp_path, registry, catalog, [_track(tmp_path / "one.jsonl", 3, 1), _track(tmp_path / "two.jsonl", 3, 2)])
    assert [(row["obj_id"], row["class_id"]) for row in records] == [(1, 1), (2, 2)]
    assert [line.split()[0] for line in (tmp_path / "labels" / "000003.txt").read_text().splitlines()] == ["1", "2"]


def test_another_video_local_obj1_can_export_drone3_as_class0(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path / "catalog.json")
    registry = _registry(tmp_path / "registry.json", [_object(1, "drone_3", 0)])
    records = _merge(tmp_path, registry, catalog, [_track(tmp_path / "track.jsonl", 2, 1)])
    assert records[0]["global_drone_id"] == "drone_3"
    assert records[0]["class_id"] == 0
    assert (tmp_path / "labels" / "000002.txt").read_text().startswith("0 ")


def test_legacy_record_is_hydrated_from_registry_not_its_obj_id(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path / "catalog.json")
    registry = _registry(tmp_path / "registry.json", [_object(9, "drone_2", 2)])
    records = _merge(tmp_path, registry, catalog, [_track(tmp_path / "track.jsonl", 1, 9)])
    assert records[0]["class_id"] == 2
    assert records[0]["global_drone_id"] == "drone_2"


def test_invalid_background_obj3_is_rejected(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path / "catalog.json")
    registry = _registry(tmp_path / "registry.json", [_object(1, "drone_1", 1)], [{"obj_id": 3, "invalid_reason": "background mis-track"}])
    with pytest.raises(ValueError, match="invalid"):
        _merge(tmp_path, registry, catalog, [_track(tmp_path / "track.jsonl", 1, 3)])


def test_registry_rejects_three_active_objects() -> None:
    registry = {"video": "v.mp4", "frames_dir": "frames", "objects": [_object(1, "drone_1", 1), _object(2, "drone_2", 2), _object(3, "drone_3", 0)], "invalid_objects": []}
    with pytest.raises(ValueError, match="at most 2"):
        validate_registry(registry)


def test_registry_rejects_duplicate_active_global_identity() -> None:
    registry = {"video": "v.mp4", "frames_dir": "frames", "objects": [_object(1, "drone_1", 1), _object(2, "drone_1", 1)], "invalid_objects": []}
    with pytest.raises(ValueError, match="duplicate active"):
        validate_registry(registry)


def test_same_frame_two_objects_remain_two_label_lines(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path / "catalog.json")
    registry = _registry(tmp_path / "registry.json", [_object(4, "drone_1", 1), _object(7, "drone_2", 2)])
    _merge(tmp_path, registry, catalog, [_track(tmp_path / "a.jsonl", 8, 4), _track(tmp_path / "b.jsonl", 8, 7)])
    assert len((tmp_path / "labels" / "000008.txt").read_text().splitlines()) == 2


def test_undefined_global_drone_id_is_rejected_by_catalog(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path / "catalog.json")
    registry = _registry(tmp_path / "registry.json", [_object(1, "unknown_drone", 9)])
    with pytest.raises(ValueError, match="undefined global_drone_id"):
        _merge(tmp_path, registry, catalog, [_track(tmp_path / "track.jsonl", 1, 1)])
