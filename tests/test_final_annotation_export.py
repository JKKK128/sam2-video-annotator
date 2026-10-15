"""Unit tests for transactional final exports and strict quality validation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import finalize_video_annotation as finalizer  # noqa: E402
from validate_annotation_export import AnnotationValidationError, validate_annotation_export  # noqa: E402


def _catalog(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"drones": [
        {"global_drone_id": "drone_1", "class_id": 1, "class_name": "drone_1"},
        {"global_drone_id": "drone_2", "class_id": 2, "class_name": "drone_2"},
        {"global_drone_id": "drone_3", "class_id": 0, "class_name": "drone_3"},
    ]}), encoding="utf-8")
    return path


def _object(obj_id: int, drone: str, class_id: int, *, status: str = "active", validated: bool = True) -> dict[str, object]:
    return {"obj_id": obj_id, "name": f"local_{obj_id}", "global_drone_id": drone, "class_id": class_id, "class_name": drone, "first_visible_frame": 10, "status": status, "validated": validated}


def _setup(tmp_path: Path, objects: list[dict[str, object]]) -> tuple[Path, Path, Path, Path]:
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"video": "videos/flight.mp4", "frames_dir": "frames", "objects": objects, "invalid_objects": []}), encoding="utf-8")
    tracks: list[dict[str, object]] = []
    timeline_objects: list[dict[str, object]] = []
    for obj in objects:
        obj_id = int(obj["obj_id"])
        source = tmp_path / f"source_{obj_id}.jsonl"
        rows = [{"frame_idx": frame, "obj_id": obj_id, "bbox_xyxy": [10, 10, 20, 20], "frame_width": 100, "frame_height": 100} for frame in (10, 11)]
        source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        tracks.append({"obj_id": obj_id, "frame_range": [10, 11], "segments": [{"start_frame": 10, "end_frame": 11, "tracks_path": str(source)}]})
    timeline = tmp_path / "timeline.json"
    timeline.write_text(json.dumps({"objects": tracks}), encoding="utf-8")
    frames = tmp_path / "frames"
    frames.mkdir()
    return timeline, registry, _catalog(tmp_path / "catalog.json"), frames


def _validate(tmp_path: Path, rows: list[dict[str, object]], lines: str) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tracks = tmp_path / "tracks.jsonl"
    tracks.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "000010.txt").write_text(lines, encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"video": "v.mp4", "frames_dir": "f", "objects": [_object(7, "drone_3", 0)], "invalid_objects": []}), encoding="utf-8")
    with pytest.raises(AnnotationValidationError):
        validate_annotation_export(tracks, labels, registry, _catalog(tmp_path / "catalog.json"))


def test_two_objects_finalize_and_quality_report_is_correct(tmp_path: Path) -> None:
    timeline, registry, catalog, frames = _setup(tmp_path, [_object(1, "drone_1", 1), _object(9, "drone_2", 2)])
    output = tmp_path / "final"
    summary = finalizer.finalize_video_annotation(timeline, registry, catalog, frames, output, False)
    report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
    assert summary["active_objects"] == [1, 9]
    assert summary["total_labels"] == 4
    assert report["total_tracks"] == 4
    assert report["records_by_obj_id"] == {"1": 2, "9": 2}
    assert (output / "objects" / "obj1" / "final_tracks.jsonl").is_file()
    assert (output / "objects" / "obj9" / "final_tracks.jsonl").is_file()


def test_one_object_finalizes_and_drone3_exports_class_zero(tmp_path: Path) -> None:
    timeline, registry, catalog, frames = _setup(tmp_path, [_object(7, "drone_3", 0)])
    output = tmp_path / "final"
    finalizer.finalize_video_annotation(timeline, registry, catalog, frames, output, False)
    assert (output / "labels" / "000010.txt").read_text(encoding="utf-8").startswith("0 ")


def test_three_active_objects_error(tmp_path: Path) -> None:
    timeline, registry, catalog, frames = _setup(tmp_path, [_object(1, "drone_1", 1), _object(2, "drone_2", 2), _object(3, "drone_3", 0)])
    with pytest.raises(ValueError, match="at most 2"):
        finalizer.finalize_video_annotation(timeline, registry, catalog, frames, tmp_path / "final", False)


def test_invalid_object_is_not_exported(tmp_path: Path) -> None:
    timeline, registry, catalog, frames = _setup(tmp_path, [_object(1, "drone_1", 1), _object(2, "drone_2", 2, validated=False)])
    output = tmp_path / "final"
    finalizer.finalize_video_annotation(timeline, registry, catalog, frames, output, False)
    assert not (output / "objects" / "obj2").exists()
    assert json.loads((output / "quality_report.json").read_text())["total_tracks"] == 2


def test_validator_rejects_count_duplicate_and_invalid_coordinates(tmp_path: Path) -> None:
    row = {"frame_idx": 10, "obj_id": 7, "global_drone_id": "drone_3", "class_id": 0, "bbox_xyxy": [1, 1, 5, 5]}
    _validate(tmp_path / "count", [row], "")
    _validate(tmp_path / "duplicate", [row, row], "0 0.3 0.3 0.2 0.2\n0 0.3 0.3 0.2 0.2\n")
    _validate(tmp_path / "coords", [row], "0 1.2 0.3 0.2 0.2\n")


def test_failed_finalization_preserves_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timeline, registry, catalog, frames = _setup(tmp_path, [_object(1, "drone_1", 1)])
    output = tmp_path / "final"
    output.mkdir()
    (output / "completed.txt").write_text("old export", encoding="utf-8")
    monkeypatch.setattr(finalizer, "merge_object_tracks", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("mid-build failure")))
    with pytest.raises(RuntimeError, match="mid-build failure"):
        finalizer.finalize_video_annotation(timeline, registry, catalog, frames, output, True)
    assert (output / "completed.txt").read_text(encoding="utf-8") == "old export"
