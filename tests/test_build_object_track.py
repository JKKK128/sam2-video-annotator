"""Tests for concatenating original and hard-restart object track segments."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_object_track import build_object_track  # noqa: E402


def _registry(path: Path) -> Path:
    path.write_text(json.dumps({"video": "videos/flight.mp4", "frames_dir": "frames/flight", "objects": [{"obj_id": 1, "name": "local_1", "global_drone_id": "drone_a", "class_id": 7, "class_name": "alpha", "first_visible_frame": 10, "status": "active", "validated": True}], "invalid_objects": []}), encoding="utf-8")
    return path


def _catalog(path: Path) -> Path:
    path.write_text(json.dumps({"drones": [{"global_drone_id": "drone_a", "class_id": 7, "class_name": "alpha"}]}), encoding="utf-8")
    return path


def _track(path: Path, frames: list[int], *, identity: bool = False, class_id: int = 7) -> Path:
    rows: list[dict[str, object]] = []
    for frame in frames:
        row: dict[str, object] = {"frame_idx": frame, "obj_id": 1, "bbox_xyxy": [1, 2, 3, 4]}
        if identity:
            row.update({"global_drone_id": "drone_a", "class_id": class_id, "class_name": "alpha", "video_id": "flight"})
        rows.append(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _timeline(path: Path, segments: list[dict[str, object]], frame_range: list[int] = [10, 14]) -> Path:
    path.write_text(json.dumps({"objects": [{"obj_id": 1, "frame_range": frame_range, "segments": segments}]}), encoding="utf-8")
    return path


def _build(tmp_path: Path, timeline: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    output = tmp_path / "out.jsonl"
    summary = build_object_track(timeline, _registry(tmp_path / "registry.json"), _catalog(tmp_path / "catalog.json"), 1, output, False)
    return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()], summary


def test_original_and_hard_restart_segments_are_concatenated(tmp_path: Path) -> None:
    original = _track(tmp_path / "original.jsonl", [10, 11, 12])
    restart = _track(tmp_path / "restart.jsonl", [13, 14])
    rows, summary = _build(tmp_path, _timeline(tmp_path / "timeline.json", [{"start_frame": 10, "end_frame": 12, "tracks_path": str(original), "source_type": "original"}, {"start_frame": 13, "end_frame": 14, "tracks_path": str(restart), "source_type": "hard_restart", "restart_frame": 13}]))
    assert [row["frame_idx"] for row in rows] == [10, 11, 12, 13, 14]
    assert [row["source_type"] for row in rows] == ["original", "original", "original", "hard_restart", "hard_restart"]
    assert [row.get("restart_frame") for row in rows[3:]] == [13, 13]
    assert summary["total_records"] == 5
    assert [segment["record_count"] for segment in summary["segments"]] == [3, 2]


@pytest.mark.parametrize(
    ("start_frame", "restart_frame"),
    [(241, 240), (329, 329)],
)
def test_hard_restart_frame_may_precede_or_equal_segment_start(
    tmp_path: Path, start_frame: int, restart_frame: int
) -> None:
    tracks = _track(tmp_path / "tracks.jsonl", [start_frame])
    timeline = _timeline(
        tmp_path / "timeline.json",
        [{"start_frame": start_frame, "end_frame": start_frame, "tracks_path": str(tracks), "source_type": "hard_restart", "restart_frame": restart_frame}],
        [start_frame, start_frame],
    )

    rows, _summary = _build(tmp_path, timeline)

    assert rows[0]["restart_frame"] == restart_frame


@pytest.mark.parametrize("restart_frame", [242, -1])
def test_hard_restart_frame_after_segment_start_or_negative_raises(tmp_path: Path, restart_frame: int) -> None:
    tracks = _track(tmp_path / "tracks.jsonl", [241])
    timeline = _timeline(
        tmp_path / "timeline.json",
        [{"start_frame": 241, "end_frame": 241, "tracks_path": str(tracks), "source_type": "hard_restart", "restart_frame": restart_frame}],
        [241, 241],
    )

    with pytest.raises(ValueError, match="restart_frame must be a non-negative integer no later than its start_frame"):
        _build(tmp_path, timeline)


def test_overlapping_segments_raise(tmp_path: Path) -> None:
    tracks = _track(tmp_path / "tracks.jsonl", [10, 11, 12, 13, 14])
    timeline = _timeline(tmp_path / "timeline.json", [{"start_frame": 10, "end_frame": 12, "tracks_path": str(tracks)}, {"start_frame": 12, "end_frame": 14, "tracks_path": str(tracks)}])
    with pytest.raises(ValueError, match="overlap"):
        _build(tmp_path, timeline)


def test_gap_between_segments_raises(tmp_path: Path) -> None:
    tracks = _track(tmp_path / "tracks.jsonl", [10, 11, 12, 13, 14])
    timeline = _timeline(tmp_path / "timeline.json", [{"start_frame": 10, "end_frame": 11, "tracks_path": str(tracks)}, {"start_frame": 13, "end_frame": 14, "tracks_path": str(tracks)}])
    with pytest.raises(ValueError, match="missing frames"):
        _build(tmp_path, timeline)


def test_missing_frame_in_input_track_raises(tmp_path: Path) -> None:
    tracks = _track(tmp_path / "tracks.jsonl", [10, 11, 13, 14])
    timeline = _timeline(tmp_path / "timeline.json", [{"start_frame": 10, "end_frame": 14, "tracks_path": str(tracks)}])
    with pytest.raises(ValueError, match=r"missing frames \[12\]"):
        _build(tmp_path, timeline)


def test_unknown_obj_id_raises(tmp_path: Path) -> None:
    tracks = _track(tmp_path / "tracks.jsonl", [10, 11, 12, 13, 14])
    timeline = _timeline(tmp_path / "timeline.json", [{"start_frame": 10, "end_frame": 14, "tracks_path": str(tracks)}])
    with pytest.raises(ValueError, match="obj_id 2"):
        build_object_track(timeline, _registry(tmp_path / "registry.json"), _catalog(tmp_path / "catalog.json"), 2, tmp_path / "out.jsonl", False)


def test_track_class_id_disagreeing_with_catalog_raises(tmp_path: Path) -> None:
    tracks = _track(tmp_path / "tracks.jsonl", [10, 11, 12, 13, 14], identity=True, class_id=99)
    timeline = _timeline(tmp_path / "timeline.json", [{"start_frame": 10, "end_frame": 14, "tracks_path": str(tracks)}])
    with pytest.raises(ValueError, match="class_id disagrees"):
        _build(tmp_path, timeline)


def test_legacy_track_identity_is_hydrated_from_registry_and_catalog(tmp_path: Path) -> None:
    tracks = _track(tmp_path / "tracks.jsonl", [10, 11, 12, 13, 14])
    timeline = _timeline(tmp_path / "timeline.json", [{"frame_range": [10, 14], "tracks_path": str(tracks)}])
    rows, _summary = _build(tmp_path, timeline)
    assert {(row["global_drone_id"], row["class_id"], row["class_name"], row["video_id"]) for row in rows} == {("drone_a", 7, "alpha", "flight")}
