from __future__ import annotations

import json
from pathlib import Path

import pytest

from sam2_annotator import ui_service


def _catalog(path: Path) -> Path:
    path.write_text(json.dumps({"drones": [
        {"global_drone_id": "drone_1", "class_id": 7, "class_name": "one"},
        {"global_drone_id": "drone_2", "class_id": 9, "class_name": "two"},
    ]}), encoding="utf-8")
    return path


def _session(tmp_path: Path, objects: int = 1, *, manual: bool = False) -> tuple[Path, Path]:
    session = tmp_path / "ui_workspace" / "sessions" / "session"
    frames = session / "frames"
    frames.mkdir(parents=True)
    (frames / "metadata.json").write_text(json.dumps({"frame_count": 5, "width": 100, "height": 50, "fps": 10}), encoding="utf-8")
    registry_objects: list[dict[str, object]] = []
    timeline_objects: list[dict[str, object]] = []
    for obj_id in range(1, objects + 1):
        drone, class_id, name = ("drone_1", 7, "one") if obj_id == 1 else ("drone_2", 9, "two")
        registry_objects.append({"obj_id": obj_id, "name": f"obj{obj_id}", "status": "active", "validated": True, "global_drone_id": drone, "class_id": class_id, "class_name": name, "first_visible_frame": 0})
        original = session / "outputs" / f"obj{obj_id}" / "tracks.jsonl"
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_text("{}\n", encoding="utf-8")
        segments: list[dict[str, object]] = [{"start_frame": 0, "end_frame": 4 if not manual else 1, "source_type": "original", "tracks_path": str(original)}]
        if manual:
            corrected = session / "outputs" / f"obj{obj_id}" / "corrections" / "manual" / "tracks.jsonl"
            corrected.parent.mkdir(parents=True, exist_ok=True)
            corrected.write_text("{}\n", encoding="utf-8")
            segments.append({"start_frame": 2, "end_frame": 4, "source_type": "manual_hard_restart", "tracks_path": str(corrected)})
        timeline_objects.append({"obj_id": obj_id, "global_drone_id": drone, "class_id": class_id, "frame_range": [0, 4], "segments": segments})
    (session / "object_registry.json").write_text(json.dumps({"video": "input.mp4", "frames_dir": str(frames), "objects": registry_objects, "invalid_objects": []}), encoding="utf-8")
    (session / "object_timelines.json").write_text(json.dumps({"objects": timeline_objects}), encoding="utf-8")
    return session, _catalog(tmp_path / "catalog.json")


def _fake_finalizer(calls: list[tuple[object, ...]]):
    def fake(*args: object) -> dict[str, object]:
        calls.append(args)
        output = Path(args[4])
        (output / "merged").mkdir(parents=True)
        (output / "labels").mkdir()
        (output / "merged" / "tracks.jsonl").touch()
        (output / "labels" / "000000.txt").touch()
        (output / "overlay.mp4").touch()
        (output / "quality_report.json").write_text(json.dumps({"valid": True, "total_tracks": 5, "total_label_lines": 5}), encoding="utf-8")
        return {"quality_valid": True, "active_objects": [1], "output_dir": str(output)}
    return fake


@pytest.mark.parametrize("missing, message", [("session", "session"), ("registry", "registry"), ("timeline", "timeline"), ("frames", "frames_dir")])
def test_final_export_rejects_missing_session_inputs(tmp_path: Path, missing: str, message: str) -> None:
    session, catalog = _session(tmp_path)
    target = {"registry": session / "object_registry.json", "timeline": session / "object_timelines.json", "frames": session / "frames"}.get(missing)
    if target is not None:
        if target.is_dir():
            target.rename(target.with_name("frames-missing"))
        else:
            target.unlink()
    if missing == "session":
        session = tmp_path / "absent"
    with pytest.raises(FileNotFoundError, match=message):
        ui_service.run_final_export(session_dir=session, catalog_path=catalog)


def test_final_export_rejects_zero_active_objects(tmp_path: Path) -> None:
    session, catalog = _session(tmp_path)
    registry = json.loads((session / "object_registry.json").read_text(encoding="utf-8"))
    registry["objects"][0]["status"] = "finished"
    (session / "object_registry.json").write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="no active"):
        ui_service.run_final_export(session_dir=session, catalog_path=catalog)


@pytest.mark.parametrize("objects, manual", [(1, False), (2, False), (1, True)])
def test_final_export_uses_session_inputs_and_existing_finalizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, objects: int, manual: bool) -> None:
    session, catalog = _session(tmp_path, objects, manual=manual)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(ui_service, "finalize_video_annotation", _fake_finalizer(calls))
    result = ui_service.run_final_export(session_dir=session, catalog_path=catalog)
    assert len(calls) == 1
    timeline, registry, used_catalog, frames, output, overwrite, render = calls[0][:7]
    assert timeline == (session / "object_timelines.json").resolve()
    assert registry == (session / "object_registry.json").resolve()
    assert used_catalog == catalog.resolve() and frames == (session / "frames").resolve()
    assert output == (session / "final").resolve() and overwrite is True and render is True
    assert result.merged_tracks_path == session / "final" / "merged" / "tracks.jsonl"
    assert result.labels_dir == session / "final" / "labels"
    assert result.quality_report_path == session / "final" / "quality_report.json"
    assert result.overlay_path == session / "final" / "overlay.mp4"
    assert result.valid is True


def test_final_export_preflight_checks_tracks_and_catalog_identity(tmp_path: Path) -> None:
    session, catalog = _session(tmp_path)
    timeline = json.loads((session / "object_timelines.json").read_text(encoding="utf-8"))
    timeline["objects"][0]["segments"][0]["tracks_path"] = str(session / "missing.jsonl")
    (session / "object_timelines.json").write_text(json.dumps(timeline), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing tracks"):
        ui_service.run_final_export(session_dir=session, catalog_path=catalog)
    timeline["objects"][0]["segments"][0]["tracks_path"] = str(session / "outputs" / "obj1" / "tracks.jsonl")
    (session / "object_timelines.json").write_text(json.dumps(timeline), encoding="utf-8")
    registry = json.loads((session / "object_registry.json").read_text(encoding="utf-8"))
    registry["objects"][0]["class_id"] = 99
    (session / "object_registry.json").write_text(json.dumps(registry), encoding="utf-8")
    timeline["objects"][0]["class_id"] = 99
    (session / "object_timelines.json").write_text(json.dumps(timeline), encoding="utf-8")
    with pytest.raises(ValueError, match="catalog"):
        ui_service.run_final_export(session_dir=session, catalog_path=catalog)


def test_final_export_failure_does_not_touch_prior_final(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session, catalog = _session(tmp_path)
    old = session / "final" / "marker.txt"
    old.parent.mkdir()
    old.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(ui_service, "finalize_video_annotation", lambda *args: (_ for _ in ()).throw(RuntimeError("finalize failed")))
    with pytest.raises(RuntimeError, match="finalize failed"):
        ui_service.run_final_export(session_dir=session, catalog_path=catalog)
    assert old.read_text(encoding="utf-8") == "preserve"
