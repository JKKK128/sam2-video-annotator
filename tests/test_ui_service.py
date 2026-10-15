from __future__ import annotations

import json
from pathlib import Path

import pytest

from sam2_annotator import ui_service


def _catalog(path: Path) -> Path:
    path.write_text(json.dumps({"drones": [{"global_drone_id": "drone_1", "class_id": 7, "class_name": "catalog-name"}]}), encoding="utf-8")
    return path


def _frames(path: Path, count: int = 4) -> Path:
    path.mkdir()
    (path / "metadata.json").write_text(json.dumps({"frame_count": count, "width": 100, "height": 50, "fps": 12.5}), encoding="utf-8")
    return path


def test_prepare_video_uses_extract_frames_and_creates_isolated_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "input.mp4"
    video.touch()
    calls: list[tuple[Path, Path, int, bool]] = []

    def fake_extract(source: Path, frames: Path, quality: int, overwrite: bool) -> dict[str, object]:
        calls.append((source, frames, quality, overwrite))
        frames.mkdir()
        return {"frame_count": 3, "width": 20, "height": 10, "fps": 24.0}

    monkeypatch.setattr(ui_service, "extract_frames", fake_extract)
    one = ui_service.prepare_ui_source("video", video, workspace=tmp_path / "workspace")
    two = ui_service.prepare_ui_source("video", video, workspace=tmp_path / "workspace")
    assert calls[0][0] == video.resolve()
    assert calls[0][3] is False
    assert one.frames_dir == calls[0][1]
    assert one.session_dir != two.session_dir
    assert one.registry_path.is_file()
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "outputs").exists()


def test_prepare_image_sequence_uses_unified_frames_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    seen: list[tuple[Path, Path, float]] = []

    class Prepared:
        frame_count = 2
        width = 30
        height = 15
        fps = 9.0

    def fake_prepare(source: Path, frames: Path, fps: float) -> Prepared:
        seen.append((source, frames, fps))
        frames.mkdir()
        return Prepared()

    monkeypatch.setattr(ui_service, "prepare_image_sequence", fake_prepare)
    result = ui_service.prepare_ui_source("image_sequence", image_dir, workspace=tmp_path / "workspace", image_fps=9)
    assert result.frames_dir == seen[0][1]
    assert result.source_type == "image_sequence"
    assert result.frame_count == 2


def test_prepare_source_rejects_invalid_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source_type"):
        ui_service.prepare_ui_source("camera", tmp_path)


def test_normalize_browser_bbox_scales_clamps_and_rejects_zero_area() -> None:
    assert ui_service.normalize_browser_bbox((-2, 4, 60, 30), original_width=100, original_height=50, display_width=50, display_height=25) == (0, 8, 99, 49)
    with pytest.raises(ValueError, match="non-zero"):
        ui_service.normalize_browser_bbox((4, 4, 4, 10), original_width=100, original_height=50)


def test_tracking_rejects_missing_session_and_unknown_global_drone(tmp_path: Path) -> None:
    frames = _frames(tmp_path / "frames")
    registry = tmp_path / "registry.json"
    registry.write_text("{}", encoding="utf-8")
    catalog = _catalog(tmp_path / "catalog.json")
    with pytest.raises(FileNotFoundError, match="session_dir"):
        ui_service.run_initial_tracking(
            frames_dir=frames, registry_path=registry, catalog_path=catalog, config_path=tmp_path / "config.yaml",
            obj_id=1, global_drone_id="drone_1", start_frame=0, end_frame=1,
            bbox_xyxy=(1, 1, 2, 2), session_dir=tmp_path / "missing",
        )
    session = tmp_path / "session"
    session.mkdir()
    with pytest.raises(ValueError, match="undefined global_drone_id"):
        ui_service.run_initial_tracking(
            frames_dir=frames, registry_path=registry, catalog_path=catalog, config_path=tmp_path / "config.yaml",
            obj_id=1, global_drone_id="not-in-catalog", start_frame=0, end_frame=1,
            bbox_xyxy=(1, 1, 2, 2), session_dir=session,
        )


def test_run_tracking_orchestrates_existing_functions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _frames(tmp_path / "frames")
    session = tmp_path / "session"
    session.mkdir()
    registry = session / "object_registry.json"
    registry.write_text(json.dumps({"video": "input", "frames_dir": str(frames), "objects": [], "invalid_objects": []}), encoding="utf-8")
    catalog = _catalog(tmp_path / "catalog.json")
    order: list[str] = []

    def register(*args: object) -> None:
        order.append("register")
        Path(args[7]).parent.mkdir(parents=True, exist_ok=True)  # type: ignore[arg-type]
        Path(args[7]).touch()  # type: ignore[arg-type]

    def segment(*args: object) -> None:
        order.append("segment")
        Path(args[3]).mkdir(parents=True)  # type: ignore[arg-type]
        Path(args[5]).touch()  # type: ignore[arg-type]

    def track(*args: object) -> None:
        order.append("track")
        output = Path(args[5])  # type: ignore[arg-type]
        output.mkdir(parents=True)
        (output / "tracks.jsonl").touch()

    def render(*args: object) -> None:
        order.append("render")
        Path(args[2]).touch()  # type: ignore[arg-type]

    monkeypatch.setattr(ui_service, "register_object", register)
    monkeypatch.setattr(ui_service, "create_tracking_segment", segment)
    monkeypatch.setattr(ui_service, "track_registered_object", track)
    monkeypatch.setattr(ui_service, "render_multi_overlay", render)
    result = ui_service.run_initial_tracking(
        frames_dir=frames, registry_path=registry, catalog_path=catalog, config_path=tmp_path / "config.yaml",
        obj_id=1, global_drone_id="drone_1", start_frame=0, end_frame=3,
        bbox_xyxy=(1, 2, 10, 12), session_dir=session,
    )
    assert order == ["register", "segment", "track", "render"]
    assert result.class_id == 7  # comes from catalog, not obj_id
    assert result.prompt_path == session / "prompts" / "obj1_global.json"
    assert result.local_prompt_path == session / "prompts" / "obj1_local.json"
    assert result.segment_dir == session / "segments" / "obj1_000000_000003"
    assert result.output_dir == session / "outputs" / "obj1"
    assert result.overlay_path.is_file()
    timeline = json.loads((session / "object_timelines.json").read_text(encoding="utf-8"))
    assert timeline["objects"][0]["segments"][0]["source_type"] == "original"
    assert not (tmp_path / "data" / "object_timelines.json").exists()


def test_two_objects_have_separate_outputs_and_failures_do_not_remove_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _frames(tmp_path / "frames")
    session = tmp_path / "session"
    session.mkdir()
    registry = session / "object_registry.json"
    registry.write_text(json.dumps({"video": "input", "frames_dir": str(frames), "objects": [], "invalid_objects": []}), encoding="utf-8")
    catalog = _catalog(tmp_path / "catalog.json")
    def register(*args: object) -> None:
        Path(args[7]).parent.mkdir(parents=True, exist_ok=True)  # type: ignore[arg-type]
        Path(args[7]).touch()  # type: ignore[arg-type]

    def segment(*args: object) -> None:
        Path(args[3]).mkdir(parents=True)  # type: ignore[arg-type]
        Path(args[5]).touch()  # type: ignore[arg-type]

    def track(*args: object) -> None:
        output = Path(args[5])  # type: ignore[arg-type]
        output.mkdir(parents=True)
        (output / "tracks.jsonl").touch()

    def render(*args: object) -> None:
        Path(args[2]).touch()  # type: ignore[arg-type]

    monkeypatch.setattr(ui_service, "register_object", register)
    monkeypatch.setattr(ui_service, "create_tracking_segment", segment)
    monkeypatch.setattr(ui_service, "track_registered_object", track)
    monkeypatch.setattr(ui_service, "render_multi_overlay", render)
    first = ui_service.run_initial_tracking(
        frames_dir=frames, registry_path=registry, catalog_path=catalog, config_path=tmp_path / "config.yaml",
        obj_id=1, global_drone_id="drone_1", start_frame=0, end_frame=1,
        bbox_xyxy=(1, 1, 2, 2), session_dir=session,
    )
    second = ui_service.run_initial_tracking(
        frames_dir=frames, registry_path=registry, catalog_path=catalog, config_path=tmp_path / "config.yaml",
        obj_id=2, global_drone_id="drone_1", start_frame=0, end_frame=1,
        bbox_xyxy=(1, 1, 2, 2), session_dir=session,
    )
    assert first.output_dir != second.output_dir
    assert first.overlay_path.is_file() and second.overlay_path.is_file()

    def fail_track(*args: object) -> None:
        raise RuntimeError("mock failure")

    monkeypatch.setattr(ui_service, "track_registered_object", fail_track)
    failure_session = tmp_path / "failure-session"
    failure_session.mkdir()
    existing = failure_session / "outputs" / "obj1" / "tracks.jsonl"
    existing.parent.mkdir(parents=True)
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="mock failure"):
        ui_service.run_initial_tracking(
            frames_dir=frames, registry_path=registry, catalog_path=catalog, config_path=tmp_path / "config.yaml",
            obj_id=2, global_drone_id="drone_1", start_frame=0, end_frame=1,
            bbox_xyxy=(1, 1, 2, 2), session_dir=failure_session,
        )
    assert existing.read_text(encoding="utf-8") == "keep"
    assert (session / "outputs" / "obj2") != existing.parent


def _restart_session(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    frames = _frames(tmp_path / "frames", 6)
    session = tmp_path / "ui_workspace" / "sessions" / "session"
    session.mkdir(parents=True)
    registry = session / "object_registry.json"
    registry.write_text(json.dumps({"video": "input.mp4", "frames_dir": str(frames), "objects": [{"obj_id": 1, "name": "obj1", "status": "active", "validated": True, "global_drone_id": "drone_1", "class_id": 7, "class_name": "catalog-name", "first_visible_frame": 0}], "invalid_objects": []}), encoding="utf-8")
    catalog = _catalog(tmp_path / "catalog.json")
    original = session / "outputs" / "obj1" / "tracks.jsonl"
    original.parent.mkdir(parents=True)
    original.write_text("{}\n", encoding="utf-8")
    timeline = session / "object_timelines.json"
    timeline.write_text(json.dumps({"objects": [{"obj_id": 1, "global_drone_id": "drone_1", "class_id": 7, "frame_range": [0, 5], "segments": [{"start_frame": 0, "end_frame": 5, "source_type": "original", "tracks_path": str(original)}]}]}), encoding="utf-8")
    return frames, session, registry, catalog, timeline


def test_manual_restart_uses_injected_box_and_keeps_timeline_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames, session, registry, catalog, timeline = _restart_session(tmp_path)
    calls: list[object] = []

    def fake_handoff(args: object, manual_box: tuple[int, int, int, int] | None = None) -> dict[str, str]:
        calls.extend([args, manual_box])
        output = args.output_dir  # type: ignore[attr-defined]
        output.mkdir(parents=True)
        (output / "tracks.jsonl").touch()
        (output / "preview.mp4").touch()
        return {"tracks_path": str(output / "tracks.jsonl"), "preview_path": str(output / "preview.mp4")}

    monkeypatch.setattr(ui_service, "create_handoff", fake_handoff)
    before = timeline.read_text(encoding="utf-8")
    result = ui_service.run_manual_restart(frames_dir=frames, registry_path=registry, catalog_path=catalog, config_path=tmp_path / "config.yaml", session_dir=session, obj_id=1, restart_frame=2, end_frame=5, bbox_xyxy=(1, 2, 10, 12))
    assert calls[1] == (1, 2, 10, 12)
    assert result.correction_dir.parent == session / "outputs" / "obj1" / "corrections"
    assert result.preview_path.is_file()
    assert timeline.read_text(encoding="utf-8") == before


@pytest.mark.parametrize(("restart", "end", "match"), [(-1, 2, "restart_frame"), (6, 6, "restart_frame"), (3, 2, "end_frame")])
def test_manual_restart_rejects_invalid_frames(tmp_path: Path, restart: int, end: int, match: str) -> None:
    frames, session, registry, catalog, _ = _restart_session(tmp_path)
    with pytest.raises(ValueError, match=match):
        ui_service.run_manual_restart(frames_dir=frames, registry_path=registry, catalog_path=catalog, config_path=tmp_path / "config.yaml", session_dir=session, obj_id=1, restart_frame=restart, end_frame=end, bbox_xyxy=(1, 2, 10, 12))


def test_manual_restart_rejects_missing_or_zero_area_bbox(tmp_path: Path) -> None:
    frames, session, registry, catalog, _ = _restart_session(tmp_path)
    for bbox in ((), (1, 1, 1, 2)):
        with pytest.raises(ValueError):
            ui_service.run_manual_restart(frames_dir=frames, registry_path=registry, catalog_path=catalog, config_path=tmp_path / "config.yaml", session_dir=session, obj_id=1, restart_frame=2, end_frame=5, bbox_xyxy=bbox)


def test_accept_manual_restart_splits_timeline_and_rebuilds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames, session, registry, catalog, timeline = _restart_session(tmp_path)
    candidate = session / "outputs" / "obj1" / "corrections" / "candidate" / "tracks.jsonl"
    candidate.parent.mkdir(parents=True)
    rows = [{"obj_id": 1, "global_drone_id": "drone_1", "class_id": 7, "class_name": "catalog-name", "frame_idx": frame, "frame_width": 100, "frame_height": 50, "bbox_xyxy": [1, 1, 2, 2], "mask_path": "mask.png", "mask_area": 4} for frame in range(2, 6)]
    candidate.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    built: list[object] = []
    rendered: list[object] = []
    monkeypatch.setattr(ui_service, "build_object_track", lambda *args: built.append(args) or {})
    monkeypatch.setattr(ui_service, "render_multi_overlay", lambda *args: rendered.append(args))
    result = ui_service.accept_manual_restart(frames_dir=frames, registry_path=registry, catalog_path=catalog, session_dir=session, obj_id=1, candidate_tracks_path=candidate, restart_frame=2, end_frame=5)
    segments = json.loads(timeline.read_text(encoding="utf-8"))["objects"][0]["segments"]
    assert [(segment["start_frame"], segment["end_frame"], segment["source_type"]) for segment in segments] == [(0, 1, "original"), (2, 5, "manual_hard_restart")]
    assert built and rendered
    assert result.final_tracks_path == session / "outputs" / "obj1" / "final_tracks.jsonl"
    assert result.current_overlay_path == session / "outputs" / "obj1" / "current_overlay.mp4"
