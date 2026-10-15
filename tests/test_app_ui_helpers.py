from __future__ import annotations

from pathlib import Path

import app
import pytest

from sam2_annotator.ui_service import InitialTrackingResult, SourceInfo


def _source(tmp_path: Path, frame_count: int = 4) -> SourceInfo:
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(frame_count):
        (frames / f"{index:06d}.jpg").touch()
    session = tmp_path / "session"
    session.mkdir()
    return SourceInfo(session, frames, session / "registry.json", "image_sequence", frames, frame_count, 100, 50, 12.0)


def test_frame_navigation_reads_canonical_frames_and_stays_in_bounds(tmp_path: Path) -> None:
    source = _source(tmp_path)
    assert app._frame_for(source, 0).endswith("000000.jpg")
    state = {"1": None, "2": None}
    previous = app._step(-1, source, state, 1, 0, 3)
    following = app._step(1, source, state, 1, 3, 3)
    assert previous[0]["value"] == 0
    assert following[0]["value"] == 3
    changed = app._set_start(source, state, 1, 2, 0)
    assert changed[0]["value"] == 2
    assert changed[1]["value"] == 2
    assert changed[2]["value"].endswith("000002.jpg")


def test_end_frame_is_independent_but_cannot_precede_start(tmp_path: Path) -> None:
    source = _source(tmp_path)
    update, message = app._set_end(source, 2, 3)
    assert update["value"] == 3 and message == ""
    update, message = app._set_end(source, 2, 1)
    assert update["value"] == 2
    assert "不能早于" in message


def test_browser_bbox_mapping_clamping_replacement_and_clear(tmp_path: Path) -> None:
    source = _source(tmp_path)
    state = app._empty_bbox_state()
    first = '{"x": 20, "y": 10, "width": 20, "height": 10, "display_width": 50, "display_height": 25}'
    state, text, _, error = app._save_bbox(first, source, state, 1, 0)
    assert error == "" and state["1"] == {"frame_idx": 0, "bbox": [40, 20, 78, 38]}
    assert "width = 39" in text
    second = '{"x": -5, "y": -3, "width": 80, "height": 40, "display_width": 50, "display_height": 25}'
    state, _, _, _ = app._save_bbox(second, source, state, 1, 0)
    assert state["1"] == {"frame_idx": 0, "bbox": [0, 0, 99, 49]}
    state, text, render, _ = app._clear_bbox(state, 1)
    assert state["1"] is None and "尚未" in text and render == ""


def test_invalid_bbox_and_object_bbox_state_do_not_enable_other_object(tmp_path: Path) -> None:
    source = _source(tmp_path)
    state = app._empty_bbox_state()
    state, _, _, error = app._save_bbox('{"x": 1, "y": 1, "width": 0, "height": 3, "display_width": 100, "display_height": 50}', source, state, 1, 0)
    assert "框选无效" in error
    state["1"] = {"frame_idx": 0, "bbox": [1, 1, 4, 4]}
    assert app._bbox_for(state, 2, 0) is None
    assert app._bbox_for(state, 1, 1) is None


def test_tracking_requires_session_bbox_and_passes_exact_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source(tmp_path)
    state = {"1": {"frame_idx": 1, "bbox": [2, 3, 9, 10]}, "2": None}
    no_box = list(app._start_tracking(source, state, 1, "drone_1", 0, 3))
    assert "请先在起始帧框选目标" in no_box[-1][0]
    captured: dict[str, object] = {}
    def fake_run(**kwargs: object) -> InitialTrackingResult:
        captured.update(kwargs)
        return InitialTrackingResult(1, "drone_1", 1, 1, 3, Path("prompt"), Path("local"), Path("segment"), Path("out"), Path("tracks"), Path("overlay.mp4"))
    monkeypatch.setattr(app, "run_initial_tracking", fake_run)
    result = list(app._start_tracking(source, state, 1, "drone_1", 1, 3))
    assert result[0][0] == "正在传播，请稍候……"
    assert result[-1][1] == "overlay.mp4"
    assert captured["start_frame"] == 1 and captured["end_frame"] == 3 and captured["bbox_xyxy"] == (2, 3, 9, 10)


def test_app_has_rectangle_overlay_and_all_primary_event_callbacks() -> None:
    demo = app.build_app()
    assert "ImageEditor" not in Path(app.__file__).read_text(encoding="utf-8")
    assert "cv2.selectROI" not in Path(app.__file__).read_text(encoding="utf-8")
    assert "subprocess" not in Path(app.__file__).read_text(encoding="utf-8")
    assert "pointerdown" in app.RECTANGLE_OVERLAY_JS
    assert len(demo.config["dependencies"]) >= 10
