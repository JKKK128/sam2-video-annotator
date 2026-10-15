"""High-level, workspace-isolated operations for the local Web UI."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Literal, Sequence
from uuid import uuid4

from scripts.create_tracking_segment import create_tracking_segment
from scripts.extract_frames import extract_frames
from scripts.register_object import register_object
from scripts.render_multi_overlay import render_multi_overlay
from scripts.track_registered_object import track_registered_object
from scripts.create_handoff import create_handoff
from scripts.build_object_track import build_object_track
from scripts.finalize_video_annotation import finalize_video_annotation
from sam2_annotator.drone_catalog import get_catalog_entry, load_catalog
from sam2_annotator.frame_source import prepare_image_sequence
from sam2_annotator.ui_bbox import normalize_browser_bbox
from sam2_annotator.handoff import (
    backup_and_write_timeline,
    identity_for_object,
    replace_timeline_range,
    validate_candidate_records,
)
from sam2_annotator.object_registry import load_registry
from sam2_annotator.tracking_io import load_jsonl


SourceType = Literal["video", "image_sequence"]


@dataclass(frozen=True)
class SourceInfo:
    """Canonical source details made available to both UI tabs."""

    session_dir: Path
    frames_dir: Path
    registry_path: Path
    source_type: SourceType
    source_path: Path
    frame_count: int
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class InitialTrackingResult:
    """Artifacts made by an independent initial propagation."""

    obj_id: int
    global_drone_id: str
    class_id: int
    start_frame: int
    end_frame: int
    prompt_path: Path
    local_prompt_path: Path
    segment_dir: Path
    output_dir: Path
    tracks_path: Path
    overlay_path: Path


@dataclass(frozen=True)
class ManualRestartResult:
    """Review-only artifacts from a fresh manual hard-restart state."""

    obj_id: int
    restart_frame: int
    end_frame: int
    correction_dir: Path
    candidate_tracks_path: Path
    preview_path: Path


@dataclass(frozen=True)
class AcceptedManualRestartResult:
    """Session-local artifacts after an explicit candidate acceptance."""

    timeline_path: Path
    final_tracks_path: Path
    current_overlay_path: Path


@dataclass(frozen=True)
class FinalExportResult:
    """Committed final-export artifacts for one UI session."""

    output_dir: Path
    merged_tracks_path: Path
    labels_dir: Path
    quality_report_path: Path
    overlay_path: Path
    valid: bool
    summary: dict[str, object]


def _session_export_paths(session: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Return the fixed session-local final export paths."""
    output = session / "final"
    return output, output / "merged" / "tracks.jsonl", output / "labels", output / "quality_report.json", output / "overlay.mp4"


def _validate_final_export_inputs(session: Path, catalog_path: Path) -> tuple[Path, Path, Path, Path]:
    """Perform UI-friendly preflight checks without changing any artifacts."""
    if not session.is_dir():
        raise FileNotFoundError(f"UI session does not exist: {session}")
    frames = session / "frames"
    registry_path = session / "object_registry.json"
    timeline_path = _timeline_path(session)
    if not frames.is_dir():
        raise FileNotFoundError(f"session frames_dir does not exist: {frames}")
    if not registry_path.is_file():
        raise FileNotFoundError(f"session registry does not exist: {registry_path}")
    if not timeline_path.is_file():
        raise FileNotFoundError(f"session timeline does not exist: {timeline_path}")
    if not catalog_path.is_file():
        raise FileNotFoundError(f"drone catalog does not exist: {catalog_path}")

    registry = load_registry(registry_path)
    catalog = load_catalog(catalog_path)
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    if not isinstance(timeline, dict):
        raise ValueError("session object_timelines.json must be a JSON object")
    entries = timeline.get("objects")
    if not isinstance(entries, list):
        raise ValueError("session timeline objects must be a list")
    active = [item for item in registry["objects"] if item["status"] == "active" and item["validated"] is True]
    if not active:
        raise ValueError("no active, validated objects are ready for final export")
    for obj in active:
        obj_id = int(obj["obj_id"])
        entry = next((item for item in entries if isinstance(item, dict) and item.get("obj_id") == obj_id), None)
        if entry is None:
            raise ValueError(f"obj{obj_id} has no timeline; complete initial propagation first")
        if entry.get("global_drone_id") != obj["global_drone_id"]:
            raise ValueError(f"obj{obj_id} timeline global_drone_id disagrees with the session registry")
        if entry.get("class_id") != obj["class_id"]:
            raise ValueError(f"obj{obj_id} timeline class_id disagrees with the session registry")
        catalog_entry = get_catalog_entry(catalog, str(obj["global_drone_id"]))
        if int(obj["class_id"]) != int(catalog_entry["class_id"]):
            raise ValueError(f"obj{obj_id} class_id disagrees with catalog global mapping")
        frame_range = entry.get("frame_range")
        if not isinstance(frame_range, list) or len(frame_range) != 2 or not all(isinstance(value, int) for value in frame_range):
            raise ValueError(f"obj{obj_id} timeline frame_range is invalid")
        if frame_range[0] < 0 or frame_range[1] < frame_range[0]:
            raise ValueError(f"obj{obj_id} timeline frame_range is invalid")
        segments = entry.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"obj{obj_id} timeline has no track segments")
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError(f"obj{obj_id} timeline has an invalid segment")
            start, end = segment.get("start_frame"), segment.get("end_frame")
            if not isinstance(start, int) or not isinstance(end, int) or start < frame_range[0] or end < start or end > frame_range[1]:
                raise ValueError(f"obj{obj_id} timeline segment frame range is invalid")
            tracks = segment.get("tracks_path")
            if not isinstance(tracks, str) or not Path(tracks).expanduser().is_file():
                raise FileNotFoundError(f"obj{obj_id} timeline references missing tracks file: {tracks}")
    return timeline_path, registry_path, catalog_path, frames


def run_final_export(*, session_dir: Path | str, catalog_path: Path | str, render_overlay: bool = True) -> FinalExportResult:
    """Finalize latest session timeline through the existing transactional exporter."""
    session = Path(session_dir).expanduser().resolve()
    catalog = Path(catalog_path).expanduser().resolve()
    timeline, registry, catalog, frames = _validate_final_export_inputs(session, catalog)
    output, merged, labels, report, overlay = _session_export_paths(session)
    summary = finalize_video_annotation(timeline, registry, catalog, frames, output, True, render_overlay)
    report_data: dict[str, object] = {}
    if report.is_file():
        loaded = json.loads(report.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            report_data = loaded
    # The report has line counts but not a label-file count; enumerate only the
    # committed label directory, never regenerate or parse label contents.
    combined = {**summary, **report_data, "label_file_count": len(list(labels.glob("*.txt")))}
    valid = bool(report_data.get("valid", summary.get("quality_valid", False)))
    return FinalExportResult(output, merged, labels, report, overlay, valid, combined)


def create_ui_session(workspace: Path | str = "ui_workspace") -> Path:
    """Create a new unique UI session without touching experiment directories."""
    workspace_path = Path(workspace).expanduser().resolve()
    sessions_dir = workspace_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    while True:
        session_dir = sessions_dir / uuid4().hex
        try:
            session_dir.mkdir()
        except FileExistsError:
            continue
        for name in ("prompts", "segments", "outputs"):
            (session_dir / name).mkdir()
        return session_dir


def _source_metadata(frames: Path) -> tuple[int, int, int, float]:
    metadata_path = frames / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"frames metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (int(metadata["frame_count"]), int(metadata["width"]), int(metadata["height"]), float(metadata["fps"]))


def _timeline_path(session: Path) -> Path:
    return session / "object_timelines.json"


def _initialize_timeline_if_needed(result: InitialTrackingResult) -> Path:
    """Add the object's original segment once, only in its UI session."""
    timeline_path = _timeline_path(result.output_dir.parents[1])
    timeline = {"objects": []}
    if timeline_path.is_file():
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    objects = timeline.setdefault("objects", [])
    if not isinstance(objects, list):
        raise ValueError("session object_timelines.json objects must be a list")
    if any(isinstance(entry, dict) and entry.get("obj_id") == result.obj_id for entry in objects):
        return timeline_path
    objects.append({
        "obj_id": result.obj_id,
        "global_drone_id": result.global_drone_id,
        "class_id": result.class_id,
        "frame_range": [result.start_frame, result.end_frame],
        "segments": [{
            "start_frame": result.start_frame,
            "end_frame": result.end_frame,
            "source_type": "original",
            "tracks_path": str(result.tracks_path),
        }],
    })
    timeline_path.write_text(json.dumps(timeline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return timeline_path


def prepare_ui_source(
    source_type: SourceType | str,
    source_path: Path | str,
    *,
    workspace: Path | str = "ui_workspace",
    image_fps: float = 10.0,
    jpeg_quality: int = 95,
) -> SourceInfo:
    """Copy/materialize a video or image sequence into a new UI session.

    The original input is never modified.  Both source kinds result in the
    same canonical zero-padded ``frames_dir`` and a session-local registry.
    """
    if source_type not in ("video", "image_sequence"):
        raise ValueError("source_type must be 'video' or 'image_sequence'")
    source = Path(source_path).expanduser().resolve()
    session_dir = create_ui_session(workspace)
    frames_dir = session_dir / "frames"

    if source_type == "video":
        if not source.is_file():
            raise FileNotFoundError(f"video source does not exist: {source}")
        metadata = extract_frames(source, frames_dir, jpeg_quality, False)
        frame_count = int(metadata["frame_count"])
        width = int(metadata["width"])
        height = int(metadata["height"])
        fps = float(metadata["fps"])
    else:
        if not source.is_dir():
            raise NotADirectoryError(f"image source directory does not exist: {source}")
        prepared = prepare_image_sequence(source, frames_dir, image_fps)
        frame_count = prepared.frame_count
        width = prepared.width
        height = prepared.height
        fps = prepared.fps

    if frame_count <= 0 or width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("source metadata must contain positive frame_count, width, height, and fps")
    registry_path = session_dir / "object_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "video": str(source),
                "frames_dir": str(frames_dir),
                "objects": [],
                "invalid_objects": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return SourceInfo(
        session_dir=session_dir,
        frames_dir=frames_dir,
        registry_path=registry_path,
        source_type=source_type,
        source_path=source,
        frame_count=frame_count,
        width=width,
        height=height,
        fps=fps,
    )


def run_initial_tracking(
    *,
    frames_dir: Path | str,
    registry_path: Path | str,
    catalog_path: Path | str,
    config_path: Path | str,
    obj_id: int,
    global_drone_id: str,
    start_frame: int,
    end_frame: int,
    bbox_xyxy: Sequence[float | int],
    session_dir: Path | str,
    fps: float | None = None,
) -> InitialTrackingResult:
    """Register, segment, propagate, and render one session-local object."""
    if obj_id not in (1, 2):
        raise ValueError("UI V1 supports obj_id 1 or 2")
    frames = Path(frames_dir).expanduser().resolve()
    registry = Path(registry_path).expanduser().resolve()
    catalog = Path(catalog_path).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    session = Path(session_dir).expanduser().resolve()
    if not frames.is_dir():
        raise FileNotFoundError(f"frames_dir does not exist: {frames}")
    if not registry.is_file():
        raise FileNotFoundError(f"registry does not exist: {registry}")
    if not session.is_dir():
        raise FileNotFoundError(f"session_dir does not exist: {session}")
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("start_frame/end_frame must be non-negative and increasing")
    frame_count, width, height, metadata_fps = _source_metadata(frames)
    if end_frame >= frame_count:
        raise ValueError(f"end_frame {end_frame} is outside source frame_count {frame_count}")
    if fps is None:
        fps = metadata_fps
    box = normalize_browser_bbox(bbox_xyxy, original_width=width, original_height=height)
    catalog_entry = get_catalog_entry(load_catalog(catalog), global_drone_id)

    prompts_dir = session / "prompts"
    segment_dir = session / "segments" / f"obj{obj_id}_{start_frame:06d}_{end_frame:06d}"
    output_dir = session / "outputs" / f"obj{obj_id}"
    prompt_path = prompts_dir / f"obj{obj_id}_global.json"
    local_prompt_path = prompts_dir / f"obj{obj_id}_local.json"
    tracks_path = output_dir / "tracks.jsonl"
    overlay_path = output_dir / "initial_overlay.mp4"
    if any(path.exists() for path in (segment_dir, output_dir, prompt_path, local_prompt_path)):
        raise FileExistsError(f"obj_id {obj_id} already has UI artifacts in this session")

    register_object(
        registry, frames, obj_id, f"obj{obj_id}", catalog, global_drone_id,
        start_frame, prompt_path, box, False,
    )
    create_tracking_segment(
        frames, start_frame, end_frame, segment_dir, prompt_path, local_prompt_path, False,
    )
    track_registered_object(config, registry, obj_id, segment_dir, local_prompt_path, output_dir, None)
    render_multi_overlay(frames, tracks_path, overlay_path, start_frame, end_frame, fps)
    result = InitialTrackingResult(
        obj_id=obj_id,
        global_drone_id=global_drone_id,
        class_id=int(catalog_entry["class_id"]),
        start_frame=start_frame,
        end_frame=end_frame,
        prompt_path=prompt_path,
        local_prompt_path=local_prompt_path,
        segment_dir=segment_dir,
        output_dir=output_dir,
        tracks_path=tracks_path,
        overlay_path=overlay_path,
    )
    _initialize_timeline_if_needed(result)
    return result


def run_manual_restart(
    *, frames_dir: Path | str, registry_path: Path | str, catalog_path: Path | str,
    config_path: Path | str, session_dir: Path | str, obj_id: int,
    restart_frame: int, end_frame: int, bbox_xyxy: Sequence[float | int],
) -> ManualRestartResult:
    """Create a review-only manual candidate using a brand-new SAM2 state."""
    frames, registry, catalog = Path(frames_dir).resolve(), Path(registry_path).resolve(), Path(catalog_path).resolve()
    config, session = Path(config_path).resolve(), Path(session_dir).resolve()
    if obj_id not in (1, 2):
        raise ValueError("UI V1 supports obj_id 1 or 2")
    if not frames.is_dir() or not registry.is_file() or not session.is_dir():
        raise FileNotFoundError("frames_dir, registry, and session_dir must exist")
    frame_count, width, height, _ = _source_metadata(frames)
    if restart_frame < 0 or restart_frame >= frame_count:
        raise ValueError("restart_frame is outside source frame_count")
    if end_frame < restart_frame or end_frame >= frame_count:
        raise ValueError("end_frame must be within source and no earlier than restart_frame")
    box = normalize_browser_bbox(bbox_xyxy, original_width=width, original_height=height)
    timeline = _timeline_path(session)
    if not timeline.is_file():
        raise FileNotFoundError("initial propagation timeline does not exist for this UI session")
    entry = next((item for item in json.loads(timeline.read_text(encoding="utf-8")).get("objects", []) if item.get("obj_id") == obj_id), None)
    if not isinstance(entry, dict):
        raise ValueError("run initial propagation for this object before a manual restart")
    first_tracks = next((segment.get("tracks_path") for segment in entry.get("segments", []) if isinstance(segment, dict)), None)
    if not isinstance(first_tracks, str):
        raise ValueError("object timeline has no source tracks")
    correction_root = session / "outputs" / f"obj{obj_id}" / "corrections"
    correction_dir = correction_root / f"manual_{restart_frame:06d}_{end_frame:06d}_{uuid4().hex[:8]}"
    args = argparse.Namespace(config=config, registry=registry, catalog=catalog, frames_dir=frames,
        tracks=Path(first_tracks), output_dir=correction_dir, obj_id=obj_id, source_frame=restart_frame,
        end_frame=end_frame, prompt_mode="manual-box", intervals=None, interval_id=None,
        render_preview=True, overwrite=False, reason="UI manual hard restart")
    summary = create_handoff(args, manual_box=box)
    return ManualRestartResult(obj_id, restart_frame, end_frame, correction_dir,
        Path(summary["tracks_path"]), Path(summary["preview_path"]))


def accept_manual_restart(
    *, frames_dir: Path | str, registry_path: Path | str, catalog_path: Path | str,
    session_dir: Path | str, obj_id: int, candidate_tracks_path: Path | str,
    restart_frame: int, end_frame: int, fps: float | None = None,
) -> AcceptedManualRestartResult:
    """Explicitly merge one candidate, then rebuild this object's current view."""
    frames, registry, catalog = Path(frames_dir).resolve(), Path(registry_path).resolve(), Path(catalog_path).resolve()
    session, tracks = Path(session_dir).resolve(), Path(candidate_tracks_path).resolve()
    timeline = _timeline_path(session)
    if not timeline.is_file() or not tracks.is_file():
        raise FileNotFoundError("session timeline and candidate tracks must exist")
    identity = identity_for_object(load_registry(registry), load_catalog(catalog), obj_id)
    validate_candidate_records(load_jsonl(tracks), identity, restart_frame, end_frame)
    current = json.loads(timeline.read_text(encoding="utf-8"))
    segment = {"source_type": "manual_hard_restart", "restart_frame": restart_frame, "tracks_path": str(tracks)}
    updated = replace_timeline_range(current, obj_id, restart_frame, end_frame, segment, force=True)
    backup_and_write_timeline(timeline, updated)
    output_dir = session / "outputs" / f"obj{obj_id}"
    final_tracks = output_dir / "final_tracks.jsonl"
    build_object_track(timeline, registry, catalog, obj_id, final_tracks, True)
    entry = next(item for item in updated["objects"] if item.get("obj_id") == obj_id)
    start, finish = (int(value) for value in entry["frame_range"])
    if fps is None:
        _, _, _, fps = _source_metadata(frames)
    overlay = output_dir / "current_overlay.mp4"
    render_multi_overlay(frames, final_tracks, overlay, start, finish, fps)
    return AcceptedManualRestartResult(timeline, final_tracks, overlay)
