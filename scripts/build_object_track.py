#!/usr/bin/env python3
"""Build one complete object track from the segments in an object timeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.drone_catalog import get_catalog_entry, load_catalog
from sam2_annotator.object_registry import get_object, load_registry
from sam2_annotator.tracking_io import ensure_unique_track_keys, load_jsonl


LOGGER = logging.getLogger("build_object_track")


def _load_timeline(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"object timeline does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("object timeline must be a JSON object")
    return data


def _as_frame_range(value: Any, context: str) -> tuple[int, int]:
    if isinstance(value, dict):
        start, end = value.get("start_frame"), value.get("end_frame")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = value
    else:
        raise ValueError(f"{context} must be [start_frame, end_frame] or an object with those keys")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        raise ValueError(f"{context} must be non-negative and increasing")
    return start, end


def _timeline_object(timeline: dict[str, Any], obj_id: int) -> dict[str, Any]:
    """Find an object entry in either a list or an obj-id keyed mapping."""
    objects = timeline.get("objects", timeline.get("object_timelines"))
    if isinstance(objects, list):
        for entry in objects:
            if isinstance(entry, dict) and entry.get("obj_id") == obj_id:
                return entry
    elif isinstance(objects, dict):
        entry = objects.get(str(obj_id), objects.get(obj_id))
        if isinstance(entry, dict):
            result = dict(entry)
            result.setdefault("obj_id", obj_id)
            return result
    # Also accept the compact {"1": {...}} form.
    entry = timeline.get(str(obj_id), timeline.get(obj_id))
    if isinstance(entry, dict):
        result = dict(entry)
        result.setdefault("obj_id", obj_id)
        return result
    raise ValueError(f"obj_id {obj_id} has no timeline entry")


def _segment_range(segment: dict[str, Any], index: int) -> tuple[int, int]:
    if "frame_range" in segment:
        return _as_frame_range(segment["frame_range"], f"segment {index} frame_range")
    return _as_frame_range(segment, f"segment {index}")


def _video_id(registry: dict[str, Any]) -> str:
    value = registry.get("video_id")
    return str(value) if isinstance(value, str) and value else Path(str(registry["video"])).stem


def _validate_authoritative_identity(registry_object: dict[str, Any], catalog_entry: dict[str, Any]) -> None:
    if int(registry_object["class_id"]) != int(catalog_entry["class_id"]):
        raise ValueError("registry class_id disagrees with catalog global mapping")
    class_name = registry_object.get("class_name")
    if class_name is not None and class_name != catalog_entry["class_name"]:
        raise ValueError("registry class_name disagrees with catalog global mapping")


def _hydrate_record(
    raw_record: dict[str, Any],
    *,
    obj_id: int,
    identity: dict[str, Any],
    video_id: str,
    source_type: str,
    source_tracks_path: Path,
    restart_frame: int | None,
) -> dict[str, Any]:
    if raw_record.get("obj_id") != obj_id:
        raise ValueError(f"track record obj_id {raw_record.get('obj_id')!r} does not match requested obj_id {obj_id}")
    for key in ("global_drone_id", "class_id", "class_name", "video_id"):
        expected = identity[key] if key in identity else video_id
        if key in raw_record and raw_record[key] != expected:
            raise ValueError(f"track record {key} disagrees with registry/catalog for obj_id {obj_id}")
    record = dict(raw_record)
    record.update(identity)
    record["video_id"] = video_id
    record["source_type"] = source_type
    record["source_tracks_path"] = str(source_tracks_path)
    if source_type == "hard_restart":
        if restart_frame is None:
            raise ValueError("hard_restart segment must define restart_frame")
        record["restart_frame"] = restart_frame
    return record


def _atomic_write_jsonl(output: Path, records: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, text=True)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_object_track(timeline_path: Path, registry_path: Path, catalog_path: Path, obj_id: int, output: Path, overwrite: bool) -> dict[str, Any]:
    """Validate and concatenate one object's contiguous timeline segments."""
    if obj_id <= 0:
        raise ValueError("obj_id must be a positive integer")
    timeline_path = timeline_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")

    timeline_entry = _timeline_object(_load_timeline(timeline_path), obj_id)
    frame_range = _as_frame_range(timeline_entry.get("frame_range"), "timeline frame_range")
    segments = timeline_entry.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"timeline obj_id {obj_id} segments must be a non-empty list")

    registry = load_registry(registry_path)
    try:
        registry_object = get_object(registry, obj_id)
    except KeyError as exc:
        raise ValueError(f"obj_id {obj_id} does not exist in the current video registry") from exc
    catalog_entry = get_catalog_entry(load_catalog(catalog_path), str(registry_object["global_drone_id"]))
    _validate_authoritative_identity(registry_object, catalog_entry)
    identity = dict(catalog_entry)
    video_id = _video_id(registry)

    normalized: list[tuple[int, int, int, dict[str, Any]]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"segment {index} must be an object")
        start, end = _segment_range(segment, index)
        normalized.append((start, end, index, segment))
    normalized.sort(key=lambda item: (item[0], item[1], item[2]))

    expected_start, expected_end = frame_range
    all_records: list[dict[str, Any]] = []
    segment_counts: list[dict[str, Any]] = []
    previous_end: int | None = None
    for start, end, index, segment in normalized:
        if previous_end is None:
            if start != expected_start:
                raise ValueError(f"first segment starts at {start}, but frame_range starts at {expected_start}")
        elif start <= previous_end:
            raise ValueError(f"segments overlap: segment {index} starts at {start} after previous end {previous_end}")
        elif start != previous_end + 1:
            raise ValueError(f"segments have missing frames between {previous_end} and {start}")
        previous_end = end

        tracks_value = segment.get("tracks_path")
        if not isinstance(tracks_value, str) or not tracks_value:
            raise ValueError(f"segment {index} tracks_path must be a non-empty string")
        tracks_path = Path(tracks_value).expanduser()
        if not tracks_path.is_absolute():
            tracks_path = timeline_path.parent / tracks_path
        tracks_path = tracks_path.resolve()
        if not tracks_path.is_file():
            raise FileNotFoundError(f"segment {index} tracks_path does not exist: {tracks_path}")
        source_type = segment.get("source_type", "original")
        if not isinstance(source_type, str) or not source_type:
            raise ValueError(f"segment {index} source_type must be a non-empty string")
        restart_frame = segment.get("restart_frame")
        if restart_frame is not None and (not isinstance(restart_frame, int) or restart_frame < 0 or restart_frame > start):
            raise ValueError(
                f"segment {index} restart_frame must be a non-negative integer no later than its start_frame"
            )

        selected = [record for record in load_jsonl(tracks_path) if record.get("obj_id") == obj_id and isinstance(record.get("frame_idx"), int) and start <= record["frame_idx"] <= end]
        selected.sort(key=lambda record: int(record["frame_idx"]))
        frames = [int(record["frame_idx"]) for record in selected]
        expected_frames = list(range(start, end + 1))
        if frames != expected_frames:
            missing = sorted(set(expected_frames) - set(frames))
            duplicates = sorted(frame for frame in set(frames) if frames.count(frame) > 1)
            detail = f"missing frames {missing}" if missing else f"duplicate frames {duplicates}"
            raise ValueError(f"segment {index} does not contain exactly one record per frame: {detail}")
        hydrated = [_hydrate_record(record, obj_id=obj_id, identity=identity, video_id=video_id, source_type=source_type, source_tracks_path=tracks_path, restart_frame=restart_frame) for record in selected]
        all_records.extend(hydrated)
        segment_counts.append({"start_frame": start, "end_frame": end, "source_type": source_type, "tracks_path": str(tracks_path), "record_count": len(hydrated)})

    if previous_end != expected_end:
        raise ValueError(f"last segment ends at {previous_end}, but frame_range ends at {expected_end}")
    all_records.sort(key=lambda record: (int(record["frame_idx"]), int(record["obj_id"])))
    ensure_unique_track_keys(all_records)
    actual_range = (int(all_records[0]["frame_idx"]), int(all_records[-1]["frame_idx"])) if all_records else None
    if actual_range != frame_range:
        raise ValueError(f"final frame range {actual_range} does not match timeline frame_range {frame_range}")
    if len(all_records) != expected_end - expected_start + 1:
        raise ValueError("final track contains missing or duplicate frames")

    _atomic_write_jsonl(output, all_records)
    summary = {"obj_id": obj_id, "global_drone_id": identity["global_drone_id"], "class_id": identity["class_id"], "frame_range": list(frame_range), "segments": segment_counts, "total_records": len(all_records), "has_missing_frames": False, "has_duplicate_frames": False, "output": str(output)}
    LOGGER.info("build summary: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build one complete, contiguous object track from original and hard-restart timeline segments.")
    parser.add_argument("--timeline", type=Path, default=Path("data/object_timelines.json"), help="Object timeline JSON. Default: data/object_timelines.json")
    parser.add_argument("--registry", type=Path, default=Path("data/object_registry.json"), help="Per-video object registry JSON. Default: data/object_registry.json")
    parser.add_argument("--catalog", type=Path, default=Path("configs/drone_catalog.json"), help="Global drone catalog JSON. Default: configs/drone_catalog.json")
    parser.add_argument("--obj-id", required=True, type=int, help="Local object id to build.")
    parser.add_argument("--output", required=True, type=Path, help="Output single-object tracks JSONL.")
    parser.add_argument("--overwrite", action="store_true", help="Atomically replace an existing output file.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    build_object_track(args.timeline, args.registry, args.catalog, args.obj_id, args.output, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
