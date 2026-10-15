"""Shared IO helpers for global-frame SAM2 object tracks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from sam2_annotator.frame_utils import frame_name
from sam2_annotator.mask_utils import BBoxXYXY, bbox_to_yolo


def load_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path} line {line_no} is not a JSON object")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def load_segment_manifest(segment_dir: Path) -> dict[str, Any]:
    manifest = load_json(segment_dir.expanduser().resolve() / "segment_manifest.json")
    validate_segment_manifest(manifest)
    return manifest


def validate_segment_manifest(manifest: dict[str, Any]) -> None:
    mapping = manifest.get("frames")
    if not isinstance(mapping, list) or not mapping:
        raise ValueError("segment_manifest frames must be a non-empty list")
    seen_local: set[int] = set()
    seen_original: set[int] = set()
    for item in mapping:
        if not isinstance(item, dict):
            raise ValueError("segment_manifest frame mapping entries must be objects")
        local = int(item["local_frame_idx"])
        original = int(item["original_frame_idx"])
        if local < 0 or original < 0:
            raise ValueError("frame indexes must be non-negative")
        if local in seen_local:
            raise ValueError(f"duplicate local_frame_idx: {local}")
        if original in seen_original:
            raise ValueError(f"duplicate original_frame_idx: {original}")
        seen_local.add(local)
        seen_original.add(original)


def local_to_original_map(manifest: dict[str, Any]) -> dict[int, int]:
    validate_segment_manifest(manifest)
    return {int(item["local_frame_idx"]): int(item["original_frame_idx"]) for item in manifest["frames"]}


def original_to_local_map(manifest: dict[str, Any]) -> dict[int, int]:
    validate_segment_manifest(manifest)
    return {int(item["original_frame_idx"]): int(item["local_frame_idx"]) for item in manifest["frames"]}


def convert_prompt_to_segment(prompt: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    original_frame = int(prompt.get("original_frame_idx", prompt["frame_idx"]))
    original_to_local = original_to_local_map(manifest)
    if original_frame not in original_to_local:
        raise ValueError(f"prompt original frame {original_frame} is outside segment")
    converted = dict(prompt)
    converted["original_frame_idx"] = original_frame
    converted["frame_idx"] = original_to_local[original_frame]
    converted["local_frame_idx"] = original_to_local[original_frame]
    return converted


def write_yolo_label(path: Path, bbox: BBoxXYXY | None, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if bbox is None:
        path.write_text("", encoding="utf-8")
        return
    cx, cy, bw, bh = bbox_to_yolo(bbox, width, height)
    path.write_text(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n", encoding="utf-8")


def yolo_line_for_record(record: dict[str, Any], width: int, height: int) -> str | None:
    bbox_value = record.get("bbox_xyxy")
    if bbox_value is None or record.get("empty_mask"):
        return None
    bbox: BBoxXYXY = tuple(int(v) for v in bbox_value)  # type: ignore[assignment]
    cx, cy, bw, bh = bbox_to_yolo(bbox, width, height)
    class_id = record.get("class_id")
    if not isinstance(class_id, int) or class_id < 0:
        raise ValueError("track record class_id must be a non-negative integer")
    return f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"


def write_multi_object_labels(records: Iterable[dict[str, Any]], labels_dir: Path, width: int, height: int, full_frame_range: tuple[int, int] | None = None) -> None:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["frame_idx"]), []).append(record)
    if full_frame_range is not None:
        start, end = full_frame_range
        if start < 0 or end < start:
            raise ValueError("full frame range must be non-negative and increasing")
        for frame_idx in range(start, end + 1):
            grouped.setdefault(frame_idx, [])
    labels_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, frame_records in sorted(grouped.items()):
        lines = [
            line
            for line in (yolo_line_for_record(record, width, height) for record in sorted(frame_records, key=lambda item: int(item["obj_id"])))
            if line is not None
        ]
        (labels_dir / f"{frame_idx:06d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def merge_records_by_object_correction(
    baseline_records: list[dict[str, Any]],
    corrected_records: list[dict[str, Any]],
    correction_frame: int,
    obj_id: int,
) -> list[dict[str, Any]]:
    merged_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for record in baseline_records:
        key = record_key(record)
        if key[1] != obj_id or key[0] < correction_frame:
            merged_by_key[key] = dict(record)
    for record in corrected_records:
        key = record_key(record)
        if key[1] == obj_id and key[0] >= correction_frame:
            merged_by_key[key] = dict(record)
    return [merged_by_key[key] for key in sorted(merged_by_key)]


def record_key(record: dict[str, Any]) -> tuple[int, int]:
    return int(record["frame_idx"]), int(record["obj_id"])


def ensure_unique_track_keys(records: Iterable[dict[str, Any]]) -> None:
    seen: set[tuple[int, int]] = set()
    for record in records:
        key = record_key(record)
        if key in seen:
            raise ValueError(f"duplicate (frame_idx, obj_id): {key}")
        seen.add(key)


def collect_mask_links(track_paths: Iterable[Path], output_dir: Path) -> list[Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    targets: dict[str, Path] = {}
    for tracks_path in track_paths:
        tracks_path = tracks_path.expanduser().resolve()
        for record in load_jsonl(tracks_path):
            mask_path_value = record.get("mask_path")
            if not mask_path_value:
                continue
            source = Path(str(mask_path_value)).expanduser()
            if not source.is_absolute():
                source = tracks_path.parent / source
            source = source.resolve()
            if not source.is_file():
                raise FileNotFoundError(f"mask does not exist: {source}")
            name = f"{int(record['frame_idx']):06d}_obj{int(record['obj_id'])}.png"
            target = output_dir / name
            previous = targets.get(name)
            if previous is not None and previous != source:
                raise FileExistsError(f"mask filename conflict for {name}: {previous} vs {source}")
            if target.exists():
                existing = target.resolve()
                if existing != source:
                    raise FileExistsError(f"mask filename conflict for {target}: {existing} vs {source}")
            else:
                os.symlink(source, target)
                created.append(target)
            targets[name] = source
    return created


def mask_filename(frame_idx: int, obj_id: int) -> str:
    return f"{frame_name(frame_idx, suffix='')}_obj{obj_id}.png"
