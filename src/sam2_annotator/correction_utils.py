"""Helpers for correction prompts and corrected track merging."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sam2_annotator.mask_utils import BBoxXYXY, bbox_to_yolo


def validate_box_xyxy(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """Validate an inclusive xyxy box inside an image."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = [int(v) for v in box]
    if min(x1, y1, x2, y2) < 0:
        raise ValueError(f"box coordinates must be non-negative: {box}")
    if x2 >= width or y2 >= height:
        raise ValueError(f"box {box} exceeds frame bounds {width}x{height}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box must have non-zero area with x2>x1 and y2>y1: {box}")
    return x1, y1, x2, y2


def validate_correction_obj_id(initial_prompt: dict[str, Any], correction_prompt: dict[str, Any]) -> int:
    """Return the shared obj_id, rejecting correction prompts that create a new object."""
    initial_obj_id = int(initial_prompt["obj_id"])
    correction_obj_id = int(correction_prompt["obj_id"])
    if correction_obj_id != initial_obj_id:
        raise ValueError(f"correction obj_id {correction_obj_id} must match initial obj_id {initial_obj_id}")
    return correction_obj_id


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically by replacing the target from a temp file in the same directory."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, indent=2) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    lines = "".join(json.dumps(record) + "\n" for record in records)
    atomic_write_text(path, lines)


def merge_track_records(
    baseline_records: list[dict[str, Any]],
    corrected_records: list[dict[str, Any]],
    correction_frame: int,
    obj_id: int | None = None,
) -> list[dict[str, Any]]:
    """Merge correction records.

    When obj_id is provided, only that object's records at correction_frame and
    after are replaced. Other objects remain exactly as they appear in baseline.
    """
    merged_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for record in baseline_records:
        key = _record_key(record)
        if obj_id is None:
            keep = key[0] < correction_frame
        else:
            keep = key[1] != obj_id or key[0] < correction_frame
        if keep:
            merged_by_key[key] = dict(record)
    for record in corrected_records:
        key = _record_key(record)
        if key[0] >= correction_frame and (obj_id is None or key[1] == obj_id):
            merged_by_key[key] = dict(record)
    merged = [merged_by_key[key] for key in sorted(merged_by_key)]
    if len(merged) != len({_record_key(record) for record in merged}):
        raise ValueError("merged tracks contain duplicate (frame_idx, obj_id)")
    return merged


def write_label_for_record(record: dict[str, Any], labels_dir: Path, image_width: int, image_height: int) -> Path:
    """Write one YOLO label txt from record['bbox_xyxy']; empty masks produce empty files."""
    frame_idx = int(record["frame_idx"])
    labels_dir.mkdir(parents=True, exist_ok=True)
    path = labels_dir / f"{frame_idx:06d}.txt"
    bbox_value = record.get("bbox_xyxy")
    if bbox_value is None:
        atomic_write_text(path, "")
        return path
    bbox: BBoxXYXY = tuple(int(v) for v in bbox_value)  # type: ignore[assignment]
    cx, cy, bw, bh = bbox_to_yolo(bbox, image_width, image_height)
    atomic_write_text(path, f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n")
    return path


def render_range(
    available_frames: list[int],
    correction_frame: int,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[int, int]:
    """Compute and clip the default correction comparison frame range."""
    if not available_frames:
        raise ValueError("available_frames must not be empty")
    min_frame = min(available_frames)
    max_frame = max(available_frames)
    start = correction_frame - 5 if start_frame is None else start_frame
    end = correction_frame + 50 if end_frame is None else end_frame
    start = max(min_frame, int(start))
    end = min(max_frame, int(end))
    if end < start:
        raise ValueError(f"invalid render range after clipping: {start}..{end}")
    return start, end


def _record_key(record: dict[str, Any]) -> tuple[int, int]:
    if "frame_idx" not in record or "obj_id" not in record:
        raise ValueError("track record missing frame_idx or obj_id")
    return int(record["frame_idx"]), int(record["obj_id"])
