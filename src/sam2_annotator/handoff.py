"""Safe candidate handoffs and timeline acceptance for independent SAM2 restarts.

This module deliberately contains no model implementation.  It validates and
prepares prompts, validates generated candidates, and atomically edits only one
object's timeline after an explicit acceptance step.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Callable

import numpy as np

from sam2_annotator.correction_utils import atomic_write_json, atomic_write_text, load_jsonl, validate_box_xyxy
from sam2_annotator.drone_catalog import get_catalog_entry

PROMPT_MODES = {"auto", "mask", "bbox", "manual-box"}
CORRECTION_MODES = {"mask": "auto_mask_handoff", "bbox": "auto_bbox_handoff", "manual-box": "manual_hard_restart"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def identity_for_object(registry: dict[str, Any], catalog: dict[str, dict[str, Any]], obj_id: int) -> dict[str, Any]:
    obj = next((x for x in registry.get("objects", []) if x.get("obj_id") == obj_id), None)
    if not isinstance(obj, dict) or obj.get("status") != "active" or obj.get("validated") is not True:
        raise ValueError(f"obj_id {obj_id} must be an active, validated current-video object")
    entry = get_catalog_entry(catalog, str(obj.get("global_drone_id")))
    if int(obj.get("class_id", -1)) != int(entry["class_id"]) or obj.get("class_name", entry["class_name"]) != entry["class_name"]:
        raise ValueError(f"registry identity for obj_id {obj_id} disagrees with catalog")
    return {"obj_id": obj_id, "global_drone_id": entry["global_drone_id"], "class_id": entry["class_id"], "class_name": entry["class_name"]}


def find_interval(path: Path, interval_id: str) -> dict[str, Any]:
    matches = [row for row in load_jsonl(path) if row.get("interval_id") == interval_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one interval-id {interval_id!r}, found {len(matches)}")
    return matches[0]


def source_record(tracks: Path, obj_id: int, frame_idx: int) -> dict[str, Any]:
    matches = [r for r in load_jsonl(tracks) if r.get("obj_id") == obj_id and r.get("frame_idx") == frame_idx]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one source track record for obj_id={obj_id} frame={frame_idx}, found {len(matches)}")
    return matches[0]


def _is_critical(record: dict[str, Any]) -> bool:
    return str(record.get("severity", "")).lower() == "critical" or bool(record.get("critical_anomaly", False))


def validate_bbox(record: dict[str, Any], width: int, height: int, min_area: int, *, reject_critical: bool = True) -> tuple[int, int, int, int]:
    if reject_critical and _is_critical(record):
        raise ValueError("source frame is a critical anomaly")
    value = record.get("bbox_xyxy")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("source record has no complete bbox_xyxy")
    box = validate_box_xyxy(tuple(int(v) for v in value), width, height)
    if (box[2] - box[0] + 1) * (box[3] - box[1] + 1) < min_area:
        raise ValueError("source bbox area is below handoff.min_bbox_area")
    return box


def validate_mask(record: dict[str, Any], tracks_path: Path, width: int, height: int, min_area: int, *, reject_critical: bool = True) -> tuple[np.ndarray, Path, tuple[int, int, int, int]]:
    import cv2
    if reject_critical and _is_critical(record):
        raise ValueError("source frame is a critical anomaly")
    value = record.get("mask_path")
    if not isinstance(value, str) or not value:
        raise ValueError("source record has no mask_path")
    path = Path(value).expanduser()
    if not path.is_absolute(): path = tracks_path.parent / path
    path = path.resolve()
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
    if mask is None or mask.ndim != 2 or mask.shape != (height, width) or not np.isfinite(mask).all():
        raise ValueError("source mask is missing, unreadable, non-2D, invalid, or mismatched to the frame")
    binary = mask > 0
    if int(binary.sum()) < min_area: raise ValueError("source mask is empty or below handoff.min_mask_area")
    ys, xs = np.where(binary)
    box = validate_box_xyxy((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), width, height)
    return binary, path, box


def select_prompt(requested: str, record: dict[str, Any] | None, tracks_path: Path, width: int, height: int, settings: dict[str, Any], manual_box: tuple[int, int, int, int] | None = None) -> dict[str, Any]:
    """Return selected prompt; auto only falls back from invalid mask to valid bbox."""
    if requested not in PROMPT_MODES: raise ValueError(f"unknown prompt mode: {requested}")
    if requested == "manual-box":
        if manual_box is None: raise ValueError("manual-box selection was cancelled")
        return {"selected": "manual-box", "box": validate_box_xyxy(manual_box, width, height), "mask": None, "mask_path": None}
    if record is None: raise ValueError("source track record is required for automatic handoff")
    reject = bool(settings.get("reject_critical_source_frame", True))
    min_mask, min_bbox = int(settings.get("min_mask_area", 1)), int(settings.get("min_bbox_area", 1))
    mask_error: Exception | None = None
    if requested in {"auto", "mask"}:
        try:
            mask, path, box = validate_mask(record, tracks_path, width, height, min_mask, reject_critical=reject)
            return {"selected": "mask", "box": box, "mask": mask, "mask_path": path}
        except Exception as exc:
            mask_error = exc
            if requested == "mask": raise ValueError(f"mask handoff rejected: {exc}") from exc
    if requested in {"auto", "bbox"}:
        try:
            if requested == "auto" and not bool(settings.get("allow_bbox_fallback", True)):
                raise ValueError("bbox fallback is disabled by handoff.allow_bbox_fallback")
            return {"selected": "bbox", "box": validate_bbox(record, width, height, min_bbox, reject_critical=reject), "mask": None, "mask_path": None}
        except Exception as exc:
            if requested == "bbox": raise ValueError(f"bbox handoff rejected: {exc}") from exc
            raise ValueError(f"auto handoff has no reliable mask ({mask_error}) or bbox ({exc}); use --prompt-mode manual-box") from exc
    raise AssertionError("unreachable")


def save_manual_prompt(path: Path, identity: dict[str, Any], video_id: str, frame_idx: int, box: tuple[int, int, int, int], width: int, height: int, reason: str | None) -> dict[str, Any]:
    prompt = {"schema_version": 1, "video_id": video_id, **identity, "frame_idx": frame_idx, "original_frame_idx": frame_idx, "box_xyxy": list(validate_box_xyxy(box, width, height)), "frame_width": width, "frame_height": height, "prompt_mode": "manual_box", "reason": reason or "manual hard restart", "created_at": utc_now()}
    atomic_write_json(path, prompt)
    return prompt


def validate_candidate_records(records: list[dict[str, Any]], identity: dict[str, Any], start: int, end: int, *, require_masks: bool = True) -> None:
    frames: list[int] = []
    for row in records:
        if row.get("obj_id") != identity["obj_id"]: raise ValueError("candidate contains another obj_id")
        for key in ("global_drone_id", "class_id", "class_name"):
            if row.get(key) != identity[key]: raise ValueError(f"candidate {key} disagrees with registry/catalog")
        frame = row.get("frame_idx")
        if not isinstance(frame, int): raise ValueError("candidate frame_idx must be an integer")
        frames.append(frame)
        width, height = int(row.get("frame_width", 0)), int(row.get("frame_height", 0))
        validate_bbox(row, width, height, 1, reject_critical=False)
        if require_masks and (not isinstance(row.get("mask_path"), str) or int(row.get("mask_area", 0)) <= 0): raise ValueError("candidate record has invalid mask fields")
    expected = list(range(start, end + 1))
    selected = [frame for frame in frames if start <= frame <= end]
    duplicates = sorted(frame for frame in set(frames) if frames.count(frame) > 1)
    if sorted(selected) != expected or duplicates:
        missing = sorted(set(expected) - set(selected))
        raise ValueError(f"candidate must cover every accepted frame exactly once; missing={missing}, duplicates={duplicates}")


def replace_timeline_range(timeline: dict[str, Any], obj_id: int, start: int, end: int, segment: dict[str, Any], *, force: bool) -> dict[str, Any]:
    """Split overlapping segments, preserving every frame outside the accepted range."""
    result = json.loads(json.dumps(timeline))
    objects = result.get("objects")
    if not isinstance(objects, list): raise ValueError("timeline objects must be a list")
    entry = next((o for o in objects if isinstance(o, dict) and o.get("obj_id") == obj_id), None)
    if entry is None: raise ValueError(f"timeline has no obj_id {obj_id}")
    frame_range = entry.get("frame_range")
    if not isinstance(frame_range, list) or len(frame_range) != 2 or start < frame_range[0] or end > frame_range[1]: raise ValueError("accepted range is outside the object's timeline frame_range")
    old = entry.get("segments")
    if not isinstance(old, list): raise ValueError("timeline segments must be a list")
    accepted_types = {"hard_restart", "auto_mask_handoff", "auto_bbox_handoff", "manual_hard_restart"}
    if not force and any(not (end < s["start_frame"] or start > s["end_frame"]) and s.get("source_type") in accepted_types for s in old):
        raise ValueError("range overlaps an accepted correction; use --force or --overwrite-existing-range")
    pieces: list[dict[str, Any]] = []
    for old_seg in old:
        a, b = int(old_seg["start_frame"]), int(old_seg["end_frame"])
        if b < start or a > end: pieces.append(dict(old_seg)); continue
        if a < start:
            left = dict(old_seg); left["end_frame"] = start - 1; pieces.append(left)
        if b > end:
            right = dict(old_seg); right["start_frame"] = end + 1; pieces.append(right)
    new_seg = dict(segment); new_seg.update({"start_frame": start, "end_frame": end})
    pieces.append(new_seg); pieces.sort(key=lambda x: int(x["start_frame"]))
    expected = int(frame_range[0])
    for item in pieces:
        if int(item["start_frame"]) != expected or int(item["end_frame"]) < expected: raise ValueError("timeline update would create a gap or overlap")
        expected = int(item["end_frame"]) + 1
    if expected != int(frame_range[1]) + 1: raise ValueError("timeline update would not fully cover frame_range")
    entry["segments"] = pieces
    return result


def backup_and_write_timeline(path: Path, updated: dict[str, Any], backup_dir: Path | None = None) -> Path:
    path = path.expanduser().resolve(); backup_dir = (backup_dir or path.parent / "backups").expanduser().resolve(); backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{path.stem}.{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    shutil.copy2(path, backup)
    atomic_write_text(path, json.dumps(updated, indent=2, ensure_ascii=False) + "\n")
    return backup
