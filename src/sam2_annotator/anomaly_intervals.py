"""Review-only temporal anomaly detection and stable interval management."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from sam2_annotator.config_utils import nested_get

SEVERITIES = ("warning", "medium", "high", "critical")
REVIEW_STATUSES = {"pending", "confirmed_correct", "needs_restart", "manually_corrected", "occluded", "out_of_view", "ignored", "resolved"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class IntervalParameters:
    baseline_window: int = 5
    bbox_area_warning_ratio: float = 1.4
    bbox_area_high_ratio: float = 1.8
    bbox_area_critical_ratio: float = 2.5
    bbox_area_weight: float = 4.0
    min_consecutive: int = 3
    bbox_area_high_min_consecutive: int = 2
    merge_gap: int = 2
    recovery_frames: int = 3
    restart_lookback: int = 20
    weights: dict[str, float] = field(default_factory=lambda: {"bbox_area_jump": 4.0, "mask_area_jump": 2.0, "center_shift": 1.5, "aspect_ratio_jump": 1.0, "edge_proximity": 0.5, "empty_mask": 5.0, "missing_record": 5.0, "identity_error": 5.0})

    def validate(self) -> None:
        if self.baseline_window < 1 or self.min_consecutive < 1 or self.recovery_frames < 1:
            raise ValueError("baseline_window, min_consecutive and recovery_frames must be >= 1")
        if self.merge_gap < 0 or self.restart_lookback < 0 or self.bbox_area_high_min_consecutive < 1:
            raise ValueError("interval limits must be non-negative and high minimum must be >= 1")
        if not (1 < self.bbox_area_warning_ratio < self.bbox_area_high_ratio < self.bbox_area_critical_ratio):
            raise ValueError("bbox-area ratios must satisfy 1 < warning < high < critical")
        if self.bbox_area_weight <= 0:
            raise ValueError("bbox_area_weight must be positive")


def parameters_from_config(config: dict[str, Any], overrides: dict[str, Any] | None = None) -> IntervalParameters:
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    params = IntervalParameters(
        baseline_window=int(overrides.get("baseline_window", nested_get(config, ("anomaly_detection", "baseline_window"), 5))),
        bbox_area_warning_ratio=float(overrides.get("bbox_area_warning_ratio", nested_get(config, ("bbox_area", "warning_ratio"), 1.4))),
        bbox_area_high_ratio=float(overrides.get("bbox_area_high_ratio", nested_get(config, ("bbox_area", "high_ratio"), 1.8))),
        bbox_area_critical_ratio=float(overrides.get("bbox_area_critical_ratio", nested_get(config, ("bbox_area", "critical_ratio"), 2.5))),
        bbox_area_weight=float(overrides.get("bbox_area_weight", nested_get(config, ("bbox_area", "weight"), nested_get(config, ("weights", "bbox_area_jump"), 4.0)))),
        min_consecutive=int(overrides.get("min_consecutive", nested_get(config, ("intervals", "min_consecutive"), 3))),
        bbox_area_high_min_consecutive=int(nested_get(config, ("intervals", "bbox_area_high_min_consecutive"), 2)),
        merge_gap=int(overrides.get("merge_gap", nested_get(config, ("intervals", "merge_gap"), 2))),
        recovery_frames=int(overrides.get("recovery_frames", nested_get(config, ("intervals", "recovery_frames"), 3))),
        restart_lookback=int(overrides.get("restart_lookback", nested_get(config, ("intervals", "restart_lookback"), 20))),
        weights={str(k): float(v) for k, v in nested_get(config, ("weights",), {}).items()} or IntervalParameters().weights,
    )
    params.validate()
    return params


def _bbox(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = record.get("raw_bbox_xyxy", record.get("bbox_xyxy"))
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(x) for x in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _bbox_metrics(record: dict[str, Any]) -> tuple[float | None, tuple[float, float] | None, float | None, bool]:
    box = _bbox(record)
    if box is None:
        return None, None, None, False
    x1, y1, x2, y2 = box
    width, height = x2 - x1 + 1, y2 - y1 + 1
    if width <= 0 or height <= 0:
        return None, None, None, False
    return width * height, ((x1 + x2) / 2, (y1 + y2) / 2), width / height, True


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or a <= 0 or b <= 0:
        return None
    return max(a / b, b / a)


def _severity(score: float, types: list[str], area_level: str) -> str:
    if area_level == "critical" or "empty_mask" in types:
        return "critical"
    if area_level == "high" or score >= 5:
        return "high"
    if score >= 2:
        return "medium"
    return "warning"


def _identity_errors(record: dict[str, Any], expected: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    global_id = expected["global_drone_id"]
    if record.get("global_drone_id") not in (None, global_id): errors.append("identity_error")
    if record.get("class_id") not in (None, expected["class_id"]): errors.append("identity_error")
    if global_id not in catalog or catalog[global_id]["class_id"] != expected["class_id"]: errors.append("identity_error")
    return list(dict.fromkeys(errors))


def detect_frame_anomalies(records: Iterable[dict[str, Any]], registry: dict[str, Any], catalog: dict[str, dict[str, Any]], params: IntervalParameters, external: dict[tuple[int, int], list[str]] | None = None) -> list[dict[str, Any]]:
    """Produce one review result per expected frame/object; invalid registry objects are excluded."""
    expected = {int(obj["obj_id"]): obj for obj in registry.get("objects", [])}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        try: grouped[int(record["obj_id"])].append(record)
        except (KeyError, TypeError, ValueError): continue
    output: list[dict[str, Any]] = []
    for obj_id, items in grouped.items():
        if obj_id not in expected:
            continue
        items.sort(key=lambda row: int(row.get("frame_idx", -1)))
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in items: by_frame[int(item["frame_idx"])].append(item)
        frames = sorted(by_frame)
        history: list[float] = []
        mask_history: list[float] = []
        stable_run = 0
        previous_center: tuple[float, float] | None = None
        previous_aspect: float | None = None
        previous_box: tuple[float, float, float, float] | None = None
        static_frames = 0
        for frame in range(frames[0], frames[-1] + 1):
            rows = by_frame.get(frame, [])
            record = rows[0] if rows else {"frame_idx": frame, "obj_id": obj_id}
            types: list[str] = []
            details: dict[str, Any] = {}
            if not rows: types.append("missing_record")
            if len(rows) > 1: types.append("duplicate_frame_idx")
            types.extend(_identity_errors(record, expected[obj_id], catalog))
            area, center, aspect, valid_bbox = _bbox_metrics(record)
            mask_area = record.get("mask_area", 0)
            try: mask_area = float(mask_area or 0)
            except (TypeError, ValueError): mask_area = 0
            if bool(record.get("empty_mask")) or mask_area <= 0: types.append("empty_mask")
            if not valid_bbox: types.append("invalid_bbox")
            if valid_bbox and record.get("frame_width") is not None and record.get("frame_height") is not None:
                w, h = float(record["frame_width"]), float(record["frame_height"])
                x1, y1, x2, y2 = _bbox(record) or (0, 0, 0, 0)
                if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h: types.append("bbox_out_of_bounds")
                elif min(x1, y1, w - 1 - x2, h - 1 - y2) <= 2: types.append("near_edge")
            baseline = float(median(history[-params.baseline_window:])) if history else None
            mask_baseline = float(median(mask_history[-params.baseline_window:])) if mask_history else None
            jump = _ratio(area, baseline)
            level = "normal"
            if area is None and baseline is not None: details["bbox_area_error"] = "invalid current bbox area"
            elif baseline is not None and jump is None: details["bbox_area_error"] = "invalid baseline or current bbox area"
            elif jump is not None:
                if jump >= params.bbox_area_critical_ratio: level = "critical"
                elif jump >= params.bbox_area_high_ratio: level = "high"
                elif jump >= params.bbox_area_warning_ratio: level = "warning"
                if level != "normal": types.append("bbox_area_jump")
            center_shift = None if center is None or previous_center is None else math.dist(center, previous_center)
            if center_shift is not None and area and center_shift > max(10.0, math.sqrt(area) * 2): types.append("center_shift")
            aspect_change = _ratio(aspect, previous_aspect)
            if aspect_change is not None and aspect_change >= 2.0: types.append("aspect_ratio_jump")
            mask_jump = _ratio(mask_area, mask_baseline)
            if mask_jump is not None and mask_jump >= params.bbox_area_high_ratio:
                types.append("mask_area_jump")
            current_box = _bbox(record)
            if valid_bbox and current_box == previous_box:
                static_frames += 1
                if static_frames >= 5:
                    types.append("bbox_static")
            else:
                static_frames = 0
            weights = params.weights
            score = sum(weights.get(t, 1.0) for t in types if t != "bbox_area_jump")
            bbox_score = 0.0
            if level == "warning": bbox_score = params.bbox_area_weight * 0.35
            elif level == "high": bbox_score = params.bbox_area_weight
            elif level == "critical": bbox_score = params.bbox_area_weight * 1.5
            score += bbox_score
            reliable = not types and valid_bbox and mask_area > 0
            # Never allow suspicious frames into the baseline.  After a bad run,
            # require a stable recovery run before resuming history updates.
            if reliable:
                stable_run += 1
                if stable_run >= params.recovery_frames or not history:
                    history.append(float(area))
                    mask_history.append(mask_area)
            else:
                stable_run = 0
            if center is not None: previous_center = center
            if aspect is not None: previous_aspect = aspect
            previous_box = current_box
            if external and (obj_id, frame) in external:
                types.extend(t for t in external[(obj_id, frame)] if t not in types)
                reliable = False
                score += sum(weights.get(t, 1.0) for t in external[(obj_id, frame)])
            output.append({"video_id": str(record.get("video_id", registry.get("video", "input"))), "frame_idx": frame, "obj_id": obj_id, "global_drone_id": expected[obj_id]["global_drone_id"], "class_id": expected[obj_id]["class_id"], "anomaly_types": types, "anomaly_score": round(score, 3), "severity": _severity(score, types, level), "bbox_area": area, "bbox_area_baseline": baseline, "bbox_area_jump_ratio": jump, "bbox_area_level": level, "bbox_area_score": round(bbox_score, 3), "mask_area": mask_area, "center_shift": center_shift, "aspect_ratio_change": aspect_change, "near_edge": "near_edge" in types, "reliable": reliable, "details": details})
    return output


def _interval_id(video_id: str, obj_id: int, start: int, end: int) -> str:
    return f"{video_id}_obj{obj_id}_{start:06d}_{end:06d}"


def build_intervals(frame_results: list[dict[str, Any]], params: IntervalParameters) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in frame_results: grouped[int(result["obj_id"])].append(result)
    for obj_id, results in grouped.items():
        results.sort(key=lambda row: int(row["frame_idx"]))
        flagged = [r for r in results if r["anomaly_types"]]
        chunks: list[list[dict[str, Any]]] = []
        for row in flagged:
            if not chunks or int(row["frame_idx"]) - int(chunks[-1][-1]["frame_idx"]) > params.merge_gap + 1: chunks.append([row])
            else: chunks[-1].append(row)
        by_frame = {int(r["frame_idx"]): r for r in results}
        for chunk in chunks:
            types = {t for r in chunk for t in r["anomaly_types"]}
            high_area = sum(r["bbox_area_level"] in {"high", "critical"} for r in chunk)
            critical = any(r["bbox_area_level"] == "critical" or "empty_mask" in r["anomaly_types"] for r in chunk)
            qualifies = critical or high_area >= params.bbox_area_high_min_consecutive or len(chunk) >= params.min_consecutive
            max_frame = max(int(r["frame_idx"]) for r in chunk)
            # Hold interval open through a short recovery period; exclude stable
            # recovery frames from anomaly_frames but give reviewers the full span.
            end = max_frame
            for candidate in range(max_frame + 1, max_frame + params.recovery_frames + 1):
                if candidate in by_frame and by_frame[candidate]["reliable"]: end = candidate
                else: break
            start = int(chunk[0]["frame_idx"])
            scores = [float(r["anomaly_score"]) for r in chunk]
            jumps = [float(r["bbox_area_jump_ratio"]) for r in chunk if r["bbox_area_jump_ratio"] is not None]
            severity = "critical" if critical else ("high" if qualifies and high_area else ("medium" if qualifies else "warning"))
            restart = recommended_restart_frame(results, start, params.restart_lookback)
            first = chunk[0]
            intervals.append({"interval_id": _interval_id(str(first["video_id"]), obj_id, start, end), "video_id": first["video_id"], "obj_id": obj_id, "global_drone_id": first["global_drone_id"], "class_id": first["class_id"], "start_frame": start, "end_frame": end, "anomaly_frames": [int(r["frame_idx"]) for r in chunk], "anomaly_types": sorted(types), "severity": severity, "max_anomaly_score": max(scores), "mean_anomaly_score": sum(scores) / len(scores), "max_bbox_area_jump_ratio": max(jumps) if jumps else None, "mean_bbox_area_jump_ratio": sum(jumps) / len(jumps) if jumps else None, "bbox_area_critical_frames": [int(r["frame_idx"]) for r in chunk if r["bbox_area_level"] == "critical"], "recommended_restart_frame": restart, "status": "pending", "reviewer_note": "", "actual_restart_frame": None, "correction_tracks_path": None, "created_at": utc_now(), "updated_at": utc_now()})
    return intervals


def recommended_restart_frame(results: list[dict[str, Any]], start_frame: int, lookback: int) -> int | None:
    candidates = {int(row["frame_idx"]): row for row in results}
    for frame in range(start_frame - 1, max(-1, start_frame - lookback - 1), -1):
        row = candidates.get(frame)
        if row and row["reliable"]:
            return frame
    return None


def inherit_review_status(intervals: list[dict[str, Any]], old: list[dict[str, Any]]) -> dict[str, int]:
    exact = overlap = 0
    for interval in intervals:
        candidate = next((row for row in old if row.get("interval_id") == interval["interval_id"]), None)
        if candidate is not None: exact += 1
        else:
            same = [row for row in old if row.get("video_id") == interval["video_id"] and row.get("obj_id") == interval["obj_id"]]
            if same:
                def ratio(row: dict[str, Any]) -> float:
                    lo, hi = max(interval["start_frame"], row["start_frame"]), min(interval["end_frame"], row["end_frame"])
                    return max(0, hi - lo + 1) / max(interval["end_frame"] - interval["start_frame"] + 1, row["end_frame"] - row["start_frame"] + 1)
                candidate = max(same, key=ratio)
                if ratio(candidate) < 0.5: candidate = None
                else: overlap += 1
        if candidate:
            for key in ("status", "reviewer_note", "actual_restart_frame", "correction_tracks_path"):
                if key in candidate: interval[key] = candidate[key]
            interval["updated_at"] = candidate.get("updated_at", interval["updated_at"])
    return {"new": len(intervals) - exact - overlap, "exact": exact, "overlap": overlap, "resolved": sum(row.get("status") == "resolved" for row in intervals), "unmatched_old": sum(not any(n.get("interval_id") == o.get("interval_id") for n in intervals) for o in old)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        for row in rows: file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def write_review_csv(path: Path, intervals: list[dict[str, Any]]) -> None:
    import csv
    fields = ["interval_id", "video_id", "obj_id", "global_drone_id", "class_id", "start_frame", "end_frame", "duration", "anomaly_types", "severity", "max_bbox_area_jump_ratio", "recommended_restart_frame", "status", "reviewer_note", "actual_restart_frame", "correction_tracks_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for row in intervals:
            value = dict(row); value["duration"] = int(row["end_frame"]) - int(row["start_frame"]) + 1; value["anomaly_types"] = ";".join(row["anomaly_types"]); writer.writerow({key: value.get(key) for key in fields})


def compare_interval_sets(original: list[dict[str, Any]], final: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize review-only anomaly reduction between two interval JSONLs."""
    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        severe = [row for row in rows if row.get("severity") in {"high", "critical"}]
        ratios = [float(row["max_bbox_area_jump_ratio"]) for row in rows if row.get("max_bbox_area_jump_ratio") is not None]
        return {"interval_count": len(rows), "high_critical_count": len(severe), "max_bbox_area_jump_ratio": max(ratios) if ratios else None}
    before, after = summary(original), summary(final)
    return {"original": before, "final": after, "interval_reduction": before["interval_count"] - after["interval_count"], "high_critical_reduction": before["high_critical_count"] - after["high_critical_count"]}


def update_interval_status(rows: list[dict[str, Any]], *, interval_id: str | None, obj_id: int | None, start_frame: int | None, end_frame: int | None, status: str | None, reviewer_note: str | None, correction_tracks_path: str | None, actual_restart_frame: int | None, force: bool) -> dict[str, Any]:
    if status is not None and status not in REVIEW_STATUSES: raise ValueError(f"invalid status: {status}; allowed: {sorted(REVIEW_STATUSES)}")
    if actual_restart_frame is not None and actual_restart_frame < 0: raise ValueError("actual_restart_frame must be non-negative")
    matches = [row for row in rows if (interval_id is not None and row.get("interval_id") == interval_id) or (interval_id is None and row.get("obj_id") == obj_id and row.get("start_frame") == start_frame and row.get("end_frame") == end_frame)]
    if len(matches) != 1: raise ValueError("interval selector must match exactly one interval")
    row = matches[0]
    if row.get("status") == "resolved" and not force: raise ValueError("resolved interval requires --force to change")
    if status is not None: row["status"] = status
    if reviewer_note is not None: row["reviewer_note"] = reviewer_note
    if correction_tracks_path is not None: row["correction_tracks_path"] = correction_tracks_path
    if actual_restart_frame is not None: row["actual_restart_frame"] = actual_restart_frame
    row["updated_at"] = utc_now()
    return row
