#!/usr/bin/env python3
"""Screen SAM2 tracking outputs for temporal anomalies and build a review queue."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import fields
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.anomaly_detector import (
    AnomalyParameters,
    AnomalyResult,
    evaluate_tracks,
    merge_review_segments,
    observation_from_record,
    should_review,
    summarize_results,
    with_duplicate_warning,
)
from sam2_annotator.config_utils import load_yaml_config
from sam2_annotator.frame_utils import frame_path

LOGGER = logging.getLogger("detect_anomalies")


def resolve_path(value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def load_tracks(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"tracks JSONL does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("frame_idx", "obj_id", "mask_area", "bbox_area"):
                if key not in record:
                    raise ValueError(f"tracks line {line_no} missing {key}")
            records.append(record)
    if not records:
        raise ValueError(f"tracks JSONL has no records: {path}")
    return records


def build_parameters(args: argparse.Namespace) -> AnomalyParameters:
    config_path = args.config.expanduser().resolve()
    raw_config = load_yaml_config(config_path)
    values = {field.name: raw_config.get(field.name) for field in fields(AnomalyParameters) if field.name in raw_config}
    for name in values:
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    params = AnomalyParameters(**values)
    params.validate()
    return params


def write_anomalies(path: Path, results: list[AnomalyResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result.to_json_dict()) + "\n")


def write_review_queue(path: Path, results: list[AnomalyResult], thumbnails_dir: Path) -> list[AnomalyResult]:
    review_results = [result for result in results if should_review(result)]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_idx",
        "obj_id",
        "object_name",
        "status",
        "score",
        "reasons",
        "raw_bbox_xyxy",
        "stable_bbox_xyxy",
        "bbox_status",
        "bbox_stability_reasons",
        "thumbnail_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in review_results:
            thumb = thumbnails_dir / thumbnail_name(result)
            writer.writerow(
                {
                    "frame_idx": result.frame_idx,
                    "obj_id": result.obj_id,
                    "object_name": result.object_name or "",
                    "status": result.status,
                    "score": result.score,
                    "reasons": ";".join(result.reasons),
                    "raw_bbox_xyxy": json.dumps(result.raw_bbox_xyxy),
                    "stable_bbox_xyxy": json.dumps(result.stable_bbox_xyxy),
                    "bbox_status": result.bbox_status or "",
                    "bbox_stability_reasons": ";".join(result.bbox_stability_reasons),
                    "thumbnail_path": str(thumb),
                }
            )
    return review_results


def render_review_thumbnails(
    review_results: list[AnomalyResult],
    result_by_key: dict[tuple[int, int], AnomalyResult],
    record_by_key: dict[tuple[int, int], dict[str, Any]],
    frames_dir: Path,
    tracks_path: Path,
    thumbnails_dir: Path,
) -> None:
    import cv2

    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    first_frame_path = next((path for path in sorted(frames_dir.glob("*.jpg")) if path.is_file()), None)
    if first_frame_path is None:
        raise FileNotFoundError(f"no JPEG frames found in {frames_dir}")
    first = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
    if first is None:
        raise OSError(f"failed to read frame: {first_frame_path}")
    height, width = first.shape[:2]

    for result in review_results:
        panels = [
            _render_panel(result.frame_idx - 1, result.obj_id, result_by_key, record_by_key, frames_dir, tracks_path, (width, height)),
            _render_panel(result.frame_idx, result.obj_id, result_by_key, record_by_key, frames_dir, tracks_path, (width, height)),
            _render_panel(result.frame_idx + 1, result.obj_id, result_by_key, record_by_key, frames_dir, tracks_path, (width, height)),
        ]
        triptych = np.hstack(panels)
        output_path = thumbnails_dir / thumbnail_name(result)
        if not cv2.imwrite(str(output_path), triptych):
            raise OSError(f"failed to write review thumbnail: {output_path}")


def thumbnail_name(result: AnomalyResult) -> str:
    return f"{result.frame_idx:06d}_obj{result.obj_id}_{result.status}.jpg"


def _render_panel(
    frame_idx: int,
    obj_id: int,
    result_by_key: dict[tuple[int, int], AnomalyResult],
    record_by_key: dict[tuple[int, int], dict[str, Any]],
    frames_dir: Path,
    tracks_path: Path,
    image_size: tuple[int, int],
) -> np.ndarray:
    import cv2

    width, height = image_size
    panel = np.full((height, width, 3), 32, dtype=np.uint8)
    source_path = frame_path(frames_dir, frame_idx) if frame_idx >= 0 else None
    if source_path is not None and source_path.is_file():
        frame = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise OSError(f"failed to read frame: {source_path}")
        if frame.shape[:2] != (height, width):
            raise ValueError(f"frame size changed at {source_path}: {frame.shape[1]}x{frame.shape[0]}")
        panel = frame

    result = result_by_key.get((frame_idx, obj_id))
    record = record_by_key.get((frame_idx, obj_id))
    if result is not None and record is not None and source_path is not None and source_path.is_file():
        mask_path_value = record.get("mask_path")
        if mask_path_value:
            mask_path = resolve_path(mask_path_value, tracks_path.parent)
            if mask_path.is_file():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise OSError(f"failed to read mask: {mask_path}")
                if mask.shape[:2] != (height, width):
                    raise ValueError(f"mask size changed at {mask_path}: {mask.shape[1]}x{mask.shape[0]}")
                mask_bool = mask > 0
                color = np.array([0, 180, 255], dtype=np.uint8)
                panel[mask_bool] = (0.55 * panel[mask_bool] + 0.45 * color).astype(np.uint8)
        if result.raw_bbox_xyxy is not None:
            x1, y1, x2, y2 = [int(v) for v in result.raw_bbox_xyxy]
            cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 255), 1)
        if result.stable_bbox_xyxy is not None:
            x1, y1, x2, y2 = [int(v) for v in result.stable_bbox_xyxy]
            cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 200, 0), 2)

    label = _panel_label(frame_idx, obj_id, result)
    _draw_label(panel, label)
    return panel


def _panel_label(frame_idx: int, obj_id: int, result: AnomalyResult | None) -> list[str]:
    if result is None:
        return [f"frame {frame_idx} obj {obj_id}", "no track record"]
    reasons = ",".join(result.reasons) if result.reasons else "none"
    return [
        f"frame {frame_idx} obj {obj_id}",
        f"{result.status} score={result.score}",
        reasons,
    ]


def _draw_label(image: np.ndarray, lines: list[str]) -> None:
    import cv2

    x = 10
    y = 24
    line_height = 24
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1)
        y += line_height


def detect_anomalies(
    tracks_path: Path,
    frames_dir: Path,
    output_dir: Path,
    params: AnomalyParameters,
) -> dict[str, Any]:
    import cv2

    tracks_path = tracks_path.expanduser().resolve()
    frames_dir = frames_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames directory does not exist: {frames_dir}")
    first_frame_path = next((path for path in sorted(frames_dir.glob("*.jpg")) if path.is_file()), None)
    if first_frame_path is None:
        raise FileNotFoundError(f"no JPEG frames found in {frames_dir}")
    first = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
    if first is None:
        raise OSError(f"failed to read frame: {first_frame_path}")
    image_size = (int(first.shape[1]), int(first.shape[0]))

    records = load_tracks(tracks_path)
    observations = [observation_from_record(record) for record in records]
    results = evaluate_tracks(observations, params, image_size)
    results = apply_duplicate_target_warnings(results, records, tracks_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnails_dir = output_dir / "review_thumbnails"
    anomalies_path = output_dir / "anomalies.jsonl"
    queue_path = output_dir / "review_queue.csv"
    summary_path = output_dir / "anomaly_summary.json"
    segments_path = output_dir / "review_segments.json"

    write_anomalies(anomalies_path, results)
    review_results = write_review_queue(queue_path, results, thumbnails_dir)

    result_by_key = {(result.frame_idx, result.obj_id): result for result in results}
    record_by_key = {(int(record["frame_idx"]), int(record["obj_id"])): record for record in records}
    render_review_thumbnails(review_results, result_by_key, record_by_key, frames_dir, tracks_path, thumbnails_dir)

    summary = summarize_results(results, params)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    segments = [segment.to_json_dict() for segment in merge_review_segments(results)]
    segments_path.write_text(json.dumps(segments, indent=2) + "\n", encoding="utf-8")

    LOGGER.info("wrote anomalies: %s", anomalies_path)
    LOGGER.info("wrote review queue: %s", queue_path)
    LOGGER.info("wrote review thumbnails: %s", thumbnails_dir)
    LOGGER.info("wrote summary: %s", summary_path)
    LOGGER.info("wrote review segments: %s", segments_path)
    return summary


def apply_duplicate_target_warnings(
    results: list[AnomalyResult],
    records: list[dict[str, Any]],
    tracks_path: Path,
    iou_threshold: float = 0.5,
) -> list[AnomalyResult]:
    import cv2

    result_by_key = {(result.frame_idx, result.obj_id): result for result in results}
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        if not record.get("mask_path"):
            continue
        by_frame.setdefault(int(record["frame_idx"]), []).append(record)

    warning_keys: set[tuple[int, int]] = set()
    for frame_idx, frame_records in by_frame.items():
        masks: list[tuple[int, np.ndarray]] = []
        for record in frame_records:
            mask_path = resolve_path(str(record["mask_path"]), tracks_path.parent)
            if not mask_path.is_file():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise OSError(f"failed to read mask: {mask_path}")
            mask_bool = mask > 0
            if np.count_nonzero(mask_bool) > 0:
                masks.append((int(record["obj_id"]), mask_bool))
        for idx, (obj_a, mask_a) in enumerate(masks):
            for obj_b, mask_b in masks[idx + 1 :]:
                if obj_a == obj_b:
                    continue
                intersection = int(np.logical_and(mask_a, mask_b).sum())
                union = int(np.logical_or(mask_a, mask_b).sum())
                iou = intersection / union if union else 0.0
                if iou > iou_threshold:
                    warning_keys.add((frame_idx, obj_a))
                    warning_keys.add((frame_idx, obj_b))

    return [with_duplicate_warning(result) if (result.frame_idx, result.obj_id) in warning_keys else result for result in results]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect review-worthy anomalies from SAM2 mask, bbox and temporal statistics.")
    parser.add_argument("--tracks", required=True, type=Path, help="tracks.jsonl produced by scripts/track_box.py.")
    parser.add_argument("--frames-dir", required=True, type=Path, help="Directory containing original extracted JPEG frames.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for anomaly outputs and review thumbnails.")
    parser.add_argument("--config", type=Path, default=Path("configs/anomaly_rules.yaml"), help="YAML rule config. Default: configs/anomaly_rules.yaml.")
    parser.add_argument("--history-size", type=int, help="Override number of recent valid frames used for per-object median baselines.")
    parser.add_argument("--periodic-interval", type=int, help="Override periodic review interval in frames; 0 disables periodic checks.")
    parser.add_argument("--area-ratio-min", type=float, help="Override minimum mask area ratio before area_drop.")
    parser.add_argument("--area-ratio-max", type=float, help="Override maximum mask area ratio before area_increase.")
    parser.add_argument("--center-jump-factor", type=float, help="Override multiplier for recent median center motion.")
    parser.add_argument("--center-jump-min-diagonal-ratio", type=float, help="Override minimum center jump as image diagonal ratio.")
    parser.add_argument("--aspect-ratio-min", type=float, help="Override minimum bbox aspect ratio change.")
    parser.add_argument("--aspect-ratio-max", type=float, help="Override maximum bbox aspect ratio change.")
    parser.add_argument("--fill-ratio-min", type=float, help="Override minimum mask fill ratio change.")
    parser.add_argument("--border-margin", type=int, help="Override image border margin in pixels for touch_border.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    params = build_parameters(args)
    detect_anomalies(args.tracks, args.frames_dir, args.output_dir, params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
