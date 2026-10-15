#!/usr/bin/env python3
"""Compute before/after correction change metrics for human review."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.correction_utils import atomic_write_json, load_jsonl

LOGGER = logging.getLogger("evaluate_correction")


def _keyed(records: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    keyed: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        key = (int(record["frame_idx"]), int(record["obj_id"]))
        if key in keyed:
            raise ValueError(f"duplicate track key {key}")
        keyed[key] = record
    return keyed


def _resolve_path(value: object, tracks_path: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = tracks_path.parent / path
    return path.resolve()


def _read_mask(record: dict[str, Any], tracks_path: Path) -> np.ndarray | None:
    import cv2

    mask_path = _resolve_path(record.get("mask_path"), tracks_path)
    if mask_path is None or not mask_path.is_file():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise OSError(f"failed to read mask: {mask_path}")
    return mask > 0


def _mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(a, b).sum()
    return float(intersection / union)


def _center(record: dict[str, Any]) -> tuple[float, float] | None:
    value = record.get("center_xy")
    if value is None:
        return None
    return float(value[0]), float(value[1])


def _distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _is_anomaly(record: dict[str, Any]) -> bool:
    status = str(record.get("status") or record.get("review_status") or "")
    if status in {"warning", "need_review", "critical"}:
        return True
    reasons = record.get("reasons") or record.get("anomaly_reasons") or []
    return bool(reasons)


def evaluate_correction(baseline_tracks: Path, corrected_tracks: Path, correction_frame: int, output_json: Path) -> dict[str, Any]:
    baseline_tracks = baseline_tracks.expanduser().resolve()
    corrected_tracks = corrected_tracks.expanduser().resolve()
    baseline = _keyed(load_jsonl(baseline_tracks))
    corrected = _keyed(load_jsonl(corrected_tracks))
    keys = sorted(key for key in baseline.keys() & corrected.keys() if key[0] >= correction_frame)

    per_frame: list[dict[str, Any]] = []
    obvious_change_count = 0
    baseline_empty_count = 0
    corrected_empty_count = 0
    baseline_anomaly_count = 0
    corrected_anomaly_count = 0
    consecutive_valid = 0

    for key in keys:
        base_record = baseline[key]
        corrected_record = corrected[key]
        base_empty = bool(base_record.get("empty_mask", False))
        corr_empty = bool(corrected_record.get("empty_mask", False))
        baseline_empty_count += 1 if base_empty else 0
        corrected_empty_count += 1 if corr_empty else 0
        baseline_anomaly_count += 1 if _is_anomaly(base_record) else 0
        corrected_anomaly_count += 1 if _is_anomaly(corrected_record) else 0
        iou = _mask_iou(_read_mask(base_record, baseline_tracks), _read_mask(corrected_record, corrected_tracks))
        center_distance = _distance(_center(base_record), _center(corrected_record))
        obvious_change = (iou is not None and iou < 0.95) or (center_distance is not None and center_distance > 2.0) or base_empty != corr_empty
        obvious_change_count += 1 if obvious_change else 0
        per_frame.append(
            {
                "frame_idx": key[0],
                "obj_id": key[1],
                "mask_iou": iou,
                "bbox_center_distance": center_distance,
                "baseline_empty": base_empty,
                "corrected_empty": corr_empty,
                "obvious_change": obvious_change,
            }
        )

    for key in sorted(key for key in corrected if key[0] >= correction_frame):
        if bool(corrected[key].get("empty_mask", False)):
            break
        consecutive_valid += 1

    result = {
        "correction_frame": correction_frame,
        "compared_frame_count": len(keys),
        "baseline_empty_count": baseline_empty_count,
        "corrected_empty_count": corrected_empty_count,
        "baseline_anomaly_count": baseline_anomaly_count,
        "corrected_anomaly_count": corrected_anomaly_count,
        "obvious_change_frame_count": obvious_change_count,
        "post_correction_consecutive_valid_frame_count": consecutive_valid,
        "metrics_note": "These metrics show before/after changes only; they do not prove the corrected track is correct. Review the comparison video.",
        "per_frame": per_frame,
    }
    atomic_write_json(output_json, result)
    LOGGER.info("wrote correction evaluation: %s", output_json)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate how corrected tracks differ from baseline tracks after a correction frame.")
    parser.add_argument("--baseline-tracks", required=True, type=Path, help="Baseline tracks.jsonl.")
    parser.add_argument("--corrected-tracks", required=True, type=Path, help="Corrected tracks JSONL.")
    parser.add_argument("--correction-frame", required=True, type=int, help="Correction frame index.")
    parser.add_argument("--output-json", required=True, type=Path, help="Output JSON metrics path.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    evaluate_correction(args.baseline_tracks, args.corrected_tracks, args.correction_frame, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
