#!/usr/bin/env python3
"""Strict validation for merged annotation tracks and YOLO label exports."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.drone_catalog import load_catalog
from sam2_annotator.object_registry import load_registry
from sam2_annotator.tracking_io import load_jsonl, write_json

LOGGER = logging.getLogger("validate_annotation_export")


class AnnotationValidationError(ValueError):
    """Raised when an annotation export does not meet final-export rules."""


def _range(value: tuple[int, int] | None) -> tuple[int, int] | None:
    if value is not None and (value[0] < 0 or value[1] < value[0]):
        raise ValueError("full frame range must be non-negative and increasing")
    return value


def _bbox(record: dict[str, Any]) -> list[float] | None:
    value = record.get("bbox_xyxy", record.get("bbox"))
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    return [float(item) for item in value]


def _report(errors: list[str], warnings: list[str], records: list[dict[str, Any]], labels: dict[int, list[tuple[int, list[str]]]], missing: list[str], empty: int, duplicates: int, frame_range: tuple[int, int] | None) -> dict[str, Any]:
    frames = (frame_range[1] - frame_range[0] + 1) if frame_range else len(set(labels) | {int(row["frame_idx"]) for row in records if isinstance(row.get("frame_idx"), int)})
    return {
        "valid": not errors,
        "total_frames": frames,
        "total_tracks": len(records),
        "total_label_lines": sum(len(lines) for lines in labels.values()),
        "records_by_obj_id": dict(sorted(Counter(str(row.get("obj_id")) for row in records).items())),
        "records_by_global_drone_id": dict(sorted(Counter(str(row.get("global_drone_id")) for row in records).items())),
        "records_by_class_id": dict(sorted(Counter(str(row.get("class_id")) for row in records).items())),
        "empty_label_files": empty,
        "missing_label_files": missing,
        "duplicate_key_count": duplicates,
        "errors": errors,
        "warnings": warnings,
    }


def validate_annotation_export(
    tracks: Path,
    labels_dir: Path,
    registry: Path,
    catalog: Path,
    full_frame_range: tuple[int, int] | None = None,
    require_contiguous_label_files: bool = False,
    report_output: Path | None = None,
    *,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Validate final artifacts and return a JSON-serializable quality report."""
    tracks, labels_dir = tracks.expanduser().resolve(), labels_dir.expanduser().resolve()
    full_frame_range = _range(full_frame_range)
    errors: list[str] = []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    labels: dict[int, list[tuple[int, list[str]]]] = {}
    missing: list[str] = []
    empty = 0
    duplicates = 0

    try:
        records = load_jsonl(tracks)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"merged tracks cannot be parsed: {exc}")
    try:
        registry_data = load_registry(registry)
        catalog_data = load_catalog(catalog)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"registry or catalog cannot be parsed: {exc}")
        registry_data, catalog_data = {"objects": []}, {}

    allowed: dict[int, dict[str, Any]] = {
        int(item["obj_id"]): item for item in registry_data.get("objects", [])
    }
    active_validated = {
        obj_id: item for obj_id, item in allowed.items() if item.get("status") == "active" and item.get("validated") is True
    }
    if len([item for item in allowed.values() if item.get("status") == "active"]) > 2:
        errors.append("a video can have at most two active objects")

    expected_by_frame: dict[int, list[int]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for index, record in enumerate(records, start=1):
        required = ("frame_idx", "obj_id", "global_drone_id", "class_id")
        if any(key not in record for key in required) or ("bbox_xyxy" not in record and "bbox" not in record):
            errors.append(f"track record {index} is missing required fields")
            continue
        frame_idx, obj_id = record.get("frame_idx"), record.get("obj_id")
        if not isinstance(frame_idx, int) or frame_idx < 0 or not isinstance(obj_id, int) or obj_id <= 0:
            errors.append(f"track record {index} has invalid frame_idx or obj_id")
            continue
        key = (frame_idx, obj_id)
        if key in seen:
            duplicates += 1
            errors.append(f"duplicate (frame_idx, obj_id): {key}")
        seen.add(key)
        obj = allowed.get(obj_id)
        if obj is None:
            errors.append(f"track record {index} obj_id {obj_id} is absent from registry")
        elif obj_id not in active_validated:
            errors.append(f"track record {index} obj_id {obj_id} is not active and validated")
        else:
            drone_id = obj["global_drone_id"]
            if record.get("global_drone_id") != drone_id:
                errors.append(f"track record {index} global_drone_id disagrees with registry")
            entry = catalog_data.get(str(drone_id))
            if entry is None:
                errors.append(f"registry obj_id {obj_id} global_drone_id is absent from catalog")
            elif record.get("class_id") != entry["class_id"] or obj.get("class_id") != entry["class_id"]:
                errors.append(f"track record {index} class_id disagrees with catalog mapping")
        bbox = _bbox(record)
        if bbox is None or bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            errors.append(f"track record {index} has invalid bbox")
        else:
            expected_by_frame[frame_idx].append(int(record.get("class_id", -1)))

    if not labels_dir.is_dir():
        errors.append(f"labels directory does not exist: {labels_dir}")
    else:
        for path in sorted(labels_dir.glob("*.txt")):
            try:
                frame_idx = int(path.stem)
            except ValueError:
                warnings.append(f"ignoring non-frame label file: {path.name}")
                continue
            rows: list[tuple[int, list[str]]] = []
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                fields = line.split()
                if len(fields) != 5:
                    errors.append(f"{path.name}:{line_no} must contain 5 columns")
                    continue
                try:
                    class_id = int(fields[0])
                    coords = [float(value) for value in fields[1:]]
                except ValueError:
                    errors.append(f"{path.name}:{line_no} has non-numeric YOLO values")
                    continue
                if class_id not in {entry["class_id"] for entry in catalog_data.values()}:
                    errors.append(f"{path.name}:{line_no} class_id {class_id} is absent from catalog")
                if any(value < 0 or value > 1 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
                    errors.append(f"{path.name}:{line_no} has invalid normalized YOLO coordinates")
                rows.append((class_id, fields))
            if not rows:
                empty += 1
            labels[frame_idx] = rows
        expected_files = set(expected_by_frame)
        if full_frame_range:
            expected_files.update(range(full_frame_range[0], full_frame_range[1] + 1))
        for frame_idx in sorted(expected_files - set(labels)):
            missing.append(f"{frame_idx:06d}.txt")
        if missing and require_contiguous_label_files:
            errors.append(f"missing label files: {', '.join(missing)}")
        elif missing:
            warnings.append(f"missing label files: {', '.join(missing)}")
        for frame_idx, expected_classes in expected_by_frame.items():
            actual_classes = [row[0] for row in labels.get(frame_idx, [])]
            if sorted(actual_classes) != sorted(expected_classes):
                errors.append(f"frame {frame_idx} label classes/count do not match track records")
    label_count = sum(len(rows) for rows in labels.values())
    if label_count != len(records):
        errors.append(f"label line count {label_count} does not equal track record count {len(records)}")

    report = _report(errors, warnings, records, labels, missing, empty, duplicates, full_frame_range)
    if report_output is not None:
        write_json(report_output, report)
    if errors and raise_on_error:
        raise AnnotationValidationError("annotation export validation failed: " + "; ".join(errors))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strictly validate merged tracks and YOLO labels for one video export.")
    parser.add_argument("--tracks", required=True, type=Path, help="Merged tracks JSONL.")
    parser.add_argument("--labels-dir", required=True, type=Path, help="YOLO label directory.")
    parser.add_argument("--registry", required=True, type=Path, help="Current-video registry JSON.")
    parser.add_argument("--catalog", required=True, type=Path, help="Global drone catalog JSON.")
    parser.add_argument("--full-frame-range", nargs=2, type=int, metavar=("START", "END"), help="Expected inclusive frame range.")
    parser.add_argument("--require-contiguous-label-files", action="store_true", help="Require a label txt file for every expected frame.")
    parser.add_argument("--report-output", type=Path, help="Write the JSON quality report here, including on validation failure.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    try:
        report = validate_annotation_export(args.tracks, args.labels_dir, args.registry, args.catalog, tuple(args.full_frame_range) if args.full_frame_range else None, args.require_contiguous_label_files, args.report_output)
    except AnnotationValidationError as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("quality validation passed: %s", json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
