#!/usr/bin/env python3
"""Merge local SAM2 tracks and export catalog-derived multi-object YOLO labels."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.drone_catalog import get_catalog_entry, load_catalog
from sam2_annotator.object_registry import load_registry
from sam2_annotator.tracking_io import ensure_unique_track_keys, load_jsonl, write_jsonl, write_multi_object_labels

LOGGER = logging.getLogger("merge_object_tracks")


def _infer_size(records: list[dict[str, Any]]) -> tuple[int, int]:
    for record in records:
        if record.get("frame_width") and record.get("frame_height"):
            return int(record["frame_width"]), int(record["frame_height"])
    raise ValueError("frame-width/frame-height are required because tracks do not contain dimensions")


def _registry_mapping(registry_path: Path, catalog_path: Path, active_only: bool) -> dict[int, dict[str, Any]]:
    registry = load_registry(registry_path)
    catalog = load_catalog(catalog_path)
    mapping: dict[int, dict[str, Any]] = {}
    for obj in registry["objects"]:
        obj_id = int(obj["obj_id"])
        entry = get_catalog_entry(catalog, str(obj["global_drone_id"]))
        if int(obj["class_id"]) != int(entry["class_id"]):
            raise ValueError(f"registry obj_id {obj_id} class_id disagrees with catalog")
        if obj.get("class_name", entry["class_name"]) != entry["class_name"]:
            raise ValueError(f"registry obj_id {obj_id} class_name disagrees with catalog")
        if active_only and obj["status"] != "active":
            continue
        mapping[obj_id] = {"global_drone_id": entry["global_drone_id"], "class_id": entry["class_id"], "class_name": entry["class_name"]}
    return mapping


def merge_object_tracks(
    tracks: list[Path],
    output_tracks: Path,
    output_labels: Path,
    registry: Path,
    catalog: Path,
    active_only: bool,
    frame_width: int | None,
    frame_height: int | None,
    overwrite: bool,
    full_frame_range: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    if not tracks:
        raise ValueError("at least one --tracks path is required")
    output_tracks = output_tracks.expanduser().resolve()
    output_labels = output_labels.expanduser().resolve()
    if output_tracks.exists() and not overwrite:
        raise FileExistsError(f"output tracks exists; pass --overwrite: {output_tracks}")
    if output_labels.exists() and any(output_labels.iterdir()) and not overwrite:
        raise FileExistsError(f"output labels dir is not empty; pass --overwrite: {output_labels}")
    mapping = _registry_mapping(registry, catalog, active_only)
    records: list[dict[str, Any]] = []
    for path in tracks:
        for raw_record in load_jsonl(path):
            obj_id = raw_record.get("obj_id")
            if not isinstance(obj_id, int):
                raise ValueError(f"track record has invalid obj_id: {obj_id!r}")
            identity = mapping.get(obj_id)
            if identity is None:
                raise ValueError(f"track obj_id {obj_id} is not a registered active object (it may be invalid)")
            # The registry/catalog are authoritative, including for legacy tracks.
            record = dict(raw_record)
            record.update(identity)
            records.append(record)
    ensure_unique_track_keys(records)
    records.sort(key=lambda item: (int(item["frame_idx"]), int(item["obj_id"])))
    if frame_width is None or frame_height is None:
        frame_width, frame_height = _infer_size(records)
    write_jsonl(output_tracks, records)
    write_multi_object_labels(records, output_labels, frame_width, frame_height, full_frame_range)
    LOGGER.info("wrote merged tracks: %s", output_tracks)
    LOGGER.info("wrote merged labels: %s", output_labels)
    LOGGER.info("record counts by global_drone_id: %s", dict(sorted(Counter(row["global_drone_id"] for row in records).items())))
    LOGGER.info("record counts by class_id: %s", dict(sorted(Counter(int(row["class_id"]) for row in records).items())))
    LOGGER.info("record counts by obj_id: %s", dict(sorted(Counter(int(row["obj_id"]) for row in records).items())))
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge tracks by (frame_idx, obj_id) and write catalog-derived YOLO labels.")
    parser.add_argument("--tracks", required=True, nargs="+", type=Path, help="One or more object tracks JSONL paths.")
    parser.add_argument("--registry", required=True, type=Path, help="Per-video object registry JSON.")
    parser.add_argument("--catalog", required=True, type=Path, help="Dataset-wide drone catalog JSON.")
    parser.add_argument("--active-only", action="store_true", help="Export only registry objects whose status is active.")
    parser.add_argument("--output-tracks", required=True, type=Path, help="Merged output tracks JSONL.")
    parser.add_argument("--output-labels", required=True, type=Path, help="Output directory for merged YOLO labels.")
    parser.add_argument("--frame-width", type=int, help="Frame width for YOLO normalization.")
    parser.add_argument("--frame-height", type=int, help="Frame height for YOLO normalization.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output tracks and labels.")
    parser.add_argument("--full-frame-range", nargs=2, type=int, metavar=("START", "END"), help="Also create empty txt files for frames with no objects.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    merge_object_tracks(args.tracks, args.output_tracks, args.output_labels, args.registry, args.catalog, args.active_only, args.frame_width, args.frame_height, args.overwrite, tuple(args.full_frame_range) if args.full_frame_range else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
