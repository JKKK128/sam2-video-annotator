#!/usr/bin/env python3
"""Merge baseline tracks before a correction frame with corrected tracks from that frame onward."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.correction_utils import load_jsonl, merge_track_records, write_jsonl, write_label_for_record

LOGGER = logging.getLogger("merge_corrected_tracks")


def _resolve_record_path(value: object, tracks_path: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = tracks_path.parent / path
    return path.resolve()


def _infer_image_size(records: list[dict[str, Any]], tracks_path: Path) -> tuple[int, int]:
    import cv2

    for record in records:
        mask_path = _resolve_record_path(record.get("mask_path"), tracks_path)
        if mask_path is None or not mask_path.is_file():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise OSError(f"failed to read mask: {mask_path}")
        height, width = mask.shape[:2]
        return int(width), int(height)
    raise ValueError("could not infer image size from merged record mask_path files")


def merge_corrected_tracks(
    baseline_tracks: Path,
    corrected_tracks: Path,
    correction_frame: int,
    output_tracks: Path,
    obj_id: int | None = None,
) -> list[dict[str, Any]]:
    baseline_tracks = baseline_tracks.expanduser().resolve()
    corrected_tracks = corrected_tracks.expanduser().resolve()
    output_tracks = output_tracks.expanduser().resolve()
    if correction_frame < 0:
        raise ValueError("correction-frame must be non-negative")

    baseline_records = load_jsonl(baseline_tracks)
    corrected_records = load_jsonl(corrected_tracks)
    if obj_id is None and corrected_records:
        corrected_obj_ids = {int(record["obj_id"]) for record in corrected_records}
        if len(corrected_obj_ids) == 1:
            obj_id = next(iter(corrected_obj_ids))
    merged = merge_track_records(baseline_records, corrected_records, correction_frame, obj_id=obj_id)
    write_jsonl(output_tracks, merged)

    width, height = _infer_image_size(merged, output_tracks)
    labels_dir = output_tracks.parent / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for old_label in labels_dir.glob("*.txt"):
        old_label.unlink()
    for record in merged:
        write_label_for_record(record, labels_dir, width, height)

    LOGGER.info("wrote merged tracks: %s", output_tracks)
    LOGGER.info("wrote merged labels: %s", labels_dir)
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge baseline and corrected tracks without modifying the baseline input.")
    parser.add_argument("--baseline-tracks", required=True, type=Path, help="Original baseline tracks.jsonl.")
    parser.add_argument("--corrected-tracks", required=True, type=Path, help="Corrected tracks_corrected.jsonl.")
    parser.add_argument("--correction-frame", required=True, type=int, help="Frame where corrected records begin replacing baseline records.")
    parser.add_argument("--output-tracks", required=True, type=Path, help="Output merged_tracks.jsonl path.")
    parser.add_argument("--obj-id", type=int, help="Only replace this object id at and after correction-frame. Defaults to the corrected tracks obj_id when unambiguous.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    merge_corrected_tracks(args.baseline_tracks, args.corrected_tracks, args.correction_frame, args.output_tracks, args.obj_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
