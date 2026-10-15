#!/usr/bin/env python3
"""Build, merge, validate, and optionally render one video's final export."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_object_track import _as_frame_range, _load_timeline, _timeline_object, _video_id, build_object_track
from merge_object_tracks import merge_object_tracks
from render_multi_overlay import render_multi_overlay
from sam2_annotator.object_registry import load_registry
from validate_annotation_export import validate_annotation_export

LOGGER = logging.getLogger("finalize_video_annotation")


def _active_validated(registry: dict[str, Any]) -> list[dict[str, Any]]:
    active = [item for item in registry["objects"] if item["status"] == "active" and item["validated"] is True]
    if len([item for item in registry["objects"] if item["status"] == "active"]) > 2:
        raise ValueError("a video can have at most two active objects")
    if not active:
        raise ValueError("registry has no active, validated objects to export")
    return sorted(active, key=lambda item: int(item["obj_id"]))


def _commit(staging: Path, output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    backup = output.with_name(f".{output.name}.backup")
    if backup.exists():
        raise FileExistsError(f"stale finalization backup exists: {backup}")
    moved_old = False
    try:
        if output.exists():
            os.replace(output, backup)
            moved_old = True
        os.replace(staging, output)
    except BaseException:
        if moved_old and not output.exists() and backup.exists():
            os.replace(backup, output)
        raise
    if moved_old:
        shutil.rmtree(backup)


def finalize_video_annotation(
    timeline: Path, registry: Path, catalog: Path, frames_dir: Path, output_dir: Path, overwrite: bool,
    render_overlay: bool = False, frame_width: int | None = None, frame_height: int | None = None,
) -> dict[str, Any]:
    """Create a complete final export transactionally and return its summary."""
    timeline, registry, catalog, frames_dir = (path.expanduser().resolve() for path in (timeline, registry, catalog, frames_dir))
    output_dir = output_dir.expanduser().resolve()
    if not frames_dir.is_dir():
        raise NotADirectoryError(f"frames directory does not exist: {frames_dir}")
    registry_data = load_registry(registry)
    objects = _active_validated(registry_data)
    timeline_data = _load_timeline(timeline)
    ranges = [_as_frame_range(_timeline_object(timeline_data, int(item["obj_id"])).get("frame_range"), f"obj_id {item['obj_id']} frame_range") for item in objects]
    full_range = (min(item[0] for item in ranges), max(item[1] for item in ranges))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        object_tracks: list[Path] = []
        object_summaries: list[dict[str, Any]] = []
        for item in objects:
            obj_id = int(item["obj_id"])
            track_path = staging / "objects" / f"obj{obj_id}" / "final_tracks.jsonl"
            object_summaries.append(build_object_track(timeline, registry, catalog, obj_id, track_path, False))
            object_tracks.append(track_path)
        merged = staging / "merged" / "tracks.jsonl"
        labels = staging / "labels"
        records = merge_object_tracks(object_tracks, merged, labels, registry, catalog, True, frame_width, frame_height, False, full_range)
        report = validate_annotation_export(merged, labels, registry, catalog, full_range, True, staging / "quality_report.json")
        if render_overlay:
            render_multi_overlay(frames_dir, merged, staging / "overlay.mp4", full_range[0], full_range[1], None)
        summary = {
            "video_id": _video_id(registry_data), "active_objects": [int(item["obj_id"]) for item in objects],
            "global_drone_ids": [str(item["global_drone_id"]) for item in objects], "class_ids": [int(item["class_id"]) for item in objects],
            "frame_range": list(full_range), "records_by_object": {str(item["obj_id"]): item["total_records"] for item in object_summaries},
            "total_labels": report["total_label_lines"], "quality_valid": report["valid"], "output_dir": str(output_dir),
        }
        _commit(staging, output_dir, overwrite)
        LOGGER.info("finalization summary: %s", json.dumps(summary, ensure_ascii=False))
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-command transactional final build and quality validation for one video.")
    parser.add_argument("--timeline", type=Path, default=Path("data/object_timelines.json"), help="Object timeline JSON.")
    parser.add_argument("--registry", type=Path, default=Path("data/object_registry.json"), help="Current-video registry JSON.")
    parser.add_argument("--catalog", type=Path, default=Path("configs/drone_catalog.json"), help="Global drone catalog JSON.")
    parser.add_argument("--frames-dir", required=True, type=Path, help="Original extracted frames directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Final export directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a prior completed export only after this build succeeds.")
    parser.add_argument("--render-overlay", action="store_true", help="Also render overlay.mp4 from the merged tracks.")
    parser.add_argument("--frame-width", type=int, help="Override label normalization frame width.")
    parser.add_argument("--frame-height", type=int, help="Override label normalization frame height.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    finalize_video_annotation(args.timeline, args.registry, args.catalog, args.frames_dir, args.output_dir, args.overwrite, args.render_overlay, args.frame_width, args.frame_height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
