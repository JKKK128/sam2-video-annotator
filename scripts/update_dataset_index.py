#!/usr/bin/env python3
"""Add or replace one finalized video's entry in a dataset-level index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.object_registry import load_registry


def update_dataset_index(index_path: Path, video_id: str, registry_path: Path, frames_dir: Path, tracks_path: Path, labels_dir: Path, review_status: str) -> dict[str, Any]:
    if not video_id:
        raise ValueError("video-id must be non-empty")
    registry = load_registry(registry_path)
    frames_dir = frames_dir.expanduser().resolve()
    tracks_path = tracks_path.expanduser().resolve()
    labels_dir = labels_dir.expanduser().resolve()
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames directory does not exist: {frames_dir}")
    if not tracks_path.is_file():
        raise FileNotFoundError(f"final tracks does not exist: {tracks_path}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"final labels directory does not exist: {labels_dir}")
    active = [obj for obj in registry["objects"] if obj["status"] == "active"]
    entry = {
        "video_id": video_id,
        "registry_path": str(registry_path.expanduser().resolve()),
        "frames_dir": str(frames_dir),
        "global_drone_ids": [obj["global_drone_id"] for obj in active],
        "class_ids": [obj["class_id"] for obj in active],
        "final_tracks_path": str(tracks_path),
        "final_labels_path": str(labels_dir),
        "review_status": review_status,
    }
    index_path = index_path.expanduser().resolve()
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("videos"), list):
            raise ValueError("dataset index must be an object containing videos list")
    else:
        data = {"videos": []}
    data["videos"] = [item for item in data["videos"] if item.get("video_id") != video_id]
    data["videos"].append(entry)
    data["videos"].sort(key=lambda item: str(item["video_id"]))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Record one finalized video in the dataset-level annotation index.")
    parser.add_argument("--index", required=True, type=Path, help="Dataset index JSON to create or update.")
    parser.add_argument("--video-id", required=True, help="Unique dataset video identifier.")
    parser.add_argument("--registry", required=True, type=Path, help="Per-video object registry JSON.")
    parser.add_argument("--frames-dir", required=True, type=Path, help="Extracted original frames directory.")
    parser.add_argument("--final-tracks", required=True, type=Path, help="Final merged tracks JSONL.")
    parser.add_argument("--final-labels", required=True, type=Path, help="Final YOLO labels directory.")
    parser.add_argument("--review-status", required=True, help="Audit/review status, for example approved or pending.")
    args = parser.parse_args()
    update_dataset_index(args.index, args.video_id, args.registry, args.frames_dir, args.final_tracks, args.final_labels, args.review_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
