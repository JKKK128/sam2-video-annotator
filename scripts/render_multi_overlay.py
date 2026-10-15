#!/usr/bin/env python3
"""Render multi-object mask and bbox overlays from merged tracks."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.frame_utils import frame_path
from sam2_annotator.tracking_io import load_jsonl

LOGGER = logging.getLogger("render_multi_overlay")
COLORS = {1: (0, 255, 0), 2: (255, 128, 0), 3: (0, 128, 255)}


def group_tracks(records: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(int(record["frame_idx"]), []).append(record)
    return grouped


def render_multi_overlay(frames_dir: Path, tracks: Path, output: Path, start_frame: int, end_frame: int, fps: float | None) -> None:
    import cv2

    frames_dir = frames_dir.expanduser().resolve()
    tracks = tracks.expanduser().resolve()
    output = output.expanduser().resolve()
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("start-frame/end-frame must be non-negative and increasing")
    first_path = frame_path(frames_dir, start_frame)
    if not first_path.is_file():
        raise FileNotFoundError(f"start frame does not exist: {first_path}")
    first = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
    if first is None:
        raise OSError(f"failed to read frame: {first_path}")
    height, width = first.shape[:2]
    if fps is None:
        metadata_path = frames_dir / "metadata.json"
        fps = float(json.loads(metadata_path.read_text(encoding="utf-8")).get("fps", 30.0)) if metadata_path.is_file() else 30.0
    if fps <= 0:
        raise ValueError("fps must be positive")
    grouped = group_tracks(load_jsonl(tracks))

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"failed to open VideoWriter: {output}")
    try:
        for frame_idx in range(start_frame, end_frame + 1):
            path = frame_path(frames_dir, frame_idx)
            if not path.is_file():
                raise FileNotFoundError(f"frame does not exist: {path}")
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise OSError(f"failed to read frame: {path}")
            if frame.shape[:2] != (height, width):
                raise ValueError(f"frame size changed at {path}")
            for record in sorted(grouped.get(frame_idx, []), key=lambda item: int(item["obj_id"])):
                _draw_record(frame, record, tracks)
            _draw_text(frame, f"frame {frame_idx}", (12, 28), (255, 255, 255))
            writer.write(frame)
    finally:
        writer.release()
    if not output.is_file() or output.stat().st_size == 0:
        raise OSError(f"failed to create overlay video: {output}")
    LOGGER.info("wrote multi-object overlay: %s", output)


def _draw_record(frame: np.ndarray, record: dict[str, object], tracks_path: Path) -> None:
    import cv2

    obj_id = int(record["obj_id"])
    color = COLORS.get(obj_id, (255, 255, 255))
    mask_path_value = record.get("mask_path")
    if mask_path_value:
        mask_path = Path(str(mask_path_value)).expanduser()
        if not mask_path.is_absolute():
            mask_path = tracks_path.parent / mask_path
        if mask_path.is_file():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise OSError(f"failed to read mask: {mask_path}")
            mask_bool = mask > 0
            overlay_color = np.array(color, dtype=np.uint8)
            frame[mask_bool] = (0.55 * frame[mask_bool] + 0.45 * overlay_color).astype(np.uint8)
    bbox = record.get("bbox_xyxy")
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]  # type: ignore[union-attr]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"obj{obj_id} {record.get('object_name', '')}".strip()
        _draw_text(frame, label, (x1, max(18, y1 - 6)), color)


def _draw_text(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    import cv2

    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an MP4 showing all tracked obj_id values present on each original frame.")
    parser.add_argument("--frames-dir", required=True, type=Path, help="Original full frame directory.")
    parser.add_argument("--tracks", required=True, type=Path, help="Merged tracks JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Output MP4 path.")
    parser.add_argument("--start-frame", required=True, type=int, help="Inclusive original start frame.")
    parser.add_argument("--end-frame", required=True, type=int, help="Inclusive original end frame.")
    parser.add_argument("--fps", type=float, help="Override FPS. Defaults to frame metadata.json fps, then 30.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    render_multi_overlay(args.frames_dir, args.tracks, args.output, args.start_frame, args.end_frame, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
