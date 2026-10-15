#!/usr/bin/env python3
"""Render a before/after correction comparison video."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.correction_utils import render_range
from sam2_annotator.frame_utils import frame_path, parse_frame_index, sorted_frame_paths

LOGGER = logging.getLogger("render_correction_compare")


def _load_tracks(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"tracks JSONL does not exist: {path}")
    records: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (int(record["frame_idx"]), int(record["obj_id"]))
            if key in records:
                raise ValueError(f"duplicate track key {key} in {path} line {line_no}")
            records[key] = record
    return records


def _resolve_path(value: object, tracks_path: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = tracks_path.parent / path
    return path.resolve()


def _draw_panel(
    frame: np.ndarray,
    record: dict[str, Any] | None,
    tracks_path: Path,
    title: str,
    frame_idx: int,
    correction_frame: int,
) -> np.ndarray:
    import cv2

    panel = frame.copy()
    if record is not None:
        mask_path = _resolve_path(record.get("mask_path"), tracks_path)
        if mask_path is not None and mask_path.is_file():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise OSError(f"failed to read mask: {mask_path}")
            if mask.shape[:2] != panel.shape[:2]:
                raise ValueError(f"mask size changed at {mask_path}: {mask.shape[1]}x{mask.shape[0]}")
            mask_bool = mask > 0
            color = np.array([0, 180, 255], dtype=np.uint8)
            panel[mask_bool] = (0.55 * panel[mask_bool] + 0.45 * color).astype(np.uint8)
        bbox = record.get("bbox_xyxy") or record.get("raw_bbox_xyxy")
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
        obj_id = int(record.get("obj_id", 1))
        status = f"frame {frame_idx} obj {obj_id}"
    else:
        status = f"frame {frame_idx} no track"

    _draw_text(panel, title, (12, 28), (255, 255, 255), scale=0.78)
    _draw_text(panel, status, (12, 58), (255, 255, 255), scale=0.68)
    if frame_idx == correction_frame:
        _draw_text(panel, "CORRECTION FRAME", (12, 92), (0, 0, 255), scale=0.82)
    return panel


def _draw_text(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int], scale: float) -> None:
    import cv2

    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def render_correction_compare(
    frames_dir: Path,
    baseline_tracks: Path,
    corrected_tracks: Path,
    correction_frame: int,
    output: Path,
    start_frame: int | None,
    end_frame: int | None,
    fps: float | None,
) -> tuple[int, int]:
    import cv2

    frames_dir = frames_dir.expanduser().resolve()
    baseline_tracks = baseline_tracks.expanduser().resolve()
    corrected_tracks = corrected_tracks.expanduser().resolve()
    output = output.expanduser().resolve()
    frame_paths = sorted_frame_paths(frames_dir)
    if not frame_paths:
        raise FileNotFoundError(f"no JPEG frames found in {frames_dir}")
    available = [parse_frame_index(path) for path in frame_paths]
    start, end = render_range(available, correction_frame, start_frame, end_frame)

    first = cv2.imread(str(frame_path(frames_dir, start)), cv2.IMREAD_COLOR)
    if first is None:
        raise OSError(f"failed to read frame: {frame_path(frames_dir, start)}")
    height, width = first.shape[:2]
    if fps is None:
        metadata_path = frames_dir / "metadata.json"
        fps = float(json.loads(metadata_path.read_text(encoding="utf-8")).get("fps", 30.0)) if metadata_path.is_file() else 30.0
    if fps <= 0:
        raise ValueError("fps must be positive")

    baseline = _load_tracks(baseline_tracks)
    corrected = _load_tracks(corrected_tracks)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
    if not writer.isOpened():
        raise OSError(f"failed to open VideoWriter: {output}")

    try:
        for frame_idx in range(start, end + 1):
            image_path = frame_path(frames_dir, frame_idx)
            if not image_path.is_file():
                continue
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise OSError(f"failed to read frame: {image_path}")
            if frame.shape[:2] != (height, width):
                raise ValueError(f"frame size changed at {image_path}: {frame.shape[1]}x{frame.shape[0]}")
            before_record = baseline.get((frame_idx, 1))
            after_record = corrected.get((frame_idx, 1)) if frame_idx >= correction_frame else before_record
            left = _draw_panel(frame, before_record, baseline_tracks, "Before correction", frame_idx, correction_frame)
            right = _draw_panel(frame, after_record, corrected_tracks if frame_idx >= correction_frame else baseline_tracks, "After correction", frame_idx, correction_frame)
            writer.write(np.hstack([left, right]))
    finally:
        writer.release()

    if not output.is_file() or output.stat().st_size == 0:
        raise OSError(f"OpenCV did not create a valid output video: {output}")
    LOGGER.info("wrote correction comparison video: %s", output)
    return start, end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render side-by-side baseline versus corrected SAM2 tracks.")
    parser.add_argument("--frames-dir", required=True, type=Path, help="Directory containing zero-padded JPEG frames.")
    parser.add_argument("--baseline-tracks", required=True, type=Path, help="Baseline tracks.jsonl.")
    parser.add_argument("--corrected-tracks", required=True, type=Path, help="Corrected tracks JSONL.")
    parser.add_argument("--correction-frame", required=True, type=int, help="Correction frame index.")
    parser.add_argument("--output", required=True, type=Path, help="Output MP4 path.")
    parser.add_argument("--start-frame", type=int, help="Optional inclusive start frame. Default: correction_frame - 5.")
    parser.add_argument("--end-frame", type=int, help="Optional inclusive end frame. Default: correction_frame + 50.")
    parser.add_argument("--fps", type=float, help="Override FPS. Defaults to frames metadata.json fps, then 30.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    render_correction_compare(
        args.frames_dir,
        args.baseline_tracks,
        args.corrected_tracks,
        args.correction_frame,
        args.output,
        args.start_frame,
        args.end_frame,
        args.fps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
