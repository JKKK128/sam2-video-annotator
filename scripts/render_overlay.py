#!/usr/bin/env python3
"""Render mask and bbox overlays from tracking outputs into an MP4."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.frame_utils import parse_frame_index, sorted_frame_paths

LOGGER = logging.getLogger("render_overlay")


def load_tracks(path: Path) -> dict[int, list[dict[str, object]]]:
    if not path.is_file():
        raise FileNotFoundError(f"tracks JSONL does not exist: {path}")
    records: dict[int, list[dict[str, object]]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "frame_idx" not in record:
                raise ValueError(f"tracks line {line_no} missing frame_idx")
            records.setdefault(int(record["frame_idx"]), []).append(record)
    return records


def render_overlay(frames_dir: Path, tracks_path: Path, output_video: Path, fps: float | None, bbox_mode: str) -> None:
    import cv2

    frames_dir = frames_dir.expanduser().resolve()
    tracks_path = tracks_path.expanduser().resolve()
    output_video = output_video.expanduser().resolve()
    frame_paths = sorted_frame_paths(frames_dir)
    if not frame_paths:
        raise FileNotFoundError(f"no JPEG frames found in {frames_dir}")

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise OSError(f"failed to read frame: {frame_paths[0]}")
    height, width = first.shape[:2]

    if fps is None:
        metadata_path = frames_dir / "metadata.json"
        if metadata_path.is_file():
            fps = float(json.loads(metadata_path.read_text(encoding="utf-8")).get("fps", 30.0))
        else:
            fps = 30.0
            LOGGER.warning("metadata.json not found; using fps=30")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"failed to open VideoWriter: {output_video}")

    tracks = load_tracks(tracks_path)
    try:
        for frame_file in frame_paths:
            frame = cv2.imread(str(frame_file), cv2.IMREAD_COLOR)
            if frame is None:
                raise OSError(f"failed to read frame: {frame_file}")
            if frame.shape[:2] != (height, width):
                raise ValueError(f"frame size changed at {frame_file}: {frame.shape[1]}x{frame.shape[0]}")

            frame_idx = parse_frame_index(frame_file)
            frame_records = tracks.get(frame_idx, [])
            if frame_records:
                for record in frame_records:
                    _draw_record_overlay(frame, record, tracks_path, bbox_mode)
                statuses = ",".join(str(record.get("bbox_status", "unknown")) for record in frame_records)
                obj_ids = ",".join(str(int(record.get("obj_id", 1))) for record in frame_records)
                cv2.putText(frame, f"frame {frame_idx} obj {obj_ids} {statuses}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                cv2.putText(frame, f"frame {frame_idx} obj {obj_ids} {statuses}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            else:
                cv2.putText(frame, f"frame {frame_idx}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                cv2.putText(frame, f"frame {frame_idx}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

            writer.write(frame)
    finally:
        writer.release()

    if not output_video.is_file() or output_video.stat().st_size == 0:
        raise OSError(f"OpenCV did not create a valid output video: {output_video}")
    LOGGER.info("wrote overlay video: %s", output_video)


def _draw_record_overlay(frame: np.ndarray, record: dict[str, object], tracks_path: Path, bbox_mode: str) -> None:
    import cv2

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
            color = np.array([0, 255, 0], dtype=np.uint8)
            frame[mask_bool] = (0.55 * frame[mask_bool] + 0.45 * color).astype(np.uint8)
    _draw_bboxes(frame, record, bbox_mode)


def _draw_bboxes(frame: np.ndarray, record: dict[str, object], bbox_mode: str) -> None:
    import cv2

    if bbox_mode == "raw":
        _draw_bbox(frame, record.get("raw_bbox_xyxy") or record.get("bbox_xyxy"), (0, 255, 255), 1)
        _draw_text(frame, f"raw bbox {record.get('bbox_status', '')}", (12, 58), (0, 255, 255))
    elif bbox_mode == "stable":
        _draw_bbox(frame, record.get("stable_bbox_xyxy") or record.get("bbox_xyxy"), (0, 255, 0), 2)
        _draw_text(frame, f"stable bbox {record.get('bbox_status', '')}", (12, 58), (0, 255, 0))
    elif bbox_mode == "both":
        _draw_bbox(frame, record.get("raw_bbox_xyxy") or record.get("bbox_xyxy"), (0, 255, 255), 1)
        _draw_bbox(frame, record.get("stable_bbox_xyxy") or record.get("bbox_xyxy"), (0, 255, 0), 3)
        _draw_text(frame, f"raw thin yellow | stable thick green {record.get('bbox_status', '')}", (12, 58), (0, 255, 0))
    else:
        raise ValueError(f"invalid bbox_mode: {bbox_mode}")


def _draw_bbox(frame: np.ndarray, bbox: object, color: tuple[int, int, int], thickness: int) -> None:
    import cv2

    if bbox is None:
        return
    x1, y1, x2, y2 = [int(v) for v in bbox]  # type: ignore[union-attr]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def _draw_text(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    import cv2

    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render masks, boxes, frame index and object id into an MP4 overlay.")
    parser.add_argument("--frames-dir", required=True, type=Path, help="Directory containing original extracted JPEG frames.")
    parser.add_argument("--tracks", required=True, type=Path, help="tracks.jsonl produced by track_box.py.")
    parser.add_argument("--output", required=True, type=Path, help="Output MP4 path.")
    parser.add_argument("--fps", type=float, help="Override FPS. Defaults to frames metadata.json fps, then 30.")
    parser.add_argument("--bbox-mode", choices=("raw", "stable", "both"), default="raw", help="Which bbox to draw. Default: raw.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    render_overlay(args.frames_dir, args.tracks, args.output, args.fps, args.bbox_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
