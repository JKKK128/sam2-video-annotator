#!/usr/bin/env python3
"""Extract every frame from an MP4 into zero-padded JPEG files."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.frame_utils import frame_path

LOGGER = logging.getLogger("extract_frames")


def _progress(iterable, total: int):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, unit="frame")


def positive_quality(value: str) -> int:
    quality = int(value)
    if quality < 1 or quality > 100:
        raise argparse.ArgumentTypeError("JPEG quality must be between 1 and 100")
    return quality


def extract_frames(video_path: Path, frames_dir: Path, jpeg_quality: int, overwrite: bool) -> dict[str, object]:
    import cv2

    video_path = video_path.expanduser().resolve()
    frames_dir = frames_dir.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"input video does not exist: {video_path}")
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0

    written = 0
    skipped_existing = 0
    read_count = 0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

    for idx in _progress(range(frame_count if frame_count > 0 else 10**18), max(frame_count, 0)):
        ok, frame = cap.read()
        if not ok:
            break
        read_count += 1
        output_path = frame_path(frames_dir, idx)
        if output_path.exists() and not overwrite:
            skipped_existing += 1
            continue
        if not cv2.imwrite(str(output_path), frame, encode_params):
            raise OSError(f"failed to write frame: {output_path}")
        written += 1

    cap.release()

    if frame_count > 0 and read_count != frame_count:
        raise RuntimeError(f"exported frame count mismatch: OpenCV reported {frame_count}, read {read_count}")

    metadata = {
        "video_path": str(video_path),
        "fps": fps,
        "frame_count": read_count,
        "opencv_reported_frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "output_pattern": str(frames_dir / "%06d.jpg"),
        "jpeg_quality": jpeg_quality,
        "written_frames": written,
        "skipped_existing_frames": skipped_existing,
    }
    metadata_path = frames_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("wrote metadata: %s", metadata_path)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract all frames from an MP4 as 000000.jpg, 000001.jpg, ...")
    parser.add_argument("--video", required=True, type=Path, help="Input MP4 video path.")
    parser.add_argument("--frames-dir", required=True, type=Path, help="Directory where JPEG frames are written.")
    parser.add_argument("--jpeg-quality", type=positive_quality, default=95, help="JPEG quality from 1 to 100. Default: 95.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frame JPEGs.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    metadata = extract_frames(args.video, args.frames_dir, args.jpeg_quality, args.overwrite)
    LOGGER.info("read=%s written=%s skipped_existing=%s", metadata["frame_count"], metadata["written_frames"], metadata["skipped_existing_frames"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
