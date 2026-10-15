#!/usr/bin/env python3
"""Create a box prompt JSON from a frame, using cv2.selectROI or manual coordinates."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.frame_utils import frame_path

LOGGER = logging.getLogger("select_box")


def validate_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        raise ValueError(f"box coordinates must be non-negative: {box}")
    if x2 >= width or y2 >= height:
        raise ValueError(f"box {box} exceeds frame bounds {width}x{height}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box must have non-zero area with x2>x1 and y2>y1: {box}")
    return box


def create_prompt(frames_dir: Path, frame_idx: int, output_json: Path, obj_id: int, box: tuple[int, int, int, int] | None) -> dict[str, object]:
    import cv2

    if frame_idx < 0:
        raise ValueError("frame_idx must be non-negative")
    image_path = frame_path(frames_dir.expanduser().resolve(), frame_idx)
    if not image_path.is_file():
        raise FileNotFoundError(f"frame does not exist: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"failed to read frame: {image_path}")
    height, width = image.shape[:2]

    if box is None:
        roi = cv2.selectROI("Select target box", image, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select target box")
        x, y, w, h = [int(v) for v in roi]
        if w <= 0 or h <= 0:
            raise ValueError("ROI selection was cancelled or has zero area")
        box = (x, y, x + w - 1, y + h - 1)

    box = validate_box(tuple(int(v) for v in box), width, height)
    output_json = output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    prompt = {
        "frame_idx": frame_idx,
        "obj_id": obj_id,
        "prompt_type": "box",
        "box_xyxy": list(box),
        "frame_width": width,
        "frame_height": height,
        "frame_path": str(image_path),
    }
    output_json.write_text(json.dumps(prompt, indent=2) + "\n", encoding="utf-8")

    preview = image.copy()
    cv2.rectangle(preview, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    preview_path = output_json.with_name(output_json.stem + "_preview.jpg")
    if not cv2.imwrite(str(preview_path), preview):
        raise OSError(f"failed to write preview image: {preview_path}")
    LOGGER.info("wrote prompt: %s", output_json)
    LOGGER.info("wrote preview: %s", preview_path)
    return prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select or provide a single object box prompt for a frame.")
    parser.add_argument("--frames-dir", required=True, type=Path, help="Directory containing zero-padded JPEG frames.")
    parser.add_argument("--frame-idx", required=True, type=int, help="Original frame index to prompt.")
    parser.add_argument("--output", required=True, type=Path, help="Output prompt JSON path.")
    parser.add_argument("--obj-id", type=int, default=1, help="Object id. First milestone supports obj_id=1. Default: 1.")
    parser.add_argument("--box", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), help="Manual inclusive xyxy box coordinates.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    if args.obj_id != 1:
        raise ValueError("first milestone supports only obj_id=1")
    box = tuple(args.box) if args.box is not None else None
    create_prompt(args.frames_dir, args.frame_idx, args.output, args.obj_id, box)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
