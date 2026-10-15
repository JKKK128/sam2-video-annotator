#!/usr/bin/env python3
"""Create a same-obj correction box prompt for a SAM2 tracking failure frame."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.correction_utils import atomic_write_json, validate_box_xyxy
from sam2_annotator.frame_utils import frame_path

LOGGER = logging.getLogger("create_correction_prompt")


def create_correction_prompt(
    frames_dir: Path,
    frame_idx: int,
    obj_id: int,
    object_name: str | None,
    output: Path,
    box: tuple[int, int, int, int] | None,
    reason: str,
    original_start_frame: int | None,
    replace: bool,
) -> dict[str, Any]:
    import cv2

    frames_dir = frames_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    if frame_idx < 0:
        raise ValueError("frame_idx must be non-negative")
    if obj_id not in {1, 2, 3}:
        raise ValueError("current project supports only obj_id=1,2,3")
    if output.exists() and not replace:
        raise FileExistsError(f"output already exists; pass --replace to overwrite: {output}")
    image_path = frame_path(frames_dir, frame_idx)
    if not image_path.is_file():
        raise FileNotFoundError(f"frame does not exist: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"failed to read frame: {image_path}")
    height, width = image.shape[:2]

    if box is None:
        roi = cv2.selectROI("Select correction box", image, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select correction box")
        x, y, w, h = [int(v) for v in roi]
        if w <= 0 or h <= 0:
            raise ValueError("ROI selection was cancelled or has zero area")
        box = (x, y, x + w - 1, y + h - 1)

    validated_box = validate_box_xyxy(tuple(int(v) for v in box), width, height)
    prompt = {
        "obj_id": obj_id,
        "object_name": object_name or f"obj{obj_id}",
        "frame_idx": frame_idx,
        "original_frame_idx": int(original_start_frame) + frame_idx if original_start_frame is not None else frame_idx,
        "prompt_type": "box",
        "box_xyxy": list(validated_box),
        "correction_reason": reason,
        "replace_prompt_on_frame": True,
        "frame_width": width,
        "frame_height": height,
        "frame_path": str(image_path),
    }
    atomic_write_json(output, prompt)

    preview = image.copy()
    cv2.rectangle(preview, (validated_box[0], validated_box[1]), (validated_box[2], validated_box[3]), (0, 0, 255), 2)
    cv2.putText(preview, "CORRECTION", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(preview, "CORRECTION", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
    preview_path = output.with_name(output.stem + "_preview.jpg")
    if not cv2.imwrite(str(preview_path), preview):
        raise OSError(f"failed to write preview image: {preview_path}")

    LOGGER.info("wrote correction prompt: %s", output)
    LOGGER.info("wrote preview: %s", preview_path)
    return prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a correction box prompt for the first truly wrong SAM2 tracking frame.")
    parser.add_argument("--frames-dir", required=True, type=Path, help="Directory containing zero-padded JPEG frames.")
    parser.add_argument("--frame-idx", required=True, type=int, help="Correction frame index within frames-dir.")
    parser.add_argument("--obj-id", required=True, type=int, help="Existing object id to correct. Must match the original object id.")
    parser.add_argument("--object-name", help="Object name to store in the correction prompt, e.g. drone_2.")
    parser.add_argument("--output", required=True, type=Path, help="Output correction JSON, e.g. data/corrections/obj1_frame000035_box.json.")
    parser.add_argument("--box", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), help="Manual inclusive xyxy correction box.")
    parser.add_argument("--reason", required=True, help="Human-readable correction reason such as mask_drift, empty_mask, wrong_target.")
    parser.add_argument("--original-start-frame", type=int, help="Original video frame index for frames-dir frame 0, when known.")
    parser.add_argument("--replace", action="store_true", help="Overwrite the output JSON if it already exists.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    box = tuple(args.box) if args.box is not None else None
    create_correction_prompt(
        args.frames_dir,
        args.frame_idx,
        args.obj_id,
        args.object_name,
        args.output,
        box,
        args.reason,
        args.original_start_frame,
        args.replace,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
