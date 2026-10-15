#!/usr/bin/env python3
"""Register one object and create its initial box prompt."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.correction_utils import validate_box_xyxy
from sam2_annotator.drone_catalog import get_catalog_entry, load_catalog
from sam2_annotator.frame_utils import frame_path
from sam2_annotator.object_registry import load_registry, save_registry, upsert_object

LOGGER = logging.getLogger("register_object")


def register_object(
    registry_path: Path,
    frames_dir: Path,
    obj_id: int,
    name: str,
    catalog_path: Path,
    global_drone_id: str,
    frame_idx: int,
    prompt_output: Path,
    box: tuple[int, int, int, int] | None,
    replace: bool,
) -> dict[str, Any]:
    import cv2

    frames_dir = frames_dir.expanduser().resolve()
    prompt_output = prompt_output.expanduser().resolve()
    if frame_idx < 0:
        raise ValueError("frame-idx must be non-negative")
    if prompt_output.exists() and not replace:
        raise FileExistsError(f"prompt output already exists; pass --replace: {prompt_output}")
    image_path = frame_path(frames_dir, frame_idx)
    if not image_path.is_file():
        raise FileNotFoundError(f"frame does not exist: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"failed to read frame: {image_path}")
    height, width = image.shape[:2]

    if box is None:
        roi = cv2.selectROI(f"Select {name} box", image, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(f"Select {name} box")
        x, y, w, h = [int(v) for v in roi]
        if w <= 0 or h <= 0:
            raise ValueError("ROI selection was cancelled or has zero area")
        box = (x, y, x + w - 1, y + h - 1)
    box = validate_box_xyxy(tuple(int(v) for v in box), width, height)

    catalog = load_catalog(catalog_path)
    catalog_entry = get_catalog_entry(catalog, global_drone_id)
    registry = load_registry(registry_path)
    registry["frames_dir"] = str(frames_dir)
    upsert_object(registry, obj_id, name, global_drone_id, int(catalog_entry["class_id"]), str(catalog_entry["class_name"]), frame_idx, "active", True)
    save_registry(registry_path, registry)

    prompt = {
        "obj_id": obj_id,
        "object_name": name,
        "global_drone_id": global_drone_id,
        "class_id": catalog_entry["class_id"],
        "class_name": catalog_entry["class_name"],
        "frame_idx": frame_idx,
        "original_frame_idx": frame_idx,
        "prompt_type": "box",
        "box_xyxy": list(box),
        "frame_width": width,
        "frame_height": height,
        "frame_path": str(image_path),
    }
    prompt_output.parent.mkdir(parents=True, exist_ok=True)
    prompt_output.write_text(json.dumps(prompt, indent=2) + "\n", encoding="utf-8")

    preview = image.copy()
    cv2.rectangle(preview, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.putText(preview, f"obj{obj_id} {name}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(preview, f"obj{obj_id} {name}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 160, 0), 1)
    preview_path = prompt_output.with_name(prompt_output.stem + "_preview.jpg")
    if not cv2.imwrite(str(preview_path), preview):
        raise OSError(f"failed to write preview image: {preview_path}")
    LOGGER.info("updated registry: %s", registry_path)
    LOGGER.info("wrote prompt: %s", prompt_output)
    LOGGER.info("wrote preview: %s", preview_path)
    return prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register one local object for this video and create its initial box prompt.")
    parser.add_argument("--registry", required=True, type=Path, help="Object registry JSON to update.")
    parser.add_argument("--catalog", required=True, type=Path, help="Dataset-wide drone catalog JSON.")
    parser.add_argument("--frames-dir", required=True, type=Path, help="Original full frame directory.")
    parser.add_argument("--obj-id", required=True, type=int, help="Positive local object id for this video.")
    parser.add_argument("--name", required=True, help="Unique local display name.")
    parser.add_argument("--global-drone-id", required=True, help="Fixed global drone identity from --catalog.")
    parser.add_argument("--frame-idx", required=True, type=int, help="Original video frame where this object is first prompted.")
    parser.add_argument("--prompt-output", required=True, type=Path, help="Output prompt JSON path.")
    parser.add_argument("--box", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), help="Manual inclusive xyxy box. If omitted, opens cv2.selectROI.")
    parser.add_argument("--replace", action="store_true", help="Overwrite prompt output if it already exists.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    register_object(args.registry, args.frames_dir, args.obj_id, args.name, args.catalog, args.global_drone_id, args.frame_idx, args.prompt_output, tuple(args.box) if args.box else None, args.replace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
