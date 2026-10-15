#!/usr/bin/env python3
"""Create a symlinked, zero-based tracking segment from global frames."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.frame_utils import frame_name, frame_path
from sam2_annotator.tracking_io import convert_prompt_to_segment, load_json, write_json

LOGGER = logging.getLogger("create_tracking_segment")


def create_tracking_segment(
    source_frames_dir: Path,
    start_frame: int,
    end_frame: int,
    output_dir: Path,
    prompt: Path | None,
    output_prompt: Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    import cv2

    source_frames_dir = source_frames_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source_frames_dir.is_dir():
        raise FileNotFoundError(f"source frames directory does not exist: {source_frames_dir}")
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("start-frame/end-frame must be non-negative and increasing")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output dir exists; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    for local_idx, original_idx in enumerate(range(start_frame, end_frame + 1)):
        source = frame_path(source_frames_dir, original_idx)
        if not source.is_file():
            raise FileNotFoundError(f"source frame does not exist: {source}")
        target = output_dir / frame_name(local_idx)
        os.symlink(source, target)
        frames.append({"local_frame_idx": local_idx, "original_frame_idx": original_idx, "source_path": str(source), "segment_path": str(target)})

    first = cv2.imread(str(output_dir / frame_name(0)), cv2.IMREAD_COLOR)
    if first is None:
        raise OSError(f"failed to read first segment frame: {output_dir / frame_name(0)}")
    height, width = first.shape[:2]
    source_metadata_path = source_frames_dir / "metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8")) if source_metadata_path.is_file() else {}
    fps = source_metadata.get("fps")
    manifest = {
        "source_frames_dir": str(source_frames_dir),
        "segment_dir": str(output_dir),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_count": len(frames),
        "frame_width": width,
        "frame_height": height,
        "fps": fps,
        "frames": frames,
    }
    write_json(output_dir / "segment_manifest.json", manifest)
    write_json(
        output_dir / "metadata.json",
        {
            "source_frames_dir": str(source_frames_dir),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "fps": fps,
            "frame_width": width,
            "frame_height": height,
            "frame_count": len(frames),
        },
    )

    if prompt is not None:
        converted = convert_prompt_to_segment(load_json(prompt), manifest)
        if output_prompt is None:
            output_prompt = output_dir / "prompt_local.json"
        write_json(output_prompt, converted)
        LOGGER.info("wrote local prompt: %s", output_prompt)

    LOGGER.info("wrote segment manifest: %s", output_dir / "segment_manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a symlinked tracking segment with local 000000.jpg numbering.")
    parser.add_argument("--source-frames-dir", required=True, type=Path, help="Original full frame directory.")
    parser.add_argument("--start-frame", required=True, type=int, help="Inclusive original start frame.")
    parser.add_argument("--end-frame", required=True, type=int, help="Inclusive original end frame.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output segment directory.")
    parser.add_argument("--prompt", type=Path, help="Optional global prompt JSON to convert into local segment coordinates.")
    parser.add_argument("--output-prompt", type=Path, help="Optional output local prompt JSON path.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output-dir if it already exists.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    create_tracking_segment(args.source_frames_dir, args.start_frame, args.end_frame, args.output_dir, args.prompt, args.output_prompt, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
