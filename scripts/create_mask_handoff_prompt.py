#!/usr/bin/env python3
"""Create a mask prompt for a fresh next-segment SAM2 state from a verified prior mask."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.tracking_io import load_jsonl, load_segment_manifest, original_to_local_map, write_json

LOGGER = logging.getLogger("create_mask_handoff_prompt")


def _resolve_mask_path(value: str, tracks_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = tracks_path.parent / path
    return path.resolve()


def create_mask_handoff_prompt(
    source_tracks: Path,
    frame_idx: int,
    obj_id: int,
    next_segment_manifest: Path,
    output: Path,
    verified: bool,
) -> dict[str, Any]:
    import cv2

    if frame_idx < 0:
        raise ValueError("frame-idx must be non-negative")
    source_tracks = source_tracks.expanduser().resolve()
    records = load_jsonl(source_tracks)
    matches = [record for record in records if int(record["frame_idx"]) == frame_idx and int(record["obj_id"]) == obj_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one source record for frame {frame_idx} obj {obj_id}, found {len(matches)}")
    record = matches[0]
    mask_path = _resolve_mask_path(str(record["mask_path"]), source_tracks)
    if not mask_path.is_file():
        raise FileNotFoundError(f"source mask does not exist: {mask_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise OSError(f"failed to read source mask: {mask_path}")
    if int(np.count_nonzero(mask)) == 0:
        raise ValueError(f"source mask is empty: {mask_path}")

    manifest = load_segment_manifest(next_segment_manifest.expanduser().resolve().parent)
    original_to_local = original_to_local_map(manifest)
    if frame_idx not in original_to_local:
        raise ValueError(f"frame {frame_idx} is outside next segment")
    prompt = {
        "obj_id": obj_id,
        "object_name": record.get("object_name", f"obj{obj_id}"),
        "frame_idx": original_to_local[frame_idx],
        "original_frame_idx": frame_idx,
        "local_frame_idx": original_to_local[frame_idx],
        "prompt_type": "mask",
        "mask_path": str(mask_path),
        "frame_width": int(mask.shape[1]),
        "frame_height": int(mask.shape[0]),
        "source_segment": record.get("segment_start_frame"),
        "source_tracks": str(source_tracks),
        "verified": bool(verified),
    }
    write_json(output, prompt)
    LOGGER.info("wrote mask handoff prompt: %s", output)
    return prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a local mask prompt for the next independent segment state from a prior confirmed mask.")
    parser.add_argument("--source-tracks", required=True, type=Path, help="Previous segment tracks.jsonl.")
    parser.add_argument("--frame-idx", required=True, type=int, help="Original frame index whose mask is used as the handoff prompt.")
    parser.add_argument("--obj-id", required=True, type=int, help="Object id to hand off.")
    parser.add_argument("--next-segment-manifest", required=True, type=Path, help="Next segment_manifest.json.")
    parser.add_argument("--output", required=True, type=Path, help="Output mask prompt JSON.")
    parser.add_argument("--verified", action="store_true", help="Mark the handoff mask as human-verified. Default is false.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    create_mask_handoff_prompt(args.source_tracks, args.frame_idx, args.obj_id, args.next_segment_manifest, args.output, args.verified)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
