#!/usr/bin/env python3
"""Recompute raw and stable bbox fields from existing mask files without SAM2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.bbox_stabilizer import BBoxStabilizer, clean_mask_components, parameters_from_mapping
from sam2_annotator.config_utils import load_yaml_config
from sam2_annotator.mask_utils import bbox_area, bbox_center, mask_area, mask_to_bbox, save_binary_mask_png


def load_tracks(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"tracks JSONL does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"tracks JSONL has no records: {path}")
    return sorted(records, key=lambda record: (int(record["obj_id"]), int(record["frame_idx"])))


def recompute(tracks_path: Path, output_tracks: Path, config_path: Path, cleaned_masks_dir: Path | None) -> None:
    import cv2

    tracks_path = tracks_path.expanduser().resolve()
    output_tracks = output_tracks.expanduser().resolve()
    config = load_yaml_config(config_path.expanduser().resolve())
    params = parameters_from_mapping(config)
    stabilizers: dict[int, BBoxStabilizer] = {}
    output_tracks.parent.mkdir(parents=True, exist_ok=True)
    if cleaned_masks_dir is not None:
        cleaned_masks_dir = cleaned_masks_dir.expanduser().resolve()
        cleaned_masks_dir.mkdir(parents=True, exist_ok=True)

    records = load_tracks(tracks_path)
    with output_tracks.open("w", encoding="utf-8") as file:
        for record in records:
            obj_id = int(record["obj_id"])
            mask_path = Path(str(record.get("raw_mask_path") or record.get("mask_path"))).expanduser()
            if not mask_path.is_absolute():
                mask_path = tracks_path.parent / mask_path
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise OSError(f"failed to read mask: {mask_path}")
            height, width = mask.shape[:2]
            stabilizer = stabilizers.setdefault(obj_id, BBoxStabilizer(params))
            cleaned, component_info = clean_mask_components(
                mask > 0,
                previous_center=stabilizer.previous_center,
                recent_area=stabilizer.recent_area,
                min_component_area=params.min_component_area,
            )
            cleaned_mask_path = record.get("cleaned_mask_path") or record.get("mask_path")
            if cleaned_masks_dir is not None:
                cleaned_mask_path = cleaned_masks_dir / Path(str(record["mask_path"])).name
                save_binary_mask_png(cleaned, cleaned_mask_path)
            raw_bbox = mask_to_bbox(mask)
            cleaned_bbox = mask_to_bbox(cleaned)
            stability = stabilizer.update(raw_bbox, (width, height), candidate_bbox_xyxy=cleaned_bbox)
            center = bbox_center(stability.stable_bbox_xyxy)
            updated = dict(record)
            updated.update(
                {
                    "raw_mask_path": str(mask_path),
                    "cleaned_mask_path": str(cleaned_mask_path) if cleaned_mask_path is not None else None,
                    "raw_bbox_xyxy": list(stability.raw_bbox_xyxy) if stability.raw_bbox_xyxy is not None else None,
                    "stable_bbox_xyxy": list(stability.stable_bbox_xyxy) if stability.stable_bbox_xyxy is not None else None,
                    "bbox_xyxy": list(stability.stable_bbox_xyxy) if stability.stable_bbox_xyxy is not None else None,
                    "cleaned_bbox_xyxy": list(cleaned_bbox) if cleaned_bbox is not None else None,
                    "bbox_status": stability.bbox_status,
                    "bbox_stabilized": stability.bbox_stabilized,
                    "bbox_stability_reasons": stability.bbox_stability_reasons,
                    "component_count": component_info.component_count,
                    "selected_component_area": component_info.selected_component_area,
                    "removed_component_count": component_info.removed_component_count,
                    "mask_area": mask_area(cleaned),
                    "raw_mask_area": mask_area(mask),
                    "bbox_area": bbox_area(stability.stable_bbox_xyxy),
                    "raw_bbox_area": bbox_area(raw_bbox),
                    "center_xy": list(center) if center is not None else None,
                    "raw_center_xy": list(stability.raw_center_xy) if stability.raw_center_xy is not None else None,
                    "stable_center_xy": list(stability.stable_center_xy) if stability.stable_center_xy is not None else None,
                    "raw_width": stability.raw_width,
                    "raw_height": stability.raw_height,
                    "stable_width": stability.stable_width,
                    "stable_height": stability.stable_height,
                    "empty_mask": cleaned_bbox is None,
                }
            )
            file.write(json.dumps(updated) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute raw/stable bbox fields from existing mask PNGs without rerunning SAM2.")
    parser.add_argument("--tracks", required=True, type=Path, help="Existing tracks.jsonl.")
    parser.add_argument("--output-tracks", required=True, type=Path, help="Output tracks.jsonl with stable bbox fields.")
    parser.add_argument("--config", type=Path, default=Path("configs/bbox_stabilization.yaml"), help="bbox stabilization config YAML.")
    parser.add_argument("--cleaned-masks-dir", type=Path, help="Optional directory to write cleaned mask PNGs.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    recompute(args.tracks, args.output_tracks, args.config, args.cleaned_masks_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
