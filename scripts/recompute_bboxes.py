#!/usr/bin/env python3
"""Recompute raw, cleaned and stable bboxes from saved masks without SAM2."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.bbox_stabilizer import BBoxStabilizer, clean_mask_components, parameters_from_mapping
from sam2_annotator.config_utils import load_yaml_config
from sam2_annotator.frame_utils import frame_name
from sam2_annotator.mask_utils import bbox_area, bbox_center, bbox_to_yolo, mask_area, mask_to_bbox, save_binary_mask_png

LOGGER = logging.getLogger("recompute_bboxes")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_tracks(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"tracks JSONL does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "frame_idx" not in record:
                raise ValueError(f"tracks line {line_no} missing frame_idx")
            record.setdefault("obj_id", 1)
            records.append(record)
    if not records:
        raise ValueError(f"tracks JSONL has no records: {path}")
    return sorted(records, key=lambda item: (int(item["frame_idx"]), int(item.get("obj_id", 1))))


def resolve_mask_path(record: dict[str, Any], tracks_path: Path, project_root: Path = PROJECT_ROOT) -> Path:
    value = record.get("mask_path") or record.get("raw_mask_path")
    if not value:
        raise ValueError(f"track record frame={record.get('frame_idx')} obj={record.get('obj_id', 1)} has no mask_path")
    raw_path = Path(str(value)).expanduser()
    candidates = [raw_path] if raw_path.is_absolute() else [tracks_path.parent / raw_path, project_root / raw_path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"mask_path not found for frame={record.get('frame_idx')} obj={record.get('obj_id', 1)}; checked: {checked}")


def load_run_summary(tracks_path: Path) -> dict[str, Any]:
    summary_path = tracks_path.parent / "run_summary.json"
    if not summary_path.is_file():
        return {}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def dimensions_from_summary(summary: dict[str, Any], frame_idx: int) -> tuple[int, int] | None:
    import cv2

    if "frame_width" in summary and "frame_height" in summary:
        return _positive_size(summary["frame_width"], summary["frame_height"], "run_summary.json")
    frames_dir_value = summary.get("frames_dir")
    if not frames_dir_value:
        return None
    frames_dir = Path(str(frames_dir_value)).expanduser()
    if not frames_dir.is_absolute():
        frames_dir = PROJECT_ROOT / frames_dir
    candidates = [frames_dir / frame_name(frame_idx)]
    candidates.extend(sorted(frames_dir.glob("*.jpg"))[:1])
    for frame_path in candidates:
        if not frame_path.is_file():
            continue
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise OSError(f"failed to read frame for image size: {frame_path}")
        height, width = frame.shape[:2]
        return int(width), int(height)
    return None


def infer_image_size(
    record: dict[str, Any],
    summary_size: tuple[int, int] | None,
    mask_shape: tuple[int, int],
    cli_width: int | None,
    cli_height: int | None,
) -> tuple[int, int]:
    if "frame_width" in record and "frame_height" in record:
        return _positive_size(record["frame_width"], record["frame_height"], "track record")
    if summary_size is not None:
        return summary_size
    height, width = mask_shape
    if width > 0 and height > 0:
        return int(width), int(height)
    if cli_width is not None and cli_height is not None:
        return _positive_size(cli_width, cli_height, "CLI image size")
    raise ValueError("could not infer image dimensions")


def recompute_bboxes(
    tracks_path: Path,
    output_tracks: Path,
    output_labels: Path,
    config_path: Path,
    output_cleaned_masks: Path | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    import cv2

    tracks_path = tracks_path.expanduser().resolve()
    output_tracks = output_tracks.expanduser().resolve()
    output_labels = output_labels.expanduser().resolve()
    if tracks_path == output_tracks:
        raise ValueError("output-tracks must not be the same file as input tracks")
    if output_tracks.exists() and not overwrite:
        raise FileExistsError(f"output tracks already exists: {output_tracks}; pass --overwrite to replace it")
    if output_labels.exists() and any(output_labels.iterdir()) and not overwrite:
        raise FileExistsError(f"output labels directory is not empty: {output_labels}; pass --overwrite to replace its txt files")

    config = load_yaml_config(config_path.expanduser().resolve())
    params = parameters_from_mapping(config)
    records = load_tracks(tracks_path)
    summary = load_run_summary(tracks_path)
    summary_size_cache: dict[int, tuple[int, int] | None] = {}
    stabilizers: dict[int, BBoxStabilizer] = {}
    output_rows: list[dict[str, Any]] = []
    label_rows: dict[int, list[str]] = {}
    status_counts: dict[str, int] = {}

    cleaned_dir = output_cleaned_masks.expanduser().resolve() if output_cleaned_masks is not None else None
    if cleaned_dir is not None:
        if cleaned_dir.exists() and any(cleaned_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"cleaned mask directory is not empty: {cleaned_dir}; pass --overwrite to replace files")
        cleaned_dir.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    output_tracks.parent.mkdir(parents=True, exist_ok=True)

    for record in _progress(records, "recomputing bboxes"):
        frame_idx = int(record["frame_idx"])
        obj_id = int(record.get("obj_id", 1))
        mask_path = resolve_mask_path(record, tracks_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise OSError(f"failed to read mask: {mask_path}")
        if frame_idx not in summary_size_cache:
            summary_size_cache[frame_idx] = dimensions_from_summary(summary, frame_idx)
        width, height = infer_image_size(record, summary_size_cache[frame_idx], mask.shape[:2], image_width, image_height)

        stabilizer = stabilizers.setdefault(obj_id, BBoxStabilizer(params))
        raw_bbox = mask_to_bbox(mask)
        raw_area = mask_area(mask)
        cleaned_mask, component_info = clean_mask_components(
            mask,
            previous_center=stabilizer.previous_center,
            recent_area=stabilizer.recent_area,
            min_component_area=params.min_component_area,
        )
        cleaned_bbox = mask_to_bbox(cleaned_mask)
        cleaned_area = mask_area(cleaned_mask)
        stability = stabilizer.update(raw_bbox, (width, height), candidate_bbox_xyxy=cleaned_bbox)
        stability_data = stability.to_dict()

        cleaned_mask_path: Path | None = None
        if cleaned_dir is not None:
            cleaned_mask_path = cleaned_dir / f"{frame_idx:06d}_obj{obj_id}.png"
            save_binary_mask_png(cleaned_mask, cleaned_mask_path)

        updated = dict(record)
        if "bbox_xyxy" in record:
            updated["legacy_bbox_xyxy"] = record["bbox_xyxy"]
        stable_bbox = stability.stable_bbox_xyxy
        center = bbox_center(stable_bbox)
        label_status = label_status_for_bbox(stability.bbox_status)
        updated.update(
            {
                "raw_mask_path": str(mask_path),
                "cleaned_mask_path": str(cleaned_mask_path) if cleaned_mask_path is not None else None,
                "raw_mask_area": raw_area,
                "cleaned_mask_area": cleaned_area,
                "mask_area": cleaned_area,
                "raw_bbox_xyxy": _bbox_to_list(raw_bbox),
                "cleaned_bbox_xyxy": _bbox_to_list(cleaned_bbox),
                "stable_bbox_xyxy": stability_data["stable_bbox_xyxy"],
                "bbox_xyxy": stability_data["stable_bbox_xyxy"],
                "bbox_area": bbox_area(stable_bbox),
                "raw_bbox_area": bbox_area(raw_bbox),
                "bbox_status": stability.bbox_status,
                "bbox_stabilized": stability.bbox_stabilized,
                "bbox_stability_reasons": stability.bbox_stability_reasons,
                "component_count": component_info.component_count,
                "valid_component_count": component_info.valid_component_count,
                "selected_component_area": component_info.selected_component_area,
                "removed_component_count": component_info.removed_component_count,
                "selected_component_center": list(component_info.selected_component_center)
                if component_info.selected_component_center is not None
                else None,
                "raw_center_xy": stability_data["raw_center_xy"],
                "stable_center_xy": stability_data["stable_center_xy"],
                "center_xy": list(center) if center is not None else None,
                "raw_width": stability.raw_width,
                "raw_height": stability.raw_height,
                "stable_width": stability.stable_width,
                "stable_height": stability.stable_height,
                "frame_width": width,
                "frame_height": height,
                "label_status": label_status,
                "empty_mask": cleaned_bbox is None,
            }
        )
        output_rows.append(updated)
        status_counts[stability.bbox_status] = status_counts.get(stability.bbox_status, 0) + 1
        label_rows.setdefault(frame_idx, [])
        if stability.bbox_status != "missing" and stable_bbox is not None:
            cx, cy, bw, bh = bbox_to_yolo(stable_bbox, width, height)
            label_rows[frame_idx].append(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")

    _write_tracks_atomic(output_tracks, output_rows)
    _write_labels(output_labels, label_rows)
    stabilized_labels = sum(len(lines) for lines in label_rows.values())
    summary_data = {
        "track_records": len(output_rows),
        "label_files": len(label_rows),
        "stable_label_rows": stabilized_labels,
        "status_counts": status_counts,
        "output_tracks": str(output_tracks),
        "output_labels": str(output_labels),
        "output_cleaned_masks": str(cleaned_dir) if cleaned_dir is not None else None,
    }
    LOGGER.info("processed %s track records; wrote %s label rows", len(output_rows), stabilized_labels)
    return summary_data


def label_status_for_bbox(status: str) -> str:
    if status in {"initialized", "stable", "smoothed"}:
        return "auto_candidate"
    if status == "unstable":
        return "need_review"
    return "missing"


def _write_tracks_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_labels(output_dir: Path, label_rows: dict[int, list[str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in sorted(label_rows):
        text = "".join(f"{line}\n" for line in label_rows[frame_idx])
        path = output_dir / f"{frame_idx:06d}.txt"
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(output_dir))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(text)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise


def _progress(records: list[dict[str, Any]], description: str) -> Iterable[dict[str, Any]]:
    try:
        from tqdm import tqdm
    except ImportError:
        LOGGER.info("%s: %s records", description, len(records))
        return records
    return tqdm(records, desc=description, unit="record")


def _positive_size(width_value: Any, height_value: Any, source: str) -> tuple[int, int]:
    width = int(width_value)
    height = int(height_value)
    if width <= 0 or height <= 0:
        raise ValueError(f"{source} image dimensions must be positive, got {width}x{height}")
    return width, height


def _bbox_to_list(bbox: tuple[int, int, int, int] | None) -> list[int] | None:
    return list(bbox) if bbox is not None else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute raw, cleaned and stable bbox fields from existing mask PNGs without rerunning SAM2.")
    parser.add_argument("--tracks", required=True, type=Path, help="Input tracks.jsonl containing mask_path fields.")
    parser.add_argument("--output-tracks", required=True, type=Path, help="Output JSONL path for stabilized tracks. Must differ from --tracks.")
    parser.add_argument("--output-labels", required=True, type=Path, help="Directory for regenerated YOLO txt labels.")
    parser.add_argument("--config", type=Path, default=Path("configs/bbox_stabilization.yaml"), help="bbox stabilization YAML config. Default: configs/bbox_stabilization.yaml.")
    parser.add_argument("--output-cleaned-masks", type=Path, help="Optional directory for cleaned binary mask PNGs.")
    parser.add_argument("--image-width", type=int, help="Fallback image width if not present in tracks, run_summary, or mask.")
    parser.add_argument("--image-height", type=int, help="Fallback image height if not present in tracks, run_summary, or mask.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing output tracks and generated files in output directories.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    if (args.image_width is None) != (args.image_height is None):
        raise ValueError("--image-width and --image-height must be provided together")
    recompute_bboxes(
        tracks_path=args.tracks,
        output_tracks=args.output_tracks,
        output_labels=args.output_labels,
        config_path=args.config,
        output_cleaned_masks=args.output_cleaned_masks,
        image_width=args.image_width,
        image_height=args.image_height,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
