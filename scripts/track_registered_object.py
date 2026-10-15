#!/usr/bin/env python3
"""Track one registered object in one segment with an independent SAM2 state."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from contextlib import nullcontext
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.config_utils import load_yaml_config, nested_get
from sam2_annotator.frame_utils import frame_name, sorted_frame_paths
from sam2_annotator.mask_utils import bbox_area, bbox_center, bbox_to_yolo, mask_area, mask_to_bbox, save_binary_mask_png
from sam2_annotator.object_registry import get_object, load_registry
from sam2_annotator.tracking_io import load_json, load_segment_manifest, local_to_original_map, mask_filename, write_json

LOGGER = logging.getLogger("track_registered_object")


def resolve_path(value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _tensor_mask_to_numpy(mask_logits: Any, mask_threshold: float) -> np.ndarray:
    if hasattr(mask_logits, "detach"):
        mask_logits = mask_logits.detach().cpu().numpy()
    array = np.asarray(mask_logits)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"expected one 2D mask after squeeze, got shape {array.shape}")
    return array > mask_threshold


def _write_yolo_label(path: Path, bbox: tuple[int, int, int, int] | None, width: int, height: int, class_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if bbox is None:
        path.write_text("", encoding="utf-8")
        return
    cx, cy, bw, bh = bbox_to_yolo(bbox, width, height)
    path.write_text(f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n", encoding="utf-8")


def _extract_obj_mask(out_obj_ids: Iterable[Any], out_mask_logits: Any, obj_id: int, mask_threshold: float) -> np.ndarray | None:
    obj_ids = [int(value) for value in out_obj_ids]
    if obj_id not in obj_ids:
        return None
    return _tensor_mask_to_numpy(out_mask_logits[obj_ids.index(obj_id)], mask_threshold)


def _load_mask_prompt(mask_path_value: str, prompt_path: Path) -> np.ndarray:
    import cv2

    mask_path = Path(mask_path_value).expanduser()
    if not mask_path.is_absolute():
        mask_path = prompt_path.parent / mask_path
    mask_path = mask_path.resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(f"mask prompt file does not exist: {mask_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise OSError(f"failed to read mask prompt: {mask_path}")
    if int(np.count_nonzero(mask)) == 0:
        raise ValueError(f"mask prompt is empty: {mask_path}")
    return mask > 0


def _add_initial_prompt(predictor: Any, state: Any, prompt: dict[str, Any], prompt_path: Path, obj_id: int) -> Any:
    local_frame_idx = int(prompt.get("local_frame_idx", prompt["frame_idx"]))
    if int(prompt["obj_id"]) != obj_id:
        raise ValueError(f"prompt obj_id {prompt['obj_id']} does not match requested obj_id {obj_id}")
    prompt_type = prompt.get("prompt_type")
    if prompt_type == "box":
        box = np.asarray(prompt["box_xyxy"], dtype=np.float32)
        return predictor.add_new_points_or_box(inference_state=state, frame_idx=local_frame_idx, obj_id=obj_id, box=box)
    if prompt_type == "mask":
        if not hasattr(predictor, "add_new_mask"):
            raise AttributeError("SAM2 predictor does not expose add_new_mask")
        mask = _load_mask_prompt(str(prompt["mask_path"]), prompt_path)
        return predictor.add_new_mask(inference_state=state, frame_idx=local_frame_idx, obj_id=obj_id, mask=mask)
    raise ValueError(f"unsupported prompt_type: {prompt_type!r}")


def track_registered_object(
    config_path: Path,
    registry_path: Path,
    obj_id: int,
    segment_dir: Path,
    prompt_path: Path,
    output_dir: Path,
    max_frames: int | None,
) -> dict[str, Any]:
    import torch

    sam2_root = (Path.home() / "projects" / "sam2").resolve()
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2_video_predictor

    config_path = config_path.expanduser().resolve()
    config = load_yaml_config(config_path)
    model_config = str(nested_get(config, ("sam2", "model_config")))
    checkpoint = resolve_path(nested_get(config, ("sam2", "checkpoint")))
    device = str(nested_get(config, ("sam2", "device"), "cuda"))
    mask_threshold = float(nested_get(config, ("tracking", "mask_threshold"), 0.0))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint}")

    registry = load_registry(registry_path)
    obj = get_object(registry, obj_id)
    object_name = str(obj["name"])
    global_drone_id = str(obj["global_drone_id"])
    class_id = int(obj["class_id"])
    class_name = str(obj.get("class_name", global_drone_id))
    video_id = str(registry.get("video_id") or Path(str(registry["video"])).stem)
    segment_dir = segment_dir.expanduser().resolve()
    if not segment_dir.is_dir():
        raise FileNotFoundError(f"segment directory does not exist: {segment_dir}")
    manifest = load_segment_manifest(segment_dir)
    local_to_original = local_to_original_map(manifest)
    prompt = load_json(prompt_path)
    local_prompt_frame = int(prompt.get("local_frame_idx", prompt["frame_idx"]))
    if local_prompt_frame not in local_to_original:
        raise ValueError(f"prompt local frame {local_prompt_frame} is outside segment")
    frame_paths = sorted_frame_paths(segment_dir)
    if not frame_paths:
        raise FileNotFoundError(f"no segment frames found in {segment_dir}")
    width = int(prompt.get("frame_width") or manifest["frame_width"])
    height = int(prompt.get("frame_height") or manifest["frame_height"])

    output_dir = output_dir.expanduser().resolve()
    masks_dir = output_dir / "masks"
    labels_dir = output_dir / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prompt_path.expanduser().resolve(), output_dir / "prompt.json")
    tracks_path = output_dir / "tracks.jsonl"

    autocast_context = nullcontext()
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("device is cuda but torch.cuda.is_available() is false")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        autocast_context = torch.autocast(device_type="cuda", dtype=dtype)

    LOGGER.info("building independent SAM2 video predictor for obj_id=%s", obj_id)
    predictor = build_sam2_video_predictor(model_config, str(checkpoint), device=device, vos_optimized=False)
    processed = 0
    empty = 0
    end_local_frame: int | None = None
    started_at = time.monotonic()

    with torch.inference_mode(), autocast_context:
        state = predictor.init_state(video_path=str(segment_dir))
        _add_initial_prompt(predictor, state, prompt, prompt_path, obj_id)
        with tracks_path.open("w", encoding="utf-8") as tracks_file:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                state,
                start_frame_idx=local_prompt_frame,
                max_frame_num_to_track=max_frames,
                reverse=False,
            ):
                local_frame_idx = int(out_frame_idx)
                if local_frame_idx not in local_to_original:
                    continue
                original_frame_idx = local_to_original[local_frame_idx]
                mask = _extract_obj_mask(out_obj_ids, out_mask_logits, obj_id, mask_threshold)
                if mask is None:
                    LOGGER.warning("obj_id=%s missing on local frame %s", obj_id, local_frame_idx)
                    continue
                bbox = mask_to_bbox(mask)
                center = bbox_center(bbox)
                mask_path = masks_dir / mask_filename(original_frame_idx, obj_id)
                label_path = labels_dir / f"{original_frame_idx:06d}.txt"
                save_binary_mask_png(mask, mask_path)
                _write_yolo_label(label_path, bbox, width, height, class_id)
                record = {
                    "frame_idx": original_frame_idx,
                    "original_frame_idx": original_frame_idx,
                    "local_frame_idx": local_frame_idx,
                    "obj_id": obj_id,
                    "object_name": object_name,
                    "video_id": video_id,
                    "global_drone_id": global_drone_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "mask_path": str(mask_path),
                    "label_path": str(label_path),
                    "bbox_xyxy": list(bbox) if bbox is not None else None,
                    "mask_area": mask_area(mask),
                    "bbox_area": bbox_area(bbox),
                    "center_xy": list(center) if center is not None else None,
                    "empty_mask": bbox is None,
                    "frame_width": width,
                    "frame_height": height,
                    "segment_start_frame": int(manifest["start_frame"]),
                    "segment_end_frame": int(manifest["end_frame"]),
                }
                tracks_file.write(json.dumps(record) + "\n")
                tracks_file.flush()
                processed += 1
                empty += 1 if bbox is None else 0
                end_local_frame = local_frame_idx

    summary = {
        "obj_id": obj_id,
        "object_name": object_name,
        "video_id": video_id,
        "global_drone_id": global_drone_id,
        "class_id": class_id,
        "class_name": class_name,
        "processed_frames": processed,
        "empty_masks": empty,
        "segment_start_frame": int(manifest["start_frame"]),
        "segment_end_frame": int(manifest["end_frame"]),
        "local_start_frame": local_prompt_frame,
        "local_end_frame": end_local_frame,
        "original_start_frame": local_to_original[local_prompt_frame],
        "original_end_frame": local_to_original[end_local_frame] if end_local_frame is not None else None,
        "prompt_type": prompt.get("prompt_type"),
        "output_tracks": str(tracks_path),
        "elapsed_seconds": time.monotonic() - started_at,
        "sam2_api": {
            "add_new_points_or_box": str(inspect.signature(predictor.add_new_points_or_box)),
            "add_new_mask": str(inspect.signature(predictor.add_new_mask)) if hasattr(predictor, "add_new_mask") else None,
            "propagate_in_video": str(inspect.signature(predictor.propagate_in_video)),
        },
    }
    write_json(output_dir / "run_summary.json", summary)
    LOGGER.info("wrote tracks: %s", tracks_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track exactly one registered object in one segment with a fresh independent SAM2 inference state.")
    parser.add_argument("--config", required=True, type=Path, help="YAML config containing SAM2 model, checkpoint and device.")
    parser.add_argument("--registry", required=True, type=Path, help="Object registry JSON.")
    parser.add_argument("--obj-id", required=True, type=int, help="Object id to track. One run tracks one object only.")
    parser.add_argument("--segment-dir", required=True, type=Path, help="Segment directory with segment_manifest.json and local 000000.jpg frames.")
    parser.add_argument("--prompt", required=True, type=Path, help="Local box or mask prompt JSON for this segment.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for this object's masks, labels, tracks and summary.")
    parser.add_argument("--max-frames", type=int, help="Maximum local frames for SAM2 propagation.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    track_registered_object(args.config, args.registry, args.obj_id, args.segment_dir, args.prompt, args.output_dir, args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
