#!/usr/bin/env python3
"""Propagate a single box prompt through extracted frames with SAM2 video predictor."""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.config_utils import load_yaml_config, nested_get
from sam2_annotator.frame_utils import frame_name, sorted_frame_paths
from sam2_annotator.mask_utils import bbox_area, bbox_center, bbox_to_yolo, mask_area, mask_to_bbox, save_binary_mask_png

LOGGER = logging.getLogger("track_box")


def resolve_path(value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def load_prompt(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"prompt JSON does not exist: {path}")
    prompt = json.loads(path.read_text(encoding="utf-8"))
    required = ("frame_idx", "obj_id", "prompt_type", "box_xyxy", "frame_width", "frame_height")
    missing = [key for key in required if key not in prompt]
    if missing:
        raise ValueError(f"prompt missing required keys: {missing}")
    if prompt["prompt_type"] != "box":
        raise ValueError(f"only box prompts are supported, got {prompt['prompt_type']!r}")
    if int(prompt["obj_id"]) != 1:
        raise ValueError("first milestone supports only obj_id=1")
    box = prompt["box_xyxy"]
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("prompt box_xyxy must be a list of four numbers")
    return prompt


def _tensor_mask_to_numpy(mask_logits: Any, mask_threshold: float) -> np.ndarray:
    if hasattr(mask_logits, "detach"):
        mask_logits = mask_logits.detach().cpu().numpy()
    array = np.asarray(mask_logits)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"expected one 2D mask after squeeze, got shape {array.shape}")
    return array > mask_threshold


def _write_empty_label(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _write_yolo_label(path: Path, bbox: tuple[int, int, int, int] | None, width: int, height: int) -> None:
    if bbox is None:
        _write_empty_label(path)
        return
    cx, cy, bw, bh = bbox_to_yolo(bbox, width, height)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n", encoding="utf-8")


def _load_confirmed_keys(path: Path | None) -> set[tuple[int, int]]:
    if path is None:
        return set()
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"confirmed frames file does not exist: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("confirmed frames JSON must be a list")
        return {(int(item["frame_idx"]), int(item.get("obj_id", 1))) if isinstance(item, dict) else (int(item), 1) for item in data}
    keys: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("label_status") == "confirmed" or record.get("status") == "confirmed":
                keys.add((int(record["frame_idx"]), int(record.get("obj_id", 1))))
            elif "frame_idx" not in record:
                raise ValueError(f"confirmed frames line {line_no} missing frame_idx")
    return keys


def run_tracking(config_path: Path, prompt_override: Path | None, max_frames_override: int | None) -> dict[str, Any]:
    import torch

    sam2_root = (Path.home() / "projects" / "sam2").resolve()
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2_video_predictor

    config_path = config_path.expanduser().resolve()
    config = load_yaml_config(config_path)
    config_base = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent

    model_config = str(nested_get(config, ("sam2", "model_config")))
    checkpoint = resolve_path(nested_get(config, ("sam2", "checkpoint")))
    device = str(nested_get(config, ("sam2", "device"), "cuda"))
    frames_dir = resolve_path(nested_get(config, ("input", "frames_dir")), config_base)
    prompt_path = prompt_override or resolve_path(nested_get(config, ("input", "prompt_file")), config_base)
    output_dir = resolve_path(nested_get(config, ("output", "output_dir")), config_base)
    final_labels_value = nested_get(config, ("output", "final_labels_dir"), None)
    final_labels_dir = resolve_path(final_labels_value, config_base) if final_labels_value is not None else None
    confirmed_frames_value = nested_get(config, ("input", "confirmed_frames_file"), None)
    confirmed_frames_path = resolve_path(confirmed_frames_value, config_base) if confirmed_frames_value is not None else None
    mask_threshold = float(nested_get(config, ("tracking", "mask_threshold"), 0.0))
    start_frame = int(nested_get(config, ("tracking", "start_frame"), 0))
    obj_id = int(nested_get(config, ("tracking", "obj_id"), 1))
    max_frames = max_frames_override if max_frames_override is not None else nested_get(config, ("tracking", "max_frames"), None)
    max_frames = int(max_frames) if max_frames is not None else None
    if obj_id != 1:
        raise ValueError("first milestone supports only obj_id=1")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint}")
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames directory does not exist: {frames_dir}")

    prompt = load_prompt(Path(prompt_path))
    prompt_frame = int(prompt["frame_idx"])
    if start_frame != prompt_frame:
        LOGGER.warning("using prompt frame %s as start frame instead of config start_frame %s", prompt_frame, start_frame)
        start_frame = prompt_frame
    box = np.asarray(prompt["box_xyxy"], dtype=np.float32)
    if box.shape != (4,):
        raise ValueError(f"invalid prompt box shape: {box.shape}")

    frame_paths = sorted_frame_paths(frames_dir)
    if not frame_paths:
        raise FileNotFoundError(f"no JPEG frames found in {frames_dir}")
    width = int(prompt["frame_width"])
    height = int(prompt["frame_height"])

    masks_dir = output_dir / "masks"
    labels_dir = output_dir / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if final_labels_dir is not None:
        final_labels_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "tracks.jsonl"
    confirmed_keys = _load_confirmed_keys(confirmed_frames_path)

    autocast_context = nullcontext()
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("device is cuda but torch.cuda.is_available() is false")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if dtype is torch.float16:
            LOGGER.warning("CUDA bfloat16 is not supported; falling back to float16 autocast")
        autocast_context = torch.autocast(device_type="cuda", dtype=dtype)

    LOGGER.info("building SAM2 video predictor")
    predictor = build_sam2_video_predictor(model_config, str(checkpoint), device=device, vos_optimized=False)

    processed = 0
    non_empty = 0
    empty = 0
    end_frame: int | None = None
    started_at = time.monotonic()

    try:
        with torch.inference_mode(), autocast_context:
            state = predictor.init_state(video_path=str(frames_dir))
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=start_frame,
                obj_id=obj_id,
                box=box,
            )
            with tracks_path.open("a", encoding="utf-8") as tracks_file:
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                    state,
                    start_frame_idx=start_frame,
                    max_frame_num_to_track=max_frames,
                    reverse=False,
                ):
                    if obj_id not in [int(v) for v in out_obj_ids]:
                        LOGGER.warning("obj_id=%s missing on frame %s", obj_id, out_frame_idx)
                        continue
                    obj_pos = [int(v) for v in out_obj_ids].index(obj_id)
                    mask = _tensor_mask_to_numpy(out_mask_logits[obj_pos], mask_threshold)
                    bbox = mask_to_bbox(mask)
                    area = mask_area(mask)
                    is_empty = bbox is None
                    mask_path = masks_dir / f"{frame_name(int(out_frame_idx), suffix='')}_obj{obj_id}.png"
                    label_path = labels_dir / f"{frame_name(int(out_frame_idx), suffix='')}.txt"
                    final_label_path = final_labels_dir / f"{frame_name(int(out_frame_idx), suffix='')}.txt" if final_labels_dir is not None else None
                    is_confirmed = (int(out_frame_idx), obj_id) in confirmed_keys
                    label_status = "confirmed" if is_confirmed else "working_unconfirmed"

                    save_binary_mask_png(mask, mask_path)
                    _write_yolo_label(label_path, bbox, width, height)
                    if final_label_path is not None and is_confirmed:
                        _write_yolo_label(final_label_path, bbox, width, height)

                    center = bbox_center(bbox)
                    record = {
                        "frame_idx": int(out_frame_idx),
                        "obj_id": obj_id,
                        "mask_path": str(mask_path),
                        "label_path": str(label_path),
                        "final_label_path": str(final_label_path) if final_label_path is not None and is_confirmed else None,
                        "label_status": label_status,
                        "bbox_xyxy": list(bbox) if bbox is not None else None,
                        "mask_area": area,
                        "bbox_area": bbox_area(bbox),
                        "center_xy": list(center) if center is not None else None,
                        "empty_mask": is_empty,
                    }
                    tracks_file.write(json.dumps(record) + "\n")
                    tracks_file.flush()

                    processed += 1
                    non_empty += 0 if is_empty else 1
                    empty += 1 if is_empty else 0
                    end_frame = int(out_frame_idx)
    except KeyboardInterrupt:
        LOGGER.warning("interrupted; writing summary for completed frames")

    elapsed = time.monotonic() - started_at
    summary = {
        "processed_frames": processed,
        "non_empty_masks": non_empty,
        "empty_masks": empty,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "elapsed_seconds": elapsed,
        "model_config": model_config,
        "checkpoint": str(checkpoint),
        "frames_dir": str(frames_dir),
        "prompt_file": str(prompt_path),
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("wrote summary: %s", summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track one prompted object forward through extracted frames using SAM2.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="YAML config path. Default: configs/default.yaml.")
    parser.add_argument("--prompt", type=Path, help="Override prompt JSON path from config.")
    parser.add_argument("--max-frames", type=int, help="Override maximum propagated frames.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    run_tracking(args.config, args.prompt, args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
