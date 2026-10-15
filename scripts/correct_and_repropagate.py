#!/usr/bin/env python3
"""Apply one same-obj correction prompt and repropagate SAM2 forward from that frame."""

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
from sam2_annotator.correction_utils import atomic_write_json, load_json, validate_correction_obj_id
from sam2_annotator.frame_utils import frame_name, sorted_frame_paths
from sam2_annotator.mask_utils import bbox_area, bbox_center, bbox_to_yolo, mask_area, mask_to_bbox, save_binary_mask_png

LOGGER = logging.getLogger("correct_and_repropagate")


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


def _write_yolo_label(path: Path, bbox: tuple[int, int, int, int] | None, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if bbox is None:
        path.write_text("", encoding="utf-8")
        return
    cx, cy, bw, bh = bbox_to_yolo(bbox, width, height)
    path.write_text(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n", encoding="utf-8")


def _extract_obj_mask(out_obj_ids: Iterable[Any], out_mask_logits: Any, obj_id: int, mask_threshold: float) -> np.ndarray | None:
    obj_ids = [int(value) for value in out_obj_ids]
    if obj_id not in obj_ids:
        return None
    obj_pos = obj_ids.index(obj_id)
    return _tensor_mask_to_numpy(out_mask_logits[obj_pos], mask_threshold)


def _save_track_record(
    frame_idx: int,
    obj_id: int,
    mask: np.ndarray,
    output_dir: Path,
    tracks_file: Any,
    width: int,
    height: int,
    is_correction_frame: bool,
    correction_applied: bool,
    correction_source: str,
    original_frame_idx: int | None,
    object_name: str | None,
) -> dict[str, Any]:
    masks_dir = output_dir / "masks"
    labels_dir = output_dir / "labels"
    bbox = mask_to_bbox(mask)
    center = bbox_center(bbox)
    mask_path = masks_dir / f"{frame_name(frame_idx, suffix='')}_obj{obj_id}.png"
    label_path = labels_dir / f"{frame_name(frame_idx, suffix='')}.txt"
    save_binary_mask_png(mask, mask_path)
    _write_yolo_label(label_path, bbox, width, height)
    record = {
        "frame_idx": frame_idx,
        "obj_id": obj_id,
        "object_name": object_name or f"obj{obj_id}",
        "mask_path": str(mask_path),
        "label_path": str(label_path),
        "bbox_xyxy": list(bbox) if bbox is not None else None,
        "mask_area": mask_area(mask),
        "bbox_area": bbox_area(bbox),
        "center_xy": list(center) if center is not None else None,
        "empty_mask": bbox is None,
        "is_correction_frame": is_correction_frame,
        "correction_applied": correction_applied,
        "correction_source": correction_source,
        "original_frame_idx": original_frame_idx,
    }
    tracks_file.write(json.dumps(record) + "\n")
    tracks_file.flush()
    return record


def _call_propagate(predictor: Any, state: Any, start_frame_idx: int, max_frame_num_to_track: int | None) -> Iterable[tuple[Any, Any, Any]]:
    signature = inspect.signature(predictor.propagate_in_video)
    kwargs: dict[str, Any] = {}
    if "start_frame_idx" in signature.parameters:
        kwargs["start_frame_idx"] = start_frame_idx
    if "max_frame_num_to_track" in signature.parameters and max_frame_num_to_track is not None:
        kwargs["max_frame_num_to_track"] = max_frame_num_to_track
    if "reverse" in signature.parameters:
        kwargs["reverse"] = False
    return predictor.propagate_in_video(state, **kwargs)


def _add_box_prompt(predictor: Any, state: Any, frame_idx: int, obj_id: int, box: np.ndarray, clear_old_points: bool) -> Any:
    signature = inspect.signature(predictor.add_new_points_or_box)
    kwargs: dict[str, Any] = {
        "inference_state": state,
        "frame_idx": frame_idx,
        "obj_id": obj_id,
        "box": box,
    }
    if "clear_old_points" in signature.parameters:
        kwargs["clear_old_points"] = clear_old_points
    return predictor.add_new_points_or_box(**kwargs)


def _max_frames_for_range(start_frame: int, end_frame: int | None) -> int | None:
    if end_frame is None:
        return None
    return max(1, end_frame - start_frame + 1)


def correct_and_repropagate(
    config_path: Path,
    initial_prompt_path: Path,
    correction_prompt_path: Path,
    baseline_tracks: Path | None,
    output_dir: Path,
    end_frame: int | None,
    max_frames_after_correction: int | None,
) -> dict[str, Any]:
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
    mask_threshold = float(nested_get(config, ("tracking", "mask_threshold"), 0.0))

    initial_prompt = load_json(initial_prompt_path)
    correction_prompt = load_json(correction_prompt_path)
    obj_id = validate_correction_obj_id(initial_prompt, correction_prompt)
    if obj_id not in {1, 2, 3}:
        raise ValueError("current project supports only obj_id=1,2,3")
    if initial_prompt.get("prompt_type") != "box" or correction_prompt.get("prompt_type") != "box":
        raise ValueError("only box prompts are supported in this milestone")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint}")
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames directory does not exist: {frames_dir}")

    frame_paths = sorted_frame_paths(frames_dir)
    if not frame_paths:
        raise FileNotFoundError(f"no JPEG frames found in {frames_dir}")
    max_available_frame = int(frame_paths[-1].stem)
    initial_frame = int(initial_prompt["frame_idx"])
    correction_frame = int(correction_prompt["frame_idx"])
    if correction_frame < initial_frame:
        raise ValueError("correction frame must be at or after the initial prompt frame")
    if correction_frame > max_available_frame:
        raise ValueError(f"correction frame {correction_frame} is outside available frames ending at {max_available_frame}")
    if end_frame is not None and end_frame < correction_frame:
        raise ValueError("end-frame must be at or after correction frame")
    corrected_end_frame = min(max_available_frame, end_frame if end_frame is not None else max_available_frame)
    if max_frames_after_correction is not None:
        if max_frames_after_correction < 0:
            raise ValueError("max-frames-after-correction must be non-negative")
        corrected_end_frame = min(corrected_end_frame, correction_frame + max_frames_after_correction)

    width = int(correction_prompt.get("frame_width") or initial_prompt["frame_width"])
    height = int(correction_prompt.get("frame_height") or initial_prompt["frame_height"])
    initial_box = np.asarray(initial_prompt["box_xyxy"], dtype=np.float32)
    correction_box = np.asarray(correction_prompt["box_xyxy"], dtype=np.float32)
    object_name = str(correction_prompt.get("object_name") or initial_prompt.get("object_name") or f"obj{obj_id}")

    output_dir = output_dir.expanduser().resolve()
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "tracks_corrected.jsonl"
    shutil.copyfile(correction_prompt_path.expanduser().resolve(), output_dir / "correction_prompt.json")

    autocast_context = nullcontext()
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("device is cuda but torch.cuda.is_available() is false")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        autocast_context = torch.autocast(device_type="cuda", dtype=dtype)

    started_at = time.monotonic()
    written_keys: set[tuple[int, int]] = set()
    corrected_count = 0
    original_offset = None
    if correction_prompt.get("original_frame_idx") is not None:
        original_offset = int(correction_prompt["original_frame_idx"]) - correction_frame

    LOGGER.info("building SAM2 video predictor")
    predictor = build_sam2_video_predictor(model_config, str(checkpoint), device=device, vos_optimized=False)

    with torch.inference_mode(), autocast_context:
        state = predictor.init_state(video_path=str(frames_dir))
        _add_box_prompt(predictor, state, initial_frame, obj_id, initial_box, clear_old_points=True)

        if correction_frame > initial_frame:
            warmup_max = _max_frames_for_range(initial_frame, correction_frame)
            for out_frame_idx, _out_obj_ids, _out_mask_logits in _call_propagate(predictor, state, initial_frame, warmup_max):
                if int(out_frame_idx) >= correction_frame:
                    break

        correction_result = _add_box_prompt(
            predictor,
            state,
            correction_frame,
            obj_id,
            correction_box,
            clear_old_points=bool(correction_prompt.get("replace_prompt_on_frame", True)),
        )

        with tracks_path.open("w", encoding="utf-8") as tracks_file:
            if correction_result is not None and len(correction_result) == 3:
                out_frame_idx, out_obj_ids, out_mask_logits = correction_result
                if int(out_frame_idx) == correction_frame:
                    mask = _extract_obj_mask(out_obj_ids, out_mask_logits, obj_id, mask_threshold)
                    if mask is not None:
                        original_frame_idx = original_offset + correction_frame if original_offset is not None else None
                        _save_track_record(
                            correction_frame,
                            obj_id,
                            mask,
                            output_dir,
                            tracks_file,
                            width,
                            height,
                            is_correction_frame=True,
                            correction_applied=True,
                            correction_source=str(correction_prompt_path),
                            original_frame_idx=original_frame_idx,
                            object_name=object_name,
                        )
                        written_keys.add((correction_frame, obj_id))
                        corrected_count += 1

            propagate_max = _max_frames_for_range(correction_frame, corrected_end_frame)
            for out_frame_idx, out_obj_ids, out_mask_logits in _call_propagate(predictor, state, correction_frame, propagate_max):
                frame_idx = int(out_frame_idx)
                if frame_idx < correction_frame:
                    continue
                if frame_idx > corrected_end_frame:
                    break
                if (frame_idx, obj_id) in written_keys:
                    continue
                mask = _extract_obj_mask(out_obj_ids, out_mask_logits, obj_id, mask_threshold)
                if mask is None:
                    LOGGER.warning("obj_id=%s missing on frame %s", obj_id, frame_idx)
                    continue
                original_frame_idx = original_offset + frame_idx if original_offset is not None else None
                _save_track_record(
                    frame_idx,
                    obj_id,
                    mask,
                    output_dir,
                    tracks_file,
                    width,
                    height,
                    is_correction_frame=frame_idx == correction_frame,
                    correction_applied=frame_idx >= correction_frame,
                    correction_source=str(correction_prompt_path),
                    original_frame_idx=original_frame_idx,
                    object_name=object_name,
                )
                written_keys.add((frame_idx, obj_id))
                corrected_count += 1

    summary = {
        "obj_id": obj_id,
        "object_name": object_name,
        "initial_prompt_frame": initial_frame,
        "correction_frame": correction_frame,
        "corrected_end_frame": corrected_end_frame,
        "corrected_frame_count": corrected_count,
        "correction_type": correction_prompt.get("prompt_type"),
        "correction_reason": correction_prompt.get("correction_reason"),
        "baseline_tracks": str(baseline_tracks.expanduser().resolve()) if baseline_tracks is not None else None,
        "output_tracks": str(tracks_path),
        "elapsed_seconds": time.monotonic() - started_at,
        "sam2_api": {
            "add_new_points_or_box": str(inspect.signature(predictor.add_new_points_or_box)),
            "add_new_mask": str(inspect.signature(predictor.add_new_mask)) if hasattr(predictor, "add_new_mask") else None,
            "propagate_in_video": str(inspect.signature(predictor.propagate_in_video)),
            "reset_state": str(inspect.signature(predictor.reset_state)) if hasattr(predictor, "reset_state") else None,
            "clear_all_prompts_in_frame": str(inspect.signature(predictor.clear_all_prompts_in_frame))
            if hasattr(predictor, "clear_all_prompts_in_frame")
            else None,
        },
    }
    atomic_write_json(output_dir / "correction_summary.json", summary)
    LOGGER.info("wrote corrected tracks: %s", tracks_path)
    LOGGER.info("wrote correction summary: %s", output_dir / "correction_summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay an initial SAM2 prompt, apply a same-obj correction prompt, and repropagate forward.")
    parser.add_argument("--config", required=True, type=Path, help="YAML config containing SAM2 model, checkpoint, device and frames_dir.")
    parser.add_argument("--initial-prompt", required=True, type=Path, help="Original box prompt JSON.")
    parser.add_argument("--correction-prompt", required=True, type=Path, help="Correction prompt JSON created by create_correction_prompt.py.")
    parser.add_argument("--baseline-tracks", type=Path, help="Optional baseline tracks.jsonl path for summary metadata.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Independent output directory for corrected masks, labels and tracks.")
    parser.add_argument("--end-frame", type=int, help="Inclusive final frame to write after correction.")
    parser.add_argument("--max-frames-after-correction", type=int, help="Maximum number of frames after the correction frame; 0 writes only the correction frame.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    correct_and_repropagate(
        args.config,
        args.initial_prompt,
        args.correction_prompt,
        args.baseline_tracks,
        args.output_dir,
        args.end_frame,
        args.max_frames_after_correction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
