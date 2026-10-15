from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from sam2_annotator.mask_utils import save_binary_mask_png

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recompute_bboxes import recompute_bboxes, resolve_mask_path  # noqa: E402


def test_relative_mask_path_resolves_from_tracks_dir(tmp_path: Path) -> None:
    masks_dir = tmp_path / "masks"
    mask_path = masks_dir / "000000_obj1.png"
    save_binary_mask_png(np.ones((4, 4), dtype=np.uint8), mask_path)
    tracks_path = tmp_path / "tracks.jsonl"
    record = {"frame_idx": 0, "obj_id": 1, "mask_path": "masks/000000_obj1.png"}

    assert resolve_mask_path(record, tracks_path) == mask_path.resolve()


def test_recompute_preserves_input_sorts_and_writes_multi_object_labels(tmp_path: Path) -> None:
    masks_dir = tmp_path / "masks"
    save_binary_mask_png(_mask_with_box((20, 20), (2, 2, 5, 5)), masks_dir / "000000_obj1.png")
    save_binary_mask_png(_mask_with_box((20, 20), (10, 10, 13, 13)), masks_dir / "000000_obj2.png")
    save_binary_mask_png(np.zeros((20, 20), dtype=np.uint8), masks_dir / "000001_obj1.png")
    tracks_path = tmp_path / "tracks.jsonl"
    rows = [
        {"frame_idx": 1, "obj_id": 1, "mask_path": "masks/000001_obj1.png", "bbox_xyxy": [99, 99, 99, 99]},
        {"frame_idx": 0, "obj_id": 2, "mask_path": "masks/000000_obj2.png", "bbox_xyxy": [1, 1, 1, 1]},
        {"frame_idx": 0, "obj_id": 1, "mask_path": "masks/000000_obj1.png", "bbox_xyxy": [0, 0, 0, 0]},
    ]
    original_text = "".join(json.dumps(row) + "\n" for row in rows)
    tracks_path.write_text(original_text, encoding="utf-8")
    config_path = tmp_path / "bbox.yaml"
    config_path.write_text("min_component_area: 3\nminimum_center_jump_pixels: 50\n", encoding="utf-8")

    summary = recompute_bboxes(
        tracks_path=tracks_path,
        output_tracks=tmp_path / "tracks_stabilized.jsonl",
        output_labels=tmp_path / "labels",
        config_path=config_path,
        output_cleaned_masks=tmp_path / "cleaned_masks",
        overwrite=True,
    )

    assert tracks_path.read_text(encoding="utf-8") == original_text
    output_rows = [json.loads(line) for line in (tmp_path / "tracks_stabilized.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(row["frame_idx"], row["obj_id"]) for row in output_rows] == [(0, 1), (0, 2), (1, 1)]
    assert output_rows[0]["legacy_bbox_xyxy"] == [0, 0, 0, 0]
    assert output_rows[0]["raw_bbox_xyxy"] == [2, 2, 5, 5]
    assert output_rows[0]["bbox_xyxy"] == output_rows[0]["stable_bbox_xyxy"]
    assert output_rows[2]["bbox_status"] == "missing"
    assert output_rows[2]["label_status"] == "missing"
    assert (tmp_path / "cleaned_masks" / "000000_obj1.png").is_file()
    assert len((tmp_path / "labels" / "000000.txt").read_text(encoding="utf-8").splitlines()) == 2
    assert (tmp_path / "labels" / "000001.txt").read_text(encoding="utf-8") == ""
    assert summary["track_records"] == 3
    assert summary["stable_label_rows"] == 2


def test_recompute_does_not_fake_cleaned_mask_path_without_output_dir(tmp_path: Path) -> None:
    masks_dir = tmp_path / "masks"
    save_binary_mask_png(_mask_with_box((10, 10), (1, 1, 3, 3)), masks_dir / "000000_obj1.png")
    tracks_path = tmp_path / "tracks.jsonl"
    tracks_path.write_text(json.dumps({"frame_idx": 0, "obj_id": 1, "mask_path": "masks/000000_obj1.png"}) + "\n", encoding="utf-8")
    config_path = tmp_path / "bbox.yaml"
    config_path.write_text("min_component_area: 3\n", encoding="utf-8")

    recompute_bboxes(
        tracks_path=tracks_path,
        output_tracks=tmp_path / "out.jsonl",
        output_labels=tmp_path / "labels",
        config_path=config_path,
        overwrite=True,
    )

    row = json.loads((tmp_path / "out.jsonl").read_text(encoding="utf-8"))
    assert row["cleaned_mask_path"] is None


def _mask_with_box(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    mask[y1 : y2 + 1, x1 : x2 + 1] = 1
    return mask
