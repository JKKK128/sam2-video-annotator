from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from sam2_annotator.mask_utils import bbox_area, bbox_center, bbox_to_yolo, mask_area, mask_to_bbox, save_binary_mask_png


def test_mask_to_bbox_uses_inclusive_xyxy() -> None:
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1

    bbox = mask_to_bbox(mask)

    assert bbox == (3, 2, 6, 4)
    assert mask_area(mask) == 12
    assert bbox_area(bbox) == 12
    assert bbox_center(bbox) == (4.5, 3.0)


def test_empty_mask() -> None:
    mask = np.zeros((4, 5), dtype=bool)

    assert mask_to_bbox(mask) is None
    assert mask_area(mask) == 0
    assert bbox_area(None) == 0
    assert bbox_center(None) is None


def test_bbox_to_yolo_inclusive_coordinates() -> None:
    yolo = bbox_to_yolo((2, 1, 5, 4), image_width=10, image_height=8)

    assert yolo == pytest.approx((0.4, 0.375, 0.4, 0.5))


def test_bbox_to_yolo_rejects_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="outside"):
        bbox_to_yolo((0, 0, 10, 3), image_width=10, image_height=8)


def test_save_binary_mask_png(tmp_path: Path) -> None:
    mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    output = tmp_path / "mask.png"

    save_binary_mask_png(mask, output)

    saved = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
    assert saved is not None
    assert saved.tolist() == [[0, 255], [255, 0]]


def test_mask_must_be_2d() -> None:
    with pytest.raises(ValueError, match="2D"):
        mask_to_bbox(np.zeros((1, 2, 3), dtype=np.uint8))
