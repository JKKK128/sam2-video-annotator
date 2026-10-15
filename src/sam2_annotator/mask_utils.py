"""Mask and bounding-box helpers.

Bounding boxes use inclusive ``xyxy`` pixel coordinates throughout this module:
``x1 <= x <= x2`` and ``y1 <= y <= y2``. Therefore bbox width is
``x2 - x1 + 1`` and height is ``y2 - y1 + 1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

BBoxXYXY: TypeAlias = tuple[int, int, int, int]
PointXY: TypeAlias = tuple[float, float]


def _as_bool_mask(mask: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {array.shape}")
    return array.astype(bool)


def mask_to_bbox(mask: npt.ArrayLike) -> BBoxXYXY | None:
    """Return the inclusive ``xyxy`` bbox around non-zero mask pixels.

    Returns ``None`` for an empty mask.
    """
    bool_mask = _as_bool_mask(mask)
    ys, xs = np.nonzero(bool_mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_area(mask: npt.ArrayLike) -> int:
    """Return the number of non-zero pixels in a 2D mask."""
    return int(np.count_nonzero(_as_bool_mask(mask)))


def bbox_area(bbox: BBoxXYXY | None) -> int:
    """Return the area of an inclusive ``xyxy`` bbox, or 0 for ``None``."""
    if bbox is None:
        return 0
    x1, y1, x2, y2 = bbox
    if x2 < x1 or y2 < y1:
        return 0
    return int((x2 - x1 + 1) * (y2 - y1 + 1))


def bbox_center(bbox: BBoxXYXY | None) -> PointXY | None:
    """Return the center point of an inclusive ``xyxy`` bbox."""
    if bbox is None or bbox_area(bbox) == 0:
        return None
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_to_yolo(bbox: BBoxXYXY, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    """Convert an inclusive ``xyxy`` bbox to normalized YOLO ``cx cy w h``."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    area = bbox_area(bbox)
    if area == 0:
        raise ValueError(f"bbox must have positive area, got {bbox}")
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 >= image_width or y2 >= image_height:
        raise ValueError(f"bbox {bbox} is outside image size {image_width}x{image_height}")
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    return center_x / image_width, center_y / image_height, width / image_width, height / image_height


def save_binary_mask_png(mask: npt.ArrayLike, output_path: Path) -> None:
    """Save a 2D mask as a binary PNG with values 0 and 255."""
    import cv2

    bool_mask = _as_bool_mask(mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png = (bool_mask.astype(np.uint8) * 255)
    if not cv2.imwrite(str(output_path), png):
        raise OSError(f"failed to write mask PNG: {output_path}")
