"""Pure UI helpers for frame navigation and browser rectangle selections."""

from __future__ import annotations

from typing import Sequence


BBoxXYXY = tuple[int, int, int, int]


def clamp_frame_index(frame_idx: int | float, frame_count: int) -> int:
    """Return a frame index constrained to a non-empty source's range."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    return min(max(int(frame_idx), 0), frame_count - 1)


def previous_frame(frame_idx: int | float, frame_count: int) -> int:
    """Move one frame backward without leaving the source."""
    return clamp_frame_index(int(frame_idx) - 1, frame_count)


def next_frame(frame_idx: int | float, frame_count: int) -> int:
    """Move one frame forward without leaving the source."""
    return clamp_frame_index(int(frame_idx) + 1, frame_count)


def normalize_browser_bbox(
    bbox_xyxy: Sequence[float | int],
    *,
    original_width: int,
    original_height: int,
    display_width: int | None = None,
    display_height: int | None = None,
) -> BBoxXYXY:
    """Map an inclusive browser bbox to inclusive original-image coordinates."""
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain exactly four coordinates")
    if original_width <= 0 or original_height <= 0:
        raise ValueError("original image dimensions must be positive")
    shown_width = original_width if display_width is None else display_width
    shown_height = original_height if display_height is None else display_height
    if shown_width <= 0 or shown_height <= 0:
        raise ValueError("display image dimensions must be positive")
    scale_x = original_width / shown_width
    scale_y = original_height / shown_height
    x1, y1, x2, y2 = (
        round(float(value) * scale)
        for value, scale in zip(bbox_xyxy, (scale_x, scale_y, scale_x, scale_y))
    )
    x1 = min(max(x1, 0), original_width - 1)
    x2 = min(max(x2, 0), original_width - 1)
    y1 = min(max(y1, 0), original_height - 1)
    y2 = min(max(y2, 0), original_height - 1)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have non-zero width and height")
    return x1, y1, x2, y2


def browser_rect_to_bbox(
    x: float | int,
    y: float | int,
    width: float | int,
    height: float | int,
    *,
    original_width: int,
    original_height: int,
    display_width: int,
    display_height: int,
) -> BBoxXYXY:
    """Convert browser ``x, y, width, height`` to an inclusive source bbox."""
    if float(width) <= 0 or float(height) <= 0:
        raise ValueError("bbox width and height must be positive")
    return normalize_browser_bbox(
        (float(x), float(y), float(x) + float(width) - 1, float(y) + float(height) - 1),
        original_width=original_width,
        original_height=original_height,
        display_width=display_width,
        display_height=display_height,
    )


def frame_range_after_start(start_frame: int | float, end_frame: int | float, frame_count: int) -> tuple[int, int]:
    """Clamp the start frame and raise the end frame if it falls before it."""
    start = clamp_frame_index(start_frame, frame_count)
    end = clamp_frame_index(end_frame, frame_count)
    return start, max(start, end)


def format_bbox(bbox: BBoxXYXY | None) -> str:
    """Format one inclusive bbox for the visible annotation status."""
    if bbox is None:
        return "BBox：尚未框选目标。"
    x1, y1, x2, y2 = bbox
    return (
        "BBox:\n"
        f"x1 = {x1}\n"
        f"y1 = {y1}\n"
        f"x2 = {x2}\n"
        f"y2 = {y2}\n"
        f"width = {x2 - x1 + 1}\n"
        f"height = {y2 - y1 + 1}"
    )
