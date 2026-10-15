"""Connected-component cleanup and temporal bbox stabilization."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import math
from statistics import median
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from sam2_annotator.mask_utils import BBoxXYXY, PointXY, bbox_area, bbox_center

BBoxStatus = Literal["initialized", "stable", "smoothed", "unstable", "missing"]


@dataclass(frozen=True)
class ComponentDiagnostics:
    component_count: int
    valid_component_count: int
    selected_component_area: int
    removed_component_count: int
    selected_component_center: PointXY | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["selected_component_center"] is not None:
            data["selected_component_center"] = list(data["selected_component_center"])
        return data


@dataclass(frozen=True)
class StabilizerParameters:
    center_alpha: float = 0.7
    size_alpha: float = 0.4
    history_size: int = 5
    max_size_ratio: float = 2.0
    min_size_ratio: float = 0.5
    max_center_jump_factor: float = 4.0
    minimum_center_jump_pixels: float = 8.0
    min_component_area: int = 3
    save_raw_masks: bool = False

    def validate(self) -> None:
        if not 0.0 <= self.center_alpha <= 1.0:
            raise ValueError("center_alpha must be between 0 and 1")
        if not 0.0 <= self.size_alpha <= 1.0:
            raise ValueError("size_alpha must be between 0 and 1")
        if self.history_size < 1:
            raise ValueError("history_size must be greater than 0")
        if self.max_size_ratio <= 0:
            raise ValueError("max_size_ratio must be positive")
        if self.min_size_ratio <= 0:
            raise ValueError("min_size_ratio must be positive")
        if self.min_size_ratio > self.max_size_ratio:
            raise ValueError("min_size_ratio must be <= max_size_ratio")
        if self.max_center_jump_factor <= 0:
            raise ValueError("max_center_jump_factor must be positive")
        if self.minimum_center_jump_pixels < 0:
            raise ValueError("minimum_center_jump_pixels must be non-negative")
        if self.min_component_area < 1:
            raise ValueError("min_component_area must be >= 1")


@dataclass(frozen=True)
class StabilizedBBox:
    raw_bbox_xyxy: BBoxXYXY | None
    stable_bbox_xyxy: BBoxXYXY | None
    bbox_status: BBoxStatus
    bbox_stabilized: bool
    bbox_stability_reasons: list[str]
    raw_center_xy: PointXY | None
    stable_center_xy: PointXY | None
    raw_width: int | None
    raw_height: int | None
    stable_width: int | None
    stable_height: int | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("raw_bbox_xyxy", "stable_bbox_xyxy"):
            if data[key] is not None:
                data[key] = list(data[key])
        for key in ("raw_center_xy", "stable_center_xy"):
            if data[key] is not None:
                data[key] = list(data[key])
        return data


@dataclass
class BBoxStabilizer:
    params: StabilizerParameters = field(default_factory=StabilizerParameters)
    initialized: bool = False
    _stable_center: PointXY | None = None
    _stable_size: tuple[float, float] | None = None
    _stable_bbox: BBoxXYXY | None = None
    _recent_widths: deque[float] = field(default_factory=deque)
    _recent_heights: deque[float] = field(default_factory=deque)
    _recent_centers: deque[PointXY] = field(default_factory=deque)
    _recent_displacements: deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.params.validate()

    @property
    def previous_center(self) -> PointXY | None:
        return self._stable_center

    @property
    def recent_area(self) -> float | None:
        if self._stable_size is None:
            return None
        return self._stable_size[0] * self._stable_size[1]

    def update(
        self,
        raw_bbox_xyxy: BBoxXYXY | None,
        image_size: tuple[int, int],
        candidate_bbox_xyxy: BBoxXYXY | None = None,
    ) -> StabilizedBBox:
        """Update one object's bbox history and return raw plus stable boxes.

        ``raw_bbox_xyxy`` is preserved in the output. ``candidate_bbox_xyxy`` may
        be a cleaned component bbox used for smoothing; when omitted, raw is used.
        """
        width, height = _validate_image_size(image_size)
        raw_bbox = _clip_bbox(raw_bbox_xyxy, width, height)
        candidate = _clip_bbox(candidate_bbox_xyxy if candidate_bbox_xyxy is not None else raw_bbox, width, height)
        raw_center = bbox_center(raw_bbox)
        raw_size = _bbox_size(raw_bbox)

        if candidate is None:
            return self._missing_result(raw_bbox, raw_center, raw_size)

        candidate_center = bbox_center(candidate)
        candidate_size = _bbox_size(candidate)
        if candidate_center is None or candidate_size is None:
            return self._missing_result(raw_bbox, raw_center, raw_size)

        if not self.initialized:
            self.initialized = True
            self._stable_center = candidate_center
            self._stable_size = (float(candidate_size[0]), float(candidate_size[1]))
            self._stable_bbox = candidate
            self._accept_history(candidate_center, candidate_size)
            return self._result(raw_bbox, candidate, "initialized", False, [], raw_center, raw_size)

        reasons = self._outlier_reasons(candidate_center, candidate_size)
        if reasons:
            return self._result(raw_bbox, self._stable_bbox, "unstable", False, reasons, raw_center, raw_size)

        assert self._stable_center is not None
        assert self._stable_size is not None
        center = (
            self.params.center_alpha * candidate_center[0] + (1.0 - self.params.center_alpha) * self._stable_center[0],
            self.params.center_alpha * candidate_center[1] + (1.0 - self.params.center_alpha) * self._stable_center[1],
        )
        size = (
            max(1.0, self.params.size_alpha * candidate_size[0] + (1.0 - self.params.size_alpha) * self._stable_size[0]),
            max(1.0, self.params.size_alpha * candidate_size[1] + (1.0 - self.params.size_alpha) * self._stable_size[1]),
        )
        stable_bbox = _bbox_from_center_size(center, size, width, height)

        self._stable_center = center
        self._stable_size = size
        self._stable_bbox = stable_bbox
        self._accept_history(candidate_center, candidate_size)
        changed = stable_bbox != candidate
        status: BBoxStatus = "smoothed" if changed else "stable"
        return self._result(raw_bbox, stable_bbox, status, changed, ["ema_smoothing"] if changed else [], raw_center, raw_size)

    def _missing_result(
        self,
        raw_bbox: BBoxXYXY | None,
        raw_center: PointXY | None,
        raw_size: tuple[int, int] | None,
    ) -> StabilizedBBox:
        stable = self._stable_bbox if self.initialized else None
        return self._result(raw_bbox, stable, "missing", False, ["missing"], raw_center, raw_size)

    def _result(
        self,
        raw_bbox: BBoxXYXY | None,
        stable_bbox: BBoxXYXY | None,
        status: BBoxStatus,
        stabilized: bool,
        reasons: list[str],
        raw_center: PointXY | None,
        raw_size: tuple[int, int] | None,
    ) -> StabilizedBBox:
        stable_size = _bbox_size(stable_bbox)
        return StabilizedBBox(
            raw_bbox_xyxy=raw_bbox,
            stable_bbox_xyxy=stable_bbox,
            bbox_status=status,
            bbox_stabilized=stabilized,
            bbox_stability_reasons=reasons,
            raw_center_xy=raw_center,
            stable_center_xy=bbox_center(stable_bbox),
            raw_width=raw_size[0] if raw_size is not None else None,
            raw_height=raw_size[1] if raw_size is not None else None,
            stable_width=stable_size[0] if stable_size is not None else None,
            stable_height=stable_size[1] if stable_size is not None else None,
        )

    def _outlier_reasons(self, center: PointXY, size: tuple[int, int]) -> list[str]:
        reasons: list[str] = []
        if self._recent_widths and self._recent_heights:
            median_w = float(median(self._recent_widths))
            median_h = float(median(self._recent_heights))
            width_ratio = _safe_ratio(size[0], median_w)
            height_ratio = _safe_ratio(size[1], median_h)
            if (
                width_ratio is not None
                and (width_ratio > self.params.max_size_ratio or width_ratio < self.params.min_size_ratio)
            ) or (
                height_ratio is not None
                and (height_ratio > self.params.max_size_ratio or height_ratio < self.params.min_size_ratio)
            ):
                reasons.append("size_outlier")

        if self._recent_centers:
            reference = self._recent_centers[-1]
            displacement = math.hypot(center[0] - reference[0], center[1] - reference[1])
            median_motion = float(median(self._recent_displacements)) if self._recent_displacements else 0.0
            threshold = max(self.params.minimum_center_jump_pixels, self.params.max_center_jump_factor * median_motion)
            if displacement > threshold:
                reasons.append("center_outlier")
        return reasons

    def _accept_history(self, center: PointXY, size: tuple[int, int]) -> None:
        if self._recent_centers:
            previous = self._recent_centers[-1]
            _append_limited(self._recent_displacements, math.hypot(center[0] - previous[0], center[1] - previous[1]), self.params.history_size)
        _append_limited(self._recent_centers, center, self.params.history_size)
        _append_limited(self._recent_widths, float(size[0]), self.params.history_size)
        _append_limited(self._recent_heights, float(size[1]), self.params.history_size)


def clean_mask_components(
    mask: npt.ArrayLike,
    previous_center: PointXY | None = None,
    recent_area: float | None = None,
    min_component_area: int = 3,
) -> tuple[npt.NDArray[np.bool_], ComponentDiagnostics]:
    """Keep the most plausible connected component without mutating ``mask``."""
    import cv2

    if min_component_area < 1:
        raise ValueError("min_component_area must be >= 1")
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {array.shape}")
    binary = array.astype(bool)
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    component_count = max(0, int(labels_count) - 1)
    valid_ids = [idx for idx in range(1, labels_count) if int(stats[idx, cv2.CC_STAT_AREA]) >= min_component_area]
    removed = component_count - len(valid_ids)
    if not valid_ids:
        diagnostics = ComponentDiagnostics(
            component_count=component_count,
            valid_component_count=0,
            selected_component_area=0,
            removed_component_count=removed,
            selected_component_center=None,
        )
        return np.zeros(binary.shape, dtype=bool), diagnostics

    selected = _select_component(valid_ids, stats, centroids, previous_center, recent_area)
    cleaned = labels == selected
    selected_center = (float(centroids[selected, 0]), float(centroids[selected, 1]))
    diagnostics = ComponentDiagnostics(
        component_count=component_count,
        valid_component_count=len(valid_ids),
        selected_component_area=int(stats[selected, cv2.CC_STAT_AREA]),
        removed_component_count=removed,
        selected_component_center=selected_center,
    )
    return cleaned.astype(bool, copy=False), diagnostics


def parameters_from_mapping(values: dict[str, Any] | None) -> StabilizerParameters:
    if values is None:
        params = StabilizerParameters()
    else:
        allowed = set(StabilizerParameters.__dataclass_fields__)
        params = StabilizerParameters(**{key: value for key, value in values.items() if key in allowed})
    params.validate()
    return params


def _select_component(
    valid_ids: list[int],
    stats: npt.NDArray[np.int32],
    centroids: npt.NDArray[np.float64],
    previous_center: PointXY | None,
    recent_area: float | None,
) -> int:
    if previous_center is None:
        return max(valid_ids, key=lambda idx: int(stats[idx, 4]))

    def score(idx: int) -> tuple[float, float]:
        cx = float(centroids[idx, 0])
        cy = float(centroids[idx, 1])
        distance = math.hypot(cx - previous_center[0], cy - previous_center[1])
        area = float(stats[idx, 4])
        if recent_area is not None and recent_area > 0:
            area_penalty = abs(math.log(max(area, 1.0) / recent_area))
        else:
            area_penalty = 0.0
        return distance + area_penalty, -area

    return min(valid_ids, key=score)


def _append_limited(queue: deque[Any], value: Any, size: int) -> None:
    queue.append(value)
    while len(queue) > size:
        queue.popleft()


def _safe_ratio(value: float, baseline: float) -> float | None:
    if baseline <= 0:
        return None
    ratio = value / baseline
    return ratio if math.isfinite(ratio) else None


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    return width, height


def _bbox_size(bbox: BBoxXYXY | None) -> tuple[int, int] | None:
    if bbox is None or bbox_area(bbox) == 0:
        return None
    x1, y1, x2, y2 = bbox
    return x2 - x1 + 1, y2 - y1 + 1


def _clip_bbox(bbox: BBoxXYXY | None, width: int, height: int) -> BBoxXYXY | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = min(max(0, x1), width - 1)
    y1 = min(max(0, y1), height - 1)
    x2 = min(max(0, x2), width - 1)
    y2 = min(max(0, y2), height - 1)
    if x2 < x1 or y2 < y1:
        return None
    return x1, y1, x2, y2


def _bbox_from_center_size(center: PointXY, size: tuple[float, float], image_width: int, image_height: int) -> BBoxXYXY:
    width = max(1, int(round(size[0])))
    height = max(1, int(round(size[1])))
    x1 = int(round(center[0] - (width - 1) / 2.0))
    y1 = int(round(center[1] - (height - 1) / 2.0))
    x2 = x1 + width - 1
    y2 = y1 + height - 1
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 >= image_width:
        shift = x2 - image_width + 1
        x1 -= shift
        x2 -= shift
    if y2 >= image_height:
        shift = y2 - image_height + 1
        y1 -= shift
        y2 -= shift
    x1 = min(max(0, x1), image_width - 1)
    y1 = min(max(0, y1), image_height - 1)
    x2 = min(max(x1, x2), image_width - 1)
    y2 = min(max(y1, y2), image_height - 1)
    return x1, y1, x2, y2
