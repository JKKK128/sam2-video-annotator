"""Anomaly screening helpers for SAM2 propagation tracks.

The checks in this module use only SAM2 outputs and temporal statistics. The
score is a review priority signal, not a model confidence value.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import math
from statistics import median
from typing import Any, Iterable, Literal

from sam2_annotator.mask_utils import BBoxXYXY, bbox_area, bbox_center

Status = Literal["auto_ok", "warning", "need_review", "critical"]


@dataclass(frozen=True)
class AnomalyParameters:
    history_size: int = 5
    area_ratio_min: float = 0.35
    area_ratio_max: float = 2.8
    center_jump_factor: float = 4.0
    center_jump_min_diagonal_ratio: float = 0.02
    aspect_ratio_min: float = 0.33
    aspect_ratio_max: float = 3.0
    fill_ratio_min: float = 0.4
    border_margin: int = 2
    periodic_interval: int = 50

    def validate(self) -> None:
        if self.history_size < 1:
            raise ValueError("history_size must be >= 1")
        if self.area_ratio_min <= 0 or self.area_ratio_max <= 0 or self.area_ratio_min >= self.area_ratio_max:
            raise ValueError("area ratio bounds must be positive and increasing")
        if self.center_jump_factor <= 0:
            raise ValueError("center_jump_factor must be positive")
        if self.center_jump_min_diagonal_ratio < 0:
            raise ValueError("center_jump_min_diagonal_ratio must be non-negative")
        if self.aspect_ratio_min <= 0 or self.aspect_ratio_max <= 0 or self.aspect_ratio_min >= self.aspect_ratio_max:
            raise ValueError("aspect ratio bounds must be positive and increasing")
        if self.fill_ratio_min <= 0:
            raise ValueError("fill_ratio_min must be positive")
        if self.border_margin < 0:
            raise ValueError("border_margin must be non-negative")
        if self.periodic_interval < 0:
            raise ValueError("periodic_interval must be non-negative")


@dataclass(frozen=True)
class TrackObservation:
    frame_idx: int
    obj_id: int
    mask_area: int
    bbox_area: int
    bbox_xyxy: BBoxXYXY | None
    center_xy: tuple[float, float] | None
    mask_path: str | None = None
    object_name: str | None = None
    raw_bbox_xyxy: BBoxXYXY | None = None
    stable_bbox_xyxy: BBoxXYXY | None = None
    bbox_status: str | None = None
    bbox_stability_reasons: list[str] = field(default_factory=list)
    component_count: int | None = None
    selected_component_area: int | None = None
    removed_component_count: int | None = None
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.mask_area <= 0 or self.bbox_area <= 0 or self.bbox_xyxy is None or self.center_xy is None

    @property
    def aspect_ratio(self) -> float | None:
        if self.bbox_xyxy is None:
            return None
        x1, y1, x2, y2 = self.bbox_xyxy
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        if width <= 0 or height <= 0:
            return None
        return width / height

    @property
    def fill_ratio(self) -> float | None:
        if self.bbox_area <= 0:
            return None
        return self.mask_area / self.bbox_area

    @property
    def bbox_diagonal(self) -> float | None:
        if self.bbox_xyxy is None:
            return None
        x1, y1, x2, y2 = self.bbox_xyxy
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        if width <= 0 or height <= 0:
            return None
        return math.hypot(width, height)


@dataclass(frozen=True)
class Baseline:
    mask_area: float | None = None
    center_xy: tuple[float, float] | None = None
    step_displacement: float | None = None
    aspect_ratio: float | None = None
    fill_ratio: float | None = None
    bbox_diagonal: float | None = None
    count: int = 0


@dataclass(frozen=True)
class AnomalyResult:
    frame_idx: int
    obj_id: int
    object_name: str | None
    status: Status
    score: int
    reasons: list[str]
    mask_area: int
    bbox_area: int
    bbox_xyxy: list[int] | None
    center_xy: list[float] | None
    raw_bbox_xyxy: list[int] | None
    stable_bbox_xyxy: list[int] | None
    bbox_status: str | None
    bbox_stability_reasons: list[str]
    area_ratio: float | None
    center_displacement: float | None
    center_jump_ratio: float | None
    aspect_ratio: float | None
    aspect_ratio_change: float | None
    fill_ratio: float | None
    fill_ratio_change: float | None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewSegment:
    obj_id: int
    start_frame: int
    end_frame: int
    peak_frame: int
    max_score: int
    reasons: list[str]
    member_frames: list[int]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def observation_from_record(record: dict[str, Any]) -> TrackObservation:
    """Build a validated observation from one tracks.jsonl record.

    Anomaly detection intentionally defaults to ``raw_bbox_xyxy`` when present
    so stable bbox smoothing cannot hide SAM2 propagation errors.
    """
    frame_idx = int(record["frame_idx"])
    obj_id = int(record["obj_id"])
    raw_bbox = _parse_bbox(record.get("raw_bbox_xyxy"), frame_idx, obj_id, "raw_bbox_xyxy")
    stable_bbox = _parse_bbox(record.get("stable_bbox_xyxy"), frame_idx, obj_id, "stable_bbox_xyxy")
    bbox = raw_bbox if "raw_bbox_xyxy" in record else _parse_bbox(record.get("bbox_xyxy"), frame_idx, obj_id, "bbox_xyxy")
    center_value = record.get("raw_center_xy") if "raw_bbox_xyxy" in record else record.get("center_xy")
    center: tuple[float, float] | None = None
    if center_value is not None:
        if not isinstance(center_value, list | tuple) or len(center_value) != 2:
            raise ValueError(f"frame {frame_idx} obj {obj_id} has invalid center_xy")
        center = (float(center_value[0]), float(center_value[1]))
    elif bbox is not None:
        center = bbox_center(bbox)
    mask_area_value = int(record.get("mask_area", 0) or 0)
    bbox_area_value = int(record.get("bbox_area", bbox_area(bbox)) or 0)
    return TrackObservation(
        frame_idx=frame_idx,
        obj_id=obj_id,
        mask_area=max(0, mask_area_value),
        bbox_area=max(0, bbox_area_value),
        bbox_xyxy=bbox,
        center_xy=center,
        mask_path=str(record["mask_path"]) if record.get("mask_path") else None,
        object_name=str(record["object_name"]) if record.get("object_name") is not None else None,
        raw_bbox_xyxy=raw_bbox,
        stable_bbox_xyxy=stable_bbox,
        bbox_status=str(record["bbox_status"]) if record.get("bbox_status") is not None else None,
        bbox_stability_reasons=[str(reason) for reason in record.get("bbox_stability_reasons", [])],
        component_count=int(record["component_count"]) if record.get("component_count") is not None else None,
        selected_component_area=int(record["selected_component_area"]) if record.get("selected_component_area") is not None else None,
        removed_component_count=int(record["removed_component_count"]) if record.get("removed_component_count") is not None else None,
        source=record,
    )


def build_baseline(history: Iterable[TrackObservation], history_size: int) -> Baseline:
    valid = [obs for obs in history if not obs.is_empty][-history_size:]
    if not valid:
        return Baseline(count=0)

    areas = [obs.mask_area for obs in valid if obs.mask_area > 0]
    centers = [obs.center_xy for obs in valid if obs.center_xy is not None]
    aspects = [obs.aspect_ratio for obs in valid if obs.aspect_ratio is not None]
    fills = [obs.fill_ratio for obs in valid if obs.fill_ratio is not None]
    diagonals = [obs.bbox_diagonal for obs in valid if obs.bbox_diagonal is not None]
    steps: list[float] = []
    for prev, current in zip(valid, valid[1:]):
        if prev.center_xy is None or current.center_xy is None:
            continue
        steps.append(_distance(prev.center_xy, current.center_xy))

    center = None
    if centers:
        center = (float(median([point[0] for point in centers])), float(median([point[1] for point in centers])))

    return Baseline(
        mask_area=float(median(areas)) if areas else None,
        center_xy=center,
        step_displacement=float(median(steps)) if steps else None,
        aspect_ratio=float(median(aspects)) if aspects else None,
        fill_ratio=float(median(fills)) if fills else None,
        bbox_diagonal=float(median(diagonals)) if diagonals else None,
        count=len(valid),
    )


def empty_mask(obs: TrackObservation) -> bool:
    return obs.is_empty


def area_drop(obs: TrackObservation, baseline: Baseline, params: AnomalyParameters) -> tuple[bool, float | None]:
    ratio = _ratio(obs.mask_area, baseline.mask_area)
    return ratio is not None and ratio < params.area_ratio_min, ratio


def area_increase(obs: TrackObservation, baseline: Baseline, params: AnomalyParameters) -> tuple[bool, float | None]:
    ratio = _ratio(obs.mask_area, baseline.mask_area)
    return ratio is not None and ratio > params.area_ratio_max, ratio


def center_jump(
    obs: TrackObservation,
    baseline: Baseline,
    params: AnomalyParameters,
    image_size: tuple[int, int] | None = None,
) -> tuple[bool, float | None, float | None]:
    if obs.center_xy is None or baseline.center_xy is None:
        return False, None, None
    displacement = _distance(obs.center_xy, baseline.center_xy)
    reference_diagonal = _reference_diagonal(baseline, image_size)
    min_jump = params.center_jump_min_diagonal_ratio * reference_diagonal if reference_diagonal is not None else 0.0
    step_jump = params.center_jump_factor * baseline.step_displacement if baseline.step_displacement is not None else 0.0
    threshold = max(min_jump, step_jump)
    ratio = displacement / threshold if threshold > 0 else None
    return threshold > 0 and displacement > threshold, displacement, ratio


def aspect_ratio_jump(obs: TrackObservation, baseline: Baseline, params: AnomalyParameters) -> tuple[bool, float | None, float | None]:
    aspect = obs.aspect_ratio
    change = _ratio(aspect, baseline.aspect_ratio)
    return change is not None and (change < params.aspect_ratio_min or change > params.aspect_ratio_max), aspect, change


def fill_ratio_drop(obs: TrackObservation, baseline: Baseline, params: AnomalyParameters) -> tuple[bool, float | None, float | None]:
    fill = obs.fill_ratio
    change = _ratio(fill, baseline.fill_ratio)
    return change is not None and change < params.fill_ratio_min, fill, change


def touch_border(obs: TrackObservation, image_size: tuple[int, int] | None, params: AnomalyParameters) -> bool:
    if obs.bbox_xyxy is None or image_size is None:
        return False
    width, height = image_size
    if width <= 0 or height <= 0:
        return False
    x1, y1, x2, y2 = obs.bbox_xyxy
    margin = params.border_margin
    return x1 <= margin or y1 <= margin or x2 >= width - 1 - margin or y2 >= height - 1 - margin


def periodic_check(frame_idx: int, params: AnomalyParameters, first_prompt_frame: int = 0) -> bool:
    offset = frame_idx - first_prompt_frame
    return params.periodic_interval > 0 and offset > 0 and offset % params.periodic_interval == 0


def evaluate_observation(
    obs: TrackObservation,
    history: Iterable[TrackObservation],
    params: AnomalyParameters,
    image_size: tuple[int, int] | None = None,
    first_prompt_frame: int = 0,
) -> AnomalyResult:
    params.validate()
    baseline = build_baseline(history, params.history_size)
    reasons: list[str] = []
    score = 0
    forced_critical = False

    area_ratio: float | None = None
    center_displacement: float | None = None
    center_jump_ratio: float | None = None
    aspect_ratio = obs.aspect_ratio
    aspect_ratio_change: float | None = None
    fill_ratio = obs.fill_ratio
    fill_ratio_change: float | None = None

    in_warmup = baseline.count < params.history_size

    if empty_mask(obs):
        reasons.append("empty_mask")
        forced_critical = True
    elif not in_warmup:
        hit, area_ratio = area_drop(obs, baseline, params)
        if hit:
            reasons.append("area_drop")
            score += 2
        hit, area_ratio_increase = area_increase(obs, baseline, params)
        area_ratio = area_ratio if area_ratio is not None else area_ratio_increase
        if hit:
            reasons.append("area_increase")
            score += 2
        hit, center_displacement, center_jump_ratio = center_jump(obs, baseline, params, image_size)
        if hit:
            reasons.append("center_jump")
            score += 2
        hit, aspect_ratio, aspect_ratio_change = aspect_ratio_jump(obs, baseline, params)
        if hit:
            reasons.append("aspect_ratio_jump")
            score += 1
        hit, fill_ratio, fill_ratio_change = fill_ratio_drop(obs, baseline, params)
        if hit:
            reasons.append("fill_ratio_drop")
            score += 1
        if touch_border(obs, image_size, params):
            reasons.append("touch_border")
            score += 1
        if obs.component_count is not None and obs.component_count > 1:
            reasons.append("disconnected_mask")
            score += 1
        if "size_outlier" in obs.bbox_stability_reasons:
            reasons.append("bbox_size_outlier")
            score += 2
        if "center_outlier" in obs.bbox_stability_reasons:
            reasons.append("bbox_center_outlier")
            score += 2
        if obs.bbox_status == "unstable":
            reasons.append("bbox_unstable")
            score += 1

    if periodic_check(obs.frame_idx, params, first_prompt_frame):
        reasons.append("periodic_check")

    status = _status_for_score(score, forced_critical)
    return AnomalyResult(
        frame_idx=obs.frame_idx,
        obj_id=obs.obj_id,
        object_name=obs.object_name,
        status=status,
        score=score,
        reasons=reasons,
        mask_area=obs.mask_area,
        bbox_area=obs.bbox_area,
        bbox_xyxy=list(obs.bbox_xyxy) if obs.bbox_xyxy is not None else None,
        center_xy=list(obs.center_xy) if obs.center_xy is not None else None,
        raw_bbox_xyxy=list(obs.raw_bbox_xyxy) if obs.raw_bbox_xyxy is not None else None,
        stable_bbox_xyxy=list(obs.stable_bbox_xyxy) if obs.stable_bbox_xyxy is not None else None,
        bbox_status=obs.bbox_status,
        bbox_stability_reasons=obs.bbox_stability_reasons,
        area_ratio=area_ratio,
        center_displacement=center_displacement,
        center_jump_ratio=center_jump_ratio,
        aspect_ratio=aspect_ratio,
        aspect_ratio_change=aspect_ratio_change,
        fill_ratio=fill_ratio,
        fill_ratio_change=fill_ratio_change,
    )


def evaluate_tracks(
    observations: Iterable[TrackObservation],
    params: AnomalyParameters,
    image_size: tuple[int, int] | None = None,
    first_prompt_frames: dict[int, int] | None = None,
) -> list[AnomalyResult]:
    observation_list = list(observations)
    histories: dict[int, list[TrackObservation]] = {}
    first_prompt_frames = dict(first_prompt_frames or {})
    for obs in observation_list:
        first_prompt_frames.setdefault(obs.obj_id, obs.frame_idx)
    results: list[AnomalyResult] = []
    for obs in sorted(observation_list, key=lambda item: (item.obj_id, item.frame_idx)):
        history = histories.setdefault(obs.obj_id, [])
        results.append(evaluate_observation(obs, history, params, image_size, first_prompt_frames.get(obs.obj_id, obs.frame_idx)))
        if not obs.is_empty:
            history.append(obs)
    return sorted(results, key=lambda item: (item.frame_idx, item.obj_id))


def with_duplicate_warning(result: AnomalyResult) -> AnomalyResult:
    if "duplicate_target_warning" in result.reasons:
        return result
    reasons = [*result.reasons, "duplicate_target_warning"]
    score = result.score + 1
    status = _status_for_score(score, False) if result.status == "auto_ok" else result.status
    return replace(result, reasons=reasons, score=score, status=status)


def merge_review_segments(results: Iterable[AnomalyResult], max_gap: int = 2) -> list[ReviewSegment]:
    segments: list[ReviewSegment] = []
    by_obj: dict[int, list[AnomalyResult]] = {}
    for result in results:
        anomalous_reasons = [reason for reason in result.reasons if reason != "periodic_check"]
        if result.score <= 0 or not anomalous_reasons:
            continue
        by_obj.setdefault(result.obj_id, []).append(result)

    for obj_id, obj_results in sorted(by_obj.items()):
        current: list[AnomalyResult] = []
        for result in sorted(obj_results, key=lambda item: item.frame_idx):
            if not current or result.frame_idx - current[-1].frame_idx <= max_gap + 1:
                current.append(result)
            else:
                segments.append(_build_segment(obj_id, current))
                current = [result]
        if current:
            segments.append(_build_segment(obj_id, current))
    return segments


def summarize_results(results: Iterable[AnomalyResult], params: AnomalyParameters) -> dict[str, Any]:
    result_list = list(results)
    status_counts = Counter(result.status for result in result_list)
    reason_counts = Counter(reason for result in result_list for reason in result.reasons)
    review_frame_count = sum(1 for result in result_list if should_review(result))
    return {
        "total_frames": len({result.frame_idx for result in result_list}),
        "total_objects": len({result.obj_id for result in result_list}),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "review_frame_count": review_frame_count,
        "parameters": asdict(params),
    }


def should_review(result: AnomalyResult) -> bool:
    return result.status in {"need_review", "critical"} or "periodic_check" in result.reasons


def _build_segment(obj_id: int, members: list[AnomalyResult]) -> ReviewSegment:
    peak = max(members, key=lambda item: (item.score, -item.frame_idx))
    reasons = sorted({reason for item in members for reason in item.reasons if reason != "periodic_check"})
    return ReviewSegment(
        obj_id=obj_id,
        start_frame=members[0].frame_idx,
        end_frame=members[-1].frame_idx,
        peak_frame=peak.frame_idx,
        max_score=peak.score,
        reasons=reasons,
        member_frames=[item.frame_idx for item in members],
    )


def _ratio(value: float | int | None, baseline: float | int | None) -> float | None:
    if value is None or baseline is None or baseline <= 0:
        return None
    ratio = float(value) / float(baseline)
    if not math.isfinite(ratio):
        return None
    return ratio


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _parse_bbox(value: Any, frame_idx: int, obj_id: int, field_name: str) -> BBoxXYXY | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"frame {frame_idx} obj {obj_id} has invalid {field_name}")
    bbox: BBoxXYXY = tuple(int(v) for v in value)  # type: ignore[assignment]
    return bbox if bbox_area(bbox) > 0 else None


def _reference_diagonal(baseline: Baseline, image_size: tuple[int, int] | None) -> float | None:
    if image_size is not None and image_size[0] > 0 and image_size[1] > 0:
        return math.hypot(float(image_size[0]), float(image_size[1]))
    return baseline.bbox_diagonal


def _status_for_score(score: int, forced_critical: bool) -> Status:
    if forced_critical or score >= 4:
        return "critical"
    if score >= 2:
        return "need_review"
    if score == 1:
        return "warning"
    return "auto_ok"
