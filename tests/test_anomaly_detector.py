from __future__ import annotations

from sam2_annotator.anomaly_detector import (
    AnomalyParameters,
    TrackObservation,
    evaluate_tracks,
    merge_review_segments,
)


def obs(frame_idx: int, *, obj_id: int = 1, bbox: tuple[int, int, int, int] | None = (10, 10, 19, 19), mask_area: int = 80) -> TrackObservation:
    bbox_area = 0 if bbox is None else (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    center = None if bbox is None else ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
    return TrackObservation(
        frame_idx=frame_idx,
        obj_id=obj_id,
        mask_area=mask_area,
        bbox_area=bbox_area,
        bbox_xyxy=bbox,
        center_xy=center,
    )


def test_constant_track_has_no_serious_anomaly() -> None:
    observations = [obs(i, bbox=(10 + i, 10, 19 + i, 19), mask_area=80) for i in range(8)]

    results = evaluate_tracks(observations, AnomalyParameters(history_size=5), image_size=(200, 100))

    assert all(result.status == "auto_ok" for result in results)
    assert all(result.score == 0 for result in results)


def test_area_increase() -> None:
    observations = [obs(i, mask_area=80) for i in range(5)]
    observations.append(obs(5, bbox=(5, 5, 24, 24), mask_area=400))

    result = evaluate_tracks(observations, AnomalyParameters(history_size=5), image_size=(200, 100))[-1]

    assert result.status == "need_review"
    assert "area_increase" in result.reasons
    assert result.score == 2


def test_area_drop() -> None:
    observations = [obs(i, mask_area=80) for i in range(5)]
    observations.append(obs(5, mask_area=20))

    result = evaluate_tracks(observations, AnomalyParameters(history_size=5), image_size=(200, 100))[-1]

    assert result.status == "need_review"
    assert "area_drop" in result.reasons


def test_center_jump() -> None:
    observations = [obs(i, bbox=(10 + i, 10, 19 + i, 19), mask_area=80) for i in range(5)]
    observations.append(obs(5, bbox=(100, 10, 109, 19), mask_area=80))

    result = evaluate_tracks(observations, AnomalyParameters(history_size=5), image_size=(200, 100))[-1]

    assert result.status == "need_review"
    assert "center_jump" in result.reasons
    assert result.center_displacement is not None


def test_empty_mask_is_critical() -> None:
    observations = [obs(i, mask_area=80) for i in range(3)]
    observations.append(obs(3, bbox=None, mask_area=0))

    result = evaluate_tracks(observations, AnomalyParameters(history_size=5), image_size=(200, 100))[-1]

    assert result.status == "critical"
    assert result.score == 0
    assert result.reasons == ["empty_mask"]


def test_short_history_does_not_crash_or_emit_nan() -> None:
    results = evaluate_tracks([obs(0), obs(1)], AnomalyParameters(history_size=5), image_size=(200, 100))

    for result in results:
        assert result.status == "auto_ok"
        assert result.area_ratio is None or result.area_ratio == result.area_ratio
        assert result.center_jump_ratio is None or result.center_jump_ratio == result.center_jump_ratio


def test_obj_histories_are_separate() -> None:
    observations = [obs(i, obj_id=1, mask_area=80) for i in range(5)]
    observations.extend(obs(i, obj_id=2, bbox=(100, 20, 119, 39), mask_area=300) for i in range(5))
    observations.append(obs(5, obj_id=1, bbox=(5, 5, 24, 24), mask_area=400))
    observations.append(obs(5, obj_id=2, bbox=(100, 20, 119, 39), mask_area=300))

    results = evaluate_tracks(observations, AnomalyParameters(history_size=5), image_size=(200, 100))
    by_key = {(result.frame_idx, result.obj_id): result for result in results}

    assert "area_increase" in by_key[(5, 1)].reasons
    assert "area_increase" not in by_key[(5, 2)].reasons


def test_merge_review_segments_with_small_gaps() -> None:
    observations = [obs(i, mask_area=80) for i in range(5)]
    observations.extend(
        [
            obs(5, bbox=(5, 5, 24, 24), mask_area=400),
            obs(8, bbox=(5, 5, 24, 24), mask_area=400),
            obs(12, bbox=(5, 5, 24, 24), mask_area=400),
        ]
    )
    results = evaluate_tracks(observations, AnomalyParameters(history_size=5), image_size=(200, 100))

    segments = merge_review_segments(results)

    assert len(segments) == 2
    assert segments[0].start_frame == 5
    assert segments[0].end_frame == 8
    assert segments[0].member_frames == [5, 8]
    assert segments[1].start_frame == 12


def test_periodic_check_enters_review_without_score() -> None:
    observations = [obs(i) for i in range(6)]

    result = evaluate_tracks(observations, AnomalyParameters(history_size=5, periodic_interval=5), image_size=(200, 100))[-1]

    assert result.frame_idx == 5
    assert result.status == "auto_ok"
    assert result.score == 0
    assert result.reasons == ["periodic_check"]


def test_zero_bbox_or_mask_area_does_not_crash() -> None:
    bad_bbox = TrackObservation(frame_idx=0, obj_id=1, mask_area=10, bbox_area=0, bbox_xyxy=(5, 5, 4, 4), center_xy=None)
    zero_mask = TrackObservation(frame_idx=1, obj_id=1, mask_area=0, bbox_area=100, bbox_xyxy=(10, 10, 19, 19), center_xy=(14.5, 14.5))

    results = evaluate_tracks([bad_bbox, zero_mask], AnomalyParameters(history_size=5), image_size=(200, 100))

    assert [result.status for result in results] == ["critical", "critical"]
    assert all("empty_mask" in result.reasons for result in results)
