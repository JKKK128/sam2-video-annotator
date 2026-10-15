from __future__ import annotations

from sam2_annotator.anomaly_intervals import (
    IntervalParameters,
    build_intervals,
    compare_interval_sets,
    detect_frame_anomalies,
    inherit_review_status,
    update_interval_status,
)


REGISTRY = {"video": "input", "frames_dir": "frames", "objects": [
    {"obj_id": 1, "name": "one", "global_drone_id": "drone_1", "class_id": 1, "first_visible_frame": 0, "validated": True, "status": "active"},
    {"obj_id": 2, "name": "two", "global_drone_id": "drone_2", "class_id": 2, "first_visible_frame": 0, "validated": True, "status": "active"},
]}
CATALOG = {"drone_1": {"class_id": 1}, "drone_2": {"class_id": 2}}


def row(frame: int, size: int = 10, obj: int = 1, **extra: object) -> dict[str, object]:
    result: dict[str, object] = {"frame_idx": frame, "obj_id": obj, "bbox_xyxy": [50 + frame, 30, 49 + frame + size, 29 + size], "mask_area": size * size, "frame_width": 200, "frame_height": 100, "video_id": "input", "global_drone_id": f"drone_{obj}", "class_id": obj}
    result.update(extra)
    return result


def detect(records: list[dict[str, object]], params: IntervalParameters | None = None) -> list[dict[str, object]]:
    return detect_frame_anomalies(records, REGISTRY, CATALOG, params or IntervalParameters())


def test_threefold_bbox_growth_is_direct_critical() -> None:
    results = detect([row(i) for i in range(5)] + [row(5, 30)])
    last = results[-1]
    assert last["bbox_area_level"] == "critical"
    assert last["severity"] == "critical"
    assert "bbox_area_jump" in last["anomaly_types"]
    assert last["bbox_area_baseline"] == 100
    assert last["reliable"] is False


def test_threefold_bbox_shrink_is_direct_critical() -> None:
    results = detect([row(i, 30) for i in range(5)] + [row(5, 10)])
    assert results[-1]["bbox_area_level"] == "critical"


def test_two_high_area_frames_create_high_interval() -> None:
    results = detect([row(i) for i in range(5)] + [row(5, 14), row(6, 14)])
    intervals = build_intervals(results, IntervalParameters())
    assert len(intervals) == 1
    assert intervals[0]["severity"] == "high"
    assert intervals[0]["recommended_restart_frame"] == 4


def test_single_warning_remains_warning_interval() -> None:
    results = detect([row(i) for i in range(5)] + [row(5, 12)])
    intervals = build_intervals(results, IntervalParameters())
    assert intervals[0]["severity"] == "warning"


def test_stable_track_has_no_intervals_and_object_baselines_are_separate() -> None:
    results = detect([row(i, 10, 1) for i in range(5)] + [row(i, 30, 2) for i in range(5)] + [row(5, 10, 1), row(5, 30, 2)])
    assert build_intervals(results, IntervalParameters()) == []


def test_anomalous_area_does_not_pollute_baseline_and_recovery_resumes() -> None:
    results = detect([row(i) for i in range(5)] + [row(5, 30), row(6), row(7), row(8), row(9)])
    assert results[5]["bbox_area_baseline"] == 100
    assert results[6]["bbox_area_baseline"] == 100
    assert results[9]["reliable"] is True


def test_missing_duplicate_and_identity_are_reported() -> None:
    records = [row(0), row(2), row(2), row(3, global_drone_id="wrong")]
    results = {r["frame_idx"]: r for r in detect(records)}
    assert "missing_record" in results[1]["anomaly_types"]
    assert "duplicate_frame_idx" in results[2]["anomaly_types"]
    assert "identity_error" in results[3]["anomaly_types"]


def test_interval_id_is_stable_and_review_is_inherited_by_overlap() -> None:
    results = detect([row(i) for i in range(5)] + [row(5, 30)])
    new = build_intervals(results, IntervalParameters())
    assert new[0]["interval_id"] == "input_obj1_000005_000005"
    old = [{**new[0], "start_frame": 5, "end_frame": 6, "interval_id": "old", "status": "resolved", "reviewer_note": "done"}]
    stats = inherit_review_status(new, old)
    assert stats["overlap"] == 1 and new[0]["status"] == "resolved" and new[0]["reviewer_note"] == "done"


def test_status_update_resolved_requires_force() -> None:
    rows = [{"interval_id": "x", "status": "resolved", "obj_id": 1, "start_frame": 1, "end_frame": 2}]
    try:
        update_interval_status(rows, interval_id="x", obj_id=None, start_frame=None, end_frame=None, status="pending", reviewer_note=None, correction_tracks_path=None, actual_restart_frame=None, force=False)
    except ValueError as error:
        assert "requires --force" in str(error)
    else: assert False
    update_interval_status(rows, interval_id="x", obj_id=None, start_frame=None, end_frame=None, status="needs_restart", reviewer_note="a", correction_tracks_path="p", actual_restart_frame=0, force=True)
    assert rows[0]["status"] == "needs_restart" and rows[0]["actual_restart_frame"] == 0


def test_compare_interval_sets_reports_reduction() -> None:
    original = [{"severity": "critical", "max_bbox_area_jump_ratio": 3.0}, {"severity": "warning", "max_bbox_area_jump_ratio": 1.5}]
    final = [{"severity": "warning", "max_bbox_area_jump_ratio": 1.4}]
    comparison = compare_interval_sets(original, final)
    assert comparison["interval_reduction"] == 1
    assert comparison["high_critical_reduction"] == 1
