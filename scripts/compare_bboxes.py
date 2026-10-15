#!/usr/bin/env python3
"""Compare raw and stable bbox jitter from a tracks.jsonl file."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
from typing import Any


def load_tracks(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"tracks JSONL does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "frame_idx" not in record or "obj_id" not in record:
                raise ValueError(f"tracks line {line_no} missing frame_idx or obj_id")
            records.append(record)
    if not records:
        raise ValueError(f"tracks JSONL has no records: {path}")
    return sorted(records, key=lambda item: (int(item["obj_id"]), int(item["frame_idx"])))


def bbox_metrics(bbox: object) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    if bbox is None:
        return None, None, None, None, None
    x1, y1, x2, y2 = [float(v) for v in bbox]  # type: ignore[union-attr]
    if x2 < x1 or y2 < y1:
        return None, None, None, None, None
    width = x2 - x1 + 1.0
    height = y2 - y1 + 1.0
    return width, height, width * height, (x1 + x2) / 2.0, (y1 + y2) / 2.0


def displacement(points: list[tuple[float | None, float | None]]) -> list[float]:
    values: list[float] = []
    previous: tuple[float, float] | None = None
    for x, y in points:
        if x is None or y is None:
            previous = None
            continue
        current = (float(x), float(y))
        if previous is not None:
            values.append(((current[0] - previous[0]) ** 2 + (current[1] - previous[1]) ** 2) ** 0.5)
        previous = current
    return values


def stdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if values else 0.0


def median_value(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = statistics.fmean(values)
    return float(stdev(values) / mean) if mean > 0 else 0.0


def compare_bboxes(tracks_path: Path, output_dir: Path) -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load_tracks(tracks_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "bbox_comparison.csv"
    summary_path = output_dir / "bbox_stability_summary.json"

    rows: list[dict[str, Any]] = []
    for record in records:
        raw = bbox_metrics(record.get("raw_bbox_xyxy") or record.get("bbox_xyxy"))
        stable = bbox_metrics(record.get("stable_bbox_xyxy") or record.get("bbox_xyxy"))
        rows.append(
            {
                "frame_idx": int(record["frame_idx"]),
                "obj_id": int(record["obj_id"]),
                "raw_width": raw[0],
                "raw_height": raw[1],
                "raw_area": raw[2],
                "raw_center_x": raw[3],
                "raw_center_y": raw[4],
                "stable_width": stable[0],
                "stable_height": stable[1],
                "stable_area": stable[2],
                "stable_center_x": stable[3],
                "stable_center_y": stable[4],
                "bbox_status": record.get("bbox_status", ""),
                "bbox_stabilized": bool(record.get("bbox_stabilized", False)),
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_obj: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_obj.setdefault(int(row["obj_id"]), []).append(row)
    per_obj = {str(obj_id): summarize_rows(obj_rows) for obj_id, obj_rows in sorted(by_obj.items())}
    summary = {"overall": summarize_rows(rows), "by_obj_id": per_obj}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    frames = [row["frame_idx"] for row in rows]
    plt.figure(figsize=(10, 5))
    plt.plot(frames, [row["raw_width"] for row in rows], label="raw width", linewidth=1)
    plt.plot(frames, [row["stable_width"] for row in rows], label="stable width", linewidth=1)
    plt.plot(frames, [row["raw_height"] for row in rows], label="raw height", linewidth=1)
    plt.plot(frames, [row["stable_height"] for row in rows], label="stable height", linewidth=1)
    plt.xlabel("frame_idx")
    plt.ylabel("pixels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bbox_size_plot.png")
    plt.close()

    raw_steps = grouped_displacements(rows, "raw_center_x", "raw_center_y")
    stable_steps = grouped_displacements(rows, "stable_center_x", "stable_center_y")
    raw_step_frames = list(range(1, 1 + len(raw_steps)))
    stable_step_frames = list(range(1, 1 + len(stable_steps)))
    plt.figure(figsize=(10, 5))
    plt.plot(raw_step_frames, raw_steps, label="raw center displacement", linewidth=1)
    plt.plot(stable_step_frames, stable_steps, label="stable center displacement", linewidth=1)
    plt.xlabel("step")
    plt.ylabel("pixels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bbox_center_displacement_plot.png")
    plt.close()
    return summary


def _nonnull(values: object) -> list[float]:
    return [float(value) for value in values if value is not None]  # type: ignore[union-attr]


def grouped_displacements(rows: list[dict[str, Any]], x_key: str, y_key: str) -> list[float]:
    values: list[float] = []
    by_obj: dict[int, list[tuple[float | None, float | None]]] = {}
    for row in rows:
        by_obj.setdefault(int(row["obj_id"]), []).append((row[x_key], row[y_key]))
    for points in by_obj.values():
        values.extend(displacement(points))
    return values


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_widths = _nonnull(row["raw_width"] for row in rows)
    raw_heights = _nonnull(row["raw_height"] for row in rows)
    raw_areas = _nonnull(row["raw_area"] for row in rows)
    stable_widths = _nonnull(row["stable_width"] for row in rows)
    stable_heights = _nonnull(row["stable_height"] for row in rows)
    stable_areas = _nonnull(row["stable_area"] for row in rows)
    raw_steps = grouped_displacements(rows, "raw_center_x", "raw_center_y")
    stable_steps = grouped_displacements(rows, "stable_center_x", "stable_center_y")
    statuses = [str(row["bbox_status"]) for row in rows]
    return {
        "record_count": len(rows),
        "raw_width_std": stdev(raw_widths),
        "stable_width_std": stdev(stable_widths),
        "raw_height_std": stdev(raw_heights),
        "stable_height_std": stdev(stable_heights),
        "raw_area_mean": float(statistics.fmean(raw_areas)) if raw_areas else 0.0,
        "stable_area_mean": float(statistics.fmean(stable_areas)) if stable_areas else 0.0,
        "raw_area_std": stdev(raw_areas),
        "stable_area_std": stdev(stable_areas),
        "raw_area_cv": coefficient_of_variation(raw_areas),
        "stable_area_cv": coefficient_of_variation(stable_areas),
        "raw_center_displacement_median": median_value(raw_steps),
        "stable_center_displacement_median": median_value(stable_steps),
        "raw_center_displacement_max": max(raw_steps) if raw_steps else 0.0,
        "stable_center_displacement_max": max(stable_steps) if stable_steps else 0.0,
        "initialized_count": statuses.count("initialized"),
        "stable_count": statuses.count("stable"),
        "smoothed_count": statuses.count("smoothed"),
        "unstable_count": statuses.count("unstable"),
        "missing_count": statuses.count("missing"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare raw and stable bbox size and center jitter from tracks.jsonl.")
    parser.add_argument("--tracks", required=True, type=Path, help="tracks.jsonl with raw_bbox_xyxy and stable_bbox_xyxy fields.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for CSV, JSON summary and plots.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    compare_bboxes(args.tracks, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
