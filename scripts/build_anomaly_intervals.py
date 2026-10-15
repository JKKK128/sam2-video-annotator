#!/usr/bin/env python3
"""Build per-object review intervals from SAM2 track JSONL (review only)."""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sam2_annotator.anomaly_intervals import compare_interval_sets, detect_frame_anomalies, build_intervals, inherit_review_status, parameters_from_config, read_jsonl, write_jsonl_atomic, write_review_csv
from sam2_annotator.config_utils import load_yaml_config
from sam2_annotator.drone_catalog import load_catalog
from sam2_annotator.object_registry import load_registry

def main() -> None:
    p=argparse.ArgumentParser(description="Generate stable, per-object anomaly intervals for manual review.")
    p.add_argument("--tracks", type=Path, required=True); p.add_argument("--registry", type=Path, required=True); p.add_argument("--catalog", type=Path, required=True); p.add_argument("--frame-anomalies", type=Path); p.add_argument("--config", type=Path, default=Path("configs/default.yaml")); p.add_argument("--output", type=Path, required=True); p.add_argument("--review-csv", type=Path); p.add_argument("--compare-to", type=Path, help="Existing original interval JSONL to compare with this result."); p.add_argument("--comparison-output", type=Path, help="Optional JSON path for comparison summary.")
    for name in ("min-consecutive", "merge-gap", "recovery-frames", "restart-lookback", "baseline-window"): p.add_argument("--"+name, type=int)
    for name in ("bbox-area-warning-ratio", "bbox-area-high-ratio", "bbox-area-critical-ratio", "bbox-area-weight"): p.add_argument("--"+name, type=float)
    p.add_argument("--overwrite", action="store_true")
    a=p.parse_args(); a.output=a.output.expanduser().resolve()
    if a.output.exists() and not a.overwrite: p.error(f"output exists (use --overwrite): {a.output}")
    for path in (a.tracks,a.registry,a.catalog,a.config):
        if not path.expanduser().is_file(): p.error(f"file does not exist: {path}")
    overrides={key:getattr(a,key) for key in ("min_consecutive","merge_gap","recovery_frames","restart_lookback","baseline_window","bbox_area_warning_ratio","bbox_area_high_ratio","bbox_area_critical_ratio","bbox_area_weight")}
    params=parameters_from_config(load_yaml_config(a.config), overrides)
    records=read_jsonl(a.tracks.expanduser().resolve())
    external={}
    if a.frame_anomalies:
        for row in read_jsonl(a.frame_anomalies.expanduser().resolve()): external[(int(row["obj_id"]),int(row["frame_idx"]))]=list(row.get("anomaly_types",row.get("reasons",[])))
    old=read_jsonl(a.output)
    frames=detect_frame_anomalies(records,load_registry(a.registry),load_catalog(a.catalog),params,external)
    intervals=build_intervals(frames,params); stats=inherit_review_status(intervals,old) if old else {"new":len(intervals),"exact":0,"overlap":0,"resolved":0,"unmatched_old":0}
    write_jsonl_atomic(a.output,intervals)
    if a.review_csv: write_review_csv(a.review_csv.expanduser().resolve(),intervals)
    if a.compare_to:
        comparison = compare_interval_sets(read_jsonl(a.compare_to.expanduser().resolve()), intervals)
        logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")
        logging.info("comparison original=%d final=%d interval_reduction=%d high_critical_reduction=%d", comparison["original"]["interval_count"], comparison["final"]["interval_count"], comparison["interval_reduction"], comparison["high_critical_reduction"])
        if a.comparison_output:
            a.comparison_output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            a.comparison_output.expanduser().resolve().write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s"); logging.info("intervals=%d new=%d exact_inherited=%d overlap_inherited=%d resolved=%d unmatched_old=%d",len(intervals),stats["new"],stats["exact"],stats["overlap"],stats["resolved"],stats["unmatched_old"])
if __name__ == "__main__": main()
