#!/usr/bin/env python3
"""Atomically update the manual-review fields of one anomaly interval."""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sam2_annotator.anomaly_intervals import REVIEW_STATUSES, read_jsonl, update_interval_status, write_jsonl_atomic
def main() -> None:
 p=argparse.ArgumentParser(description="Update status and review metadata for one anomaly interval."); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path); p.add_argument("--interval-id"); p.add_argument("--obj-id",type=int); p.add_argument("--start-frame",type=int); p.add_argument("--end-frame",type=int); p.add_argument("--status",choices=sorted(REVIEW_STATUSES)); p.add_argument("--reviewer-note"); p.add_argument("--correction-tracks-path"); p.add_argument("--actual-restart-frame",type=int); p.add_argument("--strict-correction-path",action="store_true"); p.add_argument("--force",action="store_true"); a=p.parse_args()
 if bool(a.interval_id)==bool(a.obj_id is not None and a.start_frame is not None and a.end_frame is not None): p.error("use --interval-id OR --obj-id with --start-frame and --end-frame")
 if a.correction_tracks_path and a.strict_correction_path and not Path(a.correction_tracks_path).expanduser().is_file(): p.error("correction tracks path does not exist")
 if a.correction_tracks_path and not Path(a.correction_tracks_path).expanduser().exists(): logging.warning("correction tracks path does not exist: %s",a.correction_tracks_path)
 inp=a.input.expanduser().resolve(); out=(a.output or inp).expanduser().resolve(); rows=read_jsonl(inp)
 row=update_interval_status(rows,interval_id=a.interval_id,obj_id=a.obj_id,start_frame=a.start_frame,end_frame=a.end_frame,status=a.status,reviewer_note=a.reviewer_note,correction_tracks_path=a.correction_tracks_path,actual_restart_frame=a.actual_restart_frame,force=a.force); write_jsonl_atomic(out,rows); print(row["interval_id"])
if __name__ == "__main__": main()
