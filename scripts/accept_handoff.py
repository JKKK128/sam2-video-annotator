#!/usr/bin/env python3
"""Explicitly accept one validated candidate range into exactly one timeline."""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from sam2_annotator.drone_catalog import load_catalog
from sam2_annotator.handoff import backup_and_write_timeline, find_interval, identity_for_object, replace_timeline_range, validate_candidate_records
from sam2_annotator.object_registry import load_registry
from sam2_annotator.tracking_io import load_jsonl

LOG=logging.getLogger("accept_handoff")
def accept(args: argparse.Namespace) -> dict:
    timeline_path=args.timeline.expanduser().resolve()
    if not timeline_path.is_file(): raise FileNotFoundError(f"timeline does not exist: {timeline_path}")
    interval=find_interval(args.intervals,args.interval_id) if args.intervals and args.interval_id else {}
    obj=args.obj_id if args.obj_id is not None else interval.get("obj_id")
    start=args.start_frame if args.start_frame is not None else interval.get("start_frame")
    end=args.end_frame if args.end_frame is not None else interval.get("end_frame")
    if not all(isinstance(x,int) for x in (obj,start,end)) or start<0 or end<start: raise ValueError("obj-id/start-frame/end-frame are required and must be valid")
    restart=args.restart_frame if args.restart_frame is not None else args.start_frame
    if not isinstance(restart,int) or restart<0 or restart>start: raise ValueError("restart-frame must be non-negative and no later than start-frame")
    registry=load_registry(args.registry); identity=identity_for_object(registry,load_catalog(args.catalog),obj)
    tracks=args.tracks.expanduser().resolve(); rows=load_jsonl(tracks); validate_candidate_records(rows,identity,start,end)
    inferred=next((r.get("source_type") or r.get("correction_mode") for r in rows if r.get("source_type") or r.get("correction_mode")),None)
    source_type=args.source_type or inferred
    if source_type not in {"auto_mask_handoff","auto_bbox_handoff","manual_hard_restart"}: raise ValueError("source-type must be an accepted handoff correction mode")
    current=json.loads(timeline_path.read_text(encoding="utf-8"))
    segment={"source_type":source_type,"restart_frame":restart,"tracks_path":str(tracks)}
    if args.review_note: segment["review_note"]=args.review_note
    updated=replace_timeline_range(current,obj,start,end,segment,force=args.force or args.overwrite_existing_range)
    backup=backup_and_write_timeline(timeline_path,updated,args.backup_dir)
    result={"obj_id":obj,"start_frame":start,"end_frame":end,"restart_frame":restart,"source_type":source_type,"backup":str(backup),"timeline":str(timeline_path)}
    LOG.info("accepted candidate. Next: python scripts/finalize_video_annotation.py --timeline %s",timeline_path); return result
def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Validate and explicitly accept a handoff candidate into one object's timeline; final exports are not run automatically.")
    p.add_argument("--timeline",required=True,type=Path);p.add_argument("--registry",required=True,type=Path);p.add_argument("--catalog",required=True,type=Path);p.add_argument("--tracks",required=True,type=Path);p.add_argument("--obj-id",type=int);p.add_argument("--start-frame",type=int);p.add_argument("--end-frame",type=int);p.add_argument("--restart-frame",type=int);p.add_argument("--source-type");p.add_argument("--review-note");p.add_argument("--intervals",type=Path);p.add_argument("--interval-id");p.add_argument("--overwrite-existing-range",action="store_true");p.add_argument("--force",action="store_true");p.add_argument("--backup-dir",type=Path);p.add_argument("--log-level",default="INFO");return p
def main()->int:
    a=parser().parse_args();logging.basicConfig(level=getattr(logging,a.log_level.upper(),logging.INFO),format="%(levelname)s: %(message)s");print(json.dumps(accept(a),ensure_ascii=False));return 0
if __name__=="__main__":raise SystemExit(main())
