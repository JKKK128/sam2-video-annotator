#!/usr/bin/env python3
"""Create a review-only independent SAM2 handoff candidate; never edits timelines."""
from __future__ import annotations
import argparse, json, logging, os, shutil, sys, tempfile
from pathlib import Path
from typing import Any
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam2_annotator.config_utils import load_yaml_config, nested_get
from sam2_annotator.drone_catalog import load_catalog
from sam2_annotator.frame_utils import frame_path
from sam2_annotator.handoff import CORRECTION_MODES, find_interval, identity_for_object, save_manual_prompt, select_prompt, source_record, utc_now, validate_bbox, validate_candidate_records, validate_mask
from sam2_annotator.object_registry import load_registry
from sam2_annotator.tracking_io import load_jsonl
from create_tracking_segment import create_tracking_segment
from track_registered_object import track_registered_object

LOG=logging.getLogger("create_handoff")

def _video_id(registry: dict[str, Any]) -> str: return str(registry.get("video_id") or Path(str(registry["video"])).stem)
def _roi(image: Any, name: str) -> tuple[int,int,int,int] | None:
    roi=cv2.selectROI(name,image,showCrosshair=True,fromCenter=False); cv2.destroyWindow(name)
    x,y,w,h=(int(v) for v in roi)
    return None if w<=0 or h<=0 else (x,y,x+w-1,y+h-1)
def _preview(frames: Path, tracks: Path, output: Path, source: int, mode: str, identity: dict[str,Any], context: int) -> None:
    rows={int(r["frame_idx"]):r for r in load_jsonl(tracks)}; indices=sorted(rows); start=max(indices[0],source-context); end=min(indices[-1],indices[-1]+context)
    first=cv2.imread(str(frame_path(frames,start))); 
    if first is None: raise OSError("cannot read preview frame")
    meta=frames/"metadata.json"; fps=float(json.loads(meta.read_text()).get("fps",30)) if meta.is_file() else 30.0
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"mp4v"),fps,(first.shape[1],first.shape[0]))
    if not writer.isOpened(): raise OSError(f"cannot create preview: {output}")
    try:
        label={"mask":"AUTO MASK HANDOFF","bbox":"AUTO BBOX HANDOFF","manual-box":"MANUAL HARD RESTART"}[mode]
        for idx in range(start,end+1):
            image=cv2.imread(str(frame_path(frames,idx))); row=rows.get(idx)
            if image is None: raise OSError(f"cannot read preview frame {idx}")
            if row and row.get("bbox_xyxy"):
                x1,y1,x2,y2=map(int,row["bbox_xyxy"]); cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            text=f"{label} | obj {identity['obj_id']} {identity['global_drone_id']} class {identity['class_id']} | frame {idx} source {source}"
            cv2.putText(image,text,(10,28),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3); cv2.putText(image,text,(10,28),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1); writer.write(image)
    finally: writer.release()
    if not output.is_file() or output.stat().st_size==0: raise OSError("preview video was not written")

def create_handoff(args: argparse.Namespace, manual_box: tuple[int, int, int, int] | None = None) -> dict[str, Any]:
    """Create a candidate, optionally using a caller-supplied manual bbox.

    The optional argument is deliberately an API extension: command-line
    callers retain the original OpenCV ROI workflow when it is omitted.
    """
    config=load_yaml_config(args.config); settings=nested_get(config,("handoff",),{}) or {}
    registry=load_registry(args.registry); catalog=load_catalog(args.catalog)
    interval=find_interval(args.intervals,args.interval_id) if args.intervals and args.interval_id else {}
    obj_id=args.obj_id if args.obj_id is not None else interval.get("obj_id")
    if not isinstance(obj_id,int): raise ValueError("--obj-id is required unless --intervals and --interval-id provide it")
    identity=identity_for_object(registry,catalog,obj_id)
    source=args.source_frame if args.source_frame is not None else interval.get("recommended_restart_frame")
    end=args.end_frame if args.end_frame is not None else interval.get("end_frame")
    if not isinstance(source,int) or not isinstance(end,int) or source<0 or end<source: raise ValueError("source-frame/end-frame must be non-negative and source <= end")
    image_path=frame_path(args.frames_dir.expanduser().resolve(),source); image=cv2.imread(str(image_path))
    if image is None: raise FileNotFoundError(f"cannot read source frame: {image_path}")
    h,w=image.shape[:2]; record=None if args.prompt_mode=="manual-box" else source_record(args.tracks,obj_id,source)
    if record is not None:
        for key in ("global_drone_id","class_id","class_name"):
            if record.get(key) not in (None,identity[key]):
                raise ValueError(f"source track {key} disagrees with registry/catalog for obj_id {obj_id}")
    mask_reliable=bbox_reliable=False
    if record is not None:
        try: validate_mask(record,args.tracks.resolve(),w,h,int(settings.get("min_mask_area",1)),reject_critical=bool(settings.get("reject_critical_source_frame",True))); mask_reliable=True
        except ValueError: pass
        try: validate_bbox(record,w,h,int(settings.get("min_bbox_area",1)),reject_critical=bool(settings.get("reject_critical_source_frame",True))); bbox_reliable=True
        except ValueError: pass
    box = manual_box
    if args.prompt_mode == "manual-box" and box is None:
        box = _roi(image, str(settings.get("manual_box", {}).get("window_name", "Select restart bbox")))
    selected=select_prompt(args.prompt_mode,record,args.tracks.resolve(),w,h,settings,box)
    output=args.output_dir.expanduser().resolve()
    if output.exists() and not args.overwrite: raise FileExistsError(f"output exists; pass --overwrite: {output}")
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix=f".{output.name}.",dir=output.parent)); manual_path: Path|None=None
    try:
        if selected["selected"]=="manual-box":
            manual_path=temporary/"manual_prompt.json"; save_manual_prompt(manual_path,identity,_video_id(registry),source,selected["box"],w,h,args.reason)
        prompt={"obj_id":obj_id,"object_name":f"obj{obj_id}","frame_idx":0,"local_frame_idx":0,"original_frame_idx":source,"prompt_type":"mask" if selected["selected"]=="mask" else "box","frame_width":w,"frame_height":h}
        if selected["selected"]=="mask":
            # Use a copied prompt mask so the candidate is self-contained.
            mask_dst=temporary/"prompt_mask.png"; shutil.copy2(selected["mask_path"],mask_dst); prompt["mask_path"]=str(mask_dst)
        else: prompt["box_xyxy"]=list(selected["box"])
        prompt_path=temporary/"prompt.json"; prompt_path.write_text(json.dumps(prompt,indent=2)+"\n")
        segment=temporary/"segment"; create_tracking_segment(args.frames_dir,source,end,segment,None,None,False)
        result=track_registered_object(args.config,args.registry,obj_id,segment,prompt_path,temporary/"run",end-source+1)
        tracks=temporary/"run"/"tracks.jsonl"; rows=load_jsonl(tracks)
        correction=CORRECTION_MODES[selected["selected"]]
        for row in rows:
            row.update(identity); row.update({"correction_mode":correction,"handoff_source_frame":source,"handoff_prompt_mode":selected["selected"],"handoff_source_tracks":str(args.tracks.resolve()),"manual_prompt_path":str(manual_path) if manual_path else None,"source_type":correction})
        tracks.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows))
        validate_candidate_records(rows,identity,source,end)
        # Expose conventional candidate paths rather than tracker implementation paths.
        final_tracks=temporary/"tracks.jsonl"; shutil.move(str(tracks),final_tracks); shutil.move(str(temporary/"run"/"masks"),str(temporary/"masks")); shutil.rmtree(temporary/"run"); shutil.rmtree(segment)
        final_manual=None
        if manual_path:
            final_manual=temporary/f"manual_frame_{source:06d}_box.json"; shutil.move(str(manual_path),final_manual)
            data=json.loads(final_manual.read_text()); data["prompt_path"]=str(final_manual); final_manual.write_text(json.dumps(data,indent=2)+"\n")
        preview=temporary/"preview.mp4" if args.render_preview else None
        if preview: _preview(args.frames_dir,final_tracks,preview,source,selected["selected"],identity,int(settings.get("preview_context_frames",2)))
        summary={"video_id":_video_id(registry),**identity,"source_frame":source,"end_frame":end,"selected_prompt_mode":selected["selected"],"requested_prompt_mode":args.prompt_mode,"mask_available":bool(record and record.get("mask_path")),"mask_reliable":mask_reliable,"bbox_available":bool(record and record.get("bbox_xyxy")),"bbox_reliable":bbox_reliable,"source_tracks":str(args.tracks.resolve()),"manual_prompt_path":str(final_manual) if final_manual else None,"correction_mode":correction,"output_record_count":len(rows),"missing_frames":[],"duplicate_frames":[],"tracks_path":str(output/"tracks.jsonl"),"masks_dir":str(output/"masks"),"preview_path":str(output/"preview.mp4") if preview else None,"created_at":utc_now()}
        (temporary/"handoff_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
        if output.exists(): shutil.rmtree(output)
        os.replace(temporary,output); LOG.info("candidate created: %s",output); return summary
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True); raise

def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Create a review-only fresh SAM2 handoff candidate; it never changes final tracks or timelines.")
    for n in ("tracks","frames-dir","registry","catalog","config","output-dir"): p.add_argument("--"+n,required=True,type=Path)
    p.add_argument("--obj-id",type=int); p.add_argument("--source-frame",type=int); p.add_argument("--end-frame",type=int); p.add_argument("--prompt-mode",required=True,choices=("auto","mask","bbox","manual-box")); p.add_argument("--intervals",type=Path); p.add_argument("--interval-id"); p.add_argument("--render-preview",action="store_true"); p.add_argument("--overwrite",action="store_true"); p.add_argument("--reason"); p.add_argument("--log-level",default="INFO"); return p
def main() -> int:
    a=parser().parse_args(); logging.basicConfig(level=getattr(logging,a.log_level.upper(),logging.INFO),format="%(levelname)s: %(message)s"); s=create_handoff(a); print(f"Review candidate then accept with: python scripts/accept_handoff.py --obj-id {s['obj_id']} --start-frame {s['source_frame']} --end-frame {s['end_frame']} --restart-frame {s['source_frame']} --tracks {s['tracks_path']}"); return 0
if __name__=="__main__": raise SystemExit(main())
