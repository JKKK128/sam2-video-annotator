from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from sam2_annotator.handoff import replace_timeline_range, save_manual_prompt, select_prompt, validate_candidate_records

def _record(mask: str | None = None, box: list[int] | None = None) -> dict:
    return {"obj_id":2,"frame_idx":10,"mask_path":mask,"bbox_xyxy":box or [1,2,5,7]}
def test_auto_prefers_reliable_mask(tmp_path: Path)->None:
    mask=tmp_path/"mask.png"; cv2.imwrite(str(mask),np.ones((10,10),np.uint8)*255)
    got=select_prompt("auto",_record(str(mask)),tmp_path/"tracks.jsonl",10,10,{})
    assert got["selected"]=="mask"
def test_auto_falls_back_to_bbox(tmp_path: Path)->None:
    got=select_prompt("auto",_record(str(tmp_path/"missing.png")),tmp_path/"tracks.jsonl",10,10,{})
    assert got["selected"]=="bbox"
def test_auto_rejects_when_no_prompt(tmp_path: Path)->None:
    with pytest.raises(ValueError,match="manual-box"): select_prompt("auto",_record(str(tmp_path/"missing.png"),[1,1,1,1]),tmp_path/"x",10,10,{})
def test_forced_mask_and_bbox_reject_invalid(tmp_path: Path)->None:
    with pytest.raises(ValueError): select_prompt("mask",_record(str(tmp_path/"missing")),tmp_path/"x",10,10,{})
    with pytest.raises(ValueError): select_prompt("bbox",_record(None,[1,1,1,2]),tmp_path/"x",10,10,{})
def test_critical_auto_rejected(tmp_path: Path)->None:
    row=_record(None);row["severity"]="critical"
    with pytest.raises(ValueError): select_prompt("auto",row,tmp_path/"x",10,10,{})
def test_manual_prompt_and_box(tmp_path: Path)->None:
    identity={"obj_id":2,"global_drone_id":"d2","class_id":7,"class_name":"Drone"}
    prompt=save_manual_prompt(tmp_path/"manual.json",identity,"video",9,(1,2,4,5),10,10,"check")
    assert prompt["prompt_mode"]=="manual_box" and json.loads((tmp_path/"manual.json").read_text())["original_frame_idx"]==9
    assert select_prompt("manual-box",None,tmp_path/"x",10,10,{},(1,2,4,5))["selected"]=="manual-box"
    with pytest.raises(ValueError): select_prompt("manual-box",None,tmp_path/"x",10,10,{},None)
def test_candidate_validation_and_timeline_split()->None:
    identity={"obj_id":2,"global_drone_id":"d2","class_id":7,"class_name":"Drone"}
    rows=[{**identity,"frame_idx":i,"bbox_xyxy":[1,1,3,3],"frame_width":10,"frame_height":10,"mask_path":"x","mask_area":9} for i in range(3,6)]
    validate_candidate_records(rows,identity,3,5)
    timeline={"objects":[{"obj_id":2,"frame_range":[0,9],"segments":[{"start_frame":0,"end_frame":9,"source_type":"original","tracks_path":"old"}]}]}
    out=replace_timeline_range(timeline,2,3,5,{"source_type":"auto_mask_handoff","tracks_path":"new","restart_frame":2},force=False)
    assert [(x["start_frame"],x["end_frame"]) for x in out["objects"][0]["segments"]]==[(0,2),(3,5),(6,9)]
def test_timeline_refuses_accepted_overlap_without_force()->None:
    timeline={"objects":[{"obj_id":2,"frame_range":[0,9],"segments":[{"start_frame":0,"end_frame":9,"source_type":"hard_restart","tracks_path":"old"}]}]}
    with pytest.raises(ValueError): replace_timeline_range(timeline,2,3,5,{"source_type":"auto_bbox_handoff","tracks_path":"new"},force=False)
