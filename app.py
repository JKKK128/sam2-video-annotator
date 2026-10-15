"""Local Gradio UI for initial SAM2 annotation and propagation only."""
from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import gradio as gr

from sam2_annotator.drone_catalog import get_catalog_entry, load_catalog
from sam2_annotator.frame_utils import frame_path
from sam2_annotator.ui_bbox import BBoxXYXY, browser_rect_to_bbox, format_bbox, frame_range_after_start, next_frame, previous_frame
from sam2_annotator.ui_service import (
    SourceInfo, accept_manual_restart, prepare_ui_source, run_initial_tracking,
    run_final_export, run_manual_restart,
)

PROJECT_ROOT = Path(__file__).resolve().parent
CATALOG_PATH = PROJECT_ROOT / "configs" / "drone_catalog.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _frame_for(source: SourceInfo, frame_idx: int) -> str:
    path = frame_path(source.frames_dir, int(frame_idx))
    if not path.is_file():
        raise FileNotFoundError(f"frame does not exist: {path}")
    return str(path)


def _empty_bbox_state() -> dict[str, dict[str, Any] | None]:
    return {"1": None, "2": None}


def _empty_tracking_state() -> dict[str, dict[str, int] | None]:
    return {"1": None, "2": None}


def _bbox_for(state: dict[str, dict[str, Any] | None] | None, obj_id: int, frame_idx: int) -> BBoxXYXY | None:
    record = (state or {}).get(str(obj_id))
    if not isinstance(record, dict) or record.get("frame_idx") != int(frame_idx):
        return None
    value = record.get("bbox")
    if not isinstance(value, list) or len(value) != 4:
        return None
    return tuple(int(v) for v in value)  # type: ignore[return-value]


def _bbox_render(source: SourceInfo | None, bbox: BBoxXYXY | None) -> str:
    return "" if source is None or bbox is None else json.dumps({"bbox": list(bbox), "width": source.width, "height": source.height})


def _load_source(source_type: str, path: str, image_fps: float) -> tuple[SourceInfo | None, dict[str, dict[str, Any] | None], Any, Any, Any, str, str, str]:
    try:
        source = prepare_ui_source(source_type, path, image_fps=image_fps)
        start = gr.update(minimum=0, maximum=source.frame_count - 1, value=0)
        end = gr.update(minimum=0, maximum=source.frame_count - 1, value=source.frame_count - 1)
        details = f"已加载：{source.source_type}；{source.frame_count} 帧，{source.width}×{source.height}，{source.fps:g} fps。\nUI session: {source.session_dir}"
        return source, _empty_bbox_state(), start, end, gr.update(value=_frame_for(source, 0)), details, format_bbox(None), ""
    except Exception as exc:
        return None, _empty_bbox_state(), gr.update(), gr.update(), gr.update(), f"加载失败：{exc}", format_bbox(None), ""


def _tab_open(source: SourceInfo | None) -> tuple[Any, Any, Any, str, str]:
    if source is None:
        return gr.update(), gr.update(), gr.update(), format_bbox(None), ""
    return (gr.update(minimum=0, maximum=source.frame_count - 1, value=0), gr.update(minimum=0, maximum=source.frame_count - 1, value=source.frame_count - 1), gr.update(value=_frame_for(source, 0)), format_bbox(None), "")


def _set_start(source: SourceInfo | None, state: dict[str, dict[str, Any] | None] | None, obj_id: int, start_frame: int, end_frame: int) -> tuple[Any, Any, Any, str, str]:
    if source is None:
        return gr.update(), gr.update(), gr.update(), format_bbox(None), ""
    start, end = frame_range_after_start(start_frame, end_frame, source.frame_count)
    bbox = _bbox_for(state, int(obj_id), start)
    return gr.update(value=start), gr.update(value=end), gr.update(value=_frame_for(source, start)), format_bbox(bbox), _bbox_render(source, bbox)


def _step(direction: int, source: SourceInfo | None, state: dict[str, dict[str, Any] | None] | None, obj_id: int, start: int, end: int) -> tuple[Any, Any, Any, str, str]:
    if source is None:
        return _set_start(source, state, obj_id, start, end)
    new_start = previous_frame(start, source.frame_count) if direction < 0 else next_frame(start, source.frame_count)
    return _set_start(source, state, obj_id, new_start, end)


def _set_end(source: SourceInfo | None, start: int, end: int) -> tuple[Any, str]:
    if source is None:
        return gr.update(), "请先在“数据”页加载视频或图片序列。"
    _, normalized = frame_range_after_start(start, end, source.frame_count)
    return gr.update(value=normalized), "" if normalized == int(end) else "结束帧不能早于开始帧，已调整为开始帧。"


def _select_object(source: SourceInfo | None, state: dict[str, dict[str, Any] | None] | None, obj_id: int, start: int) -> tuple[str, str]:
    bbox = _bbox_for(state, int(obj_id), int(start))
    return format_bbox(bbox), _bbox_render(source, bbox)


def _validate_drone(drone_id: str) -> str:
    try:
        get_catalog_entry(load_catalog(CATALOG_PATH), drone_id)
    except Exception as exc:
        return f"Global drone 无效：{exc}"
    return ""


def _save_bbox(raw: str, source: SourceInfo | None, state: dict[str, dict[str, Any] | None] | None, obj_id: int, start: int) -> tuple[dict[str, dict[str, Any] | None], str, str, str]:
    new_state = dict(state or _empty_bbox_state())
    if source is None:
        return new_state, format_bbox(None), "", "请先加载数据。"
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("矩形选择数据无效")
        bbox = browser_rect_to_bbox(value["x"], value["y"], value["width"], value["height"], original_width=source.width, original_height=source.height, display_width=int(value["display_width"]), display_height=int(value["display_height"]))
        new_state[str(int(obj_id))] = {"frame_idx": int(start), "bbox": list(bbox)}
        return new_state, format_bbox(bbox), _bbox_render(source, bbox), ""
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        bbox = _bbox_for(new_state, int(obj_id), int(start))
        return new_state, format_bbox(bbox), _bbox_render(source, bbox), f"框选无效：{exc}"


def _clear_bbox(state: dict[str, dict[str, Any] | None] | None, obj_id: int) -> tuple[dict[str, dict[str, Any] | None], str, str, str]:
    new_state = dict(state or _empty_bbox_state())
    new_state[str(int(obj_id))] = None
    return new_state, format_bbox(None), "", "已清除当前对象的 bbox；请重新框选。"


def _start_tracking(source: SourceInfo | None, state: dict[str, dict[str, Any] | None] | None, obj_id: int, drone_id: str, start: int, end: int, tracking: dict[str, dict[str, int] | None] | None = None) -> Generator[tuple[Any, ...], None, None]:
    new_tracking = dict(tracking or _empty_tracking_state())
    if source is None:
        yield ("请先在“数据”页加载视频或图片序列。", None, gr.update(interactive=True), new_tracking) if tracking is not None else ("请先在“数据”页加载视频或图片序列。", None, gr.update(interactive=True))
        return
    try:
        valid_start, valid_end = frame_range_after_start(start, end, source.frame_count)
        if (valid_start, valid_end) != (int(start), int(end)):
            raise ValueError("开始帧或结束帧不在有效范围内")
        if int(obj_id) not in (1, 2):
            raise ValueError("对象必须是 obj1 或 obj2")
        get_catalog_entry(load_catalog(CATALOG_PATH), drone_id)
        bbox = _bbox_for(state, int(obj_id), valid_start)
        if bbox is None:
            raise ValueError("请先在起始帧框选目标。")
        yield ("正在传播，请稍候……", None, gr.update(interactive=False), new_tracking) if tracking is not None else ("正在传播，请稍候……", None, gr.update(interactive=False))
        result = run_initial_tracking(frames_dir=source.frames_dir, registry_path=source.registry_path, catalog_path=CATALOG_PATH, config_path=CONFIG_PATH, obj_id=int(obj_id), global_drone_id=drone_id, start_frame=valid_start, end_frame=valid_end, bbox_xyxy=bbox, session_dir=source.session_dir, fps=source.fps)
        new_tracking[str(result.obj_id)] = {"start": result.start_frame, "end": result.end_frame}
        yield (f"传播完成：obj{result.obj_id} / {result.global_drone_id}，帧 {result.start_frame}–{result.end_frame}。", str(result.overlay_path), gr.update(interactive=True), new_tracking) if tracking is not None else (f"传播完成：obj{result.obj_id} / {result.global_drone_id}，帧 {result.start_frame}–{result.end_frame}。", str(result.overlay_path), gr.update(interactive=True))
    except Exception as exc:
        yield (f"传播失败：{exc}", None, gr.update(interactive=True), new_tracking) if tracking is not None else (f"传播失败：{exc}", None, gr.update(interactive=True))


def _set_restart_frame(source: SourceInfo | None, state: dict[str, dict[str, Any] | None] | None, obj_id: int, restart: int, end: int) -> tuple[Any, Any, Any, str, str]:
    if source is None:
        return gr.update(), gr.update(), gr.update(), format_bbox(None), ""
    restart = max(0, min(int(restart), source.frame_count - 1))
    end = max(restart, min(int(end), source.frame_count - 1))
    bbox = _bbox_for(state, int(obj_id), restart)
    return (gr.update(minimum=0, maximum=source.frame_count - 1, value=restart),
            gr.update(minimum=0, maximum=source.frame_count - 1, value=end),
            gr.update(value=_frame_for(source, restart)), format_bbox(bbox), _bbox_render(source, bbox))


def _restart_object_changed(source: SourceInfo | None, tracking: dict[str, dict[str, int] | None] | None, state: dict[str, dict[str, Any] | None] | None, obj_id: int) -> tuple[Any, Any, Any, str, str]:
    if source is None:
        return gr.update(), gr.update(), gr.update(), format_bbox(None), ""
    prior = (tracking or {}).get(str(int(obj_id))) or {"start": 0, "end": source.frame_count - 1}
    restart, end = int(prior["start"]), int(prior["end"])
    bbox = _bbox_for(state, int(obj_id), restart)
    return (gr.update(minimum=0, maximum=source.frame_count - 1, value=restart),
            gr.update(minimum=0, maximum=source.frame_count - 1, value=end),
            gr.update(value=_frame_for(source, restart)), format_bbox(bbox), _bbox_render(source, bbox))


def _save_restart_bbox(raw: str, source: SourceInfo | None, state: dict[str, dict[str, Any] | None] | None, obj_id: int, restart: int) -> tuple[dict[str, dict[str, Any] | None], str, str, str]:
    return _save_bbox(raw, source, state, obj_id, restart)


def _run_restart(source: SourceInfo | None, state: dict[str, dict[str, Any] | None] | None, obj_id: int, restart: int, end: int) -> Generator[tuple[str, Any, Any, Any], None, None]:
    if source is None:
        yield "请先加载数据。", None, None, gr.update(interactive=True)
        return
    try:
        bbox = _bbox_for(state, int(obj_id), int(restart))
        if bbox is None:
            raise ValueError("请先在重启帧框选目标。")
        yield "正在创建独立的人工硬重启候选……", None, None, gr.update(interactive=False)
        result = run_manual_restart(frames_dir=source.frames_dir, registry_path=source.registry_path, catalog_path=CATALOG_PATH, config_path=CONFIG_PATH, session_dir=source.session_dir, obj_id=int(obj_id), restart_frame=int(restart), end_frame=int(end), bbox_xyxy=bbox)
        candidate = {"obj_id": result.obj_id, "restart": result.restart_frame, "end": result.end_frame, "tracks": str(result.candidate_tracks_path)}
        yield f"候选已生成：帧 {result.restart_frame}–{result.end_frame}。请检查视频后再接受。", str(result.preview_path), candidate, gr.update(interactive=True)
    except Exception as exc:
        yield f"生成候选失败：{exc}", None, None, gr.update(interactive=True)


def _accept_restart(source: SourceInfo | None, candidate: dict[str, Any] | None) -> tuple[str, Any, Any]:
    if source is None or not isinstance(candidate, dict):
        return "没有可接受的候选。", None, candidate
    try:
        result = accept_manual_restart(frames_dir=source.frames_dir, registry_path=source.registry_path, catalog_path=CATALOG_PATH, session_dir=source.session_dir, obj_id=int(candidate["obj_id"]), candidate_tracks_path=candidate["tracks"], restart_frame=int(candidate["restart"]), end_frame=int(candidate["end"]), fps=source.fps)
        return "纠正已接受，并已重建当前完整 Overlay。", str(result.current_overlay_path), None
    except Exception as exc:
        return f"接受纠正失败：{exc}", None, candidate


def _discard_candidate() -> tuple[str, None, None]:
    return "已放弃当前候选；候选文件保留在 session 中供调试。", None, None


def _format_final_summary(result: Any) -> str:
    """Present only fields emitted by the existing finalizer/report."""
    summary = result.summary
    rows = [
        "导出成功" if result.valid else "导出完成，但质量检查未通过",
        f"最终输出目录：{result.output_dir}",
        f"合并轨迹：{result.merged_tracks_path}",
        f"标签目录：{result.labels_dir}",
        f"质量报告：{result.quality_report_path}",
        f"最终视频：{result.overlay_path}",
    ]
    labels = {
        "active_objects": "对象数量",
        "total_tracks": "轨迹记录数",
        "label_file_count": "标签文件数",
        "total_label_lines": "YOLO 标签行数",
        "total_labels": "YOLO 标签行数",
        "duplicate_key_count": "重复记录数",
        "missing_label_files": "缺失标签帧数",
        "records_by_global_drone_id": "身份一致性",
        "records_by_class_id": "类别一致性",
    }
    seen: set[str] = set()
    for key, label in labels.items():
        if key in summary and label not in seen:
            value = summary[key]
            if key == "missing_label_files" and isinstance(value, list):
                value = len(value)
            rows.append(f"{label}：{value}")
            seen.add(label)
    rows.append(f"Quality validation：{'通过' if result.valid else '未通过'}")
    return "\n".join(rows)


def _run_final_export(source: SourceInfo | None) -> Generator[tuple[str, Any, Any], None, None]:
    if source is None:
        yield "请先在“数据”页加载视频或图片序列。", None, gr.update(interactive=True)
        return
    try:
        yield "正在生成最终结果，请稍候……", None, gr.update(interactive=False)
        result = run_final_export(session_dir=source.session_dir, catalog_path=CATALOG_PATH, render_overlay=True)
        yield _format_final_summary(result), str(result.overlay_path), gr.update(interactive=True)
    except Exception as exc:
        yield f"最终导出失败：{exc}", None, gr.update(interactive=True)


# Gradio 5.50 has no native rectangle-annotation value contract. This tiny SVG
# overlay uses the actual rendered <img> rectangle, then sends CSS-pixel x/y/w/h
# through hidden Gradio state. It is deliberately dependency-free.
RECTANGLE_OVERLAY_JS = r'''() => { if (window.__sam2Box) return; window.__sam2Box=1; let svg,box,down=false,start,old=''; const q=id=>document.querySelector(`#${id} textarea,#${id} input`), img=()=>document.querySelector('#bbox_image img'); const make=()=>{if(svg)return;svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.style.cssText='position:fixed;z-index:1000;cursor:crosshair;touch-action:none';box=document.createElementNS('http://www.w3.org/2000/svg','rect');box.setAttribute('fill','transparent');box.setAttribute('stroke','#00d4ff');box.setAttribute('stroke-width','2');svg.append(box);document.body.append(svg);const pt=e=>{let b=svg.getBoundingClientRect();return{x:Math.max(0,Math.min(b.width,e.clientX-b.left)),y:Math.max(0,Math.min(b.height,e.clientY-b.top))}}, draw=(a,b)=>{box.style.display='';box.setAttribute('x',Math.min(a.x,b.x));box.setAttribute('y',Math.min(a.y,b.y));box.setAttribute('width',Math.abs(a.x-b.x));box.setAttribute('height',Math.abs(a.y-b.y))};svg.onpointerdown=e=>{start=pt(e);down=true;svg.setPointerCapture(e.pointerId);draw(start,start);e.preventDefault()};svg.onpointermove=e=>{if(down)draw(start,pt(e))};svg.onpointerup=e=>{if(!down)return;let end=pt(e),b=svg.getBoundingClientRect(),w=Math.abs(end.x-start.x),h=Math.abs(end.y-start.y);down=false;draw(start,end);let input=q('bbox_payload');if(input&&w>0&&h>0){input.value=JSON.stringify({x:Math.min(start.x,end.x),y:Math.min(start.y,end.y),width:w,height:h,display_width:b.width,display_height:b.height});input.dispatchEvent(new Event('input',{bubbles:true}));input.dispatchEvent(new Event('change',{bubbles:true}))}}}; const sync=()=>{let im=img();if(!im||!im.complete)return;make();let b=im.getBoundingClientRect();svg.style.left=b.left+'px';svg.style.top=b.top+'px';svg.style.width=b.width+'px';svg.style.height=b.height+'px';svg.setAttribute('viewBox',`0 0 ${b.width} ${b.height}`);let v=q('bbox_render')?.value||'';if(!down&&v!==old){old=v;box.style.display='none';try{let d=JSON.parse(v),a=d.bbox;box.style.display='';box.setAttribute('x',a[0]*b.width/d.width);box.setAttribute('y',a[1]*b.height/d.height);box.setAttribute('width',(a[2]-a[0]+1)*b.width/d.width);box.setAttribute('height',(a[3]-a[1]+1)*b.height/d.height)}catch(_){}}};addEventListener('resize',sync);addEventListener('scroll',sync,true);setInterval(sync,100) }'''
RESTART_RECTANGLE_OVERLAY_JS = RECTANGLE_OVERLAY_JS.replace("__sam2Box", "__sam2RestartBox").replace("bbox_image", "restart_bbox_image").replace("bbox_payload", "restart_bbox_payload").replace("bbox_render", "restart_bbox_render")


def build_app() -> gr.Blocks:
    drone_ids = list(load_catalog(CATALOG_PATH).keys())
    with gr.Blocks(title="SAM2 UAV 标注 UI V1") as demo:
        source_state, bbox_state = gr.State(None), gr.State(_empty_bbox_state())
        tracking_state, restart_bbox_state, candidate_state = gr.State(_empty_tracking_state()), gr.State(_empty_bbox_state()), gr.State(None)
        gr.Markdown("# SAM2 UAV 标注 UI V1\n数据加载、人工纠正与最终导出。")
        with gr.Tab("数据"):
            source_type = gr.Radio(choices=[("视频", "video"), ("连续图片目录", "image_sequence")], value="video", label="数据类型")
            source_path, image_fps = gr.Textbox(label="视频文件或图片目录路径"), gr.Number(value=10.0, minimum=0.001, label="图片序列 FPS")
            load_button, source_details = gr.Button("加载数据", variant="primary"), gr.Textbox(label="数据状态", lines=3, interactive=False)
        with gr.Tab("标注与传播") as annotation_tab:
            obj_id = gr.Radio(choices=[("obj1", 1), ("obj2", 2)], value=1, label="Object")
            drone_id = gr.Dropdown(choices=drone_ids, value=drone_ids[0], label="Global drone")
            with gr.Row():
                previous_button = gr.Button("上一帧")
                start_frame = gr.Slider(0, 0, 0, step=1, label="当前 / 开始帧")
                next_button = gr.Button("下一帧")
            end_frame = gr.Slider(0, 0, 0, step=1, label="传播结束帧")
            gr.Markdown("在图片上按住鼠标左键拖动以框选矩形；重新拖动会覆盖当前对象的旧框。")
            frame_image = gr.Image(label="当前帧 bbox", type="filepath", interactive=False, elem_id="bbox_image")
            bbox_details = gr.Textbox(label="当前 bbox（原始图像坐标）", lines=7, interactive=False, value=format_bbox(None))
            clear_button = gr.Button("清除框 / 重新框选")
            bbox_payload, bbox_render = gr.Textbox(visible="hidden", elem_id="bbox_payload"), gr.Textbox(visible="hidden", elem_id="bbox_render")
            start_button = gr.Button("开始传播", variant="primary")
            tracking_status, overlay = gr.Textbox(label="传播状态", interactive=False), gr.Video(label="初始传播 Overlay", interactive=False)
            gr.Markdown("## 人工硬重启\n观看初始或当前 Overlay 后，选择发生漂移的原始帧，重新画框并生成候选。")
            restart_obj = gr.Radio(choices=[("obj1", 1), ("obj2", 2)], value=1, label="对象")
            restart_frame = gr.Slider(0, 0, 0, step=1, label="重启帧")
            restart_end = gr.Slider(0, 0, 0, step=1, label="传播结束帧")
            restart_image = gr.Image(label="重启帧原始图像", type="filepath", interactive=False, elem_id="restart_bbox_image")
            restart_details = gr.Textbox(label="当前 bbox（原始图像坐标）", lines=7, interactive=False, value=format_bbox(None))
            restart_clear = gr.Button("清除框")
            restart_payload, restart_render = gr.Textbox(visible="hidden", elem_id="restart_bbox_payload"), gr.Textbox(visible="hidden", elem_id="restart_bbox_render")
            restart_button = gr.Button("生成硬重启候选", variant="primary")
            restart_status, candidate_video = gr.Textbox(label="人工硬重启状态", interactive=False), gr.Video(label="候选视频", interactive=False)
            accept_button, discard_button = gr.Button("接受纠正", variant="primary"), gr.Button("放弃候选")
            current_overlay = gr.Video(label="当前已接受完整 Overlay", interactive=False)
        with gr.Tab("最终导出"):
            gr.Markdown("完成需要的初始传播和人工纠正后，生成 session 内的最终多目标结果。")
            final_button = gr.Button("生成最终结果", variant="primary")
            final_status = gr.Textbox(label="最终导出结果", lines=14, interactive=False)
            final_overlay = gr.Video(label="最终 Overlay（merged tracks）", interactive=False)
        load_button.click(_load_source, [source_type, source_path, image_fps], [source_state, bbox_state, start_frame, end_frame, frame_image, source_details, bbox_details, bbox_render])
        annotation_tab.select(_tab_open, [source_state], [start_frame, end_frame, frame_image, bbox_details, bbox_render])
        obj_id.change(_select_object, [source_state, bbox_state, obj_id, start_frame], [bbox_details, bbox_render])
        drone_id.change(_validate_drone, [drone_id], [tracking_status])
        start_frame.change(_set_start, [source_state, bbox_state, obj_id, start_frame, end_frame], [start_frame, end_frame, frame_image, bbox_details, bbox_render])
        previous_button.click(lambda *x: _step(-1, *x), [source_state, bbox_state, obj_id, start_frame, end_frame], [start_frame, end_frame, frame_image, bbox_details, bbox_render])
        next_button.click(lambda *x: _step(1, *x), [source_state, bbox_state, obj_id, start_frame, end_frame], [start_frame, end_frame, frame_image, bbox_details, bbox_render])
        end_frame.change(_set_end, [source_state, start_frame, end_frame], [end_frame, tracking_status])
        bbox_payload.input(_save_bbox, [bbox_payload, source_state, bbox_state, obj_id, start_frame], [bbox_state, bbox_details, bbox_render, tracking_status])
        clear_button.click(_clear_bbox, [bbox_state, obj_id], [bbox_state, bbox_details, bbox_render, tracking_status])
        initial_event = start_button.click(_start_tracking, [source_state, bbox_state, obj_id, drone_id, start_frame, end_frame, tracking_state], [tracking_status, overlay, start_button, tracking_state], trigger_mode="once", concurrency_limit=1, concurrency_id="initial_tracking")
        initial_event.then(_restart_object_changed, [source_state, tracking_state, restart_bbox_state, restart_obj], [restart_frame, restart_end, restart_image, restart_details, restart_render])
        restart_obj.change(_restart_object_changed, [source_state, tracking_state, restart_bbox_state, restart_obj], [restart_frame, restart_end, restart_image, restart_details, restart_render])
        restart_frame.change(_set_restart_frame, [source_state, restart_bbox_state, restart_obj, restart_frame, restart_end], [restart_frame, restart_end, restart_image, restart_details, restart_render])
        restart_payload.input(_save_restart_bbox, [restart_payload, source_state, restart_bbox_state, restart_obj, restart_frame], [restart_bbox_state, restart_details, restart_render, restart_status])
        restart_clear.click(_clear_bbox, [restart_bbox_state, restart_obj], [restart_bbox_state, restart_details, restart_render, restart_status])
        restart_button.click(_run_restart, [source_state, restart_bbox_state, restart_obj, restart_frame, restart_end], [restart_status, candidate_video, candidate_state, restart_button], trigger_mode="once", concurrency_limit=1, concurrency_id="manual_restart")
        accept_button.click(_accept_restart, [source_state, candidate_state], [restart_status, current_overlay, candidate_state], concurrency_limit=1, concurrency_id="manual_restart")
        discard_button.click(_discard_candidate, [], [restart_status, candidate_video, candidate_state])
        final_button.click(_run_final_export, [source_state], [final_status, final_overlay, final_button], trigger_mode="once", concurrency_limit=1, concurrency_id="final_export")
        demo.load(None, js=RECTANGLE_OVERLAY_JS)
        demo.load(None, js=RESTART_RECTANGLE_OVERLAY_JS)
    return demo


if __name__ == "__main__":
    build_app().queue().launch()
