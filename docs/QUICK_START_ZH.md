# 快速开始

## 推荐：Web UI

先按 [README](../README.md) 安装环境、可编辑 SAM2 和 tiny checkpoint，并检查 `configs/default.yaml`。在项目根目录执行：

```bash
conda activate sam2_env
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python app.py
```

打开终端打印的 URL（默认通常为 `http://127.0.0.1:7860`）。

1. “数据”：加载视频或 JPG/JPEG/PNG 连续图片目录。图片按文件名自然排序，不改原图，内部 mapping 保存来源与 canonical frame 的对应关系。
2. “标注与传播”：选择 obj1/obj2 和 global drone，选择开始/结束帧，浏览器画框并开始传播，查看初始 overlay。
3. 漂移时人工选择 restart frame、结束帧并重画 bbox，生成全新 SAM2 state 的硬重启候选；看 candidate preview 后点击“接受纠正”。可多次重启，timeline 仅写入当前 session。无需先运行异常检测。
4. “最终导出”：生成最终结果，查看 quality report 和 overlay。输出位于 `ui_workspace/sessions/<session_id>/final/`，包含对象 final tracks、merged tracks、YOLO labels、`quality_report.json`、`overlay.mp4`。

## CLI（保留）

在 `<PROJECT_ROOT>` 执行：

```bash
conda activate sam2_env
export PYTHONPATH=src:${PYTHONPATH}
python scripts/extract_frames.py --video data/videos/video_a.mp4 --frames-dir data/frames/video_a
python scripts/register_object.py --registry data/object_registry.json --catalog configs/drone_catalog.json --frames-dir data/frames/video_a --obj-id 1 --name obj1 --global-drone-id drone_1 --frame-idx 0 --prompt-output data/prompts/obj1.json --box 100 120 180 200
python scripts/create_tracking_segment.py --source-frames-dir data/frames/video_a --start-frame 0 --end-frame 300 --output-dir data/segments/obj1 --prompt data/prompts/obj1.json --output-prompt data/prompts/obj1_local.json
python scripts/track_registered_object.py --config configs/default.yaml --registry data/object_registry.json --obj-id 1 --segment-dir data/segments/obj1 --prompt data/prompts/obj1_local.json --output-dir outputs/video_a/obj1
```

人工观察结果，必要时生成硬重启候选，查看预览后再用 `accept_handoff.py` 显式接受；候选路径以 summary 为准。异常检测是可选 CLI 工具。准备好包含原始段与已接受纠正段的 timeline 后运行：

```bash
python scripts/finalize_video_annotation.py --timeline data/object_timelines.json --registry data/object_registry.json --catalog configs/drone_catalog.json --frames-dir data/frames/video_a --output-dir outputs/video_a/final --render-overlay
```

完整步骤见 [USER_GUIDE_ZH.md](USER_GUIDE_ZH.md)。
