# 中文用户手册

## 1. 适用场景与边界

本工具用于人工给出初始框后，借助 SAM2 对无人机视频或连续图片序列做半自动标注。当前以 Gradio Web UI 为主入口，保留 CLI。它适合逐视频、最多两架无人机的审慎标注；不做自动发现、YOLO 检测或自动接受纠正。所有帧使用内部原始索引和六位文件名（如 `000257.jpg`）。

### 推荐：Web UI 主流程

完成下节环境和 SAM2 安装后，在项目根目录执行：

```bash
conda activate sam2_env
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python app.py
```

打开终端打印的地址（Gradio 默认通常为 `http://127.0.0.1:7860`，端口占用时顺延）。在“数据”页选择视频或连续图片目录，填写 Python 进程可访问的路径，点击“加载数据”。图片序列支持 JPG/JPEG/PNG，按文件名自然排序，不修改原图；内部 canonical mapping 保存来源文件与零填充帧名的对应关系，帧号是排序后的索引，不直接沿用文件名中的数字。图片序列 FPS 用于播放。

在“标注与传播”选择 obj1/obj2 及 global drone，设置开始/结束帧，在浏览器原图上拖框并“开始传播”。每个对象独立初始化 SAM2 state，查看初始 overlay 后再决定是否纠正。

发现漂移后，在“人工硬重启”选择对象、重启帧和结束帧，重新画框，点击“生成硬重启候选”。这会建立全新的 inference state，不在旧污染 state 中追加提示。先看候选视频，再“接受纠正”；session-local timeline 随后更新并重建当前完整 overlay。可以多次重复，放弃候选不会删除候选文件。UI 主流程依赖人工观察，不要求先运行异常检测。

在“最终导出”点击“生成最终结果”，检查报告和 overlay。输出在 `ui_workspace/sessions/<session_id>/final/`，包含 `objects/objN/final_tracks.jsonl`、`merged/tracks.jsonl`、`labels/`、`quality_report.json` 和 `overlay.mp4`。mask 和中间结果保留在 session 内，移动或重建标注时请保留整个 session 并检查路径引用。每次加载建立独立 session，不复用 CLI 的真实 registry/timeline。

以下章节保留 CLI 详细流程；自动异常检测与 mask/bbox handoff 为可选工具，推荐优先使用上述人工硬重启。

## 2. 环境准备

### 本机与服务器

推荐 Ubuntu 或 WSL2、Conda、CUDA GPU。服务器可使用无显示环境：所有画框命令均可传 `--box X1 Y1 X2 Y2`；需要交互画框时，使用带图形转发的会话。不要完整读入视频帧到内存，先抽帧并只让 SAM2 读取所需短片段。

```bash
cd ~/projects/sam2_video_annotator
conda env create -f environment.yml
conda activate sam2_env
cd ~/projects/sam2
pip install -e .
cd ~/projects/sam2_video_annotator
export PYTHONPATH=src:${PYTHONPATH}
```

SAM2 源码应位于与项目相邻的位置，模型配置为 `configs/sam2.1/sam2.1_hiera_t.yaml`，checkpoint 由使用者下载后在 `configs/default.yaml` 的 `sam2.checkpoint` 中配置。权重、真实视频、帧和输出均不随发布包提供。

## 3. 数据目录与身份规则

建议每个视频独立使用 `data/frames/<video_id>/`、`data/segments/`、`data/prompts/` 和 `outputs/<video_id>/`；每次抽帧产生 `metadata.json`。注册表和 timeline 是当前视频的元数据，目录可按视频保存。

固定全局目录 `configs/drone_catalog.json`：

| global_drone_id | class_id |
| --- | ---: |
| `drone_1` | 1 |
| `drone_2` | 2 |
| `drone_3` | 0 |

`obj_id` 是局部实例，不是类别。视频 A 可为 `obj1 → drone_1 → 1`、`obj2 → drone_2 → 2`；视频 B 可为 `obj1 → drone_1 → 1`、`obj2 → drone_3 → 0`。当前视频未出现的无人机不注册；背景误跟的 obj3 是无效轨迹，不代表 `drone_3`。

## 4. 从视频到初始传播

放入视频后抽帧：

```bash
python scripts/extract_frames.py --video data/videos/video_a.mp4 --frames-dir data/frames/video_a --jpeg-quality 95
```

为每架有效无人机注册局部对象，并在首次清晰出现的原始帧给框。注册信息会写入 registry，`global_drone_id` 决定 catalog 中的 `class_id`：

```bash
python scripts/register_object.py --registry data/object_registry.json --catalog configs/drone_catalog.json --frames-dir data/frames/video_a --obj-id 1 --name obj1 --global-drone-id drone_1 --frame-idx 203 --prompt-output data/prompts/obj1_global.json --box 100 120 140 155
```

创建短片段会建立局部从 `000000.jpg` 开始的软链接，并把 prompt 转为局部坐标；原始帧索引保留在 manifest：

```bash
python scripts/create_tracking_segment.py --source-frames-dir data/frames/video_a --start-frame 203 --end-frame 354 --output-dir data/segments/obj1_203_354 --prompt data/prompts/obj1_global.json --output-prompt data/prompts/obj1_local.json
python scripts/track_registered_object.py --config configs/default.yaml --registry data/object_registry.json --obj-id 1 --segment-dir data/segments/obj1_203_354 --prompt data/prompts/obj1_local.json --output-dir outputs/video_a/obj1
```

对第二架无人机重复此过程，使用另一个 `obj_id`、prompt、segment 和输出目录。每次传播只有一个对象，并使用独立的新 inference state。

## 5. 查看结果与可选 CLI 异常管理

传播会增量保存 mask、bbox、YOLO 标签和 `tracks.jsonl`。可查看单对象输出目录中的 mask PNG、JSONL bbox；多对象合并后再渲染：

```bash
python scripts/detect_anomalies.py --tracks outputs/video_a/obj1/tracks.jsonl --frames-dir data/frames/video_a --output-dir outputs/video_a/obj1/review
python scripts/build_anomaly_intervals.py --tracks outputs/video_a/obj1/tracks.jsonl --registry data/object_registry.json --catalog configs/drone_catalog.json --frame-anomalies outputs/video_a/obj1/review/anomalies.jsonl --output data/anomaly_intervals.jsonl --review-csv data/review_queue.csv --overwrite
```

异常检测只用于复核排序，不是置信度。`bbox_area` 跳变根据近期基线的面积比给 warning/high/critical 权重；默认阈值、权重和连续帧规则在 `configs/default.yaml` 的 `bbox_area`、`intervals` 中，必要时可用 `build_anomaly_intervals.py --help` 的覆盖参数。`restart_lookback` 推荐可靠重启帧；人工应检查遮挡、漂移、再出现和尺度变化。

用以下命令记录审核状态（不直接修改轨迹）：

```bash
python scripts/update_review_status.py --input data/anomaly_intervals.jsonl --interval-id interval_id_here --status needs_restart --reviewer-note "确认漂移"
```

## 6. 三种纠正与候选预览

所有纠正都从可靠来源帧创建新片段和**全新的 SAM2 inference state**。禁止在旧污染 state 追加 prompt。

1. 来源 mask 清晰准确时选 `mask`：形状保留较好，但源 mask 偏差也会继续传播。
2. mask 不可靠但 bbox 准确时选 `bbox`：操作简单，框内背景较多时可能不稳定。
3. 明显漂移、遮挡后重现、尺度变化大或自动效果差时选 `manual-box` 人工硬重启。

`create_handoff.py` 只生成候选，不覆盖正式轨迹。使用 `--render-preview` 查看结果，确认后才接受。例如 mask 候选：

```bash
python scripts/create_handoff.py --tracks outputs/video_a/obj1/tracks.jsonl --frames-dir data/frames/video_a --registry data/object_registry.json --catalog configs/drone_catalog.json --config configs/default.yaml --output-dir outputs/video_a/handoff_obj1 --obj-id 1 --source-frame 257 --end-frame 354 --prompt-mode mask --render-preview --overwrite
```

bbox 交接只将 `--prompt-mode` 改为 `bbox`。人工画框使用 `manual-box`；脚本会请求画框，可取消后重新画。无 GUI 环境应在可视化会话中操作；取消不会接受候选。候选 tracks 的实际路径以 `create_handoff.py` 输出的 summary 为准，切勿按旧文档猜测 `tracks.jsonl` 路径。

预览通过后显式写入对象 timeline：

```bash
python scripts/accept_handoff.py --timeline data/object_timelines.json --registry data/object_registry.json --catalog configs/drone_catalog.json --tracks outputs/video_a/handoff_obj1/tracks.jsonl --obj-id 1 --start-frame 257 --end-frame 354 --restart-frame 257 --source-type auto_mask_handoff --review-note "已查看候选" --overwrite-existing-range
```

`restart_frame` 可以早于替换区间：它表示候选重建 state 的可靠起点，而替换区间仍由 `start-frame`/`end-frame` 决定。若接受失败，修正候选路径、范围或 registry 后重新运行；不要直接编辑正式轨迹。

## 7. Timeline、构建与合并

`object_timelines.json` 为每个 `obj_id` 记录原始段和已接受的交接段、路径、范围、来源和审核信息。路径相对基准由该 JSON 文件所在位置决定；不要误以为它总会解析到 `data/outputs`。

构建单对象连续轨迹：

```bash
python scripts/build_object_track.py --timeline data/object_timelines.json --registry data/object_registry.json --catalog configs/drone_catalog.json --obj-id 1 --output outputs/video_a/final/obj1_tracks.jsonl --overwrite
```

为每个有效对象构建后，`--tracks` 必须只写一次并跟随多个文件，避免重复参数导致 argparse 只保留最后一组：

```bash
python scripts/merge_object_tracks.py --tracks outputs/video_a/final/obj1_tracks.jsonl outputs/video_a/final/obj2_tracks.jsonl --registry data/object_registry.json --catalog configs/drone_catalog.json --active-only --output-tracks outputs/video_a/final/merged/tracks.jsonl --output-labels outputs/video_a/final/labels --overwrite
python scripts/render_multi_overlay.py --frames-dir data/frames/video_a --tracks outputs/video_a/final/merged/tracks.jsonl --output outputs/video_a/final/overlay.mp4 --start-frame 203 --end-frame 354
```

YOLO 标签类别始终来自 registry 的 `global_drone_id` 再由 catalog 校验，绝不由 `obj_id` 推断。

## 8. 一键最终构建与质量校验

推荐在所有候选都已审核接受后执行：

```bash
python scripts/finalize_video_annotation.py --timeline data/object_timelines.json --registry data/object_registry.json --catalog configs/drone_catalog.json --frames-dir data/frames/video_a --output-dir outputs/video_a/final --render-overlay --overwrite
```

该命令构建 active 且 validated 对象、合并 tracks/YOLO 标签、写质量报告，可选渲染 overlay；失败不会替换已完成导出。单独检查：

```bash
python scripts/validate_annotation_export.py --tracks outputs/video_a/final/merged/tracks.jsonl --labels-dir outputs/video_a/final/labels --registry data/object_registry.json --catalog configs/drone_catalog.json --report-output outputs/video_a/final/quality_report.json
```

处理下一个视频时创建新的 frames、registry、timeline、prompts 和 outputs 子目录，再按本手册重复；仍共用同一份 drone catalog。

## 9. 常见错误、限制与验收

详细排错见 [TROUBLESHOOTING_ZH.md](TROUBLESHOOTING_ZH.md)。常见现象包括导入路径、缺少身份字段、mask/候选路径、Qt 字体警告及旧 state 污染。

限制：仅前向传播、每视频最多两架有效对象、无自动发现和自动接受；自动交接已跑通但并不保证稳定，必须人工看预览。结束前确认：所有有效对象映射正确、候选均审核、timeline 连续、最终质量报告 valid、labels/overlay 已抽查，并运行：

```bash
python -m compileall src scripts app.py
python -m pytest -q
```
