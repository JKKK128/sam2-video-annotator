# 数据格式

`drone_catalog.json` 定义唯一的 `global_drone_id → class_id` 映射；固定为 drone_1→1、drone_2→2、drone_3→0。`object_registry.json` 是当前视频局部对象表，每个有效对象必须有 `obj_id`、`global_drone_id`、catalog 一致的 `class_id`、`first_visible_frame`、`status=active` 和 `validated=true`。`obj_id` 不具有类别语义。

`tracks.jsonl` 每行对应一个原始 `frame_idx` 与一个 `obj_id`，包含 `bbox_xyxy`、mask 路径、尺寸和身份字段。`object_timelines.json` 将连续的原始段和已接受纠正段按范围排列；`tracks_path` 相对路径相对于 timeline 文件本身解析。YOLO 标签每帧一个 `.txt`，类别来自 catalog。

可复制 `examples/` 的四个文件作为无真实数据的起点。视频 B 的局部映射示例为 obj1→drone_1→1、obj2→drone_3→0。

UI 每次加载在 `ui_workspace/sessions/<session_id>/` 内保存独立 registry、timeline、frames、prompts 和 outputs。图片输入自然排序后建立 `000000.jpg` 等内部帧及 `frames/frame_manifest.json`，不改原始文件；`frame_idx` 是该序列索引，而不是来源文件名中的数字。

UI 接受硬重启候选后更新 session-local timeline 并重建当前对象轨迹。最终导出目录为 session 下的 `final/`：`objects/objN/final_tracks.jsonl`、`merged/tracks.jsonl`、`labels/*.txt`、`quality_report.json`、`overlay.mp4`。mask 仍由 session 内中间结果提供；JSONL 可包含绝对路径，搬迁时需检查引用，不能将 final 目录视为完整独立数据包。
