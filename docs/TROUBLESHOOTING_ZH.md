# 故障排查

- `ModuleNotFoundError: sam2_annotator`：在项目根目录执行 `export PYTHONPATH="$PWD/src:$PYTHONPATH"`，或执行 `pip install -e .` 安装现有项目。
- registry 缺少 `global_drone_id` 或 `first_visible_frame`：重新用 `register_object.py` 注册；不要手工用 obj_id 补类别。
- timeline 相对路径意外解析到 `data/outputs`：相对 `tracks_path` 是相对 timeline JSON 所在目录，而非当前终端目录；改为正确相对路径或绝对路径。
- `restart_frame` 早于替换区间：这是合法的，表示新 state 的可靠启动帧；`start-frame/end-frame` 才是替换范围。
- merge 只读取最后一个文件：仅写一个 `--tracks`，后面连续列出所有 tracks 路径。
- mask 路径不存在：检查 JSONL 的相对路径相对 tracks 文件所在目录；先确认 mask 未被移动或清理。
- 候选 `tracks.jsonl` 路径不一致：以 `create_handoff.py` 输出的 `handoff_summary.json` 中 `tracks_path` 为准。
- OpenCV Qt 字体警告：通常不影响非 GUI 处理；画框失败时改用 `--box` 或使用图形会话。
- 完整视频加载内存不足：使用 `extract_frames.py`，随后用 `create_tracking_segment.py`；不要自行一次加载全部原始分辨率帧。
- 在旧 SAM2 state 追加提示仍无效：旧 state 可能已污染。创建 mask、bbox 或 manual-box handoff 候选，确保每次均新建 inference state。
- WSL 代理导致 Gradio localhost 访问失败：启动前设置 `export NO_PROXY="127.0.0.1,localhost,::1"` 和 `export no_proxy="127.0.0.1,localhost,::1"`，并使用终端实际打印的 URL。
- 图片序列帧号与原文件名不一致：按自然排序建立 canonical frame 索引；在 session 的 `frames/frame_manifest.json` 查看映射，不改原图名。
- UI 漂移纠正：在人工硬重启区重新选帧、画框并生成候选，看预览后接受，可多次重复；无需先运行 anomaly detection。
- 最终导出：先确认至少有一个 active 且 validated 对象以及完整 timeline，再点击“生成最终结果”。检查 session 的 `final/quality_report.json` 和 `final/overlay.mp4`；报告 valid 不代表跟踪视觉上准确。
