# GitHub 发布前审计

审计日期：2026-10-15。范围：公开仓库整理，不增加功能、不运行真实 SAM2 tracking。

## 结论与待确认项

当前 UI/CLI 主流程代码未被本轮修改；实验数据、权重、outputs、ui_workspace、snapshots 和真实 registry/timeline 均保留。没有暂存、提交、push、创建远端或重写历史。

当前 Git 索引仍跟踪 12 个缓存文件和 1 个备份文件。新增 ignore 不会让它们退出版本控制；直接发布现有分支仍会包含它们。建议用户确认后仅取消这些文件的跟踪，保留本地副本；该操作会在下一次提交中删除它们的仓库条目，不删除工作区副本，也不移除旧历史。本轮没有执行此操作。

没有 LICENSE，README 标为 “License: To be determined.”。公开前请自行决定 MIT、Apache-2.0 或其他适当许可，不能把上游 SAM2 许可自动视为本项目许可。

## 初始只读 Git 审计记录

- `git branch --show-current`：`ui-v1`。
- `git remote -v`：空，没有 remote。
- `git log --oneline -10` 实际返回以下 6 条：
  - `9228bbe Add UI data import and initial tracking workflow`
  - `898a41f Complete SAM2 handoff and correction workflow`
  - `23ff877 Archive bbox stabilization experiment`
  - `81aa622 Add temporal anomaly review pipeline`
  - `2affbb8 Validate single object SAM2 propagation`
  - `443730f Implement minimal SAM2 video propagation pipeline`
- `git tag --list`：
  - `pre-ui-stable`
  - `sam2-annotator-v1.0`
  - `ui-initial-tracking-stable`

初始 `git status --short`：

```text
 M README.md
 M app.py
 M configs/default.yaml
 M configs/default.yaml.bak
 M configs/restart_000257_000302.yaml
 M configs/test_segment_100.yaml
 M configs/test_segment_20.yaml
 M docs/USER_GUIDE_ZH.md
 M scripts/__pycache__/track_box.cpython-312.pyc
 M scripts/correct_and_repropagate.py
 M scripts/create_handoff.py
 M scripts/package_release.sh
 M scripts/track_box.py
 M scripts/track_registered_object.py
 M src/sam2_annotator/ui_service.py
 M tests/test_ui_service.py
?? RELEASE_CHECKLIST.md
?? build_optical_yolo_dataset.py
?? checkpoints/
?? configs/optical_drone_catalog.json
?? convert_optical_images1.py
?? datasets/
?? dist/
?? docs/DATA_FORMAT_ZH.md
?? docs/QUICK_START_ZH.md
?? docs/RELEASE_NOTES.md
?? docs/TROUBLESHOOTING_ZH.md
?? environment.yml
?? examples/
?? prepare_timestamp_sequence_for_sam2.py
?? pyproject.toml
?? readme.md
?? scripts/build_release.py
?? scripts/prepare_sam2_frames.py
?? snapshots/
?? src/sam2_annotator/ui_bbox.py
?? tests/test_app_ui_helpers.py
?? tests/test_ui_final_export.py
?? ui_workspace/
?? visualize_yolo_random_samples.py
```

已有修改和未跟踪文件属于用户；本轮没有清理它们。README 和中文手册的 Git diff 包含任务开始前已有改动，不能把整个 diff 都归因于本轮。

## 本轮文件变更

修改已有工作区文件（部分在任务开始时尚未跟踪）：

- `.gitignore`：追加发布排除规则。
- `README.md`：英文 GitHub 首页，以真实 UI 主流程为入口。
- `AGENTS.md`：仅将个人 home 绝对路径换成 `~/projects/sam2/...`，不改规则。
- `RELEASE_CHECKLIST.md`：修正删除大文件的建议、测试命令和 UI/人工复核说明。
- `docs/USER_GUIDE_ZH.md`：局部新增 UI、图片输入、人工硬重启与导出说明，保留 CLI 章节。
- `docs/QUICK_START_ZH.md`：新增简短 UI 入口，保留 CLI。
- `docs/DATA_FORMAT_ZH.md`：补充 session、图片 manifest 与真实最终目录。
- `docs/TROUBLESHOOTING_ZH.md`：修正安装说明，补充 localhost、图片映射、硬重启与最终报告。
- `docs/RELEASE_NOTES.md`：说明旧打包工具会重建整个 dist 且缺少 UI 入口文件，避免误用。

本轮新建：`docs/GITHUB_RELEASE_AUDIT.md`（本文件）。

未修改运行代码、配置、环境文件、现有示例或测试。三个 JSON example 和 YAML example 原本已存在，使用示意名称和相对路径，没有新增重复模板。没有创建新发布检查脚本。

## .gitignore

保留原有 `data/`、`outputs/`、`__pycache__/`、`*.py[cod]` 和 `.pytest_cache/`。其中 `data/` 已覆盖 videos、frames、segments、prompts、corrections，不必重复逐项配置。

新增：

- 环境与凭证：`.venv/`、`venv/`、`env/`、`.env`、`.env.*`。
- 编辑器：`.vscode/`、`.idea/`。
- 权重和数据：`checkpoints/`、`*.pt`、`*.pth`、`*.ckpt`、`*.safetensors`、`datasets/`。
- 工作区与构建产物：`ui_workspace/`、`snapshots/`、`dist/`、`build/`、`*.egg-info/`。
- 媒体、备份、临时文件：`*.mp4`、`*.avi`、`*.mov`、`*.bak`、`*.tmp`、`*.log`、`.ipynb_checkpoints/`。
- 本地数据集专用配置：`/configs/optical_drone_catalog.json`；不是当前 UI 使用的共享 catalog，保留本地文件。

已用 `git check-ignore --no-index` 验证各类排除规则，且 src、scripts、tests、docs、examples、configs/default.yaml、app.py、environment.yml、pyproject.toml 未被误排除。没有整体忽略 configs。

仍被跟踪的忽略文件：

```text
configs/default.yaml.bak
scripts/__pycache__/extract_frames.cpython-312.pyc
scripts/__pycache__/render_overlay.cpython-312.pyc
scripts/__pycache__/select_box.cpython-312.pyc
scripts/__pycache__/track_box.cpython-312.pyc
src/sam2_annotator/__pycache__/__init__.cpython-312.pyc
src/sam2_annotator/__pycache__/config_utils.cpython-312.pyc
src/sam2_annotator/__pycache__/frame_utils.cpython-312.pyc
src/sam2_annotator/__pycache__/mask_utils.cpython-312.pyc
tests/__pycache__/test_frame_sorting.cpython-312-pytest-9.1.1.pyc
tests/__pycache__/test_frame_sorting.cpython-312.pyc
tests/__pycache__/test_mask_utils.cpython-312-pytest-9.1.1.pyc
tests/__pycache__/test_mask_utils.cpython-312.pyc
```

缓存可能包含编译时的本机源文件路径。建议确认后取消跟踪，但本轮不执行 `git rm --cached`。

## 大文件、真实数据与权重

执行了：

```bash
find . -type f -size +20M -not -path './.git/*'
```

发现以下 6 个文件，全部保留并由 ignore 排除：

| 文件 | 字节数 |
| --- | ---: |
| data/videos/input.mp4 | 203517921 |
| outputs/merged/overlay_203_354_final.mp4 | 28188328 |
| outputs/merged/overlay_203_354_valid.mp4 | 28188328 |
| outputs/merged/overlay_203_379_current.mp4 | 32925485 |
| snapshots/video_input_203_354_valid/overlay_203_354_valid.mp4 | 28188328 |
| snapshots/input_final_203_354/overlay_203_354_final.mp4 | 28188328 |

按 `git ls-files` 检查当前工作区对应的已跟踪文件大小：没有超过 20 MiB 的文件，最大文件为 app.py（25249 字节）。没有发现 checkpoint、视频、dataset image、mask、zip、tar.gz、dist artifact 被跟踪；也没有真实 data、datasets、outputs、ui_workspace 或 snapshots 被跟踪。

还检查了所有可见 refs 的提交路径记录，没有发现 data/datasets/outputs/ui_workspace/snapshots 或常用权重扩展名的历史提交路径。本检查没有对历史每个 blob 做完整安全取证。

## 路径与敏感信息

检查了待公开源码、配置、脚本、测试、文档及 examples 的个人 home 路径、Windows C/D 盘路径、指定本机代理地址，以及 api_key/token/password/secret、SSH 和常见私网地址模式。

- 当前待公开文本中的个人 home 路径仅在 AGENTS.md 发现，已换成用户无关的 `~/projects` 路径。
- configs/default.yaml 和几个已有片段配置已使用相邻 SAM2 checkout 的相对 checkpoint 路径；本轮没有改动。
- 真实 data、datasets、outputs、ui_workspace、snapshots 和旧 dist 文本中存在本机路径；按保护要求全部原样保留，并排除上传。
- 未发现真实 token、password、API key、SSH key 或本机代理凭证。扫描 build_release.py 命中的 “failed” 字符串属于 Windows 盘符模式的误匹配，不是凭证。
- README 的 localhost 地址是 UI 使用示例，可保留。
- 旧提交 898a41f、81aa622、2affbb8、443730f 中有个人 home 路径的增删记录。当前清理不会改变旧提交；推送现有历史和旧 tag 会公开其可达历史。本轮没有改写历史。

这是基于文件内容和常见模式的检查，不是对所有历史 blob 的完整密钥扫描。

## README、UI 与文档一致性

README 包含：Features、Demo / Screenshot、Workflow、Input、Object Identity、Installation、SAM2 Setup、Run Web UI、Basic Usage、Manual Hard Restart、Final Export、CLI、Testing、Known Limitations、Documentation、Data and Weights、Acknowledgements、License。

核对来源：

- environment.yml：Python 3.12、sam2_env、Gradio 5；并非完整 CUDA/PyTorch 锁文件。
- pyproject.toml：核心包依赖及可编辑安装配置。
- 本机官方 SAM2 checkout README：clone、editable install 和 checkpoint 准备方式。
- app.py：直接 `build_app().queue().launch()`，没有自定义启动参数；源输入、obj1/obj2、画框、候选、接受及最终导出控件均一致。
- 已安装 Gradio 5.50.0：默认 host 为 127.0.0.1、起始端口 7860，支持环境变量覆盖及端口顺延。
- frame_source.py：图片自然排序、原图不变、canonical frames 与 `frames/frame_manifest.json`。
- ui_service.py 与 finalize_video_annotation.py：session/final/objects/objN/final_tracks.jsonl、merged/tracks.jsonl、labels、quality_report.json、overlay.mp4。
- 每次人工硬重启都通过 manual-box 创建新 SAM2 state；接受后更新 session-local timeline，支持多次重启。

没有 docs/assets 截图，因此 README 只写预留路径，没有插入不存在的图片链接。README 所列本地文档和脚本均存在。没有执行安装、下载或真实传播命令；启动入口通过源码和现有 UI 测试核对。

中文文档保留了 CLI 细节，需要发布者最终人工浏览确认措辞和自己的数据路径。UI 已明确成为推荐入口；异常检测为可选 CLI，不是 UI 前置步骤。搬迁 session 时还应检查 mask/软链接路径，final 并不是完整独立数据包。

根目录还有未跟踪的旧 `readme.md`，与主 README 重复且内容过时；本轮保留文件、不建议暂存。几个根目录数据集转换工具也保留，未列入发布暂存建议。

## 验证

在已有 sam2_env 中设置 `PYTHONPATH="$PWD/src:$PYTHONPATH"` 后：

- `python -m compileall src scripts app.py`：成功。
- `python -m pytest -q`：156 passed in 1.77s。
- scripts 下 30 个 Python 脚本逐一执行 `--help`：全部退出码为 0。
- `git diff --check`：通过。
- 没有运行真实 SAM2 tracking；没有更改 CLI/UI 行为。

旧 scripts/build_release.py 会删除整个 dist，且白名单没有 app.py 和 .gitignore，不适用于本轮完整 UI 发布；未运行或修改它，也未新增功能重复的脚本。

## 建议暂存与提交（仅建议，未执行）

以下文件集合包括任务开始前已经跑通 UI 所需的现有修改和未跟踪源码。它不是“仅本轮新增修改”的列表。未改动且已跟踪的源码本来就在仓库，无需再次列入 git add。刻意不用目录级 src/scripts/tests/configs，避免把已跟踪缓存和备份的改动一起暂存。

在处理上述取消跟踪确认项并审阅已有修改后，可执行：

```bash
git add -- \
  .gitignore AGENTS.md README.md RELEASE_CHECKLIST.md \
  environment.yml pyproject.toml app.py \
  configs/default.yaml \
  configs/restart_000257_000302.yaml \
  configs/test_segment_100.yaml configs/test_segment_20.yaml \
  docs/USER_GUIDE_ZH.md docs/QUICK_START_ZH.md \
  docs/DATA_FORMAT_ZH.md docs/TROUBLESHOOTING_ZH.md \
  docs/RELEASE_NOTES.md docs/GITHUB_RELEASE_AUDIT.md \
  examples/default.example.yaml examples/drone_catalog.example.json \
  examples/object_registry.example.json examples/object_timelines.example.json \
  scripts/correct_and_repropagate.py scripts/create_handoff.py \
  scripts/package_release.sh scripts/track_box.py \
  scripts/track_registered_object.py scripts/build_release.py \
  scripts/prepare_sam2_frames.py \
  src/sam2_annotator/ui_service.py src/sam2_annotator/ui_bbox.py \
  tests/test_ui_service.py tests/test_app_ui_helpers.py \
  tests/test_ui_final_export.py
git diff --cached --stat
git diff --cached --name-only
git diff --cached
```

确认后由用户手动提交：

```bash
git commit -m "Prepare public GitHub release"
```

上述命令不会取消已经跟踪的缓存/备份；这件事仍需另行确认。当前没有 LICENSE，暂存命令不包含不存在的文件。

## GitHub 建议（未创建、未推送）

Repository name: `sam2-video-annotator`

Description:

A Gradio-based semi-automatic UAV video and image-sequence annotation tool built on Meta SAM2, with manual hard restart and YOLO export.

Topics:

`sam2`, `segment-anything`, `video-annotation`, `uav`, `drone`, `computer-vision`, `yolo`, `gradio`, `semi-automatic-annotation`

用户自行创建 GitHub 仓库、确定许可并检查提交后，才执行以下流程；URL 需替换为真实仓库地址：

```bash
git remote -v
git remote add origin <URL>
git branch -M main
git push -u origin main
```

当前没有 origin；未来若已有 origin，不要重复 add。分支重命名由用户决定，本轮保持 ui-v1。

已有 `sam2-annotator-v1.0` 等旧 tag；若决定为当前整理结果另建 v1.0.0，请先确认版本命名及历史内容，再手动运行：

```bash
git tag -a v1.0.0 -m "First public release"
git push origin v1.0.0
```

仍不建议上传：data、datasets、checkpoints、outputs、ui_workspace、snapshots、dist、缓存、备份、临时文件、环境/凭证、真实媒体、本地 optical catalog，以及未审阅的根目录数据转换工具和重复旧 readme。所有这些现有文件均未删除。

## 整理后的 git status --short

下方快照包含本报告自身；未跟踪数据目录从 status 消失是 ignore 生效，不是文件被删除。暂存区为空。

```text
 M .gitignore
 M AGENTS.md
 M app.py
 M configs/default.yaml
 M configs/default.yaml.bak
 M configs/restart_000257_000302.yaml
 M configs/test_segment_100.yaml
 M configs/test_segment_20.yaml
 M docs/USER_GUIDE_ZH.md
 M README.md
 M scripts/__pycache__/track_box.cpython-312.pyc
 M scripts/correct_and_repropagate.py
 M scripts/create_handoff.py
 M scripts/package_release.sh
 M scripts/track_box.py
 M scripts/track_registered_object.py
 M src/sam2_annotator/ui_service.py
 M tests/test_ui_service.py
?? build_optical_yolo_dataset.py
?? convert_optical_images1.py
?? docs/DATA_FORMAT_ZH.md
?? docs/GITHUB_RELEASE_AUDIT.md
?? docs/QUICK_START_ZH.md
?? docs/RELEASE_NOTES.md
?? docs/TROUBLESHOOTING_ZH.md
?? environment.yml
?? examples/
?? prepare_timestamp_sequence_for_sam2.py
?? pyproject.toml
?? readme.md
?? RELEASE_CHECKLIST.md
?? scripts/build_release.py
?? scripts/prepare_sam2_frames.py
?? src/sam2_annotator/ui_bbox.py
?? tests/test_app_ui_helpers.py
?? tests/test_ui_final_export.py
?? visualize_yolo_random_samples.py
```
