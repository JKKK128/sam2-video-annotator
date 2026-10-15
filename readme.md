# SAM2 Video Annotator

基于 SAM2 的无人机视频半自动标注工具。

当前主线采用：

```text
视频抽帧
→ 人工在目标首次清晰出现帧框选目标
→ SAM2 单目标独立传播
→ 输出 mask / bbox / YOLO 标签
→ 异常检测
→ 人工复核
→ 错误区间硬重启传播
→ 多目标轨迹合并
→ 生成多目标可视化视频
```

## 当前已实现

- 视频抽帧与帧目录管理
- ROI 框选并生成初始提示
- SAM2 单目标视频传播
- 原始 mask 转 bbox
- YOLO 标签导出
- 面积、中心位移、空 mask 等异常检测
- 对象注册表
- 多对象独立追踪
- 基于全新 inference state 的硬重启纠正
- 多对象 tracks 合并
- 多行 YOLO 标签生成
- 多对象 overlay 视频生成

## 当前设计原则

1. 每个目标使用独立 SAM2 inference state。
2. 多目标通过 `obj_id` 区分，当前使用 `1 / 2 / 3`。
3. 明显漂移后不在旧 state 中继续追加框，而是从可靠帧重新建立片段并硬重启。
4. bbox 直接由 SAM2 原始二值 mask 计算。
5. bbox 稳定化目前不是主线。
6. 长视频采用短片段处理，建议单段约 100 帧。

## 快速开始

详细操作见：

```text
docs/USER_GUIDE_ZH.md
```

常用帮助：

```bash
python scripts/register_object.py --help
python scripts/create_tracking_segment.py --help
python scripts/track_registered_object.py --help
python scripts/detect_anomalies.py --help
python scripts/merge_object_tracks.py --help
python scripts/render_multi_overlay.py --help
```

## 推荐环境

- Linux / WSL2
- Conda
- NVIDIA GPU
- CUDA 可用的 PyTorch
- Python 3.10–3.12
- OpenCV
- 官方 SAM2 仓库与模型权重

建议交付时同时提供：

```text
environment.yml
requirements-lock.txt
```

生成方式：

```bash
conda env export --from-history > environment.yml
python -m pip freeze > requirements-lock.txt
```

## 数据与模型

默认不将真实视频、完整帧目录、输出结果和模型权重打包进源码压缩包。

接收方需要自行准备：

```text
data/videos/input.mp4
data/frames/input/
checkpoints/sam2.1_hiera_tiny.pt
```

或者按照 `docs/USER_GUIDE_ZH.md` 修改配置路径。

## 测试

```bash
python -m compileall src scripts tests
pytest -q
```

## 重要说明

该工具是半自动标注工具，不保证所有帧完全正确。SAM2 传播结果仍需经过异常检测和人工复核。对于小目标、快速运动、遮挡、出入画面和背景相似区域，建议使用短片段与硬重启纠正。
