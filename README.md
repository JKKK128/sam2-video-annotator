# SAM2 Video Annotator

A semi-automatic UAV video and image-sequence annotation tool built on Meta SAM2, with a Gradio Web UI, manual hard restart, and YOLO label export.

[中文用户手册](docs/USER_GUIDE_ZH.md) · [中文快速开始](docs/QUICK_START_ZH.md)

## Features

- Video and image sequence input
- Precise browser bbox annotation
- Independent SAM2 tracking per object
- Manual hard restart with candidate preview and explicit acceptance
- Timeline-based correction merge
- YOLO export, final overlay, and quality validation
- Gradio Web UI and retained CLI workflow

## Demo / Screenshot

A UI screenshot has not been included yet. The intended location is `docs/assets/ui_overview.png`.

## Workflow

```text
Video / Image Sequence → Load → Draw bbox → SAM2 propagation
    → Review overlay → Manual hard restart if drift
    → Preview and accept correction → Final export
```

Review and correction are manual. Automatic anomaly detection is optional CLI tooling and is not required by the UI workflow.

## Input

- MP4 and other video formats readable by the installed OpenCV backend.
- A directory of consecutive JPG / JPEG / PNG images.

Image sequences are naturally sorted by filename. Source images remain unchanged; an internal canonical frame mapping links source names to zero-padded frames such as `000000.jpg`. Frame indices refer to this ordered sequence, not numbers inferred from source filenames. Set the image-sequence FPS in the UI for playback timing.

## Object Identity

`obj_id != class_id`: an object ID identifies a local track within one video. Choose its global drone identity explicitly.

| Global drone | YOLO class |
| --- | ---: |
| `drone_1` | 1 |
| `drone_2` | 2 |
| `drone_3` | 0 |

The same `obj_id` may map to different global drones in different videos. The current UI supports `obj1` and `obj2`, each with an independent SAM2 inference state.

## Installation

Use Ubuntu / WSL2 with Conda and a CUDA-capable GPU. `environment.yml` specifies Python 3.12 and Gradio 5; `pyproject.toml` declares the core Python package dependencies. PyTorch and SAM2 must be installed separately as described below.

From the project root:

```bash
conda env create -f environment.yml
conda activate sam2_env
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Set `PYTHONPATH` again when opening a new terminal, or install this project with `pip install -e .`.

## SAM2 Setup

SAM2 source code and model weights are not included. Follow the [official Meta SAM2 installation instructions](https://github.com/facebookresearch/sam2#installation) to prepare compatible CUDA PyTorch and TorchVision dependencies, then install SAM2 from an editable checkout. With this project at `~/projects/sam2_video_annotator`:

```bash
cd ~/projects
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ~/projects/sam2_video_annotator
```

If SAM2 is already installed, use that checkout. Download `sam2.1_hiera_tiny.pt` from the [official SAM2.1 checkpoint links](https://github.com/facebookresearch/sam2#download-checkpoints) into the SAM2 checkout's `checkpoints/` directory.

The UI reads `configs/default.yaml`. Check these values before running, adjusting the checkpoint location for your machine:

```yaml
sam2:
  model_config: configs/sam2.1/sam2.1_hiera_t.yaml
  checkpoint: ../sam2/checkpoints/sam2.1_hiera_tiny.pt
  device: cuda
```

`model_config` refers to the configuration supplied by the installed SAM2 package. `examples/default.example.yaml` is a portable reference. Run from the project root so relative checkpoint paths resolve correctly.

## Run Web UI

From the project root with `sam2_env` active and `PYTHONPATH` set:

```bash
python app.py
```

Open the URL printed in the terminal. With default Gradio settings and an available port, it is `http://127.0.0.1:7860`; Gradio may choose a later port if occupied. `app.py` uses Gradio's default `launch()` settings.

For WSL environments where a proxy interferes with local access, set these before launching:

```bash
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"
```

## Basic Usage

1. **Load data** in the 数据 tab: choose video or image sequence, enter a local path accessible to the Python process, and click 加载数据. A new session is created under `ui_workspace/sessions/`.
2. **Draw and propagate** in 标注与传播: choose `obj1` / `obj2` and its global drone, select the start and end frames, draw a bbox, and click 开始传播. Review the initial overlay. Repeat independently for the second object if needed.
3. **Correct drift** with the manual hard restart controls: choose the object and restart range, redraw its bbox, generate a candidate, inspect the preview, and accept it. Repeat as needed.
4. **Export** in 最终导出: click 生成最终结果, inspect the quality summary and final overlay, then collect the session's `final/` directory.

## Manual Hard Restart

When visual review reveals tracking drift, choose a restart frame where the target is clear and draw a fresh bbox. Click 生成硬重启候选 to create a **new SAM2 inference state** and propagate forward. This does not append a prompt to the old contaminated state.

Inspect the candidate preview before clicking 接受纠正. Acceptance updates the session-local timeline, rebuilds the object's final tracks, and refreshes its current overlay. Multiple accepted restarts are supported. 放弃候选 leaves the candidate files available locally without accepting them.

## Final Export

The UI rebuilds each active, validated object's final tracks from its timeline, merges the tracks, exports YOLO labels, validates the result, and renders the final overlay:

```text
ui_workspace/sessions/<session_id>/final/
├── objects/
│   ├── obj1/final_tracks.jsonl
│   └── obj2/final_tracks.jsonl     # when obj2 is active
├── merged/tracks.jsonl
├── labels/                       # one YOLO .txt file per frame
├── quality_report.json
└── overlay.mp4
```

Masks and intermediate segment/correction outputs remain in the session; track records reference those masks. Keep the complete session if you need to rebuild or move an annotation, and check path references when relocating it. A valid quality report checks export consistency; visually review tracking accuracy as well.

The CLI finalizer produces the same directory layout at its chosen `--output-dir`; overlay rendering requires `--render-overlay`.

## CLI

The original CLI workflow remains available through `scripts/`:

- `extract_frames.py`
- `register_object.py`
- `create_tracking_segment.py`
- `track_registered_object.py`
- `create_handoff.py`
- `accept_handoff.py`
- `finalize_video_annotation.py`

Use each script's `--help` for arguments. See the [Chinese user guide](docs/USER_GUIDE_ZH.md) for commands and file formats.

## Testing

With `sam2_env` active and `PYTHONPATH` set, run:

```bash
python -m compileall src scripts app.py
python -m pytest -q
```

All tests should pass. The utility and UI orchestration tests do not require real SAM2 tracking.

## Known Limitations

- SAM2 propagation needs human review; small targets, occlusion, and fast motion can cause drift.
- Manual hard restart is the recommended, most reliable correction workflow for this project; inspect every candidate.
- Mask / bbox automatic handoff remains available in the CLI but is not recommended as the UI's main workflow.
- Each object uses an independent SAM2 state; propagation is forward only, with at most two active objects per video.
- This is a tool for careful semi-automatic annotation, not fully automatic dataset generation. There is no automatic target discovery or object detector; YOLO is only an export format.
- The environment file is a starting environment specification, not a complete CUDA/PyTorch lockfile.

## Documentation

- [Quick start (中文)](docs/QUICK_START_ZH.md)
- [User guide (中文)](docs/USER_GUIDE_ZH.md)
- [Data formats (中文)](docs/DATA_FORMAT_ZH.md)
- [Troubleshooting (中文)](docs/TROUBLESHOOTING_ZH.md)

## Data and Weights

This repository does not provide UAV raw video, extracted frames, real annotations, outputs, checkpoints, or SAM2 model weights. Prepare your own inputs and weights. Files in `examples/` are illustrative templates, not experiment results.

## Acknowledgements

Built on [Meta SAM2](https://github.com/facebookresearch/sam2). SAM2 and its weights are subject to their upstream licenses.

## License

License: To be determined.
