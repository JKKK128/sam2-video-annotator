# Project rules

## Project goal

Build an interactive semi-automatic UAV video annotation system based only on
SAM2 video propagation.

The workflow is:

1. Read a video and extract frames.
2. Let the user select a start frame.
3. Let the user provide a box or point prompt.
4. Use the official Meta SAM2 video predictor to propagate the target mask.
5. Save masks, bounding boxes and visualization results.
6. Later add anomaly detection and manual correction.

## Scope restrictions

- Do not use YOLO or any object detector.
- Do not implement automatic target discovery.
- Do not modify the official SAM2 source repository.
- Import SAM2 from the editable installation at ~/projects/sam2.
- Initially support only one object with obj_id=1.
- Initially support forward propagation only.
- Do not implement a GUI before the CLI workflow works.
- Do not implement anomaly detection in the first milestone.
- Do not enable torch.compile or vos_optimized in the first milestone.

## Environment

- OS: Ubuntu under WSL2.
- Python environment: conda environment sam2_env.
- GPU: NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM.
- Model: sam2.1_hiera_tiny.
- SAM2 config:
  configs/sam2.1/sam2.1_hiera_t.yaml
- SAM2 checkpoint:
  ~/projects/sam2/checkpoints/sam2.1_hiera_tiny.pt

## Engineering requirements

- Use pathlib for paths.
- Add type hints.
- Validate all input paths and coordinates.
- Do not load all full-resolution frames into CPU memory at once.
- Save intermediate outputs incrementally.
- Use zero-padded frame names such as 000000.jpg.
- Preserve the original frame index.
- Store metadata in JSON or JSONL.
- Add clear CLI --help messages.
- Add logging and useful error messages.
- Add unit tests for utilities that do not require the GPU.
