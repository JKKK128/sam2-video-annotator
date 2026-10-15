"""Utilities for frame file naming and ordering."""

from __future__ import annotations

from pathlib import Path


FRAME_NAME_WIDTH = 6
FRAME_SUFFIX = ".jpg"


def frame_name(frame_idx: int, suffix: str = FRAME_SUFFIX) -> str:
    """Return the canonical zero-padded frame filename for a frame index."""
    if frame_idx < 0:
        raise ValueError(f"frame_idx must be non-negative, got {frame_idx}")
    return f"{frame_idx:0{FRAME_NAME_WIDTH}d}{suffix}"


def frame_path(frames_dir: Path, frame_idx: int, suffix: str = FRAME_SUFFIX) -> Path:
    """Return the canonical path for a frame inside a frames directory."""
    return frames_dir / frame_name(frame_idx, suffix=suffix)


def parse_frame_index(path: Path) -> int:
    """Parse a zero-padded frame index from a frame path."""
    if path.suffix.lower() != FRAME_SUFFIX:
        raise ValueError(f"expected a {FRAME_SUFFIX} frame, got {path.name}")
    stem = path.stem
    if not stem.isdigit():
        raise ValueError(f"frame filename must be numeric, got {path.name}")
    return int(stem)


def sorted_frame_paths(frames_dir: Path) -> list[Path]:
    """Return JPEG frame paths sorted by numeric frame index."""
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames directory does not exist: {frames_dir}")
    paths = [path for path in frames_dir.glob(f"*{FRAME_SUFFIX}") if path.is_file()]
    return sorted(paths, key=parse_frame_index)
