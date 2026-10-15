"""Prepare image sequences for the existing canonical JPEG-frame workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import cv2

from sam2_annotator.frame_utils import frame_name


SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
JPEG_QUALITY = 95


@dataclass(frozen=True)
class PreparedImageSequence:
    """Paths and basic properties of a prepared canonical frame directory."""

    frames_dir: Path
    frame_count: int
    width: int
    height: int
    fps: float
    metadata_path: Path
    manifest_path: Path


def _natural_sort_key(path: Path) -> list[tuple[int, int | str]]:
    """Return a case-insensitive natural-sort key for a source filename."""
    key: list[tuple[int, int | str]] = []
    for part in re.split(r"(\d+)", path.name.casefold()):
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return key


def _source_images(source_dir: Path) -> list[Path]:
    images = [
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
    ]
    if not images:
        raise ValueError(
            f"no supported images found in {source_dir}; expected .jpg, .jpeg, or .png files"
        )
    return sorted(images, key=_natural_sort_key)


def _is_jpeg(path: Path) -> bool:
    """Return whether a decoded image also carries a JPEG file signature."""
    with path.open("rb") as handle:
        return handle.read(3) == b"\xff\xd8\xff"


def _read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode image: {path.name}")
    return image


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare_image_sequence(
    source_dir: Path | str,
    frames_dir: Path | str,
    fps: float = 10.0,
) -> PreparedImageSequence:
    """Materialize a source image directory as canonical ``000000.jpg`` frames.

    Source images are read only. Valid JPEG files are symlinked; all other
    supported images are decoded and saved as RGB-compatible JPEG derivatives.
    The target is assembled in a sibling temporary directory and renamed only
    after every image has been validated and metadata has been written.
    """
    source_path = Path(source_dir).expanduser().resolve()
    target_path = Path(frames_dir).expanduser().resolve()

    if not source_path.is_dir():
        raise NotADirectoryError(f"source image directory does not exist: {source_path}")
    if source_path == target_path:
        raise ValueError("source_dir and frames_dir must be different directories")
    if target_path.is_relative_to(source_path):
        raise ValueError("frames_dir must not be inside source_dir")
    if target_path.exists():
        if target_path.is_dir() and any(target_path.iterdir()):
            raise FileExistsError(f"frames_dir already exists and is not empty: {target_path}")
        raise FileExistsError(f"frames_dir already exists: {target_path}")

    try:
        fps_value = float(fps)
    except (TypeError, ValueError) as error:
        raise ValueError(f"fps must be a positive number, got {fps!r}") from error
    if fps_value <= 0:
        raise ValueError(f"fps must be positive, got {fps_value}")

    source_images = _source_images(source_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{target_path.name}.tmp_", dir=target_path.parent))

    try:
        expected_size: tuple[int, int] | None = None
        manifest_frames: list[dict[str, object]] = []
        for frame_idx, source_image in enumerate(source_images):
            image = _read_image(source_image)
            height, width = image.shape[:2]
            image_size = (width, height)
            if expected_size is None:
                expected_size = image_size
            elif image_size != expected_size:
                raise ValueError(
                    f"image dimensions differ: first image is {expected_size[0]}x{expected_size[1]}, "
                    f"but {source_image.name} is {width}x{height}"
                )

            canonical_name = frame_name(frame_idx)
            canonical_path = temporary_dir / canonical_name
            if _is_jpeg(source_image):
                canonical_path.symlink_to(source_image)
                materialization = "symlink"
            else:
                if not cv2.imwrite(
                    str(canonical_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                ):
                    raise OSError(f"failed to write converted JPEG: {canonical_path}")
                materialization = "converted_jpeg"

            manifest_frames.append(
                {
                    "frame_idx": frame_idx,
                    "source_name": source_image.name,
                    "source_path": str(source_image),
                    "canonical_name": canonical_name,
                    "materialization": materialization,
                }
            )

        assert expected_size is not None
        width, height = expected_size
        manifest_path = temporary_dir / "frame_manifest.json"
        _write_json(
            manifest_path,
            {
                "schema_version": 1,
                "source_type": "image_sequence",
                "source_dir": str(source_path),
                "frame_count": len(manifest_frames),
                "frames": manifest_frames,
            },
        )
        metadata_path = temporary_dir / "metadata.json"
        _write_json(
            metadata_path,
            {
                "source_type": "image_sequence",
                "fps": fps_value,
                "frame_count": len(manifest_frames),
                "width": width,
                "height": height,
                "frame_manifest": manifest_path.name,
                "output_pattern": str(target_path / "%06d.jpg"),
                "jpeg_quality": JPEG_QUALITY,
            },
        )
        os.replace(temporary_dir, target_path)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return PreparedImageSequence(
        frames_dir=target_path,
        frame_count=len(manifest_frames),
        width=width,
        height=height,
        fps=fps_value,
        metadata_path=target_path / "metadata.json",
        manifest_path=target_path / "frame_manifest.json",
    )
