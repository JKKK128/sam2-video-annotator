from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sam2_annotator.frame_source import prepare_image_sequence
from sam2_annotator.frame_utils import sorted_frame_paths


def _write_image(path: Path, *, size: tuple[int, int] = (8, 6), color: tuple[int, int, int] = (1, 2, 3)) -> None:
    image = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_image_sequence_natural_sorts_and_supports_mixed_case_formats(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_image(source_dir / "frame10.PNG")
    _write_image(source_dir / "frame2.JPEG")
    _write_image(source_dir / "frame1.JPG")

    result = prepare_image_sequence(source_dir, tmp_path / "frames", fps=12.5)

    assert [path.name for path in sorted_frame_paths(result.frames_dir)] == [
        "000000.jpg", "000001.jpg", "000002.jpg"
    ]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert [frame["source_name"] for frame in manifest["frames"]] == [
        "frame1.JPG", "frame2.JPEG", "frame10.PNG"
    ]
    assert [frame["frame_idx"] for frame in manifest["frames"]] == [0, 1, 2]
    assert manifest["frames"][0]["materialization"] == "symlink"
    assert manifest["frames"][1]["materialization"] == "symlink"
    assert manifest["frames"][2]["materialization"] == "converted_jpeg"
    assert (result.frames_dir / "000000.jpg").is_symlink()
    assert not (result.frames_dir / "000002.jpg").is_symlink()
    assert cv2.imread(str(result.frames_dir / "000002.jpg"), cv2.IMREAD_COLOR) is not None
    assert (result.frames_dir / "000002.jpg").read_bytes()[:3] == b"\xff\xd8\xff"
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "source_type": "image_sequence",
        "fps": 12.5,
        "frame_count": 3,
        "width": 8,
        "height": 6,
        "frame_manifest": "frame_manifest.json",
        "output_pattern": str(result.frames_dir / "%06d.jpg"),
        "jpeg_quality": 95,
    }


def test_prepare_image_sequence_preserves_source_images(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    jpeg = source_dir / "original.jpg"
    png = source_dir / "original.png"
    _write_image(jpeg, color=(15, 20, 25))
    _write_image(png, color=(30, 35, 40))
    before = {path.name: _digest(path) for path in (jpeg, png)}

    prepare_image_sequence(source_dir, tmp_path / "frames")

    assert {path.name: _digest(path) for path in (jpeg, png)} == before
    assert sorted(path.name for path in source_dir.iterdir()) == ["original.jpg", "original.png"]


@pytest.mark.parametrize("fps", [0, -1, "not-a-number"])
def test_prepare_image_sequence_rejects_nonpositive_or_invalid_fps(tmp_path: Path, fps: object) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_image(source_dir / "frame.jpg")

    with pytest.raises(ValueError, match="fps"):
        prepare_image_sequence(source_dir, tmp_path / "frames", fps=fps)  # type: ignore[arg-type]
    assert not (tmp_path / "frames").exists()


def test_prepare_image_sequence_rejects_dimension_mismatch_without_partial_output(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_image(source_dir / "frame1.jpg", size=(8, 6))
    _write_image(source_dir / "frame2.png", size=(4, 3))

    with pytest.raises(ValueError, match=r"first image is 8x6.*frame2.png is 4x3"):
        prepare_image_sequence(source_dir, tmp_path / "frames")
    assert not (tmp_path / "frames").exists()


def test_prepare_image_sequence_rejects_corrupt_or_empty_source(tmp_path: Path) -> None:
    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "broken.PNG").write_bytes(b"not an image")
    with pytest.raises(ValueError, match=r"failed to decode image: broken.PNG"):
        prepare_image_sequence(corrupt_dir, tmp_path / "corrupt_frames")
    assert not (tmp_path / "corrupt_frames").exists()

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "notes.txt").write_text("ignored", encoding="utf-8")
    with pytest.raises(ValueError, match="no supported images"):
        prepare_image_sequence(empty_dir, tmp_path / "empty_frames")
    assert not (tmp_path / "empty_frames").exists()


def test_prepare_image_sequence_rejects_existing_target_and_source_target_conflicts(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_image(source_dir / "frame.jpg")
    target_dir = tmp_path / "frames"
    target_dir.mkdir()
    (target_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_image_sequence(source_dir, target_dir)
    assert (target_dir / "existing.txt").read_text(encoding="utf-8") == "keep"
    with pytest.raises(ValueError, match="different directories"):
        prepare_image_sequence(source_dir, source_dir)
    with pytest.raises(ValueError, match="inside source_dir"):
        prepare_image_sequence(source_dir, source_dir / "nested_frames")
