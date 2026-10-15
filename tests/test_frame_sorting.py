from __future__ import annotations

from pathlib import Path

import pytest

from sam2_annotator.frame_utils import frame_name, frame_path, parse_frame_index, sorted_frame_paths


def test_frame_name_zero_padding() -> None:
    assert frame_name(0) == "000000.jpg"
    assert frame_name(42) == "000042.jpg"
    assert frame_name(42, suffix="") == "000042"


def test_frame_name_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        frame_name(-1)


def test_sorted_frame_paths_numeric_order(tmp_path: Path) -> None:
    for name in ("000010.jpg", "000002.jpg", "000001.jpg"):
        (tmp_path / name).write_text("", encoding="utf-8")

    assert [path.name for path in sorted_frame_paths(tmp_path)] == ["000001.jpg", "000002.jpg", "000010.jpg"]


def test_parse_frame_index_validation(tmp_path: Path) -> None:
    assert parse_frame_index(tmp_path / "000123.jpg") == 123
    with pytest.raises(ValueError, match="numeric"):
        parse_frame_index(tmp_path / "frame.jpg")
    with pytest.raises(ValueError, match=".jpg"):
        parse_frame_index(tmp_path / "000001.png")


def test_frame_path(tmp_path: Path) -> None:
    assert frame_path(tmp_path, 7) == tmp_path / "000007.jpg"
