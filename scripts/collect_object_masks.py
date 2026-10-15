#!/usr/bin/env python3
"""Collect object mask files into one directory with symlinks."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam2_annotator.tracking_io import collect_mask_links

LOGGER = logging.getLogger("collect_object_masks")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Symlink masks from object tracks into one merged masks directory, failing on name conflicts.")
    parser.add_argument("--tracks", required=True, nargs="+", type=Path, help="One or more tracks JSONL files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Merged masks directory, e.g. outputs/merged/masks.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    created = collect_mask_links(args.tracks, args.output_dir)
    LOGGER.info("created %s mask links in %s", len(created), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
