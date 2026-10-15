#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import json
import re
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PATTERN = re.compile(r"frame_(\d+)$", re.IGNORECASE)

def main():
    parser = argparse.ArgumentParser(
        description="将 frame_<timestamp_ns> 图片整理为 SAM2 所需的六位连续编号。"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在: {input_dir}")

    records = []
    unmatched = []
    for path in input_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = PATTERN.fullmatch(path.stem)
        if match:
            records.append((int(match.group(1)), path))
        else:
            unmatched.append(path.name)

    if not records:
        raise SystemExit("没有找到 frame_<timestamp_ns> 格式的图片。")

    records.sort(key=lambda item: item[0])

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"输出目录非空: {output_dir}；确认后添加 --overwrite")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for index, (timestamp_ns, source_path) in enumerate(records):
        target_name = f"{index:06d}{source_path.suffix.lower()}"
        target_path = output_dir / target_name

        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()

        if args.mode == "copy":
            shutil.copy2(source_path, target_path)
        else:
            target_path.symlink_to(source_path.resolve())

        rows.append({
            "frame_idx": index,
            "sam2_filename": target_name,
            "original_filename": source_path.name,
            "timestamp_ns": timestamp_ns,
            "source_path": str(source_path.resolve()),
        })

    manifest_path = output_dir / "frame_map.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "source_dir": str(input_dir.resolve()),
        "frames_dir": str(output_dir.resolve()),
        "frame_count": len(records),
        "first_timestamp_ns": records[0][0],
        "last_timestamp_ns": records[-1][0],
        "naming": "six_digit_sequential",
        "mode": args.mode,
        "unmatched_files": unmatched,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 64)
    print(f"成功整理图片: {len(records)}")
    print(f"未匹配文件名: {len(unmatched)}")
    print(f"输出目录: {output_dir.resolve()}")
    print(f"映射表: {manifest_path.resolve()}")
    print(f"首帧: {rows[0]['sam2_filename']} <- {rows[0]['original_filename']}")
    print(f"末帧: {rows[-1]['sam2_filename']} <- {rows[-1]['original_filename']}")
    print("=" * 64)

if __name__ == "__main__":
    main()
