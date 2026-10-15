#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将原始 PNG 帧转换为保留全局帧号的 SAM2 JPG 帧"
    )
    parser.add_argument("--src-dir", required=True, help="原始 PNG 图片目录")
    parser.add_argument("--dst-dir", required=True, help="输出 JPG 目录")
    parser.add_argument("--start-frame", type=int, required=True, help="起始帧号，包含")
    parser.add_argument("--end-frame", type=int, required=True, help="结束帧号，包含")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPG质量，默认95")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有JPG")
    parser.add_argument("--strict", action="store_true", help="发现缺失或坏帧时返回非零退出码")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"原始图片目录不存在：{src_dir}")
    if not src_dir.is_dir():
        raise NotADirectoryError(f"原始图片路径不是目录：{src_dir}")
    if args.start_frame > args.end_frame:
        raise ValueError("--start-frame 不能大于 --end-frame")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality 必须位于 1～100")

    dst_dir.mkdir(parents=True, exist_ok=True)

    expected = args.end_frame - args.start_frame + 1
    written = 0
    skipped_existing = 0
    missing = []
    unreadable = []
    failed_write = []

    print("=" * 72)
    print(f"源目录：{src_dir.resolve()}")
    print(f"目标目录：{dst_dir.resolve()}")
    print(f"帧范围：{args.start_frame}～{args.end_frame}")
    print(f"期望帧数：{expected}")
    print(f"JPEG质量：{args.jpeg_quality}")
    print("=" * 72)

    for frame_idx in range(args.start_frame, args.end_frame + 1):
        source = src_dir / f"frame_{frame_idx:06d}.png"
        target = dst_dir / f"{frame_idx:06d}.jpg"

        if not source.exists():
            missing.append(frame_idx)
            continue

        if target.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            unreadable.append(frame_idx)
            continue

        ok = cv2.imwrite(
            str(target),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
        )
        if not ok:
            failed_write.append(frame_idx)
            continue

        written += 1

        processed = frame_idx - args.start_frame + 1
        if processed % 100 == 0 or frame_idx == args.end_frame:
            print(
                f"进度：{processed}/{expected}，"
                f"新写入={written}，已存在跳过={skipped_existing}"
            )

    output_files = sorted(dst_dir.glob("*.jpg"))

    print()
    print("=" * 72)
    print("转换完成")
    print(f"新写入：{written}")
    print(f"已存在并跳过：{skipped_existing}")
    print(f"缺失源文件：{len(missing)}")
    print(f"无法读取：{len(unreadable)}")
    print(f"写入失败：{len(failed_write)}")
    print(f"目标目录当前 JPG 总数：{len(output_files)}")

    if output_files:
        print(f"首个输出：{output_files[0].name}")
        print(f"最后输出：{output_files[-1].name}")

    if missing:
        print("缺失帧示例：", missing[:20])
    if unreadable:
        print("无法读取帧示例：", unreadable[:20])
    if failed_write:
        print("写入失败帧示例：", failed_write[:20])

    print("=" * 72)

    if args.strict and (missing or unreadable or failed_write):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())