#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import random
from pathlib import Path

import cv2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def main():
    parser = argparse.ArgumentParser(description="随机抽取YOLO标注并绘制bbox。")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    if not images_dir.is_dir():
        raise SystemExit(f"图片目录不存在: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"标签目录不存在: {labels_dir}")

    images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise SystemExit("没有找到图片。")

    random.seed(args.seed)
    selected = random.sample(images, min(args.count, len(images)))
    output_dir.mkdir(parents=True, exist_ok=True)

    box_count = 0
    empty_count = 0

    for image_path in selected:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        lines = []

        if label_path.exists():
            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        if not lines:
            empty_count += 1
            cv2.putText(
                image, "EMPTY LABEL", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
            )

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                print(f"标签格式异常: {label_path} -> {line}")
                continue

            class_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = int(round((cx - bw / 2) * width))
            y1 = int(round((cy - bh / 2) * height))
            x2 = int(round((cx + bw / 2) * width))
            y2 = int(round((cy + bh / 2) * height))

            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image, f"class={class_id}", (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
            )
            box_count += 1

        cv2.imwrite(str(output_dir / image_path.name), image)

    print("=" * 64)
    print(f"随机检查图片: {len(selected)}")
    print(f"绘制bbox数量: {box_count}")
    print(f"空标签样本数量: {empty_count}")
    print(f"输出目录: {output_dir.resolve()}")
    print("=" * 64)

if __name__ == "__main__":
    main()
