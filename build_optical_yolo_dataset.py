#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
SEQUENCES = {
    'seq01': ('train', 'data/frames/optical_seq01', 'outputs/optical_seq01/final/labels', 236),
    'seq02': ('train', 'data/frames/optical_seq02', 'outputs/optical_seq02/final/labels', 251),
    'seq03': ('val',   'data/frames/optical_seq03', 'outputs/optical_seq03/final/labels', 108),
    'seq04': ('test',  'data/frames/optical_seq04', 'outputs/optical_seq04/final/labels', 275),
}


def parse_args():
    p = argparse.ArgumentParser(description='将4段SAM2结果整理为单类别YOLO数据集。')
    p.add_argument('--repo-root', default='.')
    p.add_argument('--output-dir', default='datasets/optical_drone_yolo')
    p.add_argument('--mode', choices=['copy', 'symlink'], default='copy')
    p.add_argument('--allow-missing-labels-as-empty', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def transfer(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == 'copy':
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def validate_label(path: Path):
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        return 0, []
    errors, objects = [], 0
    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f'{path}:{line_no}: 字段数不是5')
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f'{path}:{line_no}: 存在非数值字段')
            continue
        if cls != 0:
            errors.append(f'{path}:{line_no}: class_id={cls}，预期为0')
        for name, value in [('cx', cx), ('cy', cy), ('w', w), ('h', h)]:
            if not 0.0 <= value <= 1.0:
                errors.append(f'{path}:{line_no}: {name}={value} 超出[0,1]')
        if w <= 0 or h <= 0:
            errors.append(f'{path}:{line_no}: bbox宽高必须大于0')
        objects += 1
    return objects, errors


def main():
    args = parse_args()
    repo = Path(args.repo_root).resolve()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = repo / out
    out = out.resolve()

    if out.exists():
        if args.overwrite:
            shutil.rmtree(out)
        elif any(out.iterdir()):
            raise SystemExit(f'输出目录非空: {out}；确认后添加 --overwrite')

    for split in ('train', 'val', 'test'):
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    rows, errors = [], []
    stats = {s: {'images': 0, 'labeled': 0, 'empty': 0, 'objects': 0} for s in ('train','val','test')}

    for seq, (split, frames_rel, labels_rel, expected) in SEQUENCES.items():
        frames = repo / frames_rel
        labels = repo / labels_rel
        if not frames.is_dir():
            errors.append(f'图片目录不存在: {frames}')
            continue
        if not labels.is_dir():
            errors.append(f'标签目录不存在: {labels}')
            continue

        images = sorted(p for p in frames.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if len(images) != expected:
            errors.append(f'{seq}: 图片数={len(images)}，预期={expected}')

        for image in images:
            source_label = labels / f'{image.stem}.txt'
            if not source_label.exists() and not args.allow_missing_labels_as_empty:
                errors.append(f'{seq}: 缺少标签 {source_label}')
                continue

            if source_label.exists():
                obj_count, label_errors = validate_label(source_label)
                errors.extend(label_errors)
            else:
                obj_count = 0

            new_stem = f'{seq}_{image.stem}'
            dst_img = out / 'images' / split / f'{new_stem}{image.suffix.lower()}'
            dst_lbl = out / 'labels' / split / f'{new_stem}.txt'
            transfer(image, dst_img, args.mode)
            if source_label.exists():
                transfer(source_label, dst_lbl, args.mode)
            else:
                dst_lbl.write_text('', encoding='utf-8')

            stats[split]['images'] += 1
            stats[split]['objects'] += obj_count
            stats[split]['labeled' if obj_count else 'empty'] += 1
            rows.append({
                'sequence': seq,
                'split': split,
                'source_image': str(image.resolve()),
                'source_label': str(source_label.resolve()) if source_label.exists() else '',
                'dataset_image': str(dst_img.resolve()),
                'dataset_label': str(dst_lbl.resolve()),
                'object_count': obj_count,
            })

    if errors:
        out.mkdir(parents=True, exist_ok=True)
        report = out / 'build_errors.txt'
        report.write_text('\n'.join(errors) + '\n', encoding='utf-8')
        print(f'发现 {len(errors)} 个问题，错误报告: {report}')
        for item in errors[:20]:
            print(' -', item)
        raise SystemExit(1)

    manifest = out / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (out / 'data.yaml').write_text(
        f'path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: drone\n',
        encoding='utf-8'
    )

    lines = [f'dataset_root={out}', f'mode={args.mode}', 'class_0=drone']
    for split in ('train', 'val', 'test'):
        s = stats[split]
        lines += [
            f'{split}_images={s["images"]}',
            f'{split}_labeled_images={s["labeled"]}',
            f'{split}_empty_images={s["empty"]}',
            f'{split}_objects={s["objects"]}',
        ]
    (out / 'dataset_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('=' * 72)
    print('YOLO数据集创建完成')
    print('数据集目录:', out)
    print('配置文件:', out / 'data.yaml')
    for split in ('train', 'val', 'test'):
        s = stats[split]
        print(f'{split:5s}: images={s["images"]}, labeled={s["labeled"]}, empty={s["empty"]}, objects={s["objects"]}')
    print('=' * 72)


if __name__ == '__main__':
    main()
