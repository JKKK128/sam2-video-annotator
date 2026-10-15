from pathlib import Path

import cv2


src = Path("/mnt/d/projects/dronedecode/images1")
dst = Path("data/frames/optical_images1")

dst.mkdir(parents=True, exist_ok=True)

count = 0
missing_count = 0
failed_count = 0

for frame_idx in range(468, 1127):
    input_path = src / f"frame_{frame_idx:06d}.png"
    output_path = dst / f"{frame_idx:06d}.jpg"

    if not input_path.exists():
        print("missing:", input_path)
        missing_count += 1
        continue

    image = cv2.imread(str(input_path))

    if image is None:
        print("read failed:", input_path)
        failed_count += 1
        continue

    ok = cv2.imwrite(
        str(output_path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    if not ok:
        print("write failed:", output_path)
        failed_count += 1
        continue

    count += 1
    if count % 20 ==0:
        print(f"converted:{count}/659",flush=True)

print()
print("converted:", count)
print("missing:", missing_count)
print("failed:", failed_count)
print("output:", dst.resolve())
