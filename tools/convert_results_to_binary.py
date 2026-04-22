"""
Convert visualization result images (overlay) to pure binary masks.
- Mask region  → pure white (255)
- Background   → pure black  (0)

The result images are MMSegmentation overlays where the predicted lesion
region is blended with a light color (pinkish-white) on top of the original
dark skin image. Otsu thresholding on the grayscale image separates them.

Usage (chạy từ thư mục gốc của project):

  # Chạy 20 ảnh đầu, lưu vào binary_masks/
  python tools/convert_results_to_binary.py --limit 20

  # Chỉ định rõ đường dẫn nguồn và đích
  python tools/convert_results_to_binary.py --src results --dst binary_masks --limit 20

  # Dùng đường dẫn tuyệt đối
  python tools/convert_results_to_binary.py \\
      --src /home/cham_anh/lsnet-base-for-medical-image-segmentation/results \\
      --dst /home/cham_anh/lsnet-base-for-medical-image-segmentation/binary_masks \\
      --limit 20

  # Threshold cố định thay vì Otsu
  python tools/convert_results_to_binary.py --limit 20 --threshold 128

Kết quả lưu tại: binary_masks/  (cùng cấp với results/)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def to_binary(img_path: Path, threshold: Optional[int]) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if threshold is None:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return binary


def main():
    parser = argparse.ArgumentParser(description="Convert result overlays to binary masks")
    parser.add_argument("--src", default="results", help="Source directory (default: results)")
    parser.add_argument("--dst", default="binary_masks", help="Output directory (default: binary_masks)")
    parser.add_argument("--threshold", type=int, default=None,
                        help="Fixed threshold 0-255. Omit to use Otsu auto-threshold.")
    parser.add_argument("--ext", default=".png", help="Output extension (default: .png)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N images (default: all)")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        print(f"[ERROR] Source directory not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    dst_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in image_exts)

    if not files:
        print(f"[WARN] No images found in {src_dir}")
        sys.exit(0)

    if args.limit is not None:
        files = files[: args.limit]

    mode = f"fixed threshold={args.threshold}" if args.threshold is not None else "Otsu auto-threshold"
    print(f"Converting {len(files)} images | mode: {mode}")
    print(f"  src : {src_dir.resolve()}")
    print(f"  dst : {dst_dir.resolve()}\n")

    ok = fail = 0
    for path in files:
        out_path = dst_dir / (path.stem + args.ext)
        try:
            binary = to_binary(path, args.threshold)
            cv2.imwrite(str(out_path), binary)
            print(f"  [OK]   {path.name}  →  {out_path.name}")
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {path.name}: {e}")
            fail += 1

    print(f"\nDone: {ok} saved, {fail} failed")
    print(f"Kết quả tại: {dst_dir.resolve()}/")


if __name__ == "__main__":
    main()
