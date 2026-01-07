"""
Preprocess images with CLAHE + edge enhancement and preview a collage:
preprocessed image | mask | overlay.

Usage:
  python code/preprocess_preview.py --image-dir train_images/forged --mask-dir train_masks
  python code/preprocess_preview.py --image-path train_images/forged/10.png --mask-dir train_masks
"""

import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


def list_images(paths: Iterable[Path]) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images: list[Path] = []
    for p in paths:
        if p.is_dir():
            images.extend([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts])
        elif p.is_file() and p.suffix.lower() in exts:
            images.append(p)
    return sorted(images)


def align_mask(mask_arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h_t, w_t = target_hw
    if mask_arr.shape[:2] != (h_t, w_t) and mask_arr.shape[:2] == (w_t, h_t):
        mask_arr = np.transpose(mask_arr)
    if mask_arr.shape[:2] != (h_t, w_t):
        mask_arr = np.array(
            Image.fromarray(mask_arr.astype(np.uint8)).resize((w_t, h_t), resample=Image.NEAREST)
        )
    return mask_arr


def load_mask(mask_dir: Path | None, stem: str, target_hw: tuple[int, int]) -> np.ndarray:
    if mask_dir is None:
        return np.zeros(target_hw, dtype=np.uint8)
    mask_path = mask_dir / f"{stem}.npy"
    if not mask_path.exists():
        return np.zeros(target_hw, dtype=np.uint8)
    arr = np.load(mask_path, allow_pickle=True)
    if arr.ndim == 3:
        arr = arr.max(axis=0)
    mask = (arr > 0).astype(np.uint8) * 255
    return align_mask(mask, target_hw)


def apply_overlay(img_np: np.ndarray, mask_bin: np.ndarray, color=(255, 0, 0), alpha=0.5) -> np.ndarray:
    overlay = img_np.copy()
    mask_bool = mask_bin.astype(bool)
    overlay[mask_bool] = (
        alpha * np.array(color, dtype=np.float32) + (1 - alpha) * overlay[mask_bool].astype(np.float32)
    ).astype(np.uint8)
    return overlay


def add_title(img_arr: np.ndarray, title: str, font, title_h: int = 24) -> Image.Image:
    panel = Image.new("RGB", (img_arr.shape[1], img_arr.shape[0] + title_h), color=(0, 0, 0))
    panel.paste(Image.fromarray(img_arr), (0, title_h))
    draw = ImageDraw.Draw(panel)
    if font:
        draw.text((5, 4), title, fill=(255, 255, 255), font=font)
    else:
        draw.text((5, 4), title, fill=(255, 255, 255))
    return panel


def preprocess_image(
    img_np: np.ndarray,
    clahe_clip: float,
    clahe_tile: int,
    canny_low: int,
    canny_high: int,
    edge_weight: float,
) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for CLAHE and edge detection.")
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    combined = cv2.addWeighted(enhanced, 1.0, edges_rgb, edge_weight, 0)
    return combined


def build_collage(orig_img: np.ndarray, pre_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_rgb = np.stack([mask] * 3, axis=-1).astype(np.uint8)
    overlay = apply_overlay(pre_img, mask > 0)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    panels = [
        ("Original", orig_img),
        ("Preprocessed", pre_img),
        ("Mask", mask_rgb),
        ("Overlay", overlay),
    ]
    titled = [add_title(arr, title, font) for title, arr in panels]
    margin = 8
    panel_widths = [img.size[0] for img in titled]
    panel_heights = [img.size[1] for img in titled]
    collage_w = sum(panel_widths) + margin * (len(titled) - 1)
    collage_h = max(panel_heights)
    collage = Image.new("RGB", (collage_w, collage_h), color=(0, 0, 0))
    x = 0
    for panel in titled:
        collage.paste(panel, (x, 0))
        x += panel.size[0] + margin
    return np.array(collage)


def process_one(
    img_path: Path,
    mask_dir: Path | None,
    clahe_clip: float,
    clahe_tile: int,
    canny_low: int,
    canny_high: int,
    edge_weight: float,
) -> tuple[str, np.ndarray, np.ndarray]:
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img_np = np.array(img)
    pre = preprocess_image(img_np, clahe_clip, clahe_tile, canny_low, canny_high, edge_weight)
    mask = load_mask(mask_dir, img_path.stem, (pre.shape[0], pre.shape[1]))
    collage = build_collage(img_np, pre, mask)
    return img_path.stem, pre, collage


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview CLAHE + edge preprocessing collages.")
    parser.add_argument("--image-dir", action="append", default=[], help="Directory with images (repeatable).")
    parser.add_argument("--image-path", action="append", default=[], help="Single image path (repeatable).")
    parser.add_argument("--mask-dir", default=None, help="Directory with .npy masks (optional).")
    parser.add_argument("--save-dir", default=None, help="Optional dir to save preprocessed images.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle images before processing.")
    parser.add_argument("--delay", type=int, default=0, help="Unused (kept for backward compatibility).")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit.")
    parser.add_argument("--clahe-tile", type=int, default=8, help="CLAHE tile size.")
    parser.add_argument("--canny-low", type=int, default=50, help="Canny low threshold.")
    parser.add_argument("--canny-high", type=int, default=150, help="Canny high threshold.")
    parser.add_argument("--edge-weight", type=float, default=0.35, help="Edge blend weight.")
    args = parser.parse_args()

    image_dirs = [Path(p) for p in args.image_dir]
    image_paths = [Path(p) for p in args.image_path]
    images = list_images(image_dirs + image_paths)
    if not images:
        print("No images found. Provide --image-dir and/or --image-path.", file=sys.stderr)
        return 1
    if args.shuffle:
        rng = np.random.default_rng(42)
        rng.shuffle(images)
    if args.max_images is not None:
        images = images[: args.max_images]

    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    worker = partial(
        process_one,
        mask_dir=mask_dir,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        edge_weight=args.edge_weight,
    )

    if cv2 is None:
        print("OpenCV not available; install it to use CLAHE + edge detection.", file=sys.stderr)
        return 1

    with Pool(processes=args.workers) as pool:
        for stem, pre_img, collage in tqdm(pool.imap(worker, images), total=len(images), desc="Preprocess"):
            if save_dir is not None:
                out_path = save_dir / f"{stem}.png"
                Image.fromarray(collage).save(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
