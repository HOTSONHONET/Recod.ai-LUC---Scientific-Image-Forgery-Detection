#!/usr/bin/env python3
"""
make_yolo_panel_dataset.py

Creates a synthetic dataset to train a YOLO detector to find panels (subfigures)
in multi-panel scientific images.

Outputs:
  outdir/
    images/
      train/*.png
      val/*.png
    labels/
      train/*.txt   (YOLO format: cls xc yc w h normalized)
      val/*.txt
    data.yaml
    meta_train.jsonl
    meta_val.jsonl

Single class:
  class_id 0 = "panel"

This script is designed to mimic supplemental-like figures:
- multi-panel layouts from 2x2 to 5x6
- optional grid lines
- optional blank panels
- white-ish backgrounds + random gutters/outer margins
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from tqdm import tqdm


# -----------------------------
# IO helpers
# -----------------------------

def pil_load_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


def load_mask_for_image(img_path: Path, masks_root: Path) -> Optional[np.ndarray]:
    """
    Same logic you already had. Not required for panel detection,
    but kept here in case you want to drive "forged-like" texture distribution.
    """
    stem = img_path.stem
    candidates = [
        masks_root / f"{stem}.npy",
        masks_root / "forged" / f"{stem}.npy",
        masks_root / img_path.parent.name / f"{stem}.npy",
    ]
    for c in candidates:
        if c.exists():
            arr = np.load(c, allow_pickle=True)
            if arr.ndim == 3:
                arr = arr.max(axis=0)
            return (arr > 0).astype(np.uint8)
    return None


def resize_with_pad(img: Image.Image, target_wh: Tuple[int, int], pad_color=(255, 255, 255)) -> Image.Image:
    return ImageOps.pad(img, target_wh, color=pad_color, method=Image.Resampling.BICUBIC)


# -----------------------------
# Coverage sampler
# -----------------------------

class CoverageSampler:
    """Cycles through shuffled lists so every file appears once per pass."""
    def __init__(self, auth_paths: List[Path], forged_paths: List[Path], seed: int = 1337):
        self.rng = random.Random(seed)
        self.auth_all = list(auth_paths)
        self.forged_all = list(forged_paths)
        self.auth_q: List[Path] = []
        self.forged_q: List[Path] = []
        self._refill_auth()
        self._refill_forged()

    def _refill_auth(self) -> None:
        self.auth_q = self.auth_all[:]
        self.rng.shuffle(self.auth_q)

    def _refill_forged(self) -> None:
        self.forged_q = self.forged_all[:]
        self.rng.shuffle(self.forged_q)

    def next_auth(self) -> Path:
        if not self.auth_q:
            self._refill_auth()
        return self.auth_q.pop()

    def next_forged(self) -> Path:
        if not self.forged_q:
            self._refill_forged()
        return self.forged_q.pop()


# -----------------------------
# Panel + YOLO label helpers
# -----------------------------

@dataclass
class PanelBox:
    x1: int
    y1: int
    x2: int
    y2: int
    panel_id: str
    is_blank: bool
    src_stem: str

    def to_yolo(self, W: int, H: int, cls_id: int = 0) -> str:
        # YOLO format: cls xc yc w h (all normalized 0..1)
        xc = (self.x1 + self.x2) / 2.0 / W
        yc = (self.y1 + self.y2) / 2.0 / H
        w = (self.x2 - self.x1) / float(W)
        h = (self.y2 - self.y1) / float(H)
        # clamp for safety
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        return f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


# -----------------------------
# Sampling knobs (match supplement vibe)
# -----------------------------

def choose_layout(rng: random.Random, min_r: int, max_r: int, min_c: int, max_c: int) -> Tuple[int, int]:
    # Bias towards common shapes while allowing full range
    rows = rng.randint(min_r, max_r)
    cols = rng.randint(min_c, max_c)

    # Small bias: 2-4 rows/cols are most common; 5-6 appear but less often
    if rng.random() < 0.65:
        rows = rng.choice([2, 3, 3, 4, 2])
    if rng.random() < 0.65:
        cols = rng.choice([2, 3, 4, 3, 2, 5])

    rows = max(min_r, min(max_r, rows))
    cols = max(min_c, min(max_c, cols))
    return rows, cols


def choose_cell_size(rng: random.Random) -> Tuple[int, int]:
    # Slightly smaller helps fit 5x6 without huge canvases
    cell_w = rng.choice([220, 256, 280, 320, 360, 420, 512])
    cell_h = rng.choice([200, 220, 256, 280, 320, 360, 420])
    cell_w += rng.randint(-20, 20)
    cell_h += rng.randint(-20, 20)
    cell_w = max(180, cell_w)
    cell_h = max(180, cell_h)
    return cell_w, cell_h


def choose_background_whiteish(rng: random.Random) -> Tuple[int, int, int]:
    v = rng.randint(242, 255)
    # tiny tint sometimes
    if rng.random() < 0.2:
        return (v, min(255, v + rng.randint(-2, 2)), min(255, v + rng.randint(-2, 2)))
    return (v, v, v)


def maybe_add_background_texture(img: Image.Image, rng: random.Random) -> Image.Image:
    # very subtle noise / blur to mimic paper scanning / compression
    if rng.random() < 0.4:
        arr = np.array(img).astype(np.int16)
        noise = rng.randint(0, 2)
        if noise > 0:
            n = np.random.randint(-noise, noise + 1, size=arr.shape, dtype=np.int16)
            arr = np.clip(arr + n, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
    if rng.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.choice([0.3, 0.5, 0.8])))
    return img


def draw_grid_lines(draw: ImageDraw.ImageDraw, rows: int, cols: int, outer: int, gutter: int,
                    cell_ws: List[int], cell_hs: List[int], line_w: int, line_color: Tuple[int, int, int]) -> None:
    """
    Draws separators between cells. cell_ws and cell_hs are per-col/per-row sizes used.
    """
    # Vertical separators
    x = outer
    for c in range(cols - 1):
        x += cell_ws[c]
        # line in the middle of gutter
        lx = x + gutter // 2
        draw.line([(lx, 0), (lx, 10_000)], fill=line_color, width=line_w)
        x += gutter

    # Horizontal separators
    y = outer
    for r in range(rows - 1):
        y += cell_hs[r]
        ly = y + gutter // 2
        draw.line([(0, ly), (10_000, ly)], fill=line_color, width=line_w)
        y += gutter


# -----------------------------
# Composite builder (with panel bboxes)
# -----------------------------

def build_composite_with_panel_labels(
    sampler: CoverageSampler,
    out_img_path: Path,
    out_lbl_path: Path,
    seed: int,
    *,
    min_rows: int,
    max_rows: int,
    min_cols: int,
    max_cols: int,
    prob_use_forged: float,
    prob_blank_panel: float,
    prob_grid_lines: float,
    expand_bbox_px: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    np.random.seed(seed)

    rows, cols = choose_layout(rng, min_rows, max_rows, min_cols, max_cols)

    # Make per-row/per-col sizes to allow irregular grids like real figures
    cell_ws = []
    cell_hs = []
    base_w, base_h = choose_cell_size(rng)
    for _ in range(cols):
        cell_ws.append(max(160, base_w + rng.randint(-30, 30)))
    for _ in range(rows):
        cell_hs.append(max(160, base_h + rng.randint(-30, 30)))

    gutter = rng.randint(8, 40)
    outer = rng.randint(10, 80)
    bg = choose_background_whiteish(rng)

    canvas_w = outer * 2 + sum(cell_ws) + (cols - 1) * gutter
    canvas_h = outer * 2 + sum(cell_hs) + (rows - 1) * gutter

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)

    panel_boxes: List[PanelBox] = []
    used_sources: List[str] = []

    # Fill panels
    y = outer
    for r in range(rows):
        x = outer
        for c in range(cols):
            cw, ch = cell_ws[c], cell_hs[r]

            x1 = x
            y1 = y
            x2 = x + cw
            y2 = y + ch

            # Decide blank
            is_blank = rng.random() < prob_blank_panel

            if is_blank:
                # Leave blank white-ish panel (common in supplements)
                # optionally draw faint border
                if rng.random() < 0.25:
                    draw = ImageDraw.Draw(canvas)
                    border_col = tuple(max(0, b - rng.randint(5, 20)) for b in bg)
                    draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=border_col, width=rng.choice([1, 1, 2]))
                src_stem = "BLANK"
            else:
                # Choose source image
                use_f = (rng.random() < prob_use_forged)
                p = sampler.next_forged() if use_f else sampler.next_auth()
                used_sources.append(p.stem)
                img = pil_load_rgb(p)
                img_r = resize_with_pad(img, (cw, ch), pad_color=bg)
                canvas.paste(img_r, (x1, y1))
                src_stem = p.stem

                # Optional: add tiny border to mimic panel boundaries
                if rng.random() < 0.35:
                    draw = ImageDraw.Draw(canvas)
                    border_col = (rng.randint(180, 230),) * 3
                    draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=border_col, width=rng.choice([1, 1, 2]))

            # bbox expand (help detector include border/title-ish edges)
            ex = expand_bbox_px + rng.randint(0, 6)
            bx1 = max(0, x1 - ex)
            by1 = max(0, y1 - ex)
            bx2 = min(canvas_w, x2 + ex)
            by2 = min(canvas_h, y2 + ex)

            panel_id = f"r{r}_c{c}"
            panel_boxes.append(PanelBox(bx1, by1, bx2, by2, panel_id, is_blank, src_stem))

            x = x2 + gutter
        y = y + cell_hs[r] + gutter

    # Optional gridlines (like your example images)
    if rng.random() < prob_grid_lines:
        draw = ImageDraw.Draw(canvas)
        line_color = (rng.randint(150, 210),) * 3
        line_w = rng.choice([1, 1, 2, 3])
        draw_grid_lines(draw, rows, cols, outer, gutter, cell_ws, cell_hs, line_w, line_color)

    canvas = maybe_add_background_texture(canvas, rng)

    # Save image
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_lbl_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_img_path, format="PNG", optimize=True)

    # Save YOLO labels
    W, H = canvas.size
    lines = [pb.to_yolo(W, H, cls_id=0) for pb in panel_boxes]
    out_lbl_path.write_text("\n".join(lines) + "\n")

    meta = {
        "seed": seed,
        "rows": rows,
        "cols": cols,
        "canvas_w": W,
        "canvas_h": H,
        "outer": outer,
        "gutter": gutter,
        "bg": bg,
        "prob_use_forged": prob_use_forged,
        "prob_blank_panel": prob_blank_panel,
        "prob_grid_lines": prob_grid_lines,
        "expand_bbox_px": expand_bbox_px,
        "panel_boxes_xyxy": [
            {
                "panel_id": pb.panel_id,
                "xyxy": [pb.x1, pb.y1, pb.x2, pb.y2],
                "is_blank": pb.is_blank,
                "src_stem": pb.src_stem,
            }
            for pb in panel_boxes
        ],
        "used_sources": used_sources,
    }
    return meta


# -----------------------------
# Main
# -----------------------------

def write_data_yaml(outdir: Path) -> None:
    yaml = f"""# Ultralytics YOLO dataset
path: {outdir.as_posix()}
train: images/train
val: images/val

names:
  0: panel
"""
    (outdir / "data.yaml").write_text(yaml)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", type=str, required=True,
                    help="Root containing authentic/ and forged/ folders (png/jpg)")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--n", type=int, default=12000, help="Total synthetic images")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)

    # layout controls
    ap.add_argument("--min-rows", type=int, default=2)
    ap.add_argument("--max-rows", type=int, default=5)
    ap.add_argument("--min-cols", type=int, default=2)
    ap.add_argument("--max-cols", type=int, default=6)

    # style controls
    ap.add_argument("--prob-use-forged", type=float, default=0.55,
                    help="probability to draw a panel from forged pool (doesn't affect bbox)")
    ap.add_argument("--prob-blank-panel", type=float, default=0.12,
                    help="probability a panel is blank (still labeled as a panel bbox)")
    ap.add_argument("--prob-grid-lines", type=float, default=0.55,
                    help="probability to draw grid/separator lines")
    ap.add_argument("--expand-bbox-px", type=int, default=8,
                    help="expand each panel bbox by this many pixels")

    args = ap.parse_args()

    train_images = Path(args.train_images)
    outdir = Path(args.outdir)
    rng = random.Random(args.seed)

    auth_paths = sorted((train_images / "authentic").glob("*.png")) + sorted((train_images / "authentic").glob("*.jpg"))
    forged_paths = sorted((train_images / "forged").glob("*.png")) + sorted((train_images / "forged").glob("*.jpg"))
    if not auth_paths:
        raise RuntimeError(f"No authentic images found at: {train_images / 'authentic'}")
    if not forged_paths:
        raise RuntimeError(f"No forged images found at: {train_images / 'forged'}")

    sampler = CoverageSampler(auth_paths, forged_paths, seed=args.seed)

    n_val = int(args.n * args.val_ratio)
    n_train = args.n - n_val

    img_train_dir = outdir / "images" / "train"
    lbl_train_dir = outdir / "labels" / "train"
    img_val_dir = outdir / "images" / "val"
    lbl_val_dir = outdir / "labels" / "val"
    img_train_dir.mkdir(parents=True, exist_ok=True)
    lbl_train_dir.mkdir(parents=True, exist_ok=True)
    img_val_dir.mkdir(parents=True, exist_ok=True)
    lbl_val_dir.mkdir(parents=True, exist_ok=True)

    meta_train_path = outdir / "meta_train.jsonl"
    meta_val_path = outdir / "meta_val.jsonl"
    if meta_train_path.exists():
        meta_train_path.unlink()
    if meta_val_path.exists():
        meta_val_path.unlink()

    # Generate train
    with meta_train_path.open("a") as fmeta:
        for i in tqdm(range(n_train), desc="Generating TRAIN composites", unit="img"):
            sid = f"synth_train_{i:06d}"
            img_path = img_train_dir / f"{sid}.png"
            lbl_path = lbl_train_dir / f"{sid}.txt"

            meta = build_composite_with_panel_labels(
                sampler=sampler,
                out_img_path=img_path,
                out_lbl_path=lbl_path,
                seed=rng.randint(0, 10**9),
                min_rows=args.min_rows,
                max_rows=args.max_rows,
                min_cols=args.min_cols,
                max_cols=args.max_cols,
                prob_use_forged=args.prob_use_forged,
                prob_blank_panel=args.prob_blank_panel,
                prob_grid_lines=args.prob_grid_lines,
                expand_bbox_px=args.expand_bbox_px,
            )
            fmeta.write(json.dumps(meta) + "\n")

    # Generate val
    with meta_val_path.open("a") as fmeta:
        for i in tqdm(range(n_val), desc="Generating VAL composites", unit="img"):
            sid = f"synth_val_{i:06d}"
            img_path = img_val_dir / f"{sid}.png"
            lbl_path = lbl_val_dir / f"{sid}.txt"

            meta = build_composite_with_panel_labels(
                sampler=sampler,
                out_img_path=img_path,
                out_lbl_path=lbl_path,
                seed=rng.randint(0, 10**9),
                min_rows=args.min_rows,
                max_rows=args.max_rows,
                min_cols=args.min_cols,
                max_cols=args.max_cols,
                prob_use_forged=args.prob_use_forged,
                prob_blank_panel=args.prob_blank_panel,
                prob_grid_lines=args.prob_grid_lines,
                expand_bbox_px=args.expand_bbox_px,
            )
            fmeta.write(json.dumps(meta) + "\n")

    write_data_yaml(outdir)
    print("✅ wrote:", (outdir / "data.yaml"))
    print("✅ train images:", img_train_dir)
    print("✅ val images:", img_val_dir)
    print("✅ train labels:", lbl_train_dir)
    print("✅ val labels:", lbl_val_dir)


if __name__ == "__main__":
    main()
