#!/usr/bin/env python3
"""
make_synth_multipanel.py

Builds "supplemental-like" multi-panel composite images from your train set.

Outputs:
  outdir/
    images/synth_000000.png ...
    masks/synth_000000.npy  (uint8 HxW, {0,1})
    synthetic.csv           (id,label)
    synthetic_meta.npy      (optional metadata list)

Key feature:
  - CoverageSampler guarantees every authentic + forged image is used at least once
    per "pass" through the queues (coverage, no random-missing forever).
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


# -----------------------------
# IO helpers
# -----------------------------

def pil_load_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


def load_mask_for_image(img_path: Path, masks_root: Path) -> Optional[np.ndarray]:
    """
    Tries a few common conventions. Adjust if your tree is different.

    Supports:
      1) train_masks/{id}.npy
      2) train_masks/forged/{id}.npy
      3) train_masks/{authentic|forged}/{id}.npy
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
    """Keep aspect ratio; pad to target."""
    return ImageOps.pad(img, target_wh, color=pad_color, method=Image.Resampling.BICUBIC)


def resize_mask_with_pad(mask: np.ndarray, target_wh: Tuple[int, int]) -> np.ndarray:
    """Keep aspect ratio; pad mask with 0; nearest neighbor."""
    m = Image.fromarray(mask * 255)
    m = ImageOps.pad(m, target_wh, color=0, method=Image.Resampling.NEAREST)
    return (np.array(m) > 0).astype(np.uint8)


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
# Layout / style sampling
# -----------------------------

def choose_layout(rng: random.Random) -> Tuple[int, int]:
    layouts = [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3)]
    probs   = [0.10, 0.10, 0.30, 0.25, 0.15, 0.10]
    return rng.choices(layouts, weights=probs, k=1)[0]


def choose_forged_count(rng: random.Random, n_panels: int) -> int:
    r = rng.random()
    if r < 0.40:
        return 0
    if r < 0.80:
        return 1
    if r < 0.95:
        return min(2, n_panels)
    return min(3, n_panels)


def choose_cell_size(rng: random.Random) -> Tuple[int, int]:
    cell_w = rng.choice([360, 420, 512, 640, 700, 800])
    cell_h = rng.choice([256, 360, 420, 512, 600, 700])

    cell_w += rng.randint(-24, 24)
    cell_h += rng.randint(-24, 24)

    cell_w = max(220, cell_w)
    cell_h = max(220, cell_h)
    return cell_w, cell_h


def choose_background_whiteish(rng: random.Random) -> Tuple[int, int, int]:
    v = rng.randint(242, 255)
    return (v, v, v)


# -----------------------------
# Composite builder
# -----------------------------

def build_composite(
    sampler: CoverageSampler,
    masks_root: Path,
    out_img_path: Path,
    out_mask_path: Path,
    seed: int,
    max_forged_mask_retries: int = 10,
) -> Tuple[int, Dict[str, Any]]:
    rng = random.Random(seed)
    np.random.seed(seed)

    rows, cols = choose_layout(rng)
    n_panels = rows * cols
    n_forged = choose_forged_count(rng, n_panels)
    forged_slots = set(rng.sample(range(n_panels), k=n_forged)) if n_forged > 0 else set()

    cell_w, cell_h = choose_cell_size(rng)

    gutter = rng.randint(10, 40)
    outer  = rng.randint(20, 80)
    bg = choose_background_whiteish(rng)

    canvas_w = outer * 2 + cols * cell_w + (cols - 1) * gutter
    canvas_h = outer * 2 + rows * cell_h + (rows - 1) * gutter

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    mask_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    panel_labels: List[int] = []
    panel_ids: List[str] = []

    for idx in range(n_panels):
        r = idx // cols
        c = idx % cols
        x0 = outer + c * (cell_w + gutter)
        y0 = outer + r * (cell_h + gutter)

        if idx in forged_slots:
            chosen_p = None
            chosen_m = None
            for _ in range(max_forged_mask_retries):
                p = sampler.next_forged()
                m = load_mask_for_image(p, masks_root)
                if m is not None:
                    chosen_p = p
                    chosen_m = m
                    break

            if chosen_p is None or chosen_m is None:
                # fallback to authentic if forged masks are unexpectedly missing
                p = sampler.next_auth()
                img = pil_load_rgb(p)
                img_r = resize_with_pad(img, (cell_w, cell_h), pad_color=bg)
                canvas.paste(img_r, (x0, y0))
                panel_labels.append(0)
                panel_ids.append(p.stem)
                continue

            img = pil_load_rgb(chosen_p)
            img_r = resize_with_pad(img, (cell_w, cell_h), pad_color=bg)
            m_r = resize_mask_with_pad(chosen_m, (cell_w, cell_h))

            canvas.paste(img_r, (x0, y0))
            mask_canvas[y0:y0 + cell_h, x0:x0 + cell_w] |= m_r

            panel_labels.append(1)
            panel_ids.append(chosen_p.stem)
        else:
            p = sampler.next_auth()
            img = pil_load_rgb(p)
            img_r = resize_with_pad(img, (cell_w, cell_h), pad_color=bg)
            canvas.paste(img_r, (x0, y0))

            panel_labels.append(0)
            panel_ids.append(p.stem)

    canvas.save(out_img_path, format="PNG", optimize=True)
    np.save(out_mask_path, mask_canvas.astype(np.uint8))

    label = int(mask_canvas.sum() > 0)
    meta: Dict[str, Any] = {
        "rows": rows,
        "cols": cols,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "gutter": gutter,
        "outer": outer,
        "bg": bg,
        "panel_labels": panel_labels,
        "panel_ids": panel_ids,
        "seed": seed,
    }
    return label, meta


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", type=str, required=True,
                    help="Root containing authentic/ and forged/ folders")
    ap.add_argument("--train-masks", type=str, required=True,
                    help="Root containing mask .npy files")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    train_images = Path(args.train_images)
    masks_root = Path(args.train_masks)
    outdir = Path(args.outdir)

    out_img_dir = outdir / "images"
    out_m_dir = outdir / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_m_dir.mkdir(parents=True, exist_ok=True)

    auth_paths = sorted((train_images / "authentic").glob("*.png"))
    forged_paths = sorted((train_images / "forged").glob("*.png"))
    if not auth_paths:
        raise RuntimeError(f"No authentic images found at: {train_images / 'authentic'}")
    if not forged_paths:
        raise RuntimeError(f"No forged images found at: {train_images / 'forged'}")

    sampler = CoverageSampler(auth_paths, forged_paths, seed=args.seed)
    rng = random.Random(args.seed)

    rows_csv: List[Tuple[str, int]] = []
    metas: List[Tuple[str, Dict[str, Any]]] = []

    for i in tqdm(range(args.n), desc="Generating composites", unit="img"):
        sid = f"synth_{i:06d}"
        out_img = out_img_dir / f"{sid}.png"
        out_mask = out_m_dir / f"{sid}.npy"

        label, meta = build_composite(
            sampler=sampler,
            masks_root=masks_root,
            out_img_path=out_img,
            out_mask_path=out_mask,
            seed=rng.randint(0, 10**9),
        )

        rows_csv.append((sid, label))
        metas.append((sid, meta))

    csv_path = outdir / "synthetic.csv"
    with open(csv_path, "w") as f:
        f.write("id,label\n")
        for sid, label in rows_csv:
            f.write(f"{sid},{label}\n")

    meta_path = outdir / "synthetic_meta.npy"
    np.save(meta_path, metas, allow_pickle=True)

    print("wrote:", csv_path)
    print("wrote:", meta_path)


if __name__ == "__main__":
    main()
