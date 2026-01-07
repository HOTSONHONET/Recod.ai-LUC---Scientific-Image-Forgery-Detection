"""
Inference script to generate predicted masks for forged images per fold, and collages:
input image | predicted mask | ground-truth mask | pred overlay | gt overlay.

Usage:
    python code/infer.py --weights-dir analysis/models --splits-dir analysis/splits --outdir analysis/preds
Options:
    --folds 0 1 2 3 4   # folds to run; default: all detected weight files
    --threshold 0.5     # probability threshold
    --img-size 448      # resize to match training
"""

import argparse
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from model import DinoSegModel


class ForgedOnlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int = 448):
        forged_df = df[df["label"] == 1].reset_index(drop=True)
        self.df = forged_df
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                self.to_tensor,
                self.normalize,
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        case_id = row["case_id"]
        mask_path = row.get("mask_path", "")
        image_path = row["image_path"]
        with open(image_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        orig_size = img.size  # (W,H)
        img_t = self.img_transform(img)
        return img_t, case_id, orig_size, mask_path, image_path


def load_split(splits_dir: Path, fold: int) -> pd.DataFrame:
    train_path = splits_dir / f"train_fold{fold}.csv"
    val_path = splits_dir / f"val_fold{fold}.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing splits for fold {fold} in {splits_dir}")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return pd.concat([train_df, val_df], ignore_index=True)


def infer_fold(
    fold: int,
    weights_path: Path,
    splits_dir: Path,
    outdir: Path,
    model_name: str,
    img_size: int,
    threshold: float,
    device: torch.device,
    samples: int | None,
) -> None:
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights for fold {fold}: {weights_path}")

    state = torch.load(weights_path, map_location=device)
    model_state = state["model_state"]

    def infer_img_size_from_state(state_dict: dict, fallback: int) -> int:
        pos_embed = state_dict.get("backbone.pos_embed")
        patch_weight = state_dict.get("backbone.patch_embed.proj.weight")
        if pos_embed is None or patch_weight is None:
            return fallback
        tokens = pos_embed.shape[1]
        grid_tokens = tokens - 1
        grid = int(math.sqrt(grid_tokens))
        if grid * grid != grid_tokens:
            return fallback
        patch = int(patch_weight.shape[-1])
        return grid * patch

    inferred_img_size = infer_img_size_from_state(model_state, img_size)
    if inferred_img_size != img_size:
        print(
            f"Adjusting img_size from {img_size} to {inferred_img_size} based on checkpoint pos_embed."
        )
        img_size = inferred_img_size

    df = load_split(splits_dir, fold)
    dataset = ForgedOnlyDataset(df, img_size=img_size)
    if samples is not None:
        if samples <= 0:
            raise ValueError("--samples must be a positive integer.")
        total = len(dataset)
        pick = min(samples, total)
        indices = random.sample(range(total), k=pick)
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    model = DinoSegModel(model_name=model_name, pretrained=False, img_size=img_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    fold_dir = outdir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    def align_mask(mask_arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        # Ensure mask is (H,W) and matches target height/width; if transposed, fix it.
        h_t, w_t = target_hw
        if mask_arr.shape[:2] != (h_t, w_t) and mask_arr.shape[:2] == (w_t, h_t):
            mask_arr = np.transpose(mask_arr)
        if mask_arr.shape[:2] != (h_t, w_t):
            mask_arr = np.array(
                Image.fromarray(mask_arr.astype(np.uint8)).resize((w_t, h_t), resample=Image.NEAREST)
            )
        return mask_arr

    with torch.no_grad():
        for images, case_ids, orig_sizes, mask_paths, image_paths in tqdm(loader, desc=f"Fold {fold}"):
            images = images.to(device, non_blocking=True)
            logits, _ = model(images)
            probs = torch.sigmoid(logits)
            masks = (probs >= threshold).float().cpu().numpy()
            def parse_size(orig_size) -> tuple[int, int]:
                if isinstance(orig_size, (list, tuple)) and len(orig_size) >= 2:
                    return int(orig_size[0]), int(orig_size[1])
                if torch.is_tensor(orig_size):
                    flat = orig_size.flatten()
                    if flat.numel() >= 2:
                        return int(flat[0].item()), int(flat[1].item())
                    if flat.numel() == 1:
                        val = int(flat[0].item())
                        return val, val
                if isinstance(orig_size, np.ndarray):
                    flat = orig_size.flatten()
                    if flat.size >= 2:
                        return int(flat[0]), int(flat[1])
                    if flat.size == 1:
                        val = int(flat[0])
                        return val, val
                raise ValueError(f"Unexpected orig_size format: {type(orig_size)}")

            for i, case_id in enumerate(case_ids):
                w, h = parse_size(orig_sizes[i])

                mask = masks[i, 0]
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_resized = np.array(mask_img.resize((w, h), resample=Image.NEAREST))
                mask_resized = align_mask(mask_resized, (h, w))
                # Save ground truth mask if available
                gt_path = mask_paths[i]
                if isinstance(gt_path, str) and gt_path and Path(gt_path).exists():
                    gt = np.load(gt_path, allow_pickle=True)
                    if gt.ndim == 3:
                        gt = gt.max(axis=0)
                    gt_mask = (gt > 0).astype(np.uint8)
                    gt_mask = align_mask(gt_mask, (h, w))
                else:
                    gt_mask = np.zeros((h, w), dtype=np.uint8)

                # Build collage with titles.
                with open(image_paths[i], "rb") as f:
                    orig_img = Image.open(f).convert("RGB")
                orig_img = orig_img.resize((w, h))
                orig_np = np.array(orig_img)

                # mask_resized is already 0/255 uint8; re-binarize to avoid overflow artifacts.
                pred_mask_bin = (mask_resized > 0).astype(np.uint8) * 255
                pred_mask_rgb = np.stack([pred_mask_bin] * 3, axis=-1).astype(np.uint8)
                gt_mask_rgb = np.stack([gt_mask * 255] * 3, axis=-1).astype(np.uint8)

                def apply_overlay(img_np: np.ndarray, mask_bin: np.ndarray, color=(255, 0, 0), alpha=0.5):
                    overlay = img_np.copy()
                    mask_bool = mask_bin.astype(bool)
                    overlay[mask_bool] = (
                        alpha * np.array(color, dtype=np.float32)
                        + (1 - alpha) * overlay[mask_bool].astype(np.float32)
                    ).astype(np.uint8)
                    return overlay

                pred_overlay = apply_overlay(orig_np, mask_resized > 0)
                gt_overlay = apply_overlay(orig_np, gt_mask > 0)

                panels = [
                    ("Input", orig_np),
                    ("Pred mask", pred_mask_rgb),
                    ("GT mask", gt_mask_rgb),
                    ("Pred overlay", pred_overlay),
                    ("GT overlay", gt_overlay),
                ]

                title_h = 24
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None

                def add_title(img_arr: np.ndarray, title: str) -> Image.Image:
                    panel = Image.new("RGB", (img_arr.shape[1], img_arr.shape[0] + title_h), color=(0, 0, 0))
                    panel.paste(Image.fromarray(img_arr), (0, title_h))
                    draw = ImageDraw.Draw(panel)
                    if font:
                        draw.text((5, 4), title, fill=(255, 255, 255), font=font)
                    else:
                        draw.text((5, 4), title, fill=(255, 255, 255))
                    return panel

                margin = 8
                titled = [add_title(arr, t) for t, arr in panels]
                panel_widths = [img.size[0] for img in titled]
                panel_heights = [img.size[1] for img in titled]
                collage_w = sum(panel_widths) + margin * (len(titled) - 1)
                collage_h = max(panel_heights)
                collage = Image.new("RGB", (collage_w, collage_h), color=(0, 0, 0))
                x = 0
                for panel in titled:
                    collage.paste(panel, (x, 0))
                    x += panel.size[0] + margin
                collage.save(fold_dir / f"{case_id}_collage.png")


def main():
    parser = argparse.ArgumentParser(description="Run inference per fold on forged images.")
    parser.add_argument("--weights-dir", default="analysis/models", help="Directory with saved fold weights.")
    parser.add_argument("--splits-dir", default="analysis/splits", help="Directory with train/val CSVs.")
    parser.add_argument("--outdir", default="analysis/preds", help="Where to save predicted masks.")
    parser.add_argument("--model-name", default="vit_base_patch16_224.dino", help="timm model name.")
    parser.add_argument("--img-size", type=int, default=448, help="Image resize size.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for masks.")
    parser.add_argument("--folds", nargs="*", type=int, default=None, help="Folds to run. Default: infer from weights.")
    parser.add_argument("--samples", type=int, default=None, help="Randomly sample N images per fold.")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    outdir = Path(args.outdir)
    splits_dir = Path(args.splits_dir)

    # Recreate output directory
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.folds is None:
        folds = []
        for p in weights_dir.glob("*.pt"):
            name = p.stem
            if "fold" in name:
                try:
                    fold_num = int(name.split("fold")[-1])
                    folds.append(fold_num)
                except ValueError:
                    continue
        folds = sorted(set(folds))
    else:
        folds = args.folds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in folds:
        weight_path = weights_dir / f"{Path(args.model_name).name}_fold{fold}.pt"
        infer_fold(
            fold=fold,
            weights_path=weight_path,
            splits_dir=splits_dir,
            outdir=outdir,
            model_name=args.model_name,
            img_size=args.img_size,
            threshold=args.threshold,
            device=device,
            samples=args.samples,
        )


if __name__ == "__main__":
    main()
