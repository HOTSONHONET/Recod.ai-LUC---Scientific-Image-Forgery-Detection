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
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

from dinov2_seg import DinoSegModel
from utils import evaluate_single_image, rle_encode, score


def _ensure_odd_positive(value: int) -> int:
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"Kernel size must be a positive odd int, got {value}.")
    return value


def enhanced_adaptive_mask(
    prob: np.ndarray,
    alpha_grad: float = 0.35,
    thr_scale: float = 0.3,
    close_ksize: int = 5,
    open_ksize: int = 3,
    blur_ksize: int = 3,
) -> tuple[np.ndarray, float]:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for adaptive postprocessing.")
    close_ksize = _ensure_odd_positive(close_ksize)
    open_ksize = _ensure_odd_positive(open_ksize)
    blur_ksize = _ensure_odd_positive(blur_ksize)
    prob = prob.astype(np.float32)
    gx = cv2.Sobel(prob, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_norm = grad_mag / (grad_mag.max() + 1e-6)
    enhanced = (1 - alpha_grad) * prob + alpha_grad * grad_norm
    enhanced = cv2.GaussianBlur(enhanced, (blur_ksize, blur_ksize), 0)
    thr = float(np.mean(enhanced) + thr_scale * np.std(enhanced))
    mask = (enhanced > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_ksize, close_ksize), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_ksize, open_ksize), np.uint8))
    return mask, thr


def finalize_mask(prob: np.ndarray, orig_size: tuple[int, int], alpha_grad: float = 0.35) -> tuple[np.ndarray, float]:
    mask, thr = enhanced_adaptive_mask(prob, alpha_grad=alpha_grad)
    mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)
    return mask, thr


def _prepare_input_tensor(
    pil_img: Image.Image,
    img_size: int,
    processor,
    to_tensor: transforms.ToTensor,
    normalize: transforms.Normalize,
) -> torch.Tensor:
    if processor is not None:
        pil_img = transforms.functional.resize(pil_img, (img_size, img_size))
        processed = processor(
            images=pil_img,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
            do_normalize=True,
            do_rescale=True,
        )
        return processed["pixel_values"].squeeze(0)
    return normalize(to_tensor(pil_img))


def predict_prob_map_tiled(
    model: torch.nn.Module,
    pil_img: Image.Image,
    img_size: int,
    processor,
    device: torch.device,
    tile_size: int,
    tile_overlap: int,
    to_tensor: transforms.ToTensor,
    normalize: transforms.Normalize,
    skip_white_tiles: bool = False,
    white_threshold: int = 250,
    white_fraction: float = 0.98,
) -> np.ndarray:
    if tile_size <= 0:
        raise ValueError("--tile-size must be a positive integer.")
    if tile_overlap >= tile_size:
        raise ValueError("--tile-overlap must be smaller than --tile-size.")
    if not 0.0 <= white_fraction <= 1.0:
        raise ValueError("--white-fraction must be between 0 and 1.")

    w, h = pil_img.size
    stride = tile_size - tile_overlap
    acc = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    xs = list(range(0, max(w - tile_size, 0) + 1, stride))
    ys = list(range(0, max(h - tile_size, 0) + 1, stride))
    if xs and xs[-1] != w - tile_size:
        xs.append(max(w - tile_size, 0))
    if ys and ys[-1] != h - tile_size:
        ys.append(max(h - tile_size, 0))
    if not xs:
        xs = [0]
    if not ys:
        ys = [0]

    model.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                crop = pil_img.crop((x, y, min(x + tile_size, w), min(y + tile_size, h)))
                tile_w, tile_h = crop.size
                if tile_w != tile_size or tile_h != tile_size:
                    padded = Image.new("RGB", (tile_size, tile_size), color=(255, 255, 255))
                    padded.paste(crop, (0, 0))
                    crop = padded
                if skip_white_tiles and _is_mostly_white(crop, threshold=white_threshold, fraction=white_fraction):
                    continue
                inp = _prepare_input_tensor(crop, img_size, processor, to_tensor, normalize).unsqueeze(0).to(device)
                out = model(inp)
                logits = out if torch.is_tensor(out) else out[0]
                prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
                if tile_w != tile_size or tile_h != tile_size:
                    prob = prob[:tile_h, :tile_w]
                prob_tile = np.array(
                    Image.fromarray(prob.astype(np.float32)).resize((tile_w, tile_h), resample=Image.BILINEAR)
                )
                acc[y : y + tile_h, x : x + tile_w] += prob_tile
                counts[y : y + tile_h, x : x + tile_w] += 1.0

    counts = np.maximum(counts, 1.0)
    return acc / counts


def _is_mostly_white(pil_img: Image.Image, threshold: int, fraction: float) -> bool:
    arr = np.asarray(pil_img)
    if arr.size == 0:
        return True
    if arr.ndim == 2:
        white_mask = arr >= threshold
    else:
        white_mask = (arr[..., 0] >= threshold) & (arr[..., 1] >= threshold) & (arr[..., 2] >= threshold)
    return float(np.mean(white_mask)) >= fraction


def _format_pred_score(score: float) -> str:
    return f"{score:.3f}".replace(".", "pt")


def _compute_image_score(
    gt_instances: list[np.ndarray],
    pred_bin: np.ndarray,
    shape: tuple[int, int],
) -> tuple[str, float]:
    gt_label = "authentic" if len(gt_instances) == 0 else "forged"
    pred_label = "authentic" if int(pred_bin.sum()) == 0 else "forged"
    if gt_label == "authentic" or pred_label == "authentic":
        score = 1.0 if gt_label == pred_label else 0.0
    else:
        label_rles = rle_encode(gt_instances)
        pred_rles = rle_encode([pred_bin.astype(np.uint8)])
        score = evaluate_single_image(label_rles, pred_rles, json.dumps([shape[0], shape[1]]))
    return gt_label, score


class InferenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int = 448,
        processor=None,
    ):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.processor = processor
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

        case_id = row["case_id"]
        mask_path = row.get("mask_path", "")
        label = int(row.get("label", 0))
        image_path = row["image_path"]
        with open(image_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        orig_size = img.size  # (W,H)
        if self.processor is not None:
            img = transforms.functional.resize(img, (self.img_size, self.img_size))
            processed = self.processor(
                images=img,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
                do_normalize=True,
                do_rescale=True,
            )
            img_t = processed["pixel_values"].squeeze(0)
        else:
            img_t = self.img_transform(img)
        return img_t, case_id, orig_size, mask_path, image_path, label


def load_split(splits_dir: Path, fold: int, split_mode: str = "all") -> pd.DataFrame:
    train_path = splits_dir / f"train_fold{fold}.csv"
    val_path = splits_dir / f"val_fold{fold}.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing splits for fold {fold} in {splits_dir}")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    if split_mode == "train":
        return train_df
    if split_mode == "val":
        return val_df
    if split_mode == "all":
        return pd.concat([train_df, val_df], ignore_index=True)
    raise ValueError(f"Unknown split_mode: {split_mode}")


def build_supplemental_dataframe(repo_root: Path) -> pd.DataFrame:
    rows = []
    supp_dir = repo_root / "supplemental_images"
    supp_mask_dir = repo_root / "supplemental_masks"
    for img_path in supp_dir.glob("*.png"):
        case_id = img_path.stem
        mask_path = supp_mask_dir / f"{case_id}.npy"
        rows.append(
            {
                "case_id": case_id,
                "image_path": str(img_path.resolve()),
                "mask_path": str(mask_path.resolve()) if mask_path.exists() else "",
                "label": 1 if mask_path.exists() else 0,
                "source": "supplemental",
                "variant": "supplemental_forged" if mask_path.exists() else "supplemental_authentic",
            }
        )
    return pd.DataFrame(rows)


def infer_fold(
    fold: int,
    weights_path: Path,
    splits_dir: Path,
    outdir: Path,
    repo_root: Path,
    arch: str,
    model_name: str,
    dinov2_id: str,
    img_size: int,
    threshold: float,
    device: torch.device,
    samples: int | None,
    split_mode: str,
    compute_score: bool,
    processor=None,
    alpha_grad: float = 0.35,
    area_thr: int = 190,
    mean_thr: float = 0.21,
    supplemental_only: bool = False,
    use_tiles: bool = False,
    tile_size: int | None = None,
    tile_overlap: int = 0,
    skip_white_tiles: bool = False,
    white_threshold: int = 250,
    white_fraction: float = 0.98,
    collect_authentic_stats: bool = False,
    calibration_out: Path | None = None,
) -> dict | None:
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

    if arch == "dino_seg":
        inferred_img_size = infer_img_size_from_state(model_state, img_size)
        if inferred_img_size != img_size:
            print(
                f"Adjusting img_size from {img_size} to {inferred_img_size} based on checkpoint pos_embed."
            )
            img_size = inferred_img_size
    else:
        if img_size % 14 != 0:
            raise ValueError(f"{arch} requires img_size divisible by 14 (DINOv2 patch size).")

    df = load_split(splits_dir, fold, split_mode=split_mode)
    if supplemental_only:
        if "source" in df.columns:
            df = df[df["source"] == "supplemental"].reset_index(drop=True)
        if df.empty:
            df = build_supplemental_dataframe(repo_root)
    dataset = InferenceDataset(df, img_size=img_size, processor=processor)
    if samples is not None:
        if samples <= 0:
            raise ValueError("--samples must be a positive integer.")
        total = len(dataset)
        pick = min(samples, total)
        indices = random.sample(range(total), k=pick)
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    if arch == "dino_seg":
        model = DinoSegModel(model_name=model_name, pretrained=False, img_size=img_size).to(device)
    elif arch == "dinov2_uperhead":
        from dinov2_uperhead import DinoV2_UPerNet

        model = DinoV2_UPerNet(dinov2_id=dinov2_id, num_classes=1).to(device)
    elif arch == "dinov2_unet":
        from dinov2_unet import DinoV2UNet

        model = DinoV2UNet(dinov2_id=dinov2_id, out_classes=1).to(device)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    model.load_state_dict(model_state)
    model.eval()

    tile_size = tile_size or img_size

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

    solution_rows: list[dict] = []
    submission_rows: list[dict] = []
    records: list[dict] = []
    auth_areas: list[int] = []
    auth_means: list[float] = []

    with torch.no_grad():
        for images, case_ids, orig_sizes, mask_paths, image_paths, labels in tqdm(
            loader, desc=f"Fold {fold}"
        ):
            images = images.to(device, non_blocking=True)
            if not use_tiles:
                out = model(images)
                logits = out if torch.is_tensor(out) else out[0]
                probs = torch.sigmoid(logits)
            else:
                probs = None
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

            def normalize_case_id(raw) -> int | str:
                if torch.is_tensor(raw):
                    raw = raw.item()
                if isinstance(raw, (np.integer, int)):
                    return int(raw)
                return str(raw)

            for i, case_id in enumerate(case_ids):
                w, h = parse_size(orig_sizes[i])
                case_id_value = normalize_case_id(case_id)

                if use_tiles:
                    with open(image_paths[i], "rb") as f:
                        pil_img = Image.open(f).convert("RGB")
                    prob_resized = predict_prob_map_tiled(
                        model=model,
                        pil_img=pil_img,
                        img_size=img_size,
                        processor=processor,
                        device=device,
                        tile_size=tile_size,
                        tile_overlap=tile_overlap,
                        to_tensor=dataset.to_tensor,
                        normalize=dataset.normalize,
                        skip_white_tiles=skip_white_tiles,
                        white_threshold=white_threshold,
                        white_fraction=white_fraction,
                    )
                else:
                    prob_t = probs[i : i + 1].detach().cpu()
                    prob_resized = F.interpolate(
                        prob_t,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0].numpy()
                # Save ground truth mask if available
                gt_path = mask_paths[i]
                if isinstance(gt_path, str) and gt_path and Path(gt_path).exists():
                    gt = np.load(gt_path, allow_pickle=True)
                    if gt.ndim == 3:
                        if gt.shape[0] == h and gt.shape[1] == w:
                            gt_instances = [gt[:, :, c] for c in range(gt.shape[2])]
                        else:
                            gt_instances = [gt[j] for j in range(gt.shape[0])]
                        gt_instances = [align_mask((g > 0).astype(np.uint8), (h, w)) for g in gt_instances]
                        gt_mask = np.maximum.reduce(gt_instances)
                    else:
                        gt_instances = [align_mask((gt > 0).astype(np.uint8), (h, w))]
                        gt_mask = gt_instances[0]
                else:
                    gt_instances = []
                    gt_mask = np.zeros((h, w), dtype=np.uint8)

                if compute_score:
                    row_id = case_id_value
                    if len(gt_instances) == 0:
                        solution_rows.append(
                            {"row_id": row_id, "annotation": "authentic", "shape": "authentic"}
                        )
                    else:
                        solution_rows.append(
                            {
                                "row_id": row_id,
                                "annotation": rle_encode(gt_instances),
                                "shape": json.dumps([h, w]),
                            }
                        )
                    records.append(
                        {
                            "row_id": row_id,
                            "prob": prob_resized.astype(np.float32),
                            "shape": (h, w),
                            "gt_instances": gt_instances,
                            "image_path": image_paths[i],
                            "label": int(labels[i]),
                        }
                    )
                    continue

                pred_bin, _ = enhanced_adaptive_mask(prob_resized, alpha_grad=alpha_grad)
                area = int(pred_bin.sum())
                mean_inside = float(prob_resized[pred_bin == 1].mean()) if area > 0 else 0.0
                if collect_authentic_stats and int(labels[i]) == 0 and area > 0:
                    auth_areas.append(area)
                    auth_means.append(mean_inside)
                if area < area_thr or mean_inside < mean_thr:
                    pred_bin = np.zeros_like(pred_bin)

                gt_label, image_score = _compute_image_score(gt_instances, pred_bin, (h, w))
                score_str = _format_pred_score(image_score)

                mask_resized = (pred_bin * 255).astype(np.uint8)
                mask_resized = align_mask(mask_resized, (h, w))

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
                collage.save(fold_dir / f"{gt_label}_{case_id_value}_{score_str}.png")

    if compute_score and solution_rows:
        submission_rows = []
        for rec in records:
            mask, _ = enhanced_adaptive_mask(rec["prob"], alpha_grad=alpha_grad)
            area = int(mask.sum())
            mean_inside = float(rec["prob"][mask == 1].mean()) if area > 0 else 0.0
            if collect_authentic_stats and rec["label"] == 0 and area > 0:
                auth_areas.append(area)
                auth_means.append(mean_inside)
            if area < area_thr or mean_inside < mean_thr:
                pred_bin = np.zeros_like(mask)
            else:
                pred_bin = mask
            if pred_bin.sum() == 0:
                submission_rows.append({"row_id": rec["row_id"], "annotation": "authentic"})
            else:
                submission_rows.append({"row_id": rec["row_id"], "annotation": rle_encode([pred_bin])})

            h, w = rec["shape"]
            mask_resized = (pred_bin * 255).astype(np.uint8)
            mask_resized = align_mask(mask_resized, (h, w))

            with open(rec["image_path"], "rb") as f:
                orig_img = Image.open(f).convert("RGB")
            orig_img = orig_img.resize((w, h))
            orig_np = np.array(orig_img)

            gt_instances = rec["gt_instances"]
            if len(gt_instances) == 0:
                gt_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                gt_mask = np.maximum.reduce(gt_instances)

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
            score_mask = align_mask((pred_bin > 0).astype(np.uint8), (h, w))
            gt_label, image_score = _compute_image_score(gt_instances, score_mask, (h, w))
            score_str = _format_pred_score(image_score)
            collage.save(fold_dir / f"{gt_label}_{rec['row_id']}_{score_str}.png")

        solution_df = pd.DataFrame(solution_rows)
        submission_df = pd.DataFrame(submission_rows)
        metric = score(solution_df, submission_df, row_id_column_name="row_id")
        print(f"[Fold {fold}] competition score={metric:.6f}")

        score_path = outdir / f"fold_{fold}_score.txt"
        with open(score_path, "w", encoding="utf-8") as f:
            f.write(f"{metric:.6f}\n")

        merged = solution_df.merge(submission_df, on="row_id", suffixes=("_label", "_pred"))
        per_image_rows: list[dict] = []
        for _, row in merged.iterrows():
            label = row["annotation_label"]
            pred = row["annotation_pred"]
            shape = row["shape"]
            if label == "authentic" or pred == "authentic":
                image_score = 1.0 if label == pred else 0.0
            else:
                image_score = evaluate_single_image(label, pred, shape)
            per_image_rows.append({"row_id": row["row_id"], "oF1": image_score})
        per_image_df = pd.DataFrame(per_image_rows)
        per_image_df.to_csv(outdir / f"fold_{fold}_per_image_of1.csv", index=False)

        submission_out = submission_df.rename(columns={"row_id": "case_id"})
        submission_out.to_csv(outdir / f"fold_{fold}_submission.csv", index=False)

    if collect_authentic_stats:
        mean_area = float(np.mean(auth_areas)) if auth_areas else 0.0
        mean_inside = float(np.mean(auth_means)) if auth_means else 0.0
        stats = {
            "fold": fold,
            "mean_area": mean_area,
            "mean_mean_inside": mean_inside,
            "count": len(auth_areas),
        }
        print(
            f"[Fold {fold}] authentic calibration: mean_area={mean_area:.2f}, "
            f"mean_mean_inside={mean_inside:.4f} (n={len(auth_areas)})"
        )
        if calibration_out is not None:
            with open(calibration_out, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
        return stats


def main():
    parser = argparse.ArgumentParser(description="Run inference per fold on forged images.")
    parser.add_argument("--repo-root", default=".", help="Repository root for supplemental lookup.")
    parser.add_argument("--weights-dir", default="analysis/models", help="Directory with saved fold weights.")
    parser.add_argument("--splits-dir", default="analysis/splits", help="Directory with train/val CSVs.")
    parser.add_argument("--outdir", default="analysis/preds", help="Where to save predicted masks.")
    parser.add_argument(
        "--arch",
        choices=["dino_seg", "dinov2_uperhead", "dinov2_unet"],
        default="dino_seg",
        help="Model architecture to use for inference.",
    )
    parser.add_argument("--model-name", default="vit_base_patch16_224.dino", help="timm model name.")
    parser.add_argument(
        "--dinov2-id",
        default="facebook/dinov2-base",
        help="HuggingFace model id for DINOv2 when using dinov2_uperhead or dinov2_unet.",
    )
    parser.add_argument(
        "--use-hf-processor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use HuggingFace AutoImageProcessor for dinov2_* architectures.",
    )
    parser.add_argument("--img-size", type=int, default=448, help="Image resize size.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for masks.")
    parser.add_argument("--alpha-grad", type=float, default=0.35, help="Gradient mix for adaptive threshold.")
    parser.add_argument("--area-thr", type=int, default=190, help="Area threshold for adaptive_gate.")
    parser.add_argument("--mean-thr", type=float, default=0.21, help="Mean-inside threshold for adaptive_gate.")
    parser.add_argument(
        "--supplemental-only",
        action="store_true",
        help="Run inference only on supplemental items from the split CSVs.",
    )
    parser.add_argument("--use-tiles", action="store_true", help="Enable tiled inference over original images.")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tile size in pixels (original image space). Defaults to img_size.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=0,
        help="Overlap in pixels between tiles.",
    )
    parser.add_argument(
        "--skip-white-tiles",
        action="store_true",
        help="Skip tiles that are mostly white (useful for supplemental images).",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=250,
        help="Pixel value threshold to consider a pixel white (0-255).",
    )
    parser.add_argument(
        "--white-fraction",
        type=float,
        default=0.98,
        help="Fraction of white pixels to treat a tile as white.",
    )
    parser.add_argument(
        "--calibrate-authentic",
        action="store_true",
        help="Compute mean area/mean-inside from authentic predictions and save for reuse.",
    )
    parser.add_argument(
        "--calibration-file",
        default=None,
        help="Path to a calibration JSON to override area/mean thresholds.",
    )
    parser.add_argument(
        "--run-val-after-calibration",
        action="store_true",
        help="After supplemental calibration, run validation inference with the calibrated thresholds.",
    )
    parser.add_argument("--folds", nargs="*", type=int, default=None, help="Folds to run. Default: infer from weights.")
    parser.add_argument("--samples", type=int, default=None, help="Randomly sample N images per fold.")
    parser.add_argument(
        "--split-mode",
        choices=["train", "val", "all"],
        default="all",
        help="Which split to use for inference/score.",
    )
    parser.add_argument("--compute-score", action="store_true", help="Compute competition score on predictions.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    weights_dir = Path(args.weights_dir)
    outdir = Path(args.outdir)
    splits_dir = Path(args.splits_dir)

    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for adaptive threshold + morphology.")
    if args.use_tiles and args.tile_size is not None and args.tile_overlap >= args.tile_size:
        raise ValueError("--tile-overlap must be smaller than --tile-size.")
    if not 0.0 <= args.white_fraction <= 1.0:
        raise ValueError("--white-fraction must be between 0 and 1.")

    if args.calibration_file:
        with open(args.calibration_file, "r", encoding="utf-8") as f:
            calib = json.load(f)
        args.area_thr = int(round(float(calib.get("mean_area", args.area_thr))))
        args.mean_thr = float(calib.get("mean_mean_inside", args.mean_thr))

    if args.run_val_after_calibration and not args.calibrate_authentic:
        raise ValueError("--run-val-after-calibration requires --calibrate-authentic.")

    if args.supplemental_only and not args.run_val_after_calibration:
        suffix = "_supplements"
        outdir = Path(f"{outdir}{suffix}") if not str(outdir).endswith(suffix) else outdir

    # Recreate output directory
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    use_hf_processor = args.use_hf_processor
    if use_hf_processor is None:
        use_hf_processor = args.arch != "dino_seg"
    hf_processor = None
    if use_hf_processor:
        from transformers import AutoImageProcessor

        hf_processor = AutoImageProcessor.from_pretrained(args.dinov2_id)

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
        if args.run_val_after_calibration:
            supp_outdir = outdir / "supplemental"
            supp_outdir.mkdir(parents=True, exist_ok=True)
            calib_path = supp_outdir / f"fold_{fold}_authentic_calibration.json"
            stats = infer_fold(
                fold=fold,
                weights_path=weight_path,
                splits_dir=splits_dir,
                outdir=supp_outdir,
                repo_root=repo_root,
                arch=args.arch,
                model_name=args.model_name,
                dinov2_id=args.dinov2_id,
                img_size=args.img_size,
                threshold=args.threshold,
                device=device,
                samples=args.samples,
                split_mode=args.split_mode,
                compute_score=args.compute_score,
                processor=hf_processor,
                alpha_grad=args.alpha_grad,
                area_thr=args.area_thr,
                mean_thr=args.mean_thr,
                supplemental_only=True,
                use_tiles=args.use_tiles,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                skip_white_tiles=args.skip_white_tiles,
                white_threshold=args.white_threshold,
                white_fraction=args.white_fraction,
                collect_authentic_stats=True,
                calibration_out=calib_path,
            )
            if not stats:
                continue
            val_outdir = outdir / "val"
            val_outdir.mkdir(parents=True, exist_ok=True)
            infer_fold(
                fold=fold,
                weights_path=weight_path,
                splits_dir=splits_dir,
                outdir=val_outdir,
                repo_root=repo_root,
                arch=args.arch,
                model_name=args.model_name,
                dinov2_id=args.dinov2_id,
                img_size=args.img_size,
                threshold=args.threshold,
                device=device,
                samples=args.samples,
                split_mode="val",
                compute_score=args.compute_score,
                processor=hf_processor,
                alpha_grad=args.alpha_grad,
                area_thr=int(round(stats["mean_area"])),
                mean_thr=float(stats["mean_mean_inside"]),
                supplemental_only=False,
                use_tiles=args.use_tiles,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                skip_white_tiles=args.skip_white_tiles,
                white_threshold=args.white_threshold,
                white_fraction=args.white_fraction,
            )
            continue

        infer_fold(
            fold=fold,
            weights_path=weight_path,
            splits_dir=splits_dir,
            outdir=outdir,
            repo_root=repo_root,
            arch=args.arch,
            model_name=args.model_name,
            dinov2_id=args.dinov2_id,
            img_size=args.img_size,
            threshold=args.threshold,
            device=device,
            samples=args.samples,
            split_mode=args.split_mode,
            compute_score=args.compute_score,
            processor=hf_processor,
            alpha_grad=args.alpha_grad,
            area_thr=args.area_thr,
            mean_thr=args.mean_thr,
            supplemental_only=args.supplemental_only,
            use_tiles=args.use_tiles,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            skip_white_tiles=args.skip_white_tiles,
            white_threshold=args.white_threshold,
            white_fraction=args.white_fraction,
            collect_authentic_stats=args.calibrate_authentic,
            calibration_out=outdir / f"fold_{fold}_authentic_calibration.json" if args.calibrate_authentic else None,
        )


if __name__ == "__main__":
    main()
