"""
Training script for ViT-DINO segmentation on the scientific forgery dataset.

Features:
- 5-fold cross-validation (configurable)
- MLflow logging (params, metrics, artifacts, model weights, split CSVs)
- Uses timm ViT DINO backbone with a lightweight upsampling head
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

from config import Config, set_seed
from data_split import build_dataframe, make_folds, save_folds
from model import DinoSegModel


class ForgeryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int = 448, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _apply_clahe(self, image):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for CLAHE preprocessing.")
        img_np = np.array(image)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        from PIL import Image

        return Image.fromarray(enhanced)

    def __len__(self) -> int:
        return len(self.df)

    def _load_mask(self, mask_path: str, target_size: Tuple[int, int]) -> np.ndarray:
        if not mask_path or not Path(mask_path).exists():
            return np.zeros(target_size, dtype=np.float32)

        arr = np.load(mask_path, allow_pickle=True)
        if arr.ndim == 3:
            arr = arr.max(axis=0)
        mask = arr.astype(np.float32)
        return mask

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        mask_path = row.get("mask_path", "")

        with open(image_path, "rb") as f:
            from PIL import Image

            image = Image.open(f).convert("RGB")

        image = self._apply_clahe(image)

        w, h = image.size
        mask_np = self._load_mask(mask_path, target_size=(h, w))
        mask_pil = transforms.functional.to_pil_image(mask_np)

        image = transforms.functional.resize(image, (self.img_size, self.img_size))
        mask_pil = transforms.functional.resize(
            mask_pil, (self.img_size, self.img_size), transforms.InterpolationMode.NEAREST
        )
        if self.augment:
            if torch.rand(1).item() < 0.5:
                image = transforms.functional.hflip(image)
                mask_pil = transforms.functional.hflip(mask_pil)
            if torch.rand(1).item() < 0.5:
                image = transforms.functional.vflip(image)
                mask_pil = transforms.functional.vflip(mask_pil)
        image = self.normalize(self.to_tensor(image))

        mask_tensor = transforms.functional.pil_to_tensor(mask_pil).float()
        mask_tensor = (mask_tensor > 0.5).float()

        label = float(row.get("label", 0))
        return image, mask_tensor, torch.tensor(label, dtype=torch.float32)


def mask_bce_loss(
    mask_logits: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    auth_mask_weight: float,
    pos_weight: float | None = None,
) -> torch.Tensor:
    pw = None
    if pos_weight is not None:
        pw = torch.tensor(pos_weight, device=mask_logits.device, dtype=mask_logits.dtype)
    loss_map = F.binary_cross_entropy_with_logits(mask_logits, targets, pos_weight=pw, reduction="none")
    per_sample = loss_map.mean(dim=(1, 2, 3))
    weights = torch.where(labels > 0.5, 1.0, auth_mask_weight).to(per_sample.dtype)
    return (per_sample * weights).mean()


def dice_score(mask_logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    probs = torch.sigmoid(mask_logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    pos_weight: float | None = None,
    auth_mask_weight: float = 0.1,
    mask_loss_weight: float = 1.0,
    cls_loss_weight: float = 1.0,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    count = 0
    use_amp = scaler is not None and scaler.is_enabled()
    device_type = device.type
    for images, masks, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device_type, enabled=use_amp):
            mask_logits, class_logits = model(images)
            mask_loss = mask_bce_loss(
                mask_logits, masks, labels, auth_mask_weight=auth_mask_weight, pos_weight=pos_weight
            )
            cls_loss = F.binary_cross_entropy_with_logits(class_logits.squeeze(1), labels)
            loss = mask_loss * mask_loss_weight + cls_loss * cls_loss_weight
            dice = dice_score(mask_logits, masks)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_dice += dice.item() * images.size(0)
        count += images.size(0)

    return total_loss / count, total_dice / count


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: float | None = None,
    auth_mask_weight: float = 0.1,
    mask_loss_weight: float = 1.0,
    cls_loss_weight: float = 1.0,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    count = 0
    with torch.no_grad():
        for images, masks, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask_logits, class_logits = model(images)
            mask_loss = mask_bce_loss(
                mask_logits, masks, labels, auth_mask_weight=auth_mask_weight, pos_weight=pos_weight
            )
            cls_loss = F.binary_cross_entropy_with_logits(class_logits.squeeze(1), labels)
            loss = mask_loss * mask_loss_weight + cls_loss * cls_loss_weight
            dice = dice_score(mask_logits, masks)
            total_loss += loss.item() * images.size(0)
            total_dice += dice.item() * images.size(0)
            count += images.size(0)
    return total_loss / count, total_dice / count


def prepare_splits(repo_root: Path, splits_dir: Path, n_splits: int = 5, seed: int = 42):
    if splits_dir.exists() and list(splits_dir.glob("train_fold*.csv")):
        return
    df = build_dataframe(repo_root)
    folds = make_folds(df, n_splits=n_splits, seed=seed)
    save_folds(folds, splits_dir)


def load_fold(splits_dir: Path, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = splits_dir / f"train_fold{fold}.csv"
    val_path = splits_dir / f"val_fold{fold}.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing split CSVs for fold {fold} in {splits_dir}")
    return pd.read_csv(train_path), pd.read_csv(val_path)


def main():
    parser = argparse.ArgumentParser(description="Train DINO-based segmentation with 5-fold CV.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--splits-dir", default=Config.TRAIN_CSV, help="Directory to read/write split CSVs.")
    parser.add_argument("--n-splits", type=int, default=Config.N_SPLITS, help="Number of folds.")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS, help="Epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH, help="Batch size.")
    parser.add_argument("--lr", type=float, default=Config.LR, help="Learning rate.")
    parser.add_argument("--img-size", type=int, default=Config.IMG_SIZE, help="Square image size for training.")
    parser.add_argument("--num-workers", type=int, default=Config.NUM_WORKERS, help="Dataloader workers.")
    parser.add_argument("--model-name", default=Config.TIMM_MODEL, help="timm model name.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights.")
    parser.add_argument("--experiment-name", default=Config.EXPERIMENT_NAME, help="MLflow experiment name.")
    parser.add_argument("--mlflow-uri", default=None, help="Optional MLflow tracking URI (e.g., http://localhost:5000).")
    parser.add_argument("--output-dir", default=Config.OUTPUT_DIR, help="Where to save model weights.")
    parser.add_argument("--seed", type=int, default=Config.SEED, help="Random seed.")
    parser.add_argument("--pos-weight", type=float, default=None, help="Positive class weight for BCE (e.g., 5.0).")
    parser.add_argument("--auth-mask-weight", type=float, default=0.1, help="Mask loss weight for authentic images.")
    parser.add_argument("--mask-loss-weight", type=float, default=1.0, help="Global weight for mask BCE loss.")
    parser.add_argument("--cls-loss-weight", type=float, default=1.0, help="Global weight for classification BCE loss.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    splits_dir = (repo_root / args.splits_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (repo_root / args.output_dir / timestamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cli_args = {"argv": sys.argv, "parsed": vars(args)}
    with open(output_dir / "cli_args.json", "w", encoding="utf-8") as f:
        json.dump(cli_args, f, indent=2, sort_keys=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    prepare_splits(repo_root, splits_dir, n_splits=args.n_splits)

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    experiment_name = f"{args.experiment_name}_{timestamp}"
    mlflow.set_experiment(experiment_name)

    for fold in range(args.n_splits):
        train_df, val_df = load_fold(splits_dir, fold)
        train_ds = ForgeryDataset(train_df, img_size=args.img_size, augment=True)
        val_ds = ForgeryDataset(val_df, img_size=args.img_size, augment=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model = DinoSegModel(model_name=args.model_name, pretrained=not args.no_pretrained, img_size=args.img_size)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        use_amp = device.type == "cuda"
        scaler = GradScaler(enabled=use_amp) if use_amp else None

        with mlflow.start_run(run_name=f"fold_{fold}", nested=False):
            run_cfg = Config.to_dict()
            run_cfg.update(
                {
                    "EPOCHS": args.epochs,
                    "BATCH": args.batch_size,
                    "LR": args.lr,
                    "IMG_SIZE": args.img_size,
                    "NUM_WORKERS": args.num_workers,
                    "N_SPLITS": args.n_splits,
                    "TIMM_MODEL": args.model_name,
                    "PRETRAINED": not args.no_pretrained,
                    "SEED": args.seed,
                    "POS_WEIGHT": args.pos_weight,
                    "AUTH_MASK_WEIGHT": args.auth_mask_weight,
                    "MASK_LOSS_WEIGHT": args.mask_loss_weight,
                    "CLS_LOSS_WEIGHT": args.cls_loss_weight,
                }
            )
            mlflow.log_dict(run_cfg, "config_used.json")
            mlflow.log_artifact(str(output_dir / "cli_args.json"))
            mlflow.log_params(
                {
                    "fold": fold,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "img_size": args.img_size,
                    "model_name": args.model_name,
                    "pretrained": not args.no_pretrained,
                    "pos_weight": args.pos_weight,
                    "auth_mask_weight": args.auth_mask_weight,
                    "mask_loss_weight": args.mask_loss_weight,
                    "cls_loss_weight": args.cls_loss_weight,
                    "machine_specs": json.dumps({"ram_gb": 64, "cpu_cores": 16, "gpu": "RTX 5070 Ti"}),
                }
            )
            mlflow.log_artifacts(str(splits_dir), artifact_path="splits")

            best_dice = 0.0
            best_path = output_dir / f"{Path(args.model_name).name}_fold{fold}.pt"
            history: list[dict] = []

            for epoch in range(1, args.epochs + 1):
                train_loss, train_dice = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scaler,
                    device,
                    pos_weight=args.pos_weight,
                    auth_mask_weight=args.auth_mask_weight,
                    mask_loss_weight=args.mask_loss_weight,
                    cls_loss_weight=args.cls_loss_weight,
                )
                val_loss, val_dice = validate(
                    model,
                    val_loader,
                    device,
                    pos_weight=args.pos_weight,
                    auth_mask_weight=args.auth_mask_weight,
                    mask_loss_weight=args.mask_loss_weight,
                    cls_loss_weight=args.cls_loss_weight,
                )

                metrics = {
                    f"train_loss_fold{fold}": train_loss,
                    f"train_dice_fold{fold}": train_dice,
                    f"val_loss_fold{fold}": val_loss,
                    f"val_dice_fold{fold}": val_dice,
                    "train_loss": train_loss,
                    "train_dice": train_dice,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                }
                mlflow.log_metrics(metrics, step=epoch)
                print(
                    f"[Fold {fold}][Epoch {epoch}/{args.epochs}] "
                    f"train_loss={train_loss:.4f}, train_dice={train_dice:.4f}, "
                    f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f}"
                )
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_dice": train_dice,
                        "val_loss": val_loss,
                        "val_dice": val_dice,
                    }
                )

                if val_dice > best_dice:
                    best_dice = val_dice
                    torch.save({"model_state": model.state_dict(), "epoch": epoch}, best_path)
                    mlflow.log_artifact(str(best_path), artifact_path=f"fold_{fold}_weights")

            mlflow.log_metric(f"best_val_dice_fold{fold}", best_dice)

            # Log curves as artifacts for convenient viewing
            if history:
                hist_df = pd.DataFrame(history)
                csv_path = output_dir / f"history_fold{fold}.csv"
                hist_df.to_csv(csv_path, index=False)
                mlflow.log_artifact(str(csv_path), artifact_path=f"fold_{fold}_history")

                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
                ax[0].plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss")
                ax[0].set_xlabel("epoch")
                ax[0].set_ylabel("loss")
                ax[0].set_title(f"Loss (fold {fold})")
                ax[0].legend()

                ax[1].plot(hist_df["epoch"], hist_df["train_dice"], label="train_dice")
                ax[1].plot(hist_df["epoch"], hist_df["val_dice"], label="val_dice")
                ax[1].set_xlabel("epoch")
                ax[1].set_ylabel("dice")
                ax[1].set_title(f"Dice (fold {fold})")
                ax[1].legend()

                fig.tight_layout()
                mlflow.log_figure(fig, f"fold_{fold}_curves.png")
                plt.close(fig)

    print("Training complete. Models saved to", output_dir)


if __name__ == "__main__":
    main()
