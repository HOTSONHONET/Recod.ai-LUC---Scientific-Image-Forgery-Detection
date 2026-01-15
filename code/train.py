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
from data_split import build_dataframe, make_area_stratified_split, make_folds, save_folds
from dinov2_seg import DinoSegModel


class ForgeryDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int = 448,
        augment: bool = False,
        processor=None,
        include_synthetic=False,
        include_supplemental=False,
    ):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment
        self.processor = processor
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.include_synthetic = include_synthetic
        self.include_supplemental = include_supplemental

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

        if not (self.include_synthetic or self.include_supplemental):
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
        if self.processor is not None:
            processed = self.processor(
                images=image,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
                do_normalize=True,
                do_rescale=True,
            )
            image = processed["pixel_values"].squeeze(0)
        else:
            image = self.normalize(self.to_tensor(image))

        mask_tensor = transforms.functional.pil_to_tensor(mask_pil).float()
        mask_tensor = (mask_tensor > 0.5).float()
        if self.processor is not None:
            _, h, w = image.shape
            if mask_tensor.shape[-2:] != (h, w):
                mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)

        return image, mask_tensor


def bce_loss(mask_logits: torch.Tensor, targets: torch.Tensor, pos_weight: float | None = None) -> torch.Tensor:
    pw = None
    if pos_weight is not None:
        pw = torch.tensor(pos_weight, device=mask_logits.device, dtype=mask_logits.dtype)
    return F.binary_cross_entropy_with_logits(mask_logits, targets, pos_weight=pw)


def dice_loss(mask_logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    probs = torch.sigmoid(mask_logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


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
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    count = 0
    use_amp = scaler is not None and scaler.is_enabled()
    device_type = device.type
    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device_type, enabled=use_amp):
            mask_logits = model(images)
            loss = bce_loss(mask_logits, masks, pos_weight=pos_weight) + dice_loss(mask_logits, masks)
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
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    count = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Val", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            mask_logits = model(images)
            loss = bce_loss(mask_logits, masks, pos_weight=pos_weight) + dice_loss(mask_logits, masks)
            dice = dice_score(mask_logits, masks)
            total_loss += loss.item() * images.size(0)
            total_dice += dice.item() * images.size(0)
            count += images.size(0)
    return total_loss / count, total_dice / count


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def prepare_splits(
    repo_root: Path,
    splits_dir: Path,
    n_splits: int = 5,
    seed: int = 42,
    include_supplemental: bool = True,
    split_strategy: str = "kfold",
    area_bins: int = 5,
    val_ratio: float = 0.2,
    include_synthetic: bool = False,
):
    meta_path = splits_dir / "split_meta.json"
    if splits_dir.exists() and list(splits_dir.glob("train_fold*.csv")):
        if split_strategy == "area_bins":
            if not meta_path.exists():
                raise ValueError(
                    f"Existing splits in {splits_dir} are missing split_meta.json; "
                    "use a new --splits-dir for area_bins."
                )
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            expected = {
                "split_strategy": split_strategy,
                "area_bins": area_bins,
                "val_ratio": val_ratio,
                "seed": seed,
                "include_supplemental": include_supplemental,
                "include_synthetic": include_synthetic,
            }
            mismatches = [
                key for key, value in expected.items() if meta.get(key) != value
            ]
            if mismatches:
                raise ValueError(
                    f"Existing splits in {splits_dir} do not match requested area_bins config "
                    f"(mismatched: {', '.join(mismatches)}); use a new --splits-dir."
                )
        return
    df = build_dataframe(repo_root, include_supplemental=include_supplemental, include_synthetic = include_synthetic)
    if split_strategy == "kfold":
        folds = make_folds(df, n_splits=n_splits, seed=seed)
    elif split_strategy == "area_bins":
        train_df, val_df = make_area_stratified_split(df, bins=area_bins, val_ratio=val_ratio, seed=seed)
        folds = [(train_df, val_df)]
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")
    save_folds(folds, splits_dir)
    meta = {
        "split_strategy": split_strategy,
        "area_bins": area_bins,
        "val_ratio": val_ratio,
        "n_splits": n_splits,
        "seed": seed,
        "include_supplemental": include_supplemental,
        "include_synthetic": include_synthetic,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


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
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop if val_dice doesn't improve for this many epochs (0 disables).",
    )
    parser.add_argument("--batch-size", type=int, default=Config.BATCH, help="Batch size.")
    parser.add_argument("--lr", type=float, default=Config.LR, help="Learning rate.")
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine", "onecycle"],
        default="none",
        help="Learning rate schedule type.",
    )
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Warmup epochs for cosine schedule.")
    parser.add_argument("--img-size", type=int, default=Config.IMG_SIZE, help="Square image size for training.")
    parser.add_argument("--num-workers", type=int, default=Config.NUM_WORKERS, help="Dataloader workers.")
    parser.add_argument(
        "--arch",
        default=Config.MODEL,
        choices=["dino_seg", "dinov2_uperhead", "dinov2_unet", "medsam"],
        help="Model architecture to train.",
    )
    parser.add_argument("--model-name", default=Config.TIMM_MODEL, help="timm model name.")
    parser.add_argument(
        "--dinov2-id",
        default="facebook/dinov2-base",
        help="HuggingFace model id for DINOv2 when using dinov2_uperhead/dinov2_unet.",
    )
    parser.add_argument(
        "--medsam-id",
        default="flaviagiammarino/medsam-vit-base",
        help="HuggingFace model id for MedSAM when using medsam.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze backbone weights during training.",
    )
    parser.add_argument(
        "--use-hf-processor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use HuggingFace AutoImageProcessor for transformer architectures.",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights.")
    parser.add_argument("--experiment-name", default=Config.EXPERIMENT_NAME, help="MLflow experiment name.")
    parser.add_argument("--mlflow-uri", default=None, help="Optional MLflow tracking URI (e.g., http://localhost:5000).")
    parser.add_argument("--output-dir", default=Config.OUTPUT_DIR, help="Where to save model weights.")
    parser.add_argument("--seed", type=int, default=Config.SEED, help="Random seed.")
    parser.add_argument("--pos-weight", type=float, default=None, help="Positive class weight for BCE (e.g., 5.0).")
    parser.add_argument("--no-folds", action="store_true", help="Skip k-fold CV and train a single split.")
    parser.add_argument(
        "--split-strategy",
        choices=["kfold", "area_bins"],
        default="kfold",
        help="Split strategy for train/val.",
    )
    parser.add_argument("--area-bins", type=int, default=5, help="Number of area bins for area_bins split.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio for area_bins split.")
    parser.add_argument(
        "--use-supplemental",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include supplemental images in training.",
    )
    parser.add_argument(
        "--use-synthetic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include synthetic multi-panel dataset in training.",
    )
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
    if args.split_strategy == "area_bins" and not args.no_folds:
        raise ValueError("split_strategy=area_bins requires --no-folds (single train/val split).")
    prepare_splits(
        repo_root,
        splits_dir,
        n_splits=args.n_splits,
        seed=args.seed,
        include_supplemental=args.use_supplemental,
        split_strategy=args.split_strategy,
        area_bins=args.area_bins,
        val_ratio=args.val_ratio,
        include_synthetic=args.use_synthetic,
    )

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    experiment_name = f"{args.experiment_name}_{timestamp}"
    mlflow.set_experiment(experiment_name)

    folds_to_run = [0] if args.no_folds else list(range(args.n_splits))

    use_hf_processor = args.use_hf_processor
    if use_hf_processor is None:
        use_hf_processor = args.arch != "dino_seg"
    hf_processor = None
    if use_hf_processor:
        from transformers import AutoImageProcessor

        processor_id = args.medsam_id if args.arch == "medsam" else args.dinov2_id
        hf_processor = AutoImageProcessor.from_pretrained(processor_id)

    for fold in folds_to_run:
        train_df, val_df = load_fold(splits_dir, fold)
        train_ds = ForgeryDataset(
            train_df,
            img_size=args.img_size,
            augment=True,
            processor=hf_processor,
            include_supplemental=args.use_supplemental,
            include_synthetic=args.use_synthetic,
        )
        val_ds = ForgeryDataset(
            val_df,
            img_size=args.img_size,
            augment=False,
            processor=hf_processor,
            include_supplemental=args.use_supplemental,
            include_synthetic=args.use_synthetic,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        if args.arch == "dino_seg":
            model = DinoSegModel(model_name=args.model_name, pretrained=not args.no_pretrained, img_size=args.img_size)
        elif args.arch == "dinov2_uperhead":
            if args.img_size % 14 != 0:
                raise ValueError("dinov2_uperhead requires img_size divisible by 14 (DINOv2 patch size).")
            from dinov2_uperhead import DinoV2_UPerNet

            model = DinoV2_UPerNet(dinov2_id=args.dinov2_id, num_classes=1)
        elif args.arch == "dinov2_unet":
            if args.img_size % 14 != 0:
                raise ValueError("dinov2_unet requires img_size divisible by 14 (DINOv2 patch size).")
            from dinov2_unet import DinoV2UNet

            model = DinoV2UNet(dinov2_id=args.dinov2_id, out_classes=1)
        elif args.arch == "medsam":
            from medsam_seg import MedSAMSegModel

            model = MedSAMSegModel(model_id=args.medsam_id, out_classes=1)
        else:
            raise ValueError(f"Unknown arch: {args.arch}")
        model.to(device)
        if args.freeze_backbone:
            if args.arch == "dino_seg":
                for p in model.backbone.parameters():
                    p.requires_grad = False
                model.backbone.eval()
            elif args.arch == "dinov2_uperhead":
                for p in model.backbone.encoder.parameters():
                    p.requires_grad = False
                model.backbone.encoder.eval()
            elif args.arch == "dinov2_unet":
                model.backbone.freeze()
                model.backbone.encoder.eval()
            elif args.arch == "medsam":
                for p in model.encoder.parameters():
                    p.requires_grad = False
                model.encoder.eval()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
        use_amp = device.type == "cuda"
        scaler = GradScaler(enabled=use_amp) if use_amp else None
        scheduler = None
        warmup_epochs = max(0, int(args.warmup_epochs))
        if args.lr_scheduler == "cosine":
            if warmup_epochs >= args.epochs:
                raise ValueError("--warmup-epochs must be < total epochs for cosine schedule.")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - warmup_epochs)
            )
        elif args.lr_scheduler == "onecycle":
            if warmup_epochs > 0:
                print("Warning: --warmup-epochs is ignored for onecycle (use its internal warmup).")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr,
                epochs=args.epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.1,
            )

        with mlflow.start_run(run_name=f"fold_{fold}", nested=False):
            run_cfg = Config.to_dict()
            run_cfg.update(
                {
                    "EPOCHS": args.epochs,
                    "BATCH": args.batch_size,
                    "LR": args.lr,
                    "LR_SCHEDULER": args.lr_scheduler,
                    "WARMUP_EPOCHS": args.warmup_epochs,
                    "EARLY_STOPPING_PATIENCE": args.early_stopping_patience,
                    "IMG_SIZE": args.img_size,
                    "NUM_WORKERS": args.num_workers,
                    "N_SPLITS": args.n_splits,
                    "TIMM_MODEL": args.model_name,
                    "PRETRAINED": not args.no_pretrained,
                    "MODEL": args.arch,
                    "DINOV2_ID": args.dinov2_id,
                    "MEDSAM_ID": args.medsam_id,
                    "SEED": args.seed,
                    "POS_WEIGHT": args.pos_weight,
                    "NO_FOLDS": args.no_folds,
                    "USE_SUPPLEMENTAL": args.use_supplemental,
                    "SPLIT_STRATEGY": args.split_strategy,
                    "AREA_BINS": args.area_bins,
                    "VAL_RATIO": args.val_ratio,
                    "USE_HF_PROCESSOR": use_hf_processor,
                    "FREEZE_BACKBONE": args.freeze_backbone,
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
                    "lr_scheduler": args.lr_scheduler,
                    "warmup_epochs": args.warmup_epochs,
                    "early_stopping_patience": args.early_stopping_patience,
                    "img_size": args.img_size,
                    "model_name": args.model_name,
                    "pretrained": not args.no_pretrained,
                    "arch": args.arch,
                    "dinov2_id": args.dinov2_id,
                    "medsam_id": args.medsam_id,
                    "pos_weight": args.pos_weight,
                    "no_folds": args.no_folds,
                    "use_supplemental": args.use_supplemental,
                    "split_strategy": args.split_strategy,
                    "area_bins": args.area_bins,
                    "val_ratio": args.val_ratio,
                    "use_hf_processor": use_hf_processor,
                    "freeze_backbone": args.freeze_backbone,
                    "machine_specs": json.dumps({"ram_gb": 64, "cpu_cores": 16, "gpu": "RTX 5070 Ti"}),
                }
            )
            mlflow.log_artifacts(str(splits_dir), artifact_path="splits")

            best_dice = 0.0
            epochs_since_best = 0
            model_tag = Path(args.medsam_id).name if args.arch == "medsam" else Path(args.model_name).name
            best_path = output_dir / f"{model_tag}_fold{fold}.pt"
            history: list[dict] = []

            for epoch in range(1, args.epochs + 1):
                if args.lr_scheduler == "cosine" and warmup_epochs > 0 and epoch <= warmup_epochs:
                    warmup_lr = args.lr * (epoch / warmup_epochs)
                    set_optimizer_lr(optimizer, warmup_lr)
                train_loss, train_dice = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scaler,
                    device,
                    pos_weight=args.pos_weight,
                )
                if args.lr_scheduler == "onecycle" and scheduler is not None:
                    scheduler.step()
                elif args.lr_scheduler == "cosine" and scheduler is not None and epoch > warmup_epochs:
                    scheduler.step()
                val_loss, val_dice = validate(
                    model,
                    val_loader,
                    device,
                    pos_weight=args.pos_weight,
                )
                current_lr = optimizer.param_groups[0]["lr"]

                metrics = {
                    f"train_loss_fold{fold}": train_loss,
                    f"train_dice_fold{fold}": train_dice,
                    f"val_loss_fold{fold}": val_loss,
                    f"val_dice_fold{fold}": val_dice,
                    "train_loss": train_loss,
                    "train_dice": train_dice,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "lr": current_lr,
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
                    epochs_since_best = 0
                    torch.save({"model_state": model.state_dict(), "epoch": epoch}, best_path)
                    mlflow.log_artifact(str(best_path), artifact_path=f"fold_{fold}_weights")
                    print(
                        f"[Fold {fold}][Epoch {epoch}] new best val_dice={best_dice:.4f} 🎉 "
                        f"(saved {best_path.name})"
                    )
                else:
                    epochs_since_best += 1
                    if args.early_stopping_patience > 0 and epochs_since_best >= args.early_stopping_patience:
                        print(
                            f"[Fold {fold}] early stopping after {epoch} epochs "
                            f"(no val_dice improvement for {epochs_since_best} epochs)"
                        )
                        break

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
