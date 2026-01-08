"""
Utility to build a dataframe of training images/masks and create K-fold CSV splits.

Usage:
    python code/data_split.py --n-splits 5 --outdir analysis/splits
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_dataframe(repo_root: Path, include_supplemental: bool = True) -> pd.DataFrame:
    """
    Build a dataframe describing images and masks.

    Columns:
        case_id: image id without extension
        image_path: absolute path to image
        mask_path: absolute path to mask (may be None/empty for authentic)
        label: 1 for forged (mask present), 0 for authentic
        source: 'train', 'supplemental'
    """
    rows = []
    root = Path(repo_root)

    # Train set: authentic and forged.
    auth_dir = root / "train_images" / "authentic"
    forged_dir = root / "train_images" / "forged"
    mask_dir = root / "train_masks"

    for img_path in auth_dir.glob("*.png"):
        case_id = img_path.stem
        mask_path = mask_dir / f"{case_id}.npy"
        rows.append(
            {
                "case_id": case_id,
                "image_path": str(img_path.resolve()),
                "mask_path": str(mask_path.resolve()) if mask_path.exists() else "",
                "label": 0,
                "source": "train",
                "variant": "authentic",
            }
        )

    for img_path in forged_dir.glob("*.png"):
        case_id = img_path.stem
        mask_path = mask_dir / f"{case_id}.npy"
        rows.append(
            {
                "case_id": case_id,
                "image_path": str(img_path.resolve()),
                "mask_path": str(mask_path.resolve()) if mask_path.exists() else "",
                "label": 1,
                "source": "train",
                "variant": "forged",
            }
        )

    if include_supplemental:
        supp_dir = root / "supplemental_images"
        supp_mask_dir = root / "supplemental_masks"
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

    df = pd.DataFrame(rows)
    return df


def make_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    y = df["label"].values
    groups = df["case_id"].values
    for train_idx, val_idx in sgkf.split(df, y, groups):
        folds.append((df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)))
    return folds


def save_folds(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]], out_dir: Path, prefix: Optional[str] = None
) -> list[tuple[Path, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[tuple[Path, Path]] = []
    for i, (train_df, val_df) in enumerate(folds):
        pre = f"{prefix}_" if prefix else ""
        train_path = out_dir / f"{pre}train_fold{i}.csv"
        val_path = out_dir / f"{pre}val_fold{i}.csv"
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        paths.append((train_path, val_path))
    return paths


def plot_fold_distributions(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]], out_dir: Path, prefix: str = ""
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, (train_df, val_df) in enumerate(folds):
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        for a, df, title in zip(ax, (train_df, val_df), ("Train", "Val")):
            counts = df["label"].value_counts().reindex([0, 1], fill_value=0)
            a.bar(["authentic", "forged"], counts.values, color=["#4c78a8", "#f58518"])
            a.set_title(f"{title} fold {i}")
            a.set_ylabel("count")
        fig.tight_layout()
        pre = f"{prefix}_" if prefix else ""
        path = out_dir / f"{pre}fold{i}_label_dist.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Create K-fold CSV splits for forgery dataset.")
    parser.add_argument("--repo-root", default=".", help="Repository root containing train_images/train_masks.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--outdir", default="analysis/splits", help="Output directory for CSV files.")
    parser.add_argument("--prefix", default="", help="Optional prefix for file names.")
    parser.add_argument(
        "--plot-distribution",
        action="store_true",
        help="Plot authentic vs forged distributions per fold into analysis/.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    df = build_dataframe(repo_root)
    folds = make_folds(df, n_splits=args.n_splits, seed=args.seed)
    paths = save_folds(folds, Path(args.outdir), prefix=args.prefix)
    print(f"Built dataframe with {len(df)} rows; saved {len(paths)} folds to {args.outdir}")
    if args.plot_distribution:
        plot_dir = Path("analysis")
        plot_paths = plot_fold_distributions(folds, plot_dir, prefix=args.prefix)
        print(f"Saved {len(plot_paths)} distribution plots to {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
