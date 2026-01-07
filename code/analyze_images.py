"""
Quick dataset inspection script for computing pixel stats and image size distribution.
Usage:
    python code/analyze_images.py
Options:
    --roots train_images supplemental_images test_images
    --outdir analysis
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # Disable interactive backends for headless runs.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from tqdm import tqdm


def find_images(roots: Sequence[Path]) -> list[tuple[Path, Path]]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    paths: list[tuple[Path, Path]] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append((root, p))
    return paths


def percentile(vals: list[int], pct: float) -> float:
    if not vals:
        return float("nan")
    return float(np.percentile(np.array(vals), pct))


def summarize_dims(dim_counts: Counter[tuple[int, int]], widths: list[int], heights: list[int]) -> str:
    lines: list[str] = []
    if not dim_counts:
        return "No images found."
    top_dims = dim_counts.most_common(10)
    lines.append(f"Unique dims: {len(dim_counts)}; top 10 (w,h,count):")
    for (w, h), cnt in top_dims:
        lines.append(f"  {w}x{h}: {cnt}")
    lines.append(
        "Width stats (min/median/mean/max): "
        f"{min(widths)} / {percentile(widths,50):.1f} / {np.mean(widths):.1f} / {max(widths)}"
    )
    lines.append(
        "Height stats (min/median/mean/max): "
        f"{min(heights)} / {percentile(heights,50):.1f} / {np.mean(heights):.1f} / {max(heights)}"
    )
    return "\n".join(lines)


def classify_root(root: Path) -> str:
    name = root.name.lower()
    if "test" in name:
        return "test"
    return "train_supplement"


def new_accumulator() -> dict:
    return {
        "total_pixels": 0,
        "channel_sum": np.zeros(3, dtype=np.float64),
        "channel_sumsq": np.zeros(3, dtype=np.float64),
        "dim_counts": Counter(),
        "widths": [],
        "heights": [],
        "images": 0,
    }


def update_acc(acc: dict, w: int, h: int, pixels: np.ndarray) -> None:
    acc["dim_counts"][(w, h)] += 1
    acc["widths"].append(w)
    acc["heights"].append(h)
    acc["channel_sum"] += pixels.sum(axis=0)
    acc["channel_sumsq"] += np.square(pixels).sum(axis=0)
    acc["total_pixels"] += pixels.shape[0]
    acc["images"] += 1


def compute_stats(acc: dict) -> dict | None:
    if acc["images"] == 0 or acc["total_pixels"] == 0:
        return None
    mean = acc["channel_sum"] / acc["total_pixels"]
    var = acc["channel_sumsq"] / acc["total_pixels"] - np.square(mean)
    std = np.sqrt(np.maximum(var, 0.0))
    dims = acc["dim_counts"]
    widths = acc["widths"]
    heights = acc["heights"]
    return {
        "images_processed": acc["images"],
        "total_pixels": int(acc["total_pixels"]),
        "mean_uint8": mean.tolist(),
        "std_uint8": std.tolist(),
        "mean_0_1": (mean / 255.0).tolist(),
        "std_0_1": (std / 255.0).tolist(),
        "width": {
            "min": int(min(widths)),
            "median": float(percentile(widths, 50)),
            "mean": float(np.mean(widths)),
            "max": int(max(widths)),
        },
        "height": {
            "min": int(min(heights)),
            "median": float(percentile(heights, 50)),
            "mean": float(np.mean(heights)),
            "max": int(max(heights)),
        },
        "top_dims": [
            {"width": int(w), "height": int(h), "count": int(cnt)}
            for (w, h), cnt in dims.most_common(50)
        ],
    }


def write_dim_counts(dim_counts: Counter[tuple[int, int]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("width,height,count\n")
        for (w, h), cnt in dim_counts.most_common():
            f.write(f"{w},{h},{cnt}\n")


def save_dim_plot(name: str, widths: list[int], heights: list[int], outdir: Path) -> Path:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(widths, bins=40, ax=axes[0], color="#1f77b4")
    axes[0].set(title=f"{name} width distribution", xlabel="Width (px)", ylabel="Count")
    sns.histplot(heights, bins=40, ax=axes[1], color="#ff7f0e")
    axes[1].set(title=f"{name} height distribution", xlabel="Height (px)", ylabel="Count")
    fig.suptitle(f"Image dimension distribution: {name}", fontsize=14)
    fig.tight_layout()
    plot_path = outdir / f"dimension_distribution_{name}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def save_dim_heatmap(name: str, widths: list[int], heights: list[int], outdir: Path) -> Path:
    if not widths or not heights:
        return outdir / f"dimension_heatmap_{name}.png"
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.histplot(x=widths, y=heights, bins=60, pthresh=0.05, cmap="mako", ax=ax)
    ax.set(title=f"Width vs Height density: {name}", xlabel="Width (px)", ylabel="Height (px)")
    fig.tight_layout()
    plot_path = outdir / f"dimension_heatmap_{name}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def main(argv: Iterable[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_roots = ["train_images", "supplemental_images", "test_images"]

    parser = argparse.ArgumentParser(description="Compute pixel mean/std and dimension distribution.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=default_roots,
        help="Relative paths (from repo root) to scan for images.",
    )
    parser.add_argument(
        "--outdir",
        default="analysis",
        help="Relative path (from repo root) to write stats and plots.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    roots = [repo_root / r for r in args.roots]
    outdir = (repo_root / args.outdir).resolve()

    image_entries = find_images(roots)
    if not image_entries:
        print("No images found in provided roots.", file=sys.stderr)
        return 1

    accs = {
        "all": new_accumulator(),
        "train_supplement": new_accumulator(),
        "test": new_accumulator(),
    }

    for root, path in tqdm(image_entries, desc="Images", unit="img"):
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                arr = np.asarray(img, dtype=np.float32)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Failed to read {path}: {exc}", file=sys.stderr)
            continue

        h, w = arr.shape[:2]
        pixels = arr.reshape(-1, 3)

        update_acc(accs["all"], w, h, pixels)
        group = classify_root(root)
        update_acc(accs.get(group, accs["all"]), w, h, pixels)

    outdir.mkdir(parents=True, exist_ok=True)

    stats_all = compute_stats(accs["all"])
    stats_train = compute_stats(accs["train_supplement"])
    stats_test = compute_stats(accs["test"])

    stats = {
        "roots": [str(r) for r in roots],
        "overall": stats_all,
        "train_supplement": stats_train,
        "test": stats_test,
    }
    (outdir / "pixel_stats.json").write_text(json.dumps(stats, indent=2))

    # Dimension counts CSVs.
    write_dim_counts(accs["all"]["dim_counts"], outdir / "dimension_counts_all.csv")
    if accs["train_supplement"]["images"] > 0:
        write_dim_counts(accs["train_supplement"]["dim_counts"], outdir / "dimension_counts_train_supplement.csv")
    if accs["test"]["images"] > 0:
        write_dim_counts(accs["test"]["dim_counts"], outdir / "dimension_counts_test.csv")

    plot_paths = []
    heatmap_paths = []
    if accs["all"]["images"] > 0:
        plot_paths.append(save_dim_plot("all", accs["all"]["widths"], accs["all"]["heights"], outdir))
        heatmap_paths.append(save_dim_heatmap("all", accs["all"]["widths"], accs["all"]["heights"], outdir))
    if accs["train_supplement"]["images"] > 0:
        plot_paths.append(
            save_dim_plot(
                "train_supplement", accs["train_supplement"]["widths"], accs["train_supplement"]["heights"], outdir
            )
        )
        heatmap_paths.append(
            save_dim_heatmap(
                "train_supplement", accs["train_supplement"]["widths"], accs["train_supplement"]["heights"], outdir
            )
        )
    if accs["test"]["images"] > 0:
        plot_paths.append(save_dim_plot("test", accs["test"]["widths"], accs["test"]["heights"], outdir))
        heatmap_paths.append(save_dim_heatmap("test", accs["test"]["widths"], accs["test"]["heights"], outdir))

    print("Scanned roots:")
    for r in roots:
        exists = "exists" if r.exists() else "missing"
        print(f"  {r} ({exists})")

    print(f"Images processed (overall): {accs['all']['images']}")
    if stats_all:
        print("\nPixel statistics (uint8 scale 0-255):")
        print(f"  Mean: {np.array(stats_all['mean_uint8'])}")
        print(f"  Std:  {np.array(stats_all['std_uint8'])}")
        print("Pixel statistics (0-1 scale):")
        print(f"  Mean: {np.array(stats_all['mean_0_1'])}")
        print(f"  Std:  {np.array(stats_all['std_0_1'])}")

        print("\nDimension distribution (overall):")
        print(summarize_dims(accs["all"]["dim_counts"], accs["all"]["widths"], accs["all"]["heights"]))

    if stats_train:
        print("\nTrain/Supplement dimension distribution:")
        print(
            summarize_dims(
                accs["train_supplement"]["dim_counts"],
                accs["train_supplement"]["widths"],
                accs["train_supplement"]["heights"],
            )
        )
    if stats_test:
        print("\nTest dimension distribution:")
        print(
            summarize_dims(
                accs["test"]["dim_counts"],
                accs["test"]["widths"],
                accs["test"]["heights"],
            )
        )

    print(f"\nStats written to: {outdir}")
    if plot_paths:
        print("Distribution plots:")
        for p in plot_paths:
            print(f"  {p}")
    if heatmap_paths:
        print("Width x Height density heatmaps:")
        for p in heatmap_paths:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
