#!/usr/bin/env python3
"""
val_inference_cc_precompute.py

Pipeline:
1) Run model inference once on CSV.
2) Cache per-image: pred_mask (orig-res), prob (model-res), thr, gt_mask, gt_label, shape_str.
3) Precompute CC-filtered variants for each min_cc in cc_range:
   - area_cc[i,k], mean_cc[i,k], rle_cc[i][k]
4) Grid search over (area_thr, mean_thr, min_cc) using ONLY precomputed arrays/strings (fast).
5) Final run uses best params to write submission.csv.
6) Optional: save collages (controlled by SAVE_COLLAGES constant, NOT a CLI arg).
7) Save analysis plots + run_metadata.json.

Requires:
- utils.py with: rle_encode, evaluate_single_image, score as competition_score
- model modules: dinov2_unet.py, dinov2_uperhead.py, dinov2_seg.py (optional)
"""

import argparse
import json
import math
import shutil
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from utils import evaluate_single_image, rle_encode, score as competition_score


# ============================================================
# Toggle this here (NOT a CLI arg)
# ============================================================
SAVE_COLLAGES = False  # set True when you want collages


# -------------------------
# Dataset
# -------------------------
class InferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, processor=None):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.processor = processor

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            self.to_tensor,
            self.normalize,
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        case_id = str(row["case_id"])
        image_path = str(row["image_path"])
        label = int(row.get("label", 0))

        with open(image_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        orig_size = img.size  # (W,H)

        if self.processor is not None:
            img_rs = transforms.functional.resize(img, (self.img_size, self.img_size))
            processed = self.processor(
                images=img_rs,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
                do_normalize=True,
                do_rescale=True,
            )
            img_t = processed["pixel_values"].squeeze(0)
        else:
            img_t = self.img_transform(img)

        return img_t, case_id, orig_size, image_path, label


def collate_fn(batch):
    images, case_ids, orig_sizes, image_paths, labels = zip(*batch)
    images = default_collate(images)
    return images, list(case_ids), list(orig_sizes), list(image_paths), list(labels)


# -------------------------
# Helpers
# -------------------------
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


def enhanced_adaptive_mask(prob: np.ndarray, alpha_grad=0.45):
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim == 3:
        prob = prob[..., 0]

    gx = cv2.Sobel(prob, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_norm = grad_mag / (grad_mag.max() + 1e-6)

    enhanced = (1 - alpha_grad) * prob + alpha_grad * grad_norm
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    thr = float(np.mean(enhanced) + 0.3 * np.std(enhanced))
    mask = (enhanced > thr).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask, thr


def finalize_mask(prob: np.ndarray, orig_size_wh: tuple[int, int]):
    # orig_size_wh MUST be (W,H)
    mask, thr = enhanced_adaptive_mask(prob)
    w, h = orig_size_wh
    mask = cv2.resize(mask, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
    return mask, thr


def remove_small_components(mask01: np.ndarray, min_area: int) -> np.ndarray:
    if min_area is None or min_area <= 0:
        return (mask01 > 0).astype(np.uint8)

    m = (mask01 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    out = np.zeros_like(m, dtype=np.uint8)
    for lab in range(1, num_labels):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lab] = 1
    return out


def load_gt_mask_from_image_path(image_path: str, w: int, h: int) -> np.ndarray:
    mask_path = (
        image_path
        .replace("train_images", "train_masks")
        .replace("supplemental_images", "supplemental_masks")
        .replace("/forged", "")
        .replace(".png", ".npy")
    )
    if not Path(mask_path).exists():
        return np.zeros((h, w), dtype=np.uint8)

    arr = np.load(mask_path, allow_pickle=True)
    if arr.ndim == 3:
        arr = arr.max(axis=0)
    gt_mask = (arr > 0).astype(np.uint8)
    if gt_mask.shape[:2] != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return gt_mask


def apply_overlay(img_np: np.ndarray, mask01: np.ndarray, color=(255, 0, 0), alpha=0.5):
    overlay = img_np.copy()
    mask_bool = mask01.astype(bool)
    overlay[mask_bool] = (
        alpha * np.array(color, dtype=np.float32)
        + (1 - alpha) * overlay[mask_bool].astype(np.float32)
    ).astype(np.uint8)
    return overlay


def create_collage(
    image_path: str,
    pred_bin: np.ndarray,
    case_id: str,
    score: float,
    out_dir: Path,
    gt_label: str,
):
    with open(image_path, "rb") as f:
        orig_img = Image.open(f).convert("RGB")
    orig_np = np.array(orig_img)
    vis_w, vis_h = orig_img.size  # (W,H)

    if pred_bin.shape[:2] != (vis_h, vis_w):
        pred_bin_vis = cv2.resize(pred_bin, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
    else:
        pred_bin_vis = pred_bin

    gt_mask = load_gt_mask_from_image_path(image_path=image_path, w=vis_w, h=vis_h)
    if gt_mask.shape[:2] != (vis_h, vis_w):
        gt_mask_vis = cv2.resize(gt_mask, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
    else:
        gt_mask_vis = gt_mask

    pred_mask_rgb = (pred_bin_vis[..., None] * 255).repeat(3, axis=2).astype(np.uint8)
    gt_mask_rgb = (gt_mask_vis[..., None] * 255).repeat(3, axis=2).astype(np.uint8)

    pred_overlay = apply_overlay(orig_np, pred_bin_vis, color=(255, 0, 0))
    gt_overlay = apply_overlay(orig_np, gt_mask_vis, color=(0, 255, 0))

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    titles = [f"Input_{case_id}", "Pred mask", "GT mask", "Pred overlay", "GT overlay"]
    imgs = [orig_np, pred_mask_rgb, gt_mask_rgb, pred_overlay, gt_overlay]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()

    score_str = f"{score:.3f}".replace(".", "pt")
    save_path = out_dir / f"{gt_label}_{case_id}_{score_str}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_hist(values, title, out_path: Path, bins=50):
    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# -------------------------
# Main pipeline
# -------------------------
def run_pipeline(
    df: pd.DataFrame,
    weights_path: Path,
    outdir: Path,
    arch: str,
    model_name: str,
    dinov2_id: str,
    img_size: int,
    device: torch.device,
    hf_processor,
):
    # -------- checkpoint
    state = torch.load(weights_path, map_location=device)
    model_state = state["model_state"]

    if arch == "dino_seg":
        inferred_img_size = infer_img_size_from_state(model_state, img_size)
        if inferred_img_size != img_size:
            print(f"[INFO] Adjusting img_size {img_size} -> {inferred_img_size} from pos_embed.")
            img_size = inferred_img_size
    else:
        if img_size % 14 != 0:
            raise ValueError(f"{arch} requires img_size divisible by 14.")

    # -------- dataset/loader
    dataset = InferenceDataset(df, img_size=img_size, processor=hf_processor)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # -------- model
    if arch == "dino_seg":
        from dinov2_seg import DinoSegModel
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
    print("[INFO] Loaded model")

    # -------- inference once + cache
    predictions = []
    solution_rows = []

    with torch.no_grad():
        for images, case_ids, orig_sizes, image_paths, labels in tqdm(loader, desc="Predicting masks"):
            images = images.to(device, non_blocking=True)
            out = model(images)
            logits = out if torch.is_tensor(out) else out[0]
            probs = torch.sigmoid(logits).float().cpu().numpy()  # (B,1,S,S)

            for i, cid in enumerate(case_ids):
                orig_w, orig_h = orig_sizes[i]
                prob = probs[i, 0]  # (S,S)

                mask_full, thr = finalize_mask(prob, (int(orig_w), int(orig_h)))  # (H,W) orig-res
                gt_mask = load_gt_mask_from_image_path(image_paths[i], int(orig_w), int(orig_h))
                gt_label = "authentic" if int(labels[i]) == 0 else "forged"
                shape_str = json.dumps([int(orig_h), int(orig_w)])  # scorer expects (H,W)

                predictions.append(dict(
                    case_id=str(cid),
                    image_path=image_paths[i],
                    orig_width=int(orig_w),
                    orig_height=int(orig_h),
                    prob=prob,
                    pred_mask=mask_full.astype(np.uint8),
                    gt_mask=gt_mask.astype(np.uint8),
                    gt_label=gt_label,
                    thr=float(thr),
                    shape_str=shape_str,
                ))

                solution_rows.append(dict(
                    case_id=str(cid),
                    annotation="authentic" if gt_label == "authentic" else rle_encode([gt_mask.astype(np.uint8)]),
                    shape=shape_str,
                ))

    solution_df = pd.DataFrame(solution_rows)

    # -------- grid ranges
    area_range = [i * 100 for i in range(1, 11)]
    mean_range = [round(x, 2) for x in np.arange(0.20, 0.9, 0.05)]
    cc_range = [0, 50, 100, 200, 400, 800, 850, 900, 950, 1000]

    combos_2d = list(itertools.product(area_range, mean_range))

    best_score = -1.0
    best_params = {"area_thres": None, "mean_thres": None, "min_cc_area": None}

    # -------------------------
    # 1) Precompute per-image stats/RLE for each CC threshold
    # -------------------------
    N = len(predictions)
    K = len(cc_range)

    area_cc = np.zeros((N, K), dtype=np.int32)
    mean_cc = np.zeros((N, K), dtype=np.float32)
    rle_cc = [["authentic"] * K for _ in range(N)]  # default authentic

    for i, pred in enumerate(tqdm(predictions, desc="Precomputing CC variants", leave=True)):
        prob = pred["prob"]  # (S,S)
        base = (pred["pred_mask"] > 0).astype(np.uint8)  # (H,W)
        prob_h, prob_w = prob.shape

        for k, min_cc in enumerate(cc_range):
            m = remove_small_components(base, int(min_cc))
            a = int(m.sum())
            area_cc[i, k] = a

            if a > 0:
                m_small = cv2.resize(m, (prob_w, prob_h), interpolation=cv2.INTER_NEAREST)
                mean_cc[i, k] = float(prob[m_small == 1].mean()) if (m_small == 1).any() else 0.0
                rle_cc[i][k] = rle_encode([m])
            else:
                mean_cc[i, k] = 0.0
                rle_cc[i][k] = "authentic"

    # -------------------------
    # 2) Grid search (fast, no cv2, no RLE inside hot loop)
    # -------------------------
    case_ids = [p["case_id"] for p in predictions]

    for k, min_cc in enumerate(cc_range):
        pbar = tqdm(combos_2d, desc=f"Grid Search (CC={min_cc})", leave=True)
        a_col = area_cc[:, k]
        m_col = mean_cc[:, k]

        for a_thr, m_thr in pbar:
            is_forged = (a_col >= a_thr) & (m_col >= m_thr)

            # keep the inner loop tight: no extra computation besides list build
            submission_rows = [
                {"case_id": case_ids[i], "annotation": (rle_cc[i][k] if is_forged[i] else "authentic")}
                for i in range(N)
            ]
            submission_df = pd.DataFrame(submission_rows)

            curr_score = competition_score(
                solution=solution_df,
                submission=submission_df,
                row_id_column_name="case_id",
            )

            if curr_score > best_score:
                best_score = float(curr_score)
                best_params = {
                    "area_thres": int(a_thr),
                    "mean_thres": float(m_thr),
                    "min_cc_area": int(min_cc),
                }
                pbar.set_postfix(best=f"{best_score:.6f}", A=a_thr, M=m_thr, CC=min_cc)

    print("\nBEST:", best_score, best_params)

    # -------- output dirs
    outdir.mkdir(parents=True, exist_ok=True)
    analysis_dir = outdir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    collage_dir = outdir / "collage"
    if SAVE_COLLAGES:
        collage_dir.mkdir(parents=True, exist_ok=True)

    best_cc = int(best_params["min_cc_area"])
    best_k = cc_range.index(best_cc)

    # -------- final submission + optional collages
    final_rows = []

    best_area_vals = area_cc[:, best_k].tolist()
    best_mean_vals = mean_cc[:, best_k].tolist()
    thr_vals = [p["thr"] for p in predictions]

    a_thr = int(best_params["area_thres"])
    m_thr = float(best_params["mean_thres"])

    for i, p in enumerate(tqdm(predictions, desc="Final submission", leave=True)):
        is_forged = (area_cc[i, best_k] >= a_thr) and (mean_cc[i, best_k] >= m_thr)
        annotation = (rle_cc[i][best_k] if is_forged else "authentic")
        final_rows.append({"case_id": p["case_id"], "annotation": annotation})

        if SAVE_COLLAGES:
            label_rles = solution_df.loc[solution_df["case_id"] == p["case_id"], "annotation"].iloc[0]
            pred_rles = annotation

            if (label_rles == "authentic") or (pred_rles == "authentic"):
                image_score = 1.0 if (label_rles == pred_rles) else 0.0
            else:
                image_score = float(evaluate_single_image(
                    label_rles=label_rles,
                    prediction_rles=pred_rles,
                    shape_str=p["shape_str"],
                ))

            if annotation == "authentic":
                final_mask = np.zeros_like(p["gt_mask"], dtype=np.uint8)
            else:
                base = (p["pred_mask"] > 0).astype(np.uint8)
                final_mask = remove_small_components(base, best_cc)

            create_collage(
                image_path=p["image_path"],
                pred_bin=final_mask,
                case_id=p["case_id"],
                score=image_score,
                out_dir=collage_dir,
                gt_label=p["gt_label"],
            )

    final_submission_df = pd.DataFrame(final_rows)
    submission_path = outdir / "submission.csv"
    final_submission_df.to_csv(submission_path, index=False)
    print(f"[INFO] Saved: {submission_path}")

    # -------- plots
    save_hist(best_area_vals, f"Mask Area (CC>={best_cc})", analysis_dir / "mask_area_hist.png")
    save_hist(best_mean_vals, f"Mask Mean-Inside (CC>={best_cc})", analysis_dir / "mask_mean_inside_hist.png")
    save_hist(thr_vals, "Finalize Threshold (thr)", analysis_dir / "thr_hist.png")

    # -------- metadata
    run_meta = {
        "best_score": best_score,
        "best_params": best_params,
        "per_case": [
            {
                "case_id": p["case_id"],
                "image_path": p["image_path"],
                "orig_width": p["orig_width"],
                "orig_height": p["orig_height"],
                "gt_label": p["gt_label"],
                "thr": p["thr"],
                "area_cc_best": int(area_cc[i, best_k]),
                "mean_cc_best": float(mean_cc[i, best_k]),
            }
            for i, p in enumerate(predictions)
        ],
    }
    meta_path = analysis_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(run_meta, indent=2))
    print(f"[INFO] Saved: {meta_path}")

    return best_score, best_params


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Val inference + CC-precompute + grid search + submission")
    parser.add_argument("--weights-path", type=str, required=True, help="Path to model weights .pt")
    parser.add_argument("--outdir", type=str, default="analysis/preds", help="Output directory (base)")
    parser.add_argument(
        "--arch",
        choices=["dino_seg", "dinov2_uperhead", "dinov2_unet"],
        default="dinov2_unet",
        help="Model architecture",
    )
    parser.add_argument("--model-name", default="vit_base_patch14_dinov2.lvd142m", help="timm model name (dino_seg)")
    parser.add_argument("--dinov2-id", default="facebook/dinov2-base", help="HF model id for DINOv2")
    parser.add_argument("--use-hf-processor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--img-size", type=int, default=532, help="Resize size")
    parser.add_argument("--csv-path", type=str, required=True, help="CSV path (must include image_path, case_id, label)")

    args = parser.parse_args()

    if cv2 is None:
        raise RuntimeError("cv2 not available. Please install opencv-python.")

    outdir = Path(args.outdir + f"_{args.arch}")
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)

    hf_processor = None
    if args.use_hf_processor:
        from transformers import AutoImageProcessor
        hf_processor = AutoImageProcessor.from_pretrained(args.dinov2_id)

    weights_path = Path(args.weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"No weights file at: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_score, best_params = run_pipeline(
        df=df,
        weights_path=weights_path,
        outdir=outdir,
        arch=args.arch,
        model_name=args.model_name,
        dinov2_id=args.dinov2_id,
        img_size=args.img_size,
        device=device,
        hf_processor=hf_processor,
    )

    (outdir / "analysis" / "args.json").write_text(json.dumps(vars(args), indent=2))

    print("\n======== SUMMARY ========")
    print("SAVE_COLLAGES:", SAVE_COLLAGES)
    print("Best score   :", best_score)
    print("Best params  :", best_params)
    print("Outdir       :", outdir)
    print("=========================\n")


if __name__ == "__main__":
    main()
