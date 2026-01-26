#!/usr/bin/env python3
"""
yolo_grid_search_cc.py

Grid-search thresholds for your **YOLO-crop + seg + merge** pipeline (exactly like your inference code).

What it does:
1) For each image in a CSV (val_fold0.csv etc.), run YOLO -> crop -> seg -> merge:
   - Build merged binary mask (union of per-crop pred_bin) at full resolution
   - Build merged probability map prob_vis (avg of per-crop prob_full) at full resolution
2) Precompute for each min_cc in cc_range:
   - area_ratio_cc[i,k]  = mask_area / (H*W)    <-- scale-invariant
   - mean_cc[i,k]        = mean(prob_vis inside CC-filtered mask)
   - rle_cc[i][k]        = RLE for that CC-filtered mask (or "authentic")
3) Grid search over (area_ratio_thr, mean_thr, min_cc) to maximize competition score.
4) Writes:
   - outdir/submission.csv (best params, on THIS CSV)
   - outdir/analysis/run_metadata.json
   - outdir/analysis/args.json

Notes:
- Uses your exact seg forward, finalize_mask(), and pred_bin union logic.
- Uses area_ratio threshold (recommended). If you insist on pixel area threshold, flip a flag below.
- Requires utils.py: rle_encode, evaluate_single_image, score (competition_score).
"""

import argparse
import itertools
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from ultralytics import YOLO
from utils import rle_encode, evaluate_single_image, score as competition_score


# -------------------------
# Config defaults (override via CLI)
# -------------------------
@dataclass
class Defaults:
    img_size: int = 532
    margin: int = 20

    yolo_img_size: int = 1024
    yolo_conf: float = 0.25
    yolo_iou: float = 0.60

    # grid search ranges
    # area_ratio ~= 0.001 => 0.1% of image pixels
    area_ratio_range: Tuple[float, ...] = (
        5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 8e-4,
        1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3,
        8e-3, 1e-2
    )
    mean_range: Tuple[float, ...] = tuple(np.round(np.arange(0.35, 0.86, 0.05), 2).tolist())
    cc_range: Tuple[int, ...] = (0, 50, 100, 200, 400, 800, 1000)

    # If True: uses area_ratio thresholds.
    # If False: uses pixel area thresholds (less transferable across sizes).
    use_area_ratio: bool = True


# -------------------------
# Pre/post helpers (match your code)
# -------------------------
def apply_clahe(image: Image.Image) -> Image.Image:
    if cv2 is None:
        raise RuntimeError("cv2 is required.")
    img_np = np.array(image)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)


def enhanced_adaptive_mask(prob: np.ndarray, alpha_grad: float = 0.45):
    if cv2 is None:
        raise RuntimeError("cv2 is required.")
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim == 3:
        prob = prob[..., 0]

    gx = cv2.Sobel(prob, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_norm = grad_mag / (grad_mag.max() + 1e-6)

    enhanced = (1 - alpha_grad) * prob + alpha_grad * grad_norm
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # keep your current coeff (0.1) from the code you pasted
    thr = float(np.mean(enhanced) + 0.1 * np.std(enhanced))
    mask = (enhanced > thr).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return mask, thr


def finalize_mask(prob_small: np.ndarray, orig_size_wh: Tuple[int, int]):
    mask, thr = enhanced_adaptive_mask(prob_small)
    w, h = orig_size_wh
    mask = cv2.resize(mask, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
    return mask, thr


def remove_small_components(mask01: np.ndarray, min_area: int) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("cv2 is required.")
    if min_area is None or min_area <= 0:
        return (mask01 > 0).astype(np.uint8)

    m = (mask01 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    out = np.zeros_like(m, dtype=np.uint8)
    for lab in range(1, num_labels):
        if stats[lab, cv2.CC_STAT_AREA] >= int(min_area):
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


# -------------------------
# Model loading
# -------------------------
def load_seg_model(arch: str, weights_path: str, device: torch.device, dinov2_id: str):
    state = torch.load(weights_path, map_location=device)
    model_state = state["model_state"]

    if arch == "dinov2_uperhead":
        from dinov2_uperhead import DinoV2_UPerNet
        model = DinoV2_UPerNet(dinov2_id=dinov2_id, num_classes=1).to(device)
    elif arch == "dinov2_unet":
        from dinov2_unet import DinoV2UNet
        model = DinoV2UNet(dinov2_id=dinov2_id, out_classes=1).to(device)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.load_state_dict(model_state)
    model.eval()
    return model


# -------------------------
# YOLO-crop + seg + merge (matches your inference)
# -------------------------
def make_img_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def seg_forward_prob_small(seg_model, img_pil: Image.Image, img_transform, device) -> np.ndarray:
    """
    Returns prob_small (S,S) float32 in [0,1].
    """
    img_pil = apply_clahe(img_pil)
    x = img_transform(img_pil).unsqueeze(0).to(device)
    out = seg_model(x)
    logits = out if torch.is_tensor(out) else out[0]
    prob_small = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    return prob_small


@torch.no_grad()
def yolo_seg_merge(
    image_pil: Image.Image,
    image_path_for_yolo: str,
    detect_model,
    seg_model,
    img_transform,
    device,
    img_size: int,
    margin: int,
    yolo_conf: float,
    yolo_iou: float,
    yolo_imgsz: int,
):
    """
    Returns:
      pred_union: (H,W) uint8 {0,1}   union of per-crop pred_bin (no CC here)
      prob_vis  : (H,W) float32      avg prob where covered (prob_accum/prob_count)
      boxes_xyxy: (N,4) float32 or None
    """
    W, H = image_pil.size
    pred_union = np.zeros((H, W), dtype=np.uint8)
    prob_accum = np.zeros((H, W), dtype=np.float32)
    prob_count = np.zeros((H, W), dtype=np.float32)

    preds = detect_model.predict(
        source=image_path_for_yolo,
        conf=yolo_conf,
        iou=yolo_iou,
        imgsz=yolo_imgsz,
        verbose=False,
    )
    boxes = preds[0].boxes
    boxes_xyxy = None if (boxes is None or len(boxes) == 0) else boxes.xyxy.cpu().numpy()

    # whole-image fallback if <=1 box (your logic)
    if boxes is None or len(boxes) <= 1:
        prob_small = seg_forward_prob_small(seg_model, image_pil, img_transform, device)
        mask_full, _ = finalize_mask(prob_small, (W, H))
        pred_bin = (mask_full > 0).astype(np.uint8)

        pred_union = np.maximum(pred_union, pred_bin)

        prob_full = cv2.resize(prob_small, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        prob_accum += prob_full
        prob_count += 1.0

        prob_vis = prob_accum / np.maximum(prob_count, 1e-6)
        return pred_union, prob_vis, boxes_xyxy

    # crop loop
    for bbox in boxes_xyxy:
        x1, y1, x2, y2 = bbox.tolist()
        x1 = int(max(0, x1)); y1 = int(max(0, y1))
        x2 = int(min(W, x2)); y2 = int(min(H, y2))

        x1m = int(min(W, x1 + margin))
        y1m = int(min(H, y1 + margin))
        x2m = int(max(0, x2 - margin))
        y2m = int(max(0, y2 - margin))

        if x2m <= x1m or y2m <= y1m:
            continue

        crop = image_pil.crop((x1m, y1m, x2m, y2m))
        cw, ch = crop.size

        prob_small = seg_forward_prob_small(seg_model, crop, img_transform, device)

        # bin mask on this crop (your current finalize_mask)
        mask_crop_full, _ = finalize_mask(prob_small, (cw, ch))
        pred_bin_crop = (mask_crop_full > 0).astype(np.uint8)

        # paste union
        pred_union[y1m:y2m, x1m:x2m] = np.maximum(pred_union[y1m:y2m, x1m:x2m], pred_bin_crop)

        # paste prob
        prob_crop_full = cv2.resize(prob_small, (cw, ch), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        prob_accum[y1m:y2m, x1m:x2m] += prob_crop_full
        prob_count[y1m:y2m, x1m:x2m] += 1.0

    prob_vis = prob_accum / np.maximum(prob_count, 1e-6)
    return pred_union, prob_vis, boxes_xyxy


# -------------------------
# Grid search
# -------------------------
def run_grid_search(
    df: pd.DataFrame,
    detect_model,
    seg_model,
    outdir: Path,
    img_size: int,
    margin: int,
    yolo_conf: float,
    yolo_iou: float,
    yolo_imgsz: int,
    cc_range: List[int],
    area_vals: List[float],
    mean_vals: List[float],
    use_area_ratio: bool,
    device: torch.device,
):
    if cv2 is None:
        raise RuntimeError("cv2 not available.")

    outdir.mkdir(parents=True, exist_ok=True)
    analysis_dir = outdir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    img_transform = make_img_transform(img_size)

    # 1) run YOLO-merge inference ONCE per image and cache base artifacts
    predictions = []
    solution_rows = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Caching YOLO-merged outputs"):
        case_id = str(getattr(row, "case_id"))
        image_path = str(getattr(row, "image_path"))
        label = int(getattr(row, "label", 0))

        with open(image_path, "rb") as f:
            image = Image.open(f).convert("RGB")
        W, H = image.size

        pred_union, prob_vis, _ = yolo_seg_merge(
            image_pil=image,
            image_path_for_yolo=image_path,
            detect_model=detect_model,
            seg_model=seg_model,
            img_transform=img_transform,
            device=device,
            img_size=img_size,
            margin=margin,
            yolo_conf=yolo_conf,
            yolo_iou=yolo_iou,
            yolo_imgsz=yolo_imgsz,
        )

        gt_mask = load_gt_mask_from_image_path(image_path, w=W, h=H)
        gt_label = "authentic" if label == 0 else "forged"
        shape_str = json.dumps([int(H), int(W)])

        predictions.append(dict(
            case_id=case_id,
            image_path=image_path,
            W=int(W),
            H=int(H),
            pred_union=pred_union.astype(np.uint8),      # (H,W)
            prob_vis=prob_vis.astype(np.float32),        # (H,W)
            gt_mask=gt_mask.astype(np.uint8),            # (H,W)
            gt_label=gt_label,
            shape_str=shape_str,
        ))

        solution_rows.append(dict(
            case_id=case_id,
            annotation="authentic" if gt_label == "authentic" else rle_encode([gt_mask.astype(np.uint8)]),
            shape=shape_str,
        ))

    solution_df = pd.DataFrame(solution_rows)

    # 2) precompute per-CC variants
    N = len(predictions)
    K = len(cc_range)

    # store either ratio or pixel area depending on flag
    area_cc = np.zeros((N, K), dtype=np.float32)
    mean_cc = np.zeros((N, K), dtype=np.float32)
    rle_cc = [["authentic"] * K for _ in range(N)]

    for i, p in enumerate(tqdm(predictions, desc="Precomputing CC variants")):
        base = (p["pred_union"] > 0).astype(np.uint8)
        prob_vis = p["prob_vis"]
        H = int(p["H"]); W = int(p["W"])
        denom = float(H * W)

        for k, min_cc in enumerate(cc_range):
            m = remove_small_components(base, int(min_cc))
            a = float(m.sum())

            area_cc[i, k] = (a / denom) if use_area_ratio else a

            if a > 0:
                mean_cc[i, k] = float(prob_vis[m == 1].mean())
                rle_cc[i][k] = rle_encode([m.astype(np.uint8)])
            else:
                mean_cc[i, k] = 0.0
                rle_cc[i][k] = "authentic"

    # 3) grid search
    combos = list(itertools.product(area_vals, mean_vals))
    best_score = -1.0
    best_params = {"area_thres": None, "mean_thres": None, "min_cc_area": None}

    case_ids = [p["case_id"] for p in predictions]

    for k, min_cc in enumerate(cc_range):
        a_col = area_cc[:, k]
        m_col = mean_cc[:, k]
        pbar = tqdm(combos, desc=f"GridSearch CC={min_cc}", leave=True)

        for a_thr, m_thr in pbar:
            is_forged = (a_col >= float(a_thr)) & (m_col >= float(m_thr))

            # build submission (fast: no cv2 here)
            submission_rows = [
                {"case_id": case_ids[i], "annotation": (rle_cc[i][k] if is_forged[i] else "authentic")}
                for i in range(N)
            ]
            submission_df = pd.DataFrame(submission_rows)

            curr = competition_score(
                solution=solution_df,
                submission=submission_df,
                row_id_column_name="case_id",
            )

            if float(curr) > best_score:
                best_score = float(curr)
                best_params = {
                    "area_thres": float(a_thr),
                    "mean_thres": float(m_thr),
                    "min_cc_area": int(min_cc),
                    "use_area_ratio": bool(use_area_ratio),
                }
                pbar.set_postfix(best=f"{best_score:.6f}", A=a_thr, M=m_thr)

    # 4) final submission using best params + per-image metrics
    best_cc = int(best_params["min_cc_area"])
    best_k = cc_range.index(best_cc)
    a_thr = float(best_params["area_thres"])
    m_thr = float(best_params["mean_thres"])

    final_rows = []
    per_case = []

    for i, p in enumerate(tqdm(predictions, desc="Final submission")):
        forged = (area_cc[i, best_k] >= a_thr) and (mean_cc[i, best_k] >= m_thr)
        ann = rle_cc[i][best_k] if forged else "authentic"
        final_rows.append({"case_id": p["case_id"], "annotation": ann})

        # compute per-image score for reporting
        label_rles = solution_df.loc[solution_df["case_id"] == p["case_id"], "annotation"].iloc[0]
        pred_rles = ann
        if (label_rles == "authentic") or (pred_rles == "authentic"):
            img_score = 1.0 if (label_rles == pred_rles) else 0.0
        else:
            img_score = float(evaluate_single_image(
                label_rles=label_rles,
                prediction_rles=pred_rles,
                shape_str=p["shape_str"],
            ))

        per_case.append({
            "case_id": p["case_id"],
            "image_path": p["image_path"],
            "W": p["W"],
            "H": p["H"],
            "gt_label": p["gt_label"],
            "area_metric": float(area_cc[i, best_k]),
            "mean_inside": float(mean_cc[i, best_k]),
            "is_forged": bool(forged),
            "image_score": float(img_score),
        })

    submission_path = outdir / "submission.csv"
    pd.DataFrame(final_rows).to_csv(submission_path, index=False)

    meta = {
        "best_score": best_score,
        "best_params": best_params,
        "n_images": int(N),
        "cc_range": list(map(int, cc_range)),
        "area_values": list(map(float, area_vals)),
        "mean_values": list(map(float, mean_vals)),
        "per_case": per_case,
    }
    (analysis_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[INFO] Best score: {best_score:.6f}")
    print(f"[INFO] Best params: {best_params}")
    print(f"[INFO] Wrote: {submission_path}")
    print(f"[INFO] Wrote: {analysis_dir/'run_metadata.json'}")

    return best_score, best_params


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser("YOLO-based grid search for postprocess thresholds")
    parser.add_argument("--csv-path", type=str, required=True, help="CSV with columns: case_id,image_path,label")
    parser.add_argument("--outdir", type=str, default="analysis/yolo_gridsearch", help="Output directory")
    parser.add_argument("--detector-weights", type=str, required=True, help="YOLO best.pt path")
    parser.add_argument("--seg-weights", type=str, required=True, help="Seg .pt path (your checkpoint)")
    parser.add_argument("--arch", choices=["dinov2_uperhead", "dinov2_unet"], default="dinov2_uperhead")
    parser.add_argument("--dinov2-id", type=str, default="facebook/dinov2-base")

    parser.add_argument("--img-size", type=int, default=Defaults.img_size)
    parser.add_argument("--margin", type=int, default=Defaults.margin)

    parser.add_argument("--yolo-img-size", type=int, default=Defaults.yolo_img_size)
    parser.add_argument("--yolo-conf", type=float, default=Defaults.yolo_conf)
    parser.add_argument("--yolo-iou", type=float, default=Defaults.yolo_iou)

    parser.add_argument("--use-area-ratio", action=argparse.BooleanOptionalAction, default=Defaults.use_area_ratio)

    # Optional: override grid ranges quickly
    parser.add_argument("--cc-range", type=str, default="0,50,100,200,400,800,1000",
                        help="Comma-separated CC min areas")
    parser.add_argument("--mean-range", type=str, default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85",
                        help="Comma-separated mean thresholds")
    parser.add_argument("--area-range", type=str, default="5e-5,1e-4,2e-4,3e-4,5e-4,8e-4,1e-3,1.5e-3,2e-3,3e-3,5e-3,8e-3,1e-2",
                        help="Comma-separated area thresholds (ratio if --use-area-ratio, else pixels)")

    args = parser.parse_args()

    if cv2 is None:
        raise RuntimeError("opencv-python not available (cv2).")

    outdir = Path(args.outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    needed = {"case_id", "image_path", "label"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # parse ranges
    cc_range = [int(x.strip()) for x in args.cc_range.split(",") if x.strip() != ""]
    mean_range = [float(x.strip()) for x in args.mean_range.split(",") if x.strip() != ""]
    area_range = [float(x.strip()) for x in args.area_range.split(",") if x.strip() != ""]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading YOLO detector...")
    detect_model = YOLO(args.detector_weights)

    print("[INFO] Loading segmentation model...")
    seg_model = load_seg_model(args.arch, args.seg_weights, device=device, dinov2_id=args.dinov2_id)

    # Save args
    (outdir / "analysis" / "args.json").write_text(json.dumps(vars(args), indent=2))

    run_grid_search(
        df=df,
        detect_model=detect_model,
        seg_model=seg_model,
        outdir=outdir,
        img_size=args.img_size,
        margin=args.margin,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_imgsz=args.yolo_img_size,
        cc_range=cc_range,
        area_vals=area_range,
        mean_vals=mean_range,
        use_area_ratio=bool(args.use_area_ratio),
        device=device,
    )


if __name__ == "__main__":
    main()
