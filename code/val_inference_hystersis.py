#!/usr/bin/env python3
"""
val_inference_cc_hysteresis_precompute.py

What it does (end-to-end):
1) Runs segmentation inference once on a CSV (val or any split that has labels + masks available locally).
2) Caches per-image: prob (S,S), orig_size, gt_mask, gt_label, shape_str.
3) Adds HYSTERESIS thresholding (low/high) to convert prob -> binary mask at model-res, then upsamples to orig.
4) Precomputes CC-filtered variants (area, mean_inside, rle) for each (low, high, min_cc).
5) Grid-searches (area_thr, mean_thr, min_cc, low, high) using ONLY precomputed arrays/strings (fast in the hot loop).
6) Writes submission.csv using best params.
7) Writes analysis plots + run_metadata.json.

Notes:
- SAVE_COLLAGES is a constant (NOT a CLI arg).
- You can choose between:
  (A) Full 4D search over (low,high,cc,area,mean) (can be heavy).
  (B) Two-stage search (recommended): tune (low,high) on a subset, then tune (area,mean,cc) with best (low,high).

Dependencies:
- utils.py must provide:
    - rle_encode(list_of_masks)
    - evaluate_single_image(label_rles, prediction_rles, shape_str)
    - score(solution_df, submission_df, row_id_column_name="case_id")  # competition scorer

- Model modules:
    - dinov2_uperhead.py -> DinoV2_UPerNet
    - dinov2_unet.py -> DinoV2UNet
    - dinov2_seg.py -> DinoSegModel (optional)

CSV must include columns:
- case_id (string/int ok), image_path, label (0 authentic, 1 forged)
"""

import argparse
import itertools
import json
import math
import random
import shutil
from pathlib import Path

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
# Toggle these here (NOT CLI args)
# ============================================================
SAVE_COLLAGES = True     # set True when you want collages
USE_TWO_STAGE = True      # recommended; if False -> full 4D grid
TWO_STAGE_SUBSET = 800    # how many samples to tune (low,high) in stage 1 (only used if USE_TWO_STAGE=True)
RANDOM_SEED = 42


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


def load_gt_mask_from_image_path(image_path: str, w: int, h: int) -> np.ndarray:
    """
    Assumes your mask lives at:
      train_images/... -> train_masks/... (same filename .png -> .npy)
      supplemental_images/... -> supplemental_masks/...
      and removes "/forged" in the path if present.
    """
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
# Hysteresis thresholding
# -------------------------
def hysteresis_mask_from_prob(prob: np.ndarray, low: float, high: float, close_ks: int = 5, open_ks: int = 3):
    p = prob.astype(np.float32)

    strong = (p >= high).astype(np.uint8)
    weak   = (p >= low).astype(np.uint8)   # include strong

    num, labels = cv2.connectedComponents(weak, connectivity=8)

    out = np.zeros_like(strong, dtype=np.uint8)
    for lab in range(1, num):
        comp = (labels == lab)
        if (strong[comp] > 0).any():       # component contains strong pixels
            out[comp] = 1

    if close_ks > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((close_ks, close_ks), np.uint8))
    if open_ks > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((open_ks, open_ks), np.uint8))

    return out


def finalize_mask_hysteresis(
    prob: np.ndarray,
    orig_size_wh: tuple[int, int],
    low: float,
    high: float,
    close_ksize: int = 3,
    open_ksize: int = 3,
):
    """
    prob: (S,S) sigmoid map
    returns: (mask_orig(H,W) {0,1}, meta dict)
    """
    mask_small = hysteresis_mask_from_prob(prob, low=low, high=high)

    if close_ksize and close_ksize > 0:
        mask_small = cv2.morphologyEx(
            mask_small, cv2.MORPH_CLOSE, np.ones((close_ksize, close_ksize), np.uint8)
        )
    if open_ksize and open_ksize > 0:
        mask_small = cv2.morphologyEx(
            mask_small, cv2.MORPH_OPEN, np.ones((open_ksize, open_ksize), np.uint8)
        )

    w, h = orig_size_wh
    mask_orig = cv2.resize(mask_small, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
    return mask_orig.astype(np.uint8), {"low": float(low), "high": float(high)}


# -------------------------
# Model loading
# -------------------------
def build_model(arch: str, model_state: dict, model_name: str, dinov2_id: str, img_size: int, device: torch.device):
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
    return model


# -------------------------
# Pipeline
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
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # ----- checkpoint
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

    # ----- dataset/loader
    dataset = InferenceDataset(df, img_size=img_size, processor=hf_processor)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ----- model
    model = build_model(
        arch=arch,
        model_state=model_state,
        model_name=model_name,
        dinov2_id=dinov2_id,
        img_size=img_size,
        device=device,
    )
    print("[INFO] Loaded model")

    # ----- inference once + cache only what we need
    predictions = []
    solution_rows = []

    with torch.no_grad():
        for images, case_ids, orig_sizes, image_paths, labels in tqdm(loader, desc="Inference (cache prob)"):
            images = images.to(device, non_blocking=True)
            out = model(images)
            logits = out if torch.is_tensor(out) else out[0]
            probs = torch.sigmoid(logits).float().cpu().numpy()  # (B,1,S,S)

            for i, cid in enumerate(case_ids):
                ow, oh = orig_sizes[i]  # (W,H)
                prob = probs[i, 0].astype(np.float32)  # (S,S)
                gt_mask = load_gt_mask_from_image_path(image_paths[i], int(ow), int(oh))
                gt_label = "authentic" if int(labels[i]) == 0 else "forged"
                shape_str = json.dumps([int(oh), int(ow)])  # scorer expects (H,W)

                predictions.append(dict(
                    case_id=str(cid),
                    image_path=image_paths[i],
                    orig_width=int(ow),
                    orig_height=int(oh),
                    prob=prob,  # (S,S)
                    gt_mask=gt_mask.astype(np.uint8),  # (H,W)
                    gt_label=gt_label,
                    shape_str=shape_str,
                ))

                solution_rows.append(dict(
                    case_id=str(cid),
                    annotation="authentic" if gt_label == "authentic" else rle_encode([gt_mask.astype(np.uint8)]),
                    shape=shape_str,
                ))

    solution_df = pd.DataFrame(solution_rows)
    case_ids = [p["case_id"] for p in predictions]
    N = len(predictions)
    print(f"[INFO] Cached predictions: {N}")

    # ----- grid ranges (edit as you like)
    area_range = [i * 100 for i in range(1, 11)]  # 100..1000
    mean_range = [round(x, 2) for x in np.arange(0.20, 0.90, 0.05)]
    cc_range = [0, 50, 100, 200, 400, 800, 850, 900, 950, 1000]

    # hysteresis ranges (reasonable small set)
    high_range = [0.65, 0.70, 0.75, 0.80]
    low_range  = [0.35, 0.40, 0.45, 0.50, 0.55]
    hyst_pairs = [(lo, hi) for hi in high_range for lo in low_range if lo < hi]

    combos_2d = list(itertools.product(area_range, mean_range))

    # ----- output dirs
    outdir.mkdir(parents=True, exist_ok=True)
    analysis_dir = outdir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    collage_dir = outdir / "collage"
    if SAVE_COLLAGES:
        collage_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # PRECOMPUTE
    # ============================================================
    print("[INFO] Precomputing CC variants for hysteresis pairs...")

    # We'll store:
    # area_cc[h, i, k], mean_cc[h, i, k], rle_cc[h][i][k]
    H = len(hyst_pairs)
    K = len(cc_range)

    area_cc = np.zeros((H, N, K), dtype=np.int32)
    mean_cc = np.zeros((H, N, K), dtype=np.float32)
    rle_cc = [[["authentic"] * K for _ in range(N)] for _ in range(H)]

    for h_idx, (lo, hi) in enumerate(tqdm(hyst_pairs, desc="Hysteresis pairs", leave=True)):
        for i, pred in enumerate(predictions):
            prob = pred["prob"]  # (S,S)
            ow, oh = pred["orig_width"], pred["orig_height"]

            base_mask, _meta = finalize_mask_hysteresis(prob, (ow, oh), low=lo, high=hi)

            prob_h, prob_w = prob.shape

            for k, min_cc in enumerate(cc_range):
                m = remove_small_components(base_mask, int(min_cc))
                a = int(m.sum())
                area_cc[h_idx, i, k] = a

                if a > 0:
                    m_small = cv2.resize(m, (prob_w, prob_h), interpolation=cv2.INTER_NEAREST)
                    mean_cc[h_idx, i, k] = float(prob[m_small == 1].mean()) if (m_small == 1).any() else 0.0
                    rle_cc[h_idx][i][k] = rle_encode([m])
                else:
                    mean_cc[h_idx, i, k] = 0.0
                    rle_cc[h_idx][i][k] = "authentic"

    # ============================================================
    # SEARCH
    # ============================================================
    def compute_score_for_params(h_idx: int, k: int, a_thr: int, m_thr: float) -> float:
        a_col = area_cc[h_idx, :, k]
        m_col = mean_cc[h_idx, :, k]
        is_forged = (a_col >= a_thr) & (m_col >= m_thr)

        submission_rows = [
            {"case_id": case_ids[i], "annotation": (rle_cc[h_idx][i][k] if is_forged[i] else "authentic")}
            for i in range(N)
        ]
        submission_df = pd.DataFrame(submission_rows)

        return float(competition_score(solution=solution_df, submission=submission_df, row_id_column_name="case_id"))

    # Two-stage search:
    # - Stage 1: pick best (low,high) using a fixed (area,mean,cc) on a subset
    # - Stage 2: with best (low,high) do full grid over (area,mean,cc)
    best_score = -1.0
    best_params = {"area_thres": None, "mean_thres": None, "min_cc_area": None, "low": None, "high": None}

    if USE_TWO_STAGE:
        # Stage 1 anchors (you can set based on your known good params)
        stage1_area = 100
        stage1_mean = 0.75
        stage1_cc = 800

        # subset indices
        idxs = list(range(N))
        random.shuffle(idxs)
        subN = min(TWO_STAGE_SUBSET, N)
        subset = set(idxs[:subN])

        # evaluate hysteresis pairs on subset (fast-ish; still uses scorer, but fewer rows)
        print(f"[INFO] Two-stage search ON. Stage1 subset={subN}, anchors A={stage1_area}, M={stage1_mean}, CC={stage1_cc}")
        try:
            stage1_k = cc_range.index(stage1_cc)
        except ValueError:
            stage1_k = 0

        best_h_idx = 0
        best_h_score = -1.0

        for h_idx, (lo, hi) in tqdm(list(enumerate(hyst_pairs)), desc="Stage1: tune (low,high)"):
            a_col = area_cc[h_idx, :, stage1_k]
            m_col = mean_cc[h_idx, :, stage1_k]
            is_forged = (a_col >= stage1_area) & (m_col >= stage1_mean)

            submission_rows = []
            solution_rows_sub = []
            for i in range(N):
                if i not in subset:
                    continue
                submission_rows.append({
                    "case_id": case_ids[i],
                    "annotation": (rle_cc[h_idx][i][stage1_k] if is_forged[i] else "authentic")
                })
                solution_rows_sub.append(solution_df.iloc[i].to_dict())

            submission_df = pd.DataFrame(submission_rows)
            solution_df_sub = pd.DataFrame(solution_rows_sub)

            curr = float(competition_score(solution=solution_df_sub, submission=submission_df, row_id_column_name="case_id"))
            if curr > best_h_score:
                best_h_score = curr
                best_h_idx = h_idx

        lo_best, hi_best = hyst_pairs[best_h_idx]
        print(f"[INFO] Stage1 best hysteresis: low={lo_best}, high={hi_best}, score={best_h_score:.6f}")

        # Stage 2: full grid over (area,mean,cc) for fixed hysteresis
        for k, min_cc in enumerate(cc_range):
            a_col = area_cc[best_h_idx, :, k]
            m_col = mean_cc[best_h_idx, :, k]

            pbar = tqdm(combos_2d, desc=f"Stage2 Grid (low={lo_best},high={hi_best},CC={min_cc})", leave=False)
            for a_thr, m_thr in pbar:
                is_forged = (a_col >= a_thr) & (m_col >= m_thr)

                submission_rows = [
                    {"case_id": case_ids[i], "annotation": (rle_cc[best_h_idx][i][k] if is_forged[i] else "authentic")}
                    for i in range(N)
                ]
                submission_df = pd.DataFrame(submission_rows)
                curr_score = float(competition_score(solution=solution_df, submission=submission_df, row_id_column_name="case_id"))

                if curr_score > best_score:
                    best_score = curr_score
                    best_params = {
                        "area_thres": int(a_thr),
                        "mean_thres": float(m_thr),
                        "min_cc_area": int(min_cc),
                        "low": float(lo_best),
                        "high": float(hi_best),
                    }
                    pbar.set_postfix(best=f"{best_score:.6f}", A=a_thr, M=m_thr)

    else:
        print("[INFO] Full 4D search ON (low/high/cc/area/mean). This can be heavy.")
        for h_idx, (lo, hi) in enumerate(hyst_pairs):
            for k, min_cc in enumerate(cc_range):
                a_col = area_cc[h_idx, :, k]
                m_col = mean_cc[h_idx, :, k]
                pbar = tqdm(combos_2d, desc=f"Grid (low={lo},high={hi},CC={min_cc})", leave=False)

                for a_thr, m_thr in pbar:
                    is_forged = (a_col >= a_thr) & (m_col >= m_thr)
                    submission_rows = [
                        {"case_id": case_ids[i], "annotation": (rle_cc[h_idx][i][k] if is_forged[i] else "authentic")}
                        for i in range(N)
                    ]
                    submission_df = pd.DataFrame(submission_rows)
                    curr_score = float(competition_score(solution=solution_df, submission=submission_df, row_id_column_name="case_id"))

                    if curr_score > best_score:
                        best_score = curr_score
                        best_params = {
                            "area_thres": int(a_thr),
                            "mean_thres": float(m_thr),
                            "min_cc_area": int(min_cc),
                            "low": float(lo),
                            "high": float(hi),
                        }
                        pbar.set_postfix(best=f"{best_score:.6f}", A=a_thr, M=m_thr)

    print("\n[RESULT] BEST:", best_score, best_params)

    # ============================================================
    # FINAL SUBMISSION (+ optional collages)
    # ============================================================
    best_cc = int(best_params["min_cc_area"])
    best_lo = float(best_params["low"])
    best_hi = float(best_params["high"])
    best_h_idx = hyst_pairs.index((best_lo, best_hi))
    best_k = cc_range.index(best_cc)
    a_thr = int(best_params["area_thres"])
    m_thr = float(best_params["mean_thres"])

    final_rows = []
    best_area_vals = area_cc[best_h_idx, :, best_k].tolist()
    best_mean_vals = mean_cc[best_h_idx, :, best_k].tolist()

    for i, p in enumerate(tqdm(predictions, desc="Final submission")):
        is_forged = (area_cc[best_h_idx, i, best_k] >= a_thr) and (mean_cc[best_h_idx, i, best_k] >= m_thr)
        annotation = (rle_cc[best_h_idx][i][best_k] if is_forged else "authentic")
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

            # reconstruct final mask for collage
            prob = p["prob"]
            ow, oh = p["orig_width"], p["orig_height"]
            base_mask, _ = finalize_mask_hysteresis(prob, (ow, oh), low=best_lo, high=best_hi)
            final_mask = remove_small_components(base_mask, best_cc) if annotation != "authentic" else np.zeros((oh, ow), dtype=np.uint8)

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

    # ----- plots
    save_hist(best_area_vals, f"Mask Area (low={best_lo},high={best_hi}, CC>={best_cc})", analysis_dir / "mask_area_hist.png")
    save_hist(best_mean_vals, f"Mask Mean-Inside (low={best_lo},high={best_hi}, CC>={best_cc})", analysis_dir / "mask_mean_inside_hist.png")

    # ----- metadata
    run_meta = {
        "best_score": best_score,
        "best_params": best_params,
        "search_mode": "two_stage" if USE_TWO_STAGE else "full_4d",
        "two_stage_subset": int(TWO_STAGE_SUBSET) if USE_TWO_STAGE else None,
        "hysteresis_pairs": [{"low": float(lo), "high": float(hi)} for (lo, hi) in hyst_pairs],
        "cc_range": cc_range,
        "area_range": area_range,
        "mean_range": mean_range,
        "per_case": [
            {
                "case_id": p["case_id"],
                "image_path": p["image_path"],
                "orig_width": p["orig_width"],
                "orig_height": p["orig_height"],
                "gt_label": p["gt_label"],
                "area_best": int(area_cc[best_h_idx, i, best_k]),
                "mean_best": float(mean_cc[best_h_idx, i, best_k]),
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
    parser = argparse.ArgumentParser(description="Val inference + hysteresis+CC precompute + grid search + submission")
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

    outdir = Path(args.outdir + f"_{args.arch}_hyst")
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
    print("SAVE_COLLAGES  :", SAVE_COLLAGES)
    print("USE_TWO_STAGE  :", USE_TWO_STAGE)
    print("Best score     :", best_score)
    print("Best params    :", best_params)
    print("Outdir         :", outdir)
    print("=========================\n")


if __name__ == "__main__":
    main()
