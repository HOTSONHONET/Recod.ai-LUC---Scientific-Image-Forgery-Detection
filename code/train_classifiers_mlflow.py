#!/usr/bin/env python3
"""
train_classifiers_mlflow.py

Train sklearn-based classifiers (optional xgboost/lightgbm) on DINOv2 embeddings.
Logs everything to MLflow: params, metrics, artifacts.

CSV requirements:
- train CSV: image_path, label (0=authentic, 1=forged). case_id optional
- val   CSV: image_path, label (0/1). case_id optional

Outputs:
analysis/classifiers/<run_name>/
  cache/
  models/<model_name>/*
  model_ranking.json
  meta.json
  cli_args.json

MLflow:
- logs params (args)
- logs metrics per model
- logs full outdir as artifacts
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
import mlflow

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


# -------------------------
# Dataset
# -------------------------
class ImageClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = str(row["image_path"])
        label = int(row["label"])
        case_id = str(row["case_id"]) if "case_id" in row else str(Path(image_path).stem)

        with open(image_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        return img, label, case_id, image_path


def collate_fn(batch):
    imgs, labels, case_ids, paths = zip(*batch)
    return list(imgs), np.array(labels, dtype=np.int64), list(case_ids), list(paths)


# -------------------------
# Embedding extractor
# -------------------------
@torch.no_grad()
def extract_embeddings_hf(
    df: pd.DataFrame,
    model_id: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_fast: bool,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Returns:
      X: (N, D) float32 embeddings (CLS token)
      y: (N,) int64
      case_ids: list[str]
      paths: list[str]
    """
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=use_fast)
    model = AutoModel.from_pretrained(model_id)
    model.eval().to(device)

    ds = ImageClsDataset(df)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    feats = []
    ys = []
    case_ids_all = []
    paths_all = []

    for imgs, labels, case_ids, paths in tqdm(loader, desc=f"Embeddings: {Path(model_id).name}"):
        # Explicitly control resize to match your segmentation style (square resize, no crop)
        inputs = processor(
            images=imgs,
            return_tensors="pt",
            do_resize=True,
            size={"height": img_size, "width": img_size},
            do_center_crop=False,
            do_normalize=True,
            do_rescale=True,
        )
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :]  # [B, D]
        feats.append(cls.float().cpu().numpy())

        ys.append(labels)
        case_ids_all.extend(case_ids)
        paths_all.extend(paths)

    X = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    return X, y, case_ids_all, paths_all


# -------------------------
# Plots / Metrics helpers
# -------------------------
def save_confusion_matrix(cm: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.xticks([0, 1], ["auth", "forged"])
    plt.yticks([0, 1], ["auth", "forged"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_roc_pr_curves(y_true, y_prob, out_roc: Path, out_pr: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()
    plt.savefig(out_roc, dpi=200, bbox_inches="tight")
    plt.close()

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.title("PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(out_pr, dpi=200, bbox_inches="tight")
    plt.close()


def best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.linspace(0.0, 1.0, 501)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int64)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


def evaluate_and_save(
    name: str,
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    out_dir: Path,
    mlflow_prefix: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    clf.fit(X_train, y_train)

    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_val)[:, 1]
    elif hasattr(clf, "decision_function"):
        s = clf.decision_function(X_val)
        y_prob = 1.0 / (1.0 + np.exp(-np.clip(s, -20, 20)))
    else:
        y_prob = clf.predict(X_val).astype(np.float32)

    y_pred_05 = (y_prob >= 0.5).astype(np.int64)

    acc = float(accuracy_score(y_val, y_pred_05))
    p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred_05, average="binary", zero_division=0)

    roc_auc = float(roc_auc_score(y_val, y_prob)) if len(np.unique(y_val)) == 2 else float("nan")
    pr_auc = float(average_precision_score(y_val, y_prob)) if len(np.unique(y_val)) == 2 else float("nan")

    t_best = best_threshold_by_f1(y_val, y_prob)
    y_pred_best = (y_prob >= t_best).astype(np.int64)
    p2, r2, f12, _ = precision_recall_fscore_support(y_val, y_pred_best, average="binary", zero_division=0)
    acc2 = float(accuracy_score(y_val, y_pred_best))

    rep = classification_report(y_val, y_pred_05, target_names=["authentic", "forged"], zero_division=0)
    (out_dir / "classification_report.txt").write_text(rep)

    cm05 = confusion_matrix(y_val, y_pred_05, labels=[0, 1])
    cmb = confusion_matrix(y_val, y_pred_best, labels=[0, 1])
    save_confusion_matrix(cm05, out_dir / "confusion_matrix_thr0.50.png", f"{name} (thr=0.50)")
    save_confusion_matrix(cmb, out_dir / "confusion_matrix_bestF1.png", f"{name} (thr={t_best:.3f})")

    save_roc_pr_curves(y_val, y_prob, out_dir / "roc_curve.png", out_dir / "pr_curve.png")

    metrics = {
        "threshold_default": 0.5,
        "acc@0.5": acc,
        "precision@0.5": float(p),
        "recall@0.5": float(r),
        "f1@0.5": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold_best_f1": float(t_best),
        "acc@best": acc2,
        "precision@best": float(p2),
        "recall@best": float(r2),
        "f1@best": float(f12),
    }
    (out_dir / "metrics.json").write_text(json.dumps({"name": name, **metrics}, indent=2))

    # Save model
    joblib.dump(clf, out_dir / "model.pkl")

    # MLflow metrics (namespaced per model)
    # Example metric name: "models/logreg/f1@best"
    for k, v in metrics.items():
        if isinstance(v, (float, int)) and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            mlflow.log_metric(f"{mlflow_prefix}/{name}/{k.replace("@", "_at_")}", float(v))

    return {"name": name, **metrics}


def make_models(seed: int):
    models = {}

    models["logreg"] = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)),
    ])

    base_svc = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svc", LinearSVC(class_weight="balanced", random_state=seed)),
    ])
    models["linear_svm_calibrated"] = CalibratedClassifierCV(base_svc, method="sigmoid", cv=3)

    models["random_forest"] = RandomForestClassifier(
        n_estimators=600, max_depth=None, n_jobs=-1, class_weight="balanced_subsample", random_state=seed
    )
    models["extra_trees"] = ExtraTreesClassifier(
        n_estimators=800, max_depth=None, n_jobs=-1, class_weight="balanced", random_state=seed
    )
    models["decision_tree"] = DecisionTreeClassifier(class_weight="balanced", random_state=seed)
    models["grad_boosting"] = GradientBoostingClassifier(random_state=seed)

    # Optional: XGBoost (requires pip install xgboost)
    try:
        import xgboost as xgb  # type: ignore
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=1200,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=os.cpu_count() or 8,
            eval_metric="logloss",
        )
    except Exception:
        pass

    # Optional: LightGBM (requires pip install lightgbm)
    try:
        import lightgbm as lgb  # type: ignore
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=3000,
            learning_rate=0.02,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=os.cpu_count() or 8,
        )
    except Exception:
        pass

    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=str, required=True)
    ap.add_argument("--val-csv", type=str, required=True)
    ap.add_argument("--dinov2-id", type=str, default="facebook/dinov2-base")
    ap.add_argument("--img-size", type=int, default=532, help="Square resize used before embedding extraction")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--use-fast", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out-root", type=str, default="exps/classifiers")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--cache-dir", type=str, default="analysis/embeddings_data")

    # MLflow
    ap.add_argument("--mlflow-uri", type=str, default=None, help="Optional tracking URI (e.g., http://localhost:5000)")
    ap.add_argument("--experiment-name", type=str, default="forgery_classifiers", help="MLflow experiment name")

    args = ap.parse_args()
    print("[INFO] Collecting train and val dfs")
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    for df, name in [(train_df, "train"), (val_df, "val")]:
        if "image_path" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{name} CSV must contain columns: image_path, label")
        df["label"] = df["label"].astype(int)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name or timestamp
    outdir = Path(args.out_root) / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Save args locally too
    print("[INFO] Saving CLI args")
    (outdir / "cli_args.json").write_text(json.dumps(vars(args), indent=2))

    # MLflow setup
    print("[INFO] Setting MLflow")
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    
    experiment_name = f"{args.experiment_name}_{timestamp}"
    print("[INFO] Setting experiment name")
    mlflow.set_experiment(experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    emb_tr_path = cache_dir / "embeddings_train.npy"
    y_tr_path = cache_dir / "labels_train.npy"
    emb_va_path = cache_dir / "embeddings_val.npy"
    y_va_path = cache_dir / "labels_val.npy"

    with mlflow.start_run(run_name=run_name):
        # 1) log args as params (require scalars/strings)
        # MLflow params expect simple types; convert anything complex to string
        safe_params = {}
        for k, v in vars(args).items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                safe_params[k] = v
            else:
                safe_params[k] = json.dumps(v)
        mlflow.log_params(safe_params)

        # 2) log dataset stats
        mlflow.log_metric("n_train", float(len(train_df)))
        mlflow.log_metric("n_val", float(len(val_df)))
        mlflow.log_metric("train_pos", float(train_df["label"].sum()))
        mlflow.log_metric("train_neg", float((train_df["label"] == 0).sum()))
        mlflow.log_metric("val_pos", float(val_df["label"].sum()))
        mlflow.log_metric("val_neg", float((val_df["label"] == 0).sum()))
        mlflow.log_param("device", str(device))

        # 3) embeddings
        if emb_tr_path.exists() and y_tr_path.exists() and emb_va_path.exists() and y_va_path.exists():
            print("[INFO] Loading cached embeddings...")
            X_train = np.load(emb_tr_path)
            y_train = np.load(y_tr_path)
            X_val = np.load(emb_va_path)
            y_val = np.load(y_va_path)
        else:
            print("[INFO] Extracting train embeddings...")
            X_train, y_train, _, _ = extract_embeddings_hf(
                train_df, args.dinov2_id, device, args.batch_size, args.num_workers, args.use_fast, args.img_size
            )
            print("[INFO] Extracting val embeddings...")
            X_val, y_val, _, _ = extract_embeddings_hf(
                val_df, args.dinov2_id, device, args.batch_size, args.num_workers, args.use_fast, args.img_size
            )
            np.save(emb_tr_path, X_train)
            np.save(y_tr_path, y_train)
            np.save(emb_va_path, X_val)
            np.save(y_va_path, y_val)

        print("[INFO] Shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)
        mlflow.log_param("embedding_dim", int(X_train.shape[1]))

        # 4) train + evaluate models
        models = make_models(args.seed)
        results = []
        models_dir = outdir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        for name, clf in models.items():
            print(f"\n[INFO] Training: {name}")
            mdir = models_dir / name
            metrics = evaluate_and_save(
                name=name,
                clf=clf,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                out_dir=mdir,
                mlflow_prefix="models",
            )
            pprint(metrics)
            results.append(metrics)

        # 5) ranking + meta
        results_sorted = sorted(results, key=lambda d: d.get("f1@best", -1), reverse=True)
        (outdir / "model_ranking.json").write_text(json.dumps(results_sorted, indent=2))

        meta = {
            "dinov2_id": args.dinov2_id,
            "device": str(device),
            "train_csv": args.train_csv,
            "val_csv": args.val_csv,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "pos_train": int(y_train.sum()),
            "neg_train": int((y_train == 0).sum()),
            "pos_val": int(y_val.sum()),
            "neg_val": int((y_val == 0).sum()),
            "run_name": run_name,
            "created_at": datetime.now().isoformat(),
        }
        (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

        # Log “best model” summary
        if results_sorted:
            best = results_sorted[0]
            mlflow.log_param("best_model", best["name"])
            mlflow.log_metric("best_model_f1_at_best", float(best["f1@best"]))
            mlflow.log_metric("best_model_pr_auc", float(best["pr_auc"]) if not np.isnan(best["pr_auc"]) else -1.0)

        # 6) log ALL artifacts produced
        mlflow.log_artifacts(str(outdir), artifact_path="classifiers_run")

        print("\n======== DONE ========")
        print("Outputs:", outdir)
        print("Top by f1@best:", results_sorted[0]["name"] if results_sorted else "N/A")
        print("======================\n")


if __name__ == "__main__":
    main()
