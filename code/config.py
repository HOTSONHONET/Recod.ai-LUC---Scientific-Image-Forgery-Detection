"""
Central configuration and reproducibility utilities.
"""

import os
import random
from typing import Any, Dict

import numpy as np
import torch

from model import DinoSegModel


class Config:
    # Data
    IMG_SIZE = 448
    N_SPLITS = 5
    TRAIN_CSV = "analysis/splits"
    IMG_DIR = "."
    SEED = 42

    # Training
    EPOCHS = 10
    BATCH = 4
    LR = 1e-4
    NUM_WORKERS = 8
    VAL_WORKERS = 8
    OUTPUT_DIR = "exps"
    EXPERIMENT_NAME = "forgery-dino"

    # Model
    MODEL = "dino_seg"
    TIMM_MODEL = "vit_base_patch16_224.dino"
    IN_CHANNELS = 3
    DROPOUT = 0.1
    PRETRAINED = True

    MODEL_MAP = {
        "dino_seg": DinoSegModel,
    }

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        cfg = {
            name: value
            for name, value in cls.__dict__.items()
            if not name.startswith("__") and not callable(value) and name.upper() == name
        }
        model_map_classnames = {name: klass.__name__ for name, klass in cfg["MODEL_MAP"].items()}
        cfg["MODEL_MAP"] = model_map_classnames
        cfg["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
        return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
