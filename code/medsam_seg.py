"""
MedSAM-based segmentation model with a lightweight decoder head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SamModel


class MedSAMSegModel(nn.Module):
    def __init__(self, model_id: str = "flaviagiammarino/medsam-vit-base", out_classes: int = 1):
        super().__init__()
        self.encoder = SamModel.from_pretrained(model_id)
        self.decode_head = nn.Sequential(
            nn.LazyConv2d(256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, out_classes, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_embeddings = self.encoder.get_image_embeddings(pixel_values=images)
        x = self.decode_head(image_embeddings)
        logits = F.interpolate(x, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return logits
