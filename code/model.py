"""
Model definitions for scientific image forgery segmentation.
"""

from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoSegModel(nn.Module):
    """
    Simple ViT-DINO backbone with a lightweight upsampling head for binary segmentation.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224.dino",
        pretrained: bool = True,
        num_classes: int = 1,
        img_size: int | None = None,
    ):
        super().__init__()
        self.img_size = img_size
        create_kwargs = {}
        if img_size is not None:
            create_kwargs["img_size"] = img_size
        self.backbone = timm.create_model(model_name, pretrained=pretrained, **create_kwargs)
        # Remove classifier if present.
        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(0)
        self.num_classes = num_classes

        patch_size = getattr(getattr(self.backbone, "patch_embed", None), "patch_size", (16, 16))
        self.patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else int(patch_size)
        embed_dim = getattr(self.backbone, "num_features", 768)

        self.decode_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1),
        )

    def _extract_tokens(self, feats: Any) -> torch.Tensor:
        """
        Normalize different timm forward_features outputs to a token tensor (B, N, C).
        """
        x = feats
        if isinstance(x, dict):
            for key in ("x", "last_hidden_state", "tokens"):
                if key in x:
                    x = x[key]
                    break
            else:
                x = next(iter(x.values()))
        if isinstance(x, (list, tuple)):
            x = x[-1]
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, _, H, W = images.shape
        feats = self.backbone.forward_features(images)
        tokens = self._extract_tokens(feats)

        # Drop class token if present.
        if tokens.shape[1] == (H // self.patch_size) * (W // self.patch_size) + 1:
            tokens = tokens[:, 1:, :]

        h_tokens = H // self.patch_size
        w_tokens = W // self.patch_size
        tokens = tokens.view(B, h_tokens, w_tokens, -1).permute(0, 3, 1, 2).contiguous()
        logits = self.decode_head(tokens)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits
