"""
DINOv2 + UNet-style decoder for binary semantic segmentation (forgery mask).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss for binary segmentation.
    logits: [B,1,H,W]
    targets: [B,1,H,W] or [B,H,W]
    """
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.float().flatten(1)
    inter = (probs * targets).sum(dim=1)
    den = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    """
    UNet-style decoder block:
    - upsample x
    - concat with skip
    - convs
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch, k=3, p=1)
        self.conv2 = ConvBNReLU(out_ch, out_ch, k=3, p=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


@dataclass
class DinoV2Config:
    model_id: str = "facebook/dinov2-base"
    out_indices: Tuple[int, int, int, int] = (2, 5, 8, 11)
    patch_size: int = 14


class DinoV2HookBackbone(nn.Module):
    """
    DINOv2 backbone that returns 4 feature maps from intermediate blocks.
    Each feature map is (B, C, H/ps, W/ps).
    """

    def __init__(self, cfg: DinoV2Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.model_id)

        self.hidden_size = int(self.encoder.config.hidden_size)

        self._feats: Dict[int, torch.Tensor] = {}
        self._out_indices = set(cfg.out_indices)

        blocks = self.encoder.encoder.layer
        for i, blk in enumerate(blocks):
            blk.register_forward_hook(self._make_hook(i))

    def _make_hook(self, idx: int):
        def hook(module, inp, out):
            self._feats[idx] = out

        return hook

    @torch.no_grad()
    def freeze(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_last_blocks(self, n_last: int = 4) -> None:
        blocks = self.encoder.encoder.layer
        n = len(blocks)
        for i, blk in enumerate(blocks):
            req = i >= (n - n_last)
            for p in blk.parameters():
                p.requires_grad = req

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._feats = {}
        _ = self.encoder(pixel_values=x, return_dict=True, output_hidden_states=False)

        b, _, h, w = x.shape
        gh = h // self.cfg.patch_size
        gw = w // self.cfg.patch_size

        feats: List[torch.Tensor] = []
        for idx in self.cfg.out_indices:
            t = self._feats[idx]
            t = t[:, 1:, :]
            fmap = t.transpose(1, 2).contiguous().view(b, self.hidden_size, gh, gw)
            feats.append(fmap)

        return feats


class DinoV2UNet(nn.Module):
    """
    UNet-like decoder operating on same-resolution token maps (gh x gw),
    using intermediate depth features as "skip" connections.
    """

    def __init__(
        self,
        dinov2_id: str = "facebook/dinov2-base",
        out_indices: Tuple[int, int, int, int] = (2, 5, 8, 11),
        patch_size: int = 14,
        dec_channels: Tuple[int, int, int, int] = (512, 256, 192, 128),
        out_classes: int = 1,
    ):
        super().__init__()
        cfg = DinoV2Config(model_id=dinov2_id, out_indices=out_indices, patch_size=patch_size)
        self.backbone = DinoV2HookBackbone(cfg)
        c = self.backbone.hidden_size

        self.proj1 = nn.Conv2d(c, dec_channels[3], kernel_size=1)
        self.proj2 = nn.Conv2d(c, dec_channels[2], kernel_size=1)
        self.proj3 = nn.Conv2d(c, dec_channels[1], kernel_size=1)
        self.proj4 = nn.Conv2d(c, dec_channels[0], kernel_size=1)

        self.dec3 = DecoderBlock(in_ch=dec_channels[0], skip_ch=dec_channels[1], out_ch=dec_channels[1])
        self.dec2 = DecoderBlock(in_ch=dec_channels[1], skip_ch=dec_channels[2], out_ch=dec_channels[2])
        self.dec1 = DecoderBlock(in_ch=dec_channels[2], skip_ch=dec_channels[3], out_ch=dec_channels[3])

        self.refine = nn.Sequential(
            ConvBNReLU(dec_channels[3], dec_channels[3], k=3, p=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(dec_channels[3], out_classes, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        f1, f2, f3, f4 = feats

        s1 = self.proj1(f1)
        s2 = self.proj2(f2)
        s3 = self.proj3(f3)
        x = self.proj4(f4)

        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        logits_small = self.refine(x)
        logits = F.interpolate(logits_small, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoV2UNet(
        dinov2_id="facebook/dinov2-base",
        out_indices=(2, 5, 8, 11),
        patch_size=14,
        dec_channels=(512, 256, 192, 128),
        out_classes=1,
    ).to(device)

    x = torch.randn(2, 3, 518, 518, device=device)
    y = model(x)
    print("out:", y.shape)
