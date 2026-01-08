import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(dim=1)
    den = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (den + eps)
    return 1 - dice.mean()


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PSPModule(nn.Module):
    """Pyramid pooling on top-level feature map."""

    def __init__(self, in_ch, out_ch, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList()
        for ps in pool_sizes:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(ps),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
        self.bottleneck = ConvBNReLU(in_ch + len(pool_sizes) * out_ch, out_ch, k=3, p=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        priors = [x]
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            priors.append(y)
        x = torch.cat(priors, dim=1)
        return self.bottleneck(x)


class UPerHead(nn.Module):
    """
    UPerNet head: PSP on last feature + FPN fusion.
    feats: list of feature maps [C2, C3, C4, C5] (low->high level)
    """

    def __init__(self, in_channels_list, channels=256, num_classes=1):
        super().__init__()
        assert len(in_channels_list) == 4, "Expect 4 feature levels"
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, channels, 1) for c in in_channels_list])
        self.fpn_convs = nn.ModuleList([ConvBNReLU(channels, channels) for _ in in_channels_list])

        self.psp = PSPModule(channels, channels)

        self.fuse = ConvBNReLU(channels * 4, channels)
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, feats):
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, feats)]

        laterals[-1] = self.psp(laterals[-1])

        for i in range(2, -1, -1):
            up = F.interpolate(laterals[i + 1], size=laterals[i].shape[-2:], mode="bilinear", align_corners=False)
            laterals[i] = laterals[i] + up

        outs = [fpn(lat) for fpn, lat in zip(self.fpn_convs, laterals)]

        target_size = outs[0].shape[-2:]
        outs = [outs[0]] + [F.interpolate(o, size=target_size, mode="bilinear", align_corners=False) for o in outs[1:]]
        x = self.fuse(torch.cat(outs, dim=1))
        return self.classifier(x)


class DinoV2Backbone(nn.Module):
    """
    Extract intermediate block tokens from DINOv2 and reshape to feature maps.
    """

    def __init__(self, model_id: str, out_indices=(2, 5, 8, 11)):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        self.out_indices = set(out_indices)

        self._features = {}
        for i, blk in enumerate(self.encoder.encoder.layer):
            blk.register_forward_hook(self._make_hook(i))

        self.hidden_size = self.encoder.config.hidden_size
        self.patch_size = 14

    def _make_hook(self, idx):
        def hook(module, inp, out):
            self._features[idx] = out

        return hook

    def forward(self, x):
        self._features = {}
        _ = self.encoder(pixel_values=x, output_hidden_states=False, return_dict=True)
        feats = []
        b, _, h, w = x.shape

        gh = h // self.patch_size
        gw = w // self.patch_size

        for idx in sorted(self.out_indices):
            t = self._features[idx]
            t = t[:, 1:, :]
            fmap = t.transpose(1, 2).contiguous().view(b, self.hidden_size, gh, gw)
            feats.append(fmap)

        return feats


class DinoV2_UPerNet(nn.Module):
    def __init__(self, dinov2_id: str, num_classes=1, uper_channels=256, out_indices=(2, 5, 8, 11)):
        super().__init__()
        self.backbone = DinoV2Backbone(dinov2_id, out_indices=out_indices)
        in_list = [self.backbone.hidden_size] * 4
        self.decode_head = UPerHead(in_list, channels=uper_channels, num_classes=num_classes)

    def forward(self, images):
        feats = self.backbone(images)
        logits_small = self.decode_head(feats)
        logits = F.interpolate(logits_small, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return logits
