"""DINOv3 trunk + simple feature pyramid.

The trunk is a plain ViT, so every block already has a global receptive field.
Following ViTDet, we do not need a hierarchical CNN: we take four intermediate
blocks and resample them to strides 8 / 16 / 32 / 64 to feed an FPN-style
detection head.

The trunk is frozen by default. That is the whole point of using DINOv3 here:
the self-supervised features are strong and generic, and freezing them
    * keeps training cheap enough for a Kaggle GPU session,
    * avoids destroying the pretrained representation on a 14k-image dataset,
    * keeps the model stable across the lighting domains we care about.
`--unfreeze-last-n` can later release the last N blocks at a much lower LR.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

DEFAULT_BACKBONE = "vit_small_patch16_dinov3.lvd1689m"
# DINOv3 uses ImageNet normalisation.
BACKBONE_MEAN = (0.485, 0.456, 0.406)
BACKBONE_STD = (0.229, 0.224, 0.225)
# Strides produced by the pyramid, from finest to coarsest.
PYRAMID_STRIDES = (8, 16, 32, 64)
PYRAMID_NAMES = ("p3", "p4", "p5", "p6")
# Every input must be divisible by the coarsest stride.
SIZE_DIVISIBLE = 64


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of an NCHW tensor, as used by ViTDet.

    Deliberately not BatchNorm: with no running statistics anywhere in the
    model we can compute a validation loss in train() mode without polluting
    anything, and small batch sizes stay safe.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


def default_out_indices(depth: int) -> tuple[int, int, int, int]:
    """Pick four reasonably deep blocks, scaled to the trunk depth."""
    return (
        max(0, depth // 2 - 1),
        max(0, (3 * depth) // 4 - 1),
        max(0, (7 * depth) // 8 - 1),
        depth - 1,
    )


def create_trunk(model_name: str, pretrained: bool, local_weights: str | None = None):
    """Build the timm ViT trunk, optionally loading weights from a local file.

    `local_weights` exists for Kaggle sessions with the internet switch off:
    download the checkpoint once, attach it as a Kaggle dataset, and point
    `--backbone-weights` at the file.
    """
    import timm

    load_from_hub = pretrained and local_weights is None
    trunk = timm.create_model(model_name, pretrained=load_from_hub, num_classes=0)

    if local_weights:
        path = Path(local_weights)
        if not path.exists():
            raise FileNotFoundError(f"--backbone-weights not found: {path}")
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(path))
        else:
            state = torch.load(str(path), map_location="cpu")
            for key in ("model", "state_dict", "model_state_dict"):
                if isinstance(state, dict) and key in state:
                    state = state[key]
                    break
        missing, unexpected = trunk.load_state_dict(state, strict=False)
        print(f"backbone weights loaded from {path}")
        if missing:
            print(f"  missing keys: {len(missing)} (first: {missing[:3]})")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")

    return trunk


class DinoPyramidBackbone(nn.Module):
    """Frozen DINOv3 trunk wrapped in a simple feature pyramid.

    Exposes the interface that torchvision FasterRCNN expects: a forward that
    returns an OrderedDict of feature maps, plus an `out_channels` attribute.
    It also stashes a pooled trunk descriptor in `last_pooled` so the
    scene-level classifier can reuse the same trunk pass.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        out_channels: int = 256,
        unfreeze_last_n: int = 0,
        local_weights: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.trunk = create_trunk(model_name, pretrained, local_weights)

        patch = self.trunk.patch_embed.patch_size
        patch_size = patch[0] if isinstance(patch, (tuple, list)) else patch
        if patch_size != 16:
            raise ValueError(
                f"{model_name} uses patch size {patch_size}; this pyramid assumes 16 "
                "so the strides land on 8/16/32/64. Pick a patch-16 DINOv3 model."
            )

        depth = len(self.trunk.blocks)
        self.out_indices = default_out_indices(depth)
        self.trunk_dim = self.trunk.embed_dim
        self.out_channels = out_channels
        self.unfreeze_last_n = unfreeze_last_n
        self.frozen = unfreeze_last_n <= 0

        self._freeze_trunk(unfreeze_last_n)

        c = self.trunk_dim
        # Finest level: upsample the stride-16 tokens back to stride 8.
        self.to_p3 = nn.Sequential(
            nn.Conv2d(c, out_channels, 1),
            nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2),
        )
        self.to_p4 = nn.Conv2d(c, out_channels, 1)
        self.to_p5 = nn.Sequential(nn.Conv2d(c, out_channels, 1), nn.MaxPool2d(2, 2))
        self.to_p6 = nn.Sequential(nn.Conv2d(c, out_channels, 1), nn.MaxPool2d(4, 4))
        self.resamplers = nn.ModuleList([self.to_p3, self.to_p4, self.to_p5, self.to_p6])

        self.smooth = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                LayerNorm2d(out_channels),
            )
            for _ in PYRAMID_NAMES
        )

        self.last_pooled: torch.Tensor | None = None
        self._init_neck()

    def _init_neck(self) -> None:
        for module in list(self.resamplers) + list(self.smooth):
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _freeze_trunk(self, unfreeze_last_n: int) -> None:
        for param in self.trunk.parameters():
            param.requires_grad_(False)
        if unfreeze_last_n > 0:
            for block in self.trunk.blocks[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad_(True)
            if hasattr(self.trunk, "norm"):
                for param in self.trunk.norm.parameters():
                    param.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # A fully frozen trunk always stays in eval mode (no dropout / droppath).
        if self.frozen:
            self.trunk.eval()
        return self

    def trunk_parameters(self):
        return [p for p in self.trunk.parameters() if p.requires_grad]

    def neck_parameters(self):
        params = []
        for module in list(self.resamplers) + list(self.smooth):
            params.extend(module.parameters())
        return params

    def _run_trunk(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.trunk.forward_intermediates(
            x,
            indices=list(self.out_indices),
            norm=False,
            output_fmt="NCHW",
            intermediates_only=True,
        )

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        if self.frozen:
            with torch.no_grad():
                feats = [f.detach() for f in self._run_trunk(x)]
        else:
            feats = self._run_trunk(x)

        deepest = feats[-1]
        pooled = torch.cat([deepest.mean(dim=(2, 3)), deepest.amax(dim=(2, 3))], dim=1)
        # Stashed so the scene head can reuse this pass instead of re-running
        # the trunk. FasterRCNN calls the backbone exactly once per forward.
        self.last_pooled = pooled

        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for name, feat, resample, smooth in zip(PYRAMID_NAMES, feats, self.resamplers, self.smooth):
            out[name] = smooth(resample(feat))
        return out


class SceneHead(nn.Module):
    """Image-level fire / smoke classifier on top of the pooled DINOv3 tokens.

    Why a second head at all:
      * an occluded flame often has no box-shaped evidence, only a warm glow
        spread over surrounding surfaces -- a whole-image classifier can still
        fire on that, and the hand-crafted glow statistics give it the cue,
      * two semi-independent signals let the deployment layer demand agreement
        before raising an alarm, which is the cheapest false-positive filter
        available to us.
    """

    def __init__(self, trunk_dim: int, glow_dim: int, hidden: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.glow_dim = glow_dim
        in_dim = trunk_dim * 2 + glow_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),  # logits for [smoke, fire]
        )

    def forward(self, pooled: torch.Tensor, glow: torch.Tensor | None) -> torch.Tensor:
        if glow is None:
            glow = pooled.new_zeros((pooled.shape[0], self.glow_dim))
        return self.net(torch.cat([pooled, glow.to(pooled.dtype)], dim=1))
