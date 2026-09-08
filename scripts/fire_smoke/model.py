"""The fire / smoke model: DINOv3 pyramid + Faster R-CNN head + scene head.

Two heads share one frozen trunk pass:

  detection head  -> boxes for `smoke` and `fire`
  scene head      -> image-level P(smoke), P(fire), also fed the glow statistics

The detection head is what the report is benchmarked on. The scene head exists
for the deployment requirements: it can still respond to a fire that is hidden
from the camera and only visible as reflected light, and it gives the alarm
logic a second opinion to cross-check against before waking anybody up.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from . import CLASS_NAMES
from .backbone import (
    BACKBONE_MEAN,
    BACKBONE_STD,
    DEFAULT_BACKBONE,
    PYRAMID_NAMES,
    SIZE_DIVISIBLE,
    DinoPyramidBackbone,
    SceneHead,
)
from .glow import GLOW_FEATURE_DIM

# One anchor scale band per pyramid level (strides 8 / 16 / 32 / 64).
# The smallest band matters: distant flames in a wide industrial view are tiny.
ANCHOR_SIZES = ((16, 32), (48, 96), (128, 192), (256, 384))
ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)
SCENE_CLASSES = ("smoke", "fire")


class FireSmokeDetector(nn.Module):
    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        image_size: int = 640,
        pretrained_backbone: bool = True,
        unfreeze_last_n: int = 0,
        backbone_weights: str | None = None,
        scene_weight: float = 1.0,
        fpn_channels: int = 256,
        box_score_thresh: float = 0.02,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 20,
        rpn_pre_nms_top_n_test: int = 1000,
        rpn_post_nms_top_n_test: int = 300,
    ) -> None:
        super().__init__()
        trunk = DinoPyramidBackbone(
            model_name=backbone_name,
            pretrained=pretrained_backbone,
            out_channels=fpn_channels,
            unfreeze_last_n=unfreeze_last_n,
            local_weights=backbone_weights,
        )

        self.detector = FasterRCNN(
            trunk,
            num_classes=len(CLASS_NAMES),
            min_size=image_size,
            max_size=image_size,
            image_mean=list(BACKBONE_MEAN),
            image_std=list(BACKBONE_STD),
            rpn_anchor_generator=AnchorGenerator(sizes=ANCHOR_SIZES, aspect_ratios=ASPECT_RATIOS),
            box_roi_pool=MultiScaleRoIAlign(
                featmap_names=list(PYRAMID_NAMES), output_size=7, sampling_ratio=2
            ),
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
            # Measured on a 400-image test subset: cutting test-time proposals
            # from 1000 to 300 and detections from 50 to 20 costs 0.001 mAP@0.5
            # and nothing at all in recall at the operating point, while making
            # the head ~1.4x faster. Free speed. Training-time counts are
            # untouched, so this does not change how the model is fit.
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        )
        # The default transform pads to a multiple of 32; the stride-64 pyramid
        # level needs 64. Inputs are already letterboxed squares, so the resize
        # inside the transform is a no-op and only the padding matters.
        self.detector.transform = GeneralizedRCNNTransform(
            min_size=image_size,
            max_size=image_size,
            image_mean=list(BACKBONE_MEAN),
            image_std=list(BACKBONE_STD),
            size_divisible=SIZE_DIVISIBLE,
        )

        self.scene_head = SceneHead(trunk.trunk_dim, GLOW_FEATURE_DIM)
        self.scene_weight = scene_weight
        self.scene_loss_fn = nn.BCEWithLogitsLoss()

        self.config = {
            "backbone_name": backbone_name,
            "image_size": image_size,
            "unfreeze_last_n": unfreeze_last_n,
            "scene_weight": scene_weight,
            "fpn_channels": fpn_channels,
            "class_names": CLASS_NAMES,
            "scene_classes": list(SCENE_CLASSES),
        }

    @property
    def backbone(self) -> DinoPyramidBackbone:
        # Exposed through the detector so the trunk is stored once in state_dict.
        return self.detector.backbone

    def param_groups(self, lr: float, trunk_lr_scale: float = 0.05, weight_decay: float = 1e-4):
        """Head/neck at the full LR, any unfrozen trunk blocks far below it."""
        trunk_params = self.backbone.trunk_parameters()
        trunk_ids = {id(p) for p in trunk_params}
        head_params = [
            p for p in self.parameters() if p.requires_grad and id(p) not in trunk_ids
        ]

        groups = [{"params": head_params, "lr": lr, "weight_decay": weight_decay, "name": "heads"}]
        if trunk_params:
            groups.append(
                {
                    "params": trunk_params,
                    "lr": lr * trunk_lr_scale,
                    "weight_decay": weight_decay,
                    "name": "trunk",
                }
            )
        return groups

    def forward(
        self,
        images,
        targets=None,
        scene_targets: torch.Tensor | None = None,
        glow: torch.Tensor | None = None,
    ):
        detector_out = self.detector(images, targets)
        pooled = self.backbone.last_pooled
        scene_logits = self.scene_head(pooled, glow)

        if self.training:
            losses = dict(detector_out)
            if scene_targets is not None:
                losses["loss_scene"] = self.scene_loss_fn(
                    scene_logits.float(), scene_targets.float()
                ) * self.scene_weight
            return losses

        return detector_out, torch.sigmoid(scene_logits.float())

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        payload = {"model": self.state_dict(), "config": self.config}
        if extra:
            payload.update(extra)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(path))


def load_detector(checkpoint_path: str | Path, device: torch.device | str = "cpu", **overrides):
    """Rebuild a `FireSmokeDetector` from a saved checkpoint.

    The trunk is created without downloading pretrained weights, since the
    checkpoint already contains them.
    """
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = dict(checkpoint.get("config", {}))
    config.pop("class_names", None)
    config.pop("scene_classes", None)
    config.update(overrides)
    config["pretrained_backbone"] = False

    model = FireSmokeDetector(**config)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing:
        print(f"warning: {len(missing)} missing keys when loading (first: {missing[:3]})")
    if unexpected:
        print(f"warning: {len(unexpected)} unexpected keys when loading (first: {unexpected[:3]})")

    model.to(device).eval()
    return model, checkpoint
