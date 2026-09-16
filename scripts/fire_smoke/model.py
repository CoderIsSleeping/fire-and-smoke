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
from torchvision.models.detection import FCOS, FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.fcos import FCOSHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from . import CLASS_NAMES
from .backbone import (
    BACKBONE_MEAN,
    BACKBONE_STD,
    DEFAULT_BACKBONE,
    PYRAMID_NAMES,
    PYRAMID_STRIDES,
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
        head: str = "frcnn",
        head_convs: int = 4,
    ) -> None:
        super().__init__()
        if head not in ("frcnn", "fcos"):
            raise ValueError(f"head must be 'frcnn' or 'fcos', got {head!r}")
        self.head_type = head
        trunk = DinoPyramidBackbone(
            model_name=backbone_name,
            pretrained=pretrained_backbone,
            out_channels=fpn_channels,
            unfreeze_last_n=unfreeze_last_n,
            local_weights=backbone_weights,
        )

        if head == "fcos":
            self.detector = self._build_fcos(trunk, image_size, box_score_thresh, box_nms_thresh,
                                             box_detections_per_img, head_convs)
        else:
            self.detector = self._build_frcnn(trunk, image_size, box_score_thresh, box_nms_thresh,
                                              box_detections_per_img, rpn_pre_nms_top_n_test,
                                              rpn_post_nms_top_n_test)

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
            "head": head,
            "head_convs": head_convs,
            "class_names": CLASS_NAMES,
            "scene_classes": list(SCENE_CLASSES),
        }

    @staticmethod
    def _build_fcos(trunk, image_size, score_thresh, nms_thresh, detections_per_img, head_convs):
        """One-stage, anchor-free head.

        Chosen over Faster R-CNN for deployment, not accuracy. Faster R-CNN has a
        data-dependent middle: the RPN picks a variable number of proposals and
        RoIAlign crops features for them before the box head runs. That was
        measured to block the multi-camera path -- the legacy ONNX exporter
        bakes in batch size 1 (batch 4 fails in ONNX Runtime) and torch.export
        fails outright. FCOS puts every learned layer *before* any
        data-dependent step, so the network exports as a fixed-shape, batchable
        graph and decoding plus NMS happen afterwards, where TensorRT's
        EfficientNMS plugin or a few lines of numpy can do them.

        num_classes stays 3 with an unused background channel so label ids
        (1 = smoke, 2 = fire) match the Faster R-CNN model and every existing
        evaluation and video script works unchanged.
        """
        return FCOS(
            trunk,
            num_classes=len(CLASS_NAMES),
            min_size=image_size,
            max_size=image_size,
            image_mean=list(BACKBONE_MEAN),
            image_std=list(BACKBONE_STD),
            # One point per location; the size is the stride, which FCOS uses
            # to assign each ground-truth box to a pyramid level by scale.
            anchor_generator=AnchorGenerator(
                sizes=tuple((s,) for s in PYRAMID_STRIDES),
                aspect_ratios=((1.0,),) * len(PYRAMID_STRIDES),
            ),
            head=FCOSHead(trunk.out_channels, num_anchors=1, num_classes=len(CLASS_NAMES),
                          num_convs=head_convs),
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
            topk_candidates=300,
        )

    @staticmethod
    def _build_frcnn(trunk, image_size, score_thresh, nms_thresh, detections_per_img,
                     pre_nms_top_n_test, post_nms_top_n_test):
        return FasterRCNN(
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
            box_score_thresh=score_thresh,
            box_nms_thresh=nms_thresh,
            box_detections_per_img=detections_per_img,
            # Measured on a 400-image test subset: cutting test-time proposals
            # from 1000 to 300 and detections from 50 to 20 costs 0.001 mAP@0.5
            # and nothing at all in recall at the operating point, while making
            # the head ~1.4x faster. Free speed. Training-time counts are
            # untouched, so this does not change how the model is fit.
            rpn_pre_nms_top_n_test=pre_nms_top_n_test,
            rpn_post_nms_top_n_test=post_nms_top_n_test,
        )

    def forward_raw(self, images: torch.Tensor, glow: torch.Tensor):
        """Fixed-shape network outputs, for export. FCOS head only.

        Takes a (B, 3, S, S) batch in 0..1 with S a multiple of 64 and returns
        the head's raw per-location tensors plus the scene logits -- everything
        learned, nothing data-dependent. Decoding and NMS happen outside the
        graph (see `decode_fcos_raw`), which is what lets TensorRT build a
        batched engine from it.
        """
        if self.head_type != "fcos":
            raise RuntimeError("forward_raw is only defined for the FCOS head; "
                               "the Faster R-CNN head has no fixed-shape export boundary.")
        mean = images.new_tensor(BACKBONE_MEAN)[None, :, None, None]
        std = images.new_tensor(BACKBONE_STD)[None, :, None, None]
        features = list(self.backbone((images - mean) / std).values())
        head = self.detector.head(features)
        scene_logits = self.scene_head(self.backbone.last_pooled, glow)
        return head["cls_logits"], head["bbox_regression"], head["bbox_ctrness"], scene_logits

    @torch.no_grad()
    def decode_raw(self, cls_logits, bbox_regression, bbox_ctrness, image_size: int | None = None):
        """Turn `forward_raw` outputs back into per-image detections.

        Mirrors FCOS.forward's own evaluation path step for step -- split per
        pyramid level, build the per-level anchor points, then the model's own
        postprocess_detections (score threshold, top-k, box decode, NMS). Using
        the model's code rather than a re-implementation is what guarantees an
        exported engine produces the same boxes as the PyTorch model.

        Accepts numpy arrays or tensors, so ONNX Runtime / TensorRT outputs can
        be passed straight in.
        """
        from torchvision.models.detection.image_list import ImageList

        size = image_size or self.config["image_size"]
        as_tensor = lambda a: a if isinstance(a, torch.Tensor) else torch.from_numpy(a)
        cls_logits, bbox_regression, bbox_ctrness = map(as_tensor, (cls_logits, bbox_regression, bbox_ctrness))
        batch = cls_logits.shape[0]

        counts = [(size // stride) ** 2 for stride in PYRAMID_STRIDES]
        if sum(counts) != cls_logits.shape[1]:
            raise ValueError(f"raw outputs have {cls_logits.shape[1]} locations but a {size}px input "
                             f"should give {sum(counts)}; pass the image_size the engine was built for")

        dummy = torch.zeros(batch, 3, size, size)
        image_sizes = [(size, size)] * batch
        features = [torch.zeros(batch, 1, size // s, size // s) for s in PYRAMID_STRIDES]
        anchors = self.detector.anchor_generator(ImageList(dummy, image_sizes), features)

        split = {
            "cls_logits": list(cls_logits.float().split(counts, dim=1)),
            "bbox_regression": list(bbox_regression.float().split(counts, dim=1)),
            "bbox_ctrness": list(bbox_ctrness.float().split(counts, dim=1)),
        }
        split_anchors = [list(a.split(counts)) for a in anchors]
        return self.detector.postprocess_detections(split, split_anchors, image_sizes)

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
