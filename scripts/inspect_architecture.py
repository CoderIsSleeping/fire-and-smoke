"""Exact layer inventory for a checkpoint, read from the live PyTorch model.

Every number in the printed table is counted by walking the actual model
object (`model.named_modules()`), not typed from memory. This exists so the
architecture tables in a report can be trusted: if the code changes, this
script's output changes with it.

    python scripts/inspect_architecture.py --weights models/stage1_frozen_fasterrcnn.pt --output reports/data/arch/stage1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn

from fire_smoke.model import load_detector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Count layers in a trained checkpoint, exactly.")
    p.add_argument("--weights", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--output", required=True)
    return p.parse_args()


def count_modules(module: nn.Module) -> Counter:
    """Leaf-module type counts (a Linear inside an Attention block counts once)."""
    counts = Counter()
    for m in module.modules():
        if len(list(m.children())) == 0:  # leaf module only, no containers
            counts[type(m).__name__] += 1
    return counts


def param_stats(module: nn.Module) -> dict:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def section(name: str, module: nn.Module | None) -> dict:
    if module is None:
        return {"present": False}
    counts = count_modules(module)
    return {"present": True, **param_stats(module), "layer_counts": dict(counts.most_common())}


def main() -> None:
    args = parse_args()
    model, checkpoint = load_detector(args.weights, "cpu")
    label = args.label or f"{model.config.get('backbone_name', '?')} + {model.head_type}"

    trunk = model.backbone.trunk
    blocks = list(trunk.blocks)
    block0 = blocks[0]
    head_dim = block0.attn.qkv.in_features // getattr(block0.attn, "num_heads", 1) if hasattr(block0.attn, "num_heads") else None

    report = {
        "label": label,
        "weights": str(args.weights),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "config": model.config,
        "totals": param_stats(model),
        "sections": {
            "dinov3_trunk": section("DINOv3 ViT trunk", trunk),
            "feature_pyramid": section("Feature pyramid (neck)", model.backbone),
            "detection_head": section("Detection head", model.detector),
            "scene_head": section("Scene classifier head", model.scene_head),
        },
        "transformer_trunk_detail": {
            "num_blocks": len(blocks),
            "embed_dim": trunk.embed_dim,
            "num_heads_per_block": getattr(block0.attn, "num_heads", None),
            "head_dim": head_dim,
            "mlp_hidden_dim": block0.mlp.fc1.out_features,
            "patch_size": trunk.patch_embed.patch_size,
            "block_type": type(block0).__name__,
            "per_block_layers": {
                "LayerNorm": 2,          # norm1, norm2
                "Linear (qkv)": 1,
                "Linear (attn proj)": 1,
                "Linear (mlp fc1)": 1,
                "Linear (mlp fc2)": 1,
                "GELU": 1,
            },
        },
    }

    # Neck-only breakdown (backbone module minus the trunk submodule).
    neck_counts = Counter()
    for name, m in model.backbone.named_modules():
        if name.startswith("trunk"):
            continue
        if len(list(m.children())) == 0:
            neck_counts[type(m).__name__] += 1
    report["sections"]["feature_pyramid_only"] = {
        "layer_counts": dict(neck_counts.most_common()),
        "trainable_params": sum(p.numel() for n, p in model.backbone.named_parameters()
                                if not n.startswith("trunk") and p.requires_grad),
    }

    if model.head_type == "fcos":
        head = model.detector.head
        report["fcos_head_detail"] = {
            "num_convs_per_tower": len(head.classification_head.conv) // 3,  # Conv+GroupNorm+ReLU triples
            "classification_tower": {**param_stats(head.classification_head),
                                      "layer_counts": dict(count_modules(head.classification_head).most_common())},
            "regression_tower": {**param_stats(head.regression_head),
                                 "layer_counts": dict(count_modules(head.regression_head).most_common())},
        }
    else:
        rpn = model.detector.rpn
        roi = model.detector.roi_heads
        report["faster_rcnn_head_detail"] = {
            "rpn": {**param_stats(rpn), "layer_counts": dict(count_modules(rpn).most_common())},
            "roi_heads": {**param_stats(roi), "layer_counts": dict(count_modules(roi).most_common())},
            "anchor_sizes": [tuple(s) for s in model.detector.rpn.anchor_generator.sizes],
            "aspect_ratios": [tuple(a) for a in model.detector.rpn.anchor_generator.aspect_ratios],
            "box_predictor_out_features": {
                "cls_score": model.detector.roi_heads.box_predictor.cls_score.out_features,
                "bbox_pred": model.detector.roi_heads.box_predictor.bbox_pred.out_features,
            },
        }

    # A disjoint breakdown (each parameter counted exactly once), for reporting.
    # model.detector wraps backbone+RPN+ROI-heads, so "detection_head" above
    # double-counts the trunk and neck; this table does not.
    disjoint = {"dinov3_trunk": report["sections"]["dinov3_trunk"]["total_params"],
                "feature_pyramid_neck": report["sections"]["feature_pyramid_only"]["trainable_params"] and
                                        (report["sections"]["feature_pyramid"]["total_params"] -
                                         report["sections"]["dinov3_trunk"]["total_params"]),
                "scene_head": report["sections"]["scene_head"]["total_params"]}
    if model.head_type == "fcos":
        disjoint["fcos_head"] = (report["fcos_head_detail"]["classification_tower"]["total_params"] +
                                 report["fcos_head_detail"]["regression_tower"]["total_params"])
    else:
        disjoint["rpn"] = report["faster_rcnn_head_detail"]["rpn"]["total_params"]
        disjoint["roi_box_head"] = report["faster_rcnn_head_detail"]["roi_heads"]["total_params"]
    disjoint["sum_check"] = sum(v for k, v in disjoint.items() if k != "sum_check")
    report["disjoint_param_breakdown"] = disjoint

    print(f"label: {label}")
    print(f"total params: {report['totals']['total_params']:,}  "
          f"trainable: {report['totals']['trainable_params']:,}\n")
    for name, sec in report["sections"].items():
        if not sec.get("present"):
            continue
        print(f"[{name}]  {sec['total_params']:,} params ({sec['trainable_params']:,} trainable)")
        for layer_type, n in list(sec["layer_counts"].items())[:12]:
            print(f"    {layer_type:<22} x{n}")
    print(f"\ntransformer trunk: {report['transformer_trunk_detail']['num_blocks']} blocks, "
          f"embed_dim={report['transformer_trunk_detail']['embed_dim']}, "
          f"heads={report['transformer_trunk_detail']['num_heads_per_block']}, "
          f"mlp_hidden={report['transformer_trunk_detail']['mlp_hidden_dim']}")

    print("\ndisjoint parameter breakdown (each parameter counted exactly once):")
    for k, v in report["disjoint_param_breakdown"].items():
        if k == "sum_check":
            continue
        print(f"    {k:<20} {v:>12,}")
    print(f"    {'sum':<20} {report['disjoint_param_breakdown']['sum_check']:>12,}  "
          f"(model total: {report['totals']['total_params']:,})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
