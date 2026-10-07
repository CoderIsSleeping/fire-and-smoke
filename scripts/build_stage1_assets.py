"""Diagrams, charts and the data bundle for the stage-1 model-selection report
and slide deck (DINOv3 vs MobileNetV3-Large vs ResNet-18, Faster R-CNN vs FCOS).

Every number is read from files: reports/data/arch/stage1_models.json holds
parameter and layer counts taken from the trained checkpoints, the *_results.csv
files hold the per-epoch validation results, the *_eval_test.md files the test
results. Nothing in a diagram is typed from memory where a file can supply it.

    python scripts/build_stage1_assets.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from build_training_report_pdf import parse_alarm_sweep, parse_detection_table

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "reports" / "data"
FIGS = ROOT / "reports" / "figures"
FONT_DIR = Path("C:/Windows/Fonts")
DPI = 320  # Calibri's embedded bitmap strikes render blank in matplotlib at low pixel sizes

INK, MUTED = "#1a1a1a", "#5b6670"
STYLE = {  # edge, fill
    "io": ("#5b6670", "#f2f4f6"), "bb": ("#1f6f8b", "#dceaf1"), "neck": ("#2e7d32", "#e3f1e4"),
    "head": ("#b5410f", "#fbe9df"), "scene": ("#6a1b9a", "#f1e6f6"), "plain": ("#9aa3ab", "#ffffff"),
}
MODELS = {  # key: (label, short, results csv, test report, best metrics, colour, arch key)
    "dino": ("DINOv3 ViT-S + Faster R-CNN", "DINOv3 ViT-S", "stage1", "stage1_eval_test", "stage1", "#1f6f8b", "dino"),
    "mbv3": ("MobileNetV3-L + Faster R-CNN", "MobileNetV3-L", "mbv3_stage1", "mbv3_stage1_eval_test", "mbv3_stage1",
             "#2e7d32", "mbv3"),
    "r18": ("ResNet-18 + Faster R-CNN", "ResNet-18", "resnet18_stage1", "resnet18_stage1_eval_test",
            "resnet18_stage1", "#c62828", "r18"),
    "fcos": ("DINOv3 ViT-Ti + FCOS", "ViT-Ti + FCOS", "light_stage1", "light_stage1_eval_test", "light_stage1",
             "#ef6c00", "fcos"),
}
# Laptop CPU, batch 1, 640 px, fp32, median of 15 runs (PROJECT_LOG 6f); midpoint of the two sessions.
CPU_MS = {"dino": (894, 912), "mbv3": (346, 385), "r18": (414, 414), "fcos": (338, 398)}
# Same ViT-S trunk, head swapped (PROJECT_LOG 6c), CPU ms per frame.
HEAD_SPEED = [("ViT-S + Faster R-CNN", 884, "39.3M"), ("ViT-S + FCOS, 4 convs", 1177, "-"),
              ("ViT-S + FCOS, 2 convs", 924, "27.2M"), ("ViT-S + FCOS, 2 convs, 128-ch pyramid", 746, "23.2M"),
              ("ViT-Ti + FCOS, 2 convs, 128-ch pyramid", 329, "6.9M")]


def setup_fonts() -> None:
    for f in ("calibri.ttf", "calibrib.ttf", "calibrii.ttf"):
        font_manager.fontManager.addfont(str(FONT_DIR / f))
    plt.rcParams["font.family"] = "Calibri"
    plt.rcParams["font.size"] = 10


# ------------------------------------------------------------------ data


def load() -> dict:
    arch = json.loads((DATA / "arch" / "stage1_models.json").read_text(encoding="utf-8"))
    out = {"models": {}}
    for key, (label, short, csv_name, test_name, best_name, colour, akey) in MODELS.items():
        with (DATA / f"{csv_name}_results.csv").open(encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("epoch")][:40]
        text = (DATA / f"{test_name}.md").read_text(encoding="utf-8")
        best = json.loads((DATA / f"{best_name}_best_metrics.json").read_text(encoding="utf-8"))
        epochs = [{
            "epoch": int(float(r["epoch"])), "mAP50": float(r["mAP50"]), "smoke": float(r["AP50_smoke"]),
            "fire": float(r["AP50_fire"]), "mAP50_95": float(r["mAP50_95"]), "loss": float(r["train_total"]),
            "recall": float(r["recall_any_at_op"]), "fpr": float(r["fpr_at_op"]), "seconds": float(r["seconds"]),
            "losses": {k: float(v) for k, v in r.items() if k.startswith("loss_")},
        } for r in rows]
        b = max(epochs, key=lambda e: e["mAP50"])
        ok = [r for r in best["alarm_sweep"] if r["fpr"] <= 0.01]
        op = min(ok, key=lambda r: r["threshold"])
        out["models"][key] = {
            "label": label, "short": short, "colour": colour, "epochs": epochs, "arch": arch[akey],
            "best_epoch": b["epoch"], "best_val": b["mAP50"], "best_val_smoke": b["smoke"], "best_val_fire": b["fire"],
            "min_per_epoch": sum(e["seconds"] for e in epochs) / len(epochs) / 60,
            "test": parse_detection_table(text), "test_sweep": parse_alarm_sweep(text),
            "val_sweep": best["alarm_sweep"], "val_op": op, "ckpt_epoch": best["epoch"],
            "ckpt_val": best["detection"]["mAP50"], "cpu_ms": CPU_MS[key],
        }
    out["head_speed"] = HEAD_SPEED
    return out


# ------------------------------------------------------------------ drawing helpers


def canvas(w_in: float, h_in: float, xmax: float, ymax: float):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, sub=None, kind="plain", tsize=9.6, ssize=7.9, lw=1.5):
    edge, fill = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fill, ec=edge, lw=lw))
    if sub:
        n = sub.count("\n") + 1
        ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="top", fontsize=tsize, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + (h - 0.62) / 2 + 0.02, sub, ha="center", va="center", fontsize=ssize, color=MUTED,
                linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=INK)


def arrow(ax, x1, y1, x2, y2, colour=MUTED, lw=1.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=colour, lw=lw,
                                 shrinkA=0, shrinkB=0))


def chain(ax, items, x0, y, h, gap=0.45):
    """Row of boxes joined by arrows. items = (width, title, sub, kind). Returns box x-centres."""
    centres, x = [], x0
    for i, (w, title, sub, kind) in enumerate(items):
        box(ax, x, y, w, h, title, sub, kind)
        centres.append(x + w / 2)
        if i < len(items) - 1:
            arrow(ax, x + w + 0.04, y + h / 2, x + w + gap - 0.04, y + h / 2)
        x += w + gap
    return centres


def panel_title(ax, x, y, text, colour=INK, size=10.5):
    ax.text(x, y, text, fontsize=size, fontweight="bold", color=colour, va="center")


def save(fig, name: str) -> None:
    fig.savefig(FIGS / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ architecture diagrams


def diag_pipeline() -> None:
    fig, ax = canvas(10.5, 3.9, 21, 7.8)
    chain(ax, [
        (2.6, "Camera frame", "any size,\nletterboxed to\n640 x 640", "io"),
        (3.3, "Backbone", "DINOv3, MobileNetV3\nor ResNet-18\n(pretrained, frozen)", "bb"),
        (3.3, "Feature pyramid", "4 maps: 80x80, 40x40,\n20x20, 10x10\n(strides 8/16/32/64)", "neck"),
        (3.5, "Detection head", "Faster R-CNN or FCOS\nboxes + class + score\n(smoke, fire)", "head"),
        (3.2, "Temporal check", "alarm only after\n6 detections in\n15 frames", "io"),
    ], 0.3, 4.4, 2.9)
    box(ax, 7.2, 0.5, 3.3, 2.5, "Scene classifier", "whole-image\nP(smoke), P(fire)\n+ 7 glow statistics", "scene")
    arrow(ax, 5.25, 4.4, 8.2, 3.05, colour=STYLE["scene"][0])
    ax.text(11.0, 1.75, "A second opinion on the whole frame: it can still respond when the flame is hidden\n"
                        "behind equipment and only its glow on nearby surfaces is visible.",
            fontsize=8.6, color=MUTED, va="center")
    ax.text(0.3, 7.5, "What is being compared", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(4.9, 7.5, "the backbone (3 candidates)", fontsize=9.5, color=STYLE["bb"][0], fontweight="bold")
    ax.text(11.6, 7.5, "the detection head (2 candidates)", fontsize=9.5, color=STYLE["head"][0], fontweight="bold")
    save(fig, "s1_pipeline.png")


def _backbone_bottom(ax, block_title, block_items, pyramid_note, x_split=11.3):
    panel_title(ax, 0.3, 3.75, block_title, STYLE["bb"][0])
    x = 0.3
    for i, (w, title, sub) in enumerate(block_items):
        box(ax, x, 1.55, w, 1.75, title, sub, "plain", tsize=8.6, ssize=7.4, lw=1.2)
        if i < len(block_items) - 1:
            arrow(ax, x + w + 0.03, 2.42, x + w + 0.27, 2.42)
        x += w + 0.3
    panel_title(ax, x_split, 3.75, "What our detector takes from it", STYLE["neck"][0])
    ax.text(x_split, 2.35, pyramid_note, fontsize=8.5, color=INK, va="center", linespacing=1.35)


def diag_dino(d: dict) -> None:
    a = d["models"]["dino"]["arch"]
    fig, ax = canvas(10.5, 4.9, 21, 9.8)
    panel_title(ax, 0.3, 9.45, f"DINOv3 ViT-S/16  -  {a['trunk'] / 1e6:.1f}M parameters, 12 transformer blocks", STYLE["bb"][0], 11.5)
    chain(ax, [
        (2.5, "Input", "3 x 640 x 640\nRGB image", "io"),
        (3.6, "Patch embedding", "cut into 16 x 16 px patches\n40 x 40 = 1,600 tokens\neach a 384-number vector", "bb"),
        (5.2, "12 transformer blocks", "every token looks at every other token\n(6 attention heads, 384-dim)\nposition given by rotary embedding (RoPE)", "bb"),
        (3.0, "Final LayerNorm", "", "bb"),
        (3.4, "Output", "1,600 tokens x 384\na feature map,\nnot classes or boxes", "io"),
    ], 0.3, 5.0, 3.6)
    ax.text(9.9, 4.55, "blocks 6, 9, 10 and 12 are tapped", fontsize=8.2, color=STYLE["neck"][0], style="italic", ha="center")
    _backbone_bottom(ax, "Inside one transformer block (repeated 12 times)", [
        (1.55, "LayerNorm", None), (2.5, "Self-attention", "6 heads\nqkv 384->1152"), (1.15, "+ skip", None),
        (1.55, "LayerNorm", None), (2.3, "MLP (GELU)", "384 -> 1536\n-> 384"), (1.15, "+ skip", None),
    ], "The 4 tapped maps (all 40 x 40) are\nresampled into a pyramid: p3 80 x 80,\n"
       "p4 40 x 40, p5 20 x 20, p6 10 x 10;\n256 channels each (as in ViTDet).\n"
       f"Neck: {a['neck'] / 1e6:.2f}M trainable parameters.", x_split=13.4)
    ax.text(0.3, 0.75, "Pretraining: self-supervised (no labels) on LVD-1689M, 1.689 billion images. One global receptive field from the first block.",
            fontsize=8.4, color=MUTED)
    save(fig, "s1_arch_dino.png")


def diag_resnet(d: dict) -> None:
    a = d["models"]["r18"]["arch"]
    fig, ax = canvas(10.5, 4.9, 21, 9.8)
    panel_title(ax, 0.3, 9.45, f"ResNet-18  -  {a['trunk'] / 1e6:.1f}M parameters, 8 residual blocks, "
                               f"{a['trunk_leaf_layers']['Conv2d']} convolutions", STYLE["bb"][0], 11.5)
    chain(ax, [
        (2.2, "Input", "3 x 640\nx 640", "io"),
        (2.9, "Stem", "7x7 conv, stride 2\n+ 3x3 max-pool\n64 ch, 160 x 160", "bb"),
        (2.7, "layer1", "2 blocks\n64 ch\n160 x 160", "bb"),
        (2.7, "layer2", "2 blocks\n128 ch\n80 x 80", "bb"),
        (2.7, "layer3", "2 blocks\n256 ch\n40 x 40", "bb"),
        (2.7, "layer4", "2 blocks\n512 ch\n20 x 20", "bb"),
    ], 0.3, 5.0, 3.6)
    for cx, name in ((11.95, "C3"), (15.1, "C4"), (18.25, "C5")):
        ax.text(cx, 4.55, f"{name} tapped", fontsize=8.2, color=STYLE["neck"][0], style="italic", ha="center")
    _backbone_bottom(ax, "Inside one residual block (BasicBlock)", [
        (2.0, "3x3 conv", "+ BatchNorm"), (1.2, "ReLU", None), (2.0, "3x3 conv", "+ BatchNorm"),
        (1.5, "+ skip", "input added\nback"), (1.2, "ReLU", None),
    ], "C3, C4, C5 (128 / 256 / 512 channels) go through a\n"
       "standard Feature Pyramid Network: p3 80 x 80, p4 40 x 40,\n"
       "p5 20 x 20, and p6 10 x 10 pooled from p5; 256 channels each.\n"
       f"Neck: {a['neck'] / 1e6:.2f}M trainable parameters.", x_split=10.6)
    ax.text(0.3, 0.75, "Pretraining: supervised classification on ImageNet-1k (1.28 million labelled images, 1,000 classes). "
                       "BatchNorm statistics kept frozen.", fontsize=8.4, color=MUTED)
    save(fig, "s1_arch_resnet18.png")


def diag_mobilenet(d: dict) -> None:
    a = d["models"]["mbv3"]["arch"]
    fig, ax = canvas(10.5, 4.9, 21, 9.8)
    panel_title(ax, 0.3, 9.45, f"MobileNetV3-Large  -  {a['trunk'] / 1e6:.1f}M parameters, 15 inverted-residual blocks, "
                               f"{a['trunk_leaf_layers']['Conv2d']} convolutions", STYLE["bb"][0], 11.5)
    chain(ax, [
        (2.0, "Input", "3 x 640\nx 640", "io"),
        (2.6, "Stem", "3x3 conv, stride 2\n16 ch, 320 x 320\nhard-swish", "bb"),
        (2.9, "Stages 1-2", "3 blocks\n-> 24 ch\n160 x 160", "bb"),
        (2.7, "Stage 3", "3 blocks, 5x5\n-> 40 ch\n80 x 80", "bb"),
        (2.9, "Stages 4-5", "6 blocks\n-> 112 ch\n40 x 40", "bb"),
        (3.2, "Stages 6-7", "3 blocks + 1x1 conv\n-> 960 ch\n20 x 20", "bb"),
    ], 0.3, 5.0, 3.6)
    for cx, name in ((11.55, "C3"), (14.8, "C4"), (18.3, "C5")):
        ax.text(cx, 4.55, f"{name} tapped", fontsize=8.2, color=STYLE["neck"][0], style="italic", ha="center")
    _backbone_bottom(ax, "Inside one inverted-residual block", [
        (2.0, "1x1 conv", "expand\nchannels"), (2.3, "Depthwise conv", "3x3 or 5x5,\none per channel"),
        (2.2, "Squeeze-excite", "channel attention\n(some blocks)"), (2.0, "1x1 conv", "project\n(linear)"),
        (1.3, "+ skip", None),
    ], "C3, C4, C5 (40 / 112 / 960 channels) go through the same\n"
       "Feature Pyramid Network as ResNet-18: p3 80 x 80,\n"
       "p4 40 x 40, p5 20 x 20, p6 10 x 10; 256 channels each.\n"
       f"Neck: {a['neck'] / 1e6:.2f}M trainable parameters.", x_split=11.7)
    ax.text(0.3, 0.75, "Pretraining: supervised classification on ImageNet-1k. Designed by architecture search for phones: "
                       "depthwise convolutions make it cheap.", fontsize=8.4, color=MUTED)
    save(fig, "s1_arch_mobilenetv3.png")


def diag_frcnn() -> None:
    fig, ax = canvas(10.5, 4.6, 21, 9.2)
    panel_title(ax, 0.3, 8.85, "Faster R-CNN  -  two-stage detector: first propose regions, then examine each one", STYLE["head"][0], 11.5)
    chain(ax, [
        (2.75, "Feature pyramid", "p3 ... p6\n256 channels", "neck"),
        (4.5, "Stage 1: proposals (RPN)", "3x3 conv, then at every location\n6 anchor boxes (2 sizes x 3 shapes):\n'object or not' + box correction", "head"),
        (2.75, "Proposals", "best ~300 boxes\nafter NMS\n(2,000 in training)", "io"),
        (2.75, "RoIAlign", "crop each proposal\nto a 7 x 7 x 256\nfeature patch", "head"),
        (3.6, "Stage 2: box head", "FC 12,544 -> 1,024\nFC 1,024 -> 1,024\nclass scores + box refinement", "head"),
        (2.3, "Output", "boxes, class\n(smoke / fire),\nscore 0-1", "io"),
    ], 0.15, 4.5, 3.6, gap=0.38)
    ax.text(0.3, 3.55, "Parameters (ours): RPN 0.60M, box head 13.91M  ->  14.5M, all trained.", fontsize=9, color=INK, fontweight="bold")
    ax.text(0.3, 2.45, "Training losses (4): RPN objectness (binary cross-entropy), RPN box (smooth L1),\n"
                       "box-head class (cross-entropy over background / smoke / fire), box-head box (smooth L1).",
            fontsize=8.8, color=MUTED, va="center", linespacing=1.35)
    ax.text(0.3, 1.15, "Anchors: sizes 16-32 / 48-96 / 128-192 / 256-384 px on p3 / p4 / p5 / p6, shapes 1:2, 1:1, 2:1.\n"
                       "The number of proposals depends on the image, so the middle of the network has no fixed shape.",
            fontsize=8.8, color=MUTED, va="center", linespacing=1.35)
    save(fig, "s1_arch_frcnn.png")


def diag_fcos(d: dict) -> None:
    a = d["models"]["fcos"]["arch"]
    fig, ax = canvas(10.5, 4.6, 21, 9.2)
    panel_title(ax, 0.3, 8.85, "FCOS  -  one-stage, anchor-free detector: every location predicts directly", STYLE["head"][0], 11.5)
    box(ax, 0.2, 4.9, 2.9, 2.9, "Feature pyramid", "p3 ... p6\n8,500 locations\nat 640 px", "neck")
    box(ax, 3.9, 6.5, 4.7, 1.9, "Classification tower", "2 x (3x3 conv + GroupNorm + ReLU)", "head")
    box(ax, 3.9, 4.2, 4.7, 1.9, "Regression tower", "2 x (3x3 conv + GroupNorm + ReLU)", "head")
    box(ax, 9.5, 6.5, 5.9, 1.9, "Class score", "per location: smoke / fire", "plain")
    box(ax, 9.5, 5.25, 5.9, 0.95, "Box: 4 distances (left, top, right, bottom)", None, "plain", tsize=8.8)
    box(ax, 9.5, 4.2, 5.9, 0.95, "Centre-ness: how central the point is", None, "plain", tsize=8.8)
    box(ax, 16.3, 4.9, 4.5, 2.9, "Output", "score = sqrt(class x centre-ness)\nkeep top 300, then NMS\nboxes, class, score", "io", ssize=7.6)
    for (x1, y1, x2, y2) in ((3.15, 6.8, 3.85, 7.3), (3.15, 5.9, 3.85, 5.3), (8.65, 7.45, 9.45, 7.45),
                             (8.65, 5.3, 9.45, 5.7), (8.65, 5.0, 9.45, 4.7), (15.45, 7.3, 16.25, 6.8),
                             (15.45, 5.2, 16.25, 5.8)):
        arrow(ax, x1, y1, x2, y2)
    ax.text(0.3, 3.4, f"Parameters (ours): {a['det_head'] / 1e6:.2f}M, all trained.", fontsize=9, color=INK, fontweight="bold")
    ax.text(0.3, 2.35, "Training losses (3): class (focal loss, handles the many empty locations),\n"
                       "box (GIoU loss), centre-ness (binary cross-entropy).", fontsize=8.8, color=MUTED, va="center", linespacing=1.35)
    ax.text(0.3, 1.1, "No anchors and no proposals: the network always outputs the same fixed-size tensors,\n"
                      "so frames from many cameras can be processed together as one batch.",
            fontsize=8.8, color=MUTED, va="center", linespacing=1.35)
    save(fig, "s1_arch_fcos.png")


# ------------------------------------------------------------------ charts


def chart_val_map(d: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    for key, m in d["models"].items():
        ax.plot([e["epoch"] for e in m["epochs"]], [e["mAP50"] for e in m["epochs"]], color=m["colour"], lw=2,
                label=f"{m['label']}  (best {m['best_val']:.4f} at epoch {m['best_epoch']})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation mAP@0.5")
    ax.set_xlim(1, 40)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.3, loc="lower right")
    fig.tight_layout()
    save(fig, "s1_val_map.png")


def chart_class_ap(d: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    for ax, cls, title in ((axes[0], "smoke", "Smoke AP@0.5"), (axes[1], "fire", "Fire AP@0.5")):
        for key, m in d["models"].items():
            ax.plot([e["epoch"] for e in m["epochs"]], [e[cls] for e in m["epochs"]], color=m["colour"], lw=1.8, label=m["short"])
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_xlim(1, 40)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("validation AP@0.5")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "s1_class_ap.png")


def chart_loss(d: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for key in ("dino", "mbv3", "r18"):
        m = d["models"][key]
        axes[0].plot([e["epoch"] for e in m["epochs"]], [e["loss"] for e in m["epochs"]], color=m["colour"], lw=1.8, label=m["short"])
    axes[0].set_title("Training loss - Faster R-CNN models (same loss)")
    m = d["models"]["fcos"]
    axes[1].plot([e["epoch"] for e in m["epochs"]], [e["loss"] for e in m["epochs"]], color=m["colour"], lw=1.8, label=m["short"])
    axes[1].set_title("Training loss - FCOS model (different loss)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_xlim(1, 40)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "s1_train_loss.png")


def chart_test_bars(d: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    keys = list(d["models"])
    w = 0.26
    for i, (cls, col, lab) in enumerate((("smoke", "#90a4ae", "smoke AP"), ("fire", "#ff8a65", "fire AP"), ("mean", "#37474f", "mAP@0.5"))):
        vals = [d["models"][k]["test"][cls]["AP50"] for k in keys]
        xs = [j + (i - 1) * w for j in range(len(keys))]
        ax.bar(xs, vals, w, label=lab, color=col)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([d["models"][k]["label"].replace(" + ", "\n+ ") for k in keys], fontsize=8.4)
    ax.set_ylim(0.45, 0.88)
    ax.set_ylabel("test AP@0.5")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    fig.tight_layout()
    save(fig, "s1_test_bars.png")


def chart_speed_acc(d: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for key, m in d["models"].items():
        ms = sum(m["cpu_ms"]) / 2
        v = m["test"]["mean"]["AP50"]
        ax.scatter(ms, v, s=130, color=m["colour"], zorder=3)
        ax.annotate(m["label"], (ms, v), textcoords="offset points", xytext=(9, -3), fontsize=8.4)
    ax.set_xlabel("laptop CPU time per frame, ms (lower is faster)")
    ax.set_ylabel("test mAP@0.5")
    ax.set_xlim(250, 1500)
    ax.set_ylim(0.59, 0.74)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "s1_speed_acc.png")


def chart_params(d: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    keys = list(d["models"])
    parts = (("trunk", "backbone (frozen)", "#1f6f8b"), ("neck", "feature pyramid", "#2e7d32"),
             ("det_head", "detection head", "#b5410f"), ("scene", "scene classifier", "#6a1b9a"))
    left = [0.0] * len(keys)
    for part, lab, col in parts:
        vals = [d["models"][k]["arch"][part] / 1e6 for k in keys]
        ax.barh(range(len(keys)), vals, left=left, label=lab, color=col, height=0.6)
        left = [a + b for a, b in zip(left, vals)]
    for i, k in enumerate(keys):
        a = d["models"][k]["arch"]
        ax.text(left[i] + 0.4, i, f"{a['total'] / 1e6:.1f}M total, {a['trainable_frozen_trunk'] / 1e6:.1f}M trained", va="center", fontsize=8.2)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([d["models"][k]["label"] for k in keys], fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 58)
    ax.set_xlabel("parameters (millions)")
    ax.legend(fontsize=7.8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "s1_params.png")


def chart_alarm(d: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for key, m in d["models"].items():
        sw = sorted(m["val_sweep"], key=lambda r: r["fpr"])
        ax.plot([r["fpr"] * 100 for r in sw], [r["recall_any"] * 100 for r in sw], "o-", ms=3, lw=1.6, color=m["colour"], label=m["label"])
    ax.axvline(1.0, color=MUTED, ls=":", lw=1)
    ax.text(1.03, 52, "1% false-alarm budget", fontsize=7.8, color=MUTED)
    ax.set_xlim(0, 4)
    ax.set_ylim(50, 100)
    ax.set_xlabel("false-alarm rate on normal images, %")
    ax.set_ylabel("fire/smoke images detected, %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "s1_alarm.png")


def chart_training_panel(d: dict, key: str) -> None:
    m = d["models"][key]
    ep = [e["epoch"] for e in m["epochs"]]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))
    ax = axes[0]
    ax.plot(ep, [e["loss"] for e in m["epochs"]], color=INK, lw=2, label="total")
    for name in m["epochs"][0]["losses"]:
        ax.plot(ep, [e["losses"][name] for e in m["epochs"]], lw=1.2, label=name.replace("loss_", ""))
    ax.set_title("Training losses")
    ax.legend(fontsize=6.8, ncol=2)
    ax = axes[1]
    ax.plot(ep, [e["mAP50"] for e in m["epochs"]], color=m["colour"], lw=2.2, label="mAP@0.5")
    ax.plot(ep, [e["smoke"] for e in m["epochs"]], color="#78909c", lw=1.3, ls="--", label="smoke AP")
    ax.plot(ep, [e["fire"] for e in m["epochs"]], color="#ff7043", lw=1.3, ls="--", label="fire AP")
    ax.plot(ep, [e["mAP50_95"] for e in m["epochs"]], color="#9e9e9e", lw=1.3, label="mAP@0.5:0.95")
    ax.set_title("Validation accuracy")
    ax.set_ylim(0, 0.9)
    ax.legend(fontsize=6.8, loc="lower right")
    ax = axes[2]
    ax.plot(ep, [e["recall"] * 100 for e in m["epochs"]], color="#2e7d32", lw=1.8, label="fire/smoke images detected")
    ax.plot(ep, [e["fpr"] * 100 for e in m["epochs"]], color="#c62828", lw=1.8, label="false alarms on normal images")
    ax.set_title("Alarm behaviour at <= 1% false alarms (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6.8, loc="center right")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_xlim(1, 40)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{m['label']}  -  40 epochs", fontsize=10.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, f"s1_train_{key}.png")


def main() -> None:
    setup_fonts()
    FIGS.mkdir(parents=True, exist_ok=True)
    d = load()
    diag_pipeline()
    diag_dino(d)
    diag_resnet(d)
    diag_mobilenet(d)
    diag_frcnn()
    diag_fcos(d)
    chart_val_map(d)
    chart_class_ap(d)
    chart_loss(d)
    chart_test_bars(d)
    chart_speed_acc(d)
    chart_params(d)
    chart_alarm(d)
    for key in d["models"]:
        chart_training_panel(d, key)
    (DATA / "stage1_presentation.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    print("wrote figures s1_*.png and reports/data/stage1_presentation.json")
    for key, m in d["models"].items():
        print(f"  {m['label']:30s} val {m['best_val']:.4f}@{m['best_epoch']}  test {m['test']['mean']['AP50']:.4f}  "
              f"{m['min_per_epoch']:.1f} min/epoch")


if __name__ == "__main__":
    main()
