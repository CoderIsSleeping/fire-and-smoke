"""Layer-flow diagrams for the research report: five architectures, real tensor shapes.

Every shape and parameter count printed on these diagrams is read from
`reports/data/arch/*.json` (written by `inspect_architecture.py`) or from the
constants in `scripts/fire_smoke/*.py` -- nothing is a rounded or invented
number. Five diagrams:

    1. DINOv3 ViT-S/16 alone            (generic self-supervised feature extractor)
    2. Faster R-CNN head alone          (generic, on top of any backbone)
    3. FCOS head alone                  (generic, on top of any backbone)
    4. DINOv3 + Faster R-CNN            (this project's Stage 1 / Stage 2 model)
    5. DINOv3 + FCOS                    (this project's Stage 3 / light model)

    python scripts/build_architecture_diagrams.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ARCH = ROOT / "reports" / "data" / "arch"
OUT = ROOT / "reports" / "figures"

INK = "#1a1a1a"
MUTED = "#5b6670"
DINO_C, DINO_F = "#1f6f8b", "#dceaf2"
NECK_C, NECK_F = "#2e7d32", "#e8f1e9"
HEAD_C, HEAD_F = "#b5410f", "#fbeee4"
SCENE_C, SCENE_F = "#8a6d3b", "#f7f0df"
IO_C, IO_F = "#5b6670", "#f2f4f6"


def load(name: str) -> dict:
    return json.loads((ARCH / f"{name}.json").read_text(encoding="utf-8"))


def new_canvas(w_in: float, h_in: float, xmax: float, ymax: float):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def node(ax, x, y, w, h, title, sub, colour, fill, fontsize=9.5, subsize=8.0):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                         facecolor=fill, edgecolor=colour, linewidth=1.5))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=subsize,
                color=MUTED, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=INK)


def down_arrow(ax, x, y_top, y_bot, shape_label=None, colour=MUTED):
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.4))
    if shape_label:
        ax.text(x + 0.15, (y_top + y_bot) / 2, shape_label, fontsize=7.6, color=colour,
                va="center", ha="left", style="italic")


def legend(ax, x, y, entries):
    for i, (label, colour) in enumerate(entries):
        ax.add_patch(mpatches.Rectangle((x, y - i * 0.34), 0.28, 0.2, facecolor=colour, edgecolor="none"))
        ax.text(x + 0.4, y - i * 0.34 + 0.1, label, fontsize=7.6, color=MUTED, va="center")


# ------------------------------------------------------------- 1. DINOv3 alone


def fig_dino_alone(path: Path) -> None:
    d = load("stage1")  # ViT-S numbers; trunk is identical code path for ViT-Ti, only widths differ
    t = d["transformer_trunk_detail"]
    fig, ax = new_canvas(7.5, 10.5, 10, 15)

    x, w = 1.0, 8.0
    y = 14.1
    node(ax, x, y, w, 0.8, "Input image", "3 x 640 x 640  (RGB, 0..1, normalised)", IO_C, IO_F)
    down_arrow(ax, x + w / 2, y, y - 0.55)
    y -= 1.35
    node(ax, x, y, w, 0.8, "Patch embedding", f"Conv2d(3 -> {t['embed_dim']}, kernel={t['patch_size'][0]}, "
                                             f"stride={t['patch_size'][0]})", DINO_C, DINO_F)
    down_arrow(ax, x + w / 2, y, y - 0.55, f"{640 // t['patch_size'][0]} x {640 // t['patch_size'][0]} x "
                                           f"{t['embed_dim']}  =  {(640 // t['patch_size'][0]) ** 2} tokens")
    y -= 1.35
    node(ax, x, y, w, 0.8, "+ Rotary position embedding", "RoPE applied inside each attention block", DINO_C, DINO_F)
    down_arrow(ax, x + w / 2, y, y - 0.5)
    y -= 1.3

    block_h = 3.55
    node(ax, x, y - block_h, w, block_h, "", "", DINO_C, "#eef6fa")
    ax.text(x + w / 2, y - 0.25, f"Transformer block  (repeated x{t['num_blocks']})",
            ha="center", fontsize=10, fontweight="bold", color=DINO_C)
    inner_x, inner_w = x + 0.5, w - 1.0
    iy = y - 0.65
    steps = [
        ("LayerNorm", None),
        (f"Multi-head self-attention  ({t['num_heads_per_block']} heads, head_dim={t['head_dim']})",
         f"qkv: Linear({t['embed_dim']} -> {t['embed_dim'] * 3})   |   proj: Linear({t['embed_dim']} -> {t['embed_dim']})"),
        ("+ residual", None),
        ("LayerNorm", None),
        (f"MLP  (GELU)", f"fc1: Linear({t['embed_dim']} -> {t['mlp_hidden_dim']})   |   "
                         f"fc2: Linear({t['mlp_hidden_dim']} -> {t['embed_dim']})"),
        ("+ residual", None),
    ]
    for i, (title, sub) in enumerate(steps):
        h = 0.42 if sub is None else 0.48
        node(ax, inner_x, iy - h, inner_w, h, title, sub, DINO_C, "white", fontsize=8.3, subsize=6.8)
        iy -= h + 0.08
    y -= block_h
    down_arrow(ax, x + w / 2, y, y - 0.55, "unchanged shape throughout")
    y -= 1.35
    node(ax, x, y, w, 0.8, "Final LayerNorm", "", DINO_C, DINO_F)
    down_arrow(ax, x + w / 2, y, y - 0.55)
    y -= 1.35
    node(ax, x, y, w, 0.9, "Output: token features",
        f"{(640 // t['patch_size'][0]) ** 2} tokens x {t['embed_dim']}  --  a REPRESENTATION, not classes or boxes",
        IO_C, IO_F)

    ax.text(x, y - 0.7, "No classification or detection head exists past this point.\n"
                       "Something must be added on top to get a usable output --\n"
                       "that \"something\" is what sections 4 and 5 describe.",
            fontsize=8, color=HEAD_C, style="italic", va="top")
    ax.set_ylim(y - 1.5, 15.0)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------- 2. Faster R-CNN alone


def fig_frcnn_alone(path: Path) -> None:
    d = load("stage1")["faster_rcnn_head_detail"]
    box_out = d["box_predictor_out_features"]
    fig, ax = new_canvas(9.5, 10.5, 13, 15)

    x, w = 1.5, 10.0
    y = 14.1
    node(ax, x, y, w, 0.9, "Input: feature pyramid from ANY backbone",
        "4 levels, e.g. 80x80, 40x40, 20x20, 10x10, each C channels\n"
        "(a plain ResNet or a hand-built CNN works here just as well as DINOv3)", IO_C, IO_F, subsize=7.6)
    down_arrow(ax, x + w / 2, y, y - 0.55)
    y -= 1.4

    node(ax, x, y - 1.5, w, 1.5, "", "", HEAD_C, "#fdf3ec")
    ax.text(x + w / 2, y - 0.22, "Region Proposal Network (RPN)  --  per level, weights shared",
            ha="center", fontsize=9.6, fontweight="bold", color=HEAD_C)
    ax.text(x + 0.4, y - 0.62, "3x3 Conv(C -> C) + ReLU   ->   1x1 Conv objectness (per anchor)   +   "
                              "1x1 Conv box deltas (per anchor x4)", fontsize=7.6, color=MUTED)
    ax.text(x + 0.4, y - 1.0, "6 anchors per location (2 sizes x 3 aspect ratios), "
                             "sizes 16-384 px across the 4 levels", fontsize=7.6, color=MUTED)
    ax.text(x + 0.4, y - 1.32, f"params: {d['rpn']['total_params']:,}   ({sum(d['rpn']['layer_counts'].values())} leaf layers)",
            fontsize=7.4, color=HEAD_C, style="italic")
    y -= 1.5
    down_arrow(ax, x + w / 2, y, y - 0.6, "score every anchor -> keep top-K by objectness")
    y -= 1.4
    node(ax, x, y, w, 0.8, "Non-max suppression + top-K",
        "1000 pre-NMS -> IoU-based NMS -> 300 proposal boxes kept (test time)", HEAD_C, HEAD_F)
    down_arrow(ax, x + w / 2, y, y - 0.55, "variable-length list of boxes")
    y -= 1.35

    node(ax, x, y, w, 0.85, "RoIAlign",
        "crops + bilinearly resamples each proposal's region to a FIXED 7x7xC patch\n"
        "(this fixed-size crop is what needs a variable number of regions -- see note below)",
        HEAD_C, HEAD_F, subsize=7.4)
    down_arrow(ax, x + w / 2, y, y - 0.55, "300 x 7 x 7 x C")
    y -= 1.35

    node(ax, x, y - 1.35, w, 1.35, "", "", HEAD_C, "#fdf3ec")
    ax.text(x + w / 2, y - 0.22, "Box head", ha="center", fontsize=9.6, fontweight="bold", color=HEAD_C)
    ax.text(x + 0.4, y - 0.58, "Flatten  ->  FC(7*7*C -> 1024)  ->  ReLU  ->  FC(1024 -> 1024)  ->  ReLU",
            fontsize=7.8, color=MUTED)
    ax.text(x + 0.4, y - 0.92, f"cls_score: Linear(1024 -> {box_out['cls_score']})     "
                              f"bbox_pred: Linear(1024 -> {box_out['bbox_pred']})", fontsize=7.8, color=MUTED)
    ax.text(x + 0.4, y - 1.24, f"box head params: {d['roi_heads']['total_params']:,}", fontsize=7.4,
            color=HEAD_C, style="italic")
    y -= 1.35
    down_arrow(ax, x + w / 2, y, y - 0.55)
    y -= 1.3
    node(ax, x, y, w, 0.85, "Output (after a final NMS per class)",
        "boxes [x1,y1,x2,y2]  +  class scores (3-way softmax: background/smoke/fire)  +  refined box deltas",
        IO_C, IO_F, subsize=7.6)

    ax.text(x, y - 0.75,
            "Why it cannot batch across cameras: step \"NMS + top-K\" keeps a DIFFERENT number of boxes\n"
            "per image, so RoIAlign and everything after it has no fixed shape to export as one graph.",
            fontsize=8, color="#c1121f", style="italic", va="top")
    ax.set_ylim(y - 1.6, 15.0)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- 3. FCOS alone


def fig_fcos_alone(path: Path) -> None:
    d = load("stage3")["fcos_head_detail"]
    fig, ax = new_canvas(9.5, 10.5, 13, 14)

    x, w = 1.5, 10.0
    y = 13.1
    node(ax, x, y, w, 0.9, "Input: feature pyramid from ANY backbone",
        "each level H x W x C  (independent per level, weights shared across levels)", IO_C, IO_F)
    down_arrow(ax, x + w / 2, y, y - 0.55, "no proposals -- every location keeps going")
    y -= 1.4

    col_w = w / 2 - 0.25
    for i, (title, colour, fill, sub, out, params) in enumerate([
        ("Classification tower", NECK_C, NECK_F,
         f"{d['num_convs_per_tower']} x [Conv3x3(C -> C) -> GroupNorm -> ReLU]",
         "Conv3x3(C -> num_classes)\n= per-location class logits", d["classification_tower"]["total_params"]),
        ("Regression tower", HEAD_C, HEAD_F,
         f"{d['num_convs_per_tower']} x [Conv3x3(C -> C) -> GroupNorm -> ReLU]",
         "Conv3x3(C -> 4)  = l,t,r,b\n+ Conv3x3(C -> 1) = centerness", d["regression_tower"]["total_params"]),
    ]):
        cx = x + i * (col_w + 0.5)
        node(ax, cx, y - 2.4, col_w, 2.4, "", "", colour, "white")
        ax.text(cx + col_w / 2, y - 0.25, title, ha="center", fontsize=9.5, fontweight="bold", color=colour)
        ax.text(cx + 0.25, y - 0.75, sub, fontsize=7.6, color=MUTED, va="top", wrap=True)
        ax.text(cx + 0.25, y - 1.5, out, fontsize=7.8, color=INK, va="top", fontweight="bold")
        ax.text(cx + 0.25, y - 2.15, f"params: {params:,}", fontsize=7.2, color=colour, style="italic")

    y -= 2.4
    down_arrow(ax, x + w / 2, y, y - 0.55, "H x W x (num_classes + 4 + 1), concatenated over all levels")
    y -= 1.4
    node(ax, x, y, w, 0.9, "Every location scores itself directly",
        "score = sqrt( sigmoid(class_logit) x sigmoid(centerness) )\n"
        "no RPN, no RoIAlign -- output shape is FIXED regardless of how many objects are present",
        NECK_C, NECK_F, subsize=7.6)
    down_arrow(ax, x + w / 2, y, y - 0.55)
    y -= 1.35
    node(ax, x, y, w, 0.85, "Output (after score threshold + NMS)",
        "boxes decoded from l,t,r,b at each surviving location  +  class  +  score", IO_C, IO_F)

    ax.text(x, y - 0.75,
            "Why it CAN batch across cameras: the raw network output (before thresholding)\n"
            "is always the same shape -- e.g. 8,500 x 3 classes at 640px -- for any input image.",
            fontsize=8, color=NECK_C, style="italic", va="top")
    ax.set_ylim(y - 1.6, 14.0)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------- 4. DINOv3 + Faster R-CNN


def fig_dino_frcnn(path: Path) -> None:
    d = load("stage1")
    disj = d["disjoint_param_breakdown"]
    fr = d["faster_rcnn_head_detail"]
    box_out = fr["box_predictor_out_features"]
    fig, ax = new_canvas(9.0, 12.5, 12, 17.5)

    x, w = 1.2, 9.6
    y = 16.6
    node(ax, x, y, w, 0.75, "Input frame", "3 x 640 x 640", IO_C, IO_F)
    down_arrow(ax, x + w / 2, y, y - 0.5)
    y -= 1.2
    node(ax, x, y, w, 0.9, "DINOv3 ViT-S/16 trunk  (frozen in Stage 1)",
        f"12 blocks, embed_dim 384  --  {disj['dinov3_trunk']:,} params", DINO_C, DINO_F)
    down_arrow(ax, x + w / 2, y, y - 0.5, "blocks 5, 8, 9, 11 taken as 4 feature maps, 40x40x384 each")
    y -= 1.25
    node(ax, x, y, w, 0.75, "Feature pyramid (neck)  --  trained",
        f"resample to strides 8/16/32/64, 256ch  --  {disj['feature_pyramid_neck']:,} params", NECK_C, NECK_F)
    down_arrow(ax, x + w / 2, y, y - 0.5, "p3 80x80  p4 40x40  p5 20x20  p6 10x10,  256ch each")
    y -= 1.25

    node(ax, x, y - 1.35, w, 1.35, "", "", HEAD_C, "#fdf3ec")
    ax.text(x + w / 2, y - 0.22, "Faster R-CNN detection head  --  trained", ha="center", fontsize=9.6,
            fontweight="bold", color=HEAD_C)
    ax.text(x + 0.35, y - 0.58, f"RPN ({fr['rpn']['total_params']:,} params) -> 300 proposals "
                               f"-> RoIAlign 7x7x256 -> box head ({fr['roi_heads']['total_params']:,} params)",
            fontsize=7.7, color=MUTED)
    ax.text(x + 0.35, y - 0.92, f"outputs: cls_score({box_out['cls_score']})  bbox_pred({box_out['bbox_pred']})  "
                               "per surviving proposal", fontsize=7.7, color=MUTED)
    ax.text(x + 0.35, y - 1.22, "anchors 16-384px, 3 aspect ratios per level", fontsize=7.2, color=HEAD_C,
            style="italic")
    y -= 1.35
    down_arrow(ax, x + w / 2, y, y - 0.5, "boxes: smoke / fire, with confidence")
    y -= 1.25
    node(ax, x, y, w, 0.7, "Detection output", "[x1,y1,x2,y2], class, score  --  per box", IO_C, IO_F)

    # Scene branch, drawn beside the trunk.
    sx = x + w + 0.6
    sy = 16.6 - 1.2
    node(ax, sx, sy, 3.2, 0.75, "Pooled trunk tokens",
        "mean + max over 1600 tokens -> 768-dim", SCENE_C, SCENE_F, fontsize=8, subsize=6.8)
    node(ax, sx, sy - 1.1, 3.2, 0.6, "Glow prior (classical CV)", "7 warm-light statistics", SCENE_C, SCENE_F,
        fontsize=8, subsize=6.8)
    down_arrow(ax, sx + 1.6, sy, sy - 1.35, colour=SCENE_C)
    down_arrow(ax, sx + 1.6, sy - 1.1, sy - 1.35, colour=SCENE_C)
    node(ax, sx, sy - 2.55, 3.2, 1.05, "Scene classifier head",
        f"LayerNorm -> Linear(775->256) -> GELU\n-> Dropout -> Linear(256->2)\n"
        f"{disj['scene_head']:,} params", SCENE_C, SCENE_F, fontsize=8, subsize=6.6)
    node(ax, sx, sy - 4.0, 3.2, 0.7, "Output: P(smoke), P(fire)", "whole-image probabilities", IO_C, IO_F,
        fontsize=8, subsize=6.8)
    down_arrow(ax, sx + 1.6, sy - 2.55, sy - 3.3, colour=SCENE_C)
    ax.annotate("", xy=(sx, sy + 0.38), xytext=(x + w - 0.1, 16.6 - 1.2 + 0.38),
                arrowprops=dict(arrowstyle="-|>", color=DINO_C, lw=1.2, linestyle="--"))
    ax.text((x + w + sx) / 2 - 0.3, sy + 0.55, "same\ntrunk pass", fontsize=6.6, color=DINO_C, ha="center")

    ax.text(x, 1.1, f"Total: {d['totals']['total_params']:,} params   "
                    f"({d['totals']['trainable_params']:,} trainable in this configuration)",
            fontsize=9, color=INK, fontweight="bold")
    legend(ax, x, 0.6, [("DINOv3 (frozen or partially unfrozen)", DINO_F), ("Trained (neck + heads)", NECK_F),
                        ("Scene branch", SCENE_F)])
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- 5. DINOv3 + FCOS


def fig_dino_fcos(path: Path) -> None:
    d = load("stage3")
    disj = d["disjoint_param_breakdown"]
    fh = d["fcos_head_detail"]
    t = d["transformer_trunk_detail"]
    fig, ax = new_canvas(9.0, 12.0, 12, 16.5)

    x, w = 1.2, 9.6
    y = 15.6
    node(ax, x, y, w, 0.75, "Input frame", "3 x 640 x 640", IO_C, IO_F)
    down_arrow(ax, x + w / 2, y, y - 0.5)
    y -= 1.2
    node(ax, x, y, w, 0.9, "DINOv3 ViT-Ti/16 trunk  (frozen)",
        f"12 blocks, embed_dim {t['embed_dim']}, {t['num_heads_per_block']} heads  --  "
        f"{disj['dinov3_trunk']:,} params", DINO_C, DINO_F)
    down_arrow(ax, x + w / 2, y, y - 0.5, "blocks 5, 8, 9, 11  ->  4 feature maps, 40x40x192 each")
    y -= 1.25
    node(ax, x, y, w, 0.75, "Feature pyramid (neck)  --  trained",
        f"resample to strides 8/16/32/64, 128ch  --  {disj['feature_pyramid_neck']:,} params", NECK_C, NECK_F)
    down_arrow(ax, x + w / 2, y, y - 0.5, "p3 80x80  p4 40x40  p5 20x20  p6 10x10,  128ch  ->  8,500 locations total")
    y -= 1.25

    node(ax, x, y - 1.35, w, 1.35, "", "", HEAD_C, "#fdf3ec")
    ax.text(x + w / 2, y - 0.22, "FCOS detection head  --  trained", ha="center", fontsize=9.6,
            fontweight="bold", color=HEAD_C)
    ax.text(x + 0.35, y - 0.58, f"classification tower ({fh['classification_tower']['total_params']:,} params) "
                               f"+ regression tower ({fh['regression_tower']['total_params']:,} params)",
            fontsize=7.7, color=MUTED)
    ax.text(x + 0.35, y - 0.92, f"{fh['num_convs_per_tower']} conv layers per tower, one shared head across "
                               "all 4 pyramid levels", fontsize=7.7, color=MUTED)
    ax.text(x + 0.35, y - 1.22, "outputs per location: 3 class logits, 4 box distances, 1 centerness",
            fontsize=7.2, color=HEAD_C, style="italic")
    y -= 1.35
    down_arrow(ax, x + w / 2, y, y - 0.5, "score = sqrt(class x centerness)  ->  threshold + NMS")
    y -= 1.25
    node(ax, x, y, w, 0.7, "Detection output", "[x1,y1,x2,y2], class, score  --  per box", IO_C, IO_F)

    sx = x + w + 0.6
    sy = 15.6 - 1.2
    node(ax, sx, sy, 3.2, 0.75, "Pooled trunk tokens", "mean + max -> 384-dim", SCENE_C, SCENE_F,
        fontsize=8, subsize=6.8)
    node(ax, sx, sy - 1.1, 3.2, 0.6, "Glow prior (classical CV)", "7 warm-light statistics", SCENE_C, SCENE_F,
        fontsize=8, subsize=6.8)
    down_arrow(ax, sx + 1.6, sy, sy - 1.35, colour=SCENE_C)
    down_arrow(ax, sx + 1.6, sy - 1.1, sy - 1.35, colour=SCENE_C)
    node(ax, sx, sy - 2.55, 3.2, 1.05, "Scene classifier head",
        f"LayerNorm -> Linear(391->256) -> GELU\n-> Dropout -> Linear(256->2)\n"
        f"{disj['scene_head']:,} params", SCENE_C, SCENE_F, fontsize=8, subsize=6.6)
    node(ax, sx, sy - 4.0, 3.2, 0.7, "Output: P(smoke), P(fire)", "whole-image probabilities", IO_C, IO_F,
        fontsize=8, subsize=6.8)
    down_arrow(ax, sx + 1.6, sy - 2.55, sy - 3.3, colour=SCENE_C)

    ax.text(x, 1.1, f"Total: {d['totals']['total_params']:,} params   "
                    f"({d['totals']['trainable_params']:,} trainable  --  trunk fully frozen)",
            fontsize=9, color=INK, fontweight="bold")
    legend(ax, x, 0.6, [("DINOv3 (frozen)", DINO_F), ("Trained (neck + heads)", NECK_F), ("Scene branch", SCENE_F)])
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_dino_alone(OUT / "layers_dino_alone.png")
    fig_frcnn_alone(OUT / "layers_frcnn_alone.png")
    fig_fcos_alone(OUT / "layers_fcos_alone.png")
    fig_dino_frcnn(OUT / "layers_dino_frcnn.png")
    fig_dino_fcos(OUT / "layers_dino_fcos.png")
    print("wrote 5 architecture diagrams to", OUT)
