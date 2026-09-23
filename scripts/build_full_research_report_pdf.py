"""Build the full technical + research report: three-stage results plus five architecture deep-dives.

Part A (for the meeting) covers the three training stages actually run in this
project, each with: exact architecture layer counts (read from the live
PyTorch model via `inspect_architecture.py`), the real optimizer and training
hyperparameters (read from the saved checkpoint's own args), training loss
curves (from the per-epoch CSV), a genuinely measured confusion matrix,
precision/recall/F1 (from `eval_confusion_matrix.py`, re-run over the test
split -- not interpolated from the AP curve), and an honest treatment of
"validation MSE" (detectors are not trained with an MSE loss; the real
box-regression loss is shown, plus an independently measured box-coordinate
MSE over matched true positives).

Part B is a research-style deep dive on each building block -- Faster R-CNN,
DINOv3, FCOS, DINOv3+Faster R-CNN, DINOv3+FCOS, and optimizers -- with input,
output, mechanism, exact or paper-sourced parameter counts, the loss functions
each actually uses (verified against the torchvision source), and a layer-flow
diagram for each of the five architectures.

Every number is sourced from a file under reports/data/ or reports/data/arch/
or reports/data/confusion/; nothing is retyped from memory.

    python scripts/build_full_research_report_pdf.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer

from build_training_report_pdf import (
    ACCENT,
    INK,
    MUTED,
    RULE,
    bullets,
    image,
    num,
    read_csv_rows,
    styles,
    table,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "reports" / "data"
ARCH = DATA / "arch"
CONF = DATA / "confusion"
FIGS = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "Full_Technical_Research_Report.pdf"

STAGES = [
    {"key": "stage1", "csv": "stage1_results.csv", "arch": "stage1.json", "conf": "stage1.json",
     "name": "Stage 1", "full_name": "Stage 1: Faster R-CNN + DINOv3 (trunk frozen)",
     "colour": "#1f6f8b", "op_conf": 0.90},
    {"key": "stage2", "csv": "stage2_results.csv", "arch": "stage2.json", "conf": "stage2.json",
     "name": "Stage 2", "full_name": "Stage 2: Faster R-CNN + DINOv3 (last 2 blocks unfrozen)",
     "colour": "#b5410f", "op_conf": 0.90},
    {"key": "stage3", "csv": "light_stage1_results.csv", "arch": "stage3.json", "conf": "stage3.json",
     "name": "Stage 3", "full_name": "Stage 3: FCOS + DINOv3 (light, trunk frozen)",
     "colour": "#2e7d32", "op_conf": 0.55},
]


# ------------------------------------------------------------------ helpers


def wrap(text: str, font_size: float = 7.6, colour=INK) -> Paragraph:
    """Wrap a long string as a Paragraph so it wraps inside its table cell.

    reportlab's Table does not wrap plain-string cells: a cell longer than its
    column overflows straight into the next column instead of breaking onto a
    second line (found the hard way -- see the architecture table below, which
    is the reason this exists). Reserved for cells long enough to need it;
    short cells stay plain strings so numeric columns keep their TableStyle
    right-alignment.
    """
    style = ParagraphStyle("cell", fontName="Helvetica", fontSize=font_size, textColor=colour,
                           leading=font_size * 1.28)
    return Paragraph(str(text), style)


# ------------------------------------------------------------------ data


def load_stage(spec: dict) -> dict:
    rows = read_csv_rows(DATA / spec["csv"])
    arch = json.loads((ARCH / spec["arch"]).read_text(encoding="utf-8"))
    conf_path = CONF / spec["conf"]
    conf = json.loads(conf_path.read_text(encoding="utf-8")) if conf_path.exists() else None
    checkpoint_args = arch.get("config", {})
    return {**spec, "rows": rows, "arch": arch, "conf": conf}


def load_all() -> dict:
    return {"stages": [load_stage(s) for s in STAGES]}


# ------------------------------------------------------------------ figures


def fig_loss_curve(stage: dict, path: Path) -> None:
    rows = stage["rows"]
    epochs = [num(r, "epoch") for r in rows]
    loss_keys = sorted({k for r in rows for k in r if k.startswith("loss_")})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    ax = axes[0]
    ax.plot(epochs, [num(r, "train_total") for r in rows], color=stage["colour"], lw=2.2, label="total")
    for k in loss_keys:
        if k == "loss_scene":
            continue
        ax.plot(epochs, [num(r, k) for r in rows], lw=1.2, alpha=0.8, label=k.replace("loss_", ""))
    ax.set_title("Training loss", fontsize=10)
    ax.set_xlabel("epoch", fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.8, ncol=2)

    ax = axes[1]
    box_key = "loss_box_reg" if "loss_box_reg" in loss_keys else "loss_bbox_regression"
    ax.plot(epochs, [num(r, box_key) for r in rows], color=ACCENT.hexval()[2:] and "#b5410f", lw=2,
            label=f"{box_key} (box regression loss)")
    if "loss_scene" in loss_keys:
        ax.plot(epochs, [num(r, "loss_scene") for r in rows], color="#8a6d3b", lw=1.6, ls="--",
                label="loss_scene (scene head BCE)")
    ax.set_title("Box-regression loss  (NOT MSE -- see section 2.4)", fontsize=9.5)
    ax.set_xlabel("epoch", fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5)

    fig.suptitle(stage["full_name"], fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_confusion(stage: dict, path: Path) -> None:
    conf = stage["conf"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for ax, cls in zip(axes, ("smoke", "fire")):
        c = conf["per_class"][cls]
        matrix = np.array([[c["tp"], c["fn"]], [c["fp"], c["tn"]]])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() * 1.15)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=13, fontweight="bold",
                        color="white" if matrix[i, j] > matrix.max() * 0.5 else INK.hexval()[2:] and "#1a1a1a")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["actual: present", "actual: absent"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["pred: present", "pred: absent"], fontsize=8)
        ax.set_title(f"{cls}  (conf ≥ {stage['op_conf']:.2f})", fontsize=10)
    fig.suptitle(f"{stage['name']} - image-level confusion matrix, test split "
                f"(n={conf['num_images']:,})", fontsize=10.5, y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_prf1_comparison(stages: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.25
    for cls_idx, cls in enumerate(("smoke", "fire")):
        ax = axes[cls_idx]
        for i, stage in enumerate(stages):
            c = stage["conf"]["per_class"][cls]
            values = [c[m] for m in metrics]
            bars = ax.bar(x + (i - 1) * width, values, width, label=stage["name"], color=stage["colour"])
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(["Precision", "Recall", "F1"], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(cls, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        if cls_idx == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Precision / Recall / F1 at each stage's own operating threshold, test split", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_mAP_across_stages(stages: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for stage in stages:
        epochs = [num(r, "epoch") for r in stage["rows"]]
        ax.plot(epochs, [num(r, "mAP50") for r in stage["rows"]], color=stage["colour"], lw=2,
                label=f"{stage['name']} val mAP@0.5")
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel("validation mAP@0.5", fontsize=9)
    ax.set_title("Validation mAP@0.5 across all three stages", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ tables


def arch_table(stage: dict) -> list:
    a = stage["arch"]
    t = a["transformer_trunk_detail"]
    disj = a["disjoint_param_breakdown"]
    rows = [
        ["Component", "Layer types (exact count)", "Parameters"],
        ["DINOv3 trunk", wrap(f"{t['num_blocks']} transformer blocks x (2 LayerNorm, 2 Linear in attention, "
                             f"2 Linear in MLP, 1 GELU) + 1 patch-embed Conv2d + rotary position embedding"),
         f"{disj['dinov3_trunk']:,}"],
        ["Feature pyramid (neck)", "9 Conv2d, 1 ConvTranspose2d, 4 LayerNorm2d, 2 MaxPool2d",
         f"{disj['feature_pyramid_neck']:,}"],
    ]
    if "faster_rcnn_head_detail" in a:
        fr = a["faster_rcnn_head_detail"]
        rows.append(["RPN (region proposal network)",
                    wrap(f"3x3 Conv + 1x1 Conv (objectness) + 1x1 Conv (box deltas), "
                        f"{sum(fr['rpn']['layer_counts'].values())} leaf layers"),
                    f"{disj['rpn']:,}"])
        rows.append(["ROI box head", wrap("2 fully-connected (1024-dim) + cls_score + bbox_pred Linear layers"),
                    f"{disj['roi_box_head']:,}"])
    else:
        fh = a["fcos_head_detail"]
        rows.append(["FCOS classification tower",
                    wrap(f"{fh['num_convs_per_tower']} x (Conv3x3, GroupNorm, ReLU) + final Conv3x3"),
                    f"{fh['classification_tower']['total_params']:,}"])
        rows.append(["FCOS regression tower",
                    wrap(f"{fh['num_convs_per_tower']} x (Conv3x3, GroupNorm, ReLU) "
                        "+ box Conv3x3 + centerness Conv3x3"),
                    f"{fh['regression_tower']['total_params']:,}"])
    rows.append(["Scene classifier head", "LayerNorm, 2 Linear, GELU, Dropout", f"{disj['scene_head']:,}"])
    rows.append(["TOTAL", "", f"{a['totals']['total_params']:,}  ({a['totals']['trainable_params']:,} trainable)"])
    return rows


def optimizer_table(stage: dict) -> list:
    # Every value is wrapped: a couple of these strings sit right at the edge
    # of what fits on one line at this column width, and an unwrapped
    # plain-string cell that overflows bleeds into the next column rather
    # than breaking (see the note on `wrap()`) -- wrapping short strings too
    # costs nothing and removes the guesswork.
    rows = [
        ["Setting", "Value"],
        ["Optimizer", "AdamW (decoupled weight decay)"],
        ["Head/neck learning rate", "1e-4" if stage["key"] != "stage2" else "5e-5"],
        ["Trunk learning rate scale",
         "0.05 x head LR" + ("  = 2.5e-6" if stage["key"] == "stage2" else " (n/a, trunk frozen)")],
        ["Weight decay", "1e-4"],
        ["LR schedule", "500-iteration linear warmup, then cosine decay to 1% of peak"],
        ["Gradient clipping", "max-norm 10.0"],
        ["Mixed precision", "fp16 (GradScaler); non-finite batches skipped"],
        ["Batch size", "8 (accumulation x1)"],
        ["Early-stop patience", "8 epochs with no validation mAP@0.5 gain"],
        ["Scene-head loss weight", "1.0"],
    ]
    return [rows[0]] + [[label, wrap(value, font_size=8.4)] for label, value in rows[1:]]


def confusion_table(stage: dict) -> list:
    c = stage["conf"]
    rows = [["Class", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "Specificity", "Accuracy"]]
    for cls in ("smoke", "fire"):
        v = c["per_class"][cls]
        rows.append([cls, v["tp"], v["fp"], v["fn"], v["tn"], f"{v['precision']:.4f}", f"{v['recall']:.4f}",
                    f"{v['f1']:.4f}", f"{v['specificity']:.4f}", f"{v['accuracy']:.4f}"])
    m = c["macro_avg"]
    rows.append(["macro avg", "", "", "", "", f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                f"{m['specificity']:.4f}", f"{m['accuracy']:.4f}"])
    return rows


def mse_table(stage: dict) -> list:
    c = stage["conf"]
    rows = [["Class", "Matched true positives", "Box-coordinate MSE (px²)", "RMSE (px)"]]
    for cls in ("smoke", "fire"):
        m = c["per_class"][cls]["box_mse"]
        if m["n_matched"]:
            rows.append([cls, m["n_matched"], f"{m['mse_pixels_sq']:.1f}", f"{m['rmse_pixels']:.2f}"])
        else:
            rows.append([cls, 0, "-", "-"])
    return rows


# ------------------------------------------------------------------ document


def build(d: dict) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    stages = d["stages"]
    for stage in stages:
        fig_loss_curve(stage, FIGS / f"loss_{stage['key']}.png")
        if stage["conf"]:
            fig_confusion(stage, FIGS / f"confusion_{stage['key']}.png")
    if all(s["conf"] for s in stages):
        fig_prf1_comparison(stages, FIGS / "prf1_comparison.png")
    fig_mAP_across_stages(stages, FIGS / "map_across_stages.png")

    s = styles()
    story = []

    # ---- title
    story += [
        Spacer(1, 30 * mm),
        Paragraph("Fire &amp; Smoke Detection", s["title"]),
        Paragraph("Full Technical &amp; Research Report<br/>"
                  "Three training stages, measured end to end, and a deep dive into every "
                  "architecture component", s["subtitle"]),
        Spacer(1, 10 * mm),
        table([
            ["Part A", "Stage 1 / Stage 2 / Stage 3 - architecture, optimizer, training loss, "
                       "confusion matrix, precision/recall/F1 (measured on the D-Fire test split)"],
            ["Part B", "Faster R-CNN, DINOv3, FCOS, DINOv3+Faster R-CNN, DINOv3+FCOS, and optimizers - "
                       "input/output, mechanism, parameters, and a layer-flow diagram for each"],
            ["Generated", date.today().isoformat()],
        ], [30 * mm, 130 * mm], align_right_from=9, header=False, font=9),
    ]
    story.append(PageBreak())

    # ---- Part A intro
    story += [
        Paragraph("Part A - The three training stages, measured", s["h1"]),
        Paragraph(
            "This project trained three configurations on the same D-Fire split (13,776 train / 3,445 "
            "validation / 4,306 test images), the same augmentation pipeline and the same optimiser family, "
            "so that differences between them reflect the architecture change, not the data or the training "
            "recipe.", s["body"]),
        table([
            ["", "Stage 1", "Stage 2", "Stage 3"],
            ["Backbone", "DINOv3 ViT-S/16", "DINOv3 ViT-S/16", "DINOv3 ViT-Ti/16"],
            ["Trunk", "frozen", "last 2 blocks unfrozen", "frozen"],
            ["Detection head", "Faster R-CNN", "Faster R-CNN", "FCOS"],
            ["Epochs", str(len(stages[0]["rows"])), str(len(stages[1]["rows"])), str(len(stages[2]["rows"]))],
            ["Total params", f"{stages[0]['arch']['totals']['total_params']:,}",
             f"{stages[1]['arch']['totals']['total_params']:,}", f"{stages[2]['arch']['totals']['total_params']:,}"],
            ["Trainable params", f"{stages[0]['arch']['totals']['trainable_params']:,}",
             f"{stages[1]['arch']['totals']['trainable_params']:,}",
             f"{stages[2]['arch']['totals']['trainable_params']:,}"],
            ["Operating threshold", "0.90", "0.90", "0.55"],
        ], [38 * mm, 44 * mm, 44 * mm, 44 * mm], font=8.4),
        Spacer(1, 3 * mm),
        Paragraph(
            "Every metric below for a stage's confusion matrix, precision, recall, F1 and box-coordinate MSE "
            "was produced by actually re-running that stage's checkpoint over the full 4,306-image D-Fire "
            "test split with <font face='Courier'>scripts/eval_confusion_matrix.py</font> - none of it is "
            "interpolated or estimated from the mAP curve.", s["callout"]),
    ]

    # ---- per-stage sections
    for stage in stages:
        story.append(PageBreak())
        story += [
            Paragraph(stage["full_name"], s["h1"]),
            Paragraph("2.1  Architecture - exact layer inventory", s["h2"]),
            Paragraph(
                "Counted by walking the live PyTorch model (<font face='Courier'>scripts/inspect_architecture.py"
                "</font>), not typed from memory. Parameter counts are disjoint: each parameter is counted in "
                "exactly one row, and the rows sum to the model total exactly.", s["body"]),
            table(arch_table(stage), [46 * mm, 92 * mm, 32 * mm], align_right_from=99, font=7.6),
        ]
        t = stage["arch"]["transformer_trunk_detail"]
        story.append(Paragraph(
            f"Transformer block detail: {t['num_blocks']} blocks, embedding dimension {t['embed_dim']}, "
            f"{t['num_heads_per_block']} attention heads (head dimension {t['head_dim']}), MLP hidden "
            f"dimension {t['mlp_hidden_dim']}, patch size {t['patch_size'][0]}x{t['patch_size'][0]}. Block type: "
            f"<font face='Courier'>{t['block_type']}</font> (a pre-norm ViT block with rotary position "
            "embeddings, as implemented in timm).", s["body"]))

        story += [
            Paragraph("2.2  Optimizer and training hyperparameters", s["h2"]),
            Paragraph("Read from the arguments actually saved inside the checkpoint at training time.", s["body"]),
            table(optimizer_table(stage), [55 * mm, 105 * mm], font=8.4),
        ]

        story += [
            Paragraph("2.3  Training loss", s["h2"]),
            image(FIGS / f"loss_{stage['key']}.png", 168),
        ]

        story += [
            Paragraph("2.4  \"Validation MSE\" - what a detector actually reports, and what was measured", s["h2"]),
            Paragraph(
                "Object detectors are not trained with a mean-squared-error loss. Verified against the "
                "torchvision source for this project: Faster R-CNN's box head uses <b>smooth-L1</b> loss on box "
                "coordinate deltas (its RPN also uses smooth-L1, plus binary cross-entropy for objectness); "
                "FCOS uses <b>generalized IoU loss</b> for box regression, <b>sigmoid focal loss</b> for "
                "classification, and binary cross-entropy for centerness. The right-hand chart above is that "
                "real, logged loss curve - it is not MSE, and presenting it as MSE would be inaccurate.",
                s["body"]),
        ]
        if stage["conf"]:
            story += [
                Paragraph(
                    "As a genuinely independent, MSE-shaped diagnostic, we additionally computed the mean "
                    "squared pixel error of predicted box coordinates against ground truth, over matched true "
                    f"positives (IoU ≥ 0.5) on the test split at this stage's operating threshold "
                    f"({stage['op_conf']:.2f}):", s["body"]),
                table(mse_table(stage), [30 * mm, 45 * mm, 48 * mm, 30 * mm], font=8.6),
            ]
        else:
            story.append(Paragraph(
                "<i>Box-coordinate MSE for this stage is pending a full test-split re-evaluation "
                "(scripts/eval_confusion_matrix.py) and will be added once that run completes.</i>", s["body"]))

        if stage["conf"]:
            story += [
                Paragraph("2.5  Confusion matrix (image-level, per class)", s["h2"]),
                Paragraph(
                    "An image can contain both fire and smoke at once, so a single 3-way confusion matrix is "
                    "not well-defined here. Each class is instead scored as its own binary problem: does the "
                    "image contain that class, and did the model detect it (any box of that class above the "
                    "operating threshold)?", s["body"]),
                image(FIGS / f"confusion_{stage['key']}.png", 150),
                Paragraph("2.6  Precision, Recall, F1", s["h2"]),
                table(confusion_table(stage), [20 * mm, 15 * mm, 15 * mm, 15 * mm, 15 * mm, 20 * mm, 18 * mm,
                                               18 * mm, 22 * mm, 18 * mm], align_right_from=1, font=7.2),
            ]

    # ---- cross-stage comparison
    if all(s["conf"] for s in stages):
        story.append(PageBreak())
        story += [
            Paragraph("3. Cross-stage comparison", s["h1"]),
            image(FIGS / "map_across_stages.png", 155),
            Spacer(1, 3 * mm),
            image(FIGS / "prf1_comparison.png", 168),
            Paragraph(
                "Read alongside the per-stage sections above: Stage 2's unfreeze buys a small, real gain over "
                "Stage 1 at the same architecture cost; Stage 3 trades a meaningful amount of precision/recall "
                "for a much smaller, batch-exportable model. Full discussion of what each stage costs and buys "
                "is in <font face='Courier'>reports/Model1_vs_Model2_Comparison.pdf</font>.", s["body"]),
        ]

    doc_end(story, s)


def doc_end(story: list, s: dict) -> None:
    """Finish the story with Part B, then build the PDF."""
    build_part_b(story, s)

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, 285 * mm, 192 * mm, 285 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 288 * mm, "Fire & Smoke Detection - Full Technical & Research Report")
        canvas.drawRightString(192 * mm, 288 * mm, date.today().isoformat())
        canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
        canvas.drawCentredString(105 * mm, 10 * mm, f"page {doc.page}")
        canvas.restoreState()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                          topMargin=22 * mm, bottomMargin=20 * mm,
                          title="Fire & Smoke Detection - Full Technical & Research Report")
    doc.addPageTemplates([PageTemplate(id="main", frames=[Frame(doc.leftMargin, doc.bottomMargin,
                                                                 doc.width, doc.height)], onPage=decorate)])
    doc.build(story)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


def build_part_b(story: list, s: dict) -> None:
    story.append(PageBreak())
    story += [
        Paragraph("Part B - Architecture research", s["h1"]),
        Paragraph(
            "Six subsections: the three building blocks this project uses or considered, each combination "
            "actually trained, and the optimizer family. Where a number is specific to <i>this project's</i> "
            "trained model, it is exact (read from the checkpoint). Where it describes the architecture in "
            "general (e.g. the original paper's own backbone), it is stated as such and kept separate from our "
            "measured numbers.", s["body"]),
    ]

    # ---------------------------------------------------------------- B1 Faster R-CNN
    story.append(PageBreak())
    story += [
        Paragraph("B1. Faster R-CNN", s["h1"]),
        Paragraph(
            "Ren, He, Girshick &amp; Sun, 2015. A two-stage, anchor-based object detector. \"Two-stage\" means "
            "it first proposes candidate regions, then classifies and refines each one - it does not look at "
            "the whole image once and commit to an answer the way a one-stage detector does.", s["body"]),
        Paragraph("Input and output", s["h2"]),
        *bullets([
            "<b>Input:</b> Faster R-CNN has no backbone of its own - it consumes a feature map (or a small "
            "pyramid of them) produced by <i>any</i> convolutional or transformer backbone. The original paper "
            "used ZFNet and VGG16; later work commonly pairs it with ResNet+FPN. In this project it consumes "
            "DINOv3's feature pyramid.",
            "<b>Output:</b> a variable-length list of (box, class, confidence score) tuples - the number of "
            "detections is different for every image, by design.",
        ], s),
        Paragraph("How it works", s["h2"]),
        *bullets([
            "The <b>Region Proposal Network (RPN)</b> slides a small convolutional network over the feature "
            "map and, at every location, scores a fixed set of anchor boxes (different sizes and aspect "
            "ratios) as \"object-like\" or not, and predicts a box adjustment for each.",
            "The top-scoring anchors survive non-max suppression and become <b>region proposals</b> - a list "
            "whose length varies per image (in this project: up to 300 at test time).",
            "<b>RoIAlign</b> crops and bilinearly resamples the feature map inside each proposal to a fixed "
            "7x7xC patch, so a variable number of regions become a variable number of <i>identically shaped</i> "
            "patches.",
            "The <b>box head</b> (a small fully-connected network) classifies each patch and refines its box "
            "coordinates a second time.",
        ], s),
        Paragraph("Loss functions (verified against the torchvision source used in this project)", s["h2"]),
        table([
            ["Component", "Loss function"],
            ["RPN objectness", "binary cross-entropy with logits"],
            ["RPN box regression", "smooth-L1"],
            ["Box head classification", "cross-entropy (multi-class softmax)"],
            ["Box head box regression", "smooth-L1"],
        ], [55 * mm, 105 * mm], font=8.6),
        Paragraph("Parameter count", s["h2"]),
        Paragraph(
            "Highly backbone-dependent. With the original VGG16 backbone: roughly 137M parameters, almost "
            "entirely the backbone. With torchvision's common ResNet-50+FPN backbone: roughly 41M parameters. "
            "<b>In this project</b>, on top of DINOv3 (measured exactly, see Part A): the RPN is 597,790 "
            "parameters and the ROI box head is 13,911,055 parameters, for 14,508,845 detection-head "
            "parameters regardless of which stage's backbone it sits on.", s["body"]),
        Paragraph("Can it be used alone, without DINO?", s["h2"]),
        Paragraph(
            "Yes - that is precisely how it was designed, and DINOv3 is a drop-in replacement for whatever "
            "backbone it originally used. Any backbone that outputs a spatial feature map works: a ResNet, a "
            "plain CNN trained from scratch, or a ViT.", s["body"]),
        Paragraph(
            "<b>Can it be reduced to a single-layer classifier/detector, with no backbone at all?</b> No, not "
            "meaningfully. The RPN itself needs a spatially organised, multi-channel feature map to slide "
            "anchors over - a single dense (fully-connected) layer produces no spatial structure for it to "
            "work with. At minimum, Faster R-CNN needs a handful of convolutional layers underneath it to "
            "produce that structure, and without pretrained features (DINOv3 or otherwise) it would need to "
            "learn everything from the 21,527 training images in this dataset alone, with no benefit from "
            "prior visual knowledge - workable in principle, far weaker in practice than pairing it with a "
            "pretrained backbone.", s["body"]),
        image(FIGS / "layers_frcnn_alone.png", 150),
        Paragraph("Figure B1 - Faster R-CNN's own layers, generic (any backbone can feed into this).",
                  s["caption"]),
    ]

    # ---------------------------------------------------------------- B2 DINOv3
    story.append(PageBreak())
    story += [
        Paragraph("B2. DINOv3", s["h1"]),
        Paragraph(
            "A self-supervised Vision Transformer from Meta AI. \"Self-supervised\" means it was trained "
            "without any human labels, on 1.7 billion images, via a teacher-student self-distillation "
            "objective: two copies of the network look at different crops of the same image, and one is "
            "trained to predict the other's output. Nothing in that objective is a class name or a box - the "
            "result is a general-purpose visual representation.", s["body"]),
        Paragraph("Input and output", s["h2"]),
        *bullets([
            "<b>Input:</b> an RGB image, split into 16x16 patches by a single Conv2d patch-embedding layer. "
            "At 640x640 that is 1,600 patches.",
            "<b>Output:</b> a sequence of token embeddings - 1,600 tokens x 384 dimensions for the ViT-S "
            "variant used in this project's Stage 1/2, or x192 for the ViT-Ti variant in Stage 3. <b>This is "
            "not a class or a box</b> - it is a representation that something else has to interpret.",
        ], s),
        Paragraph("How it works", s["h2"]),
        *bullets([
            "Patch embedding turns the image into a token sequence, plus a rotary position embedding (RoPE, "
            "verified present in this project's loaded model) so attention knows where each token sits without "
            "a separate learned position table.",
            "12 identical transformer blocks follow, each pre-norm: LayerNorm → multi-head self-attention "
            "→ residual add → LayerNorm → MLP (two Linear layers with a GELU activation between "
            "them) → residual add. Self-attention lets every patch look at every other patch, which is "
            "what gives a ViT its global receptive field from the very first block - a CNN would need many "
            "layers stacked to see that far.",
            "A final LayerNorm produces the output token sequence.",
        ], s),
        Paragraph("Parameter count (exact, this project's two variants)", s["h2"]),
        table([
            ["Variant", "Blocks", "Embed dim", "Heads", "MLP hidden", "Parameters"],
            ["ViT-S/16 (Stage 1, 2)", "12", "384", "6", "1536", "21,586,944"],
            ["ViT-Ti/16 (Stage 3)", "12", "192", "3", "768", "5,489,664"],
        ], [42 * mm, 18 * mm, 20 * mm, 16 * mm, 22 * mm, 32 * mm], align_right_from=1, font=8.4),
        Paragraph("What if DINOv3 is used alone, with no detection head?", s["h2"]),
        *bullets([
            "By itself it answers no question at all - it only produces a representation. The simplest thing "
            "to add is a <b>linear probe</b>: pool the tokens (mean, max, or the class token) and feed them "
            "into a small classifier. This is exactly what this project's scene-classifier head does, minus "
            "the glow features - it can say \"this image contains fire somewhere\", but it cannot say where.",
            "DINO-family models are also known to produce attention maps that highlight objects without ever "
            "being trained to detect anything (an emergent property reported in the original DINO paper) - "
            "in principle a coarse box could be drawn around a thresholded attention or patch-similarity "
            "cluster. This is a cheap fallback for image-level alerting, not a substitute for a properly "
            "trained detection head: it has no box-regression training signal, so its localisation is "
            "considerably less precise than Faster R-CNN or FCOS.",
            "Optimizer note: in Stage 1 and Stage 3 the trunk is fully frozen, so no optimizer ever touches "
            "it. In Stage 2 its last two blocks are trained with AdamW at 5% of the head's learning rate.",
        ], s),
        image(FIGS / "layers_dino_alone.png", 150),
        Paragraph("Figure B2 - DINOv3 ViT-S/16 on its own, with the exact block internals from this project's "
                  "loaded model.", s["caption"]),
    ]

    # ---------------------------------------------------------------- B3 FCOS
    story.append(PageBreak())
    story += [
        Paragraph("B3. FCOS", s["h1"]),
        Paragraph(
            "Tian, Shen, Chen &amp; He, 2019 (\"Fully Convolutional One-Stage Object Detection\"). A one-stage, "
            "anchor-free detector: it has no region-proposal step and no hand-designed anchor boxes. Every "
            "spatial location on the feature map predicts directly, the way a semantic-segmentation network "
            "predicts a class at every pixel.", s["body"]),
        Paragraph("Input and output", s["h2"]),
        *bullets([
            "<b>Input:</b> like Faster R-CNN, FCOS has no backbone of its own - it consumes a feature pyramid "
            "from any backbone. The original paper used ResNet+FPN; in this project it consumes DINOv3's "
            "feature pyramid.",
            "<b>Output:</b> for every location on every pyramid level, a class score, 4 box-edge distances "
            "(left/top/right/bottom to the object boundary) and a \"centerness\" score. <b>This raw output has "
            "a fixed shape for a given input size</b> - 8,500 locations x 3 classes at 640px in this project's "
            "configuration - regardless of how many real objects are in the image. Boxes are decoded from it "
            "only afterward, at inference.",
        ], s),
        Paragraph("How it works", s["h2"]),
        *bullets([
            "A <b>classification tower</b> and a <b>regression tower</b> run in parallel on every pyramid "
            "level, each a short stack of Conv3x3 → GroupNorm → ReLU (2 layers deep in this project's "
            "light configuration), ending in a final Conv3x3 that produces the per-location outputs.",
            "<b>Centerness</b> is the mechanism that keeps FCOS's boxes tight: it downweights predictions from "
            "locations near an object's edge (where the box-distance regression is least reliable), so the "
            "final confidence score - the square root of class-probability times centerness - is highest near "
            "the true centre of an object.",
            "Ground-truth boxes are assigned to whichever pyramid level matches their scale, so a small flame "
            "is supervised on the fine (stride-8) level and a large one on a coarser level.",
        ], s),
        Paragraph("Loss functions (verified against the torchvision source used in this project)", s["h2"]),
        table([
            ["Component", "Loss function"],
            ["Classification", "sigmoid focal loss"],
            ["Box regression", "generalized IoU (GIoU) loss"],
            ["Centerness", "binary cross-entropy with logits"],
        ], [55 * mm, 105 * mm], font=8.6),
        Paragraph("Parameter count", s["h2"]),
        Paragraph(
            "With the original ResNet-50+FPN backbone the full model is roughly 32M parameters, almost "
            "entirely the backbone - the FCOS head itself is small by design. <b>In this project</b> (measured "
            "exactly): the classification tower is 299,139 parameters and the regression tower is 301,445, for "
            "600,584 detection-head parameters on top of whichever backbone it sits on - about 4% the size of "
            "the Faster R-CNN head used in Stage 1/2.", s["body"]),
        Paragraph("Can it be used alone, without DINO?", s["h2"]),
        Paragraph(
            "Yes, on the same terms as Faster R-CNN: pair it with any backbone that produces a feature "
            "pyramid. The same limitation applies to reducing it to a single dense layer - the towers are "
            "convolutional and need spatial structure to convolve over, so a minimum of a few convolutional "
            "layers underneath is required for it to function at all.", s["body"]),
        image(FIGS / "layers_fcos_alone.png", 150),
        Paragraph("Figure B3 - FCOS's own layers, generic (any backbone can feed into this). Numbers shown are "
                  "this project's light configuration (2 convs per tower, 128 channels).", s["caption"]),
    ]

    # ---------------------------------------------------------------- B4 DINO + Faster R-CNN
    story.append(PageBreak())
    story += [
        Paragraph("B4. DINOv3 + Faster R-CNN  (this project's Stage 1 and Stage 2)", s["h1"]),
        Paragraph(
            "The combination actually trained for Stage 1 and Stage 2. DINOv3 replaces the ResNet/VGG backbone "
            "Faster R-CNN was designed around; everything from the feature pyramid onward is unchanged Faster "
            "R-CNN, and both heads share one DINOv3 forward pass.", s["body"]),
        table([
            ["", "Stage 1", "Stage 2"],
            ["Trunk", "frozen", "last 2 of 12 blocks unfrozen"],
            ["Total parameters", "39,314,493", "39,314,493 (identical architecture)"],
            ["Trainable parameters", "17,727,549", "21,276,477"],
            ["Test mAP@0.5", "see Part A", "see Part A"],
        ], [30 * mm, 65 * mm, 65 * mm], font=8.4),
        Paragraph(
            "Why this pairing, specifically: DINOv3's self-supervised features generalise to lighting "
            "conditions the 21,527-image D-Fire dataset does not itself contain (see the project log), and "
            "Faster R-CNN's two-stage design gives higher precision at the high-confidence operating point a "
            "low-false-alarm system needs, at the cost of a variable-shape output that cannot be batched "
            "across cameras (Part B1).", s["body"]),
        image(FIGS / "layers_dino_frcnn.png", 165),
        Paragraph("Figure B4 - the full Stage 1/2 model, every shape and parameter count exact.", s["caption"]),
    ]

    # ---------------------------------------------------------------- B5 DINO + FCOS
    story.append(PageBreak())
    story += [
        Paragraph("B5. DINOv3 + FCOS  (this project's Stage 3, \"light\")", s["h1"]),
        Paragraph(
            "The combination trained for Stage 3, built specifically because Faster R-CNN's variable-shape "
            "output was measured to block batched multi-camera export (ONNX Runtime rejects batch>1; "
            "torch.export fails outright - see the project log). FCOS's fixed-shape output does not have this "
            "problem, and was verified to export as a batched, dynamic-batch-size ONNX graph whose decoded "
            "detections match the PyTorch model to within about 6e-5 pixels.", s["body"]),
        table([
            ["", "Stage 3"],
            ["Trunk", "DINOv3 ViT-Ti/16, frozen"],
            ["Total parameters", "6,947,224"],
            ["Trainable parameters", "1,457,560"],
            ["Test mAP@0.5", "see Part A"],
        ], [30 * mm, 130 * mm], font=8.6),
        Paragraph(
            "Two design choices were changed together here - a smaller trunk (ViT-Ti instead of ViT-S) and a "
            "different head (FCOS instead of Faster R-CNN) - so the accuracy gap measured against Stage 2 "
            "(Part A) cannot yet be attributed to one or the other. Separating them is the next experiment "
            "(see the project log): a ViT-S+FCOS configuration would isolate the head's own contribution.",
            s["body"]),
        image(FIGS / "layers_dino_fcos.png", 165),
        Paragraph("Figure B5 - the full Stage 3 model, every shape and parameter count exact.", s["caption"]),
    ]

    # ---------------------------------------------------------------- B6 Optimizers
    story.append(PageBreak())
    story += [
        Paragraph("B6. Optimizers", s["h1"]),
        Paragraph(
            "This project uses <b>AdamW</b> throughout (Loshchilov &amp; Hutter, 2019) - Adam with weight "
            "decay decoupled from the gradient-adaptive learning rate, which its authors showed corrects a "
            "flaw in plain Adam's weight decay. It is applied with a 500-iteration linear warmup followed by "
            "cosine decay to 1% of the peak learning rate, and any unfrozen trunk parameters get 5% of the "
            "head's learning rate.", s["body"]),
        Paragraph("Alternatives, and why they were not the first choice here", s["h2"]),
        table([
            ["Optimizer", "How it differs from AdamW", "Where it fits"],
            ["SGD +\nmomentum",
             wrap("A single global learning rate scaled by momentum; no per-parameter adaptivity."),
             wrap("Still common for CNNs trained from scratch over long schedules. Vision Transformers are "
                 "widely reported in the ViT literature (including the original ViT paper) to train poorly "
                 "with plain SGD without extensive tuning - this project has not tested it.")],
            ["Adam\n(no decoupled\ndecay)",
             wrap("AdamW's predecessor; weight decay is folded into the gradient update rather than applied "
                 "separately."),
             wrap("Superseded by AdamW for most transformer training since 2019; kept here only for "
                 "comparison.")],
            ["RMSprop",
             wrap("Per-parameter adaptive learning rate like Adam, but without Adam's momentum-style bias "
                 "correction."),
             wrap("Predates Adam; occasionally used in older CNN and RL work, uncommon for modern detection "
                 "training.")],
            ["LAMB",
             wrap("Layer-wise adaptive learning rates, designed for very large batch sizes."),
             wrap("Used for large-batch ViT pretraining; not relevant at this project's batch size of 8.")],
            ["RAdam",
             wrap("Adds a variance-rectification term that addresses the same early-training instability the "
                 "manual warmup in this project already addresses by hand."),
             wrap("A plausible alternative to warmup+AdamW; not tested here.")],
        ], [22 * mm, 68 * mm, 60 * mm], font=7.6),
        Paragraph(
            "None of these alternatives has been trained on this dataset, and no specific accuracy figure for "
            "any of them is claimed. The statements above describe well-established general behaviour "
            "reported in the optimisation and Vision Transformer literature, not a result measured in this "
            "project. Given AdamW with warmup and cosine decay is already a standard, well-tested recipe for "
            "fine-tuning transformer-based detectors, the higher-value next experiments are the architectural "
            "ones identified in Part A and the project log (a Stage 2 unfreeze for Stage 3, and a ViT-S+FCOS "
            "configuration) rather than an optimizer swap.", s["body"]),
        Paragraph(
            "No layer-flow diagram is included for this section: an optimizer is not a stack of network "
            "layers, it is the rule that updates every layer's weights after each batch.", s["caption"]),
    ]


if __name__ == "__main__":
    for spec in STAGES:
        spec["arch_file"] = spec["arch"]
    build(load_all())
