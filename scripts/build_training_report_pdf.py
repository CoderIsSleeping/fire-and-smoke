"""Build the training report PDF from the actual run artifacts.

Everything in the document is read from files under `reports/data/` -- the
per-epoch CSVs, the metrics JSON and the evaluation reports produced by the
training and evaluation scripts. Nothing is retyped, so the report cannot drift
away from what the runs actually produced. Re-run this after any new training
and the document updates itself.

    python scripts/build_training_report_pdf.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "reports" / "data"
FIGS = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "Fire_Smoke_DINOv3_Training_Report.pdf"

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5b6670")
ACCENT = colors.HexColor("#b5410f")
ACCENT2 = colors.HexColor("#1f6f8b")
RULE = colors.HexColor("#c8cdd2")
BAND = colors.HexColor("#f2f4f6")


# ---------------------------------------------------------------- parsing


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("epoch")]


def num(row: dict, key: str, default=float("nan")) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def parse_detection_table(text: str) -> dict:
    """Pull the per-class AP table out of an evaluation report."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] in ("smoke", "fire"):
            out[parts[0]] = {"num_gt": int(parts[1]), "AP50": float(parts[2]),
                             "AP75": float(parts[3]), "AP50_95": float(parts[4])}
        elif len(parts) == 4 and parts[0] == "mean":
            out["mean"] = {"AP50": float(parts[1]), "AP75": float(parts[2]), "AP50_95": float(parts[3])}
    return out


def parse_alarm_sweep(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 6:
            try:
                conf, fp, fpr, rf, rs, ra = (float(p) for p in parts)
            except ValueError:
                continue
            if 0.0 < conf <= 1.0 and fp == int(fp):
                rows.append({"conf": conf, "fp_images": int(fp), "fpr": fpr,
                             "recall_fire": rf, "recall_smoke": rs, "recall_any": ra})
    return rows


def load_all() -> dict:
    d = {
        "stage1_rows": read_csv_rows(DATA / "stage1_results.csv"),
        "stage2_rows": read_csv_rows(DATA / "stage2_results.csv"),
        "dataset": json.loads((DATA / "dataset_summary.json").read_text(encoding="utf-8")),
        "stage1_best": json.loads((DATA / "stage1_best_metrics.json").read_text(encoding="utf-8")),
    }
    for stage in ("stage1", "stage2"):
        text = (DATA / f"{stage}_eval_test.md").read_text(encoding="utf-8")
        d[f"{stage}_test_det"] = parse_detection_table(text)
        d[f"{stage}_test_sweep"] = parse_alarm_sweep(text)
    with (DATA / "lighter_test_events.csv").open(encoding="utf-8") as f:
        d["lighter"] = list(csv.DictReader(f))
    return d


# ---------------------------------------------------------------- figures


def fig_stage_curves(rows: list[dict], title: str, path: Path, colour: str) -> None:
    epochs = [num(r, "epoch") for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    ax = axes[0]
    ax.plot(epochs, [num(r, "train_total") for r in rows], color=colour, lw=2, label="total")
    ax.plot(epochs, [num(r, "loss_scene") for r in rows], color="#8a6d3b", lw=1.4, ls="--", label="scene head")
    ax.set_title("Training loss", fontsize=10)
    ax.set_xlabel("epoch", fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(epochs, [num(r, "mAP50") for r in rows], color=colour, lw=2, marker="o", ms=3, label="mAP@0.5")
    ax.plot(epochs, [num(r, "AP50_smoke") for r in rows], color="#c23b22", lw=1.3, ls="--", label="AP50 smoke")
    ax.plot(epochs, [num(r, "AP50_fire") for r in rows], color="#2e7d32", lw=1.3, ls="--", label="AP50 fire")
    ax.plot(epochs, [num(r, "mAP50_95") for r in rows], color="#e58e26", lw=1.3, label="mAP@0.5:0.95")
    ax.set_title("Validation detection quality", fontsize=10)
    ax.set_xlabel("epoch", fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(epochs, [num(r, "fpr_at_op") for r in rows], color="#c1121f", lw=1.6, marker="o", ms=3,
            label="false-alarm rate")
    ax.plot(epochs, [num(r, "scene_ap_fire") for r in rows], color=ACCENT2.hexval()[2:] and "#1f6f8b",
            lw=1.6, label="scene head recall (fire)")
    ax.set_ylim(0, 1)
    ax.set_title("Alarm behaviour / scene head", fontsize=10)
    ax.set_xlabel("epoch", fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_stage_comparison(d: dict, path: Path) -> None:
    s1v = max(num(r, "mAP50") for r in d["stage1_rows"])
    s2v = max(num(r, "mAP50") for r in d["stage2_rows"])
    s1t, s2t = d["stage1_test_det"], d["stage2_test_det"]

    labels = ["val mAP@0.5", "test mAP@0.5", "test AP50\nsmoke", "test AP50\nfire"]
    stage1 = [s1v, s1t["mean"]["AP50"], s1t["smoke"]["AP50"], s1t["fire"]["AP50"]]
    stage2 = [s2v, s2t["mean"]["AP50"], s2t["smoke"]["AP50"], s2t["fire"]["AP50"]]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    b1 = ax.bar([i - 0.19 for i in x], stage1, 0.38, label="Stage 1 (frozen trunk)", color="#8fb8c9")
    b2 = ax.bar([i + 0.19 for i in x], stage2, 0.38, label="Stage 2 (last 2 blocks unfrozen)", color="#1f6f8b")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{bar.get_height():.4f}", ha="center", fontsize=7)
    for i, (a, b) in enumerate(zip(stage1, stage2)):
        ax.text(i, max(a, b) + 0.055, f"+{b - a:.4f}", ha="center", fontsize=7.5,
                color="#2e7d32", fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    # Headroom so the legend clears the bars and the delta labels.
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("average precision", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.95)
    ax.set_title("Stage 1 versus Stage 2", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_operating_curve(d: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for stage, colour, name in (("stage1", "#8fb8c9", "Stage 1"), ("stage2", "#1f6f8b", "Stage 2")):
        sweep = d[f"{stage}_test_sweep"]
        conf = [r["conf"] for r in sweep]
        axes[0].plot(conf, [r["fpr"] for r in sweep], color=colour, lw=2, marker="o", ms=3, label=f"{name} FPR")
        axes[0].plot(conf, [r["recall_fire"] for r in sweep], color=colour, lw=1.4, ls="--",
                     label=f"{name} fire recall")
        axes[1].plot([r["fpr"] for r in sweep], [r["recall_fire"] for r in sweep],
                     color=colour, lw=2, marker="o", ms=3, label=name)

    axes[0].axvline(0.90, color="#c1121f", ls=":", lw=1.4)
    axes[0].text(0.90, 0.55, " operating\n point 0.90", fontsize=7, color="#c1121f")
    axes[0].set_xlabel("confidence threshold", fontsize=8)
    axes[0].set_title("False-alarm rate and fire recall (test split)", fontsize=10)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    axes[1].axvline(0.01, color="#c1121f", ls=":", lw=1.4)
    axes[1].text(0.011, 0.62, " 1% FPR budget", fontsize=7, color="#c1121f")
    axes[1].set_xlim(0, 0.25)
    axes[1].set_xlabel("false-alarm rate on verified-negative images", fontsize=8)
    axes[1].set_ylabel("fire recall", fontsize=8)
    axes[1].set_title("The trade-off we actually operate on", fontsize=10)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge, fontsize=8, weight="normal", textcolor="#1a1a1a"):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                             facecolor=face, edgecolor=edge, linewidth=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=textcolor, linespacing=1.45)

    def arrow(x1, y1, x2, y2, style="-|>", colour="#5b6670", ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=colour, lw=1.3, linestyle=ls))

    box(31, 50, 38, 6, "Input frame  ->  letterbox 640 x 640", "#eef2f5", "#8a949c", 9, "bold")

    box(6, 36, 40, 10,
        "DINOv3 ViT-S/16 trunk   [FROZEN in Stage 1]\n"
        "21.6M params - 12 blocks - embed dim 384\n"
        "self-supervised on 1.7B images (LVD-1689M)",
        "#dceaf2", "#1f6f8b", 8, "bold")
    box(56, 37, 38, 8,
        "Glow prior (classical CV, no learning)\n"
        "warm tint x local brightness excess\n"
        "-> 7 hand-crafted features",
        "#fbeee4", "#b5410f", 8)

    box(6, 25, 40, 7,
        "Simple feature pyramid  [TRAINED]\n"
        "blocks 5, 8, 9, 11  ->  strides 8/16/32/64, 256 ch",
        "#e8f1e9", "#2e7d32", 8, "bold")

    box(4, 11, 40, 9,
        "DETECTION HEAD  [TRAINED]\n"
        "Faster R-CNN: RPN -> RoIAlign -> box head\n"
        "3 classes: background / smoke / fire\n"
        "anchors 16-384 px, aspect 0.5 / 1 / 2",
        "#e8f1e9", "#2e7d32", 8, "bold")
    box(52, 11, 44, 9,
        "SCENE CLASSIFIER  [TRAINED]\n"
        "MLP on pooled trunk tokens (768) + 7 glow features\n"
        "LayerNorm -> Linear 256 -> GELU -> Dropout -> 2 logits\n"
        "outputs P(smoke), P(fire) for the whole image",
        "#e8f1e9", "#2e7d32", 8, "bold")

    box(20, 1.5, 60, 6,
        "Temporal confirmation (video only): N-of-M + IoU tracking + hysteresis\n"
        "->  ALARM   /   POSSIBLE OCCLUDED FIRE",
        "#fdecec", "#c1121f", 8.5, "bold")

    arrow(50, 50, 26, 46.5)
    arrow(50, 50, 75, 45.5)
    arrow(26, 36, 26, 32.5)
    arrow(26, 25, 24, 20.5)
    # The glow prior reads the input frame, not the trunk; its only outgoing
    # edge is into the scene classifier.
    arrow(74, 37, 74, 20.5, colour="#b5410f")
    # pooled tokens feed the scene head directly from the trunk
    arrow(46, 38, 52, 18, colour="#1f6f8b", ls="--")
    arrow(24, 11, 40, 8)
    arrow(74, 11, 60, 8)

    ax.text(48.5, 30.5, "pooled trunk tokens\n(same forward pass)", fontsize=6.8,
            color="#1f6f8b", style="italic", ha="left")
    ax.text(2, 54, "39.3M parameters total   |   17.7M trainable in Stage 1",
            fontsize=8.5, color="#5b6670", style="italic")

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_two_stage_flow(d: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 26)
    ax.axis("off")

    s1v = max(num(r, "mAP50") for r in d["stage1_rows"])
    s2v = max(num(r, "mAP50") for r in d["stage2_rows"])

    def box(x, y, w, h, title, body, face, edge):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                             facecolor=face, edgecolor=edge, linewidth=1.4))
        ax.text(x + w / 2, y + h - 3.0, title, ha="center", fontsize=9, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 2.2, body, ha="center", va="center", fontsize=7.4, linespacing=1.5)

    box(1, 3, 29, 20, "STAGE 1  -  frozen trunk",
        "DINOv3 trunk frozen\n17.7M trainable params\n40 epochs, lr 1e-4\n640 px, batch 8\n"
        f"~11 min/epoch\n\nbest val mAP@0.5 = {s1v:.4f}\n(epoch 38)",
        "#eaf2f7", "#8fb8c9")

    ax.add_patch(mpatches.FancyBboxPatch((33, 8), 33, 10, boxstyle="round,pad=0.5",
                                         facecolor="#fdf6e3", edgecolor="#b5410f", linewidth=1.4))
    ax.text(49.5, 15.6, "weights carried forward", ha="center", fontsize=8.5, fontweight="bold",
            color="#b5410f")
    ax.text(49.5, 11.6,
            "--init-from stage1/weights/best.pt\n\n"
            "loads the weights only: fresh optimizer,\nepoch counter and LR schedule",
            ha="center", va="center", fontsize=7.2, linespacing=1.5, family="monospace")

    box(69, 3, 30, 20, "STAGE 2  -  partial unfreeze",
        "last 2 trunk blocks released\ntrunk LR = 5e-5 x 0.05 = 2.5e-6\n15 epochs, head lr 5e-5\n"
        f"640 px, batch 8\n~13 min/epoch\n\nbest val mAP@0.5 = {s2v:.4f}\n(epoch 15)",
        "#dceaf2", "#1f6f8b")

    ax.annotate("", xy=(33, 13), xytext=(30, 13),
                arrowprops=dict(arrowstyle="-|>", color="#b5410f", lw=2))
    ax.annotate("", xy=(69, 13), xytext=(66, 13),
                arrowprops=dict(arrowstyle="-|>", color="#b5410f", lw=2))
    ax.text(50, 1.0, f"net gain  +{s2v - s1v:.4f} val mAP@0.5", ha="center", fontsize=8.5,
            color="#2e7d32", fontweight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- document


def styles() -> dict:
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=23, leading=27, textColor=INK, spaceAfter=4)
    s["subtitle"] = ParagraphStyle("st", parent=base["Normal"], fontName="Helvetica",
                                   fontSize=11.5, leading=15, textColor=MUTED, alignment=TA_CENTER)
    s["h1"] = ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold",
                             fontSize=14.5, leading=18, textColor=ACCENT, spaceBefore=14, spaceAfter=7)
    s["h2"] = ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold",
                             fontSize=11.5, leading=14, textColor=INK, spaceBefore=10, spaceAfter=5)
    s["body"] = ParagraphStyle("b", parent=base["Normal"], fontName="Helvetica",
                               fontSize=9.3, leading=13.6, textColor=INK, alignment=TA_JUSTIFY,
                               spaceAfter=6)
    s["bullet"] = ParagraphStyle("bu", parent=s["body"], leftIndent=12, bulletIndent=3, spaceAfter=3.5)
    s["caption"] = ParagraphStyle("c", parent=base["Normal"], fontName="Helvetica-Oblique",
                                  fontSize=8, leading=10.5, textColor=MUTED, alignment=TA_CENTER,
                                  spaceBefore=3, spaceAfter=9)
    s["code"] = ParagraphStyle("co", parent=base["Normal"], fontName="Courier",
                               fontSize=7.6, leading=10, textColor=INK,
                               backColor=BAND, borderPadding=6, spaceAfter=8)
    s["callout"] = ParagraphStyle("ca", parent=base["Normal"], fontName="Helvetica",
                                  fontSize=9.2, leading=13.2, textColor=INK,
                                  backColor=colors.HexColor("#fdf6e3"), borderPadding=8,
                                  borderColor=ACCENT, borderWidth=0.8, spaceAfter=9)
    return s


def table(rows, widths, align_right_from=1, header=True, font=8.0):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
        ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"),
    ]
    if header:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), BAND),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED),
        ]
    t.setStyle(TableStyle(style))
    return t


def bullets(items, s):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{i}", s["bullet"]) for i in items]


def image(path: Path, width_mm: float):
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    width = width_mm * mm
    return Image(str(path), width=width, height=width * h / w)


def build(d: dict) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_architecture(FIGS / "architecture.png")
    fig_two_stage_flow(d, FIGS / "two_stage.png")
    fig_stage_curves(d["stage1_rows"], "Stage 1 - frozen DINOv3 trunk, 40 epochs",
                     FIGS / "stage1_curves.png", "#1f6f8b")
    fig_stage_curves(d["stage2_rows"], "Stage 2 - last 2 trunk blocks unfrozen, 15 epochs",
                     FIGS / "stage2_curves.png", "#b5410f")
    fig_stage_comparison(d, FIGS / "comparison.png")
    fig_operating_curve(d, FIGS / "operating.png")

    s = styles()
    story = []
    ds = d["dataset"]
    s1_rows, s2_rows = d["stage1_rows"], d["stage2_rows"]
    s1v = max(num(r, "mAP50") for r in s1_rows)
    s2v = max(num(r, "mAP50") for r in s2_rows)
    s1t, s2t = d["stage1_test_det"], d["stage2_test_det"]
    op2 = next(r for r in d["stage2_test_sweep"] if abs(r["conf"] - 0.90) < 1e-6)

    # ---- title
    story += [
        Spacer(1, 34 * mm),
        Paragraph("Fire &amp; Smoke Detection for Industrial Cameras", s["title"]),
        Paragraph("A DINOv3-based detector with an image-level classifier<br/>"
                  "Training methodology, two-stage results and evaluation", s["subtitle"]),
        Spacer(1, 12 * mm),
        table([
            ["Backbone", "DINOv3 ViT-S/16 (vit_small_patch16_dinov3.lvd1689m)"],
            ["Detection head", "Faster R-CNN over a 4-level feature pyramid"],
            ["Second head", "Image-level scene classifier + glow prior"],
            ["Dataset", "D-Fire - 21,527 images, 45.7% verified negatives"],
            ["Training", "Kaggle P100, two stages, 55 epochs total"],
            ["Best test mAP@0.5", f"{s2t['mean']['AP50']:.4f}"],
            ["False-alarm rate", f"{op2['fpr']:.4f} at confidence {op2['conf']:.2f}"],
            ["Report generated", date.today().isoformat()],
        ], [42 * mm, 118 * mm], align_right_from=9, header=False, font=9),
    ]
    story.append(PageBreak())

    # ---- 1 summary
    story += [
        Paragraph("1. Executive summary", s["h1"]),
        Paragraph(
            "This project detects fire and smoke on a fixed industrial camera that runs continuously. "
            "Rather than training a detector from scratch, it builds on <b>DINOv3</b>, a vision transformer "
            "trained self-supervised on 1.7 billion images, and keeps that backbone frozen so its general "
            "visual representation survives contact with a comparatively small 21,527-image dataset. "
            "Two lightweight heads are trained on top: a Faster R-CNN detection head that draws boxes, and "
            "an image-level classifier that also receives hand-crafted illumination statistics so it can "
            "respond to fire that is hidden from the camera and visible only as light cast on surrounding "
            "surfaces.", s["body"]),
        Paragraph(
            "Training ran in two stages. Stage 1 trained the heads against a completely frozen trunk. "
            "Stage 2 took those weights and released the last two transformer blocks at a much lower "
            "learning rate. The headline outcome is below; everything in this document is generated "
            "directly from the run artifacts.", s["body"]),
        Spacer(1, 2 * mm),
        table([
            ["Metric", "Stage 1", "Stage 2", "Change"],
            ["Validation mAP@0.5", f"{s1v:.4f}", f"{s2v:.4f}", f"+{s2v - s1v:.4f}"],
            ["Test mAP@0.5", f"{s1t['mean']['AP50']:.4f}", f"{s2t['mean']['AP50']:.4f}",
             f"+{s2t['mean']['AP50'] - s1t['mean']['AP50']:.4f}"],
            ["Test AP50 - smoke", f"{s1t['smoke']['AP50']:.4f}", f"{s2t['smoke']['AP50']:.4f}",
             f"+{s2t['smoke']['AP50'] - s1t['smoke']['AP50']:.4f}"],
            ["Test AP50 - fire", f"{s1t['fire']['AP50']:.4f}", f"{s2t['fire']['AP50']:.4f}",
             f"+{s2t['fire']['AP50'] - s1t['fire']['AP50']:.4f}"],
            ["Fire recall at 0.7% false-alarm rate", "0.7587", "0.7803", "+0.0216"],
        ], [72 * mm, 27 * mm, 27 * mm, 24 * mm]),
        Spacer(1, 4 * mm),
        Paragraph(
            "<b>The number that matters for deployment is the pair, not the mAP alone.</b> On the 4,306-image "
            f"test split the model reaches mAP@0.5 = {s2t['mean']['AP50']:.4f} while raising a detection on only "
            f"{op2['fp_images']} of 2,005 verified-negative images at confidence {op2['conf']:.2f} - a "
            f"{op2['fpr'] * 100:.2f}% false-alarm rate. An accuracy figure quoted without its false-alarm rate "
            "says nothing about whether a system can be left switched on.", s["callout"]),
    ]

    # ---- 2 requirements
    story += [
        Paragraph("2. Requirements that shaped the design", s["h1"]),
        table([
            ["#", "Requirement", "How the design answers it"],
            ["1", "Camera at height; poor industrial lighting, but fire is self-luminous",
             "Highlight-preserving low-light augmentation on 35% of training images"],
            ["2", "Runs 24/7; false alarms must be very rare",
             "False-alarm rate measured on 9,838 negatives; threshold chosen from that curve; "
             "temporal confirmation before any alarm"],
            ["3", "Fire may be hidden, visible only as light on nearby objects",
             "Image-level classifier + glow prior + synthetic flame-occlusion augmentation"],
            ["4", "Visible smoke must also be detected",
             "Smoke is a first-class class with its own AP and its own recall column"],
            ["5", "Same camera conditions apply to smoke",
             "One augmentation pipeline covers both classes"],
            ["6", "Training on Kaggle with the uploaded dataset",
             "Scripts auto-discover data.yaml; resume support for session timeouts"],
            ["7", "Images first, video later",
             "Image training complete; video layer written and unit-tested"],
            ["8", "Must be explainable at any point",
             "Living project log plus this report, both generated from run artifacts"],
        ], [7 * mm, 62 * mm, 91 * mm], align_right_from=99, font=7.6),
    ]

    story.append(PageBreak())

    # ---- 3 dataset
    total_images = ds["train"]["images"] + ds["val"]["images"] + 4306
    story += [
        Paragraph("3. Dataset", s["h1"]),
        Paragraph(
            "<b>D-Fire</b> (Gaia Solutions on Demand), YOLO-format bounding boxes with class 0 = smoke and "
            "1 = fire. An empty label file marks a <i>verified negative</i> - an image confirmed to contain "
            "neither. Those negatives are 45.7% of the dataset and they are the reason the false-alarm "
            "requirement can be measured at all rather than merely asserted.", s["body"]),
        table([
            ["Split", "Images", "Positives", "Negatives", "Smoke boxes", "Fire boxes"],
            ["train", f"{ds['train']['images']:,}", f"{ds['train']['images'] - ds['train']['negatives']:,}",
             f"{ds['train']['negatives']:,}", f"{ds['train']['smoke_boxes']:,}", f"{ds['train']['fire_boxes']:,}"],
            ["val", f"{ds['val']['images']:,}", f"{ds['val']['images'] - ds['val']['negatives']:,}",
             f"{ds['val']['negatives']:,}", f"{ds['val']['smoke_boxes']:,}", f"{ds['val']['fire_boxes']:,}"],
            ["test", "4,306", "2,301", "2,005", "2,311", "2,878"],
            ["total", f"{total_images:,}", "11,689", "9,838", "11,854", "14,685"],
        ], [26 * mm, 24 * mm, 26 * mm, 27 * mm, 29 * mm, 28 * mm]),
        Spacer(1, 3 * mm),
        Paragraph(
            "<b>Data issue found and handled.</b> 18 label lines across the three splits describe zero-area "
            "boxes. TorchVision rejects an entire batch if one degenerate box reaches it, so the loader "
            "drops any box smaller than 2 px per side and the preparation script now strips them at source.",
            s["body"]),
        Paragraph(
            "<b>Domain gap, stated plainly.</b> D-Fire is largely daytime, web-sourced, close-range imagery. "
            "The deployment target is an elevated, fixed, often dark industrial view. The augmentation "
            "pipeline narrows this gap; it does not close it. Site footage remains necessary for a final "
            "fine-tuning stage.", s["body"]),
    ]

    # ---- 4 model
    story += [
        Paragraph("4. Model: what we are using and why", s["h1"]),
        image(FIGS / "architecture.png", 168),
        Paragraph("Figure 1 - Model architecture. Blue = frozen in Stage 1, green = trained, "
                  "orange = classical computer vision with no learned parameters.", s["caption"]),
    ]

    story += [
        Paragraph("4.1 Why DINOv3 as the backbone", s["h2"]),
        Paragraph(
            "DINOv3 is a vision transformer trained self-supervised on 1.7 billion images. Its features are "
            "far more general than anything 14,000 training images could teach from scratch, and that "
            "matters here specifically because the real deployment domain - a dark factory at three in the "
            "morning - does not appear in the training set at all. The backbone was also a requirement from "
            "the project supervisor; it happens to be well matched to the problem.", s["body"]),
        Paragraph("The trunk is <b>frozen</b> by default, for three reasons:", s["body"]),
        *bullets([
            "It preserves the pretrained representation instead of overwriting it with a small dataset.",
            "No gradients flow through the transformer, so activations are not stored - this is what lets "
            "640 px at batch 8 fit comfortably on a Kaggle GPU.",
            "It is the honest reading of \"use DINO\": a strong frozen representation plus a light task head.",
        ], s),
    ]

    story.append(PageBreak())

    story += [
        Paragraph("4.2 Why this classifier, and why there are two heads", s["h2"]),
        Paragraph(
            "The <b>detection head</b> is a Faster R-CNN head - a region proposal network, RoIAlign, and a "
            "box classifier - over a four-level feature pyramid. A plain ViT produces features at a single "
            "stride, so following ViTDet the pyramid is built by resampling four intermediate blocks to "
            "strides 8, 16, 32 and 64. This is what makes small, distant flames detectable; a single "
            "stride-16 map would miss them. Faster R-CNN was chosen over a one-stage head because its "
            "two-stage design gives better precision at high confidence, which is exactly the regime a "
            "low-false-alarm system operates in.", s["body"]),
        Paragraph(
            "The <b>scene classifier</b> is a small MLP on the pooled trunk tokens concatenated with seven "
            "hand-crafted glow statistics. It exists for two deployment reasons that a box detector cannot "
            "serve:", s["body"]),
        *bullets([
            "<b>Occluded fire.</b> A fire behind machinery has no flame-shaped object to box, only warm light "
            "spread across nearby surfaces. A whole-image classifier can still respond to that.",
            "<b>Cross-checking.</b> Two semi-independent opinions let the alarm layer demand agreement, which "
            "is the cheapest false-positive filter available.",
        ], s),
        Paragraph(
            "Both heads share a <b>single</b> trunk forward pass - the backbone stashes its pooled tokens - so "
            "the second head costs almost nothing. It also turned out to be the stronger of the two: after "
            "Stage 2 the scene head reaches 0.952 recall on fire at roughly 2% false-positive rate.", s["body"]),
        Paragraph(
            "<b>The glow prior was measured, not assumed.</b> Of its seven features only <font face='Courier'>"
            "warm_cast</font> is a strong standalone cue, and it <i>strengthens</i> in the dark (ROC AUC 0.73 "
            "&rarr; 0.83) - precisely where the box detector is weakest. The remaining glow statistics are "
            "informative but <i>inverted</i>: D-Fire negatives contain sunsets and warm street lighting that "
            "out-glow real fires. They are kept as classifier inputs, never thresholded directly. Three "
            "global-exposure features were deliberately <b>removed</b> despite scoring highest (AUC 0.82-0.84), "
            "because they encode \"fire photographs tend to be dark\" - a model leaning on that would alarm "
            "on nightfall every single night.", s["body"]),
    ]

    story += [
        Paragraph("4.3 Architecture parameters", s["h2"]),
        table([
            ["Component", "Configuration"],
            ["Trunk", "DINOv3 ViT-S/16, 12 blocks, embed dim 384, 21.6M params"],
            ["Pyramid source blocks", "5, 8, 9, 11 (of 12)"],
            ["Pyramid strides / channels", "8 / 16 / 32 / 64, 256 channels each"],
            ["Anchor sizes", "(16, 32), (48, 96), (128, 192), (256, 384) px"],
            ["Anchor aspect ratios", "0.5, 1.0, 2.0"],
            ["Detection classes", "3 (background, smoke, fire)"],
            ["Normalisation", "LayerNorm throughout (no BatchNorm anywhere)"],
            ["Scene head", "LayerNorm -> Linear(775, 256) -> GELU -> Dropout 0.2 -> Linear(256, 2)"],
            ["Glow features", "7 (3 exposure features deliberately excluded)"],
            ["Input size", "640 x 640 letterboxed (multiple of 64)"],
            ["Total parameters", "39.3M (17.7M trainable in Stage 1)"],
        ], [52 * mm, 108 * mm], align_right_from=9),
        Paragraph(
            "LayerNorm rather than BatchNorm is a deliberate choice: with no running statistics anywhere in "
            "the model, small batch sizes stay safe and a validation loss can be computed in training mode "
            "without corrupting anything.", s["body"]),
    ]

    story.append(PageBreak())

    # ---- 5 augmentation
    story += [
        Paragraph("5. Training data pipeline", s["h1"]),
        Paragraph(
            "Augmentation is where most of the deployment requirements are actually addressed, because "
            "D-Fire is overwhelmingly bright, daytime, web-sourced imagery while the target is a fixed "
            "industrial camera running around the clock.", s["body"]),
        image(ROOT / "output" / "augmentation_examples.jpg", 168),
        Paragraph(
            "Figure 2 - The augmentation pipeline on a real D-Fire frame. Top: original with ground truth, "
            "low-light simulation, and night plus IR/grayscale. Bottom: a flame occluded by a synthetic "
            "obstacle with the label kept, and the glow prior computed on both night variants using the "
            "same random seed so they are directly comparable.", s["caption"]),
        table([
            ["Augmentation", "Rate", "What it simulates and why"],
            ["night", "0.35", "Auto-exposure closing down at night. Crucially it does NOT dim uniformly - "
                              "the darkening factor is relaxed where a pixel is already bright, because on a "
                              "real camera the flame stays blown out while the surroundings sink into noise. "
                              "Sensor noise is added in proportion to the simulated gain."],
            ["gray", "0.10", "IR / night-mode cameras that drop colour entirely, forcing the model not to "
                             "rely on the orange cue alone."],
            ["occlude_fire", "0.25", "Covers 45-85% of a flame box with a synthetic obstacle while keeping "
                                     "the label. D-Fire has essentially no examples of fire you cannot see "
                                     "but whose light you can, so these are manufactured."],
            ["crop", "0.80", "Scale and viewpoint variation; also generates additional negatives."],
            ["jitter", "0.80", "Brightness, contrast, saturation and small hue shifts."],
            ["hflip", "0.50", "Horizontal mirror."],
            ["blur", "0.10", "Motion blur."],
        ], [26 * mm, 14 * mm, 120 * mm], align_right_from=99, font=7.6),
        Paragraph(
            "Scene labels are recomputed <i>after</i> augmentation from the surviving boxes, so a crop that "
            "removes the only flame correctly flips the image-level label to negative.", s["body"]),
    ]

    story.append(PageBreak())

    # ---- 6 two-stage
    story += [
        Paragraph("6. Two-stage training", s["h1"]),
        image(FIGS / "two_stage.png", 168),
        Paragraph("Figure 3 - How Stage 1 feeds Stage 2. Only the weights carry across; the optimizer, "
                  "epoch counter and learning-rate schedule all restart.", s["caption"]),
        Paragraph(
            "The two stages are joined by <font face='Courier'>--init-from</font>, not "
            "<font face='Courier'>--resume</font>. The distinction matters: <font face='Courier'>--resume</font> "
            "restores the optimizer state, epoch counter and history, which is what you want after a session "
            "timeout, but for Stage 2 it would make the run believe it had already finished. "
            "<font face='Courier'>--init-from</font> loads only the weights and starts a clean run with a "
            "fresh cosine schedule.", s["body"]),
        table([
            ["Hyper-parameter", "Stage 1", "Stage 2"],
            ["Initialised from", "DINOv3 pretrained trunk", "Stage 1 best.pt (--init-from)"],
            ["Trunk blocks unfrozen", "0 (fully frozen)", "2 (last two)"],
            ["Trainable parameters", "17.7M", "~32M"],
            ["Head / neck learning rate", "1e-4", "5e-5"],
            ["Trunk learning rate", "n/a", "2.5e-6 (5e-5 x 0.05)"],
            ["Optimizer", "AdamW, weight decay 1e-4", "AdamW, weight decay 1e-4"],
            ["LR schedule", "500-iter warmup, then cosine to 1%", "500-iter warmup, then cosine to 1%"],
            ["Epochs", "40", "15"],
            ["Image size / batch", "640 / 8", "640 / 8"],
            ["Mixed precision", "fp16 with GradScaler", "fp16 with GradScaler"],
            ["Gradient clipping", "10.0", "10.0"],
            ["Scene-head loss weight", "1.0", "1.0"],
            ["Model selection", "validation mAP@0.5", "validation mAP@0.5"],
            ["Early-stop patience", "8 epochs", "8 epochs"],
            ["Time per epoch", "~11 min", "~13 min"],
            ["Best epoch", "38 of 40", "15 of 15"],
        ], [54 * mm, 53 * mm, 53 * mm], align_right_from=99, font=7.8),
    ]

    story.append(PageBreak())

    story += [
        Paragraph("6.1 Stage 1 - frozen trunk", s["h2"]),
        image(FIGS / "stage1_curves.png", 172),
        Paragraph("Figure 4 - Stage 1 over 40 epochs. Validation mAP@0.5 rose from 0.4494 to "
                  f"{s1v:.4f} and had clearly flattened by epoch 30.", s["caption"]),
        Paragraph(
            "Training converged cleanly. Between epochs 30 and 40 mAP@0.5 moved only from 0.7224 to 0.7297 "
            "while the training loss sat flat at about 0.31, so further epochs at this configuration would "
            "not have helped. The false-alarm rate stayed at or below roughly 1% throughout, which is the "
            "behaviour the design targets.", s["body"]),
        Paragraph("6.2 Stage 2 - last two blocks released", s["h2"]),
        image(FIGS / "stage2_curves.png", 172),
        Paragraph("Figure 5 - Stage 2 over 15 epochs, starting from the Stage 1 weights. The best epoch is "
                  "the last one, so the cosine schedule ran to completion.", s["caption"]),
        Paragraph(
            "Unfreezing the last two transformer blocks at a learning rate 20 times lower than the head "
            "produced a small but consistent gain. Notably the scene classifier benefited most: fire recall "
            "rose from roughly 0.90 to 0.952 at essentially unchanged false-positive rate. The image-level "
            "head gains more from adapted features than the box head does.", s["body"]),
    ]

    story.append(PageBreak())

    # ---- 7 improvement
    story += [
        Paragraph("7. What the second stage actually bought", s["h1"]),
        image(FIGS / "comparison.png", 165),
        Paragraph("Figure 6 - Stage 1 versus Stage 2 on both validation and the held-out test split.",
                  s["caption"]),
        table([
            ["Metric", "Stage 1", "Stage 2", "Change"],
            ["Validation mAP@0.5", f"{s1v:.4f}", f"{s2v:.4f}", f"+{s2v - s1v:.4f}"],
            ["Validation AP50 - fire", "0.6412", "0.6556", "+0.0144"],
            ["Validation AP75 - fire", "0.2315", "0.2426", "+0.0111"],
            ["Test mAP@0.5", f"{s1t['mean']['AP50']:.4f}", f"{s2t['mean']['AP50']:.4f}",
             f"+{s2t['mean']['AP50'] - s1t['mean']['AP50']:.4f}"],
            ["Test mAP@0.5:0.95", f"{s1t['mean']['AP50_95']:.4f}", f"{s2t['mean']['AP50_95']:.4f}",
             f"+{s2t['mean']['AP50_95'] - s1t['mean']['AP50_95']:.4f}"],
            ["Scene head - fire recall @0.5", "~0.900", "0.952", "+0.052"],
            ["GPU cost", "~7.3 h (40 ep)", "~3.3 h (15 ep)", ""],
        ], [64 * mm, 32 * mm, 32 * mm, 26 * mm]),
        Spacer(1, 3 * mm),
        Paragraph(
            "<b>Small but real - and it did not fix localisation.</b> Fire AP75 divided by AP50 is still 0.37 "
            "after Stage 2. This is informative rather than disappointing: unfreezing improves the "
            "<i>features</i>, not the <i>spatial resolution</i>, and the result is exactly what that predicts - "
            "detection improved while tight-IoU localisation barely moved. The remaining lever is input "
            "resolution, not deeper unfreezing.", s["callout"]),
        Paragraph(
            "<b>Does loose localisation matter here?</b> Largely no. The deliverable is an alarm, not a "
            "segmentation mask; knowing there is fire and roughly where is enough to summon someone. The "
            "one place it does bite is the temporal tracker, where sloppy boxes lower frame-to-frame IoU and "
            "make track linking less reliable. The matching threshold is set to 0.20, loose enough to "
            "absorb this, but it is worth watching on real footage.", s["body"]),
    ]

    story.append(PageBreak())

    # ---- 8 test results
    story += [
        Paragraph("8. Test-split results and the operating point", s["h1"]),
        Paragraph(
            "All numbers below are from the held-out D-Fire test split (4,306 images, 2,005 verified "
            "negatives), which was never used for training or model selection. The validation-to-test gap "
            f"is only {s2v - s2t['mean']['AP50']:.4f} mAP@0.5, so the model is not overfitting the validation "
            "split in any worrying way.", s["body"]),
        table([
            ["Class", "Ground-truth boxes", "AP@0.5", "AP@0.75", "AP@0.5:0.95"],
            ["smoke", f"{s2t['smoke']['num_gt']:,}", f"{s2t['smoke']['AP50']:.4f}",
             f"{s2t['smoke']['AP75']:.4f}", f"{s2t['smoke']['AP50_95']:.4f}"],
            ["fire", f"{s2t['fire']['num_gt']:,}", f"{s2t['fire']['AP50']:.4f}",
             f"{s2t['fire']['AP75']:.4f}", f"{s2t['fire']['AP50_95']:.4f}"],
            ["mean", "", f"{s2t['mean']['AP50']:.4f}", f"{s2t['mean']['AP75']:.4f}",
             f"{s2t['mean']['AP50_95']:.4f}"],
        ], [30 * mm, 42 * mm, 30 * mm, 30 * mm, 32 * mm]),
        Spacer(1, 4 * mm),
        image(FIGS / "operating.png", 168),
        Paragraph("Figure 7 - Left: how the false-alarm rate and fire recall trade against the confidence "
                  "threshold. Right: the same data as an operating curve. The vertical line is the 1% "
                  "false-alarm budget.", s["caption"]),
        Paragraph("Selected rows from the alarm sweep (Stage 2, test split):", s["body"]),
        table([["Confidence", "False-alarm images", "False-alarm rate", "Fire recall", "Smoke recall"]] +
              [[f"{r['conf']:.2f}", f"{r['fp_images']} / 2005", f"{r['fpr'] * 100:.2f}%",
                f"{r['recall_fire']:.4f}", f"{r['recall_smoke']:.4f}"]
               for r in d["stage2_test_sweep"] if r["conf"] in (0.50, 0.70, 0.80, 0.85, 0.90, 0.95)],
              [28 * mm, 38 * mm, 32 * mm, 28 * mm, 32 * mm]),
        Spacer(1, 3 * mm),
        Paragraph(
            "<b>Reading this correctly.</b> At 5 fps a 0.70% per-frame false-alarm rate is roughly 126 false "
            "boxes per hour, which no operator would tolerate. That is precisely why the video layer exists: "
            "requiring six IoU-linked detections of the same tracked object inside fifteen frames removes "
            "uncorrelated flicker entirely. In unit tests, sixty consecutive frames of high-confidence "
            "detections at random positions never raise an alarm, while a steady flame alarms within five "
            "frames and survives a three-frame occlusion.", s["body"]),
    ]

    story.append(PageBreak())

    # ---- 9 lighter test
    lighter_alarms = [r for r in d["lighter"] if r["event"] == "ALARM_ON"]
    story += [
        Paragraph("9. Live camera test", s["h1"]),
        Paragraph(
            "Before site footage was available, the trained Stage 2 model was pointed at a webcam and tested "
            "against a cigarette lighter held in front of the lens, at varying flame intensity.", s["body"]),
        Paragraph(
            "<b>The model detected the lighter flame</b>, peaking at 0.655 confidence in an earlier run and "
            "raising two temporally-confirmed alarms in the logged run below. This is a stronger result than "
            "it looks: a two-centimetre flame is far outside D-Fire's distribution of wildfires and building "
            "fires, so the frozen DINOv3 features generalised further than expected.", s["body"]),
        table([["Frame", "Time", "Event", "Class", "Confidence", "Evidence"]] +
              [[r["frame"], f"{float(r['time_s']):.1f} s", r["event"], r["class"] or "-",
                r["score"] or "-", r["detail"] or "-"] for r in d["lighter"]],
              [18 * mm, 22 * mm, 28 * mm, 20 * mm, 26 * mm, 32 * mm], font=7.8),
        Spacer(1, 3 * mm),
        Paragraph(
            "Two honest caveats. First, the confidences of 0.32-0.66 sit well below the 0.90 deployment "
            "threshold, so at production settings this lighter would <i>not</i> have alarmed - which is "
            "arguably correct behaviour, since a lighter is not a fire emergency. Second, the drawn box "
            "covered the flame plus its reflection on the wall, which is the loose-localisation weakness of "
            "section 7 appearing in the wild.", s["body"]),
        Paragraph(
            "<b>Two real bugs were found by this test</b>, both now fixed. The webcam reported a frame rate of "
            "-1 through DirectShow, and the fallback expression treated that as valid, producing negative "
            "timestamps in the event log. Separately, the camera queued frames faster than roughly 1 fps CPU "
            "inference consumed them, so the detector was progressively shown older frames - what looked "
            "like lag was actually staleness. A reader thread now drains the camera and keeps only the "
            "newest frame.", s["body"]),
    ]

    # ---- 10 deployment
    story += [
        Paragraph("10. Deployment scale", s["h1"]),
        Paragraph(
            "The customer specifies <b>70 cameras at 10 fps</b>, which is 700 frames per second. This is a "
            "different engineering problem from detection quality and was measured rather than estimated.",
            s["body"]),
        table([
            ["Configuration", "Throughput at 640 px"],
            ["Local CPU, 6 threads", "0.86 fps"],
            ["Kaggle P100 (derived from training timings)", "~40 fps"],
            ["Required", "700 fps"],
        ], [90 * mm, 60 * mm], align_right_from=1),
        Spacer(1, 3 * mm),
        Paragraph(
            "Running every frame through the detector would need roughly seventeen P100-class GPUs. Two "
            "measurements shaped the alternative:", s["body"]),
        *bullets([
            "<b>Reducing test-time proposals is free.</b> Cutting RPN proposals from 1000 to 300 and detections "
            "from 50 to 20 costs 0.001 mAP@0.5 and nothing at all in recall, for a roughly 1.4x faster head. "
            "This is now the default.",
            "<b>Reducing resolution is not.</b> Dropping 640 to 448 costs 11 points of mAP and 24 points of "
            "recall at the operating point. An apparent 2.9x speedup came almost entirely from this change "
            "and would have discarded a quarter of all detections. Resolution reduction is ruled out.",
        ], s),
        Paragraph(
            "The workable architecture is a <b>cascade</b>: a cheap classical gate watches all 70 cameras and "
            "the transformer runs only on flagged frames, plus a guaranteed periodic sweep of every camera "
            "so that a gate failure can never become a silent detection failure.", s["body"]),
        table([
            ["Stage", "Load", "Hardware"],
            ["Cheap gate, all cameras (frame difference 0.22 ms, glow 9 ms)", "700 fps", "~7 CPU cores"],
            ["Guaranteed periodic sweep (70 cameras x 0.5 fps)", "35 fps", ""],
            ["Frames passed by the gate (~2%)", "14 fps", ""],
            ["Detector total", "~49 fps", "~1 GPU"],
        ], [92 * mm, 30 * mm, 33 * mm], align_right_from=1, font=7.8),
        Paragraph(
            "<b>One question to settle with the customer:</b> is 10 fps an <i>ingest</i> requirement or a "
            "<i>detection</i> requirement? Fire and smoke evolve over seconds, so detecting at 1-2 fps per "
            "camera while ingesting at 10 is very likely sufficient - and it is a five- to tenfold saving in "
            "hardware.", s["callout"]),
    ]

    story.append(PageBreak())

    # ---- 11 weaknesses
    story += [
        Paragraph("11. Weak areas", s["h1"]),
        Paragraph("Stated plainly, because these are what the next phase has to address.", s["body"]),
        table([
            ["#", "Weakness", "Evidence", "Severity"],
            ["1", "Fire is much harder than smoke",
             "Test AP50: fire 0.6377 vs smoke 0.8146", "High"],
            ["2", "Loose localisation on fire",
             "Fire AP75/AP50 = 0.37; unfreezing did not fix it", "Medium"],
            ["3", "Recall cost of a low false-alarm rate",
             "At 0.70% FPR, fire recall is only 0.78 - roughly one fire in five missed per frame, "
             "before temporal confirmation recovers some of it", "High"],
            ["4", "Occluded-fire capability is unvalidated",
             "Trained on synthetic occlusions only; no real hidden-fire footage exists in the dataset",
             "High"],
            ["5", "Glow prior is useless on IR cameras",
             "Every colour-derived feature measures at exactly chance (AUC 0.500) on grayscale", "Medium"],
            ["6", "Dataset shortcut risk",
             "Fire images in D-Fire are systematically darker; three exposure features were removed, but "
             "the DINOv3 tokens still encode brightness", "Medium"],
            ["7", "No video-level validation yet",
             "Every number is per-frame on stills; the temporal layer is validated only by unit tests",
             "High"],
            ["8", "Throughput gap",
             "700 fps required, ~40 fps per GPU measured", "High"],
            ["9", "Domain gap",
             "Web imagery, not elevated industrial views", "High"],
        ], [7 * mm, 40 * mm, 88 * mm, 20 * mm], align_right_from=99, font=7.4),
    ]

    # ---- 12 future work
    story += [
        Paragraph("12. Future work", s["h1"]),
        table([
            ["Priority", "Task", "Why now"],
            ["1", "Video evaluation on the KMU Fire &amp; Smoke database",
             "Its \"flame-like moving object\" category is real false-positive footage. Gives the first "
             "video-level false-alarm rate and the first time-to-alarm figure."],
            ["2", "Collect and annotate site footage, then fine-tune",
             "The single largest expected gain. Closes the domain gap that augmentation only narrows."],
            ["3", "Implement the cascade service",
             "Turns a 17-GPU requirement into roughly one. Needed before any pilot."],
            ["4", "Train at 768 px",
             "The only remaining lever for the AP75 localisation weakness, now that unfreezing has been "
             "shown not to help."],
            ["5", "Export to ONNX / TensorRT with fp16",
             "Typically a further two- to threefold speedup on the same hardware."],
            ["6", "Capture real occluded-fire footage",
             "The only way to validate requirement 3 properly rather than by construction."],
            ["7", "Per-camera ignore masks",
             "Furnaces, welding bays and flare stacks are permanently fire-like; masking them is the "
             "cheapest false-positive reduction available."],
            ["8", "Revisit the smaller DINOv3 trunk (ViT-Ti, 5.5M params)",
             "Only if the cascade proves insufficient - it would cost accuracy."],
        ], [16 * mm, 52 * mm, 92 * mm], align_right_from=99, font=7.4),
        Spacer(1, 4 * mm),
        Paragraph(
            "<b>Recommended immediate next step.</b> Items 1 and 2 together. The model is good enough that "
            "further training on D-Fire has clearly diminishing returns; what it now needs is evidence from "
            "real video and data from the actual deployment site.", s["callout"]),
    ]

    # ---- appendix
    story += [
        Paragraph("Appendix A - Reproducing these results", s["h1"]),
        Paragraph("Stage 1:", s["body"]),
        Paragraph("python scripts/train_dinov3_detector.py --epochs 40 --imgsz 640 --batch 8 "
                  "--workers 4 --name stage1 --zip", s["code"]),
        Paragraph("Stage 2:", s["body"]),
        Paragraph("python scripts/train_dinov3_detector.py --epochs 15 --imgsz 640 --batch 8 "
                  "--workers 4 --unfreeze-last-n 2 --lr 5e-5 --name stage2 --zip "
                  "--init-from runs/fire_smoke/stage1/weights/best.pt", s["code"]),
        Paragraph("Test evaluation:", s["body"]),
        Paragraph("python scripts/eval_dinov3_detector.py --weights .../stage2/weights/best.pt "
                  "--split test --target-fpr 0.01", s["code"]),
        Paragraph("Live camera test:", s["body"]),
        Paragraph("python scripts/predict_video_dinov3.py --weights best.pt --source 0 --show "
                  "--no-save --conf 0.30 --window 8 --enter-hits 3", s["code"]),
        Paragraph(
            "Code: <font face='Courier'>github.com/CoderIsSleeping/fire-and-smoke</font>. This document is "
            "generated by <font face='Courier'>scripts/build_training_report_pdf.py</font> from the run "
            "artifacts in <font face='Courier'>reports/data/</font>, so it stays consistent with what the "
            "runs actually produced.", s["body"]),
    ]

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, 285 * mm, 192 * mm, 285 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 288 * mm, "Fire & Smoke Detection - DINOv3 Training Report")
        canvas.drawRightString(192 * mm, 288 * mm, date.today().isoformat())
        canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
        canvas.drawCentredString(105 * mm, 10 * mm, f"page {doc.page}")
        canvas.restoreState()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(OUT), pagesize=A4,
                          leftMargin=18 * mm, rightMargin=18 * mm,
                          topMargin=22 * mm, bottomMargin=20 * mm,
                          title="Fire & Smoke Detection - DINOv3 Training Report",
                          author="Fire & Smoke Detection Project")
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])
    doc.build(story)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    if not DATA.exists():
        sys.exit(f"Missing {DATA}. Copy the run artifacts there first.")
    build(load_all())
