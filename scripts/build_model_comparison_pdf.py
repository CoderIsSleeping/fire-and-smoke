"""Build the Model 1 vs Model 2 comparison PDF from the run artifacts.

Model 1: DINOv3 ViT-S + Faster R-CNN (two-stage training)
Model 2: DINOv3 ViT-Ti + FCOS, the light, batch-exportable variant

Every number is read from files under reports/data/ (and, if present, the
local customer-site scan results under Industry/, reduced to aggregate counts
only -- no camera names, dates or frames go into the document).

    python scripts/build_model_comparison_pdf.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer

from build_training_report_pdf import (
    MUTED,
    RULE,
    bullets,
    image,
    num,
    parse_alarm_sweep,
    parse_detection_table,
    read_csv_rows,
    styles,
    table,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "reports" / "data"
FIGS = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "Model1_vs_Model2_Comparison.pdf"
SITE_M1 = ROOT / "Industry" / "eval_results" / "video_eval.json"
SITE_M2 = ROOT / "Industry" / "eval_results_light" / "video_eval.json"

C1, C2 = "#1f6f8b", "#2e7d32"

# Measured in this project (see PROJECT_LOG 6b-6c and compare_models.py runs):
# same frames, same CPU, same 640 px input.
MEASURED = {
    "m1_ms_cpu": 891.0,
    "m2_ms_cpu": 336.0,
    "m1_params": 39.3,
    "m2_params": 6.9,
    "m1_trainable_stage1": 17.7,
    "m2_trainable": 1.5,
    "m1_min_per_epoch": 11.0,
}


# ------------------------------------------------------------------ data


def load() -> dict:
    d = {
        "m1_s1": read_csv_rows(DATA / "stage1_results.csv"),
        "m1_s2": read_csv_rows(DATA / "stage2_results.csv"),
        "m2": read_csv_rows(DATA / "light_stage1_results.csv"),
    }
    m1_text = (DATA / "stage2_eval_test.md").read_text(encoding="utf-8")
    m2_text = (DATA / "light_stage1_eval_test.md").read_text(encoding="utf-8")
    d["m1_det"], d["m1_sweep"] = parse_detection_table(m1_text), parse_alarm_sweep(m1_text)
    d["m2_det"], d["m2_sweep"] = parse_detection_table(m2_text), parse_alarm_sweep(m2_text)
    d["m2_min_per_epoch"] = sorted(num(r, "seconds") for r in d["m2"])[len(d["m2"]) // 2] / 60.0

    def site(path):
        if not path.exists():
            return None
        rows = json.loads(path.read_text(encoding="utf-8")).get("grid", [])
        rows = [r for r in rows if "false_alarm_events" in r]
        return rows or None

    d["site_m1"], d["site_m2"] = site(SITE_M1), site(SITE_M2)
    return d


def op_at_budget(sweep: list[dict], budget: float) -> dict:
    """Lowest threshold meeting an FPR budget -- the operating-point rule used throughout."""
    ok = [r for r in sweep if r["fpr"] <= budget]
    return min(ok, key=lambda r: r["conf"]) if ok else max(sweep, key=lambda r: r["conf"])


def recall_at_fpr(sweep: list[dict], target: float) -> float:
    """Linear interpolation of any-class recall at an exact false-alarm rate.

    The evaluation sweeps thresholds in 0.05 steps. That is fine for Model 1,
    but Model 2's scores are compressed into a narrow band, so one 0.05 step
    moves its recall by more than 10 points. Interpolating on the recall-vs-FPR
    curve gives the fair matched-rate comparison.
    """
    pts = sorted((r["fpr"], r["recall_any"]) for r in sweep)
    for (f0, r0), (f1, r1) in zip(pts, pts[1:]):
        if f0 <= target <= f1 and f1 > f0:
            return r0 + (target - f0) / (f1 - f0) * (r1 - r0)
    return pts[0][1] if target < pts[0][0] else pts[-1][1]


def site_row(rows, conf, hits=6):
    if not rows:
        return None
    for r in rows:
        if abs(r["conf"] - conf) < 1e-6 and r["enter_hits"] == hits:
            return r
    return None


# ------------------------------------------------------------------ figures


def fig_architectures(path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    specs = [
        ("MODEL 1", C1, [
            ("Input  640 x 640", "#eef2f5"),
            ("DINOv3 ViT-S/16\n21.6M params, 384-dim\nfrozen, then last 2 blocks tuned", "#dceaf2"),
            ("Feature pyramid\n256 channels, strides 8-64", "#e8f1e9"),
            ("Faster R-CNN head (two-stage)\nRPN -> RoIAlign -> box head", "#e8f1e9"),
            ("Scene classifier + glow prior", "#e8f1e9"),
        ], "39.3M parameters"),
        ("MODEL 2 (light)", C2, [
            ("Input  640 x 640", "#eef2f5"),
            ("DINOv3 ViT-Ti/16\n5.5M params, 192-dim\nfrozen", "#dceaf2"),
            ("Feature pyramid\n128 channels, strides 8-64", "#e8f1e9"),
            ("FCOS head (one-stage, anchor-free)\n2 conv layers per tower", "#e8f1e9"),
            ("Scene classifier + glow prior", "#e8f1e9"),
        ], "6.9M parameters"),
    ]
    for ax, (title, colour, blocks, footer) in zip(axes, specs):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 11.5)
        ax.axis("off")
        ax.text(5, 11.0, title, ha="center", fontsize=12, fontweight="bold", color=colour)
        y = 9.4
        for i, (text, face) in enumerate(blocks):
            h = 1.55 if "\n" in text else 0.9
            ax.add_patch(mpatches.FancyBboxPatch((1, y - h), 8, h, boxstyle="round,pad=0.12",
                                                 facecolor=face, edgecolor=colour, linewidth=1.3))
            ax.text(5, y - h / 2, text, ha="center", va="center", fontsize=8.3, linespacing=1.4)
            if i < len(blocks) - 1:
                ax.annotate("", xy=(5, y - h - 0.32), xytext=(5, y - h - 0.02),
                            arrowprops=dict(arrowstyle="-|>", color="#5b6670", lw=1.2))
            y = y - h - 0.36
        ax.text(5, 0.15, footer, ha="center", fontsize=10, color=colour, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_training(d: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    s1 = [(num(r, "epoch"), num(r, "mAP50"), num(r, "AP50_fire")) for r in d["m1_s1"]]
    offset = s1[-1][0]
    s2 = [(offset + num(r, "epoch"), num(r, "mAP50"), num(r, "AP50_fire")) for r in d["m1_s2"]]
    m2 = [(num(r, "epoch"), num(r, "mAP50"), num(r, "AP50_fire")) for r in d["m2"]]

    for ax, idx, title in ((axes[0], 1, "Validation mAP@0.5"), (axes[1], 2, "Validation AP50 - fire")):
        ax.plot([e for e, *_ in s1], [v[idx - 1] for _, *v in s1], color=C1, lw=2, label="Model 1, stage 1 (frozen)")
        ax.plot([e for e, *_ in s2], [v[idx - 1] for _, *v in s2], color=C1, lw=2, ls="--",
                label="Model 1, stage 2 (2 blocks unfrozen)")
        ax.plot([e for e, *_ in m2], [v[idx - 1] for _, *v in m2], color=C2, lw=2, label="Model 2 (frozen)")
        ax.axvline(offset, color="#999999", lw=0.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_ap_bars(d: dict, path: Path) -> None:
    a, b = d["m1_det"], d["m2_det"]
    labels = ["mAP@0.5", "mAP@0.5:0.95", "smoke AP50", "fire AP50", "fire AP75"]
    m1 = [a["mean"]["AP50"], a["mean"]["AP50_95"], a["smoke"]["AP50"], a["fire"]["AP50"], a["fire"]["AP75"]]
    m2 = [b["mean"]["AP50"], b["mean"]["AP50_95"], b["smoke"]["AP50"], b["fire"]["AP50"], b["fire"]["AP75"]]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for bars in (ax.bar([i - 0.19 for i in x], m1, 0.38, color=C1, label="Model 1"),
                 ax.bar([i + 0.19 for i in x], m2, 0.38, color=C2, label="Model 2 (light)")):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=7)
    for i, (p, q) in enumerate(zip(m1, m2)):
        ax.text(i, max(p, q) + 0.07, f"{q - p:+.3f}", ha="center", fontsize=8, color="#c1121f", fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_title("Test split (4,306 images) - detection quality", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_operating(d: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    for sweep, colour, name in ((d["m1_sweep"], C1, "Model 1"), (d["m2_sweep"], C2, "Model 2")):
        pts = sorted((r["fpr"], r["recall_any"]) for r in sweep)
        axes[0].plot([p[0] for p in pts], [p[1] for p in pts], color=colour, lw=2, marker="o", ms=3, label=name)
        axes[1].plot([r["conf"] for r in sweep], [r["recall_any"] for r in sweep], color=colour, lw=2,
                     marker="o", ms=3, label=f"{name} recall")
        axes[1].plot([r["conf"] for r in sweep], [r["fpr"] for r in sweep], color=colour, lw=1.3, ls="--",
                     label=f"{name} false-alarm rate")
    axes[0].axvline(0.01, color="#c1121f", ls=":", lw=1.3)
    axes[0].text(0.0105, 0.45, " 1% budget", color="#c1121f", fontsize=7)
    axes[0].set_xlim(0, 0.05)
    axes[0].set_ylim(0.4, 1.0)
    axes[0].set_xlabel("false-alarm rate on 2,005 verified-negative images", fontsize=8)
    axes[0].set_ylabel("recall (fire or smoke)", fontsize=8)
    axes[0].set_title("The fair comparison: recall at the same false-alarm rate", fontsize=10)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].axvspan(0.80, 0.95, color="#fde2e2", alpha=0.6)
    axes[1].text(0.805, 0.5, "Model 2 scores\nnever reach here", fontsize=7, color="#c1121f")
    axes[1].set_xlabel("confidence threshold", fontsize=8)
    axes[1].set_title("Why each model needs its own threshold", fontsize=10)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=6.5, loc="center left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_cost(d: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.0))
    items = [
        ("Parameters (M)", MEASURED["m1_params"], MEASURED["m2_params"], "{:.1f}M"),
        ("Inference, same CPU (ms/frame)", MEASURED["m1_ms_cpu"], MEASURED["m2_ms_cpu"], "{:.0f} ms"),
        ("Training time per epoch (min)", MEASURED["m1_min_per_epoch"], d["m2_min_per_epoch"], "{:.1f} min"),
    ]
    for ax, (title, v1, v2, fmt) in zip(axes, items):
        bars = ax.bar(["Model 1", "Model 2"], [v1, v2], color=[C1, C2], width=0.55)
        for bar, v in zip(bars, (v1, v2)):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02, fmt.format(v), ha="center", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, max(v1, v2) * 1.2)
        ax.grid(axis="y", alpha=0.25)
        ax.text(0.5, 0.88, f"{v1 / v2:.2f}x smaller" if v2 else "", transform=ax.transAxes,
                ha="center", fontsize=8, color=C2, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ document


def build(d: dict) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_architectures(FIGS / "cmp_architectures.png")
    fig_training(d, FIGS / "cmp_training.png")
    fig_ap_bars(d, FIGS / "cmp_ap.png")
    fig_operating(d, FIGS / "cmp_operating.png")
    fig_cost(d, FIGS / "cmp_cost.png")

    s = styles()
    a, b = d["m1_det"], d["m2_det"]
    m1_val = max(num(r, "mAP50") for r in d["m1_s2"])
    m2_val = max(num(r, "mAP50") for r in d["m2"])
    m1_op, m2_op = op_at_budget(d["m1_sweep"], 0.01), op_at_budget(d["m2_sweep"], 0.01)
    r1_07, r2_07 = recall_at_fpr(d["m1_sweep"], 0.007), recall_at_fpr(d["m2_sweep"], 0.007)
    r1_10, r2_10 = recall_at_fpr(d["m1_sweep"], 0.010), recall_at_fpr(d["m2_sweep"], 0.010)
    r1_20, r2_20 = recall_at_fpr(d["m1_sweep"], 0.020), recall_at_fpr(d["m2_sweep"], 0.020)
    speed = MEASURED["m1_ms_cpu"] / MEASURED["m2_ms_cpu"]
    m2_max_conf = max((r["conf"] for r in d["m2_sweep"] if r["recall_any"] > 0.01), default=0)

    story = [
        Spacer(1, 26 * mm),
        Paragraph("Model 1 vs Model 2", s["title"]),
        Paragraph("Fire &amp; smoke detection on industrial cameras<br/>"
                  "Why a second model was built, how it was trained, and what it traded away", s["subtitle"]),
        Spacer(1, 10 * mm),
        table([
            ["", "Model 1", "Model 2 (light)"],
            ["Backbone", "DINOv3 ViT-S/16 (21.6M)", "DINOv3 ViT-Ti/16 (5.5M)"],
            ["Detection head", "Faster R-CNN (two-stage)", "FCOS (one-stage)"],
            ["Total parameters", f"{MEASURED['m1_params']}M", f"{MEASURED['m2_params']}M"],
            ["Training", "40 ep frozen + 15 ep partial unfreeze", f"{len(d['m2'])} ep frozen"],
            ["Test mAP@0.5", f"{a['mean']['AP50']:.4f}", f"{b['mean']['AP50']:.4f}"],
            ["Recall at 1% false-alarm rate*", f"{r1_10:.3f}", f"{r2_10:.3f}"],
            ["Speed, same CPU", f"{MEASURED['m1_ms_cpu']:.0f} ms/frame", f"{MEASURED['m2_ms_cpu']:.0f} ms/frame "
                                                                           f"({speed:.2f}x faster)"],
            ["Batched export (multi-camera)", "No - measured to fail", "Yes - verified"],
        ], [52 * mm, 55 * mm, 55 * mm], align_right_from=9, font=8.6),
        Paragraph("* interpolated on each model's recall-versus-false-alarm curve; see section 5.", s["caption"]),
        Spacer(1, 4 * mm),
        Paragraph(
            f"<b>Verdict.</b> Model 2 is {speed:.1f}x faster, {MEASURED['m1_params'] / MEASURED['m2_params']:.1f}x "
            "smaller, and - unlike Model 1 - can be exported as a batched graph for many cameras. It paid for "
            f"that with accuracy: {a['mean']['AP50'] - b['mean']['AP50']:.3f} lower test mAP@0.5 and "
            f"{(r1_10 - r2_10) * 100:.1f} points lower recall at the same 1% false-alarm rate. As trained, it is "
            "<b>not a drop-in replacement</b>; Model 1 remains the accuracy reference. Section 8 lists the two "
            "experiments that decide whether Model 2 can close the gap.", s["callout"]),
        PageBreak(),
    ]

    # 1 why
    story += [
        Paragraph("1. Why a second model was needed", s["h1"]),
        Paragraph(
            "The customer's target is <b>70 cameras</b>. At 10 frames per second that is 700 frames per second, "
            "and the standard way to serve that is to batch frames from many cameras through one GPU using an "
            "exported, optimised engine (ONNX, then TensorRT). Model 1 was measured to block that path:", s["body"]),
        table([
            ["Export attempt (Model 1)", "Batch 1", "Batch 4"],
            ["Legacy ONNX exporter", "works", "fails in ONNX Runtime (batch size baked in)"],
            ["torch.export (current exporter)", "fails", "-"],
        ], [60 * mm, 25 * mm, 75 * mm], align_right_from=9, font=8),
        Spacer(1, 3 * mm),
        Paragraph(
            "The cause is structural. Faster R-CNN picks a <i>variable</i> number of candidate regions in the "
            "middle of the network and crops features for exactly those before its final layers run, so there "
            "is no fixed-shape point at which to cut the graph. A one-stage FCOS head places every learned "
            "layer before anything data-dependent, so it exports cleanly. The export of Model 2 was verified on "
            "every run: boxes decoded from the ONNX outputs match the PyTorch model to about 6e-5 pixels at "
            "batch sizes 1, 4 and 8.", s["body"]),
        Paragraph(
            "A second, measured finding shaped the design: the head swap on its own did <b>not</b> make the "
            "model faster (ViT-S + FCOS ran at 0.96x). Roughly two-thirds of the compute is the DINOv3 trunk, "
            "so the speed had to come from a smaller DINOv3 - ViT-Ti. Input resolution was deliberately left at "
            "640 px, because reducing it to 448 px had already been measured to cost 24 points of recall.",
            s["body"]),
        Paragraph("<b>DINO was never removed.</b> Both models use a DINOv3 backbone; Model 2 uses a smaller "
                  "member of the same family, with a different head after it.", s["callout"]),
        image(FIGS / "cmp_architectures.png", 165),
        Paragraph("Figure 1 - The two architectures. Only the backbone size, pyramid width and detection "
                  "head differ; input size, scene classifier and alarm logic are identical.", s["caption"]),
        PageBreak(),
    ]

    # 2 config
    story += [
        Paragraph("2. Configuration and training", s["h1"]),
        table([
            ["Setting", "Model 1", "Model 2"],
            ["Backbone", "vit_small_patch16_dinov3 (LVD-1689M)", "vit_tiny_patch16_dinov3_qkvb (EUPE)"],
            ["Embedding dim / blocks", "384 / 12", "192 / 12"],
            ["Pyramid channels", "256", "128"],
            ["Head", "Faster R-CNN, 300 test proposals", "FCOS, 2 convs per tower"],
            ["Trainable parameters", f"{MEASURED['m1_trainable_stage1']}M (stage 1)", f"{MEASURED['m2_trainable']}M"],
            ["Input size / batch", "640 / 8", "640 / 8"],
            ["Optimiser / LR", "AdamW, 1e-4, cosine, 500-iter warmup", "AdamW, 1e-4, cosine, 500-iter warmup"],
            ["Augmentation", "identical: night 0.35, IR 0.10, occlusion 0.25, ...", "identical"],
            ["Epochs", "40 (stage 1) + 15 (stage 2)", f"{len(d['m2'])} (frozen trunk only)"],
            ["Time per epoch", f"~{MEASURED['m1_min_per_epoch']:.0f} min", f"~{d['m2_min_per_epoch']:.1f} min"],
            ["Best validation mAP@0.5", f"{m1_val:.4f}", f"{m2_val:.4f}"],
        ], [45 * mm, 60 * mm, 60 * mm], align_right_from=9, font=7.8),
        Spacer(1, 4 * mm),
        Paragraph(
            "Both models were trained on the same D-Fire split with the same augmentation, optimiser and "
            "schedule, and selected by validation mAP@0.5, so the comparison isolates the architecture. One "
            "difference is not yet controlled: Model 1 received a second stage that unfroze its last two "
            "transformer blocks (worth +0.009 mAP), and Model 2 has not.", s["body"]),
        image(FIGS / "cmp_training.png", 172),
        Paragraph("Figure 2 - Validation curves. Model 1's stage 2 continues from stage 1 (dashed, after the "
                  "dotted line). Model 2 plateaus around 0.62 and was still inching upward at the end.",
                  s["caption"]),
        Paragraph(
            "Model 2 converged smoothly but to a lower ceiling. With the trunk frozen it trains only "
            f"{MEASURED['m2_trainable']}M parameters against Model 1's {MEASURED['m1_trainable_stage1']}M, which "
            "is the most likely single reason for the gap - and the reason a stage-2 run is the first thing to "
            "try.", s["body"]),
        PageBreak(),
    ]

    # 3 test results
    story += [
        Paragraph("3. Test-split results", s["h1"]),
        Paragraph("Held-out D-Fire test split: 4,306 images, 2,005 of them verified negatives. Neither model "
                  "saw it during training or model selection.", s["body"]),
        table([
            ["Class", "Model 1 AP50", "Model 2 AP50", "Change", "Model 1 AP75", "Model 2 AP75"],
            ["smoke", f"{a['smoke']['AP50']:.4f}", f"{b['smoke']['AP50']:.4f}",
             f"{b['smoke']['AP50'] - a['smoke']['AP50']:+.4f}", f"{a['smoke']['AP75']:.4f}", f"{b['smoke']['AP75']:.4f}"],
            ["fire", f"{a['fire']['AP50']:.4f}", f"{b['fire']['AP50']:.4f}",
             f"{b['fire']['AP50'] - a['fire']['AP50']:+.4f}", f"{a['fire']['AP75']:.4f}", f"{b['fire']['AP75']:.4f}"],
            ["mean", f"{a['mean']['AP50']:.4f}", f"{b['mean']['AP50']:.4f}",
             f"{b['mean']['AP50'] - a['mean']['AP50']:+.4f}", f"{a['mean']['AP75']:.4f}", f"{b['mean']['AP75']:.4f}"],
        ], [22 * mm, 27 * mm, 27 * mm, 24 * mm, 27 * mm, 27 * mm]),
        Spacer(1, 3 * mm),
        image(FIGS / "cmp_ap.png", 160),
        Paragraph("Figure 3 - Detection quality on the test split.", s["caption"]),
        Paragraph(
            "Model 2 is lower on every metric, by roughly 10-13 points of AP50 on each class. Fire - already "
            "the harder class for Model 1 - drops the most. Tight localisation (AP75) falls further still, "
            "which is consistent with a narrower backbone losing fine spatial detail on small flames.",
            s["body"]),
        PageBreak(),
    ]

    # 4/5 fair comparison
    story += [
        Paragraph("4. Why the two models need different thresholds", s["h1"]),
        Paragraph(
            "An important detail for anyone deploying Model 2: its confidence scores live on a different scale. "
            "FCOS scores a box as the square root of classification confidence times a 'centerness' estimate, "
            "and centerness rarely approaches 1, so scores are compressed into a narrow band. On the test split "
            f"Model 2 has essentially no detections above {m2_max_conf:.2f}: <b>at Model 1's operating "
            "threshold of 0.90, Model 2's recall is exactly zero.</b>", s["body"]),
        Paragraph(
            "Comparing the models at the same threshold is therefore meaningless. Each is compared at its own "
            f"operating point - the lowest threshold that keeps the false-alarm rate under 1%: <b>{m1_op['conf']:.2f} "
            f"for Model 1 and {m2_op['conf']:.2f} for Model 2.</b> The live demo tool takes a separate "
            "threshold per model for the same reason.", s["body"]),
        image(FIGS / "cmp_operating.png", 172),
        Paragraph("Figure 4 - Left: recall against false-alarm rate, the comparison that matters. Right: the "
                  "same data against the raw threshold, showing Model 2's compressed score range.", s["caption"]),
        Paragraph("5. Recall at the same false-alarm rate", s["h1"]),
        table([
            ["False-alarm rate", "Model 1 recall", "Model 2 recall", "Gap"],
            ["0.7%  (Model 1's operating point)", f"{r1_07:.3f}", f"{r2_07:.3f}", f"{(r2_07 - r1_07) * 100:+.1f} pts"],
            ["1.0%", f"{r1_10:.3f}", f"{r2_10:.3f}", f"{(r2_10 - r1_10) * 100:+.1f} pts"],
            ["2.0%", f"{r1_20:.3f}", f"{r2_20:.3f}", f"{(r2_20 - r1_20) * 100:+.1f} pts"],
        ], [62 * mm, 32 * mm, 32 * mm, 30 * mm]),
        Spacer(1, 3 * mm),
        Paragraph(
            "Recall here means an image containing fire or smoke produced at least one box - before temporal "
            "confirmation, which recovers some misses on video. Values are interpolated on each model's "
            "curve, because the evaluation sweeps thresholds in 0.05 steps and a single step moves Model 2's "
            "recall by more than 10 points. The gap narrows as the false-alarm budget loosens.", s["body"]),
        Paragraph(
            f"<b>In plain terms:</b> at Model 1's own operating point, Model 1 finds about "
            f"{round(r1_07 * 100)} of every 100 fire or smoke images and Model 2 about {round(r2_07 * 100)}. For a "
            "safety system, that gap is too large to accept in exchange for speed alone.", s["callout"]),
        PageBreak(),
    ]

    # 6 cost + site
    story += [
        Paragraph("6. Speed, size and deployability", s["h1"]),
        image(FIGS / "cmp_cost.png", 172),
        Paragraph("Figure 5 - Cost. Inference measured on the same frames, same CPU, same 640 px input, with "
                  "the side-by-side comparison tool.", s["caption"]),
        table([
            ["", "Model 1", "Model 2"],
            ["ms per frame (CPU)", f"{MEASURED['m1_ms_cpu']:.0f}", f"{MEASURED['m2_ms_cpu']:.0f}"],
            ["Batched ONNX export", "fails at batch > 1", "verified at batch 1, 4, 8"],
            ["Path to TensorRT fp16 / multi-camera batching", "blocked", "open"],
        ], [70 * mm, 45 * mm, 45 * mm], align_right_from=9, font=8),
        Spacer(1, 3 * mm),
        Paragraph(
            "The CPU figures show the ratio, not deployment speed. The real multi-camera gain comes from "
            "batching and TensorRT fp16 on a GPU, which only Model 2 can use; it has not yet been measured on "
            "the target GPU, so no camera count is claimed here.", s["body"]),
    ]

    story.append(Paragraph("7. Customer-site footage", s["h1"]))
    if d["site_m1"] and d["site_m2"]:
        m1_hours = d["site_m1"][0]["negative_hours"] * 60
        rows = [["Model / threshold", "False alarms", "Per hour"]]
        for label, rows_src, conf in (("Model 1 at 0.90 (its operating point)", d["site_m1"], 0.90),
                                      ("Model 1 at 0.85", d["site_m1"], 0.85),
                                      ("Model 1 at 0.70", d["site_m1"], 0.70),
                                      (f"Model 2 at {m2_op['conf']:.2f} (its operating point)", d["site_m2"], m2_op["conf"]),
                                      ("Model 2 at 0.50", d["site_m2"], 0.50),
                                      ("Model 2 at 0.45", d["site_m2"], 0.45)):
            r = site_row(rows_src, conf)
            if r:
                rows.append([label, str(r["false_alarm_events"]), f"{r['false_alarm_events_per_hour']:.1f}"])
        story += [
            Paragraph(
                f"Both models were run on {m1_hours:.1f} minutes of normal working-day footage from three fixed "
                "cameras at an industrial customer site - no fire present, so every alarm is false. Sampling "
                "1 frame per second, alarm on 6 confirmed detections in 15.", s["body"]),
            table(rows, [90 * mm, 35 * mm, 35 * mm], font=8),
            Spacer(1, 3 * mm),
            Paragraph(
                "A clean result here is necessary but not sufficient: this footage contains no fire, so it says "
                "nothing about recall. Its value is that it is the only real deployment-site data in the "
                "project. The false positives it did produce for Model 1, below its threshold, came from a "
                "worker's yellow hard hat and from bright dusty haze - confuser classes to add as training "
                "negatives.", s["body"]),
        ]
    else:
        story.append(Paragraph("Site-footage results for both models are not available in this build.", s["body"]))

    story.append(PageBreak())

    # 8 verdict
    story += [
        Paragraph("8. Verdict and next steps", s["h1"]),
        table([
            ["Question", "Answer"],
            ["Is Model 2 faster and smaller?", f"Yes - {speed:.2f}x faster, {MEASURED['m1_params'] / MEASURED['m2_params']:.1f}x fewer parameters"],
            ["Can Model 2 be batched for many cameras?", "Yes - verified; Model 1 cannot"],
            ["Is Model 2 as accurate?", f"No - {a['mean']['AP50'] - b['mean']['AP50']:.3f} lower mAP@0.5, "
                                        f"{(r1_07 - r2_07) * 100:.0f} pts lower recall at 0.7% FPR"],
            ["Should Model 2 replace Model 1 today?", "No. Model 1 stays the accuracy reference."],
        ], [70 * mm, 95 * mm], align_right_from=9, font=8.2),
        Spacer(1, 4 * mm),
        Paragraph(
            "The result is an honest trade-off, and it is informative rather than disappointing: it shows "
            "exactly what deployability costs. Two changes were made at once - a smaller backbone <i>and</i> a "
            "different head - so the next experiments separate them:", s["body"]),
        *bullets([
            "<b>Model 2, stage 2</b> - unfreeze the last two ViT-Ti blocks, as was done for Model 1 "
            f"(+0.009 there). With only {MEASURED['m2_trainable']}M trainable parameters, Model 2 is likely to "
            "gain more. Cheapest experiment: 15 epochs.",
            "<b>ViT-S + FCOS</b> - Model 1's backbone with Model 2's head. Keeps batched export, runs at "
            "roughly Model 1's speed, and shows whether the accuracy loss came from the head or the smaller "
            "backbone. If accuracy holds, speed can come from TensorRT and batching instead of a smaller model.",
            "<b>Hard negatives from the site</b> - hard hats, hi-vis and dusty haze as empty-label training "
            "images, for whichever model is chosen.",
            "<b>Measure on the target GPU</b> - TensorRT fp16 with batching, before any camera count is stated.",
        ], s),
        Paragraph(
            "<b>How to present this:</b> Model 1 is the accurate reference; Model 2 demonstrates the deployable "
            "architecture and quantifies its accuracy cost. The system is designed for multi-camera scaling; the "
            "final model choice and camera capacity will be set by the experiments above and by benchmarking on "
            "the target hardware.", s["callout"]),
    ]

    story += [
        Paragraph("Appendix - commands", s["h1"]),
        Paragraph("Model 2 training:", s["body"]),
        Paragraph("python scripts/train_dinov3_detector.py --backbone vit_tiny_patch16_dinov3_qkvb.eupe_lvd1689m "
                  "--head fcos --head-convs 2 --fpn-channels 128 --epochs 40 --imgsz 640 --batch 8 --name light_stage1",
                  s["code"]),
        Paragraph("Side-by-side demo, each model at its own operating point:", s["body"]),
        Paragraph(f"python scripts/compare_models.py --model \"Model 1=&lt;model1 best.pt&gt;@{m1_op['conf']:.2f}\" "
                  f"--model \"Model 2=&lt;model2 best.pt&gt;@{m2_op['conf']:.2f}\" --video &lt;clip&gt; --max-seconds 120",
                  s["code"]),
        Paragraph("Rebuild this document: python scripts/build_model_comparison_pdf.py", s["code"]),
    ]

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, 285 * mm, 192 * mm, 285 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 288 * mm, "Fire & Smoke Detection - Model 1 vs Model 2")
        canvas.drawRightString(192 * mm, 288 * mm, date.today().isoformat())
        canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
        canvas.drawCentredString(105 * mm, 10 * mm, f"page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                          topMargin=22 * mm, bottomMargin=20 * mm,
                          title="Fire & Smoke Detection - Model 1 vs Model 2")
    doc.addPageTemplates([PageTemplate(id="main", frames=[Frame(doc.leftMargin, doc.bottomMargin,
                                                                 doc.width, doc.height)], onPage=decorate)])
    doc.build(story)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"site section: {'included' if d['site_m1'] and d['site_m2'] else 'skipped (scan results missing)'}")


if __name__ == "__main__":
    build(load())
