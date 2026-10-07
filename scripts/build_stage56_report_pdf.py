"""Stage 5 / Stage 6 report: what was done for small and distant fires, what
improved, what did not, and the plan for making the model more robust.

Numbers come from reports/data (test-split evaluation reports and the video
benchmark summary). The video screenshots and the training-curve image are
read from Industry/report_assets, which is outside the repository: the footage
is confidential, so the finished PDF is written there too and never committed.

    python scripts/build_stage56_report_pdf.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

from build_final_results_pdf import BANDS, load, row_at

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "reports" / "figures"
ASSETS = ROOT / "Industry" / "report_assets"
OUT = ROOT / "Industry" / "reports" / "Stage5_Stage6_Small_Fire_Report.pdf"
FONT_DIR = Path("C:/Windows/Fonts")

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5b6670")
ACCENT = colors.HexColor("#b5410f")
RULE = colors.HexColor("#c8cdd2")
BAND = colors.HexColor("#f2f4f6")
GOOD = colors.HexColor("#e8f5e9")
COL = {"s2": "#0b3d91", "s5": "#8d6e63", "s6": "#00897b"}

# Validation results per epoch, from the Kaggle training logs (mAP@0.5 and the
# share of fire/smoke images detected at <= 1% false alarms).
VAL = {
    "s5": {"map": [0.7123, 0.7166, 0.7182, 0.7206, 0.7221, 0.7237],
           "recall": [0.5896, 0.7673, 0.7482, 0.8512, 0.7766, 0.7711]},
    "s6": {"map": [0.7280, 0.7281, 0.7291, 0.7340, 0.7349, 0.7334],
           "recall": [0.7798, 0.8283, 0.8441, 0.8861, 0.8610, 0.8578]},
}


def register_fonts() -> None:
    for name, file in (("Calibri", "calibri.ttf"), ("Calibri-Bold", "calibrib.ttf"),
                       ("Calibri-Italic", "calibrii.ttf"), ("Calibri-BoldItalic", "calibriz.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
    pdfmetrics.registerFontFamily("Calibri", normal="Calibri", bold="Calibri-Bold", italic="Calibri-Italic",
                                  boldItalic="Calibri-BoldItalic")
    for file in ("calibri.ttf", "calibrib.ttf", "calibrii.ttf"):
        font_manager.fontManager.addfont(str(FONT_DIR / file))
    plt.rcParams["font.family"] = "Calibri"
    plt.rcParams["font.size"] = 10
    # Charts are saved at 320 dpi on purpose: Calibri carries embedded bitmap
    # strikes for small pixel sizes, which matplotlib draws as blank text
    # (legends and bar labels vanished at 180 dpi).


def styles() -> dict:
    s = {}
    s["title"] = ParagraphStyle("t", fontName="Calibri-Bold", fontSize=25, leading=29, textColor=INK,
                                alignment=TA_CENTER, spaceAfter=5)
    s["subtitle"] = ParagraphStyle("st", fontName="Calibri", fontSize=12.5, leading=16, textColor=MUTED,
                                   alignment=TA_CENTER)
    s["h1"] = ParagraphStyle("h1", fontName="Calibri-Bold", fontSize=15.5, leading=19, textColor=ACCENT,
                             spaceBefore=13, spaceAfter=6)
    s["h2"] = ParagraphStyle("h2", fontName="Calibri-Bold", fontSize=12.3, leading=15, textColor=INK,
                             spaceBefore=9, spaceAfter=4)
    s["body"] = ParagraphStyle("b", fontName="Calibri", fontSize=10.4, leading=14.4, textColor=INK,
                               alignment=TA_JUSTIFY, spaceAfter=6)
    s["bullet"] = ParagraphStyle("bu", parent=s["body"], leftIndent=12, bulletIndent=3, spaceAfter=3.5)
    s["caption"] = ParagraphStyle("c", fontName="Calibri-Italic", fontSize=8.8, leading=11.2, textColor=MUTED,
                                  alignment=TA_CENTER, spaceBefore=3, spaceAfter=8)
    s["callout"] = ParagraphStyle("ca", fontName="Calibri", fontSize=10.3, leading=14.2, textColor=INK,
                                  backColor=colors.HexColor("#fdf6e3"), borderPadding=8, borderColor=ACCENT,
                                  borderWidth=0.8, spaceAfter=9)
    s["cell"] = ParagraphStyle("cell", fontName="Calibri", fontSize=9.2, leading=11.6, textColor=INK)
    return s


def table(rows, widths, right_from=1, font=9.2, highlight_col=None):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Calibri"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
        ("ALIGN", (right_from, 0), (-1, -1), "RIGHT"),
        ("FONTNAME", (0, 0), (-1, 0), "Calibri-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), BAND),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED),
    ]
    if highlight_col is not None:
        style.append(("BACKGROUND", (highlight_col, 1), (highlight_col, -1), GOOD))
    t.setStyle(TableStyle(style))
    return t


def bullets(items, s):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{i}", s["bullet"]) for i in items]


def image(path: Path, width_mm: float):
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    return Image(str(path), width=width_mm * mm, height=width_mm * mm * h / w)


# ------------------------------------------------------------------ figures


def fig_training(path: Path, stage2_val: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    epochs = range(1, 7)
    for key, label in (("s5", "Stage 5: 896 px + mosaic + flame paste"), ("s6", "Stage 6: 896 px + mosaic")):
        axes[0].plot(epochs, VAL[key]["map"], "o-", color=COL[key], label=label)
        axes[1].plot(epochs, [v * 100 for v in VAL[key]["recall"]], "o-", color=COL[key], label=label)
    axes[0].axhline(stage2_val, color=COL["s2"], ls="--", lw=1.1, label=f"Stage 2 at 640 px ({stage2_val:.4f})")
    axes[0].set_title("Validation mAP@0.5")
    axes[0].set_ylim(0.705, 0.745)
    axes[1].set_title("Fire/smoke images detected at <= 1% false alarms (%)")
    axes[1].set_ylim(50, 95)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.6, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=320)
    plt.close(fig)


def fig_small(d: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    models = (("s2", "Stage 2 (640 px)"), ("m5", "Stage 5 (paste)"), ("m6", "Stage 6 (Model 6)"))
    colour = {"s2": COL["s2"], "m5": COL["s5"], "m6": COL["s6"]}
    w = 0.26
    for ax, key, title in ((axes[0], "AP50", "Fire AP@0.5 by fire size"),
                           (axes[1], "rec50", "Fires found at confidence >= 0.5")):
        for i, (k, label) in enumerate(models):
            vals = [d["size"][k]["fire"][b][key] for b in BANDS]
            xs = [j + (i - 1) * w for j in range(len(BANDS))]
            ax.bar(xs, vals, w, label=label, color=colour[k])
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.6)
        ax.set_xticks(range(len(BANDS)))
        ax.set_xticklabels(["tiny\n<16 px", "small\n16-32 px", "medium\n32-96 px", "large\n>96 px"], fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7.6, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=320)
    plt.close(fig)


def fig_false_alarms(d: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    labels = ["Stage 2\n640 px", "Stage 5\n896 + mosaic\n+ paste", "Stage 6\n896 + mosaic"]
    vals = [row_at(d["test"][k][1], 0.90)["fp_images"] for k in ("s2", "m5", "m6")]
    bars = ax.bar(labels, vals, color=[COL["s2"], COL["s5"], COL["s6"]], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, str(v), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("normal test images with a false detection")
    ax.set_title("False alarms at confidence 0.90 (of 2,005 normal images)", fontsize=9.5)
    ax.set_ylim(0, 54)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=320)
    plt.close(fig)


# ------------------------------------------------------------------ document


def build() -> None:
    register_fonts()
    d = load()
    FIGS.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    stage2_val = max(float(r["mAP50"]) for r in d["val"]["s2"])
    fig_training(FIGS / "s56_training.png", stage2_val)
    fig_small(d, FIGS / "s56_small_fire.png")
    fig_false_alarms(d, FIGS / "s56_false_alarms.png")

    s = styles()
    T, SZ, V = d["test"], d["size"], d["video"]
    s2, m5, m6 = T["s2"][0], T["m5"][0], T["m6"][0]
    r2, r5, r6 = (row_at(T[k][1], 0.90) for k in ("s2", "m5", "m6"))
    f2, f5, f6 = (SZ[k]["fire"] for k in ("s2", "m5", "m6"))
    vs = {x["key"]: x for x in V["setups"]}

    def pct(v, digits=0):
        return f"{v * 100:.{digits}f}%"

    def vcell(key, i):
        st = vs[key]
        med, delay = st["median"][i], st["delay_s"][i]
        alarm = f"alarm +{delay:.1f} s" if delay is not None else "MISSED"
        return alarm if med is None else f"{med:.2f}  |  {alarm}"

    story = [
        Spacer(1, 16 * mm),
        Paragraph("Stage 5 and Stage 6", s["title"]),
        Paragraph("Detecting small and distant fires<br/>What was done, what improved, and the plan to make the "
                  f"model more robust<br/>{date.today():%d %B %Y}", s["subtitle"]),
        Spacer(1, 8 * mm),
        Paragraph(
            "<b>Result in one paragraph.</b> On real CCTV-style video, the earlier model (stage 2) saw small and "
            "distant fires but was not confident enough to raise an alarm. Stage 6 retrains that model at a "
            "higher input resolution (896 px instead of 640 px) with mosaic augmentation. It now raises an alarm "
            "for <b>all 7 fires</b> of the video benchmark (stage 2: 5 of 7), has the <b>best test accuracy of the "
            f"project</b> (mAP@0.5 {m6['mean']['AP50']:.4f}), and is <b>much better on tiny fires</b> "
            f"(AP {f2['tiny']['AP50']:.3f} -> {f6['tiny']['AP50']:.3f}). Stage 5, which also pasted synthetic "
            "flames into training images, was rejected because it tripled false alarms.", s["callout"]),
        table([
            ["", "Stage 2 (before)", "Stage 5", "Stage 6 (new model)"],
            ["Input resolution", "640 px", "896 px", "896 px"],
            ["Added training augmentation", "-", "mosaic + flame paste", "mosaic"],
            ["Test mAP@0.5", f"{s2['mean']['AP50']:.4f}", f"{m5['mean']['AP50']:.4f}", f"{m6['mean']['AP50']:.4f}"],
            ["Test fire AP", f"{s2['fire']['AP50']:.4f}", f"{m5['fire']['AP50']:.4f}", f"{m6['fire']['AP50']:.4f}"],
            ["Tiny fire AP (under 16 px)", f"{f2['tiny']['AP50']:.3f}", f"{f5['tiny']['AP50']:.3f}",
             f"{f6['tiny']['AP50']:.3f}"],
            ["Video benchmark: fires alarmed (threshold 0.80)", "5 of 7", "7 of 7 *", "7 of 7"],
            ["False-alarm images at confidence 0.90 (of 2,005)", str(r2["fp_images"]), str(r5["fp_images"]),
             str(r6["fp_images"])],
            ["Decision", "previous model", "rejected", "new candidate model"],
        ], [70 * mm, 32 * mm, 36 * mm, 36 * mm], highlight_col=3),
        Paragraph("* Stage 5 was trained on pasted flames and the benchmark fires are pasted flames, so this "
                  "result is biased in its favour (section 4).", s["caption"]),
        PageBreak(),
    ]

    # 1 problem
    story += [
        Paragraph("1. The problem these stages address", s["h1"]),
        Paragraph(
            "Stages 1 to 4 were judged on still photos from the D-Fire test set. When the stage 2 model was run "
            "on industrial CCTV video (1920 x 1080) with fires of different sizes, a weakness appeared that the "
            "photo benchmark hides:", s["body"]),
        *bullets([
            "Large and medium fires (about 100-150 px wide) scored 0.97 or more and raised an alarm within one "
            "second.",
            "Small and distant fires (about 45 px wide) were <b>detected but never confirmed</b>. Their scores "
            "were 0.61 to 0.86, below the alarm threshold, so no alarm was raised.",
            "The cause is size. The model reads a 640 px copy of the 1920 px frame, so a 45 px flame becomes "
            "about 15 px. The tracking logic was checked and ruled out.",
        ], s),
        Paragraph("How the stages are numbered", s["h2"]),
        table([
            ["Stage", "What was trained", "Starts from", "Outcome"],
            ["1", "DINOv3 backbone frozen, 640 px", "pretrained DINOv3", "first working model"],
            ["2", "Last 2 backbone blocks unfrozen, 640 px", "stage 1", "Model 1, proven on site footage"],
            ["3", "More fire data (DFS + FASDD)", "stage 2", "no gain on the test set"],
            ["4", "Fine-tune of stage 3 on D-Fire only", "stage 3", "ties stage 2"],
            ["5", "896 px + mosaic + flame paste", "stage 2", "rejected: false alarms tripled"],
            ["6", "896 px + mosaic", "stage 2", "Model 6: best accuracy, small fires fixed"],
        ], [14 * mm, 62 * mm, 36 * mm, 62 * mm], right_from=9),
        Paragraph("2. What was done", s["h1"]),
        Paragraph("Two measurement tools, built first", s["h2"]),
        *bullets([
            "<b>Accuracy by fire size.</b> The evaluation now reports results separately for tiny (under 16 px), "
            "small (16-32 px), medium (32-96 px) and large (over 96 px) objects, measured at the model input. "
            f"This showed that stage 2 finds only {pct(f2['tiny']['rec50'])} of tiny fires.",
            "<b>A video benchmark.</b> Seven fires in two CCTV clips, each scored separately: the model's "
            "confidence on that fire, how long the alarm takes, and false alarms in the rest of the clip.",
        ], s),
        Paragraph("Two training changes", s["h2"]),
        *bullets([
            "<b>Higher resolution (896 px).</b> The model sees a 45 px flame as about 21 px instead of 15 px, and "
            "it is <i>trained</i> at that size, not only run at it.",
            "<b>Mosaic augmentation.</b> Four or nine training images are tiled into one frame, so every fire "
            "appears at half or a third of its normal size inside a busy scene. The existing crop augmentation "
            "only ever zoomed in.",
            "<b>Flame paste (stage 5 only).</b> Real flames cut out of training images and blended into other "
            "images at 8-40 px, to manufacture small fires in new scenes.",
        ], s),
        Paragraph("Both stages start from the stage 2 weights and train for 6 epochs: AdamW, learning rate 3e-5, "
                  "last 2 backbone blocks unfrozen, effective batch 8, on a Kaggle T4 GPU (about 28 minutes per "
                  "epoch).", s["body"]),
        PageBreak(),
    ]

    # 3 training
    story += [
        Paragraph("3. Training", s["h1"]),
        image(FIGS / "s56_training.png", 168),
        Paragraph("Validation after each epoch. Left: box accuracy. Right: share of fire/smoke images detected "
                  "while keeping false alarms on normal images at or below 1%.", s["caption"]),
        Paragraph(
            "Stage 6 is ahead of stage 5 from the first epoch on both measures, and the gap in the right-hand "
            "chart is the important one: with flame pasting, the model needed a much higher threshold to keep "
            "false alarms down, and lost detections as a result. Stage 5 was run twice by accident with "
            "identical settings; the second run gave almost the same numbers (test mAP 0.7166 against 0.7190), "
            "so the differences between stages are real and not run-to-run noise.", s["body"]),
    ]
    if (ASSETS / "train_stage6.png").exists():
        story += [
            Paragraph("Stage 6 training curves", s["h2"]),
            image(ASSETS / "train_stage6.png", 160),
            Paragraph("Stage 6 run as recorded by the training script: losses fall steadily, validation accuracy "
                      "rises to its best at epoch 5 (0.7349), and the false-alarm rate at the operating threshold "
                      "stays under 1% throughout.", s["caption"]),
        ]
    story += [PageBreak()]

    # 4 results
    story += [
        Paragraph("4. Results", s["h1"]),
        Paragraph("Test set: accuracy by fire size", s["h2"]),
        image(FIGS / "s56_small_fire.png", 165),
        Paragraph("D-Fire test split (4,306 images). Fire size is measured in pixels of a 640 px model input.",
                  s["caption"]),
        table([
            ["Test split", "Stage 2", "Stage 5", "Stage 6"],
            ["mAP@0.5 (smoke / fire)", f"{s2['mean']['AP50']:.4f} ({s2['smoke']['AP50']:.3f} / {s2['fire']['AP50']:.3f})",
             f"{m5['mean']['AP50']:.4f} ({m5['smoke']['AP50']:.3f} / {m5['fire']['AP50']:.3f})",
             f"{m6['mean']['AP50']:.4f} ({m6['smoke']['AP50']:.3f} / {m6['fire']['AP50']:.3f})"],
            ["Tiny fire AP", f"{f2['tiny']['AP50']:.3f}", f"{f5['tiny']['AP50']:.3f}", f"{f6['tiny']['AP50']:.3f}"],
            ["Small fire AP", f"{f2['small']['AP50']:.3f}", f"{f5['small']['AP50']:.3f}", f"{f6['small']['AP50']:.3f}"],
            ["Tiny fires found (confidence >= 0.5)", pct(f2["tiny"]["rec50"]), pct(f5["tiny"]["rec50"]),
             pct(f6["tiny"]["rec50"])],
            ["Small fires found (confidence >= 0.5)", pct(f2["small"]["rec50"]), pct(f5["small"]["rec50"]),
             pct(f6["small"]["rec50"])],
        ], [62 * mm, 37 * mm, 37 * mm, 37 * mm], highlight_col=3),
        Spacer(1, 3 * mm),
        Paragraph("False alarms", s["h2"]),
        KeepTogether([
            image(FIGS / "s56_false_alarms.png", 100),
            Paragraph(
                f"Flame pasting made the model treat small warm blobs as fire: {r5['fp_images']} normal images "
                f"triggered a detection against {r2['fp_images']} for stage 2, and the scene classifier's false "
                "'fire' rate rose from 1.4% to 6.6%. Without pasting (stage 6) most of that disappears "
                f"({r6['fp_images']} images, scene classifier 1.6%). The remaining increase over stage 2 comes "
                "from the higher resolution itself, which makes more small bright objects visible.", s["body"]),
        ]),
        PageBreak(),
        Paragraph("Video benchmark", s["h2"]),
        Paragraph(
            "Seven fires in two CCTV clips. An alarm requires 6 detections within 15 checked frames at "
            "confidence 0.80. Each cell shows the model's typical (median) confidence on that fire and the time "
            "from the fire appearing to the alarm.", s["body"]),
        table([
            ["Fire", "Size", "Stage 2 (640 px)", "Stage 6 (896 px)", "Stage 5 (paste) *"],
            *[[f["id"], f"{f['size_px']} px", vcell("s2_640", i), vcell("m6_896", i), vcell("m5_896_paste", i)]
              for i, f in enumerate(V["fires"])],
            ["Fires alarmed", "", "5 of 7", "7 of 7", "7 of 7"],
            ["False alarms in the clips", "", "0", "0", "0"],
        ], [50 * mm, 15 * mm, 36 * mm, 36 * mm, 36 * mm], highlight_col=3, font=8.8),
        Paragraph("* Biased in stage 5's favour: it was trained on pasted flames and these test fires are pasted "
                  "flames. Stage 6 never saw one, so its improvement (distant fire 0.61-0.63 -> 0.84-0.88) is the "
                  "fair measure of what higher resolution does.", s["caption"]),
    ]

    # screenshots
    shots = [("test_three_fires.png", 118, "Three small fires at different positions, all confirmed at the same "
                                           "time by stage 6 (0.89, 0.86 and 0.81). Stage 2 confirmed none of "
                                           "these at its original threshold."),
             ("test_distant.png", 80, "The distant fire at the far end, confirmed."),
             ("test_medium.png", 80, "A medium fire, confirmed.")]
    if all((ASSETS / n).exists() for n, _, _ in shots):
        pair = Table([[image(ASSETS / shots[1][0], shots[1][1]), image(ASSETS / shots[2][0], shots[2][1])],
                      [Paragraph(shots[1][2], s["caption"]), Paragraph(shots[2][2], s["caption"])]],
                     colWidths=[87 * mm, 87 * mm], hAlign="LEFT")
        pair.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "Calibri"), ("VALIGN", (0, 0), (-1, 0), "BOTTOM"),
                                  ("NOSPLIT", (0, 0), (-1, -1))]))
        story += [
            Paragraph("Stage 6 on the test video", s["h2"]),
            image(ASSETS / shots[0][0], shots[0][1]),
            Paragraph(shots[0][2], s["caption"]),
            pair,
        ]
    story += [PageBreak()]

    # 5 improved / not
    story += [
        Paragraph("5. What improved, and what did not", s["h1"]),
        Paragraph("Improved", s["h2"]),
        *bullets([
            f"<b>Small and distant fires now raise an alarm.</b> 7 of 7 benchmark fires against 5 of 7; the "
            "distant fire's confidence rose from about 0.62 to 0.84-0.88.",
            f"<b>Tiny-fire accuracy</b> on the test set rose from {f2['tiny']['AP50']:.3f} to "
            f"{f6['tiny']['AP50']:.3f}, and tiny fires found from {pct(f2['tiny']['rec50'])} to "
            f"{pct(f6['tiny']['rec50'])}.",
            f"<b>Best overall accuracy of the project</b>: mAP@0.5 {m6['mean']['AP50']:.4f} "
            f"(stage 2: {s2['mean']['AP50']:.4f}), with fire AP up by "
            f"{(m6['fire']['AP50'] - s2['fire']['AP50']) * 100:.1f} points.",
            "<b>One pass is enough.</b> The earlier workaround ran stage 2 at two sizes (640 + 960 px) per frame; "
            "stage 6 reaches the same 7 of 7 in a single pass.",
        ], s),
        Paragraph("Not improved, or still open", s["h2"]),
        *bullets([
            f"<b>More false detections on normal photos.</b> At confidence 0.90, {r6['fp_images']} normal test "
            f"images trigger a detection against {r2['fp_images']} for stage 2. At a strict 1% false-alarm budget "
            "stage 2 still detects more fire/smoke images (90% against 83%).",
            "<b>Not yet checked on the site footage.</b> Stage 2 gave zero false alarms on 54.5 minutes of real "
            "industrial video. Stage 6 has not been run on it yet.",
            "<b>The benchmark fires are edited in.</b> They do not light up their surroundings like real fire. "
            "A real small flame filmed on the site camera is still needed.",
            "<b>Cost.</b> 896 px needs about 2.2 times the computation of 640 px.",
            "<b>Smoke</b> is slightly lower than stage 2 (AP "
            f"{m6['smoke']['AP50']:.3f} against {s2['smoke']['AP50']:.3f}).",
        ], s),
        Paragraph("6. Next plan: making the model more robust", s["h1"]),
        table([
            ["#", "Step", "Why", "How it is checked"],
            ["1", Paragraph("<b>Hard-negative training</b>: add images of fire-coloured objects labelled 'not "
                            "fire' - yellow/orange hard hats, hi-vis vests, lamps, bright haze", s["cell"]),
             Paragraph("These are the known false triggers on site; it is the direct way to bring stage 6's false "
                       "alarms down", s["cell"]),
             Paragraph("False-alarm images at 0.90 fall from 27 toward 14; small-fire scores do not drop",
                       s["cell"])],
            ["2", Paragraph("<b>Site-footage false-alarm check</b> for stage 6", s["cell"]),
             Paragraph("The requirement is very few false alarms in 24/7 use", s["cell"]),
             Paragraph("Zero false alarms on the 54.5-minute footage, as stage 2 achieved", s["cell"])],
            ["3", Paragraph("<b>Real-fire test</b>: a small controlled flame filmed on the site camera at "
                            "several distances", s["cell"]),
             Paragraph("All small-fire evidence so far uses edited-in flames", s["cell"]),
             Paragraph("Same per-fire benchmark: confidence and time to alarm", s["cell"])],
            ["4", Paragraph("<b>Own labelled images</b> from the site camera, with one consistent box rule",
                            s["cell"]),
             Paragraph("Matches the model to the real camera, light and scene", s["cell"]),
             Paragraph("A held-out 'our camera' test set", s["cell"])],
            ["5", Paragraph("<b>Fast version</b>: train MobileNetV3 at 896 px with mosaic", s["cell"]),
             Paragraph("70 cameras need a cheaper model; MobileNetV3 was about 2.5 times faster at 640 px",
                       s["cell"]),
             Paragraph("Speed and the same three checks", s["cell"])],
            ["6", Paragraph("<b>Per-camera settings</b>: thresholds and far-end zones per camera", s["cell"]),
             Paragraph("Fixed cameras differ in distance, light and background", s["cell"]),
             Paragraph("Alarm log per camera", s["cell"])],
        ], [8 * mm, 62 * mm, 55 * mm, 49 * mm], right_from=9, font=9),
        Spacer(1, 3 * mm),
        Paragraph(
            "<b>Order.</b> Steps 1 and 2 come first: stage 6 becomes the main model only after hard-negative "
            "training and a clean result on the site footage. Step 3 can be prepared in parallel, since it only "
            "needs a recording.", s["callout"]),
    ]

    doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                          bottomMargin=16 * mm, title="Stage 5 and Stage 6 - small and distant fires",
                          author="Fire & Smoke project")

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Calibri", 8.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 9 * mm, "Fire & smoke detection - stage 5 and stage 6")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"page {_doc.page}")
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width,
                                                             doc.height, id="f")], onPage=footer)])
    doc.build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
