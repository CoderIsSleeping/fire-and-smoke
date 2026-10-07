"""Stage-1 model-selection report: the problem, the three backbones and two
detection heads compared, 40 epochs of results for each, and why the project
continues with DINOv3 + Faster R-CNN.

Run `python scripts/build_stage1_assets.py` first; it writes the figures and
reads every number from the training CSVs, evaluation reports and checkpoints.

    python scripts/build_stage1_report_pdf.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

import build_stage1_assets as assets
from build_stage56_report_pdf import BAND, GOOD, INK, MUTED, RULE, bullets, image, register_fonts, styles, table

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "Stage1_Model_Selection_Report.pdf"


def facts_table(rows, s, widths=(42 * mm, 132 * mm)):
    data = [[Paragraph(f"<b>{k}</b>", s["cell"]), Paragraph(v, s["cell"])] for k, v in rows]
    t = Table(data, colWidths=list(widths), hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2), ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE), ("BACKGROUND", (0, 0), (0, -1), BAND),
        ("FONTNAME", (0, 0), (-1, -1), "Calibri"),
    ]))
    return t


def pros_cons(pros, cons, s):
    def col(title, items, colour):
        body = "<br/>".join(f"&bull;&nbsp;{i}" for i in items)
        return Paragraph(f"<font color='{colour}'><b>{title}</b></font><br/>{body}", s["cell"])

    t = Table([[col("Advantages", pros, "#2e7d32"), col("Disadvantages", cons, "#c62828")]],
              colWidths=[87 * mm, 87 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (0, 0), GOOD),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#fdecea")), ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("FONTNAME", (0, 0), (-1, -1), "Calibri"),
    ]))
    return t


def epoch_table(m):
    head = ["Epoch", "Train loss", "Smoke AP", "Fire AP", "mAP@0.5", "mAP@.5:.95", "Detected *", "False alarms *"]
    rows = [head] + [[str(e["epoch"]), f"{e['loss']:.4f}", f"{e['smoke']:.4f}", f"{e['fire']:.4f}",
                      f"{e['mAP50']:.4f}", f"{e['mAP50_95']:.4f}", f"{e['recall'] * 100:.1f}%",
                      f"{e['fpr'] * 100:.2f}%"] for e in m["epochs"]]
    t = Table(rows, colWidths=[15 * mm, 22 * mm, 21 * mm, 21 * mm, 22 * mm, 24 * mm, 23 * mm, 26 * mm], hAlign="LEFT")
    best_row = m["best_epoch"]
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Calibri"), ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("LEADING", (0, 0), (-1, -1), 8.6),
        ("FONTNAME", (0, 0), (-1, 0), "Calibri-Bold"), ("BACKGROUND", (0, 0), (-1, 0), BAND),
        ("ALIGN", (0, 0), (-1, -1), "RIGHT"), ("TOPPADDING", (0, 0), (-1, -1), 0.75),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.75), ("LINEBELOW", (0, 0), (-1, -1), 0.2, RULE),
        ("BACKGROUND", (0, best_row), (-1, best_row), GOOD), ("FONTNAME", (0, best_row), (-1, best_row), "Calibri-Bold"),
    ]))
    return t


def build() -> None:
    register_fonts()
    d = assets.load()
    M = d["models"]
    dino, mb, rn, fc = M["dino"], M["mbv3"], M["r18"], M["fcos"]
    s = styles()
    H1, H2, P, C = (lambda t: Paragraph(t, s["h1"])), (lambda t: Paragraph(t, s["h2"])), \
                   (lambda t: Paragraph(t, s["body"])), (lambda t: Paragraph(t, s["caption"]))

    def mp(x):  # millions
        return f"{x / 1e6:.2f}M" if x < 1e7 else f"{x / 1e6:.1f}M"

    def ms(m):
        a, b = m["cpu_ms"]
        return f"{a}" if a == b else f"{a}-{b}"

    def tmap(m, cls="mean"):
        return m["test"][cls]["AP50"]

    def at(m, ep, key="mAP50"):
        return m["epochs"][ep - 1][key]

    order = (dino, mb, rn, fc)
    story = [
        Spacer(1, 14 * mm),
        Paragraph("Fire and Smoke Detection", s["title"]),
        Paragraph("Model selection report (Stage 1)<br/>Three backbones and two detection heads compared over 40 "
                  f"training epochs<br/>{date.today():%d %B %Y}", s["subtitle"]),
        Spacer(1, 7 * mm),
        Paragraph(
            "<b>Summary.</b> The task is an industrial fire and smoke detector for fixed cameras that must run "
            "around the clock with very few false alarms. Before investing in one design, four models were "
            "trained under identical conditions for 40 epochs: three with the same Faster R-CNN detector on "
            "different backbones (DINOv3 ViT-S, MobileNetV3-Large, ResNet-18), and one with the FCOS detector. "
            f"<b>DINOv3 + Faster R-CNN was the most accurate on every measure</b> (validation mAP@0.5 "
            f"{dino['best_val']:.4f}, test {tmap(dino):.4f}) and kept false alarms lowest at the same detection "
            "rate, so the project continues with it. MobileNetV3 is kept as the candidate for a faster version.",
            s["callout"]),
        table([
            ["Model", "Best val mAP@0.5", "Test mAP@0.5", "Parameters", "CPU time / frame"],
            *[[m["label"], f"{m['best_val']:.4f} (epoch {m['best_epoch']})", f"{tmap(m):.4f}",
               mp(m["arch"]["total"]), f"{ms(m)} ms"] for m in order],
        ], [62 * mm, 36 * mm, 26 * mm, 23 * mm, 27 * mm], highlight_col=None),
        C("All four trained for 40 epochs on the same data with the backbone frozen. CPU time measured on the project "
          "laptop (batch 1, 640 px). Notes on measurement are in Appendix B."),
        PageBreak(),
    ]

    # 1 problem statement
    cell = lambda t: Paragraph(t, s["cell"])
    story += [
        H1("1. Problem statement"),
        P("An industrial site needs fire and smoke detected automatically from its existing surveillance cameras. "
          "The conditions make this harder than ordinary fire detection in photographs:"),
        table([
            ["#", "Problem to solve", "What it demands from the system"],
            ["1", cell("Cameras are mounted high, in poor industrial lighting; fire is a bright light source"),
             cell("Robust features under low light and glare; training with low-light and infrared-style images")],
            ["2", cell("The system runs 24 hours a day, 7 days a week"),
             cell("<b>False alarms must be very rare.</b> The false-alarm rate is measured alongside accuracy, "
                  "and an alarm needs repeated detections over time")],
            ["3", cell("A fire can be hidden behind equipment"),
             cell("Detect it from the light it casts on nearby surfaces: a whole-image classifier plus "
                  "glow statistics")],
            ["4", cell("Smoke must be detected when it is visible (invisible smoke is out of scope)"),
             cell("Two classes, smoke and fire, each with its own boxes")],
            ["5", cell("Fires can be small or far from the camera"), cell("Features at several scales (a feature pyramid)")],
            ["6", cell("Many cameras (about 70), at least 10 frames per second overall"),
             cell("Speed and model size matter as well as accuracy; a lighter variant must be possible")],
            ["7", cell("Work first on images, then on video"),
             cell("Train and benchmark on a labelled image dataset; add temporal confirmation for video")],
        ], [8 * mm, 78 * mm, 88 * mm], right_from=9),
        Spacer(1, 2 * mm),
        P("These requirements lead to one central design question, which this report answers with measurements: "
          "<b>which backbone and which detection head should the detector be built on?</b>"),
        H1("2. Approach"),
        image(FIGS / "s1_pipeline.png", 172),
        C("The detector. Stage 1 compares three choices for the backbone and two for the detection head; everything "
          "else stays the same."),
        *bullets([
            "<b>Backbone</b>: a pretrained network that turns the image into features. Candidates: DINOv3 ViT-S/16, "
            "MobileNetV3-Large, ResNet-18.",
            "<b>Feature pyramid</b>: four feature maps at different scales, so both large and small fires are covered.",
            "<b>Detection head</b>: turns features into boxes with a class and a score. Candidates: Faster R-CNN "
            "(two-stage) and FCOS (one-stage).",
            "<b>Scene classifier</b>: a second, whole-image output, P(smoke) and P(fire), that also reads seven "
            "'glow' statistics. It addresses problem 3 (hidden fire).",
            "<b>Stage 1 rule</b>: the backbone is <i>frozen</i>. Only the pyramid, the head and the scene classifier "
            "are trained. This tests how good each backbone's ready-made features are for fire and smoke.",
        ], s),
        PageBreak(),
    ]

    # 3 backbones
    a = dino["arch"]
    story += [
        H1("3. The backbone candidates"),
        H2("3.1 DINOv3 ViT-S/16"),
        P("DINOv3 is a Vision Transformer trained by Meta AI with <i>self-supervised</i> learning: it learned from "
          "1.689 billion unlabelled images by matching its own predictions across different views of the same "
          "image (self-distillation with no labels, hence DI-NO). Because no labels were used, its features are "
          "general rather than tuned to one classification task. ViT-S/16 is the small member of the family."),
        image(FIGS / "s1_arch_dino.png", 172),
        facts_table([
            ("Input", "RGB image, 3 x 640 x 640, normalised"),
            ("How it works", "The image is cut into 16 x 16-pixel patches (40 x 40 = 1,600 patches). Each patch becomes "
                             "a 384-number vector, a 'token'. Twelve transformer blocks then let every token exchange "
                             "information with every other token (self-attention), so each one comes to describe its "
                             "patch in the context of the whole image."),
            ("Output", "1,600 tokens x 384 values: a feature map at 1/16 resolution. It gives no classes or boxes by "
                       "itself; a detection head is added on top."),
            ("Layers", "12 transformer blocks. Each has 2 LayerNorms, a 6-head self-attention layer and a 2-layer MLP "
                       "(384 -> 1,536 -> 384, GELU), with skip connections. 48 linear layers in total, plus one "
                       "convolution for the patch embedding."),
            ("Parameters", f"{mp(a['trunk'])} in the backbone (frozen in stage 1). Whole detector: {mp(a['total'])}, of which "
                           f"{mp(a['trainable_frozen_trunk'])} are trained."),
            ("How we train it", "Backbone frozen. Four of its blocks (6, 9, 10, 12) are tapped and resampled into a "
                                f"4-level pyramid ({mp(a['neck'])} parameters), which feeds the detection head."),
        ], s),
        Spacer(1, 2 * mm),
        pros_cons([
            "Strongest ready-made features: best accuracy with the backbone frozen",
            "Every token sees the whole image from the first block (global context), useful for smoke, which has no "
            "fixed shape",
            "Trained on 1,300 times more images than the two CNNs, with no labels",
            "Lowest false-alarm rate at the same detection rate in our tests",
        ], [
            "Slowest: about 900 ms per frame on CPU, 2.5 times the CNNs",
            "Self-attention cost grows quickly with image size",
            "Produces one scale only, so a pyramid has to be built from it",
            "Largest model (39.3M parameters with the detector)",
        ], s),
        PageBreak(),
    ]
    a = mb["arch"]
    story += [
        H2("3.2 MobileNetV3-Large"),
        P("MobileNetV3 (Google, 2019) is a convolutional network designed for phones and embedded devices. Its "
          "structure was found partly by automated architecture search, with the goal of the best accuracy for a "
          "given amount of computation. It was pretrained with labels on ImageNet-1k (1.28 million images, "
          "1,000 classes)."),
        image(FIGS / "s1_arch_mobilenetv3.png", 172),
        facts_table([
            ("Input", "RGB image, 3 x 640 x 640, normalised"),
            ("How it works", "A stack of 'inverted residual' blocks. Each block widens the channels with a 1 x 1 "
                             "convolution, filters each channel separately with a cheap <i>depthwise</i> convolution, "
                             "optionally re-weights channels (squeeze-and-excite), then narrows again. The image gets "
                             "smaller and the channels more numerous stage by stage."),
            ("Output", "Feature maps at several scales. We use three: 80 x 80 (40 channels), 40 x 40 (112 channels) and "
                       "20 x 20 (960 channels)."),
            ("Layers", f"15 inverted-residual blocks in 7 stages, {a['trunk_leaf_layers']['Conv2d']} convolutions; "
                       "hard-swish and ReLU activations."),
            ("Parameters", f"{mp(a['trunk'])} in the backbone (frozen in stage 1). Whole detector: {mp(a['total'])}, of which "
                           f"{mp(a['trainable_frozen_trunk'])} are trained."),
            ("How we train it", "Backbone and its BatchNorm statistics frozen. The three maps go through a standard "
                                f"Feature Pyramid Network ({mp(a['neck'])} parameters) that also adds a fourth, coarser level."),
        ], s),
        Spacer(1, 2 * mm),
        pros_cons([
            "Smallest backbone (2.9M parameters) and about 2.5 times faster than DINOv3 on CPU",
            "Gives several scales naturally, which suits a feature pyramid",
            "Second-best accuracy, only 2.3-2.6 points of mAP behind DINOv3",
            "Well supported on edge hardware",
        ], [
            "Features come from a classification task, so they are less general",
            "Each layer sees only a local neighbourhood",
            "More false alarms than DINOv3 at the same detection rate (0.87% against 0.37% on validation)",
            "Lower smoke accuracy than DINOv3",
        ], s),
        PageBreak(),
    ]
    a = rn["arch"]
    story += [
        H2("3.3 ResNet-18"),
        P("ResNet (Microsoft Research, 2015) introduced the <i>skip connection</i>: each block adds its input back "
          "to its output, which made deep networks trainable. ResNet-18 is the smallest standard member and a "
          "common baseline. It was pretrained with labels on ImageNet-1k."),
        image(FIGS / "s1_arch_resnet18.png", 172),
        facts_table([
            ("Input", "RGB image, 3 x 640 x 640, normalised"),
            ("How it works", "A 7 x 7 convolution and a pooling layer shrink the image to a quarter. Four stages follow, "
                             "each with two residual blocks of two 3 x 3 convolutions. Each stage after the first "
                             "halves the resolution and doubles the channels (64, 128, 256, 512)."),
            ("Output", "We use three maps: 80 x 80 (128 channels), 40 x 40 (256 channels) and 20 x 20 (512 channels)."),
            ("Layers", f"8 residual blocks, {a['trunk_leaf_layers']['Conv2d']} convolutions (17 main + 3 on shortcuts), "
                       "each followed by BatchNorm; ReLU activations. The name counts 17 convolutions + 1 classifier "
                       "layer, which we remove."),
            ("Parameters", f"{mp(a['trunk'])} in the backbone (frozen in stage 1). Whole detector: {mp(a['total'])}, of which "
                           f"{mp(a['trainable_frozen_trunk'])} are trained."),
            ("How we train it", "Same as MobileNetV3: backbone and BatchNorm statistics frozen, a standard Feature "
                                f"Pyramid Network on top ({mp(a['neck'])} parameters)."),
        ], s),
        Spacer(1, 2 * mm),
        pros_cons([
            "Simple and very well understood; supported everywhere",
            "Regular 3 x 3 convolutions run efficiently on most hardware",
            "Several scales naturally",
        ], [
            "Lowest accuracy of the three backbones (validation mAP 0.6567)",
            "Slower than MobileNetV3 here (414 ms against about 365 ms)",
            "Misses many more fires at a low false-alarm rate: 68% detected against 87% for the other two",
            "Its frozen features fit the task worst: training loss stays highest",
        ], s),
        PageBreak(),
        H2("3.4 Backbones side by side"),
        table([
            ["", "DINOv3 ViT-S/16", "MobileNetV3-Large", "ResNet-18"],
            ["Type", "Vision Transformer", "Convolutional (mobile)", "Convolutional (residual)"],
            ["Building block", "12 transformer blocks", "15 inverted-residual blocks", "8 residual blocks"],
            ["Backbone parameters", mp(dino["arch"]["trunk"]), mp(mb["arch"]["trunk"]), mp(rn["arch"]["trunk"])],
            ["Pretraining", "self-supervised, 1.689 billion images", "supervised, ImageNet-1k", "supervised, ImageNet-1k"],
            ["What each layer sees", "the whole image", "a local neighbourhood", "a local neighbourhood"],
            ["Feature scales produced", "one (pyramid built from it)", "several", "several"],
            ["Detector total / trained", f"{mp(dino['arch']['total'])} / {mp(dino['arch']['trainable_frozen_trunk'])}",
             f"{mp(mb['arch']['total'])} / {mp(mb['arch']['trainable_frozen_trunk'])}",
             f"{mp(rn['arch']['total'])} / {mp(rn['arch']['trainable_frozen_trunk'])}"],
        ], [44 * mm, 46 * mm, 42 * mm, 42 * mm], font=8.8),
        Spacer(1, 4 * mm),
        image(FIGS / "s1_params.png", 150),
        C("Where the parameters are. In all three Faster R-CNN detectors most trained parameters sit in the detection "
          "head, so the three train nearly the same amount; what differs is the frozen backbone."),
        PageBreak(),
    ]

    # 4 heads
    story += [
        H1("4. The detection-head candidates"),
        H2("4.1 Faster R-CNN (two-stage)"),
        image(FIGS / "s1_arch_frcnn.png", 172),
        facts_table([
            ("Input", "The 4-level feature pyramid (256 channels per level)"),
            ("How it works", "<b>Stage 1</b>, the Region Proposal Network, slides over every location of every level "
                             "and, for 6 preset 'anchor' boxes there, predicts whether an object is present and how to "
                             "adjust the box. The best ~300 are kept as proposals. <b>Stage 2</b> crops the features "
                             "of each proposal to a fixed 7 x 7 patch (RoIAlign) and passes them through two "
                             "fully-connected layers, which decide the class and refine the box."),
            ("Output", "Boxes, each with a class (smoke or fire) and a confidence score from 0 to 1"),
            ("Parameters", "14.5M, all trained: proposal network 0.60M, box head 13.91M"),
            ("Training losses", "Four: objectness and box loss for the proposal network, class and box loss for the box head"),
        ], s),
        Spacer(1, 2 * mm),
        pros_cons([
            "Most accurate in our tests",
            "Every detection is examined twice, so confident detections are well separated from doubtful ones: a "
            "threshold of 0.90 is usable",
            "Mature and widely used; handles objects of very different sizes",
        ], [
            "Large head (14.5M parameters)",
            "The number of proposals changes from image to image, so the network cannot be exported as one "
            "fixed-shape graph for batching many cameras (tested: export fails for batch sizes above 1)",
            "Needs anchor sizes to be chosen",
        ], s),
        PageBreak(),
        H2("4.2 FCOS (one-stage, anchor-free)"),
        image(FIGS / "s1_arch_fcos.png", 172),
        facts_table([
            ("Input", "The same 4-level feature pyramid"),
            ("How it works", "No proposals and no anchors. Every location of every level (8,500 locations at 640 px) "
                             "directly predicts three things through two small convolution towers: a class score, "
                             "the distances to the four sides of the box, and a 'centre-ness' value that lowers the "
                             "score of points near an object's edge."),
            ("Output", "Boxes, class and score. The score is sqrt(class score x centre-ness)."),
            ("Parameters", f"{mp(fc['arch']['det_head'])}, all trained (2 convolutions per tower, 128 channels in our configuration)"),
            ("Training losses", "Three: focal loss (class), GIoU loss (box), binary cross-entropy (centre-ness)"),
        ], s),
        Spacer(1, 2 * mm),
        pros_cons([
            "Simple: no anchors, no proposals, 24 times fewer head parameters than Faster R-CNN",
            "Always outputs the same fixed-size tensors, so many cameras can be batched (export verified)",
            "Fewer settings to tune",
        ], [
            "Lower accuracy in our trained configuration",
            "Scores are compressed (they never exceed about 0.75), so its alarm threshold has to sit at 0.55",
            "Not faster by itself: on the same backbone it took 924 ms against 884 ms for Faster R-CNN",
        ], s),
        PageBreak(),
    ]

    # 5 setup
    e0 = dino["arch"]["args"]
    story += [
        H1("5. How the comparison was run"),
        P("The comparison is only meaningful if nothing changes except the part being compared. All four models "
          "used the same data, the same splits, the same augmentation and the same training settings."),
        H2("Data"),
        table([
            ["D-Fire dataset split", "Images", "With no fire or smoke", "Smoke boxes", "Fire boxes"],
            ["Training", "13,776", "6,223", "7,660", "9,659"],
            ["Validation (results after every epoch)", "3,445", "1,610", "1,890", "2,155"],
            ["Test (used once, at the end)", "4,306", "2,005", "2,311", "2,878"],
        ], [66 * mm, 22 * mm, 36 * mm, 26 * mm, 24 * mm]),
        C("About 45% of the images contain no fire or smoke. They are what makes it possible to measure false alarms."),
        H2("Training settings (identical for all four models)"),
        facts_table([
            ("Epochs", "40 (the FCOS model ran 45; its first 40 are reported here)"),
            ("Backbone", "Frozen. Only the feature pyramid, the detection head and the scene classifier are trained"),
            ("Optimiser", f"AdamW, learning rate {e0['lr']}, weight decay {e0['weight_decay']}"),
            ("Schedule", f"{e0['warmup_iters']}-step warm-up, then cosine decay to 1% of the starting rate"),
            ("Batch and precision", f"Batch size {e0['batch']}, mixed precision (16-bit), gradient clipping at 10"),
            ("Input size", f"{e0['imgsz']} x {e0['imgsz']} pixels (aspect ratio kept, padded)"),
            ("Augmentation", "Low-light simulation, infrared-style grey images, hiding part of a flame, random crop, "
                             "colour jitter, flip, motion blur"),
            ("Model selection", "The epoch with the best validation mAP@0.5 is kept"),
            ("Hardware", "Kaggle T4 GPU, about 8-11 minutes per epoch"),
        ], s),
        H2("What is measured after every epoch"),
        *bullets([
            "<b>AP@0.5</b> for smoke and for fire: how accurately boxes are found, counting a box as correct when it "
            "overlaps the labelled box by at least 50%. <b>mAP@0.5</b> is their average and is the main benchmark.",
            "<b>mAP@0.5:0.95</b>: the same averaged over stricter overlap requirements.",
            "<b>Detected at <= 1% false alarms</b>: the confidence threshold is raised until at most 1% of the "
            "fire-free images trigger a detection, then the share of fire/smoke images still detected is reported. "
            "This is the number that matters for an alarm system.",
            "<b>Training loss</b>: how well the model fits the training images.",
        ], s),
        PageBreak(),
    ]

    # 6 results backbones
    story += [
        H1("6. Results: backbone comparison"),
        P("Three models, identical except for the backbone, each with the Faster R-CNN head."),
        image(FIGS / "s1_val_map.png", 160),
        C("Validation mAP@0.5 after each of the 40 epochs. The order DINOv3 > MobileNetV3 > ResNet-18 holds at every "
          "epoch. The FCOS model is shown for reference and discussed in section 7."),
        table([
            ["Validation mAP@0.5 at epoch", "1", "5", "10", "20", "30", "40", "Best"],
            *[[m["label"], *[f"{at(m, e):.4f}" for e in (1, 5, 10, 20, 30, 40)],
               f"{m['best_val']:.4f} (ep {m['best_epoch']})"] for m in order],
        ], [56 * mm, 15 * mm, 15 * mm, 15 * mm, 15 * mm, 15 * mm, 15 * mm, 28 * mm], font=8.6),
        C("Results for every epoch of every model are in Appendix A."),
        image(FIGS / "s1_class_ap.png", 160),
        C("Per class. Fire is harder than smoke for every model; DINOv3 leads on both."),
        PageBreak(),
        H2("Test set"),
        image(FIGS / "s1_test_bars.png", 150),
        table([
            ["Test split (4,306 images)", "Smoke AP", "Fire AP", "mAP@0.5", "mAP@0.5:0.95"],
            *[[m["label"], f"{tmap(m, 'smoke'):.4f}", f"{tmap(m, 'fire'):.4f}", f"{tmap(m):.4f}",
               f"{m['test']['mean']['AP50_95']:.4f}"] for m in order],
        ], [66 * mm, 26 * mm, 26 * mm, 26 * mm, 30 * mm]),
        C("The test set was not used for training or for choosing the epoch. The ranking is the same as on validation."),
        H2("False alarms against detections"),
        image(FIGS / "s1_alarm.png", 128),
        table([
            ["At <= 1% false alarms (validation)", "Threshold", "False-alarm rate", "Fire/smoke images detected"],
            *[[m["label"], f"{m['val_op']['threshold']:.2f}", f"{m['val_op']['fpr'] * 100:.2f}%",
               f"{m['val_op']['recall_any'] * 100:.1f}%"] for m in order],
        ], [70 * mm, 24 * mm, 34 * mm, 46 * mm]),
        C("DINOv3 and MobileNetV3 detect the same share of fire/smoke images, but DINOv3 does it with less than half "
          "the false alarms. ResNet-18 detects far fewer."),
        PageBreak(),
        H2("Speed"),
        image(FIGS / "s1_speed_acc.png", 128),
        C("Accuracy against speed on the project laptop's CPU. MobileNetV3 is the best trade-off for a fast version; "
          "ResNet-18 is both slower and less accurate than MobileNetV3."),
        *bullets([
            f"<b>DINOv3</b> is the most accurate (test mAP {tmap(dino):.4f}) and the slowest ({ms(dino)} ms).",
            f"<b>MobileNetV3</b> is {(tmap(dino) - tmap(mb)) * 100:.1f} points behind on test and about 2.5 times faster.",
            f"<b>ResNet-18</b> is {(tmap(dino) - tmap(rn)) * 100:.1f} points behind and not faster than MobileNetV3.",
            "The three Faster R-CNN detectors train almost the same number of parameters (16.8M-17.7M), because most "
            "of them are in the head. The difference in results therefore comes from the frozen backbone's features.",
        ], s),
        PageBreak(),
    ]

    # 7 heads
    hs = d["head_speed"]
    story += [
        H1("7. Results: detection-head comparison"),
        P("Two detectors were trained to completion: DINOv3 ViT-S with Faster R-CNN, and a lighter DINOv3 ViT-Ti with "
          "FCOS. FCOS was built as the deployable, light option, so this pair differs in <b>two</b> things at once: "
          "the head and the size of the backbone. The accuracy gap below therefore cannot be attributed to the "
          "head alone; the speed and export measurements, made on the same backbone, can."),
        table([
            ["", "DINOv3 ViT-S + Faster R-CNN", "DINOv3 ViT-Ti + FCOS"],
            ["Detector type", "two-stage, anchor-based", "one-stage, anchor-free"],
            ["Backbone / head parameters", f"{mp(dino['arch']['trunk'])} / {mp(dino['arch']['det_head'])}",
             f"{mp(fc['arch']['trunk'])} / {mp(fc['arch']['det_head'])}"],
            ["Trained parameters", mp(dino["arch"]["trainable_frozen_trunk"]), mp(fc["arch"]["trainable_frozen_trunk"])],
            ["Best validation mAP@0.5 (40 epochs)", f"{dino['best_val']:.4f}", f"{fc['best_val']:.4f}"],
            ["Test mAP@0.5 (smoke / fire)", f"{tmap(dino):.4f} ({tmap(dino, 'smoke'):.3f} / {tmap(dino, 'fire'):.3f})",
             f"{tmap(fc):.4f} ({tmap(fc, 'smoke'):.3f} / {tmap(fc, 'fire'):.3f})"],
            ["Detected at <= 1% false alarms (validation)", f"{dino['val_op']['recall_any'] * 100:.1f}%",
             f"{fc['val_op']['recall_any'] * 100:.1f}%"],
            ["Usable alarm threshold", f"{dino['val_op']['threshold']:.2f}", f"{fc['val_op']['threshold']:.2f} (scores top out near 0.75)"],
            ["CPU time per frame", f"{ms(dino)} ms", f"{ms(fc)} ms"],
            ["Export as one batched graph (many cameras)", "fails above batch size 1", "works, verified"],
        ], [66 * mm, 54 * mm, 54 * mm], font=8.8),
        Spacer(1, 3 * mm),
        H2("Does the head make the model faster? Measured on the same backbone"),
        table([["Configuration", "CPU ms / frame", "Parameters"], *[[n, str(v), p] for n, v, p in hs]],
              [86 * mm, 34 * mm, 30 * mm]),
        C("Same DINOv3 ViT-S backbone in the first four rows. Swapping the head alone does not speed the model up; "
          "the backbone is about two thirds of the computation."),
        *bullets([
            "<b>Accuracy.</b> Faster R-CNN's model is "
            f"{(tmap(dino) - tmap(fc)) * 100:.1f} points of test mAP ahead, and detects "
            f"{(dino['val_op']['recall_any'] - fc['val_op']['recall_any']) * 100:.0f} points more fire/smoke images at "
            "the 1% false-alarm budget. Part of this comes from the smaller backbone.",
            "<b>Scores.</b> Faster R-CNN gives clearly separated scores, so a strict threshold (0.90) still keeps "
            "most detections. FCOS multiplies two numbers below 1, so its scores are compressed and the threshold "
            "must be set low and precisely.",
            "<b>Speed.</b> FCOS is not faster on the same backbone (924 ms against 884 ms). The light model is "
            "faster because of its smaller backbone.",
            "<b>Deployment.</b> FCOS's real advantage is the fixed-shape output, which allows frames from many "
            "cameras to be processed as one batch. Faster R-CNN cannot be exported that way.",
        ], s),
        PageBreak(),
    ]

    # 8 training behaviour
    def gain(m):
        return at(m, 40) - at(m, 30)

    story += [
        H1("8. Training behaviour: did any model overfit?"),
        image(FIGS / "s1_train_loss.png", 160),
        C("Training loss over 40 epochs. Faster R-CNN and FCOS use different loss functions, so their values are not "
          "comparable with each other, only their shapes."),
        table([
            ["Model", "Train loss, epoch 1 -> 40", "Val mAP, epoch 30 -> 40", "Best epoch", "Trained parameters"],
            *[[m["label"], f"{at(m, 1, 'loss'):.3f} -> {at(m, 40, 'loss'):.3f}",
               f"{at(m, 30):.4f} -> {at(m, 40):.4f} ({gain(m) * 100:+.1f} pts)", str(m["best_epoch"]),
               mp(m["arch"]["trainable_frozen_trunk"])] for m in order],
        ], [56 * mm, 34 * mm, 44 * mm, 18 * mm, 28 * mm], font=8.6),
        Spacer(1, 2 * mm),
        Paragraph(
            "<b>No model overfitted in 40 epochs.</b> Overfitting would show as training loss still falling while "
            "validation accuracy turns downward. That did not happen for any of the four: validation mAP was "
            "still rising in the last ten epochs for every model, and the best epoch was 38 or 39 in every case.",
            s["callout"]),
        H2("Why there was no overfitting"),
        *bullets([
            "<b>The backbone is frozen.</b> The part of the network that could memorise the training images most "
            "easily is not being updated at all.",
            "<b>Strong augmentation.</b> Each image is seen in a different crop, brightness and colour every epoch.",
            "<b>45% of the training images contain no fire or smoke</b>, which constantly penalises false detections.",
            "<b>The best validation epoch is kept</b>, so a late decline would not have been used in any case.",
        ], s),
        H2("What the smaller and weaker models show instead: underfitting"),
        *bullets([
            f"<b>ResNet-18</b> ends with the highest training loss of the Faster R-CNN models "
            f"({at(rn, 40, 'loss'):.3f} against {at(dino, 40, 'loss'):.3f} for DINOv3) while training almost the same "
            "number of parameters. With the same head and the same data, the only difference is the frozen "
            "features: they describe fire and smoke less well, so the head cannot fit even the training set as "
            "closely.",
            f"<b>The FCOS model</b> trains only {mp(fc['arch']['trainable_frozen_trunk'])} parameters, twelve times fewer "
            "than the others, on a backbone half as wide. It was still improving fastest at the end "
            f"({gain(fc) * 100:+.1f} points in the last ten epochs): it has too little capacity rather than too much.",
            "<b>Being small and fast did not cause overfitting here; it limited how much the model could learn.</b> "
            "The risk of overfitting appears later, when backbone blocks are unfrozen, and that is why the next "
            "stage uses a much lower learning rate for the backbone.",
        ], s),
        PageBreak(),
    ]

    # 9 decision
    story += [
        H1("9. Decision: continue with DINOv3 + Faster R-CNN"),
        table([
            ["Criterion", "DINOv3 ViT-S", "MobileNetV3-L", "ResNet-18", "DINOv3 ViT-Ti"],
            ["Detection head", "Faster R-CNN", "Faster R-CNN", "Faster R-CNN", "FCOS"],
            ["Validation mAP@0.5", *[f"{m['best_val']:.4f}" for m in order]],
            ["Test mAP@0.5", *[f"{tmap(m):.4f}" for m in order]],
            ["Test fire AP", *[f"{tmap(m, 'fire'):.4f}" for m in order]],
            ["Detected at <= 1% false alarms", *[f"{m['val_op']['recall_any'] * 100:.1f}%" for m in order]],
            ["False-alarm rate at that point", *[f"{m['val_op']['fpr'] * 100:.2f}%" for m in order]],
            ["CPU ms per frame", *[ms(m) for m in order]],
            ["Batched export", "no", "no", "no", "yes"],
        ], [50 * mm, 31 * mm, 31 * mm, 31 * mm, 31 * mm], font=8.8, highlight_col=1),
        Spacer(1, 3 * mm),
        H2("Reasons"),
        *bullets([
            "<b>1. It is the most accurate.</b> Best mAP on validation and on test, best on both smoke and fire, and "
            "its validation mAP was the highest of the three backbones at every one of the 40 epochs.",
            "<b>2. It gives the fewest false alarms for the same detections.</b> The central requirement is very few "
            "false alarms in 24/7 use. At the 1% budget DINOv3 detects 86.9% of fire/smoke images with a 0.37% "
            "false-alarm rate; MobileNetV3 needs 0.87% for the same detections.",
            "<b>3. Its features are good without being trained.</b> With all backbones frozen, DINOv3's features gave "
            "the best result. Self-supervised training on 1.689 billion images transfers better to fire and smoke "
            "than classification training on ImageNet.",
            "<b>4. Faster R-CNN gives clean scores.</b> A strict threshold (0.90) remains usable, which is what a "
            "low-false-alarm system needs. FCOS's compressed scores make its threshold fragile.",
            "<b>5. It has room to improve.</b> Its curve was still rising at epoch 40, and none of its backbone has "
            "been fine-tuned yet.",
        ], s),
        H2("What this choice costs"),
        *bullets([
            "<b>Speed.</b> About 900 ms per frame on a laptop CPU, 2.5 times slower than MobileNetV3. Acceptable on a "
            "GPU, but it has to be addressed for 70 cameras.",
            "<b>No batched export</b> with Faster R-CNN.",
        ], s),
        P("Both costs concern deployment rather than detection quality, so accuracy is settled first. "
          "<b>MobileNetV3</b> is kept as the candidate for a fast version (close in accuracy, 2.5 times faster), and "
          "<b>FCOS</b> as the head that makes multi-camera batching possible. ResNet-18 is dropped: MobileNetV3 is "
          "both faster and more accurate."),
        PageBreak(),
        H1("10. Next goals"),
        table([
            ["#", "Goal", "Purpose"],
            ["1", cell("<b>Stage 2: fine-tune part of the backbone.</b> Unfreeze the last 2 transformer blocks of "
                       "DINOv3 with a learning rate 20 times lower than the head's"),
             cell("Adapt the features to fire and smoke without overfitting; raise accuracy further")],
            ["2", cell("<b>Small and distant fires.</b> Higher input resolution and scale augmentation (mosaic)"),
             cell("Fires far from the camera cover very few pixels")],
            ["3", cell("<b>Hard-negative training.</b> Add fire-coloured objects labelled as 'not fire': hard hats, "
                       "hi-vis clothing, lamps, bright haze"),
             cell("Bring false alarms down further for 24/7 use")],
            ["4", cell("<b>Video.</b> Run on industrial footage with temporal confirmation (6 detections in 15 frames)"),
             cell("Measure false alarms per hour and time to alarm")],
            ["5", cell("<b>A fast version for many cameras.</b> MobileNetV3 backbone and/or FCOS head, exported for "
                       "GPU batching"),
             cell("Meet the 70-camera, 10 fps requirement")],
            ["6", cell("<b>Data from the site's own cameras</b>, labelled with one consistent rule"),
             cell("Match the model to the real scene and lighting")],
        ], [8 * mm, 100 * mm, 66 * mm], right_from=9),
        H1("11. Project documents"),
        table([
            ["Document", "Content"],
            ["PROJECT_LOG.md", cell("Running log of every decision, measurement and change")],
            ["Model selection report (this document)", cell("Stage 1: problem, models, 40-epoch comparison, decision")],
            ["Model selection slides", cell("Ten-slide summary of this report")],
            ["Training report", cell("Dataset, architecture and training details of the DINOv3 detector")],
            ["Technical and research report", cell("Layer-by-layer architecture, loss functions and background research")],
            ["Model comparison report", cell("Faster R-CNN model against the light FCOS model in detail")],
            ["Source code and training scripts", cell("Training, evaluation, video inference and report builders")],
        ], [70 * mm, 104 * mm], right_from=9),
        PageBreak(),
    ]

    # appendix A
    story += [H1("Appendix A. Results of every epoch")]
    for i, m in enumerate(order):
        key = [k for k, v in M.items() if v is m][0]
        story += [
            H2(f"A.{i + 1}  {m['label']}"),
            image(FIGS / f"s1_train_{key}.png", 150),
            epoch_table(m),
            C("* Detected = share of fire/smoke validation images detected at the threshold that keeps false alarms on "
              "fire-free images at or below 1%; False alarms = the rate at that threshold. Best epoch highlighted."),
        ]
        story += [PageBreak()]

    # appendix B
    story += [
        H1("Appendix B. Notes on measurement"),
        *bullets([
            "<b>Epochs.</b> Three models were trained for 40 epochs. The FCOS model was trained for 45; this report "
            f"shows its first 40 (best {fc['best_val']:.4f} at epoch {fc['best_epoch']}). Its test result comes from "
            f"its best checkpoint overall (epoch {fc['ckpt_epoch']}, validation {fc['ckpt_val']:.4f}).",
            "<b>Test-time settings.</b> Part-way through the project the detector's test-time settings were changed "
            "for speed (300 proposals instead of 1,000, at most 20 boxes per image instead of 50). The DINOv3 "
            "results in this report were recorded before the change and the two CNN results after it. The change "
            "was measured to cost about 0.005 mAP, so DINOv3's lead over the CNNs is about half a point smaller "
            "than tabulated. It remains ahead by about 2 points over MobileNetV3 and 6-7 over ResNet-18.",
            "<b>Head comparison.</b> The two trained detectors differ in both head and backbone size; see section 7.",
            "<b>CPU timings</b> are from the project laptop (batch 1, 640 px, 32-bit, median of 15 runs); ranges show "
            "two measurement sessions. They indicate relative speed, not deployment speed on a GPU.",
            "<b>Alarm figures</b> are per image. On video an alarm additionally needs 6 detections within 15 frames, "
            "which removes most single-frame mistakes.",
        ], s),
    ]

    doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                          bottomMargin=15 * mm, title="Fire and smoke detection - model selection (stage 1)",
                          author="Fire & Smoke project")

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Calibri", 8.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 8.5 * mm, "Fire and smoke detection - model selection (stage 1)")
        canvas.drawRightString(A4[0] - 18 * mm, 8.5 * mm, f"page {_doc.page}")
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width,
                                                             doc.height, id="f")], onPage=footer)])
    doc.build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
