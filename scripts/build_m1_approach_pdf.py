from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT = "output/pdf/m1_fire_smoke_detection_approach.pdf"


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#C9D4DF"))
    canvas.setLineWidth(0.6)
    canvas.line(doc.leftMargin, height - 0.55 * inch, width - doc.rightMargin, height - 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#55606A"))
    canvas.drawString(doc.leftMargin, height - 0.42 * inch, "M1 Fire and Smoke Detection - ML/DL Approach")
    canvas.drawRightString(width - doc.rightMargin, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


def make_doc():
    doc = BaseDocTemplate(
        OUTPUT,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.78 * inch,
        bottomMargin=0.65 * inch,
        title="M1 Fire and Smoke Detection Approach",
        author="Codex",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=header_footer)])

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#14213D"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubTitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#34495E"),
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=15,
            textColor=colors.HexColor("#0F5C7A"),
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=13.2,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#1F2933"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PlanBullet",
            parent=styles["Body"],
            leftIndent=13,
            firstLineIndent=-7,
            bulletIndent=0,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["Body"],
            fontSize=8.4,
            leading=11.5,
            textColor=colors.HexColor("#3E4C59"),
        )
    )

    story = []
    p = lambda text, style="Body": story.append(Paragraph(text, styles[style]))
    b = lambda text: story.append(Paragraph(text, styles["PlanBullet"], bulletText="-"))
    cell = lambda text: Paragraph(text, styles["Small"])
    head = lambda text: Paragraph(f"<b>{text}</b>", styles["Small"])

    p("M1: Fire and Smoke Detection Model", "TitleCenter")
    p(
        "Proposed approach for building a public-data base model first, followed by industry-specific fine-tuning and deployment calibration for elevated CCTV cameras.",
        "SubTitle",
    )

    p("1. Project Understanding", "Section")
    p(
        "The M1 module should detect visible fire and smoke from industrial CCTV feeds. The first version will be a base model trained on public fire/smoke datasets. After the industry provides real camera videos or images, the same model will be fine-tuned and calibrated on site-specific data. This staged approach reduces early dependency on unavailable plant footage while still keeping the final system realistic for the target environment."
    )
    b("Primary output: bounding boxes for fire and smoke with confidence scores.")
    b("Target environment: elevated industrial cameras, likely wide-angle views, changing illumination, dust, steam, glare, and partial occlusion.")
    b("Deployment direction: on-premises inference for multiple camera streams, eventually optimized through ONNX/TensorRT.")

    p("2. Why YOLO Is Suitable", "Section")
    p(
        "YOLO is a strong fit for M1 because it is fast, detection-focused, easy to train on bounding-box datasets, and widely supported for real-time video deployment. YOLOv8n should be used as the documented baseline because the M1 requirement mentions YOLOv8-nano. YOLO11n or YOLO11s can be trained as an improved comparison if allowed, then the final model can be selected using recall, false alarms, and FPS."
    )
    b("YOLOv8n: safest baseline for professor alignment.")
    b("YOLO11n/YOLO11s: newer comparison models with better practical scope if the evaluation supports them.")
    b("Higher input size such as 960 or 1280 is recommended because elevated cameras make smoke and flame regions small.")

    p("3. Public Datasets Selected for Base Training", "Section")
    dataset_table = Table(
        [
            [head("Dataset"), head("Use in M1"), head("Important Notes")],
            [
                cell("D-Fire"),
                cell("Main base dataset for fire/smoke object detection."),
                cell("YOLO-format labels, fire and smoke boxes, and many negative images. Good first dataset for the base detector."),
            ],
            [
                cell("FASDD"),
                cell("Adds larger open-flame/smoke detection coverage."),
                cell("Useful for additional visual diversity. Labels and format should be inspected before merging."),
            ],
            [
                cell("FLAME / FLAME 2"),
                cell("Adds aerial/elevated-view fire and smoke samples."),
                cell("Useful because industrial cameras are mounted high. Some labels may be classification-style, so conversion or selective use may be needed."),
            ],
        ],
        colWidths=[1.18 * inch, 2.05 * inch, 3.35 * inch],
        repeatRows=1,
    )
    dataset_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F1F5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F3D56")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("LEADING", (0, 0), (-1, -1), 10.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB7C4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(dataset_table)
    story.append(Spacer(1, 8))

    p("4. Immediate Workflow After Dataset Download", "Section")
    b("Verify folder structure, image counts, label availability, and license/citation notes for each dataset.")
    b("Convert all datasets into one YOLO-style project layout: images/train, images/val, images/test, labels/train, labels/val, labels/test.")
    b("Standardize the class names to exactly two classes: 0 = smoke and 1 = fire.")
    b("Preserve negative images using empty label files because they are essential for controlling false alarms.")
    b("Do not use random frame-level splitting for videos. Split by source video, camera, or scene so the test set is genuinely unseen.")

    p("5. Training Plan", "Section")
    training_table = Table(
        [
            [head("Stage"), head("Action"), head("Purpose")],
            [cell("Baseline"), cell("Train YOLOv8n on standardized public data."), cell("Creates the professor-aligned M1 base model.")],
            [cell("Comparison"), cell("Train YOLO11n or YOLO11s on the same data."), cell("Checks whether the newer model gives better recall/FPS.")],
            [cell("Validation"), cell("Evaluate on public test split plus hard-negative samples."), cell("Measures fire/smoke recall and false positives.")],
            [cell("Fine-tuning"), cell("Later train on industry CCTV frames/videos."), cell("Adapts model to real camera height, lighting, steam, dust, and plant background.")],
            [cell("Optimization"), cell("Export to ONNX/TensorRT."), cell("Prepares for real-time multi-camera deployment.")],
        ],
        colWidths=[1.18 * inch, 2.55 * inch, 2.85 * inch],
        repeatRows=1,
    )
    training_table.setStyle(dataset_table._cellStyles and TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF3E8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#254117")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.2),
            ("LEADING", (0, 0), (-1, -1), 10.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB7C4")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
    ))
    story.append(training_table)

    p("6. Industrial Fine-Tuning Data to Request Later", "Section")
    p(
        "The public base model will not be enough for final industry use. The industry should later provide representative video from the actual cameras, especially normal non-fire conditions. These hard negatives are as important as actual fire examples because the target false-alarm requirement is strict."
    )
    b("Normal daytime, night, and shift-change footage.")
    b("Steam vents, soot blowing, dust clouds, ash movement, welding, sparks, sunlight glare, rain, haze, and hot equipment glow.")
    b("Any controlled smoke/fire drills allowed by safety rules, captured from the same camera height and distance.")
    b("Camera metadata: camera ID, location, resolution, FPS, approximate mounting height, viewing angle, and ROI notes.")

    p("7. False Alarm Control", "Section")
    p(
        "Industrial fire/smoke detection should not trigger alarms from a single frame. The base YOLO detector should propose candidate boxes, then a temporal decision layer should confirm persistence and motion before raising an alert."
    )
    b("Use rolling-window voting across 8 to 16 frames.")
    b("Use optical flow or simple frame-difference features to verify smoke-like movement and growth.")
    b("Use camera-specific ROI masks to ignore known steam outlets or irrelevant background regions.")
    b("Keep separate thresholds for fire and smoke. Fire may be higher confidence; smoke may need stronger temporal confirmation.")

    p("8. Evaluation Metrics", "Section")
    b("Recall for confirmed fire/smoke events should be prioritized because missing a real fire is the worst failure.")
    b("Precision and false alarms per camera per day must be tracked, not only mAP.")
    b("Report mAP50, mAP50-95, class-wise precision, class-wise recall, confusion matrix, FPS, and event-level false alarms.")
    b("For deployment readiness, test at 10 FPS per camera and estimate performance for 10+ concurrent camera streams.")

    p("9. Final M1 Position", "Section")
    p(
        "The proposed M1 work should be presented as a staged industrial fire/smoke detection system: first, build a YOLO-based public-data base detector using D-Fire, FASDD, and FLAME/FLAME 2; second, standardize all datasets into two classes; third, train and compare YOLOv8n with a newer YOLO variant; fourth, add temporal confirmation to reduce false positives; and fifth, fine-tune with real industrial CCTV data when available. This approach is technically realistic, aligned with the M1 requirement, and suitable for later fusion with other safety modules."
    )

    p("Dataset Links", "Section")
    p("D-Fire: https://github.com/gaia-solutions-on-demand/DFireDataset", "Small")
    p("FASDD: https://github.com/OyamingO/FASDD", "Small")
    p("FLAME 2: https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset", "Small")
    p("YOLOv8 docs: https://docs.ultralytics.com/models/yolov8/", "Small")
    p("YOLO11 docs: https://docs.ultralytics.com/models/yolo11/", "Small")

    doc.build(story)


if __name__ == "__main__":
    make_doc()
