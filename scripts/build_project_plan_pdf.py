from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle


OUTPUT = "output/pdf/m1_dataset_model_training_future_plan.pdf"


def page_header(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#B8C5D1"))
    canvas.setLineWidth(0.6)
    canvas.line(doc.leftMargin, height - 0.55 * inch, width - doc.rightMargin, height - 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#52606D"))
    canvas.drawString(doc.leftMargin, height - 0.42 * inch, "M1 Fire and Smoke Detection - Project Plan")
    canvas.drawRightString(width - doc.rightMargin, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(
        OUTPUT,
        pagesize=A4,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.78 * inch,
        bottomMargin=0.65 * inch,
        title="M1 Fire and Smoke Detection Project Plan",
        author="Codex",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=page_header)])

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="PlanTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#102A43"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PlanSubtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.3,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#334E68"),
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.4,
            leading=15,
            textColor=colors.HexColor("#0B5E7A"),
            spaceBefore=11,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.8,
            textColor=colors.HexColor("#1F2933"),
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["Body"],
            fontSize=8.2,
            leading=10.8,
            textColor=colors.HexColor("#334E68"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletPlan",
            parent=styles["Body"],
            leftIndent=13,
            firstLineIndent=-7,
            bulletIndent=0,
            spaceAfter=3.8,
        )
    )

    story = []
    p = lambda text, style="Body": story.append(Paragraph(text, styles[style]))
    b = lambda text: story.append(Paragraph(text, styles["BulletPlan"], bulletText="-"))
    cell = lambda text: Paragraph(text, styles["Small"])
    head = lambda text: Paragraph(f"<b>{text}</b>", styles["Small"])

    p("M1 Fire and Smoke Detection", "PlanTitle")
    p(
        "Dataset preparation, YOLO model training plan, Kaggle workflow, external video verification, and future industry fine-tuning strategy.",
        "PlanSubtitle",
    )

    p("1. What We Are Building", "Section")
    p(
        "The current M1 goal is to build a visible fire and smoke detection model for industrial CCTV footage. The first model will be a base detector trained on public D-Fire data. This model is not the final plant-ready version; it is the foundation that will later be fine-tuned using actual industry camera videos and hard-negative scenes."
    )
    b("Detection classes: smoke and fire.")
    b("Detector output: bounding boxes, class labels, and confidence scores.")
    b("Target use case: elevated industrial cameras monitoring plant areas.")
    b("Final system direction: real-time multi-camera inference with temporal confirmation to reduce false alarms.")

    p("2. Dataset Work", "Section")
    p(
        "For the first training stage, only the D-Fire dataset is being used. The raw dataset was kept untouched, and a clean processed YOLO dataset was generated from it. The official split files were used so that training, validation, and testing remain separated."
    )
    dataset_table = Table(
        [
            [head("Split"), head("Images"), head("Labels"), head("Purpose")],
            [cell("Train"), cell("13,776"), cell("13,776"), cell("Used to learn fire and smoke detection patterns.")],
            [cell("Validation"), cell("3,445"), cell("3,445"), cell("Used during training to select the best model and trigger early stopping.")],
            [cell("Test"), cell("4,306"), cell("4,306"), cell("Used after training for final benchmark metrics.")],
        ],
        colWidths=[1.1 * inch, 1.1 * inch, 1.1 * inch, 3.25 * inch],
        repeatRows=1,
    )
    dataset_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F1F5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0B3D56")),
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
    story.append(Spacer(1, 6))
    b("Class mapping is standardized as 0 = smoke and 1 = fire.")
    b("Empty label files are preserved because they represent negative images with no fire/smoke.")
    b("A small number of raw test labels had coordinates slightly outside 0 to 1; these were fixed only in the processed dataset.")
    b("Processed dataset path: datasets/processed/fire_smoke_yolo/data.yaml.")

    p("3. Model Plan", "Section")
    p(
        "The base model is YOLOv8n. This is selected because the M1 requirement mentions YOLOv8-nano, and because the nano model is lightweight enough for real-time CCTV deployment. Later, YOLO11n or YOLO11s can be trained as a comparison model if time and GPU availability allow."
    )
    b("Baseline model: YOLOv8n pretrained weights.")
    b("Training image size: 960, chosen to help with small fire/smoke regions in elevated camera views.")
    b("Batch size: auto batch on Kaggle, so memory is used efficiently.")
    b("Metrics to track: mAP50, mAP50-95, precision, recall, confusion matrix, and class-wise performance.")

    p("4. Training Workflow", "Section")
    p(
        "Training will be performed on Kaggle. The GitHub repository will contain code only, while the processed dataset will be uploaded separately as a Kaggle input. This keeps the repository lightweight and avoids pushing large images or labels to GitHub."
    )
    train_table = Table(
        [
            [head("Step"), head("Action")],
            [cell("1"), cell("Clone the GitHub repository in the Kaggle notebook.")],
            [cell("2"), cell("Install Ultralytics and supporting Python packages.")],
            [cell("3"), cell("Attach the processed D-Fire YOLO dataset as Kaggle input.")],
            [cell("4"), cell("Run the Kaggle training script with GPU enabled.")],
            [cell("5"), cell("Use validation performance to save best.pt and stop early if no new best model appears for 10 epochs.")],
            [cell("6"), cell("Evaluate the trained model on the D-Fire test split.")],
            [cell("7"), cell("Zip and download training outputs, graphs, weights, and evaluation results.")],
        ],
        colWidths=[0.7 * inch, 5.85 * inch],
        repeatRows=1,
    )
    train_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF3E8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#254117")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB7C4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(train_table)

    p("5. Early Stopping and Progress Records", "Section")
    p(
        "The training script uses early stopping with patience set to 10. This means that if the validation score does not produce a new best model for 10 consecutive epochs, training stops automatically. This reduces wasted GPU time and helps avoid unnecessary overfitting."
    )
    b("Best model: weights/best.pt.")
    b("Last model: weights/last.pt.")
    b("Progress graph: results.png.")
    b("Training metrics table: results.csv.")
    b("Confusion matrix and prediction examples are saved by Ultralytics when plots are enabled.")
    b("The Kaggle training script creates a zip file of the run folder for easy download.")

    p("6. External Video Verification", "Section")
    p(
        "A separate video will be used only after training. This video will not be included in the training, validation, or test dataset. It should be described as an external unseen verification video, mainly for qualitative demonstration of how the trained detector behaves on real footage."
    )
    b("The D-Fire test split is used for benchmark metrics.")
    b("The separate video is used for visual verification and presentation/demo output.")
    b("The prediction script saves an annotated video, text labels, and confidence scores.")
    b("This separation avoids data leakage and keeps the evaluation explanation honest.")

    p("7. Future Plan", "Section")
    p(
        "After the public-data base model is trained, the next phase is industry-specific fine-tuning. The industry should provide representative CCTV footage from the actual camera locations. The most valuable footage is not only fire footage, but also normal non-fire footage that may cause false alarms."
    )
    b("Request normal plant footage across day, night, rain, haze, and shift changes.")
    b("Collect hard negatives: steam, dust, ash clouds, welding, sparks, glare, and hot equipment glow.")
    b("Add any permitted controlled smoke/fire drill footage from the same camera height and distance.")
    b("Fine-tune the base model on this industry-specific dataset.")
    b("Add temporal confirmation using rolling-window voting and smoke motion checks.")
    b("Later optimize for deployment using ONNX/TensorRT and multi-camera inference.")

    p("8. Final Position", "Section")
    p(
        "The project is currently in the base-model stage. The dataset has been standardized, the training scripts are ready for Kaggle, early stopping is configured, progress artifacts will be saved, and external video verification is planned separately from benchmark testing. This gives a clean academic workflow now and a practical route toward real industrial deployment later."
    )

    doc.build(story)


if __name__ == "__main__":
    build()
