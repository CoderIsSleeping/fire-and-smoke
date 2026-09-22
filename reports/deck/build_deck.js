const pptxgen = require("pptxgenjs");
const path = require("path");

const ROOT = "C:/Users/BIT/Desktop/fire and smoke";
const OUT = path.join(ROOT, "reports", "Fire_Smoke_Pipeline.pptx");
const AUG_IMG = path.join(ROOT, "output", "augmentation_examples.jpg");

// Palette: charcoal + ember for a fire topic; model colours match the PDFs.
const DARK = "1C2127", EMBER = "E4572E", AMBER = "E9A23B";
const M1 = "1F6F8B", M2 = "2E7D32", INK = "1A1A1A", MUTED = "5B6670";
const TINT = "F3F5F7", LINE = "D5DAE0", WHITE = "FFFFFF";
const HF = "Cambria", BF = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "Fire & Smoke Detection - DINOv3 pipeline";

function title(slide, text, sub) {
  slide.addText(text, { x: 0.6, y: 0.35, w: 12.1, h: 0.75, fontFace: HF, fontSize: 32, bold: true,
    color: INK, margin: 0, isTextBox: true });
  if (sub) slide.addText(sub, { x: 0.6, y: 1.05, w: 12.1, h: 0.4, fontFace: BF, fontSize: 14,
    color: MUTED, margin: 0, isTextBox: true });
}

function box(slide, x, y, w, h, head, sub, border, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1,
    fill: { color: fill || TINT }, line: { color: border, width: 1.75 } });
  slide.addText([
    { text: head, options: { bold: true, fontSize: 13, color: INK, breakLine: true } },
    { text: sub, options: { fontSize: 10, color: MUTED } },
  ], { x: x + 0.05, y, w: w - 0.1, h, align: "center", valign: "middle", fontFace: BF, margin: 2,
    isTextBox: true });
}

function arrow(slide, x1, y1, x2, y2, color) {
  slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1,
    line: { color: color || MUTED, width: 1.75, endArrowType: "triangle" },
  });
}

function numCircle(slide, x, y, n, color) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.46, h: 0.46, fill: { color }, line: { color } });
  slide.addText(String(n), { x, y, w: 0.46, h: 0.46, align: "center", valign: "middle",
    fontFace: BF, fontSize: 14, bold: true, color: WHITE, margin: 0, isTextBox: true });
}

function headerRow(cells, fill) {
  return cells.map((t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: fill || DARK } } }));
}

// ---------------------------------------------------------------- 1 title
{
  const s = pres.addSlide();
  s.background = { color: DARK };
  s.addText("Fire & Smoke Detection", { x: 0.8, y: 1.3, w: 11.7, h: 1.0, fontFace: HF, fontSize: 44,
    bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText("on industrial cameras, built on DINOv3", { x: 0.8, y: 2.25, w: 11.7, h: 0.6, fontFace: HF,
    fontSize: 24, italic: true, color: AMBER, margin: 0, isTextBox: true });
  s.addText("Input  \u00B7  parameters  \u00B7  output  \u00B7  how it runs  \u00B7  the pipeline",
    { x: 0.8, y: 3.0, w: 11.7, h: 0.45, fontFace: BF, fontSize: 16, color: "C9D1D9", margin: 0, isTextBox: true });

  const steps = ["Camera frame", "DINOv3 backbone", "Detection + scene heads", "Temporal confirmation", "Alarm"];
  const colors = [MUTED, M1, M2, AMBER, EMBER];
  steps.forEach((t, i) => {
    const x = 0.8 + i * 2.45;
    numCircle(s, x, 4.55, i + 1, colors[i]);
    s.addText(t, { x: x + 0.55, y: 4.5, w: 1.8, h: 0.56, fontFace: BF, fontSize: 12, color: WHITE,
      valign: "middle", margin: 0, isTextBox: true });
  });
  s.addText("Model 1: DINOv3 ViT-S + Faster R-CNN      Model 2: DINOv3 ViT-Ti + FCOS (light)",
    { x: 0.8, y: 6.3, w: 11.7, h: 0.4, fontFace: BF, fontSize: 13, color: "9AA4AE", margin: 0, isTextBox: true });
  s.addNotes("Two models share one pipeline. Model 1 is the accurate reference; Model 2 is the light, batch-exportable variant.");
}

// ---------------------------------------------------------------- 2 pipeline
{
  const s = pres.addSlide();
  title(s, "The pipeline, end to end", "One frame in, boxes and an alarm decision out");
  const W = 2.65, H = 1.05, xs = [0.6, 3.72, 6.84, 9.96], y1 = 1.75, y2 = 3.45, y3 = 5.15;

  box(s, xs[0], y1, W, H, "1  Camera frame", "webcam, video file or CCTV\nany resolution", MUTED);
  box(s, xs[1], y1, W, H, "2  Pre-process", "letterbox to 640 x 640\nRGB, ImageNet mean/std", MUTED);
  box(s, xs[2], y1, W, H, "3  DINOv3 backbone", "ViT-S (M1) or ViT-Ti (M2)\nfrozen, self-supervised", M1, "E3EEF4");
  box(s, xs[3], y1, W, H, "4  Feature pyramid", "blocks 5/8/9/11 ->\nstrides 8, 16, 32, 64", M2, "E8F1E9");

  box(s, xs[1], y2, W, H, "Glow prior", "classical CV, no learning\n7 warm-light statistics", AMBER, "FBF0DF");
  box(s, xs[2], y2, W, H, "Scene classifier", "whole image -> P(smoke), P(fire)\nsees hidden-fire glow", M2, "E8F1E9");
  box(s, xs[3], y2, W, H, "5  Detection head", "Faster R-CNN (M1) / FCOS (M2)\nboxes: smoke, fire + score", M2, "E8F1E9");

  box(s, xs[2], y3, W + 3.12, H, "6  Temporal confirmation",
      "alarm only if the SAME tracked object is seen in 6 of the last 15 sampled frames", AMBER, "FBF0DF");
  box(s, xs[0], y3, W + 3.12, H, "7  ALARM + event log",
      "on-screen alarm, CSV of every alarm on/off, annotated video", EMBER, "FCE7E1");

  for (let i = 0; i < 3; i++) arrow(s, xs[i] + W, y1 + H / 2, xs[i + 1], y1 + H / 2);
  arrow(s, xs[1] + W / 2, y1 + H, xs[1] + W / 2, y2);            // pre-process -> glow
  arrow(s, xs[2] + W / 2, y1 + H, xs[2] + W / 2, y2, M1);        // backbone -> scene (pooled tokens)
  arrow(s, xs[3] + W / 2, y1 + H, xs[3] + W / 2, y2);            // pyramid -> detection
  arrow(s, xs[1] + W, y2 + H / 2, xs[2], y2 + H / 2, AMBER);      // glow -> scene
  arrow(s, xs[3] + W / 2, y2 + H, xs[3] + W / 2, y3);            // detection -> temporal
  arrow(s, xs[2] + W / 2, y2 + H, xs[2] + W / 2, y3);            // scene -> temporal
  arrow(s, xs[2], y3 + H / 2, xs[0] + W + 3.12, y3 + H / 2, EMBER);

  s.addText("Blue = frozen DINOv3   Green = trained   Amber = rule-based   Red = output",
    { x: 0.6, y: 6.55, w: 12.1, h: 0.35, fontFace: BF, fontSize: 11, color: MUTED, margin: 0, isTextBox: true });
  s.addNotes("DINO is never removed: both models run DINOv3 first. The two heads share one backbone pass. The temporal step is rules, not learning.");
}


// ---------------------------------------------------------------- 2b detection heads
{
  const s = pres.addSlide();
  title(s, "The two detection heads", "Both turn DINOv3 features into boxes, in different ways");
  const cols = [
    { x: 0.6, c: M1, fill: "E3EEF4", name: "Faster R-CNN  (Model 1)", kind: "two-stage: propose, then check",
      steps: [
        ["Region proposal network", "looks over the feature maps and proposes ~300 candidate regions: \"something might be here\""],
        ["RoIAlign", "crops a fixed 7 x 7 feature patch for each proposed region"],
        ["Box head", "classifies each patch as smoke, fire or background, and refines its box"],
        ["Non-max suppression", "keeps the best box where several overlap"],
      ],
      pro: "Precise at high confidence; the stronger model here",
      con: "The number of regions varies per image, so it cannot be exported as a batched graph" },
    { x: 6.85, c: M2, fill: "E8F1E9", name: "FCOS  (Model 2)", kind: "one-stage: every location predicts directly",
      steps: [
        ["Every location votes", "all 8,500 points across the 4 pyramid levels predict at once, no proposals"],
        ["Class score", "how likely this point is on smoke or on fire"],
        ["Box + centerness", "distances to the 4 box edges, and how central the point is in its object"],
        ["Score + NMS", "score = sqrt(class x centerness), then keep the best boxes"],
      ],
      pro: "Fixed-shape output: batches across cameras, exports to TensorRT",
      con: "Scores are compressed (max ~0.75), so it needs its own threshold; less accurate in our run" },
  ];
  cols.forEach((col) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: col.x, y: 1.6, w: 5.85, h: 5.3, rectRadius: 0.1,
      fill: { color: col.fill }, line: { color: col.c, width: 1.75 } });
    s.addText([{ text: col.name, options: { bold: true, fontSize: 17, color: col.c, breakLine: true } },
               { text: col.kind, options: { fontSize: 12, italic: true, color: MUTED } }],
      { x: col.x + 0.3, y: 1.72, w: 5.3, h: 0.75, fontFace: BF, margin: 0, isTextBox: true });
    col.steps.forEach(([h, d], i) => {
      const y = 2.6 + i * 0.78;
      numCircle(s, col.x + 0.3, y + 0.06, i + 1, col.c);
      s.addText([{ text: h, options: { bold: true, fontSize: 13, color: INK, breakLine: true } },
                 { text: d, options: { fontSize: 11, color: MUTED } }],
        { x: col.x + 0.9, y, w: 4.75, h: 0.74, fontFace: BF, margin: 0, valign: "top", isTextBox: true });
    });
    s.addText([{ text: "+  ", options: { bold: true, color: M2 } }, { text: col.pro, options: { breakLine: true } },
               { text: "-  ", options: { bold: true, color: EMBER } }, { text: col.con }],
      { x: col.x + 0.3, y: 5.75, w: 5.3, h: 1.05, fontFace: BF, fontSize: 12, color: INK, margin: 0,
        valign: "top", paraSpaceAfter: 4, isTextBox: true });
  });
  s.addNotes("Faster R-CNN asks 'where might something be?' then checks each candidate. FCOS asks every location at once. That fixed-shape output is what makes FCOS batchable for many cameras.");
}

// ---------------------------------------------------------------- 3 input
{
  const s = pres.addSlide();
  title(s, "Input", "What the model receives, and in what shape");
  const items = [
    ["Video frame", "Any resolution (site cameras: 1280 x 720). Webcam, video file or CCTV stream."],
    ["Letterbox to 640 x 640", "Aspect ratio kept, grey padding. Resolution is fixed: 448 px was measured to cost 24 points of recall."],
    ["Normalise", "RGB scaled to 0-1, then ImageNet mean/std, as DINOv3 expects."],
    ["Glow statistics", "7 numbers from the frame (warm cast, highlights, glow blobs) for the scene classifier."],
    ["Sampling", "Deployment samples 1-5 frames per second per camera, not every frame."],
  ];
  items.forEach(([h, d], i) => {
    const y = 1.7 + i * 0.95;
    numCircle(s, 0.6, y + 0.05, i + 1, i === 3 ? AMBER : M1);
    s.addText([{ text: h, options: { bold: true, fontSize: 15, color: INK, breakLine: true } },
               { text: d, options: { fontSize: 12, color: MUTED } }],
      { x: 1.2, y, w: 5.4, h: 0.85, fontFace: BF, valign: "top", margin: 0, isTextBox: true });
  });
  s.addImage({ path: AUG_IMG, x: 7.0, y: 1.7, w: 5.7, h: 3.8 });
  s.addText("Training inputs are augmented to match the site: original, night, IR, and a flame hidden behind an obstacle.",
    { x: 7.0, y: 5.55, w: 5.7, h: 0.55, fontFace: BF, fontSize: 11, italic: true, color: MUTED, margin: 0, isTextBox: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.0, y: 6.2, w: 5.7, h: 0.6, rectRadius: 0.08,
    fill: { color: TINT }, line: { color: LINE, width: 1 } });
  s.addText("Tensors:  images (B, 3, 640, 640)  \u00B7  glow (B, 7)", { x: 7.1, y: 6.2, w: 5.5, h: 0.6,
    fontFace: "Courier New", fontSize: 12, color: INK, valign: "middle", margin: 0, isTextBox: true });
  s.addNotes("B is the batch: several cameras' frames at once. Only Model 2 can run batched after export.");
}

// ---------------------------------------------------------------- 4 model parameters
{
  const s = pres.addSlide();
  title(s, "Model parameters", "Same pipeline, two sizes");
  const rows = [
    headerRow(["", "Model 1", "Model 2 (light)"]),
    ["Backbone", "DINOv3 ViT-S/16", "DINOv3 ViT-Ti/16"],
    ["Embedding dim / blocks", "384 / 12", "192 / 12"],
    ["Backbone parameters", "21.6M", "5.5M"],
    ["Feature pyramid", "256 channels", "128 channels"],
    ["Detection head", "Faster R-CNN (two-stage)", "FCOS (one-stage), 2 convs"],
    ["Total parameters", "39.3M", "6.9M"],
    ["Checkpoint file", "157 MB", "28 MB"],
    ["Input size", "640 x 640", "640 x 640"],
    ["Batched ONNX export", "No - measured to fail", "Yes - verified"],
  ];
  rows[0][1].options.fill = { color: M1 };
  rows[0][2].options.fill = { color: M2 };
  s.addTable(rows, { x: 0.6, y: 1.65, w: 8.2, colW: [2.9, 2.65, 2.65], fontFace: BF, fontSize: 13,
    color: INK, border: { type: "solid", pt: 0.75, color: LINE }, rowH: 0.44, valign: "middle" });

  const stats = [["5.7x", "fewer parameters", M2], ["2.8x", "faster per frame", M2], ["0.726", "Model 1 test mAP@0.5", M1]];
  stats.forEach(([big, small, c], i) => {
    const y = 1.65 + i * 1.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.3, y, w: 3.4, h: 1.3, rectRadius: 0.1,
      fill: { color: TINT }, line: { color: LINE, width: 1 } });
    s.addText(big, { x: 9.3, y: y + 0.1, w: 3.4, h: 0.7, align: "center", fontFace: HF, fontSize: 36,
      bold: true, color: c, margin: 0, isTextBox: true });
    s.addText(small, { x: 9.3, y: y + 0.8, w: 3.4, h: 0.4, align: "center", fontFace: BF, fontSize: 12,
      color: MUTED, margin: 0, isTextBox: true });
  });
  s.addNotes("Model 2 changes the backbone size and the head. Input size stays 640 because smaller inputs lost too much recall.");
}

// ---------------------------------------------------------------- 5 training parameters
{
  const s = pres.addSlide();
  title(s, "Training parameters", "D-Fire dataset, trained on a Kaggle P100");
  const rows = [
    headerRow(["Setting", "Value"]),
    ["Dataset split", "train 13,776 / val 3,445 / test 4,306"],
    ["Verified negatives", "9,838 images (45.7%)"],
    ["Optimiser", "AdamW, weight decay 1e-4"],
    ["Learning rate", "1e-4, warmup 500 iters, cosine"],
    ["Stage 2 (Model 1)", "last 2 blocks unfrozen, lr 5e-5"],
    ["Batch / precision", "8, fp16 mixed precision"],
    ["Epochs", "M1: 40 + 15    M2: 45"],
    ["Model selection", "best validation mAP@0.5"],
  ];
  s.addTable(rows, { x: 0.6, y: 1.65, w: 6.3, colW: [2.3, 4.0], fontFace: BF, fontSize: 13, color: INK,
    border: { type: "solid", pt: 0.75, color: LINE }, rowH: 0.47, valign: "middle" });

  s.addChart(pres.charts.BAR, [{
    name: "probability",
    labels: ["crop", "colour jitter", "flip", "night (low light)", "flame occlusion", "IR / grayscale", "motion blur"],
    values: [0.80, 0.80, 0.50, 0.35, 0.25, 0.10, 0.10],
  }], {
    x: 7.3, y: 1.55, w: 5.4, h: 4.6, barDir: "bar", chartColors: [EMBER],
    showTitle: true, title: "Augmentation: chance per training image", titleFontSize: 13, titleColor: INK,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: INK,
    valAxisMaxVal: 1, valAxisMinVal: 0, valAxisLabelFormatCode: "0%", dataLabelFormatCode: "0%",
    catAxisLabelColor: INK, valAxisLabelColor: MUTED, catAxisLabelFontSize: 11,
    valGridLine: { color: "E6E9EC", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    catAxisOrientation: "maxMin",
  });
  s.addText("Night, IR and occlusion exist to match the deployment site, not the web images in D-Fire.",
    { x: 7.3, y: 6.25, w: 5.4, h: 0.5, fontFace: BF, fontSize: 11, italic: true, color: MUTED, margin: 0, isTextBox: true });
  s.addNotes("Same data, optimiser and augmentation for both models, so the comparison isolates the architecture.");
}

// ---------------------------------------------------------------- 6 runtime parameters
{
  const s = pres.addSlide();
  title(s, "Runtime and alarm parameters", "How a detection becomes an alarm");
  const cards = [
    ["Alarm threshold", "M1 0.90  \u00B7  M2 0.55", "Each model at its own operating point: M2 never scores above ~0.75."],
    ["Release threshold", "M1 0.75  \u00B7  M2 0.40", "Below this, a detection no longer counts as evidence."],
    ["Window", "15 sampled frames", "The sliding window the confirmation looks back over."],
    ["Enter hits", "6 of 15", "Detections of the same tracked object needed to raise the alarm."],
    ["Exit hits", "2 of 15", "Hysteresis: a briefly hidden flame does not switch the alarm off."],
    ["Tracking", "IoU 0.20 \u00B7 min box 4 px", "Links boxes frame to frame; drops tiny junk boxes."],
  ];
  cards.forEach(([h, v, d], i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.6 + col * 4.1, y = 1.7 + row * 2.45;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 3.8, h: 2.15, rectRadius: 0.1,
      fill: { color: TINT }, line: { color: LINE, width: 1 } });
    s.addText(h, { x: x + 0.25, y: y + 0.2, w: 3.3, h: 0.4, fontFace: BF, fontSize: 14, bold: true,
      color: MUTED, margin: 0, isTextBox: true });
    s.addText(v, { x: x + 0.25, y: y + 0.6, w: 3.3, h: 0.6, fontFace: HF, fontSize: 21, bold: true,
      color: i < 2 ? EMBER : INK, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.25, y: y + 1.25, w: 3.3, h: 0.8, fontFace: BF, fontSize: 12, color: MUTED,
      margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("Random flicker never lands on the same track, so it never reaches 6 hits. Stationary look-alikes can, which is why hard negatives matter.");
}

// ---------------------------------------------------------------- 7 output
{
  const s = pres.addSlide();
  title(s, "Output", "What comes out for every processed frame");
  const outs = [
    ["Boxes", "[x1, y1, x2, y2] in the original camera's pixels"],
    ["Class + score", "smoke or fire, confidence 0-1"],
    ["Scene probabilities", "P(smoke), P(fire) for the whole image"],
    ["Alarm state", "confirmed track, class, hits in window"],
    ["Recordings", "annotated video + CSV event log + alarm snapshots"],
  ];
  outs.forEach(([h, d], i) => {
    const y = 1.7 + i * 0.9;
    numCircle(s, 0.6, y + 0.04, i + 1, i === 3 ? EMBER : M2);
    s.addText([{ text: h, options: { bold: true, fontSize: 15, color: INK, breakLine: true } },
               { text: d, options: { fontSize: 12, color: MUTED } }],
      { x: 1.2, y, w: 4.9, h: 0.8, fontFace: BF, margin: 0, valign: "top", isTextBox: true });
  });

  s.addText("Event log from the live lighter test", { x: 6.6, y: 1.65, w: 6.1, h: 0.4, fontFace: BF,
    fontSize: 14, bold: true, color: INK, margin: 0, isTextBox: true });
  const ev = [
    headerRow(["time", "event", "class", "score", "evidence"], EMBER),
    ["19.5 s", "ALARM_ON", "fire", "0.315", "3/8 hits"],
    ["28.3 s", "ALARM_OFF", "-", "-", "track 1"],
    ["81.8 s", "ALARM_ON", "fire", "0.375", "3/8 hits"],
  ];
  s.addTable(ev, { x: 6.6, y: 2.1, w: 6.1, colW: [1.0, 1.4, 0.9, 0.9, 1.9], fontFace: BF, fontSize: 12,
    color: INK, border: { type: "solid", pt: 0.75, color: LINE }, rowH: 0.42, valign: "middle" });

  s.addText("Raw export outputs (Model 2, for TensorRT)", { x: 6.6, y: 4.2, w: 6.1, h: 0.4, fontFace: BF,
    fontSize: 14, bold: true, color: INK, margin: 0, isTextBox: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.6, y: 4.65, w: 6.1, h: 1.75, rectRadius: 0.08,
    fill: { color: TINT }, line: { color: LINE, width: 1 } });
  s.addText([
    { text: "cls_logits       (B, 8500, 3)", options: { breakLine: true } },
    { text: "bbox_regression  (B, 8500, 4)", options: { breakLine: true } },
    { text: "bbox_ctrness     (B, 8500, 1)", options: { breakLine: true } },
    { text: "scene_logits     (B, 2)" },
  ], { x: 6.8, y: 4.7, w: 5.8, h: 1.65, fontFace: "Courier New", fontSize: 13, color: INK,
    valign: "middle", margin: 0, isTextBox: true });
  s.addNotes("8500 = candidate locations over the 4 pyramid levels at 640 px. Decoding and NMS run after the network.");
}

// ---------------------------------------------------------------- 8 how it runs
{
  const s = pres.addSlide();
  title(s, "How it runs", "Per frame, in order");
  const steps = [
    "Read the newest frame (stale frames are dropped, so it never lags behind)",
    "Letterbox to 640 x 640, normalise, compute the 7 glow statistics",
    "DINOv3 backbone -> feature pyramid (one pass, shared by both heads)",
    "Detection head -> boxes; scene classifier -> P(smoke), P(fire)",
    "Map boxes back to camera pixels; drop tiny boxes and masked regions",
    "Update tracks; raise or release the alarm by the 6-of-15 rule",
    "Draw the on-screen display, write the video and the event log",
  ];
  steps.forEach((t, i) => {
    const y = 1.6 + i * 0.7;
    numCircle(s, 0.6, y, i + 1, i === 5 ? AMBER : i === 6 ? EMBER : M1);
    s.addText(t, { x: 1.2, y: y - 0.02, w: 6.2, h: 0.5, fontFace: BF, fontSize: 13, color: INK,
      valign: "middle", margin: 0, isTextBox: true });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.8, y: 1.6, w: 4.9, h: 1.95, rectRadius: 0.1,
    fill: { color: DARK }, line: { color: DARK } });
  s.addText([
    { text: "Run it", options: { bold: true, fontSize: 15, color: AMBER, breakLine: true } },
    { text: "Double-click run_both_models.bat", options: { fontSize: 12, color: WHITE, breakLine: true } },
    { text: "1-2  one model, webcam", options: { fontSize: 12, color: "C9D1D9", breakLine: true } },
    { text: "3-4  one model, site video", options: { fontSize: 12, color: "C9D1D9", breakLine: true } },
    { text: "5-7  both side by side", options: { fontSize: 12, color: "C9D1D9", breakLine: true } },
    { text: "press q to stop", options: { fontSize: 11, italic: true, color: "9AA4AE" } },
  ], { x: 8.05, y: 1.7, w: 4.5, h: 1.8, fontFace: BF, valign: "top", margin: 0, isTextBox: true });

  const t = [["Model 1", "849 ms", M1], ["Model 2", "305 ms", M2]];
  t.forEach(([n, ms, c], i) => {
    const x = 7.8 + i * 2.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 3.85, w: 2.4, h: 1.6, rectRadius: 0.1,
      fill: { color: TINT }, line: { color: LINE, width: 1 } });
    s.addText(ms, { x, y: 4.0, w: 2.4, h: 0.75, align: "center", fontFace: HF, fontSize: 32, bold: true,
      color: c, margin: 0, isTextBox: true });
    s.addText(`${n} per frame`, { x, y: 4.75, w: 2.4, h: 0.4, align: "center", fontFace: BF, fontSize: 12,
      color: MUTED, margin: 0, isTextBox: true });
  });
  s.addText("Laptop CPU, same frames. On a GPU with batching, only Model 2 can use TensorRT.",
    { x: 7.8, y: 5.6, w: 4.9, h: 0.55, fontFace: BF, fontSize: 11, italic: true, color: MUTED, margin: 0, isTextBox: true });
  s.addNotes("Step 1 matters for live cameras: without dropping stale frames the system would alarm on old video.");
}

// ---------------------------------------------------------------- 9 results
{
  const s = pres.addSlide();
  title(s, "Results", "Held-out test split of 4,306 images, plus real site footage");
  s.addChart(pres.charts.BAR, [
    { name: "Model 1", labels: ["Test mAP@0.5", "Recall at 0.7% FPR", "Fire AP50", "Smoke AP50"], values: [0.726, 0.875, 0.638, 0.815] },
    { name: "Model 2 (light)", labels: ["Test mAP@0.5", "Recall at 0.7% FPR", "Fire AP50", "Smoke AP50"], values: [0.613, 0.763, 0.509, 0.716] },
  ], {
    x: 0.6, y: 1.55, w: 7.6, h: 5.0, barDir: "col", barGrouping: "clustered", chartColors: [M1, M2],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelFormatCode: "0.000",
    valAxisMaxVal: 1, valAxisMinVal: 0, catAxisLabelColor: INK, valAxisLabelColor: MUTED,
    catAxisLabelFontSize: 12, valGridLine: { color: "E6E9EC", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendFontSize: 12,
  });
  const stats = [
    ["0", "false alarms on 54 min of site footage", "both models, at their own thresholds", EMBER],
    ["2.8x", "faster: Model 2", "305 vs 849 ms per frame", M2],
    ["-11 pts", "recall cost of Model 2", "at the same false-alarm rate", M1],
  ];
  stats.forEach(([big, lab, sub, c], i) => {
    const y = 1.6 + i * 1.65;
    s.addText(big, { x: 8.7, y, w: 4.0, h: 0.75, fontFace: HF, fontSize: 36, bold: true, color: c,
      margin: 0, isTextBox: true });
    s.addText([{ text: lab, options: { bold: true, fontSize: 13, color: INK, breakLine: true } },
               { text: sub, options: { fontSize: 11, color: MUTED } }],
      { x: 8.7, y: y + 0.72, w: 4.0, h: 0.75, fontFace: BF, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("Model 2 is faster, smaller and deployable, but not yet as accurate. Model 1 remains the accuracy reference.");
}

// ---------------------------------------------------------------- 10 next steps
{
  const s = pres.addSlide();
  s.background = { color: DARK };
  s.addText("Next steps", { x: 0.8, y: 0.6, w: 11.7, h: 0.9, fontFace: HF, fontSize: 40, bold: true,
    color: WHITE, margin: 0, isTextBox: true });
  const items = [
    ["Model 2, stage 2", "Unfreeze its last 2 blocks. It trains only 1.5M parameters today, so it should gain more than Model 1 did."],
    ["ViT-S + FCOS", "Model 1's backbone with Model 2's head: batched export without the accuracy loss, if it holds."],
    ["Hard negatives from the site", "Hard hats, hi-vis and dusty haze as training negatives."],
    ["Benchmark on the target GPU", "TensorRT fp16 with batching, before any camera count is stated."],
  ];
  items.forEach(([h, d], i) => {
    const y = 1.85 + i * 1.2;
    numCircle(s, 0.8, y + 0.05, i + 1, [M2, M1, AMBER, EMBER][i]);
    s.addText([{ text: h, options: { bold: true, fontSize: 18, color: WHITE, breakLine: true } },
               { text: d, options: { fontSize: 13, color: "C9D1D9" } }],
      { x: 1.45, y, w: 10.9, h: 1.05, fontFace: BF, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("The system is designed for multi-camera scaling; the final model and camera capacity come from these experiments and GPU benchmarking.");
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote " + f));
