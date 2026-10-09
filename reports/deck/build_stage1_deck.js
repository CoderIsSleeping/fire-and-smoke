// Stage-1 model-selection deck. Numbers come from reports/data/stage1_presentation.json
// (written by scripts/build_stage1_assets.py), so the slides and the report cannot disagree.
//
//   NODE_PATH=<node_modules with pptxgenjs> node reports/deck/build_stage1_deck.js <apply_theme.js>
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const ROOT = path.resolve(__dirname, "..", "..");
// Optional, git-ignored: team names, guide and annotated sample frames for the presented copy.
const PRIV_FILE = path.join(ROOT, "Industry", "presentation_private.json");
const PRIV = fs.existsSync(PRIV_FILE) ? JSON.parse(fs.readFileSync(PRIV_FILE, "utf8")) : null;
const OUT = PRIV && PRIV.out ? path.join(ROOT, PRIV.out) : path.join(ROOT, "reports", "Stage1_Model_Selection_Slides.pptx");
const FIG = (n) => path.join(ROOT, "reports", "figures", n);
const D = JSON.parse(fs.readFileSync(path.join(ROOT, "reports", "data", "stage1_presentation.json"), "utf8"));
const M = D.models;

const THEME = {
  name: "Fire and Smoke",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1A1A1A", lt1: "FFFFFF", dk2: "1C2127", lt2: "F3F5F7",
    accent1: "E4572E", // ember: emphasis
    accent2: "1F6F8B", // DINOv3
    accent3: "2E7D32", // MobileNetV3
    accent4: "C62828", // ResNet-18
    accent5: "EF6C00", // FCOS model
    accent6: "5B6670", // muted
    hlink: "1F6F8B", folHlink: "5B6670",
  },
};
const HEX = THEME.colors;
const LINE = "D5DAE0";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Fire and smoke detection - model selection (stage 1)";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.5, fontSize: 44, bold: true,
        color: C.background1, align: "left", valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 1.4, fontSize: 20,
        color: C.background2, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.9, fontSize: 30, bold: true,
        color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "Fire and smoke detection  |  model selection (stage 1)",
        options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: HEX.accent6, align: "right" },
});

// ---------------------------------------------------------------- helpers
const f4 = (v) => v.toFixed(4);
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const mp = (v) => (v < 1e7 ? (v / 1e6).toFixed(2) : (v / 1e6).toFixed(1)) + "M";
const ms = (m) => (m.cpu_ms[0] === m.cpu_ms[1] ? `${m.cpu_ms[0]}` : `${m.cpu_ms[0]}-${m.cpu_ms[1]}`);
const tmap = (m, c = "mean") => m.test[c].AP50;
const MODEL_COLOR = { dino: C.accent2, mbv3: C.accent3, r18: C.accent4, fcos: C.accent5 };
const MODEL_HEX = { dino: HEX.accent2, mbv3: HEX.accent3, r18: HEX.accent4, fcos: HEX.accent5 };
const KEYS = ["dino", "mbv3", "r18", "fcos"];

function text(slide, t, o) {
  slide.addText(t, Object.assign({ isTextBox: true, margin: 0, color: C.text1, fontSize: 14, valign: "top" }, o));
}
function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || C.background2 },
    line: { color: LINE, width: 0.75 } });
}
function badge(slide, x, y, label, color, size = 0.46) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: size, h: size, fill: { color: color || C.accent1 }, line: { color: color || C.accent1, width: 0 } });
  slide.addText(String(label), { x, y, w: size, h: size, align: "center", valign: "middle", fontSize: 15, bold: true,
    color: C.background1, margin: 0, isTextBox: true });
}
// A left-to-right row of boxes joined by arrows; items = [{t, s, w}] with w a relative width.
function flow(slide, x, y, w, h, items, color, tint) {
  const gap = 0.3;
  const total = items.reduce((a, it) => a + (it.w || 1), 0);
  const unit = (w - gap * (items.length - 1)) / total;
  let cx = x;
  items.forEach((it, i) => {
    const bw = unit * (it.w || 1);
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y, w: bw, h, rectRadius: 0.06,
      fill: { color: it.plain ? C.background1 : tint }, line: { color: it.plain ? HEX.accent6 : color, width: 1.5 } });
    const runs = [{ text: it.t, options: { bold: true, fontSize: 13, color: C.text1, breakLine: !!it.s } }];
    if (it.s) runs.push({ text: it.s, options: { fontSize: 11, color: C.accent6 } });
    slide.addText(runs, { x: cx + 0.05, y, w: bw - 0.1, h, align: "center", valign: "middle", margin: 0, isTextBox: true });
    if (i < items.length - 1) {
      slide.addShape(pres.shapes.LINE, { x: cx + bw + 0.03, y: y + h / 2, w: gap - 0.06, h: 0,
        line: { color: HEX.accent6, width: 1.5, endArrowType: "triangle" } });
    }
    cx += bw + gap;
  });
}
function cell(t, o) { return { text: String(t), options: Object.assign({ fontSize: 13, color: HEX.dk1, valign: "middle" }, o || {}) }; }
function head(t, o) { return cell(t, Object.assign({ bold: true, fill: { color: HEX.lt2 } }, o || {})); }
const TABLE = { border: { type: "solid", pt: 0.75, color: LINE }, margin: [0.05, 0.1, 0.05, 0.1] };

// ---------------------------------------------------------------- 1 title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
s.addText("Fire and Smoke Detection: Choosing the Model", { placeholder: "title" });
s.addText("Stage 1: three backbones and two detection heads compared over 40 training epochs", { placeholder: "body" });
text(s, "DINOv3   |   MobileNetV3-Large   |   ResNet-18   |   Faster R-CNN   |   FCOS",
  { x: 0.8, y: 5.6, w: 11.7, h: 0.4, fontSize: 16, color: C.accent1, bold: true });
if (PRIV) {
  text(s, [{ text: "Presented by:  ", options: { bold: true } }, { text: PRIV.team.join("   |   ") }],
    { x: 0.8, y: 6.2, w: 11.7, h: 0.35, fontSize: 14, color: C.background2 });
  text(s, [{ text: "Guide:  ", options: { bold: true } }, { text: PRIV.guide + "   |   " + PRIV.dept }],
    { x: 0.8, y: 6.6, w: 11.7, h: 0.35, fontSize: 14, color: C.background2 });
}
s.addNotes("Introduce the project: an industrial fire and smoke detector for fixed cameras. Today covers stage 1 only: "
  + "how we compared the candidate models and why we continue with DINOv3 plus Faster R-CNN.");

// ---------------------------------------------------------------- 2 problem
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("The problem: fire and smoke on industrial cameras, around the clock", { placeholder: "title" });
const problems = [
  ["Poor light, high cameras", "Cameras are mounted high in dim industrial lighting, with glare and reflections"],
  ["Runs 24 hours a day", "False alarms must be very rare, or operators stop trusting the system"],
  ["Fire can be hidden", "A flame behind equipment may show only as a glow on nearby surfaces"],
  ["Smoke and fire", "Two classes to detect; smoke has no fixed shape or edge"],
  ["Small, distant fires", "A fire far from the camera covers only a few pixels"],
  ["About 70 cameras", "At least 10 frames per second overall: speed and size matter too"],
];
problems.forEach(([h, d], i) => {
  const x = 0.6 + (i % 3) * 4.1, y = 1.5 + Math.floor(i / 3) * 2.15;
  card(s, x, y, 3.85, 1.9);
  badge(s, x + 0.25, y + 0.25, i + 1);
  text(s, h, { x: x + 0.9, y: y + 0.24, w: 2.8, h: 0.5, fontSize: 18, bold: true, valign: "middle" });
  text(s, d, { x: x + 0.25, y: y + 0.9, w: 3.4, h: 0.9, fontSize: 14, color: C.accent6 });
});
text(s, [{ text: "Question for stage 1:  ", options: { bold: true, color: C.accent1 } },
  { text: "which backbone and which detection head should the detector be built on?" }],
  { x: 0.6, y: 6.0, w: 12.1, h: 0.6, fontSize: 20, valign: "middle" });
s.addNotes("Six requirements from the site. The second one drives most decisions: because the system runs 24/7, the false-alarm "
  + "rate is as important as accuracy. Stage 1 answers one question with measurements: which backbone and which head.");

// ---------------------------------------------------------------- literature review
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("Literature review: what exists, and what is missing", { placeholder: "title" });
{
  const lit = [
    ["Colour and motion rules", "Chen 2004; Toreyin 2006; Celik 2009", "Fast, no training data", "Depends on colour; many false alarms"],
    ["CNN frame classifiers", "Muhammad 2018; Dunnings and Breckon 2018", "Learned features, real time", "Says fire or no fire, not where"],
    ["Two-stage detectors", "Ren 2015 (Faster R-CNN); Zhang 2018", "Accurate boxes, clear scores", "Heavier head"],
    ["One-stage detectors", "Tian 2019 (FCOS); de Venancio 2022 (D-Fire)", "Simple, fixed-shape output", "Daytime data; false alarms unreported"],
    ["CNN backbones", "He 2016 (ResNet); Howard 2019 (MobileNetV3)", "Efficient, several scales", "ImageNet features, local view"],
    ["Self-supervised transformers", "Caron 2021 (DINO); Simeoni 2025 (DINOv3); Li 2022", "General features, no labels needed", "Slower; hardly tried for fire"],
  ];
  const f = { fontSize: 12 };
  s.addTable([
    [head("Approach", f), head("Representative work", f), head("Strength", f), head("Limitation for our task", f)],
    ...lit.map(([a, w, st, li]) => [cell(a, { fontSize: 12, bold: true }), cell(w, { fontSize: 11, color: HEX.accent6 }), cell(st, f), cell(li, f)]),
  ], Object.assign({ x: 0.6, y: 1.35, w: 7.9, colW: [1.75, 2.45, 1.75, 1.95], rowH: 0.62 }, TABLE));
  text(s, "Gaps we address", { x: 8.85, y: 1.35, w: 3.85, h: 0.4, fontSize: 16, bold: true, color: C.accent1, valign: "middle" });
  const gaps = [
    ["False alarms are not measured", "we report the false-alarm rate with accuracy"],
    ["Data does not match industrial scenes", "low-light and infrared-style augmentation"],
    ["Hidden fire is not handled", "scene classifier with glow statistics"],
    ["Self-supervised backbones hardly tried for fire", "DINOv3 evaluated"],
    ["No equal-conditions comparison", "3 backbones, 2 heads, same data and settings"],
  ];
  gaps.forEach(([g, a], i) => {
    const y = 1.85 + i * 0.95;
    card(s, 8.85, y, 3.85, 0.83);
    badge(s, 8.97, y + 0.22, i + 1, C.accent1, 0.38);
    text(s, [{ text: g, options: { bold: true, fontSize: 12, breakLine: true } }, { text: a, options: { fontSize: 11, color: C.accent6 } }],
      { x: 9.47, y, w: 3.15, h: 0.83, valign: "middle" });
  });
}
s.addNotes("Summary of the literature survey; the full survey with 24 references is in the report. Early methods used colour and "
  + "motion rules: fast, but any orange object triggers them. CNN classifiers learn features but only say whether a frame has fire. "
  + "Detectors give boxes: two-stage ones such as Faster R-CNN are accurate with clear scores, one-stage ones such as FCOS are "
  + "simpler. The D-Fire paper gave a public dataset. Backbones are mostly ImageNet CNNs; self-supervised transformers such as "
  + "DINOv3 give general features but have hardly been tried for fire. Five gaps follow, and each one is something our work addresses.");

// ---------------------------------------------------------------- 3 approach
pres.addSection({ title: "Models" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Models" });
s.addText("Approach: one detector, with two parts to choose", { placeholder: "title" });
s.addImage({ path: FIG("s1_pipeline.png"), x: 0.6, y: 1.3, w: 12.1, h: 4.1, sizing: { type: "contain", w: 12.1, h: 4.1 } });
const fair = [["Same data", "D-Fire: 13,776 training, 3,445 validation and 4,306 test images; 45% contain no fire or smoke"],
  ["Same training", "40 epochs, AdamW, learning rate 1e-4, batch 8, same augmentation; backbone frozen"],
  ["Same measures", "After every epoch: mAP@0.5, and detections at no more than 1% false alarms"]];
fair.forEach(([h, d], i) => {
  const x = 0.6 + i * 4.1;
  card(s, x, 5.55, 3.85, 1.25);
  text(s, h, { x: x + 0.25, y: 5.62, w: 3.4, h: 0.4, fontSize: 16, bold: true, color: C.accent1, valign: "middle" });
  text(s, d, { x: x + 0.25, y: 6.02, w: 3.4, h: 0.75, fontSize: 12 });
});
s.addNotes("The detector has a backbone that extracts features, a feature pyramid for different object sizes, a detection head "
  + "that outputs boxes, and a scene classifier for hidden fire. We compare three backbones and two heads. To keep it fair, "
  + "everything else is identical. The backbone is frozen in stage 1, so we measure how good each backbone's ready-made "
  + "features are.");

// ---------------------------------------------------------------- methodology
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Models" });
s.addText("Methodology: how fire and smoke are detected", { placeholder: "title" });
const steps = [
  ["Data collection", ["21,527 images labelled with boxes for smoke and fire (D-Fire)", "45% show no fire or smoke, so false alarms can be measured"]],
  ["Pre-processing", ["Each frame resized to 640 x 640, shape kept", "Training images imitate low light, infrared and hidden flames"]],
  ["Feature extraction", ["A pretrained backbone turns the image into features", "A feature pyramid covers large and small fires"]],
  ["Detection", ["Detection head: box, class (smoke or fire) and confidence score", "Scene classifier: whole-image check for a hidden fire's glow"]],
  ["Train and evaluate", ["Transfer learning on the fire data; best epoch kept", "Measured by mAP@0.5 and the false-alarm rate"]],
  ["Alarm decision", ["Confidence threshold set for at most 1% false alarms", "On video, an alarm needs 6 detections in 15 frames"]],
];
const sw = 3.85, sgx = 0.275, sh = 2.05, sgy = 0.2;
steps.forEach(([h, lines], i) => {
  const col = i % 3, row = Math.floor(i / 3);
  const x = 0.6 + col * (sw + sgx), y = 1.45 + row * (sh + sgy);
  card(s, x, y, sw, sh);
  badge(s, x + 0.22, y + 0.2, i + 1);
  text(s, h, { x: x + 0.82, y: y + 0.2, w: sw - 1.0, h: 0.46, fontSize: 17, bold: true, valign: "middle" });
  text(s, lines.map((l, j) => ({ text: l, options: { bullet: true, breakLine: j < lines.length - 1 } })),
    { x: x + 0.2, y: y + 0.8, w: sw - 0.4, h: sh - 0.9, fontSize: 13, paraSpaceAfter: 5 });
  if (col < 2) {
    s.addShape(pres.shapes.LINE, { x: x + sw + 0.03, y: y + 0.43, w: sgx - 0.06, h: 0,
      line: { color: HEX.accent6, width: 1.5, endArrowType: "triangle" } });
  }
});
card(s, 0.6, 5.95, 12.1, 0.8);
text(s, [{ text: "Input:  ", options: { bold: true, color: C.accent1 } }, { text: "a camera frame          " },
  { text: "Output:  ", options: { bold: true, color: C.accent1 } },
  { text: "where the smoke or fire is, how confident the system is, and whether to raise an alarm" }],
  { x: 0.85, y: 5.95, w: 11.6, h: 0.8, fontSize: 15, valign: "middle" });
s.addNotes("The method in six steps. One: collect labelled images, including many with no fire, so that false alarms can be measured. "
  + "Two: resize each frame and, during training, alter the images to imitate the site conditions: low light, infrared cameras, "
  + "flames hidden behind objects. Three: a pretrained backbone extracts features and a pyramid makes them available at several "
  + "scales. Four: a detection head outputs boxes with a class and a score, and a scene classifier gives a second opinion on the "
  + "whole image for hidden fire. Five: train by transfer learning and evaluate with mAP and the false-alarm rate. Six: choose "
  + "the confidence threshold for very few false alarms, and on video require repeated detections before raising an alarm.");

// ---------------------------------------------------------------- data collection and annotation
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Models" });
s.addText("Data: collection and annotation", { placeholder: "title" });
{
  const steps = [
    ["400+", "online videos of fire, smoke and industrial incidents collected"],
    ["150+", "videos kept after filtering: surveillance-style view, industrial scene, low or mixed light"],
    ["2,100", "frames extracted at a fixed interval"],
    ["2,100", "frames annotated in Roboflow: fire and smoke marked with boxes or polygons"],
    ["21,527", "images of the public D-Fire benchmark, labels checked; used for the model comparison"],
  ];
  steps.forEach(([n, d], i) => {
    const y = 1.45 + i * 1.02;
    card(s, 0.6, y, 5.3, 0.88);
    text(s, n, { x: 0.75, y, w: 1.35, h: 0.88, fontSize: 24, bold: true, color: i === 4 ? C.accent2 : C.accent1, valign: "middle" });
    text(s, d, { x: 2.15, y, w: 3.65, h: 0.88, fontSize: 13, valign: "middle" });
  });
  const imgs = PRIV && PRIV.images ? PRIV.images.map((f) => path.join(ROOT, "Industry", "annotated_samples", f)).filter((f) => fs.existsSync(f)) : [];
  imgs.slice(0, 6).forEach((f, i) => {
    const w = 3.2, h = 1.8, x = 6.2 + (i % 2) * (w + 0.1), y = 1.45 + Math.floor(i / 2) * (h + 0.1);
    s.addImage({ path: f, x, y, w, h });
  });
  if (imgs.length) text(s, "Annotated frames from our collection: red = fire, blue or violet = smoke",
    { x: 6.2, y: 7.0, w: 5.9, h: 0.3, fontSize: 10, color: C.accent6, valign: "middle" });
}
s.addNotes("Data shortage was the first problem: there is no fire footage from the site. We collected more than 400 videos online, "
  + "kept more than 150 that match our conditions, extracted 2,100 frames and annotated all of them in Roboflow with fire and "
  + "smoke regions. For the model comparison in this presentation every model is trained on the public D-Fire benchmark, so the "
  + "numbers are comparable with published work; our own frames are for fine-tuning and testing in the next stage.");

// ---------------------------------------------------------------- 4 backbones at a glance
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Models" });
s.addText("Three backbone candidates", { placeholder: "title" });
const bb = [
  { k: "dino", name: "DINOv3 ViT-S/16", type: "Vision Transformer", blocks: "12 transformer blocks",
    pre: "Self-supervised, no labels, 1.689 billion images", plus: "Strongest general features; sees the whole image",
    minus: "Slowest; needs a pyramid built on top" },
  { k: "mbv3", name: "MobileNetV3-Large", type: "Convolutional network for mobile devices", blocks: "15 inverted-residual blocks",
    pre: "Supervised, ImageNet-1k, 1.28 million images", plus: "Smallest and about 2.5 times faster",
    minus: "Local view; more false alarms" },
  { k: "r18", name: "ResNet-18", type: "Convolutional network with skip connections", blocks: "8 residual blocks, 20 convolutions",
    pre: "Supervised, ImageNet-1k, 1.28 million images", plus: "Simple, standard, supported everywhere",
    minus: "Lowest accuracy of the three" },
];
bb.forEach((b, i) => {
  const x = 0.6 + i * 4.1, m = M[b.k];
  card(s, x, 1.4, 3.85, 5.35);
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: 1.62, w: 0.3, h: 0.3, fill: { color: MODEL_COLOR[b.k] }, line: { color: MODEL_HEX[b.k], width: 0 } });
  text(s, b.name, { x: x + 0.68, y: 1.55, w: 3.0, h: 0.45, fontSize: 20, bold: true, valign: "middle" });
  text(s, b.type, { x: x + 0.25, y: 2.08, w: 3.4, h: 0.55, fontSize: 13, color: C.accent6 });
  text(s, mp(m.arch.trunk), { x: x + 0.25, y: 2.65, w: 3.4, h: 0.8, fontSize: 44, bold: true, color: MODEL_COLOR[b.k], valign: "middle" });
  text(s, "backbone parameters", { x: x + 0.25, y: 3.42, w: 3.4, h: 0.3, fontSize: 12, color: C.accent6 });
  text(s, [{ text: "Structure  ", options: { bold: true } }, { text: b.blocks }], { x: x + 0.25, y: 3.9, w: 3.4, h: 0.55, fontSize: 13 });
  text(s, [{ text: "Pretraining  ", options: { bold: true } }, { text: b.pre }], { x: x + 0.25, y: 4.5, w: 3.4, h: 0.6, fontSize: 13 });
  text(s, [{ text: "+  ", options: { bold: true, color: C.accent3 } }, { text: b.plus }], { x: x + 0.25, y: 5.3, w: 3.4, h: 0.6, fontSize: 13 });
  text(s, [{ text: "-  ", options: { bold: true, color: C.accent4 } }, { text: b.minus }], { x: x + 0.25, y: 5.95, w: 3.4, h: 0.6, fontSize: 13 });
});
s.addNotes("DINOv3 is a Vision Transformer trained by Meta without labels on 1.689 billion images. MobileNetV3 and ResNet-18 are "
  + "convolutional networks trained with labels on ImageNet, about 1,300 times fewer images. Parameters shown are for the "
  + "backbone alone; the detection head adds about 14.5 million to each.");

// ---------------------------------------------------------------- 5 DINOv3 inside
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Models" });
s.addText("Inside DINOv3 ViT-S/16", { placeholder: "title" });
flow(s, 0.6, 1.45, 12.1, 1.25, [
  { t: "Input", s: "3 x 640 x 640 image", w: 0.9, plain: true },
  { t: "Patch embedding", s: "16 x 16 px patches\n1,600 tokens x 384", w: 1.15 },
  { t: "12 transformer blocks", s: "every token attends to every\nother token (6 heads)", w: 1.5 },
  { t: "Feature map", s: "1,600 tokens x 384\n(1/16 resolution)", w: 1.1 },
  { t: "Feature pyramid", s: "4 levels, 256 channels\n80, 40, 20, 10", w: 1.15, plain: true },
], HEX.accent2, C.background2);
text(s, "Inside one transformer block (repeated 12 times)", { x: 0.6, y: 2.95, w: 8, h: 0.35, fontSize: 14, bold: true, color: C.accent2 });
flow(s, 0.6, 3.35, 12.1, 0.8, [
  { t: "LayerNorm", w: 0.8, plain: true }, { t: "Self-attention", s: "6 heads", w: 1.1, plain: true }, { t: "+ skip", w: 0.6, plain: true },
  { t: "LayerNorm", w: 0.8, plain: true }, { t: "MLP", s: "384 > 1,536 > 384", w: 1.2, plain: true }, { t: "+ skip", w: 0.6, plain: true },
], HEX.accent2, C.background2);
const dstats = [[mp(M.dino.arch.trunk), "parameters"], ["12", "transformer blocks"], ["1.689 B", "training images, no labels"]];
dstats.forEach(([big, small], i) => {
  const x = 0.6 + i * 2.25;
  text(s, big, { x, y: 4.5, w: 2.15, h: 0.8, fontSize: 38, bold: true, color: C.accent2, valign: "middle" });
  text(s, small, { x, y: 5.3, w: 2.15, h: 0.6, fontSize: 12, color: C.accent6 });
});
card(s, 7.5, 4.45, 5.2, 2.3);
text(s, [
  { text: "Advantages", options: { bold: true, color: C.accent3, breakLine: true } },
  { text: "Best features with the backbone frozen", options: { bullet: true, breakLine: true } },
  { text: "Global context from the first block", options: { bullet: true, breakLine: true } },
  { text: "Disadvantages", options: { bold: true, color: C.accent4, breakLine: true } },
  { text: "About 900 ms per frame on CPU", options: { bullet: true, breakLine: true } },
  { text: "One scale only; cost grows fast with image size", options: { bullet: true } },
], { x: 7.7, y: 4.55, w: 4.85, h: 2.1, fontSize: 13, paraSpaceAfter: 3 });
text(s, "Our training: backbone frozen; blocks 6, 9, 10 and 12 feed the pyramid.", { x: 0.6, y: 6.1, w: 6.6, h: 0.6, fontSize: 13, color: C.accent6 });
s.addNotes("The image is cut into 16 by 16 pixel patches, giving 1,600 tokens. Twelve transformer blocks let every token look at "
  + "every other token, so each patch is described in the context of the whole image. That global view helps with smoke. "
  + "The output is a feature map, not boxes, so we build a four-level pyramid from four of the blocks and add a detection head.");

// ---------------------------------------------------------------- 6 CNNs inside
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Models" });
s.addText("Inside MobileNetV3-Large and ResNet-18", { placeholder: "title" });
text(s, `MobileNetV3-Large:  ${mp(M.mbv3.arch.trunk)} parameters, 15 inverted-residual blocks, ${M.mbv3.arch.trunk_leaf_layers.Conv2d} convolutions`,
  { x: 0.6, y: 1.35, w: 12.1, h: 0.4, fontSize: 16, bold: true, color: C.accent3, valign: "middle" });
flow(s, 0.6, 1.8, 12.1, 1.15, [
  { t: "Input", s: "640 x 640", w: 0.7, plain: true }, { t: "Stem", s: "3x3 conv\n16 ch", w: 0.8 },
  { t: "Stages 1-2", s: "3 blocks\n24 ch, 160x160", w: 1 }, { t: "Stage 3", s: "3 blocks\n40 ch, 80x80", w: 1 },
  { t: "Stages 4-5", s: "6 blocks\n112 ch, 40x40", w: 1 }, { t: "Stages 6-7", s: "3 blocks\n960 ch, 20x20", w: 1 },
  { t: "Pyramid", s: "FPN\n4 levels", w: 0.8, plain: true },
], HEX.accent3, C.background2);
text(s, "One block:  1x1 conv (expand)  >  depthwise conv  >  squeeze-and-excite  >  1x1 conv (project)  >  + skip.   Cheap by design: built for phones.",
  { x: 0.6, y: 3.05, w: 12.1, h: 0.4, fontSize: 13, color: C.accent6, valign: "middle" });
text(s, `ResNet-18:  ${mp(M.r18.arch.trunk)} parameters, 8 residual blocks, ${M.r18.arch.trunk_leaf_layers.Conv2d} convolutions`,
  { x: 0.6, y: 3.85, w: 12.1, h: 0.4, fontSize: 16, bold: true, color: C.accent4, valign: "middle" });
flow(s, 0.6, 4.3, 12.1, 1.15, [
  { t: "Input", s: "640 x 640", w: 0.7, plain: true }, { t: "Stem", s: "7x7 conv\n+ max-pool", w: 0.8 },
  { t: "layer1", s: "2 blocks\n64 ch, 160x160", w: 1 }, { t: "layer2", s: "2 blocks\n128 ch, 80x80", w: 1 },
  { t: "layer3", s: "2 blocks\n256 ch, 40x40", w: 1 }, { t: "layer4", s: "2 blocks\n512 ch, 20x20", w: 1 },
  { t: "Pyramid", s: "FPN\n4 levels", w: 0.8, plain: true },
], HEX.accent4, C.background2);
text(s, "One block:  3x3 conv + BatchNorm  >  ReLU  >  3x3 conv + BatchNorm  >  + skip  >  ReLU.   The skip connection is what made deep networks trainable.",
  { x: 0.6, y: 5.55, w: 12.1, h: 0.4, fontSize: 13, color: C.accent6, valign: "middle" });
text(s, "Both: pretrained on ImageNet-1k with labels; frozen in stage 1; the 80x80, 40x40 and 20x20 maps feed a standard Feature Pyramid Network.",
  { x: 0.6, y: 6.2, w: 12.1, h: 0.5, fontSize: 14, valign: "middle" });
s.addNotes("Both are convolutional networks that shrink the image step by step while adding channels, so they produce several "
  + "scales naturally. MobileNetV3 uses depthwise convolutions to stay cheap. ResNet-18 uses plain 3 by 3 convolutions with "
  + "skip connections. Each layer sees only a local neighbourhood, unlike the transformer.");

// ---------------------------------------------------------------- 7 heads
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Models" });
s.addText("Two detection heads: Faster R-CNN and FCOS", { placeholder: "title" });
text(s, "Faster R-CNN:  two stages, 14.5M parameters, 4 losses", { x: 0.6, y: 1.35, w: 12.1, h: 0.4, fontSize: 16, bold: true, color: C.accent1, valign: "middle" });
flow(s, 0.6, 1.8, 12.1, 1.15, [
  { t: "Pyramid", s: "4 levels", w: 0.7, plain: true },
  { t: "Stage 1: proposals", s: "6 anchor boxes per location:\nobject or not?", w: 1.35 },
  { t: "~300 proposals", s: "after NMS", w: 0.9, plain: true },
  { t: "RoIAlign", s: "crop each to 7x7", w: 0.9 },
  { t: "Stage 2: box head", s: "2 FC layers:\nclass + refined box", w: 1.2 },
  { t: "Output", s: "boxes, class,\nscore", w: 0.75, plain: true },
], HEX.accent1, C.background2);
text(s, [{ text: "+ ", options: { bold: true, color: C.accent3 } }, { text: "most accurate; clean scores, so a strict threshold (0.90) works      " },
  { text: "- ", options: { bold: true, color: C.accent4 } }, { text: "large head; cannot be exported as one batch for many cameras" }],
  { x: 0.6, y: 3.05, w: 12.1, h: 0.45, fontSize: 13, valign: "middle" });
text(s, `FCOS:  one stage, no anchors, ${mp(M.fcos.arch.det_head)} parameters, 3 losses`, { x: 0.6, y: 3.95, w: 12.1, h: 0.4, fontSize: 16, bold: true, color: C.accent5, valign: "middle" });
flow(s, 0.6, 4.4, 12.1, 1.15, [
  { t: "Pyramid", s: "8,500 locations", w: 0.8, plain: true },
  { t: "Two conv towers", s: "classification tower\nregression tower", w: 1.2 },
  { t: "Every location predicts", s: "class score, 4 box distances,\ncentre-ness", w: 1.5 },
  { t: "Score", s: "sqrt(class x centre-ness)\ntop 300, NMS", w: 1.25 },
  { t: "Output", s: "boxes, class,\nscore", w: 0.75, plain: true },
], HEX.accent5, C.background2);
text(s, [{ text: "+ ", options: { bold: true, color: C.accent3 } }, { text: "simple, 24 times smaller head; fixed-size output, so many cameras can be batched      " },
  { text: "- ", options: { bold: true, color: C.accent4 } }, { text: "compressed scores (max about 0.75); lower accuracy as trained" }],
  { x: 0.6, y: 5.65, w: 12.1, h: 0.45, fontSize: 13, valign: "middle" });
card(s, 0.6, 6.25, 12.1, 0.55);
text(s, "Both read the same 4-level feature pyramid and output the same thing: boxes with a class and a score. Only the way they get there differs.",
  { x: 0.85, y: 6.25, w: 11.6, h: 0.55, fontSize: 13, valign: "middle" });
s.addNotes("Faster R-CNN works in two stages: a proposal network suggests about 300 regions, then a box head examines each one. "
  + "That second look gives well-separated scores. FCOS has no proposals and no anchors: every location predicts a class, a "
  + "box and a centre-ness value directly. It is simpler and can be batched across cameras, but its scores are compressed.");

// ---------------------------------------------------------------- 8 results backbones
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Results: DINOv3 leads at every one of the 40 epochs", { placeholder: "title" });
const labels = M.dino.epochs.map((e) => String(e.epoch));
s.addChart(pres.charts.LINE, KEYS.map((k) => ({ name: M[k].label, labels, values: M[k].epochs.map((e) => e.mAP50) })), {
  x: 0.5, y: 1.3, w: 7.3, h: 5.5, chartColors: KEYS.map((k) => MODEL_HEX[k]), lineSize: 3, lineDataSymbol: "none",
  showTitle: true, title: "Validation mAP@0.5 after each epoch", titleFontSize: 14, titleFontFace: "+mn-lt", titleColor: HEX.dk1,
  showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt", legendColor: HEX.dk1,
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelColor: HEX.accent6, valAxisLabelColor: HEX.accent6, catAxisLabelFrequency: 5,
  valAxisMinVal: 0.2, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0",
  valGridLine: { color: "E3E7EB", size: 0.75 }, catGridLine: { style: "none" },
});
s.addTable([
  [head("Model"), head("Best val", { align: "right" }), head("Test", { align: "right" })],
  ...KEYS.map((k) => [cell(M[k].short, { bold: true, color: MODEL_HEX[k] }), cell(f4(M[k].best_val), { align: "right" }), cell(f4(tmap(M[k])), { align: "right" })]),
], Object.assign({ x: 8.1, y: 1.4, w: 4.6, colW: [2.3, 1.15, 1.15], rowH: 0.42 }, TABLE));
text(s, "mAP@0.5", { x: 8.1, y: 3.6, w: 4.6, h: 0.3, fontSize: 11, color: C.accent6 });
text(s, [
  { text: "Same head, same data, same settings: only the backbone differs", options: { bullet: true, breakLine: true } },
  { text: `DINOv3 is ${((tmap(M.dino) - tmap(M.mbv3)) * 100).toFixed(1)} points ahead of MobileNetV3 and ${((tmap(M.dino) - tmap(M.r18)) * 100).toFixed(1)} ahead of ResNet-18 on test`, options: { bullet: true, breakLine: true } },
  { text: "No model overfitted: all four were still improving at epoch 40; the smaller ones underfit", options: { bullet: true } },
], { x: 8.1, y: 4.0, w: 4.6, h: 2.8, fontSize: 14, paraSpaceAfter: 8 });
s.addNotes("Validation mAP after each of the 40 epochs. The order DINOv3, MobileNetV3, ResNet-18 holds at every epoch. The three "
  + "Faster R-CNN models train almost the same number of parameters, so the difference comes from the frozen backbone features. "
  + "On overfitting: none of the models overfitted. Validation accuracy was still rising at the end for all four, and the best "
  + "epoch was 38 or 39. The smaller models underfit instead: ResNet-18 ends with the highest training loss, and the FCOS model "
  + "trains only 1.5 million parameters. Note for questions: DINOv3's numbers were recorded before a test-time speed setting was "
  + "changed, which costs about 0.005 mAP, so its lead is about half a point smaller than shown. It still leads clearly.");

// ---------------------------------------------------------------- 9 false alarms, speed, heads
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Results: false alarms, speed, and the two heads", { placeholder: "title" });
s.addTable([
  [head("Model"), head("Detected at 1% false alarms", { align: "right" }), head("False-alarm rate", { align: "right" }),
   head("CPU ms / frame", { align: "right" }), head("Parameters", { align: "right" })],
  ...KEYS.map((k) => [cell(M[k].label, { bold: true, color: MODEL_HEX[k] }), cell(pct(M[k].val_op.recall_any), { align: "right" }),
    cell(pct(M[k].val_op.fpr, 2), { align: "right" }), cell(ms(M[k]), { align: "right" }), cell(mp(M[k].arch.total), { align: "right" })]),
], Object.assign({ x: 0.6, y: 1.4, w: 12.1, colW: [4.3, 2.6, 1.9, 1.7, 1.6], rowH: 0.45 }, TABLE));
text(s, "Validation images; threshold raised until at most 1% of fire-free images trigger a detection. CPU time on the project laptop.",
  { x: 0.6, y: 3.75, w: 12.1, h: 0.3, fontSize: 11, color: C.accent6 });
const obs = [
  ["False alarms", `DINOv3 and MobileNetV3 detect the same share of fire/smoke images, but DINOv3 needs less than half the false alarms (${pct(M.dino.val_op.fpr, 2)} against ${pct(M.mbv3.val_op.fpr, 2)})`, C.accent2],
  ["Speed", "MobileNetV3 is about 2.5 times faster than DINOv3. ResNet-18 is slower than MobileNetV3 and less accurate", C.accent3],
  ["Faster R-CNN and FCOS", "On the same backbone FCOS is not faster (924 ms against 884 ms). Its gain is batched export for many cameras; its scores top out near 0.75", C.accent5],
];
obs.forEach(([h, d, col], i) => {
  const x = 0.6 + i * 4.1;
  card(s, x, 4.3, 3.85, 2.3);
  text(s, h, { x: x + 0.25, y: 4.45, w: 3.4, h: 0.45, fontSize: 18, bold: true, color: col, valign: "middle" });
  text(s, d, { x: x + 0.25, y: 5.0, w: 3.4, h: 1.5, fontSize: 14 });
});
s.addNotes("For an alarm system the key number is how many fires are detected while false alarms stay under 1%. DINOv3 and "
  + "MobileNetV3 both detect about 87%, but DINOv3 does it at 0.37% false alarms against 0.87%. ResNet-18 detects only 68%. "
  + "On the heads: the FCOS model here also uses a smaller backbone, so its lower accuracy is not due to the head alone. What we "
  + "measured on the same backbone is speed and export: FCOS is not faster, but it can be exported as one batch.");

// ---------------------------------------------------------------- 10 decision
pres.addSection({ title: "Decision" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Decision" });
s.addText("Decision: continue with DINOv3 + Faster R-CNN", { placeholder: "title" });
const reasons = [
  ["Most accurate", "Best mAP on validation and test, on both smoke and fire"],
  ["Fewest false alarms", "Same detections as MobileNetV3 at less than half the false-alarm rate"],
  ["Features work without training", "Best result with every backbone frozen"],
  ["Clean scores", "Faster R-CNN keeps a strict 0.90 threshold usable"],
];
reasons.forEach(([h, d], i) => {
  const y = 1.45 + i * 1.05;
  badge(s, 0.6, y + 0.05, i + 1, C.accent2);
  text(s, h, { x: 1.25, y, w: 6.0, h: 0.4, fontSize: 18, bold: true, valign: "middle" });
  text(s, d, { x: 1.25, y: y + 0.42, w: 6.0, h: 0.45, fontSize: 14, color: C.accent6 });
});
const stats = [[f4(M.dino.best_val), "validation mAP@0.5"], [f4(tmap(M.dino)), "test mAP@0.5"],
  [pct(M.dino.val_op.recall_any), `detected at ${pct(M.dino.val_op.fpr, 2)} false alarms`]];
card(s, 7.9, 1.4, 4.8, 4.1);
stats.forEach(([big, small], i) => {
  const y = 1.55 + i * 1.3;
  text(s, big, { x: 8.2, y, w: 4.2, h: 0.75, fontSize: 40, bold: true, color: C.accent2, valign: "middle" });
  text(s, small, { x: 8.2, y: y + 0.75, w: 4.2, h: 0.35, fontSize: 13, color: C.accent6 });
});
card(s, 0.6, 5.75, 12.1, 1.0);
text(s, [{ text: "What it costs:  ", options: { bold: true, color: C.accent1 } },
  { text: "about 900 ms per frame on CPU and no batched export. MobileNetV3 (2.5 times faster) and FCOS (batchable) are kept for the fast version; ResNet-18 is dropped." }],
  { x: 0.85, y: 5.8, w: 11.6, h: 0.9, fontSize: 14, valign: "middle" });
s.addNotes("Four reasons. It is the most accurate. It gives the fewest false alarms for the same detections, which is the central "
  + "requirement. Its features are the best even without training them. And Faster R-CNN gives clean scores. The cost is speed "
  + "and export, which concern deployment, so accuracy is settled first and MobileNetV3 and FCOS are kept for a fast version.");

// ---------------------------------------------------------------- complete structure
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Decision" });
s.addText("Complete structure of the selected model", { placeholder: "title" });
{
  const A = M.dino.arch;
  const rows = [
    { t: "Camera frame and pre-processing", sub: "resized to 640 x 640, shape kept", plain: true,
      d: "The camera frame is resized to 640 x 640 with its shape kept, then normalised." },
    { t: "Backbone: DINOv3 ViT-S/16", sub: `12 transformer blocks  |  ${mp(A.trunk)} parameters, frozen`,
      d: "The image becomes 1,600 patch tokens that pass through 12 transformer blocks. Pretrained on 1.689 billion images." },
    { t: "Feature pyramid", sub: `4 levels, 256 channels  |  ${mp(A.neck)}, trained`,
      d: "Four feature maps (80, 40, 20 and 10 cells wide), so that both large and small fires are covered." },
    { split: true },
    { t: "Alarm logic", sub: "threshold  +  6 detections in 15 frames", plain: true,
      d: "The threshold is set for at most 1% false alarms. On video, an alarm needs 6 detections within 15 frames." },
  ];
  const X = 0.6, W = 5.7, H = 0.74, G = 0.3, Y0 = 1.4;
  let y = Y0, n = 0;
  const explain = (num, yy, title, body) => {
    badge(s, 6.85, yy + 0.14, num, C.accent2, 0.42);
    text(s, [{ text: title + "  ", options: { bold: true } }, { text: body }],
      { x: 7.45, y: yy, w: 5.25, h: H, fontSize: 13, valign: "middle" });
  };
  rows.forEach((r, i) => {
    if (r.split) {
      const hw = (W - 0.25) / 2;
      [["Detection head: Faster R-CNN", `boxes, class, score  |  ${mp(A.det_head)}, trained`, HEX.accent1],
       ["Scene classifier", `P(smoke), P(fire)  |  ${mp(A.scene)}, trained`, "6A1B9A"]].forEach(([t, sub, col], j) => {
        const bx = X + j * (hw + 0.25);
        s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y, w: hw, h: H, rectRadius: 0.06, fill: { color: C.background2 }, line: { color: col, width: 1.5 } });
        s.addText([{ text: t, options: { bold: true, fontSize: 12, color: C.text1, breakLine: true } },
          { text: sub, options: { fontSize: 10, color: C.accent6 } }],
          { x: bx + 0.05, y, w: hw - 0.1, h: H, align: "center", valign: "middle", margin: 0, isTextBox: true });
      });
      n += 1;
      badge(s, 6.85, y - 0.02, n, C.accent1, 0.34);
      text(s, [{ text: "Detection head  ", options: { bold: true } },
        { text: "Proposes about 300 regions, then classifies each as smoke or fire and refines its box." }],
        { x: 7.35, y: y - 0.12, w: 5.35, h: 0.5, fontSize: 12, valign: "middle" });
      n += 1;
      badge(s, 6.85, y + 0.44, n, C.accent1, 0.34);
      text(s, [{ text: "Scene classifier  ", options: { bold: true } },
        { text: "Whole-image check using 7 glow statistics, for fire hidden behind objects." }],
        { x: 7.35, y: y + 0.36, w: 5.35, h: 0.5, fontSize: 12, valign: "middle" });
    } else {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: X, y, w: W, h: H, rectRadius: 0.06,
        fill: { color: r.plain ? C.background1 : C.background2 }, line: { color: r.plain ? HEX.accent6 : HEX.accent2, width: 1.5 } });
      s.addText([{ text: r.t, options: { bold: true, fontSize: 14, color: C.text1, breakLine: true } },
        { text: r.sub, options: { fontSize: 11, color: C.accent6 } }],
        { x: X + 0.05, y, w: W - 0.1, h: H, align: "center", valign: "middle", margin: 0, isTextBox: true });
      n += 1;
      explain(n, y, r.t.split(":")[0], r.d);
    }
    if (i < rows.length - 1) {
      s.addShape(pres.shapes.LINE, { x: X + W / 2, y: y + H + 0.03, w: 0, h: G - 0.06,
        line: { color: HEX.accent6, width: 1.5, endArrowType: "triangle" } });
    }
    y += H + G;
  });
  card(s, 0.6, 6.55, 12.1, 0.36);
  text(s, [{ text: "Whole model:  ", options: { bold: true, color: C.accent1 } },
    { text: `${mp(A.total)} parameters, of which ${mp(A.trainable_frozen_trunk)} are trained in stage 1 (the backbone is frozen).` }],
    { x: 0.85, y: 6.55, w: 11.6, h: 0.36, fontSize: 12, valign: "middle" });
}
s.addNotes("The full structure of the model we continue with, top to bottom. The frame is resized to 640 by 640. DINOv3 turns it into "
  + "1,600 tokens and processes them through 12 transformer blocks; this part has 21.6 million parameters and is frozen. A feature "
  + "pyramid produces four scales. Then two outputs work side by side: the Faster R-CNN head gives boxes with a class and a score, "
  + "and the scene classifier gives a whole-image probability for smoke and fire, using glow statistics, for the case where the "
  + "flame itself is hidden. Finally the alarm logic applies a threshold and, on video, requires six detections in fifteen frames. "
  + "In total 39.3 million parameters, 17.7 million of them trained in stage 1.");

// ---------------------------------------------------------------- problems faced
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Decision" });
s.addText("Problems faced up to stage 1, and how they were solved", { placeholder: "title" });
{
  const rows = [
    ["Data shortage: no fire footage from the site",
     "Collected 400+ online videos, kept 150+, and annotated 2,100 frames in Roboflow. The public D-Fire dataset (21,527 images, labels checked) is used for the benchmark training"],
    ["The dataset is mostly daytime and outdoor",
     "Augmentation that imitates the site: low light with sensor noise, infrared-style grey images and motion blur"],
    ["No examples of fire hidden behind objects",
     "Flames are partly covered during training while the label is kept, and a scene classifier reads 7 glow statistics"],
    ["False alarms had to be measured, not guessed",
     "45% of the images contain no fire or smoke; the threshold is set where at most 1% of them trigger a detection"],
    ["Limited GPU time: free Kaggle sessions with a time limit",
     "The backbone is frozen, which makes training cheap; a checkpoint is saved after every epoch so a run can resume"],
    ["DINOv3 gives features at one scale only",
     "A 4-level feature pyramid was built on top of it, so small and large fires are both covered"],
  ];
  text(s, "Problem", { x: 0.6, y: 1.3, w: 4.2, h: 0.35, fontSize: 14, bold: true, color: C.accent4, valign: "middle" });
  text(s, "How it was solved", { x: 5.35, y: 1.3, w: 7.35, h: 0.35, fontSize: 14, bold: true, color: C.accent3, valign: "middle" });
  rows.forEach(([p, sol], i) => {
    const y = 1.72 + i * 0.86;
    card(s, 0.6, y, 4.2, 0.74);
    badge(s, 0.75, y + 0.16, i + 1, C.accent4, 0.42);
    text(s, p, { x: 1.3, y, w: 3.4, h: 0.74, fontSize: 13, bold: true, valign: "middle" });
    s.addShape(pres.shapes.LINE, { x: 4.86, y: y + 0.37, w: 0.43, h: 0, line: { color: HEX.accent6, width: 1.5, endArrowType: "triangle" } });
    card(s, 5.35, y, 7.35, 0.74);
    text(s, sol, { x: 5.55, y, w: 6.95, h: 0.74, fontSize: 13, valign: "middle" });
  });
}
s.addNotes("Six problems met on the way to stage 1. First, data: the site has no recorded fires, so we could not collect fire images "
  + "there. We collected and annotated 2,100 frames of our own in Roboflow, and used the public D-Fire dataset for the benchmark "
  + "training after checking every label file: 8 boxes that ran outside the image were corrected, 18 zero-area boxes are filtered, "
  + "and we created a validation split, keeping the test set untouched. Second, D-Fire is mostly daytime, so we simulate low light and infrared "
  + "cameras. Third, hidden fire: we cover part of the flame during training and add a scene classifier. Fourth, false alarms: "
  + "almost half the images have no fire, which lets us measure them. Fifth, compute: freezing the backbone and saving a "
  + "checkpoint every epoch let us train within free Kaggle sessions. Sixth, DINOv3 outputs one scale, so we built a pyramid.");

// ---------------------------------------------------------------- 11 next goals
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Decision" });
s.addText("Next goals", { placeholder: "title" });
const goals = [
  ["Fine-tune the backbone (stage 2)", "Unfreeze the last 2 transformer blocks at a 20 times lower learning rate"],
  ["Small and distant fires", "Higher input resolution and scale augmentation"],
  ["Hard-negative training", "Hard hats, hi-vis clothing, lamps and haze labelled as 'not fire'"],
  ["Video testing", "Industrial footage with temporal confirmation: 6 detections in 15 frames"],
  ["A fast version for 70 cameras", "MobileNetV3 backbone and FCOS head, exported for GPU batching"],
  ["Data from the site's own cameras", "Labelled with one consistent rule"],
];
goals.forEach(([h, d], i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 1.45 + Math.floor(i / 2) * 1.3;
  card(s, x, y, 5.95, 1.12);
  badge(s, x + 0.25, y + 0.33, i + 1);
  text(s, h, { x: x + 0.95, y: y + 0.12, w: 4.8, h: 0.42, fontSize: 17, bold: true, valign: "middle" });
  text(s, d, { x: x + 0.95, y: y + 0.55, w: 4.8, h: 0.5, fontSize: 13, color: C.accent6 });
});
text(s, [{ text: "Documents:  ", options: { bold: true } },
  { text: "project log, model selection report (21 pages, results of every epoch), training report, technical and research report, source code" }],
  { x: 0.6, y: 5.55, w: 12.1, h: 0.9, fontSize: 14, color: C.accent6, valign: "middle" });
s.addNotes("The plan from here. First fine-tune part of the DINOv3 backbone at a low learning rate, which is where overfitting must be "
  + "watched. Then small and distant fires, hard negatives to push false alarms down, testing on video, a fast version for the "
  + "70-camera requirement, and data from the site's own cameras. The full report has the results of every epoch for every model.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  const applyThemePath = process.argv[2];
  if (applyThemePath) {
    const { applyTheme } = require(applyThemePath);
    await applyTheme(OUT, THEME);
    console.log("theme applied");
  }
  console.log("wrote " + OUT);
})();
