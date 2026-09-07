# Project Log — Industrial Fire & Smoke Detection (DINOv3)

Living document. Everything about *what exists*, *what changed*, and *why* lives
here, so the project can be explained to a supervisor, an examiner or a new
teammate without re-reading the code.

Last updated: **2026-09-07**

---

## 1. What this project is

A fire and smoke detection system for a **fixed industrial camera**, built on a
self-supervised **DINOv3** vision backbone with a detection head and an
image-level classifier on top.

The deployment constraints that shape every design decision:

| # | Constraint | Consequence for the design |
|---|---|---|
| 1 | Camera mounted at height; industrial lighting is often poor, but fire is self-luminous | Heavy low-light augmentation; brightness-based cues must work without colour |
| 2 | Runs 24/7; false alarms must be very rare | False-alarm rate is a first-class metric; temporal confirmation before any alarm |
| 3 | Fire may be hidden behind objects, visible only as light cast on surroundings | Image-level classifier + a hand-crafted glow prior + flame-occlusion augmentation |
| 4 | Smoke must be detected too (visible smoke only — invisible/clear smoke is out of scope) | `smoke` is a first-class class, evaluated separately from `fire` |
| 5 | Same camera conditions apply to smoke | Same augmentation pipeline covers both classes |
| 6 | Training happens on Kaggle with the already-uploaded dataset | Scripts auto-discover `data.yaml` under `/kaggle/input`; offline weight path supported |
| 7 | Images first; video verification later | Image training and evaluation are complete; the video layer is written and ready for footage |
| 8 | The project must be explainable at any point | This document |

**Explicitly out of scope right now:** invisible/transparent smoke, thermal/IR
sensor fusion, multi-camera tracking, and any claim about certified fire-safety
compliance.

---

## 2. Decision log

### 2026-09-01/02 — Original plan: YOLOv8
The first pipeline prepared D-Fire into YOLO format and trained `yolov8n`.
That work produced the dataset preparation and validation tooling that is
still in use.

### 2026-09-02 — Switch to DINOv3 (supervisor's requirement)
The supervisor asked for a DINO-based approach with a suitable classifier.
A partial DINOv3 conversion was started but left incomplete.

**This is not just compliance — it fits the problem.** DINOv3 is trained
self-supervised on 1.7B images. Its features are far more general than anything
14k D-Fire images can teach from scratch, which matters because our real
deployment domain (a dark factory at 3 a.m.) is *not* in the training set.
Freezing the trunk keeps that generality intact.

### 2026-09-06 — Completed and rebuilt the DINOv3 pipeline
The inherited draft had several issues that would have wasted GPU time:

| Problem in the draft | Why it mattered | Fix |
|---|---|---|
| Single stride-16 feature map into Faster R-CNN, no pyramid | Small distant flames were effectively undetectable | ViTDet-style 4-level pyramid at strides 8/16/32/64 |
| Whole trunk fine-tuned at `lr=1e-4` | Destroys DINOv3 features on a small dataset, and 3× the training cost | Trunk frozen by default; `--unfreeze-last-n` with a 0.05× LR for stage 2 |
| Validation ran with `model.train()` and an optimizer argument | Confusing and fragile | Proper eval pass; LayerNorm everywhere so no running stats exist to corrupt |
| Model selection by validation **loss** | Loss is a poor proxy for detection quality | Selection by validation **mAP@0.5** |
| No augmentation at all | Model would only ever have seen bright daytime web images | Low-light, IR/grayscale, occlusion, crop, jitter, blur |
| No mAP, no false-positive measurement | The single most important deployment number was invisible | Full metrics module with a negative-image false-alarm sweep |
| No AMP, no warmup, no LR schedule, no resume | Slow, unstable, and fatal if a Kaggle session times out | All added |
| `hf_hub:` prefix on the backbone name, no offline path | Fails in a no-internet Kaggle session | Plain timm name (verified downloadable) + `--backbone-weights` |
| Absolute Windows `path:` in `data.yaml` | Breaks immediately on Kaggle | Loader falls back to the yaml's own directory |
| Video script had no temporal logic | A per-frame detector cannot be deployed 24/7 | N-of-M confirmation, hysteresis, ignore-mask, event log |

---

## 3. Architecture

```
                 letterboxed 640x640 RGB frame
                              |
              +---------------+----------------+
              |                                |
   DINOv3 ViT-S/16 trunk (FROZEN)      glow prior (classical CV)
   21.6M params, 12 blocks              warm tint x local brightness excess
              |                                |
   blocks 5, 8, 9, 11 -> NCHW              7-number summary (see 6a)
              |                                |
   simple feature pyramid (trainable)          |
   strides  8 / 16 / 32 / 64, 256ch            |
              |                     \          |
   Faster R-CNN head              pooled trunk tokens (768)
   RPN + RoIAlign + box head              \    |
              |                          scene classifier (MLP)
     boxes: smoke / fire                  P(smoke), P(fire)
              |                                |
              +--------------+-----------------+
                             |
              temporal confirmation (video only)
              N-of-M + IoU tracking + hysteresis
                             |
                    ALARM / OCCLUDED-FIRE CUE
```

Parameter budget: **39.3M total, 17.7M trainable** (the frozen trunk is 21.6M).

### Why two heads

The **detection head** is what the report benchmarks (mAP, per-class AP).

The **scene head** exists for the deployment requirements:
- it can respond to a fire that never appears as a flame-shaped object,
  because it sees the whole image plus the glow statistics;
- it gives the alarm layer a second, partly independent opinion to cross-check,
  which is the cheapest false-positive filter available.

Both heads share **one** trunk forward pass — the backbone stashes its pooled
tokens so the scene head costs almost nothing.

### Why the trunk is frozen

1. It preserves the DINOv3 representation, which generalises to lighting
   conditions D-Fire does not contain.
2. No gradients flow through the ViT, so activations are not stored — this is
   what makes a 640px, batch-8 run fit comfortably on a Kaggle T4.
3. It is the honest interpretation of "use DINO": a strong frozen
   representation plus a light task head.

Stage 2 (`--unfreeze-last-n 2 --lr 5e-5`) can release the last blocks once the
head has converged, if the metrics justify it.

---

## 4. How each requirement is addressed

### R1 / R5 — Low light, both classes
`night_augment` in `scripts/fire_smoke/dataset.py` (default 35% of training images).

It does **not** simply multiply the image down. On a real camera at night the
auto-exposure closes, the surroundings sink into the noise floor, and the flame
stays blown out. So the darkening factor is relaxed where a pixel is already
bright (smoothstep over luminance 0.60–0.95), a cool colour cast is applied, and
sensor noise is added in proportion to the simulated gain.

`gray` augmentation (10%) removes colour entirely, simulating IR/night-mode
cameras and forcing the model not to rely on the orange cue alone.

### R2 — Very low false positives
Three separate mechanisms, because no single one is enough:

1. **Measurement.** D-Fire contains 9,838 verified-negative images (45.7% of the
   dataset). `alarm_sweep` in `scripts/fire_smoke/metrics.py` reports, for every
   confidence threshold, the fraction of those that produce any box. The
   operating threshold is *chosen from this curve* against an explicit budget
   (`--target-fpr`, default 1%), not guessed at 0.25.
2. **Temporal confirmation.** `scripts/fire_smoke/temporal.py` requires N
   detections inside a sliding window of M frames, all linked to the same track
   by IoU, before raising an alarm — with hysteresis so a briefly obscured flame
   does not toggle the alarm off. Uncorrelated flicker (glare, a headlight, a
   welding flash, a bird) does not accumulate on one track.
3. **Ignore mask.** `--roi-mask` suppresses regions that are legitimately hot or
   bright all day: a furnace mouth, a welding bay, a flare stack.

Worked example: at 5 fps, a 1% per-frame image-level FPR is ~180 false boxes per
hour. Requiring 6 linked hits in 15 frames removes essentially all of them,
because they are not the same object in the same place.

### R3 — Fire hidden behind objects
Three parts:

1. **Occlusion augmentation** (`occlude_fire`, 25% of images containing fire).
   A synthetic obstacle covers 45–85% of a flame box while the label is kept.
   D-Fire has essentially no examples of "fire you cannot see but whose light
   you can", so we manufacture them: the box stays, the flame does not, and the
   only remaining evidence is the illumination cast on the rest of the scene.
2. **Glow prior** (`scripts/fire_smoke/glow.py`). `warm_tint × local_brightness_excess`
   — high where a surface is redder than it is blue *and* brighter than its own
   neighbourhood, which is what a hidden light source spilling onto a wall looks
   like, as opposed to uniform daylight. A 7-number summary is fed to the scene
   classifier as extra input, so the network learns when those statistics mean
   fire and when they just mean a sunset.

   **This was measured, not assumed** — see §6a. The headline: `warm_cast` is a
   genuinely useful cue that gets *stronger* in the dark (AUC 0.73 → 0.83),
   the `glow_*` statistics are informative but *inverted*, and the whole prior
   goes to chance on IR/mono cameras.
3. **Distinct, lower-severity cue at inference.** When the scene head is
   confident about fire, the detector has no confirmed flame box, and the glow
   prior finds a strong warm region, the video script raises
   `OCCLUDED_FIRE_CUE` and draws a dashed box on the glow — deliberately *not*
   the same thing as a confirmed flame alarm.

**Honest limitation:** this is trained on synthetic occlusions and validated by
whatever the scene head learns. It has not yet been measured on real occluded-fire
footage, because no such footage exists in the dataset. That measurement is
pending the videos being collected (see §8).

### R4 — Smoke
`smoke` is a full class with its own AP, its own recall column in the alarm
sweep, and its own scene-head output. Visible smoke only; transparent or
invisible smoke is explicitly out of scope and stated as such in the report.

### R6 — Kaggle training
`find_data_yaml` searches `/kaggle/input` recursively, so whichever name the
uploaded dataset has, it is found. `load_data_config` ignores a stale absolute
`path:` field. Weights, plots, metrics and an optional zip land in
`/kaggle/working`. `--resume` recovers from a session timeout.

If a session has no internet, `--backbone-weights` loads the DINOv3 trunk from a
file attached as a Kaggle dataset.

### R7 — Images now, video later
Image training and evaluation are complete and tested. The video layer is
written and syntax-clean but **not yet run against real footage** — that is the
next milestone.

### R8 — Explainability
This document.

---

## 5. Repository map

```
scripts/
  fire_smoke/                     the reusable package
    __init__.py                   class names (background / smoke / fire)
    backbone.py                   DINOv3 trunk + simple feature pyramid + scene head module
    glow.py                       classical fire-glow prior: map, 10-dim features, hotspot
    dataset.py                    D-Fire loader, letterboxing, all augmentation
    model.py                      FireSmokeDetector = Faster R-CNN + scene head, save/load
    metrics.py                    AP/mAP, negative-image alarm sweep, operating-point picker
    temporal.py                   N-of-M confirmation with IoU tracking and hysteresis

  prepare_dfire_detection.py      raw D-Fire  ->  clean detection dataset + data.yaml
  validate_detection_dataset.py   integrity + positive/negative balance check
  check_training_readiness.py     environment / GPU / dataset / DINOv3 weight pre-flight
  train_dinov3_detector.py        the training run
  eval_dinov3_detector.py         test-split metrics, operating point, false-positive dump
  predict_video_dinov3.py         video inference with alarm logic and an event log
  analyze_glow_prior.py           diagnostic: AUC of each glow feature, fire vs negative
  list_kaggle_inputs.py           lists what Kaggle actually mounted
  build_project_plan_pdf.py       report/plan PDF builder (from the earlier stage)

datasets/
  raw/D-Fire/                     original download (git-ignored)
  processed/fire_smoke_yolo/      prepared dataset currently in use (git-ignored)

PROJECT_LOG.md                    this file
README_TRAINING.md                local training instructions
README_KAGGLE.md                  Kaggle workflow
requirements_dinov3.txt           dependencies
```

### Removed in the DINOv3 switch
`train_yolo_fire_smoke.py`, `kaggle_train_yolo.py`, `prepare_dfire_yolo.py`,
`validate_yolo_dataset.py`, `preflight_yolo.py`, `predict_video.py`,
`build_m1_approach_pdf.py`. Their useful logic was carried into the new scripts.
`yolov8n.pt` is no longer used.

---

## 6. Dataset status

Source: **D-Fire** (Gaulia/Pedro Vinícius et al.), YOLO-format boxes,
`0 = smoke`, `1 = fire`, empty label file = verified negative.

Current prepared split (`datasets/processed/fire_smoke_yolo`, verified 2026-09-06):

| split | images | positives | negatives | smoke boxes | fire boxes |
|---|---|---|---|---|---|
| train | 13,776 | 7,553 | 6,223 | 7,654 | 9,653 |
| val | 3,445 | 1,835 | 1,610 | 1,889 | 2,154 |
| test | 4,306 | 2,301 | 2,005 | 2,311 | 2,878 |
| **total** | **21,527** | 11,689 | **9,838 (45.7%)** | 11,854 | 14,685 |

The large negative fraction is the reason the false-alarm requirement can be
measured at all, rather than merely asserted.

**Known data issue:** 18 label lines across the three splits describe zero-area
boxes. The dataset loader drops any box smaller than 2px per side, so training
is unaffected (torchvision would otherwise refuse the batch). Re-running
`prepare_dfire_detection.py` also strips them.

**Domain gap, stated plainly:** D-Fire is mostly daytime, web-sourced, close-range
imagery. Our target is an elevated, fixed, often dark industrial view. The
augmentation pipeline narrows the gap; it does not close it. Real site footage
is required for the fine-tuning stage.

---

## 6a. Measured: is the glow prior real?

Run `python scripts/analyze_glow_prior.py --split val --per-class 100` to
reproduce. ROC AUC, fire vs verified-negative (0.5 = says nothing, <0.5 = the
feature is inverted, and `|AUC − 0.5|` is the information content):

| feature | as-is | night | night + IR |
|---|---|---|---|
| `warm_cast` | **0.725** | **0.830** | 0.500 |
| `glow_p99` | 0.145 | 0.169 | 0.500 |
| `glow_mean` | 0.145 | 0.202 | 0.500 |
| `frac_fire_chroma` | 0.189 | 0.493 | 0.500 |
| `frac_highlight` | 0.534 | 0.627 | 0.627 |

Three conclusions, all of which changed the design:

1. **`warm_cast` earns its place.** It is the only strong positive cue, and it
   gets *better* in the dark — exactly where the box detector is weakest. This
   is the concrete mechanism behind requirement 3.
2. **The `glow_*` statistics are inverted.** D-Fire negatives contain sunsets and
   warm street/indoor lighting that out-glow real fires. They stay in the
   feature vector because an inverted-but-informative feature is still usable by
   a learned classifier — but they must never be thresholded directly, and the
   occluded-fire cue is therefore gated by the *learned scene head*, with the
   glow map used only to decide *where* to draw the marker, never *whether*.
3. **On IR/mono cameras the prior is worthless.** Every colour-derived feature
   goes to exactly 0.5. Only `frac_highlight` survives. If the site cameras
   switch to IR at night, requirement 3 rests entirely on the DINOv3 features.
   This is a limitation to state in the report, not a bug to fix.

### Features deliberately excluded

`mean_luma`, `std_luma` and `frac_dark` are the **strongest** discriminators on
D-Fire (AUC 0.82–0.84) and were removed anyway.

They work for the wrong reason: fire photographs are disproportionately taken at
night, so on this dataset "dark image" predicts "fire". A camera that is dark
every night from 18:00 would drive P(fire) up every night, forever — precisely
the failure mode requirement 2 forbids. The `night` augmentation attacks the
same shortcut from the data side by darkening both classes.

This does not *eliminate* the shortcut, because the DINOv3 tokens also encode
scene brightness. It removes the most direct path to it. Worth re-checking on
real footage, where the correlation should not exist at all.

---

## 7. Change log

### 2026-09-06 — DINOv3 pipeline completed
- Added the `scripts/fire_smoke/` package: `backbone`, `glow`, `dataset`,
  `model`, `metrics`, `temporal`.
- Rewrote `train_dinov3_detector.py`: frozen trunk, feature pyramid, AMP,
  warmup + cosine schedule, gradient accumulation and clipping, resume,
  mAP-based model selection, per-epoch false-alarm reporting, 4-panel plots.
- Rewrote `eval_dinov3_detector.py`: full metrics, operating-point selection
  against an FPR budget, and a dump of the worst false positives as images.
- Rewrote `predict_video_dinov3.py`: temporal confirmation, ignore mask,
  occluded-fire cue, HUD, CSV event log.
- Rewrote `validate_detection_dataset.py` and `check_training_readiness.py`.
- Retired the YOLO `requirements.txt` (now a pointer to `requirements_dinov3.txt`).
- Split the checkpoints: `best.pt` is weights-only (~160 MB, the file you
  download), `last.pt` carries the optimizer for `--resume` (~300 MB). Added
  `--init-from` so stage 2 starts a clean run from stage-1 weights, rather than
  inheriting stage 1's epoch counter and optimizer state through `--resume`.
- Added a minimum-box-size filter in the video path (see verification below).
- **Measured the glow prior instead of trusting it** (new `analyze_glow_prior.py`).
  Cut the feature vector from 10 to 7 by removing `mean_luma`, `std_luma` and
  `frac_dark`: they were the strongest features on D-Fire but only because fire
  photos are disproportionately dark, which would have taught the model to alarm
  on nightfall. Corrected the claims in `glow.py` and §4/§6a to match what was
  actually measured. See §6a.
- Found and documented the 18 zero-area boxes in the prepared dataset;
  `prepare_dfire_detection.py` now drops them and reports the count.

#### Verification performed (2026-09-06, CPU)

| Check | Result |
|---|---|
| DINOv3 weights download through timm 1.0.29 | OK — `vit_small_patch16_dinov3.lvd1689m`, 21.6M params, ungated |
| `forward_intermediates` gives NCHW maps | OK — 4 × (B, 384, H/16, W/16) |
| Pyramid shapes at 640px | OK — p3 80², p4 40², p5 20², p6 10², 256ch |
| Full model forward + backward | OK — 39.3M params, 17.7M trainable |
| **Can the head fit the data?** | **OK — 4 images, 260 steps: loss 2.53 → 0.01; 5/5 GT boxes matched at IoU≥0.5, conf ≥0.97; scene head exact** |
| Full training loop (epochs, eval, plots, checkpoints, early stop) | OK |
| Eval script: mAP, alarm sweep, threshold picker, FP dump | OK |
| Dataset integrity across all 21,527 images | OK — 18 zero-area boxes found and handled |
| Video pipeline end-to-end (letterbox → alarm → event CSV) | OK |
| Letterbox inversion arithmetic | OK — verified against hand-computed coordinates |

Alarm-logic unit tests (`TemporalConfirmer`, window 15, 6-of-15, hysteresis):

| Scenario | Expected | Result |
|---|---|---|
| Steady fire, 30 frames | alarm | raised at frame 5 |
| 60 frames of high-confidence flicker at random positions | **no alarm** | **never raised** |
| Persistent detection at conf 0.40 (below the 0.50 gate) | no alarm | never raised |
| Fire, 3-frame occlusion, fire | alarm holds through the gap | held |
| Fire extinguished | alarm clears | cleared |

The flicker test is the important one: 60 consecutive frames of 0.9-confidence
detections never raise an alarm, because they never land on the same track.
That is the false-positive requirement working as designed.

**Bug found and fixed during video testing.** An uncertain box head emits
sub-pixel boxes pinned to the canvas edge (e.g. x 319.9→320.0 at 320px). These
are invisible to an operator but would enter the tracker and accumulate hits.
`clean_boxes` now clips detections to the frame and drops anything thinner than
`--min-box-size` (default 4px) before the confirmer sees it. Eval is
deliberately left unfiltered so the reported metrics measure the raw model.

### 2026-09-02 — DINOv3 conversion started (incomplete)
Partial scripts written; work stopped part-way.

### 2026-09-01 — D-Fire prepared, YOLOv8 pipeline
Dataset preparation, validation and YOLO training scripts.

---

## 8. Current status and next steps

**Done**
- [x] Dataset prepared and verified
- [x] DINOv3 backbone verified against real weights
- [x] Model, training, evaluation and video scripts written and smoke-tested
- [x] False-alarm measurement and operating-point selection
- [x] This log
- [x] Stage-1 training run (40 epochs, val mAP@0.5 = 0.7312)

**Next, in order**
1. ~~Run the real training on Kaggle~~ — **done 2026-09-07**, val mAP@0.5 = 0.7312 (§9).
2. **Evaluate on the D-Fire test split**, record mAP and the operating threshold.
   This is the number the report quotes; val was only used for model selection.
3. **Inspect the saved false positives** and note the failure modes (welding,
   sunset, steam, headlights) — this directly informs the report.
4. **Collect site videos**, then run `predict_video_dinov3.py` and tune the
   confirmation window against real frame rates.
5. **Stage 2 fine-tune** (`--unfreeze-last-n 2`) only if stage 1 metrics justify it.
6. **Collect real occluded-fire footage** to properly validate R3.

---

## 9. Results

*(Fill in after the Kaggle run. Keep every row, including the bad ones — the
comparison between configurations is the interesting part of the report.)*

| date | config | val mAP@0.5 | test mAP@0.5 | AP50 fire | AP50 smoke | op. thr | FPR at op. | fire recall at op. |
|---|---|---|---|---|---|---|---|---|
| 2026-09-07 | stage 1, frozen trunk, 640px, batch 8, 40 ep | **0.7312** | *pending* | 0.6412 | 0.8212 | 0.90 | 0.0037 | 0.758 |

Stage 1, best epoch 38 of 40 (~11 min/epoch, ~7.4 h on Kaggle P100, 0 skipped
batches so AMP was stable). Full sweep in `best_metrics.json`.

### Reading the first run

| metric | smoke | fire |
|---|---|---|
| AP@0.5 | 0.8212 | 0.6412 |
| AP@0.75 | 0.4285 | 0.2315 |
| AP@0.5:0.95 | 0.4466 | 0.2981 |
| AP75 / AP50 | 0.52 | **0.36** |

**Fire is the weak class, and the problem is localisation, not detection.**
Fire AP50 is 0.64 while AP75 is 0.23 — the model finds the fire and then boxes
it sloppily. Smoke holds up far better at the tighter IoU.

The likely cause is object size. In the training split, fire appears in 3,819
images but contributes 9,659 boxes (2.5 per image), whereas smoke appears in
6,778 images with 7,660 boxes (1.1 per image). Fire in D-Fire is many small
flames; smoke is one big plume. Small objects are exactly what a frozen
stride-16 trunk with an upsampled P3 localises worst.

**The alarm curve is steep in the right place.**

| conf | FPR (1,610 negatives) | fire recall | smoke recall |
|---|---|---|---|
| 0.80 | 1.55% | 0.901 | 0.921 |
| 0.85 | 1.18% | 0.860 | 0.887 |
| 0.90 | **0.37%** | 0.758 | 0.828 |

The 1% budget forces 0.90, which costs ~14 points of fire recall against 0.85.
That trade is worth revisiting once temporal confirmation is in the loop — see
the caveat in §8, step 4.

**The scene head is a genuinely useful second opinion.** At 0.5 it gives fire
recall 0.901 at precision 0.927 (FPR 0.025); at 0.95 it reaches precision 1.000
with recall 0.554 and *zero* false positives on 1,610 negatives. That is a
strong cross-check signal, and it validates the two-head design.

**Training had converged.** mAP@0.5 moved 0.7224 → 0.7297 over epochs 30–40 and
train loss was flat at ~0.31. More epochs at this configuration will not help;
the next gain has to come from resolution, unfreezing, or better data.

---

## 10. Sixty-second explanation

> We detect fire and smoke on a fixed industrial camera. Instead of training a
> detector from scratch, we use DINOv3 — a vision transformer Meta trained
> self-supervised on 1.7 billion images — and freeze it, because its features
> generalise to lighting conditions our 14,000 training images never show. On
> top of the frozen backbone we train two heads: a Faster R-CNN head that draws
> boxes around fire and smoke, and an image-level classifier that also receives
> hand-crafted "glow" statistics, so it can react to a fire that is hidden
> behind machinery and only visible as warm light on the surrounding surfaces.
>
> Because the camera runs 24 hours a day, the number we optimise is not just
> accuracy but the false-alarm rate. We measure it directly on the 9,838
> verified-negative images in the dataset, pick the confidence threshold from
> that curve against a stated budget, and then require six confirmed detections
> of the *same tracked object* within fifteen frames before raising an alarm.
> Flicker, glare and headlights do not survive that; a real fire does.
