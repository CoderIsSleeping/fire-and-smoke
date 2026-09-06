# Training — DINOv3 Fire & Smoke Detector

Model: frozen **DINOv3 ViT-S/16** trunk → trainable feature pyramid →
**Faster R-CNN** head (`smoke`, `fire`) **+ an image-level scene classifier**.

For the *why* behind every choice, and the running change log, see
[`PROJECT_LOG.md`](PROJECT_LOG.md).

---

## 1. Install

```bash
pip install -r requirements_dinov3.txt
```

On Kaggle, only `timm` and `huggingface_hub` are usually missing.

## 2. Prepare the dataset (only if starting from raw D-Fire)

```bash
python scripts/prepare_dfire_detection.py --raw-root datasets/raw/D-Fire --out-root datasets/processed/fire_smoke_detection
```

The already-prepared `datasets/processed/fire_smoke_yolo` works as-is — the
label format is identical and every script auto-discovers `data.yaml`.

## 3. Pre-flight

```bash
python scripts/check_training_readiness.py --check-backbone
```

```bash
python scripts/validate_detection_dataset.py
```

`--check-backbone` actually downloads the DINOv3 trunk, which is the check most
worth doing before a Kaggle session: if it fails, see *Offline weights* below.

## 4. Train

```bash
python scripts/train_dinov3_detector.py --epochs 40 --imgsz 640 --batch 8
```

`--imgsz` must be a multiple of 64 (the coarsest pyramid level is stride 64).
Good values: 576, 640, 704.

### Options worth knowing

| Flag | Default | What it does |
|---|---|---|
| `--batch` | 8 | Lower to 4 if the GPU runs out of memory; raise with `--accum` instead of the batch size if you want a larger effective batch |
| `--accum` | 1 | Gradient accumulation steps |
| `--unfreeze-last-n` | 0 | Stage 2: release the last N trunk blocks at `lr × 0.05` |
| `--night-prob` | 0.35 | Low-light simulation rate |
| `--gray-prob` | 0.10 | IR / night-mode (colour removed) rate |
| `--occlude-prob` | 0.25 | Hide a flame behind a synthetic obstacle, keeping the label |
| `--no-augment` | off | Ablation baseline — useful for the report |
| `--target-fpr` | 0.01 | False-alarm budget used to report an operating threshold each epoch |
| `--patience` | 8 | Early stop after N epochs with no mAP@0.5 gain |
| `--resume` | — | Continue an interrupted run from `last.pt` (restores optimizer, epoch, history) |
| `--init-from` | — | Start a *fresh* run from another checkpoint's weights (stage-2 fine-tuning) |
| `--max-train-images` | 0 (all) | Subsample for a fast smoke test |

### Two-stage recipe

```bash
# stage 1 - frozen trunk, head learns the task
python scripts/train_dinov3_detector.py --epochs 40 --imgsz 640 --batch 8 --name stage1

# stage 2 - release the last two blocks, only if stage 1 justifies it
python scripts/train_dinov3_detector.py --epochs 15 --imgsz 640 --batch 4 \
    --unfreeze-last-n 2 --lr 5e-5 --name stage2 \
    --init-from runs/fire_smoke/stage1/weights/best.pt
```

`--init-from` takes the weights and starts a clean run. `--resume` is only for
picking up an *interrupted* run from its `last.pt`.

### Outputs

```text
runs/fire_smoke/<name>/
  weights/best.pt          selected by validation mAP@0.5; weights only (~160 MB)
  weights/last.pt          resume point: weights + optimizer + scaler + history (~300 MB)
  results.csv              per-epoch metrics
  results.png              losses / mAP / alarm behaviour / scene head
  best_metrics.json        full metrics at the best epoch
  training_summary.md      human-readable summary
  dataset_summary.json     what was actually loaded
```

## 5. Evaluate and pick the operating threshold

```bash
python scripts/eval_dinov3_detector.py --weights runs/fire_smoke/stage1/weights/best.pt --split test
```

This prints mAP, then the table that matters for deployment:

```text
  conf  FP imgs      FPR  rec fire  rec smoke  rec any
```

`FPR` is the fraction of **verified-negative** images (empty label file) that
produced at least one box at that confidence. The script recommends the lowest
threshold that stays inside `--target-fpr`, and writes the highest-scoring false
positives to `eval_test/false_positives/` so the failure modes can be looked at.

## 6. Video

```bash
python scripts/predict_video_dinov3.py \
    --weights runs/fire_smoke/stage1/weights/best.pt \
    --source my_site_video.mp4 \
    --output out/site_annotated.mp4 \
    --conf 0.5 --window 15 --enter-hits 6
```

Use the threshold that step 5 recommended for `--conf`.

An alarm needs `--enter-hits` detections of the *same IoU-linked track* inside
`--window` frames. Alongside the annotated video you get a CSV event log with
one row per state change (`ALARM_ON`, `ALARM_OFF`, `OCCLUDED_FIRE_CUE`).

### Ignoring permanently bright regions

```json
{
  "normalized": true,
  "ignore_polygons": [
    [[0.62, 0.40], [0.88, 0.40], [0.88, 0.72], [0.62, 0.72]]
  ]
}
```

```bash
python scripts/predict_video_dinov3.py ... --roi-mask furnace_mask.json
```

Any detection whose centre lands inside a polygon is discarded — the practical
way to live with a furnace mouth or a welding bay in frame.

---

## Offline weights (Kaggle with the internet switch off)

Download the trunk once on a machine that has internet:

```bash
python -c "import timm; timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True)"
```

The file lands in `~/.cache/huggingface/hub/models--timm--vit_small_patch16_dinov3.lvd1689m/`.
Attach it as a Kaggle dataset and pass:

```bash
--backbone-weights /kaggle/input/dinov3-vits16/model.safetensors
```

---

## Notes

- The trunk is **frozen** by default. That is deliberate — see `PROJECT_LOG.md` §3.
- Model selection uses **validation mAP@0.5**, not loss.
- Mixed precision is on automatically on CUDA; non-finite batches are skipped
  rather than allowed to poison the weights.
- CPU training is only useful for correctness checks. Use
  `--imgsz 320 --batch 2 --max-train-images 24 --workers 0 --device cpu`.
