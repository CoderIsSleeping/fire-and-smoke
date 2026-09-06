# Kaggle Workflow — DINOv3 Fire & Smoke Detector

GitHub holds the code, Kaggle holds the data and the GPU.

---

## 1. What goes where

**GitHub (code only):**

```text
scripts/            requirements_dinov3.txt
PROJECT_LOG.md      README_TRAINING.md      README_KAGGLE.md      .gitignore
```

**Kaggle inputs (never in git):**

```text
<your-dataset>/          the prepared detection dataset
  data.yaml
  images/{train,val,test}/
  labels/{train,val,test}/

<your-video-dataset>/    verification footage, kept out of every split
```

`.gitignore` already excludes `datasets/`, `runs/`, `.venv/` and `*.pt`.

> The `path:` line inside `data.yaml` is a Windows path from the machine that
> prepared the dataset. Leave it — the loader ignores it when it does not exist
> and falls back to the yaml's own folder.

## 2. Notebook setup

Notebook settings → Accelerator → **GPU (T4 x2 or P100)**, and add both inputs.

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
!pip install -q "timm>=1.0.20" huggingface_hub
```

Check what Kaggle actually mounted, and that the environment is sound:

```python
!python scripts/list_kaggle_inputs.py
!python scripts/check_training_readiness.py --check-backbone
```

`--check-backbone` downloads the DINOv3 trunk (~86 MB). If the notebook has no
internet, this is where you find out — see *Offline weights* at the bottom.

## 3. Train

The scripts find `data.yaml` under `/kaggle/input` on their own.

```python
!python scripts/train_dinov3_detector.py \
    --epochs 40 --imgsz 640 --batch 8 --workers 2 \
    --name dinov3_stage1 --zip
```

Fast sanity run first (about two minutes) — always worth it before spending a
GPU quota:

```python
!python scripts/train_dinov3_detector.py \
    --epochs 1 --imgsz 640 --batch 8 \
    --max-train-images 200 --max-val-images 200 --name smoketest
```

Outputs land in `/kaggle/working/fire_smoke/<name>/`, and `--zip` also writes
`/kaggle/working/fire_smoke/<name>_results.zip` for downloading.

**If the session times out**, save the run as a Kaggle dataset and resume:

```python
!python scripts/train_dinov3_detector.py --epochs 40 --imgsz 640 --batch 8 \
    --name dinov3_stage1 --resume /kaggle/input/PREVIOUS_RUN/weights/last.pt
```

### Memory guide

| GPU | imgsz | batch |
|---|---|---|
| T4 (16 GB) | 640 | 8 |
| T4 (16 GB) | 704 | 6 |
| P100 (16 GB) | 640 | 8 |

If you hit OOM, halve `--batch` and set `--accum 2` to keep the effective batch.

## 4. Evaluate on the test split

```python
!python scripts/eval_dinov3_detector.py \
    --weights /kaggle/working/fire_smoke/dinov3_stage1/weights/best.pt \
    --split test --target-fpr 0.01
```

Record two things in `PROJECT_LOG.md` §9:
- **mAP@0.5** and per-class AP — the benchmark numbers for the report,
- the **recommended confidence threshold** and the false-alarm rate at it — the
  deployment number.

Then open `eval_test/false_positives/` and look at what the model actually got
wrong. Those images are the most useful output of the whole run: they tell you
what to collect for the fine-tuning stage.

## 5. Verify on the external video

```python
!python scripts/predict_video_dinov3.py \
    --weights /kaggle/working/fire_smoke/dinov3_stage1/weights/best.pt \
    --source /kaggle/input/YOUR_VIDEO_DATASET/YOUR_VIDEO.mp4 \
    --output /kaggle/working/verification.mp4 \
    --conf 0.5 --window 15 --enter-hits 6
```

Use the threshold that step 4 recommended. You get `verification.mp4` plus
`verification.events.csv`, one row per alarm state change.

If the video is long, `--stride 3` runs the model on every third frame while
still writing every frame to the output. Remember the confirmation window is
counted in *processed* frames, so at `--stride 3` a `--window 15` covers 45
source frames.

## 6. Reporting note

The external video is not part of train, val or test. Describe it as:

```text
External unseen verification footage, used for qualitative demonstration only.
```

Benchmark numbers come from the **D-Fire test split**. Two numbers belong
together in the report, and they should be quoted as a pair:

```text
mAP@0.5 = <x> at confidence <t>, with a false-alarm rate of <f> on the
2,005 verified-negative test images.
```

An mAP quoted without its false-alarm rate says nothing about whether the
system can be left switched on.

---

## Offline weights

If the notebook has no internet, run this once somewhere that does:

```bash
python -c "import timm; timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True)"
```

Upload `~/.cache/huggingface/hub/models--timm--vit_small_patch16_dinov3.lvd1689m/snapshots/*/model.safetensors`
as a Kaggle dataset, then add to the training command:

```bash
--backbone-weights /kaggle/input/dinov3-vits16/model.safetensors
```
