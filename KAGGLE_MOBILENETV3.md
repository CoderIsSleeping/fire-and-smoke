# Kaggle: MobileNetV3-Large baseline (branch `backbone-mobilenetv3`)

This branch is the DINOv3 project with **one change: the backbone is
MobileNetV3-Large** (`mobilenetv3_large_100.ra_in1k`). Faster R-CNN head, scene head, glow prior,
augmentation, optimizer and schedule are identical to `main`, so the results
compare directly against Model 1 (DINOv3 ViT-S + Faster R-CNN). See
`PROJECT_LOG.md` section 6f.

Same dataset input as before. Accelerator: **GPU T4 x2 or P100**, internet
**on** (the ImageNet weights come from the Hugging Face Hub, a few MB).

## 1. Setup

```python
!git clone -b backbone-mobilenetv3 https://github.com/CoderIsSleeping/fire-and-smoke.git
%cd fire-and-smoke
!pip install -q "timm>=1.0.20" huggingface_hub
!python scripts/check_training_readiness.py --check-backbone
```

The readiness check should show `"model": "mobilenetv3_large_100.ra_in1k"` and `"ok": true`.

## 2. Stage 1: frozen backbone (same as Model 1 stage 1)

```python
!python scripts/train_dinov3_detector.py     --epochs 40 --imgsz 640 --batch 8 --workers 2     --name mobilenetv3_stage1 --zip
```

Use **Save Version -> Save & Run All (Commit)** so it keeps running after you
close the tab. Output: `/kaggle/working/fire_smoke/mobilenetv3_stage1/weights/best.pt`.

## 3. Stage 2: last 2 stages unfrozen (same as Model 1 stage 2)

Unfreezes `blocks.5` + `blocks.6` (2.2M trunk parameters). Add the stage-1 output as a notebook input, then:

```python
!python scripts/train_dinov3_detector.py     --epochs 15 --imgsz 640 --batch 8 --workers 2     --unfreeze-last-n 2 --lr 5e-5     --init-from /kaggle/input/<stage1-output>/fire_smoke/mobilenetv3_stage1/weights/best.pt     --name mobilenetv3_stage2 --zip
```

If a session times out, rerun with `--resume .../weights/last.pt` instead of
`--init-from`.

## 4. Test-split numbers for the comparison

```python
!python scripts/eval_dinov3_detector.py     --weights /kaggle/working/fire_smoke/mobilenetv3_stage2/weights/best.pt     --split test --target-fpr 0.01
```

Bring back `best.pt`, `history.csv` and `eval_test/eval_report.md` for each
stage. Those give mAP@0.5, per-class AP, recall at the 1% false-alarm
operating point, and the loss curves -- the same numbers Model 1 and Model 2
already have.
