# M1 Fire/Smoke YOLO Training Setup

This project is prepared for the D-Fire base-model stage.

## Current Dataset

Prepared YOLO dataset:

```text
datasets/processed/fire_smoke_yolo
```

Class mapping:

```text
0: smoke
1: fire
```

Dataset status:

```text
train: 13776 images / labels
val:   3445 images / labels
test:  4306 images / labels

empty negative labels: 9838
smoke boxes: 11865
fire boxes: 14692
label errors: 0
```

## Environment

A local virtual environment has been created:

```powershell
.venv\Scripts\Activate.ps1
```

Installed packages include Ultralytics, PyTorch, TorchVision, OpenCV, NumPy, PyYAML, Matplotlib, Pandas, and tqdm.

Current PyTorch status:

```text
torch: 2.13.0+cpu
CUDA available: false
```

This means the machine is ready for correctness checks, but full training on CPU will be very slow. For practical training, use a CUDA-capable NVIDIA GPU machine or cloud GPU runtime.

## Pre-Training Checks

Run this before training:

```powershell
.venv\Scripts\python.exe scripts\check_training_readiness.py
.venv\Scripts\python.exe scripts\validate_yolo_dataset.py
.venv\Scripts\python.exe scripts\preflight_yolo.py
```

## Training Command

Start the base YOLOv8n training with:

```powershell
.venv\Scripts\python.exe scripts\train_yolo_fire_smoke.py
```

Useful options:

```powershell
.venv\Scripts\python.exe scripts\train_yolo_fire_smoke.py --epochs 80 --imgsz 960 --batch -1 --device auto
```

If CUDA is available, `--device auto` uses GPU `0`. If CUDA is not available, it falls back to CPU and prints a warning.

## Expected Outputs

Training results will be written under:

```text
runs/fire_smoke/
```

The most important files after training will be:

```text
runs/fire_smoke/dfire_yolov8n_base/weights/best.pt
runs/fire_smoke/dfire_yolov8n_base/results.png
runs/fire_smoke/dfire_yolov8n_base/confusion_matrix.png
runs/fire_smoke/dfire_yolov8n_base_test_eval/
```

## Next Project Step

After the D-Fire base model is trained, collect industry-specific CCTV footage and fine-tune the model on:

- real camera height and angle
- steam/soot-blowing scenes
- dust and ash clouds
- welding/sparks/glare
- night and low-light views
- normal non-fire negatives
- any safe controlled smoke/fire drill data

The fine-tuned model should then be combined with temporal confirmation to reduce false alarms before deployment.
