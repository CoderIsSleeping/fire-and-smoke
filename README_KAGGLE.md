# Kaggle Workflow for M1 Fire/Smoke Detection

Use this workflow when training on Kaggle and keeping GitHub as code-only storage.

## What Goes to GitHub

Push the code repository only:

```text
scripts/
requirements.txt
README_TRAINING.md
README_KAGGLE.md
.gitignore
```

Do not push:

```text
.venv/
datasets/
runs/
*.pt
*.onnx
*.engine
```

These are already ignored by `.gitignore`.

## What Goes to Kaggle Input

Upload the processed YOLO dataset as a Kaggle dataset/input:

```text
datasets/processed/fire_smoke_yolo/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Also upload your external verification video as a separate Kaggle input. Do not include that video in training, validation, or test folders.

## Kaggle Notebook Setup

In Kaggle:

1. Create a new notebook.
2. Enable GPU: Notebook settings -> Accelerator -> GPU.
3. Add your processed dataset as input.
4. Add your external verification video as another input.
5. Clone your GitHub repository.

Example notebook cells:

```bash
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
```

Install dependencies:

```bash
!pip install -q ultralytics opencv-python pyyaml numpy matplotlib pandas tqdm
```

Confirm Kaggle can see a GPU:

```bash
!python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

## Train on Kaggle

The script auto-finds `data.yaml` under `/kaggle/input`.

```bash
!python scripts/kaggle_train_yolo.py --epochs 80 --imgsz 960 --batch -1 --device 0
```

If you want a faster first trial:

```bash
!python scripts/kaggle_train_yolo.py --epochs 10 --imgsz 640 --batch -1 --device 0 --name smoke_test_dfire_yolov8n
```

Training outputs are saved in:

```text
/kaggle/working/runs/fire_smoke/
```

The best model should be:

```text
/kaggle/working/runs/fire_smoke/dfire_yolov8n_base/weights/best.pt
```

The training script also zips the run folder:

```text
/kaggle/working/dfire_yolov8n_base_results.zip
```

## Verify on External Video

After training, run your separate video through the trained model.

Example:

```bash
!python scripts/predict_video.py \
  --weights /kaggle/working/runs/fire_smoke/dfire_yolov8n_base/weights/best.pt \
  --source /kaggle/input/YOUR_VIDEO_DATASET/YOUR_VIDEO.mp4 \
  --imgsz 960 \
  --conf 0.25 \
  --device 0 \
  --project /kaggle/working/runs/fire_smoke_video
```

The annotated verification output will be saved under:

```text
/kaggle/working/runs/fire_smoke_video/
```

## Important Reporting Note

Your external video is not part of training, validation, or test. In the report, describe it as:

```text
External unseen verification video used for qualitative demonstration.
```

Use the D-Fire test split for actual benchmark metrics, and use the separate video for demonstration and sanity checking.
