import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO fire/smoke detector on the prepared D-Fire dataset.")
    parser.add_argument("--data", default="datasets/processed/fire_smoke_yolo/data.yaml", help="YOLO data.yaml path.")
    parser.add_argument("--model", default="yolov8n.pt", help="Starting model, for example yolov8n.pt or yolo11n.pt.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size. Use -1 for Ultralytics auto-batch.")
    parser.add_argument("--device", default="auto", help="auto, CUDA device index such as 0, cpu, or comma-separated device list.")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--name", default="dfire_yolov8n_base", help="Run name.")
    parser.add_argument("--project", default="runs/fire_smoke", help="Output runs folder.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data.yaml: {data_path}")

    try:
        from ultralytics import YOLO
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install it first with:\n"
            "  python -m pip install ultralytics\n"
            "Then rerun this script."
        ) from exc

    device = args.device
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
        print(f"Auto-selected training device: {device}")
        if device == "cpu":
            print("Warning: CUDA is not available. CPU training will be very slow for this dataset.")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        plots=True,
        save=True,
        exist_ok=False,
    )

    model.val(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=args.project,
        name=f"{args.name}_test_eval",
        plots=True,
    )


if __name__ == "__main__":
    main()
