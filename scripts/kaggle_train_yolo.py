import argparse
import shutil
from pathlib import Path


def find_data_yaml(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Given data.yaml does not exist: {path}")
        return path

    candidates = sorted(Path("/kaggle/input").rglob("data.yaml"))
    if not candidates:
        raise FileNotFoundError(
            "No data.yaml found under /kaggle/input. Upload the processed YOLO dataset as a Kaggle input."
        )

    preferred = [path for path in candidates if "fire_smoke_yolo" in str(path).lower()]
    return preferred[0] if preferred else candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle training entry point for M1 fire/smoke YOLO model.")
    parser.add_argument("--data", default=None, help="Optional explicit path to Kaggle data.yaml.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model checkpoint, e.g. yolov8n.pt or yolo11n.pt.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default=0, help="Use 0 on Kaggle GPU, or cpu if no GPU is enabled.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="/kaggle/working/runs/fire_smoke")
    parser.add_argument("--name", default="dfire_yolov8n_base")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--zip-name", default="/kaggle/working/dfire_yolov8n_base_results")
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO
    import torch

    args = parse_args()
    data_yaml = find_data_yaml(args.data)

    print(f"Using data.yaml: {data_yaml}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        plots=True,
        save=True,
    )

    model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=f"{args.name}_test_eval",
        plots=True,
    )

    run_dir = Path(args.project) / args.name
    if run_dir.exists():
        archive = shutil.make_archive(args.zip_name, "zip", run_dir)
        print(f"Zipped training run: {archive}")


if __name__ == "__main__":
    main()
