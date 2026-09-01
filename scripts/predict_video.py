import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained fire/smoke YOLO model on an external verification video.")
    parser.add_argument("--weights", required=True, help="Path to trained best.pt.")
    parser.add_argument("--source", required=True, help="Path to verification video.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="auto, 0, cpu, etc.")
    parser.add_argument("--project", default="runs/fire_smoke_video")
    parser.add_argument("--name", default="external_video_verification")
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO
    import torch

    args = parse_args()
    weights = Path(args.weights)
    source = Path(args.source)
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights file: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Missing video file: {source}")

    device = args.device
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Using weights: {weights}")
    print(f"Using source video: {source}")
    print(f"Using device: {device}")

    model = YOLO(str(weights))
    model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=True,
        save_conf=True,
        vid_stride=1,
    )


if __name__ == "__main__":
    main()
