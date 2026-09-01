from pathlib import Path


DATA_ROOT = Path("datasets/processed/fire_smoke_yolo")
WEIGHTS = "yolov8n.pt"


def first_image() -> Path:
    image_dir = DATA_ROOT / "images" / "val"
    for path in sorted(image_dir.glob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return path
    raise FileNotFoundError(f"No validation image found in {image_dir}")


def main() -> None:
    from ultralytics import YOLO
    import torch

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
    else:
        print("cuda_device=none")

    model = YOLO(WEIGHTS)
    image_path = first_image()
    results = model.predict(source=str(image_path), imgsz=640, device="cpu", verbose=False)
    print(f"weights_ready={WEIGHTS}")
    print(f"sample_image={image_path}")
    print(f"sample_prediction_boxes={len(results[0].boxes)}")


if __name__ == "__main__":
    main()
