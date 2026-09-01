import importlib.util
import json
import platform
import sys
from pathlib import Path


DATA_YAML = Path("datasets/processed/fire_smoke_yolo/data.yaml")


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_dataset() -> dict:
    result = {
        "data_yaml_exists": DATA_YAML.exists(),
        "splits": {},
        "errors": [],
    }

    root = DATA_YAML.parent
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = list(image_dir.glob("*")) if image_dir.exists() else []
        labels = list(label_dir.glob("*.txt")) if label_dir.exists() else []
        result["splits"][split] = {
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
            "images": len(images),
            "labels": len(labels),
        }
        if len(images) != len(labels):
            result["errors"].append(f"{split}: image/label count mismatch")
    return result


def check_torch() -> dict:
    if not module_available("torch"):
        return {"installed": False}

    import torch

    info = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


def main() -> None:
    package_names = ["ultralytics", "torch", "torchvision", "cv2", "yaml", "numpy"]
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: module_available(name) for name in package_names},
        "dataset": check_dataset(),
        "torch": check_torch(),
    }

    print(json.dumps(report, indent=2))

    missing = [name for name, ok in report["packages"].items() if not ok]
    dataset_errors = report["dataset"]["errors"]

    if missing or dataset_errors:
        if missing:
            print("Missing packages:", ", ".join(missing))
        if dataset_errors:
            print("Dataset errors:", "; ".join(dataset_errors))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
