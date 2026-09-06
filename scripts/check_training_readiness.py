"""Pre-flight check: environment, GPU, dataset and DINOv3 weight availability.

Run this first on any new machine or Kaggle session. It fails loudly for the
things that actually stop a training run: a missing package, no data.yaml, or
DINOv3 weights that cannot be fetched because the notebook has no internet.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REQUIRED = ["torch", "torchvision", "timm", "cv2", "yaml", "numpy", "PIL", "matplotlib", "tqdm"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check that everything needed for training is in place.")
    p.add_argument("--data", default=None, help="Path to data.yaml (auto-detected if omitted).")
    p.add_argument("--backbone", default="vit_small_patch16_dinov3.lvd1689m")
    p.add_argument("--check-backbone", action="store_true", help="Actually try to build the pretrained trunk.")
    return p.parse_args()


def available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_torch() -> dict:
    if not available("torch"):
        return {"installed": False}
    import torch

    info = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = ".".join(str(v) for v in torch.cuda.get_device_capability(0))
        info["total_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    return info


def check_dataset(data_arg) -> dict:
    from fire_smoke.dataset import IMAGE_EXTS, find_data_yaml, load_data_config

    try:
        data_yaml = find_data_yaml(data_arg)
    except FileNotFoundError as exc:
        return {"found": False, "error": str(exc)}

    root, cfg = load_data_config(data_yaml)
    result = {"found": True, "data_yaml": str(data_yaml), "root": str(root), "splits": {}, "errors": []}
    for split in ("train", "val", "test"):
        image_dir = root / cfg.get(split, f"images/{split}")
        label_dir = root / "labels" / split
        images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS] if image_dir.exists() else []
        labels = list(label_dir.glob("*.txt")) if label_dir.exists() else []
        result["splits"][split] = {"images": len(images), "labels": len(labels)}
        if not images:
            result["errors"].append(f"{split}: no images at {image_dir}")
        elif len(images) != len(labels):
            result["errors"].append(f"{split}: {len(images)} images but {len(labels)} labels")
    return result


def check_backbone(name: str) -> dict:
    if not available("timm"):
        return {"ok": False, "error": "timm not installed"}
    import timm

    known = name in timm.list_pretrained(f"*{name.split('.')[0]}*")
    info = {"model": name, "known_to_timm": known}
    try:
        model = timm.create_model(name, pretrained=True, num_classes=0)
        info["ok"] = True
        info["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
        info["embed_dim"] = model.embed_dim
        info["depth"] = len(model.blocks)
    except Exception as exc:  # noqa: BLE001 - we want the message, whatever it is
        info["ok"] = False
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["hint"] = (
            "If this session has no internet, download the checkpoint elsewhere, "
            "attach it as a Kaggle dataset, and pass --backbone-weights to the trainer."
        )
    return info


def main() -> None:
    args = parse_args()
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: available(name) for name in REQUIRED},
        "torch": check_torch(),
        "dataset": check_dataset(args.data),
    }
    if args.check_backbone:
        report["backbone"] = check_backbone(args.backbone)

    print(json.dumps(report, indent=2))

    problems = []
    missing = [n for n, ok in report["packages"].items() if not ok]
    if missing:
        problems.append("missing packages: " + ", ".join(missing))
    if not report["dataset"].get("found"):
        problems.append("dataset: " + report["dataset"].get("error", "not found"))
    problems.extend(report["dataset"].get("errors", []))
    if args.check_backbone and not report["backbone"].get("ok"):
        problems.append("backbone: " + report["backbone"].get("error", "unavailable"))

    if not report["torch"].get("cuda_available"):
        print("\nnote: no CUDA device. Correctness checks will run, but training will be far too slow.")

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("\nready to train.")


if __name__ == "__main__":
    main()
