import argparse
import json
import os
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["smoke", "fire"]


def read_split_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "existing"

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"

    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def validate_label(label_path: Path) -> tuple[int, list[str]]:
    errors = []
    box_count = 0
    text = label_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return 0, errors

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}:{line_no}: expected 5 columns, got {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"{label_path}:{line_no}: non-numeric annotation value")
            continue
        if class_id not in (0, 1):
            errors.append(f"{label_path}:{line_no}: unexpected class id {class_id}")
        if any(value < 0 or value > 1 for value in coords):
            errors.append(f"{label_path}:{line_no}: coordinate outside normalized range 0..1")
        box_count += 1

    return box_count, errors


def sanitize_label(src: Path, dst: Path) -> tuple[int, int, int, list[str]]:
    """Copy a label file, clamping coordinates and dropping degenerate boxes.

    Returns (kept_boxes, coordinate_fixes, dropped_boxes, errors).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    output_lines = []
    fixes = 0
    dropped = 0
    errors = []
    text = src.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        dst.write_text("", encoding="utf-8")
        return 0, fixes, dropped, errors

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{src}:{line_no}: expected 5 columns, got {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"{src}:{line_no}: non-numeric annotation value")
            continue

        if class_id not in (0, 1):
            errors.append(f"{src}:{line_no}: unexpected class id {class_id}")
            continue

        clean_coords = []
        for value in coords:
            clean_value = min(1.0, max(0.0, value))
            if clean_value != value:
                fixes += 1
            clean_coords.append(clean_value)

        # D-Fire contains a handful of zero-area annotations. torchvision
        # refuses a whole batch if one degenerate box reaches it, so drop them
        # here rather than at training time.
        if clean_coords[2] <= 0.0 or clean_coords[3] <= 0.0:
            dropped += 1
            continue

        output_lines.append(
            f"{class_id} "
            + " ".join(f"{value:.10f}".rstrip("0").rstrip(".") for value in clean_coords)
        )

    dst.write_text(("\n".join(output_lines) + ("\n" if output_lines else "")), encoding="utf-8")
    return len(output_lines), fixes, dropped, errors


def prepare_split(raw_root: Path, out_root: Path, split_name: str, filenames: list[str], source_split: str, mode: str) -> dict:
    src_images = raw_root / source_split / "images"
    src_labels = raw_root / source_split / "labels"
    dst_images = out_root / "images" / split_name
    dst_labels = out_root / "labels" / split_name

    stats = {
        "images": 0,
        "labels": 0,
        "empty_labels": 0,
        "boxes": 0,
        "class_counts": {"0_smoke": 0, "1_fire": 0},
        "missing": [],
        "label_errors": [],
        "coordinate_fixes": 0,
        "dropped_zero_area_boxes": 0,
        "link_mode": {"linked": 0, "copied": 0, "existing": 0},
    }

    for image_name in filenames:
        image_src = src_images / image_name
        label_src = src_labels / f"{Path(image_name).stem}.txt"

        if not image_src.exists():
            stats["missing"].append(str(image_src))
            continue
        if image_src.suffix.lower() not in IMAGE_EXTS:
            stats["label_errors"].append(f"{image_src}: unsupported image extension")
            continue
        if not label_src.exists():
            stats["missing"].append(str(label_src))
            continue

        result = link_or_copy(image_src, dst_images / image_name, mode)
        stats["link_mode"][result] += 1

        text = label_src.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            stats["empty_labels"] += 1
        else:
            for line in text.splitlines():
                parts = line.split()
                if parts:
                    if parts[0] == "0":
                        stats["class_counts"]["0_smoke"] += 1
                    elif parts[0] == "1":
                        stats["class_counts"]["1_fire"] += 1

        box_count, fixes, dropped, errors = sanitize_label(label_src, dst_labels / label_src.name)
        stats["boxes"] += box_count
        stats["coordinate_fixes"] += fixes
        stats["dropped_zero_area_boxes"] += dropped
        stats["label_errors"].extend(errors[:20])
        stats["images"] += 1
        stats["labels"] += 1

    return stats


def write_yaml(out_root: Path) -> None:
    yaml_text = "\n".join(
        [
            f"path: {out_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
            "  0: smoke",
            "  1: fire",
            "",
        ]
    )
    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare D-Fire as a clean normalized fire/smoke detection dataset.")
    parser.add_argument("--raw-root", default="datasets/raw/D-Fire", help="Path to raw D-Fire folder.")
    parser.add_argument("--out-root", default="datasets/processed/fire_smoke_detection", help="Output detection dataset folder.")
    parser.add_argument("--mode", choices=["link", "copy"], default="link", help="Use hardlinks when possible, or copy files.")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    split_root = raw_root / "Data splitting"
    fold_root = split_root / "5-fold cross validation"

    train_files = read_split_file(fold_root / "dfire_train1.txt")
    val_files = read_split_file(fold_root / "dfire_valid1.txt")
    test_files = read_split_file(split_root / "dfire_test.txt")

    out_root.mkdir(parents=True, exist_ok=True)

    report = {
        "source": str(raw_root.resolve()),
        "output": str(out_root.resolve()),
        "classes": {"0": "smoke", "1": "fire"},
        "splits": {
            "train": prepare_split(raw_root, out_root, "train", train_files, "train", args.mode),
            "val": prepare_split(raw_root, out_root, "val", val_files, "train", args.mode),
            "test": prepare_split(raw_root, out_root, "test", test_files, "test", args.mode),
        },
    }

    write_yaml(out_root)
    (out_root / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Prepared dataset: {out_root}")
    print(f"data.yaml: {out_root / 'data.yaml'}")
    for split, stats in report["splits"].items():
        print(
            f"{split}: images={stats['images']} labels={stats['labels']} "
            f"empty={stats['empty_labels']} boxes={stats['boxes']} "
            f"smoke={stats['class_counts']['0_smoke']} fire={stats['class_counts']['1_fire']} "
            f"coordinate_fixes={stats['coordinate_fixes']} "
            f"dropped_zero_area={stats['dropped_zero_area_boxes']}"
        )
        if stats["missing"]:
            print(f"  missing files: {len(stats['missing'])}")
        if stats["label_errors"]:
            print(f"  label errors: {len(stats['label_errors'])}")


if __name__ == "__main__":
    main()
