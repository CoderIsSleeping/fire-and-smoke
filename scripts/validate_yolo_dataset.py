from pathlib import Path


DATASET_ROOT = Path("datasets/processed/fire_smoke_yolo")


def main() -> None:
    labels_root = DATASET_ROOT / "labels"
    image_root = DATASET_ROOT / "images"
    errors = []
    class_counts = {0: 0, 1: 0}
    empty_labels = 0
    label_files = 0

    for split in ("train", "val", "test"):
        image_count = len(list((image_root / split).glob("*")))
        split_label_files = sorted((labels_root / split).glob("*.txt"))
        print(f"{split}: images={image_count}, labels={len(split_label_files)}")

        for label_path in split_label_files:
            label_files += 1
            text = label_path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                empty_labels += 1
                continue

            for line_no, line in enumerate(text.splitlines(), start=1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label_path}:{line_no}: expected 5 columns")
                    continue
                try:
                    class_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                except ValueError:
                    errors.append(f"{label_path}:{line_no}: non-numeric value")
                    continue

                if class_id not in (0, 1):
                    errors.append(f"{label_path}:{line_no}: invalid class id {class_id}")
                    continue
                if any(value < 0.0 or value > 1.0 for value in coords):
                    errors.append(f"{label_path}:{line_no}: coordinate outside 0..1")
                    continue
                class_counts[class_id] += 1

    print(f"label_files={label_files}")
    print(f"empty_negative_labels={empty_labels}")
    print(f"smoke_boxes={class_counts[0]}")
    print(f"fire_boxes={class_counts[1]}")
    print(f"errors={len(errors)}")

    if errors:
        print("First errors:")
        for error in errors[:20]:
            print(error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
