"""Sanity-check a prepared detection dataset before spending GPU time on it.

Verifies that every image has a label file, that coordinates are inside 0..1,
that class ids are only 0/1, and reports the positive/negative balance. The
negative count matters as much as the positive one here: those empty label
files are what the false-alarm metric is measured on.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fire_smoke.dataset import IMAGE_EXTS, find_data_yaml, load_data_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a prepared fire/smoke detection dataset.")
    p.add_argument("--data", default=None, help="Path to data.yaml (auto-detected if omitted).")
    p.add_argument("--max-errors", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = find_data_yaml(args.data)
    root, cfg = load_data_config(data_yaml)
    print(f"data.yaml : {data_yaml}")
    print(f"root      : {root}\n")

    errors: list[str] = []
    totals = Counter()

    for split in ("train", "val", "test"):
        rel = cfg.get(split, f"images/{split}")
        image_dir = root / rel
        label_dir = root / "labels" / split
        if not image_dir.exists():
            print(f"{split:>6}: MISSING image dir {image_dir}")
            errors.append(f"missing image dir: {image_dir}")
            continue

        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        counts = Counter()
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                errors.append(f"missing label: {label_path}")
                continue
            counts["labels"] += 1

            text = label_path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                counts["negatives"] += 1
                continue

            has = set()
            for line_no, line in enumerate(text.splitlines(), start=1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label_path}:{line_no}: expected 5 columns, got {len(parts)}")
                    continue
                try:
                    class_id = int(float(parts[0]))
                    coords = [float(v) for v in parts[1:]]
                except ValueError:
                    errors.append(f"{label_path}:{line_no}: non-numeric value")
                    continue
                if class_id not in (0, 1):
                    errors.append(f"{label_path}:{line_no}: invalid class id {class_id}")
                    continue
                if any(v < 0.0 or v > 1.0 for v in coords):
                    errors.append(f"{label_path}:{line_no}: coordinate outside 0..1")
                    continue
                if coords[2] <= 0 or coords[3] <= 0:
                    errors.append(f"{label_path}:{line_no}: zero-area box")
                    continue
                counts["smoke_boxes" if class_id == 0 else "fire_boxes"] += 1
                has.add(class_id)

            counts["images_with_smoke"] += int(0 in has)
            counts["images_with_fire"] += int(1 in has)

        counts["images"] = len(images)
        totals.update(counts)
        positives = counts["images"] - counts["negatives"]
        print(
            f"{split:>6}: images={counts['images']:<6} labels={counts['labels']:<6} "
            f"positives={positives:<6} negatives={counts['negatives']:<6} "
            f"smoke_boxes={counts['smoke_boxes']:<6} fire_boxes={counts['fire_boxes']:<6}"
        )

    print(
        f"\n total: images={totals['images']} negatives={totals['negatives']} "
        f"({100.0 * totals['negatives'] / max(totals['images'], 1):.1f}% of the data) "
        f"smoke_boxes={totals['smoke_boxes']} fire_boxes={totals['fire_boxes']}"
    )
    print(f" errors: {len(errors)}")

    if errors:
        for error in errors[: args.max_errors]:
            print(f"  {error}")
        if len(errors) > args.max_errors:
            print(f"  ... and {len(errors) - args.max_errors} more")
        raise SystemExit(1)

    print("\ndataset looks consistent.")


if __name__ == "__main__":
    main()
