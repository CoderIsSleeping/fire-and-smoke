"""Split a Roboflow YOLO export into train / test *by source video*.

Frames from one video are near-copies of each other. A random frame split puts
copies of the test frames into training and the test score goes up for the
wrong reason, so every frame of a video goes to the same side.

The video a frame belongs to is read from its file name. Roboflow renames
`clip07_frame_0123.jpg` to `clip07_frame_0123_jpg.rf.<hash>.jpg`; the group key
strips the Roboflow suffix and then the trailing frame number, giving `clip07`.
Check the printed groups; if the naming is different, pass --group-regex with
one capture group for the video name.

Also handled:
- class order: Roboflow's names list is mapped onto this project's
  0 = smoke, 1 = fire by name, never by index;
- polygon labels (YOLO segmentation rows) are converted to their enclosing box;
- an image with an empty label file stays as a negative.

    python scripts/split_roboflow_by_video.py --export <unzipped export> \\
        --output datasets/own_split --test-frac 0.2
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUT_NAMES = ["smoke", "fire"]
NAME_MAP = {"smoke": "smoke", "fire": "fire", "flame": "fire", "flames": "fire"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", required=True, help="Unzipped Roboflow export (folder holding data.yaml).")
    p.add_argument("--output", required=True)
    p.add_argument("--test-frac", type=float, default=0.2, help="Share of videos (not frames) held out for test.")
    p.add_argument("--group-regex", default=None, help="Regex with one group that captures the video name.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="Only print the groups and the split.")
    return p.parse_args()


def video_key(stem: str, rx: re.Pattern | None) -> str:
    if rx:
        m = rx.search(stem)
        if not m:
            raise SystemExit(f"--group-regex does not match {stem!r}")
        return m.group(1)
    s = re.sub(r"\.rf\.[0-9a-f]+$", "", stem)          # Roboflow hash
    s = re.sub(r"_(jpe?g|png|bmp|webp)$", "", s, flags=re.I)  # original extension
    s = re.sub(r"[-_ .]*(frame|frm|img|image)?[-_ .]*\d+$", "", s, flags=re.I)  # frame number
    return s or stem


def read_labels(path: Path, class_map: dict[int, int | None]) -> list[str]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        v = line.split()
        if not v:
            continue
        cls = class_map.get(int(float(v[0])))
        if cls is None:
            continue
        nums = [float(x) for x in v[1:]]
        if len(nums) == 4:
            cx, cy, w, h = nums
        elif len(nums) >= 6 and len(nums) % 2 == 0:   # polygon: x1 y1 x2 y2 ...
            xs, ys = nums[0::2], nums[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        else:
            continue
        if w <= 0 or h <= 0:
            continue
        out.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.export)
    cfg = yaml.safe_load((root / "data.yaml").read_text())
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    class_map: dict[int, int | None] = {}
    for i, n in enumerate(names):
        target = NAME_MAP.get(str(n).strip().lower())
        if target is None:
            raise SystemExit(f"unknown class {n!r} in the export; add it to NAME_MAP")
        class_map[i] = OUT_NAMES.index(target)
    print("class mapping:", {names[i]: OUT_NAMES[c] for i, c in class_map.items()})

    rx = re.compile(args.group_regex) if args.group_regex else None
    frames = []
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS or "images" not in img.parts:
            continue
        lab = Path(str(img.parent).replace("images", "labels")) / (img.stem + ".txt")
        frames.append((img, read_labels(lab, class_map)))
    if not frames:
        raise SystemExit("no images found under an 'images' folder")

    groups: dict[str, list] = defaultdict(list)
    for img, labels in frames:
        groups[video_key(img.stem, rx)].append((img, labels))
    print(f"{len(frames)} frames in {len(groups)} video groups")
    for k in list(sorted(groups, key=lambda g: -len(groups[g])))[:10]:
        print(f"  {len(groups[k]):5d}  {k}")
    if len(groups) < 10:
        print("WARNING: very few groups; the file names may not carry the video name. Use --group-regex.")

    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    # fill test with whole videos until it holds about test_frac of the frames
    target = args.test_frac * len(frames)
    test_keys, n = set(), 0
    for k in keys:
        if n >= target:
            break
        test_keys.add(k)
        n += len(groups[k])

    def summary(ks):
        items = [x for k in ks for x in groups[k]]
        fire = sum(any(l.startswith("1 ") for l in lab) for _, lab in items)
        smoke = sum(any(l.startswith("0 ") for l in lab) for _, lab in items)
        neg = sum(not lab for _, lab in items)
        return f"{len(ks)} videos, {len(items)} frames (fire {fire}, smoke {smoke}, negative {neg})"

    train_keys = [k for k in keys if k not in test_keys]
    print("train:", summary(train_keys))
    print("test :", summary(test_keys))
    if args.dry_run:
        return

    out = Path(args.output)
    for split, ks in (("train", train_keys), ("test", sorted(test_keys))):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        for k in ks:
            for img, labels in groups[k]:
                shutil.copy2(img, out / split / "images" / img.name)
                (out / split / "labels" / (img.stem + ".txt")).write_text("\n".join(labels) + ("\n" if labels else ""))
    # train: added to training through merge_extra_datasets.py (YOLO source)
    (out / "train" / "data.yaml").write_text(yaml.safe_dump(
        {"path": ".", "train": "images", "val": "images", "names": OUT_NAMES}))
    # test: evaluated on its own with eval_dinov3_detector.py --split test
    (out / "test" / "data.yaml").write_text(yaml.safe_dump(
        {"path": ".", "train": "images", "val": "images", "test": "images", "names": OUT_NAMES}))
    (out / "split.yaml").write_text(yaml.safe_dump({"train_videos": train_keys, "test_videos": sorted(test_keys)}))
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
