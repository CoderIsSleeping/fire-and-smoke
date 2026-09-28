"""Add extra fire / smoke datasets to the training split, safely.

Three things can silently ruin a merged dataset, and this script guards
against each one:

1. **Test leakage.** Public fire datasets are often built partly from other
   public fire datasets (FASDD, for one, includes "open-access fire datasets").
   If a D-Fire test image slips into training, the test score goes up for the
   wrong reason. Every new image is compared, by perceptual hash, against
   *every* image already in the base dataset -- train, val and test -- and
   dropped if it is a near-duplicate. New images are also de-duplicated
   against each other (video-frame datasets repeat themselves heavily).

2. **Split contamination.** New data goes into **train only**. val and test
   are carried over byte-for-byte, so scores before and after the merge are
   measured on exactly the same images and stay comparable.

3. **Label mismatch.** Each source's class names are mapped explicitly to
   `smoke` / `fire` / `negative`. A `negative` class (e.g. DFS's "other":
   lamps, sunsets, orange objects that look like fire) has its boxes removed,
   and an image left with no boxes becomes a *hard negative* -- the kind of
   image that teaches the model what fire is not. Unknown class names stop
   the script rather than being guessed.

Formats read: YOLO (`.txt` per image + a names list), Pascal VOC (`.xml`),
COCO (`.json`). New images can be downscaled on copy (`--resize 640`, the
training size anyway), which keeps a 100k-image source inside Kaggle's 20 GB
working-disk limit.

    python scripts/merge_extra_datasets.py \\
        --base /kaggle/input/<dfire-dataset> \\
        --source fasdd=/kaggle/input/<fasdd>:coco \\
        --source dfs=/kaggle/input/<dfs>:voc \\
        --output /kaggle/working/merged --resize 640

Then train with `--data /kaggle/working/merged/data.yaml`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUT_CLASSES = ["smoke", "fire"]  # must match the base data.yaml
DEFAULT_MAP = {
    "smoke": "smoke",
    "fire": "fire", "flame": "fire", "flames": "fire",
    "fire_large": "fire", "fire_medium": "fire", "fire_small": "fire",
    "large_fire": "fire", "medium_fire": "fire", "small_fire": "fire",
    "other": "negative", "others": "negative",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge extra datasets into the training split, with leakage checks.")
    p.add_argument("--base", required=True, help="Folder holding the base data.yaml (the D-Fire dataset).")
    p.add_argument("--source", action="append", required=True,
                   help="NAME=PATH:FORMAT, FORMAT one of yolo|voc|coco. Repeatable.")
    p.add_argument("--map", action="append", default=[],
                   help="Extra class mapping NAME=smoke|fire|negative, e.g. --map Flame=fire. Repeatable.")
    p.add_argument("--yolo-names", action="append", default=[],
                   help="Class names for a YOLO source with no data.yaml: NAME=cls0,cls1,... Repeatable.")
    p.add_argument("--output", required=True)
    p.add_argument("--resize", type=int, default=0, help="Downscale new images so the long side is at most N px.")
    p.add_argument("--hash-distance", type=int, default=3,
                   help="Max Hamming distance (of 64 bits) that counts as a duplicate. 0-3 supported.")
    p.add_argument("--max-per-source", type=int, default=0, help="Cap on kept images per source (0 = no cap).")
    p.add_argument("--max-negatives-per-source", type=int, default=-1,
                   help="Cap on box-free images kept per source (-1 = same as positives, 0 = none).")
    p.add_argument("--only-with-fire", action="store_true",
                   help="Keep positives only if they contain fire (use to fix the fire/smoke image imbalance).")
    p.add_argument("--copy-base", action="store_true", help="Copy base files instead of hard-linking them.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------- hashing

THUMB = 64


def fingerprint(path: Path):
    """(64-bit difference hash, 64x64 grayscale thumbnail), or None if unreadable.

    The hash is only a fast candidate filter. On its own it is far too coarse:
    two frames from the same fixed camera, one with a small fire and one
    without, were measured at 3 bits apart -- a hash-only check would throw
    away exactly the images worth adding. So every hash match is confirmed on
    the thumbnail, pixel by pixel, and only true copies (resized, re-encoded,
    slightly recoloured) count as duplicates.
    """
    try:
        with Image.open(path) as im:
            gray = im.convert("L")
            g = np.asarray(gray.resize((9, 8), Image.BILINEAR), dtype=np.int16)
            thumb = np.asarray(gray.resize((THUMB, THUMB), Image.BILINEAR), dtype=np.uint8)
    except Exception:  # noqa: BLE001 - unreadable images are simply skipped
        return None
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2), thumb


def same_image(a: np.ndarray, b: np.ndarray) -> bool:
    """True copy test: small average difference AND no local region that differs.

    The percentile term is what separates "same frame, re-encoded" from "same
    camera, but now there is a fire": a flame changes a small patch a lot.
    """
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(diff.mean()) < 6.0 and float(np.percentile(diff, 99.5)) < 40.0


class HashIndex:
    """Near-duplicate lookup. With distance <= 3 split over 4 x 16-bit chunks,
    any match must agree exactly on at least one chunk (pigeonhole), so only
    those buckets need checking. Matches are then confirmed by `same_image`."""

    def __init__(self, max_distance: int) -> None:
        if not 0 <= max_distance <= 3:
            raise ValueError("--hash-distance must be between 0 and 3")
        self.max_distance = max_distance
        self.buckets = [defaultdict(list) for _ in range(4)]

    @staticmethod
    def _chunks(h: int):
        return [(h >> (16 * i)) & 0xFFFF for i in range(4)]

    def add(self, fp, tag: str) -> None:
        h, thumb = fp
        for i, c in enumerate(self._chunks(h)):
            self.buckets[i][c].append((h, thumb, tag))

    def find(self, fp) -> str | None:
        h, thumb = fp
        seen = set()
        for i, c in enumerate(self._chunks(h)):
            for other, other_thumb, tag in self.buckets[i].get(c, ()):
                if id(other_thumb) in seen:
                    continue
                seen.add(id(other_thumb))
                if bin(h ^ other).count("1") <= self.max_distance and same_image(thumb, other_thumb):
                    return tag
        return None


# ---------------------------------------------------------------- readers
# Every reader yields (image_path, [(class_name, x1, y1, x2, y2) in pixels]).

def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def read_yolo(root: Path, names: list[str] | None):
    if names is None:
        yaml_files = list(root.rglob("data.yaml")) + list(root.rglob("*.yaml"))
        for y in yaml_files:
            names = yaml_names(y)
            if names:
                print(f"  class names from {y}: {names}")
                break
    if not names:
        raise SystemExit(f"YOLO source {root} has no data.yaml with names; pass --yolo-names NAME=cls0,cls1")
    for img in find_images(root):
        parts = list(img.parts)
        label = None
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                label = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
                break
        if label is None:
            label = img.with_suffix(".txt")
        if not label.exists():
            continue  # unlabelled image: we cannot tell whether it is negative, skip it
        w, h = image_size(img)
        boxes = []
        for line in label.read_text(encoding="utf-8", errors="replace").splitlines():
            vals = line.split()
            if len(vals) < 5:
                continue
            c, cx, cy, bw, bh = int(float(vals[0])), *map(float, vals[1:5])
            name = names[c] if c < len(names) else f"class_{c}"
            boxes.append((name, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        yield img, boxes


def yaml_names(path: Path) -> list[str] | None:
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names) if names else None


def read_voc(root: Path):
    images = {p.stem: p for p in find_images(root)}
    for xml in sorted(root.rglob("*.xml")):
        try:
            tree = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        fname = tree.findtext("filename") or xml.stem
        img = images.get(Path(fname).stem) or images.get(xml.stem)
        if img is None:
            continue
        boxes = []
        for obj in tree.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            boxes.append((obj.findtext("name", "").strip(),
                          *(float(bb.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax"))))
        yield img, boxes


def read_coco(root: Path):
    images = {p.name: p for p in find_images(root)}
    for js in sorted(root.rglob("*.json")):
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
            continue
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        anns = defaultdict(list)
        for a in data["annotations"]:
            x, y, w, h = a["bbox"]
            anns[a["image_id"]].append((cats.get(a["category_id"], str(a["category_id"])), x, y, x + w, y + h))
        print(f"  {js.name}: {len(data['images'])} images, categories {sorted(set(cats.values()))}")
        for im in data["images"]:
            img = images.get(Path(im["file_name"]).name)
            if img is not None:
                yield img, anns.get(im["id"], [])


# ---------------------------------------------------------------- merge

def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def write_image(src: Path, dst: Path, resize: int) -> tuple[int, int, float]:
    """Copy (optionally downscaled) and return (w, h, scale) of the written image."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = 1.0
        if resize and max(w, h) > resize:
            scale = resize / max(w, h)
            im = im.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        im.save(dst, "JPEG", quality=92)
        return im.size[0], im.size[1], scale


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    base = Path(args.base)
    base_yaml = base / "data.yaml"
    if not base_yaml.exists():
        found = list(base.rglob("data.yaml"))
        if not found:
            raise SystemExit(f"no data.yaml under {base}")
        base_yaml = found[0]
        base = base_yaml.parent
    base_names = yaml_names(base_yaml)
    if base_names != OUT_CLASSES:
        raise SystemExit(f"base classes {base_names} != expected {OUT_CLASSES}")

    class_map = dict(DEFAULT_MAP)
    for m in args.map:
        k, v = m.split("=", 1)
        if v not in ("smoke", "fire", "negative"):
            raise SystemExit(f"--map target must be smoke|fire|negative, got {v!r}")
        class_map[k.strip().lower()] = v
    yolo_names = {k: v.split(",") for k, v in (s.split("=", 1) for s in args.yolo_names)}

    out = Path(args.output)
    report: dict = {"base": str(base), "sources": {}, "hash_distance": args.hash_distance}

    # 1. carry the base dataset over unchanged, and hash every image of it
    index = HashIndex(args.hash_distance)
    base_counts = Counter()
    for split in ("train", "val", "test"):
        img_dir, lbl_dir = base / "images" / split, base / "labels" / split
        if not img_dir.exists():
            continue
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        for img in tqdm(imgs, desc=f"base {split}"):
            link_or_copy(img, out / "images" / split / img.name, args.copy_base)
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                link_or_copy(lbl, out / "labels" / split / lbl.name, args.copy_base)
            fp = fingerprint(img)
            if fp is not None:
                index.add(fp, f"base/{split}")
            base_counts[split] += 1
    report["base_images"] = dict(base_counts)

    # 2. add each source to train
    for spec in args.source:
        name, rest = spec.split("=", 1)
        path_str, fmt = rest.rsplit(":", 1)
        root = Path(path_str)
        print(f"\nsource {name}: {root} ({fmt})")
        reader = {"yolo": lambda r: read_yolo(r, yolo_names.get(name)), "voc": read_voc, "coco": read_coco}.get(fmt)
        if reader is None:
            raise SystemExit(f"unknown format {fmt!r}")

        items = list(reader(root))
        random.shuffle(items)
        unknown = Counter()
        stats = Counter(read=len(items))
        positives, negatives = [], []
        for img, boxes in items:
            mapped = []
            had_negative_class = False
            for cls, x1, y1, x2, y2 in boxes:
                target = class_map.get(cls.strip().lower())
                if target is None:
                    unknown[cls] += 1
                elif target == "negative":
                    had_negative_class = True
                elif x2 - x1 >= 2 and y2 - y1 >= 2:
                    mapped.append((OUT_CLASSES.index(target), x1, y1, x2, y2))
            if mapped:
                if args.only_with_fire and not any(c == 1 for c, *_ in mapped):
                    stats["skipped_no_fire"] += 1
                    continue
                positives.append((img, mapped))
            else:
                negatives.append((img, had_negative_class))
        if unknown:
            raise SystemExit(f"source {name}: unmapped class names {dict(unknown)}. "
                             f"Add e.g. --map '{next(iter(unknown))}=fire'.")

        # hard negatives (from a 'negative' class like DFS 'other') first
        negatives.sort(key=lambda t: not t[1])
        neg_cap = len(positives) if args.max_negatives_per_source < 0 else args.max_negatives_per_source
        kept = Counter()
        for kind, pool, cap in (("positive", positives, args.max_per_source or len(positives)),
                                ("negative", negatives, neg_cap)):
            for entry in tqdm(pool, desc=f"{name} {kind}s"):
                if kept[kind] >= cap:
                    break
                img = entry[0]
                fp = fingerprint(img)
                if fp is None:
                    stats["unreadable"] += 1
                    continue
                dup = index.find(fp)
                if dup is not None:
                    stats[f"duplicate_of_{dup.replace('/', '_')}"] += 1
                    continue
                index.add(fp, f"{name}")
                stem = f"{name}_{kept[kind] if kind == 'positive' else 'neg' + str(kept[kind])}_{img.stem}"[:120]
                dst = out / "images" / "train" / f"{stem}.jpg"
                w, hgt, scale = write_image(img, dst, args.resize)
                lines = []
                if kind == "positive":
                    for c, x1, y1, x2, y2 in entry[1]:
                        x1, y1, x2, y2 = (v * scale for v in (x1, y1, x2, y2))
                        x1, x2 = max(0.0, x1), min(float(w), x2)
                        y1, y2 = max(0.0, y1), min(float(hgt), y2)
                        if x2 - x1 < 2 or y2 - y1 < 2:
                            continue
                        lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / hgt:.6f} "
                                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / hgt:.6f}")
                        stats[f"{OUT_CLASSES[c]}_boxes"] += 1
                    if not lines:
                        dst.unlink()
                        stats["all_boxes_degenerate"] += 1
                        continue
                    stats["images_with_fire"] += any(l.startswith("1 ") for l in lines)
                    stats["images_with_smoke"] += any(l.startswith("0 ") for l in lines)
                else:
                    stats["hard_negatives" if entry[1] else "plain_negatives"] += 1
                lbl = out / "labels" / "train" / f"{stem}.txt"
                lbl.parent.mkdir(parents=True, exist_ok=True)
                lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                kept[kind] += 1
        stats["kept_positive"], stats["kept_negative"] = kept["positive"], kept["negative"]
        report["sources"][name] = dict(stats)
        print(f"  {dict(stats)}")

    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        "names:\n  0: smoke\n  1: fire\n", encoding="utf-8")
    (out / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    leaks = sum(v for s in report["sources"].values() for k, v in s.items() if k.startswith("duplicate_of_base"))
    print(f"\nremoved {leaks} images that duplicated the base dataset (would have leaked into train)")
    print(f"wrote {out / 'data.yaml'}  and  merge_report.json")


if __name__ == "__main__":
    sys.exit(main())
