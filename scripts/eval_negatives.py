"""False alarms on a folder of fire-free images, for one or more models.

The D-Fire negatives contain few of the objects that fool the detector on an
industrial site (yellow helmets, hi-vis clothing), so a model trained against
those objects can look unchanged on the D-Fire test set. This script measures
it directly: every image in --folder is assumed to contain no fire and no
smoke, and the report is the share of images with at least one detection at
each confidence threshold, per class.

Images that were used for training must be left out, or the newer model is
being tested on its own training data. --exclude-dir takes the merged training
folder: merge_extra_datasets.py names copied files `<source>_neg<k>_<stem>.jpg`,
and any image in --folder whose stem appears there is skipped.

    python scripts/eval_negatives.py \\
        --model "Model 6=/kaggle/input/.../model6.pt" --model "Model 7=.../best.pt" \\
        --folder /kaggle/input/datasets/andrewmvd/hard-hat-detection/images \\
        --exclude-dir /kaggle/working/merged/images/train --exclude-prefix hardhat
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from fire_smoke.dataset import letterbox
from fire_smoke.glow import glow_features
from fire_smoke.model import load_detector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = {1: "smoke", 2: "fire"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True, help="LABEL=path/to/weights.pt (repeatable).")
    p.add_argument("--folder", required=True, help="Folder of fire-free images (searched recursively).")
    p.add_argument("--exclude-dir", default=None, help="Training image folder; images used there are skipped.")
    p.add_argument("--exclude-prefix", default=None, help="Only names starting with this source prefix count.")
    p.add_argument("--thresholds", default="0.50,0.70,0.80,0.85,0.90,0.93")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-examples", default=None, help="Folder for the highest-scoring false detections.")
    p.add_argument("--output", default=None, help="Write the results as JSON here.")
    return p.parse_args()


def excluded_stems(folder: str | None, prefix: str | None) -> set[str]:
    if not folder:
        return set()
    stems = set()
    pattern = re.compile(r"^(?P<src>[^_]+)_neg\d+_(?P<stem>.+)$")
    for f in Path(folder).iterdir():
        m = pattern.match(f.stem)
        if m and (prefix is None or m.group("src") == prefix):
            stems.add(m.group("stem"))
    return stems


def best_scores(model, files, device, batch, use_amp):
    """Highest fire and smoke score in each image."""
    size = model.config["image_size"]
    transform = model.detector.transform
    transform.min_size, transform.max_size = (size,), size
    out = []
    for i in tqdm(range(0, len(files), batch), desc="images", leave=False):
        chunk = files[i:i + batch]
        tensors, glows = [], []
        for f in chunk:
            bgr = cv2.imread(str(f))
            canvas, _ = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size)
            tensors.append(torch.from_numpy(canvas.transpose(2, 0, 1).copy()).to(device).float().div_(255.0))
            glows.append(glow_features(canvas))
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            dets, _ = model(tensors, glow=torch.from_numpy(np.stack(glows)).to(device))
        for d in dets:
            s, l = d["scores"].float().cpu().numpy(), d["labels"].cpu().numpy()
            out.append({name: float(s[l == c].max()) if (l == c).any() else 0.0 for c, name in CLASSES.items()})
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    use_amp = device.type == "cuda"
    skip = excluded_stems(args.exclude_dir, args.exclude_prefix)
    files = sorted(p for p in Path(args.folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    total = len(files)
    files = [f for f in files if f.stem not in skip]
    if args.max_images:
        files = files[:args.max_images]
    if not files:
        raise SystemExit("no images left to test")
    print(f"{total} images in {args.folder}; {total - len(files) if not args.max_images else len(skip)} skipped "
          f"as used in training; testing {len(files)}")
    thresholds = [float(t) for t in args.thresholds.split(",")]

    results = {}
    for spec in args.model:
        label, path = spec.split("=", 1)
        model, _ = load_detector(path, device)
        model.eval()
        scores = best_scores(model, files, device, args.batch, use_amp)
        rows = []
        for t in thresholds:
            fire = sum(s["fire"] >= t for s in scores)
            smoke = sum(s["smoke"] >= t for s in scores)
            anyc = sum(max(s.values()) >= t for s in scores)
            rows.append({"conf": t, "fire": fire, "smoke": smoke, "any": anyc, "rate": anyc / len(scores)})
        results[label] = rows
        if args.save_examples:
            out = Path(args.save_examples) / re.sub(r"\W+", "_", label)
            out.mkdir(parents=True, exist_ok=True)
            top = sorted(range(len(scores)), key=lambda k: -max(scores[k].values()))[:12]
            for rank, k in enumerate(top):
                s = scores[k]
                cls = max(s, key=s.get)
                cv2.imwrite(str(out / f"{rank:02d}_{cls}_{s[cls]:.2f}_{files[k].stem}.jpg"), cv2.imread(str(files[k])))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nimages with a detection (out of {len(files)} fire-free images)")
    labels = list(results)
    head = f"{'conf':>5}" + "".join(f" | {l[:22]:>22s}" for l in labels)
    print(head + "\n" + "-" * len(head))
    for i, t in enumerate(thresholds):
        cells = "".join(f" | {r[i]['any']:>5d} ({r[i]['rate']:6.2%}) f{r[i]['fire']:<4d}s{r[i]['smoke']:<4d}"
                        for r in (results[l] for l in labels))
        print(f"{t:>5.2f}{cells}")
    print("(f = images with a fire detection, s = images with a smoke detection)")
    if args.output:
        Path(args.output).write_text(json.dumps({"images": len(files), "results": results}, indent=2))
        print("wrote", args.output)


if __name__ == "__main__":
    main()
