"""Find the fire-free training images that the current model still fires on.

A negative image the model already ignores teaches it little; one where it
still produces a confident fire or smoke box is exactly what the next
fine-tune should see more often. This script runs a trained model over every
*negative* image of a split (empty or missing label file) and writes the ones
whose best detection reaches --min-score, highest first:

    <image name> <best score> <class>

Pass the file to train_dinov3_detector.py with --repeat-list so these images
are drawn --repeat-factor times as often. Only the training split should be
mined; mining validation or test images would leak them into training.

    python scripts/mine_hard_negatives.py --weights best.pt \\
        --data /kaggle/working/merged/data.yaml --output hard_negatives.txt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from fire_smoke.dataset import FireSmokeDataset, letterbox
from fire_smoke.glow import glow_features
from fire_smoke.model import load_detector

CLASSES = {1: "smoke", 2: "fire"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, help="data.yaml of the training set.")
    p.add_argument("--split", default="train", choices=["train"], help="Only the training split may be mined.")
    p.add_argument("--min-score", type=float, default=0.30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="hard_negatives.txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    use_amp = device.type == "cuda"
    model, _ = load_detector(args.weights, device)
    model.eval()
    size = model.config["image_size"]
    transform = model.detector.transform
    transform.min_size, transform.max_size = (size,), size

    ds = FireSmokeDataset(args.data, args.split, size)
    negatives = [p for p in ds.images
                 if not ds.label_path(p).exists() or not ds.label_path(p).read_text(errors="replace").strip()]
    print(f"{len(ds.images)} {args.split} images, {len(negatives)} negatives to check at {size}px")

    found = []
    for i in tqdm(range(0, len(negatives), args.batch), desc="mining"):
        chunk = negatives[i:i + args.batch]
        tensors, glows, keep = [], [], []
        for f in chunk:
            bgr = cv2.imread(str(f))
            if bgr is None:
                continue
            canvas, _ = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size)
            tensors.append(torch.from_numpy(canvas.transpose(2, 0, 1).copy()).to(device).float().div_(255.0))
            glows.append(glow_features(canvas))
            keep.append(f)
        if not tensors:
            continue
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            dets, _ = model(tensors, glow=torch.from_numpy(np.stack(glows)).to(device))
        for f, d in zip(keep, dets):
            if len(d["scores"]):
                k = int(d["scores"].argmax())
                score = float(d["scores"][k])
                if score >= args.min_score:
                    found.append((f.name, score, CLASSES.get(int(d["labels"][k]), "?")))

    found.sort(key=lambda t: -t[1])
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(f"# {len(found)} of {len(negatives)} negatives with a detection >= {args.min_score} ({args.weights})\n")
        for name, score, cls in found:
            fh.write(f"{name} {score:.3f} {cls}\n")
    by_source = Counter(n.split("_neg")[0] if "_neg" in n else "base" for n, _, _ in found)
    print(f"{len(found)} hard negatives (>= {args.min_score}) of {len(negatives)}; by source: {dict(by_source)}")
    for t in (0.5, 0.7, 0.9):
        print(f"  >= {t}: {sum(s >= t for _, s, _ in found)}")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
