"""Diagnostic: how much does each glow feature actually say about fire?

Reports the ROC AUC of every hand-crafted glow feature, fire images versus
verified-negative images, in two conditions: the dataset as-is, and with the
low-light augmentation applied.

Why this exists. It is very easy to invent a "fire-glow" heuristic that looks
convincing and measures at chance. It is even easier to invent one that scores
brilliantly for the wrong reason -- global brightness is the strongest
discriminator on D-Fire (AUC ~0.82) purely because fire photographs tend to be
taken at night. A model that learns that will alarm on darkness, every night,
forever. This script is how that gets caught rather than shipped.

Reading the output:
    AUC ~ 0.5   the feature says nothing
    AUC > 0.5   higher values indicate fire
    AUC < 0.5   higher values indicate *negative* (still informative, inverted)
    |AUC - 0.5| is the amount of information, regardless of direction

Run it again after changing anything in `fire_smoke/glow.py`, and after
switching to real site footage -- the numbers below are D-Fire's, not yours.

    python scripts/analyze_glow_prior.py --split val --per-class 150
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from fire_smoke.dataset import (
    AugmentConfig,
    FireSmokeDataset,
    find_data_yaml,
    night_augment,
    read_yolo_label,
    to_gray,
)
from fire_smoke.glow import EXCLUDED_EXPOSURE_FEATURES, GLOW_FEATURE_NAMES, glow_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure how informative each glow feature is.")
    p.add_argument("--data", default=None)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--per-class", type=int, default=150, help="Images sampled per group.")
    p.add_argument("--target", default="fire", choices=["fire", "smoke"], help="Positive class.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), tolerant of ties."""
    values = np.concatenate([positive, negative])
    order = values.argsort()
    ranks = np.empty(len(values), float)
    ranks[order] = np.arange(1, len(values) + 1)
    # average ranks within tied groups
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]

    n_pos, n_neg = len(positive), len(negative)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def collect(paths, transform) -> np.ndarray:
    rows = []
    for index, path in enumerate(paths):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if transform is not None:
            image = transform(image, index)
        rows.append(glow_features(image))
    return np.array(rows, np.float32)


def main() -> None:
    args = parse_args()
    data_yaml = find_data_yaml(args.data)
    dataset = FireSmokeDataset(data_yaml, args.split, 640, AugmentConfig.disabled())
    target_id = 1 if args.target == "fire" else 0

    positives, negatives = [], []
    for path in dataset.images:
        _, labels = read_yolo_label(dataset.label_path(path), 1, 1)
        if (labels == target_id).any():
            if len(positives) < args.per_class:
                positives.append(path)
        elif len(labels) == 0 and len(negatives) < args.per_class:
            negatives.append(path)
        if len(positives) >= args.per_class and len(negatives) >= args.per_class:
            break

    print(f"data  : {data_yaml}  (split={args.split})")
    print(f"groups: {len(positives)} images containing {args.target}, "
          f"{len(negatives)} verified negatives\n")

    conditions = {
        "as-is": None,
        "night-augmented": lambda img, i: night_augment(img, random.Random(args.seed + i)),
        "night + IR/gray": lambda img, i: to_gray(night_augment(img, random.Random(args.seed + i))),
    }

    for condition, transform in conditions.items():
        pos = collect(positives, transform)
        neg = collect(negatives, transform)
        print(f"=== {condition} ===")
        print(f"{'feature':<20} {'AUC':>6} {'info':>6}  {'pos mean':>9} {'neg mean':>9}  direction")
        rows = []
        for i, name in enumerate(GLOW_FEATURE_NAMES):
            auc = roc_auc(pos[:, i], neg[:, i])
            rows.append((abs(auc - 0.5), auc, name, pos[:, i].mean(), neg[:, i].mean()))
        for info, auc, name, pm, nm in sorted(rows, reverse=True):
            direction = "higher = fire" if auc > 0.55 else ("higher = normal" if auc < 0.45 else "-- uninformative")
            print(f"{name:<20} {auc:>6.3f} {info:>6.3f}  {pm:>9.4f} {nm:>9.4f}  {direction}")
        print()

    print("Excluded from the feature vector on purpose:")
    print(f"  {', '.join(EXCLUDED_EXPOSURE_FEATURES)}")
    print("  These global-exposure statistics score highest on D-Fire, but only because")
    print("  fire photographs are disproportionately dark. See fire_smoke/glow.py.")


if __name__ == "__main__":
    main()
