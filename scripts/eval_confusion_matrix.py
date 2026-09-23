"""Confusion matrix, precision/recall/F1 and box-coordinate MSE, actually measured.

The standard evaluation (`eval_dinov3_detector.py`) reports mAP and an
image-level alarm sweep (false-alarm rate, recall). Neither is a confusion
matrix, and neither is precision or F1 at a single operating point. This
script computes exactly those, by re-running the trained model over real
images -- nothing here is estimated or interpolated from the AP curve.

Two confusion matrices are produced, per class (fire, smoke), not one 3-way
matrix. That is the correct construction for this problem: an image can
legitimately contain both fire and smoke at once, so "the" true class is not
well-defined, and a multi-label detector should be scored one class at a time.
For each class, at the image level:

    TP  = image contains that class AND the model detected it above threshold
    FN  = image contains that class AND the model did NOT detect it
    FP  = image does not contain that class AND the model detected it anyway
    TN  = image does not contain that class AND the model stayed silent

From those four counts: precision, recall, F1, specificity and accuracy follow
directly and are reported exactly, not approximated.

Box-coordinate MSE is computed only over matched true positives (IoU >= 0.5
against ground truth): the mean squared pixel error of the box coordinates.
This is the closest honest analogue to "validation MSE" for a detector --
detectors are not trained with an MSE loss (Faster R-CNN uses smooth-L1 on box
deltas, FCOS uses IoU loss + BCE), so box-coordinate MSE is reported as a
supplementary, independently-computed diagnostic, not the training objective.

    python scripts/eval_confusion_matrix.py --weights <ckpt> --split test --max-images 1200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fire_smoke.dataset import AugmentConfig, FireSmokeDataset, collate_fn, find_data_yaml
from fire_smoke.metrics import box_iou
from fire_smoke.model import load_detector

CLASS_IDS = {"smoke": 1, "fire": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Confusion matrix, precision/recall/F1 and box MSE.")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--conf", type=float, required=True, help="Operating threshold for this checkpoint.")
    p.add_argument("--iou-match", type=float, default=0.5, help="IoU to count a box as a true positive.")
    p.add_argument("--max-images", type=int, default=0, help="0 = full split.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", required=True)
    p.add_argument("--label", default=None, help="Name for this run in the output JSON.")
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def confusion_counts(preds, gts, class_id, conf):
    """Image-level TP/FP/FN/TN for one class at one confidence threshold."""
    tp = fp = fn = tn = 0
    for pred, gt in zip(preds, gts):
        has_gt = bool((gt["labels"] == class_id).any())
        keep = (pred["labels"] == class_id) & (pred["scores"] >= conf)
        has_pred = bool(keep.any())
        if has_gt and has_pred:
            tp += 1
        elif has_gt and not has_pred:
            fn += 1
        elif not has_gt and has_pred:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def prf1(c: dict) -> dict:
    precision = c["tp"] / max(c["tp"] + c["fp"], 1)
    recall = c["tp"] / max(c["tp"] + c["fn"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    specificity = c["tn"] / max(c["tn"] + c["fp"], 1)
    accuracy = (c["tp"] + c["tn"]) / max(sum(c.values()), 1)
    return {"precision": precision, "recall": recall, "f1": f1,
            "specificity": specificity, "accuracy": accuracy}


def box_coordinate_mse(preds, gts, class_id, conf, iou_match) -> dict:
    """MSE of [x1,y1,x2,y2] in pixels, over matched true-positive boxes only."""
    errors = []
    for pred, gt in zip(preds, gts):
        gt_boxes = gt["boxes"][gt["labels"] == class_id]
        if len(gt_boxes) == 0:
            continue
        keep = (pred["labels"] == class_id) & (pred["scores"] >= conf)
        pred_boxes = pred["boxes"][keep]
        if len(pred_boxes) == 0:
            continue
        ious = box_iou(pred_boxes, gt_boxes)
        for i in range(len(pred_boxes)):
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= iou_match:
                errors.append((pred_boxes[i] - gt_boxes[j]) ** 2)
    if not errors:
        return {"n_matched": 0, "mse_pixels_sq": None, "rmse_pixels": None}
    errors = np.array(errors)
    mse = float(errors.mean())
    return {"n_matched": len(errors), "mse_pixels_sq": mse, "rmse_pixels": float(np.sqrt(mse))}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_yaml = find_data_yaml(args.data)

    model, checkpoint = load_detector(args.weights, device)
    image_size = model.config["image_size"]
    label = args.label or f"{model.config.get('backbone_name', '?')[-24:]} / {model.head_type}"

    dataset = FireSmokeDataset(data_yaml, args.split, image_size, AugmentConfig.disabled(),
                               max_images=args.max_images or None)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=collate_fn, pin_memory=device.type == "cuda")

    print(f"label     : {label}")
    print(f"weights   : {args.weights}  (checkpoint epoch {checkpoint.get('epoch', '?')})")
    print(f"data      : {data_yaml}  split={args.split}  images={len(dataset)}")
    print(f"threshold : conf >= {args.conf}, IoU >= {args.iou_match}\n")

    preds, gts = [], []
    for images, targets, _scene, glow in tqdm(loader, desc="eval"):
        images = [img.to(device) for img in images]
        detections, _ = model(images, glow=glow.to(device))
        for det in detections:
            preds.append({"boxes": det["boxes"].float().cpu().numpy(),
                         "scores": det["scores"].float().cpu().numpy(),
                         "labels": det["labels"].cpu().numpy()})
        for target in targets:
            gts.append({"boxes": target["boxes"].numpy(), "labels": target["labels"].numpy()})

    report = {"label": label, "weights": str(args.weights), "split": args.split,
              "num_images": len(dataset), "conf_threshold": args.conf, "iou_match": args.iou_match,
              "per_class": {}}

    print(f"{'class':<8} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  {'precision':>9} {'recall':>7} "
          f"{'F1':>6} {'specificity':>11} {'accuracy':>8}")
    for name, class_id in CLASS_IDS.items():
        counts = confusion_counts(preds, gts, class_id, args.conf)
        metrics = prf1(counts)
        mse = box_coordinate_mse(preds, gts, class_id, args.conf, args.iou_match)
        report["per_class"][name] = {**counts, **metrics, "box_mse": mse}
        print(f"{name:<8} {counts['tp']:>5} {counts['fp']:>5} {counts['fn']:>5} {counts['tn']:>5}  "
              f"{metrics['precision']:>9.4f} {metrics['recall']:>7.4f} {metrics['f1']:>6.4f} "
              f"{metrics['specificity']:>11.4f} {metrics['accuracy']:>8.4f}")
        if mse["n_matched"]:
            print(f"         box-coordinate MSE (matched TPs, n={mse['n_matched']}): "
                  f"{mse['mse_pixels_sq']:.1f} px^2  (RMSE {mse['rmse_pixels']:.2f} px)")
        else:
            print("         box-coordinate MSE: no matched true positives at this threshold")

    macro = {k: float(np.mean([report["per_class"][c][k] for c in CLASS_IDS])) for k in
             ("precision", "recall", "f1", "specificity", "accuracy")}
    report["macro_avg"] = macro
    print(f"\nmacro avg (smoke, fire): precision {macro['precision']:.4f}  recall {macro['recall']:.4f}  "
          f"F1 {macro['f1']:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
