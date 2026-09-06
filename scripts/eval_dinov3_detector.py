"""Evaluate a trained detector and choose its deployment operating point.

Beyond mAP, this answers the two questions that decide whether the system can
actually be switched on:

  1. At what confidence threshold does the false-alarm rate on verified-negative
     images drop under the budget, and what recall survives at that point?
  2. What is it actually false-alarming *on*? `--save-fp-examples` writes out the
     highest-scoring false positives so the failure modes can be looked at
     (welding arcs, sunsets, steam, headlights) and fed back into the next
     fine-tuning round.

Example:
    python scripts/eval_dinov3_detector.py --weights runs/fire_smoke/dinov3_fire_smoke/weights/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fire_smoke import CLASS_NAMES
from fire_smoke.dataset import AugmentConfig, FireSmokeDataset, collate_fn, find_data_yaml, letterbox
from fire_smoke.metrics import (
    alarm_sweep,
    detection_metrics,
    format_alarm_table,
    format_detection_table,
    pick_operating_threshold,
    scene_sweep,
)
from fire_smoke.model import load_detector

BOX_COLORS = {1: (180, 180, 180), 2: (0, 140, 255)}  # BGR: smoke grey, fire orange


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the DINOv3 fire/smoke detector.")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=0, help="Override the checkpoint image size.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--target-fpr", type=float, default=0.01, help="False-alarm budget for the operating point.")
    p.add_argument("--output", default=None, help="Directory for the report (defaults next to the weights).")
    p.add_argument("--save-fp-examples", type=int, default=24, help="Dump the N worst false positives as images.")
    p.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def run_inference(model, loader, device, use_amp):
    model.eval()
    preds, gts, scene_probs, scene_targets = [], [], [], []
    for images, targets, scene, glow in tqdm(loader, desc="eval"):
        images = [img.to(device, non_blocking=True) for img in images]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
            detections, probs = model(images, glow=glow.to(device))
        for detection in detections:
            preds.append(
                {
                    "boxes": detection["boxes"].float().cpu().numpy(),
                    "scores": detection["scores"].float().cpu().numpy(),
                    "labels": detection["labels"].cpu().numpy(),
                }
            )
        for target in targets:
            gts.append({"boxes": target["boxes"].numpy(), "labels": target["labels"].numpy()})
        scene_probs.append(probs.float().cpu().numpy())
        scene_targets.append(scene.numpy())
    return preds, gts, np.concatenate(scene_probs), np.concatenate(scene_targets)


def save_false_positive_examples(dataset, preds, gts, threshold, out_dir: Path, limit: int) -> list[dict]:
    """Write out the negative images that alarmed most confidently."""
    offenders = []
    for index, (pred, gt) in enumerate(zip(preds, gts)):
        if len(gt["labels"]) != 0 or len(pred["scores"]) == 0:
            continue
        keep = pred["scores"] >= threshold
        if not keep.any():
            continue
        offenders.append((float(pred["scores"][keep].max()), index, keep))

    offenders.sort(key=lambda item: -item[0])
    offenders = offenders[:limit]
    if not offenders:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for rank, (score, index, keep) in enumerate(offenders, start=1):
        image_path = dataset.images[index]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        # Re-letterbox exactly as the dataset did, so the predicted boxes land
        # in the right place on the saved image.
        canvas, _ = letterbox(bgr, dataset.image_size)
        pred = preds[index]
        for box, box_score, label in zip(pred["boxes"][keep], pred["scores"][keep], pred["labels"][keep]):
            x1, y1, x2, y2 = (int(v) for v in box)
            color = BOX_COLORS.get(int(label), (0, 255, 0))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, f"{CLASS_NAMES[int(label)]} {box_score:.2f}", (x1, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        name = f"{rank:03d}_{score:.2f}_{image_path.stem}.jpg"
        cv2.imwrite(str(out_dir / name), canvas)
        records.append({"rank": rank, "score": score, "source": str(image_path), "saved": name})
    return records


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_yaml = find_data_yaml(args.data)

    overrides = {"image_size": args.imgsz} if args.imgsz else {}
    model, checkpoint = load_detector(args.weights, device, **overrides)
    image_size = model.config["image_size"]

    out_dir = Path(args.output) if args.output else Path(args.weights).parent.parent / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"weights   : {args.weights}")
    print(f"data      : {data_yaml}  (split={args.split})")
    print(f"device    : {device}, image size {image_size}")
    if "epoch" in checkpoint:
        print(f"checkpoint: epoch {checkpoint['epoch']}, best val mAP50 {checkpoint.get('best_map', float('nan')):.4f}")

    dataset = FireSmokeDataset(data_yaml, args.split, image_size, AugmentConfig.disabled(),
                               max_images=args.max_images or None)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=collate_fn, pin_memory=device.type == "cuda")

    preds, gts, scene_probs, scene_targets = run_inference(model, loader, device, args.amp)

    det = detection_metrics(preds, gts)
    sweep = alarm_sweep(preds, gts)
    operating = pick_operating_threshold(sweep, args.target_fpr)
    scene = scene_sweep(scene_probs, scene_targets)

    print("\n" + format_detection_table(det))
    print("\nimage-level alarm sweep (FPR is over verified-negative images)")
    print(format_alarm_table(sweep))
    print(
        f"\nrecommended operating threshold for FPR <= {args.target_fpr:.3f}: "
        f"conf >= {operating['threshold']:.2f}"
    )
    print(
        f"  false-alarm rate {operating['fpr']:.4f} "
        f"({operating['false_alarm_images']}/{operating['neg_images']} negative images)"
    )
    print(f"  recall: fire {operating['recall_fire']:.4f}, smoke {operating['recall_smoke']:.4f}, "
          f"any {operating['recall_any']:.4f}")

    print("\nscene classifier head @ 0.5")
    for name in ("smoke", "fire"):
        row = min(scene[name], key=lambda r: abs(r["threshold"] - 0.5))
        print(f"  {name:>5}: precision {row['precision']:.4f}  recall {row['recall']:.4f}  fpr {row['fpr']:.4f}")

    fp_records = []
    if args.save_fp_examples:
        fp_records = save_false_positive_examples(
            dataset, preds, gts, operating["threshold"], out_dir / "false_positives", args.save_fp_examples
        )
        if fp_records:
            print(f"\nsaved {len(fp_records)} false-positive examples to {out_dir / 'false_positives'}")
        else:
            print("\nno false positives above the operating threshold on this split")

    report = {
        "weights": str(args.weights),
        "data": str(data_yaml),
        "split": args.split,
        "image_size": image_size,
        "num_images": len(dataset),
        "detection": det,
        "alarm_sweep": sweep,
        "operating_point": operating,
        "target_fpr": args.target_fpr,
        "scene": scene,
        "false_positive_examples": fp_records,
    }
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    frames_per_second = 5
    (out_dir / "eval_report.md").write_text(
        "\n".join(
            [
                f"# Evaluation report - {args.split} split",
                "",
                f"- weights: `{args.weights}`",
                f"- images: {len(dataset)} ({operating['neg_images']} verified negatives)",
                f"- image size: {image_size}",
                "",
                "## Detection metrics",
                "",
                "```",
                format_detection_table(det),
                "```",
                "",
                "## Alarm sweep",
                "",
                "```",
                format_alarm_table(sweep),
                "```",
                "",
                "## Recommended operating point",
                "",
                f"- confidence threshold: **{operating['threshold']:.2f}** (budget: FPR <= {args.target_fpr:.3f})",
                f"- false-alarm rate: {operating['fpr']:.4f} "
                f"({operating['false_alarm_images']}/{operating['neg_images']} negative images)",
                f"- recall at that threshold: fire {operating['recall_fire']:.4f}, "
                f"smoke {operating['recall_smoke']:.4f}",
                "",
                f"At {frames_per_second} fps that is roughly "
                f"{operating['fpr'] * frames_per_second * 3600:.0f} false boxes per hour "
                "**before** temporal confirmation. The N-of-M confirmation in "
                "`predict_video_dinov3.py` is what turns this into a usable alarm rate; "
                "requiring 6 linked hits inside 15 frames removes uncorrelated flicker.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out_dir / 'eval_report.md'}")


if __name__ == "__main__":
    main()
