"""Detection metrics, plus the false-alarm analysis this project actually needs.

mAP tells the report how good the detector is. It does not tell us whether the
system can be left running on a factory camera for a week without crying wolf,
which is requirement 2. For that we need a different number:

    image-level false positive rate
        = fraction of *verified negative* images (empty label file) on which the
          model emits at least one box above the operating threshold.

D-Fire ships roughly 9.8k such negatives, so we can measure it directly and then
pick the confidence threshold from the curve instead of guessing 0.25. At 5 fps,
an FPR of 1% is one false box every 20 seconds -- still unusable on its own,
which is exactly why the video layer adds temporal confirmation on top.
"""

from __future__ import annotations

import numpy as np

# 0.025 steps, not 0.05: the FCOS head compresses its scores into a narrow band,
# and at 0.05 steps a single step moved the light model's recall by 14 points,
# so the operating point it picked was far from the best one available.
DEFAULT_THRESHOLDS = tuple(round(x, 3) for x in np.arange(0.05, 0.976, 0.025))
COCO_IOUS = tuple(round(x, 2) for x in np.arange(0.5, 1.0, 0.05))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between two sets of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return (inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)).astype(np.float32)


def _ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style 101-point interpolated average precision."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    recall_points = np.linspace(0.0, 1.0, 101)
    indices = np.searchsorted(mrec, recall_points, side="left")
    values = np.where(indices < len(mpre), mpre[np.clip(indices, 0, len(mpre) - 1)], 0.0)
    return float(values.mean())


def average_precision(preds: list[dict], gts: list[dict], class_id: int, iou_threshold: float) -> tuple[float, int]:
    """AP for one class at one IoU threshold. Returns (ap, num_ground_truth)."""
    gt_boxes = []
    gt_used = []
    total_gt = 0
    for gt in gts:
        mask = gt["labels"] == class_id
        boxes = gt["boxes"][mask]
        gt_boxes.append(boxes)
        gt_used.append(np.zeros(len(boxes), bool))
        total_gt += len(boxes)

    if total_gt == 0:
        return float("nan"), 0

    rows = []
    for image_index, pred in enumerate(preds):
        mask = pred["labels"] == class_id
        for box, score in zip(pred["boxes"][mask], pred["scores"][mask]):
            rows.append((float(score), image_index, box))
    if not rows:
        return 0.0, total_gt

    rows.sort(key=lambda r: -r[0])
    tp = np.zeros(len(rows), np.float32)
    fp = np.zeros(len(rows), np.float32)

    for i, (_, image_index, box) in enumerate(rows):
        candidates = gt_boxes[image_index]
        if len(candidates) == 0:
            fp[i] = 1.0
            continue
        ious = box_iou(box[None, :], candidates)[0]
        ious[gt_used[image_index]] = -1.0
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold:
            gt_used[image_index][best] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return _ap_from_pr(recall, precision), total_gt


def detection_metrics(preds: list[dict], gts: list[dict], class_ids=(1, 2), class_names=("smoke", "fire")) -> dict:
    """AP@0.5, AP@0.75 and AP@0.5:0.95 per class, plus the means."""
    result = {"per_class": {}}
    ap50s, ap75s, apcocos = [], [], []

    for class_id, name in zip(class_ids, class_names):
        per_iou = {}
        for iou in COCO_IOUS:
            ap, n_gt = average_precision(preds, gts, class_id, iou)
            per_iou[iou] = ap
        valid = [v for v in per_iou.values() if not np.isnan(v)]
        entry = {
            "num_gt": n_gt,
            "AP50": per_iou[0.5],
            "AP75": per_iou[0.75],
            "AP50_95": float(np.mean(valid)) if valid else float("nan"),
        }
        result["per_class"][name] = entry
        if not np.isnan(entry["AP50"]):
            ap50s.append(entry["AP50"])
            ap75s.append(entry["AP75"])
            apcocos.append(entry["AP50_95"])

    result["mAP50"] = float(np.mean(ap50s)) if ap50s else 0.0
    result["mAP75"] = float(np.mean(ap75s)) if ap75s else 0.0
    result["mAP50_95"] = float(np.mean(apcocos)) if apcocos else 0.0
    return result


# Box size = sqrt(area) in pixels of a 640 px model input. The bands follow the
# project's own problem, not COCO's: an edited-in flame 35-45 px wide in a
# 1920 px CCTV frame lands at ~11-15 px here, which is where the detector's
# confidence was measured to drop below the alarm threshold.
SIZE_BANDS = (("tiny", 0, 16), ("small", 16, 32), ("medium", 32, 96), ("large", 96, float("inf")))


def box_side(boxes: np.ndarray, image_size: int = 640) -> np.ndarray:
    """sqrt(area), rescaled to a 640 px input so bands mean the same at any imgsz."""
    if len(boxes) == 0:
        return np.zeros(0, np.float32)
    area = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    return np.sqrt(area) * (640.0 / image_size)


def average_precision_in_band(preds, gts, class_id: int, iou_threshold: float, lo: float, hi: float,
                              image_size: int = 640) -> tuple[float, int]:
    """AP restricted to ground truth whose size falls in [lo, hi), COCO-style.

    Ground truth outside the band is not deleted but *ignored*: a detection
    that matches it counts as neither a hit nor a false positive, and an
    unmatched detection whose own size is outside the band is ignored too.
    Deleting it instead would turn every correct large-fire box into a false
    positive when scoring the small band.
    """
    gt_boxes, gt_ignore, gt_used, total = [], [], [], 0
    for gt in gts:
        boxes = gt["boxes"][gt["labels"] == class_id]
        side = box_side(boxes, image_size)
        ignore = (side < lo) | (side >= hi)
        order = np.argsort(ignore, kind="stable")  # in-band first, so they are matched first
        gt_boxes.append(boxes[order])
        gt_ignore.append(ignore[order])
        gt_used.append(np.zeros(len(boxes), bool))
        total += int((~ignore).sum())
    if total == 0:
        return float("nan"), 0

    rows = []
    for image_index, pred in enumerate(preds):
        mask = pred["labels"] == class_id
        for box, score in zip(pred["boxes"][mask], pred["scores"][mask]):
            rows.append((float(score), image_index, box))
    if not rows:
        return 0.0, total
    rows.sort(key=lambda r: -r[0])

    tp, fp = [], []
    for _, image_index, box in rows:
        candidates = gt_boxes[image_index]
        matched_ignored, hit = False, False
        if len(candidates):
            ious = box_iou(box[None, :], candidates)[0]
            ious[gt_used[image_index]] = -1.0
            valid = ious >= iou_threshold
            in_band = valid & ~gt_ignore[image_index]
            if in_band.any():
                best = int(np.argmax(np.where(in_band, ious, -1.0)))
                gt_used[image_index][best] = True
                hit = True
            elif valid.any():
                best = int(np.argmax(np.where(valid, ious, -1.0)))
                gt_used[image_index][best] = True
                matched_ignored = True
        if hit:
            tp.append(1.0); fp.append(0.0)
        elif matched_ignored:
            continue
        else:
            side = box_side(box[None, :], image_size)[0]
            if lo <= side < hi:
                tp.append(0.0); fp.append(1.0)
    if not tp:
        return 0.0, total
    # float32, as in average_precision: with the full size range this then
    # reproduces the standard AP exactly.
    tp_cum, fp_cum = np.cumsum(np.array(tp, np.float32)), np.cumsum(np.array(fp, np.float32))
    return _ap_from_pr(tp_cum / total, tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)), total


def size_breakdown(preds, gts, image_size: int = 640, class_ids=(1, 2), class_names=("smoke", "fire"),
                   conf_levels=(0.5, 0.8, 0.9)) -> dict:
    """Per class and size band: AP@0.5, recall at fixed confidences, and the
    median confidence the model gives those objects when it finds them."""
    out = {}
    for class_id, name in zip(class_ids, class_names):
        out[name] = {}
        for band, lo, hi in SIZE_BANDS:
            ap, n = average_precision_in_band(preds, gts, class_id, 0.5, lo, hi, image_size)
            best_scores = []
            for pred, gt in zip(preds, gts):
                boxes = gt["boxes"][gt["labels"] == class_id]
                side = box_side(boxes, image_size)
                boxes = boxes[(side >= lo) & (side < hi)]
                if len(boxes) == 0:
                    continue
                mask = pred["labels"] == class_id
                pboxes, pscores = pred["boxes"][mask], pred["scores"][mask]
                ious = box_iou(boxes, pboxes) if len(pboxes) else np.zeros((len(boxes), 0))
                for row in ious:
                    hits = pscores[row >= 0.5]
                    best_scores.append(float(hits.max()) if len(hits) else 0.0)
            scores = np.array(best_scores)
            entry = {"num_gt": n, "AP50": ap}
            for c in conf_levels:
                entry[f"recall@{c}"] = float((scores >= c).mean()) if len(scores) else float("nan")
            found = scores[scores > 0]
            entry["median_conf_when_found"] = float(np.median(found)) if len(found) else float("nan")
            out[name][band] = entry
    return out


def format_size_table(breakdown: dict, conf_levels=(0.5, 0.8, 0.9)) -> str:
    rec_cols = "".join(f" {'rec@' + str(c):>8}" for c in conf_levels)
    header = f"{'class':>6} {'size (px@640)':>14} {'#GT':>6} {'AP50':>7}{rec_cols} {'med conf':>9}"
    lines = [header, "-" * len(header)]
    bands = {b: (lo, hi) for b, lo, hi in SIZE_BANDS}
    for name, per_band in breakdown.items():
        for band, e in per_band.items():
            lo, hi = bands[band]
            label = f"{band} {int(lo)}-{int(hi)}" if hi != float("inf") else f"{band} >{int(lo)}"
            recs = "".join(f" {e[f'recall@{c}']:>8.3f}" for c in conf_levels)
            lines.append(f"{name:>6} {label:>14} {e['num_gt']:>6d} {e['AP50']:>7.4f}{recs} "
                         f"{e['median_conf_when_found']:>9.3f}")
    return "\n".join(lines)


def alarm_sweep(preds: list[dict], gts: list[dict], thresholds=DEFAULT_THRESHOLDS) -> list[dict]:
    """Image-level alarm behaviour as a function of the confidence threshold.

    `fpr` is over images that carry an empty (verified-negative) label file,
    which is the closest offline proxy we have for "camera pointed at a normal
    working scene".
    """
    is_negative = np.array([len(gt["labels"]) == 0 for gt in gts])
    has_fire = np.array([bool((gt["labels"] == 2).any()) for gt in gts])
    has_smoke = np.array([bool((gt["labels"] == 1).any()) for gt in gts])

    max_any = np.zeros(len(preds), np.float32)
    max_fire = np.zeros(len(preds), np.float32)
    max_smoke = np.zeros(len(preds), np.float32)
    for i, pred in enumerate(preds):
        scores = pred["scores"]
        labels = pred["labels"]
        if len(scores):
            max_any[i] = scores.max()
            if (labels == 2).any():
                max_fire[i] = scores[labels == 2].max()
            if (labels == 1).any():
                max_smoke[i] = scores[labels == 1].max()

    n_neg = int(is_negative.sum())
    rows = []
    for threshold in thresholds:
        alarm = max_any >= threshold
        row = {
            "threshold": float(threshold),
            "neg_images": n_neg,
            "false_alarm_images": int((alarm & is_negative).sum()),
            "fpr": float((alarm & is_negative).sum() / max(n_neg, 1)),
            "recall_fire": float(((max_fire >= threshold) & has_fire).sum() / max(int(has_fire.sum()), 1)),
            "recall_smoke": float(((max_smoke >= threshold) & has_smoke).sum() / max(int(has_smoke.sum()), 1)),
            "recall_any": float((alarm & ~is_negative).sum() / max(int((~is_negative).sum()), 1)),
        }
        rows.append(row)
    return rows


def scene_sweep(probs: np.ndarray, targets: np.ndarray, thresholds=DEFAULT_THRESHOLDS) -> dict:
    """Precision / recall / negative-FPR for the image-level classifier head."""
    out = {}
    for index, name in enumerate(("smoke", "fire")):
        p, t = probs[:, index], targets[:, index] > 0.5
        rows = []
        for threshold in thresholds:
            predicted = p >= threshold
            tp = int((predicted & t).sum())
            fp = int((predicted & ~t).sum())
            fn = int((~predicted & t).sum())
            rows.append(
                {
                    "threshold": float(threshold),
                    "precision": tp / max(tp + fp, 1),
                    "recall": tp / max(tp + fn, 1),
                    "fpr": fp / max(int((~t).sum()), 1),
                }
            )
        out[name] = rows
    return out


def pick_operating_threshold(rows: list[dict], max_fpr: float = 0.01) -> dict:
    """Lowest threshold (so highest recall) that still meets the FPR budget."""
    feasible = [r for r in rows if r["fpr"] <= max_fpr]
    if feasible:
        return min(feasible, key=lambda r: r["threshold"])
    return max(rows, key=lambda r: r["threshold"])


def format_alarm_table(rows: list[dict]) -> str:
    header = f"{'conf':>6} {'FP imgs':>8} {'FPR':>8} {'rec fire':>9} {'rec smoke':>10} {'rec any':>8}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['threshold']:>6.2f} {r['false_alarm_images']:>8d} {r['fpr']:>8.4f} "
            f"{r['recall_fire']:>9.4f} {r['recall_smoke']:>10.4f} {r['recall_any']:>8.4f}"
        )
    return "\n".join(lines)


def format_detection_table(metrics: dict) -> str:
    header = f"{'class':>8} {'#GT':>7} {'AP50':>8} {'AP75':>8} {'AP50-95':>9}"
    lines = [header, "-" * len(header)]
    for name, entry in metrics["per_class"].items():
        lines.append(
            f"{name:>8} {entry['num_gt']:>7d} {entry['AP50']:>8.4f} "
            f"{entry['AP75']:>8.4f} {entry['AP50_95']:>9.4f}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'mean':>8} {'':>7} {metrics['mAP50']:>8.4f} {metrics['mAP75']:>8.4f} {metrics['mAP50_95']:>9.4f}"
    )
    return "\n".join(lines)
