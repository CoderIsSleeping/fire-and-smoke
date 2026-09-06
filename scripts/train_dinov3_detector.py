"""Train the DINOv3 fire/smoke detector.

Stage 1 (what this script does by default):
    frozen DINOv3 ViT-S/16 trunk + trainable feature pyramid + Faster R-CNN head
    + image-level scene classifier, trained on D-Fire with low-light, IR and
    flame-occlusion augmentation.

Model selection is by validation mAP@0.5, not by loss, and every epoch also
logs the image-level false-alarm rate on the verified-negative images so the
false-positive requirement is visible while training rather than discovered
afterwards.

Typical Kaggle run:
    python scripts/train_dinov3_detector.py --epochs 40 --imgsz 640 --batch 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fire_smoke import CLASS_NAMES
from fire_smoke.backbone import DEFAULT_BACKBONE
from fire_smoke.dataset import (
    AugmentConfig,
    FireSmokeDataset,
    collate_fn,
    describe_split,
    find_data_yaml,
)
from fire_smoke.metrics import (
    alarm_sweep,
    detection_metrics,
    format_alarm_table,
    format_detection_table,
    pick_operating_threshold,
    scene_sweep,
)
from fire_smoke.model import FireSmokeDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the DINOv3 fire/smoke detector.")

    data = p.add_argument_group("data")
    data.add_argument("--data", default=None, help="Path to data.yaml (auto-detected under /kaggle/input if omitted).")
    data.add_argument("--imgsz", type=int, default=640, help="Letterbox size; must be a multiple of 64.")
    data.add_argument("--max-train-images", type=int, default=0, help="Subsample the train split (0 = all). For smoke tests.")
    data.add_argument("--max-val-images", type=int, default=0, help="Subsample the val split (0 = all).")

    model = p.add_argument_group("model")
    model.add_argument("--backbone", default=DEFAULT_BACKBONE)
    model.add_argument("--backbone-weights", default=None, help="Local trunk checkpoint, for offline Kaggle sessions.")
    model.add_argument("--unfreeze-last-n", type=int, default=0, help="Unfreeze the last N trunk blocks (stage 2).")
    model.add_argument("--scene-weight", type=float, default=1.0, help="Weight of the image-level classifier loss.")
    model.add_argument("--fpn-channels", type=int, default=256)

    optim = p.add_argument_group("optimisation")
    optim.add_argument("--epochs", type=int, default=40)
    optim.add_argument("--batch", type=int, default=8)
    optim.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps.")
    optim.add_argument("--lr", type=float, default=1e-4)
    optim.add_argument("--trunk-lr-scale", type=float, default=0.05)
    optim.add_argument("--weight-decay", type=float, default=1e-4)
    optim.add_argument("--warmup-iters", type=int, default=500)
    optim.add_argument("--clip-grad", type=float, default=10.0)
    optim.add_argument("--patience", type=int, default=8, help="Early stop after N epochs with no mAP50 gain.")

    aug = p.add_argument_group("augmentation")
    aug.add_argument("--night-prob", type=float, default=0.35, help="Low-light simulation probability.")
    aug.add_argument("--gray-prob", type=float, default=0.10, help="IR / night-mode (colour removed) probability.")
    aug.add_argument("--occlude-prob", type=float, default=0.25, help="Hide a flame behind an obstacle, keep the label.")
    aug.add_argument("--crop-prob", type=float, default=0.80)
    aug.add_argument("--jitter-prob", type=float, default=0.80)
    aug.add_argument("--blur-prob", type=float, default=0.10)
    aug.add_argument("--no-augment", action="store_true", help="Disable all augmentation (ablation baseline).")

    run = p.add_argument_group("run")
    run.add_argument("--device", default="auto")
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--amp", dest="amp", action="store_true", default=None)
    run.add_argument("--no-amp", dest="amp", action="store_false")
    run.add_argument("--output", default=None, help="Run directory (defaults to a Kaggle-aware path).")
    run.add_argument("--name", default="dinov3_fire_smoke")
    run.add_argument("--resume", default=None,
                     help="Continue an interrupted run from last.pt (restores optimizer, epoch and history).")
    run.add_argument("--init-from", default=None,
                     help="Start a fresh run from another checkpoint's weights (for stage-2 fine-tuning).")
    run.add_argument("--target-fpr", type=float, default=0.01, help="False-alarm budget used to report an operating threshold.")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--zip", action="store_true", help="Zip the run directory when finished (handy on Kaggle).")
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("warning: CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def default_output_dir(name: str) -> Path:
    base = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("runs")
    return base / "fire_smoke" / name


def build_augment_config(args: argparse.Namespace) -> AugmentConfig:
    if args.no_augment:
        return AugmentConfig.disabled()
    return AugmentConfig(
        hflip=0.5,
        crop=args.crop_prob,
        jitter=args.jitter_prob,
        night=args.night_prob,
        gray=args.gray_prob,
        blur=args.blur_prob,
        occlude_fire=args.occlude_prob,
    )


def lr_factor(step: int, warmup: int, total: int, floor: float = 0.01) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train_one_epoch(model, loader, optimizer, scaler, device, args, global_step, total_steps) -> tuple[dict, int]:
    model.train()
    sums: dict[str, float] = {}
    batches = 0
    skipped = 0
    optimizer.zero_grad(set_to_none=True)
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    progress = tqdm(loader, desc="train", leave=False)
    for step, (images, targets, scene, glow) in enumerate(progress):
        factor = lr_factor(global_step, args.warmup_iters, total_steps)
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * factor

        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        scene = scene.to(device, non_blocking=True)
        glow = glow.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            losses = model(images, targets, scene, glow)
            loss = sum(losses.values())

        if not torch.isfinite(loss):
            # fp16 occasionally blows up an RPN regression term; drop the batch
            # rather than poisoning the weights.
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            continue

        scaler.scale(loss / args.accum).backward()

        if (step + 1) % args.accum == 0:
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.clip_grad
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batches += 1
        global_step += 1
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach())
        sums["total"] = sums.get("total", 0.0) + float(loss.detach())
        progress.set_postfix(loss=sums["total"] / batches, lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base

    averaged = {k: v / max(batches, 1) for k, v in sums.items()}
    averaged["skipped_batches"] = skipped
    return averaged, global_step


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray]:
    model.eval()
    preds, gts, scene_probs, scene_targets = [], [], [], []

    for images, targets, scene, glow in tqdm(loader, desc="val", leave=False):
        images = [img.to(device, non_blocking=True) for img in images]
        glow_device = glow.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            detections, probs = model(images, glow=glow_device)

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


def write_history(history: list[dict], out_dir: Path) -> None:
    if not history:
        return
    fields = sorted({key for row in history for key in row})
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0][0]
    for key in ("train_total", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", "loss_scene"):
        values = [row.get(key) for row in history]
        if any(v is not None for v in values):
            ax.plot(epochs, values, label=key)
    ax.set_title("Training losses")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0][1]
    ax.plot(epochs, [row.get("mAP50") for row in history], label="mAP@0.5", marker="o")
    ax.plot(epochs, [row.get("mAP50_95") for row in history], label="mAP@0.5:0.95", marker="s")
    ax.plot(epochs, [row.get("AP50_fire") for row in history], label="AP50 fire", linestyle="--")
    ax.plot(epochs, [row.get("AP50_smoke") for row in history], label="AP50 smoke", linestyle="--")
    ax.set_title("Validation detection quality")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1][0]
    ax.plot(epochs, [row.get("fpr_at_op") for row in history], label="false-alarm rate (negatives)", color="crimson", marker="o")
    ax.plot(epochs, [row.get("recall_any_at_op") for row in history], label="image recall", color="seagreen", marker="s")
    ax.set_title("Alarm behaviour at the operating threshold")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1][1]
    ax.plot(epochs, [row.get("scene_ap_fire") for row in history], label="scene recall @0.5 fire", marker="o")
    ax.plot(epochs, [row.get("scene_ap_smoke") for row in history], label="scene recall @0.5 smoke", marker="s")
    ax.plot(epochs, [row.get("scene_fpr_fire") for row in history], label="scene FPR fire", linestyle="--", color="crimson")
    ax.set_title("Image-level classifier head")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("DINOv3 fire / smoke detector")
    fig.tight_layout()
    fig.savefig(out_dir / "results.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.imgsz % 64 != 0:
        raise SystemExit(f"--imgsz must be a multiple of 64 (got {args.imgsz}); try 576, 640 or 704.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    use_amp = args.amp if args.amp is not None else device.type == "cuda"
    data_yaml = find_data_yaml(args.data)
    out_dir = Path(args.output) if args.output else default_output_dir(args.name)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"data      : {data_yaml}")
    print(f"device    : {device} (amp={use_amp})")
    print(f"backbone  : {args.backbone} (unfreeze_last_n={args.unfreeze_last_n})")
    print(f"output    : {out_dir}")

    augment_cfg = build_augment_config(args)
    train_ds = FireSmokeDataset(data_yaml, "train", args.imgsz, augment_cfg,
                                max_images=args.max_train_images or None, seed=args.seed)
    val_ds = FireSmokeDataset(data_yaml, "val", args.imgsz, AugmentConfig.disabled(),
                              max_images=args.max_val_images or None, seed=args.seed)

    print("\ndataset summary")
    summary = {}
    for name, dataset in (("train", train_ds), ("val", val_ds)):
        summary[name] = describe_split(dataset)
        print(f"  {name}: {summary[name]}")
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        collate_fn=collate_fn, pin_memory=device.type == "cuda", drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, args.batch), shuffle=False, num_workers=args.workers,
        collate_fn=collate_fn, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    model = FireSmokeDetector(
        backbone_name=args.backbone,
        image_size=args.imgsz,
        pretrained_backbone=True,
        unfreeze_last_n=args.unfreeze_last_n,
        backbone_weights=args.backbone_weights,
        scene_weight=args.scene_weight,
        fpn_channels=args.fpn_channels,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nparameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable\n")

    optimizer = torch.optim.AdamW(model.param_groups(args.lr, args.trunk_lr_scale, args.weight_decay))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 1
    global_step = 0
    best_map = -1.0
    stale = 0
    history: list[dict] = []

    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        print(f"initialised weights from {args.init_from} "
              f"(epoch counter, optimizer and history start fresh)")
        if missing or unexpected:
            print(f"  {len(missing)} missing / {len(unexpected)} unexpected keys")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and use_amp:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_map = float(checkpoint.get("best_map", -1.0))
        history = list(checkpoint.get("history", []))
        print(f"resumed from {args.resume} at epoch {start_epoch} (best mAP50={best_map:.4f})")

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        train_stats, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args, global_step, total_steps
        )

        preds, gts, scene_probs, scene_targets = evaluate(model, val_loader, device, use_amp)
        det = detection_metrics(preds, gts)
        sweep = alarm_sweep(preds, gts)
        operating = pick_operating_threshold(sweep, args.target_fpr)
        scene = scene_sweep(scene_probs, scene_targets)

        def scene_at(name: str, key: str, threshold: float = 0.5) -> float:
            row = min(scene[name], key=lambda r: abs(r["threshold"] - threshold))
            return row[key]

        row = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "train_total": round(train_stats.get("total", 0.0), 5),
            "skipped_batches": train_stats.get("skipped_batches", 0),
            "mAP50": round(det["mAP50"], 5),
            "mAP50_95": round(det["mAP50_95"], 5),
            "AP50_fire": round(det["per_class"]["fire"]["AP50"], 5),
            "AP50_smoke": round(det["per_class"]["smoke"]["AP50"], 5),
            "op_threshold": operating["threshold"],
            "fpr_at_op": round(operating["fpr"], 5),
            "recall_any_at_op": round(operating["recall_any"], 5),
            "recall_fire_at_op": round(operating["recall_fire"], 5),
            "recall_smoke_at_op": round(operating["recall_smoke"], 5),
            "scene_ap_fire": round(scene_at("fire", "recall"), 5),
            "scene_ap_smoke": round(scene_at("smoke", "recall"), 5),
            "scene_fpr_fire": round(scene_at("fire", "fpr"), 5),
            "scene_fpr_smoke": round(scene_at("smoke", "fpr"), 5),
        }
        for key in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", "loss_scene"):
            if key in train_stats:
                row[key] = round(train_stats[key], 5)
        history.append(row)
        write_history(history, out_dir)

        print(f"\nepoch {epoch}/{args.epochs}  ({row['seconds']}s)")
        print(format_detection_table(det))
        print(
            f"operating point (FPR budget {args.target_fpr:.3f}): conf>={operating['threshold']:.2f}  "
            f"false-alarm rate {operating['fpr']:.4f} on {operating['neg_images']} negatives, "
            f"image recall {operating['recall_any']:.4f}"
        )

        metadata = {
            "epoch": epoch,
            "global_step": global_step,
            "history": history,
            "best_map": max(best_map, det["mAP50"]),
            "args": vars(args),
        }
        # last.pt is the resume point, so it carries the optimizer state.
        model.save(
            weights_dir / "last.pt",
            metadata | {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()},
        )

        if det["mAP50"] > best_map:
            best_map = det["mAP50"]
            stale = 0
            # best.pt is the deployment artifact and gets downloaded off Kaggle,
            # so it stays weights-only - roughly half the size of last.pt.
            model.save(weights_dir / "best.pt", metadata | {"best_map": best_map})
            (out_dir / "best_metrics.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "detection": det,
                        "alarm_sweep": sweep,
                        "operating_point": operating,
                        "scene": scene,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"new best mAP50 = {best_map:.4f}  (weights saved)")
        else:
            stale += 1
            print(f"no improvement for {stale}/{args.patience} epochs (best mAP50 = {best_map:.4f})")
            if stale >= args.patience:
                print("early stopping")
                break

    print("\nfinal validation alarm sweep")
    print(format_alarm_table(sweep))

    summary_path = out_dir / "training_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Training summary",
                "",
                f"- data: `{data_yaml}`",
                f"- backbone: `{args.backbone}` (unfrozen blocks: {args.unfreeze_last_n})",
                f"- image size: {args.imgsz}, batch {args.batch} x accum {args.accum}",
                f"- epochs run: {len(history)}",
                f"- best val mAP@0.5: {best_map:.4f}",
                "",
                "## Validation detection metrics (best epoch)",
                "",
                "```",
                format_detection_table(det),
                "```",
                "",
                "## Alarm sweep on the validation split",
                "",
                "```",
                format_alarm_table(sweep),
                "```",
                "",
                "`FPR` is the fraction of verified-negative images that produced at least one",
                "box at that confidence. Multiply by the frame rate to get false boxes per",
                "second before temporal confirmation.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {summary_path}")
    print(f"best weights: {weights_dir / 'best.pt'}")

    if args.zip:
        archive = shutil.make_archive(str(out_dir.parent / f"{args.name}_results"), "zip", str(out_dir))
        print(f"zipped run: {archive}")


if __name__ == "__main__":
    main()
