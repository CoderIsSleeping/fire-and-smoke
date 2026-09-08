"""Measure the *full pipeline* -- detector plus temporal confirmation -- on video.

Everything reported so far has been per-frame on still images. That cannot
answer the two questions that decide whether this system is deployable:

    On footage of things that merely look like fire, how often does the
    complete pipeline actually raise an alarm?

    When there is a real fire, how many seconds until it does?

This script answers both, over a folder of videos.

## Two passes, and why

Inference is expensive; the alarm logic is free. So pass 1 runs the model once
per video at a deliberately low threshold and caches every raw detection to
disk. Pass 2 replays the temporal confirmer over that cache for a whole grid of
(confidence, enter_hits) settings.

The consequence is that you can retune the alarm parameters in seconds instead
of re-running hours of inference -- and the operating point you pick is chosen
against measured video behaviour rather than guessed from a still-image curve.

## Naming convention

Videos are classified by filename prefix. D-Fire ships 50 `FP*.mp4` and 50
`VP*.mp4`; FP appears to mean fire-like footage containing no real fire, and VP
footage that does. **Verify that on a couple of clips before trusting the
numbers** -- the D-Fire README does not define the abbreviations, so this is
inferred. Override with --negative-prefix / --positive-prefix, or use
--labels to point at a JSON mapping filenames to "positive"/"negative".

Example:
    python scripts/eval_videos_dinov3.py --weights best.pt --videos /kaggle/input/dfire-videos --stride 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from fire_smoke.dataset import letterbox, undo_letterbox
from fire_smoke.glow import glow_features
from fire_smoke.model import load_detector
from fire_smoke.temporal import TemporalConfirmer

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
# Cache detections down to this score so pass 2 can explore any threshold above it.
CACHE_FLOOR = 0.15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate detector + temporal confirmation on a folder of videos.")
    p.add_argument("--weights", required=True)
    p.add_argument("--videos", required=True, help="Folder of videos (searched recursively).")
    p.add_argument("--output", default=None, help="Report directory (default: alongside the weights).")
    p.add_argument("--cache", default=None, help="Detection cache file (default: <output>/detections.npz).")
    p.add_argument("--reuse-cache", action="store_true", help="Skip inference and replay an existing cache.")

    p.add_argument("--imgsz", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--stride", type=int, default=1, help="Run the model every Nth frame.")
    p.add_argument("--max-seconds", type=float, default=0, help="Cap each video (0 = whole video).")
    p.add_argument("--min-box-size", type=float, default=4.0)
    p.add_argument("--no-amp", dest="amp", action="store_false", default=True)

    p.add_argument("--negative-prefix", default="FP", help="Filename prefix for no-fire footage.")
    p.add_argument("--positive-prefix", default="VP", help="Filename prefix for real-fire footage.")
    p.add_argument("--positive-dirs", default="positive",
                   help="Comma-separated folder names holding real-fire footage.")
    p.add_argument("--negative-dirs", default="negative",
                   help="Comma-separated folder names holding no-fire footage.")
    p.add_argument("--labels", default=None,
                   help='JSON: {"clip.mp4": "positive", ...}. Overrides folders and prefixes.')

    p.add_argument("--conf-grid", default="0.5,0.6,0.7,0.8,0.85,0.9,0.95")
    p.add_argument("--hits-grid", default="3,4,6,8")
    p.add_argument("--window", type=int, default=15)
    p.add_argument("--exit-hits", type=int, default=2)
    p.add_argument("--iou-match", type=float, default=0.20)
    p.add_argument("--max-age", type=int, default=20)
    p.add_argument("--target-video-fpr", type=float, default=0.05,
                   help="Budget: fraction of no-fire videos allowed to alarm at all.")
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("warning: CUDA requested but unavailable; using CPU. This will be slow.")
        return torch.device("cpu")
    return torch.device(requested)


def label_for(path: Path, root: Path, args, overrides: dict) -> str | None:
    """Decide whether a clip contains real fire.

    Three mechanisms, most explicit first:
      1. an entry in the --labels JSON,
      2. a parent folder named in --positive-dirs / --negative-dirs, which is
         how datasets that ship as separate zips (KMU, for instance) naturally
         extract,
      3. a filename prefix, which is how D-Fire names its FP*/VP* clips.
    """
    relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
    if relative in overrides:
        return overrides[relative]
    if path.name in overrides:
        return overrides[path.name]

    positive_dirs = {d.strip().lower() for d in args.positive_dirs.split(",") if d.strip()}
    negative_dirs = {d.strip().lower() for d in args.negative_dirs.split(",") if d.strip()}
    try:
        parts = {p.lower() for p in path.relative_to(root).parts[:-1]}
    except ValueError:
        parts = set()
    if parts & negative_dirs:
        return "negative"
    if parts & positive_dirs:
        return "positive"

    stem = path.name.upper()
    if stem.startswith(args.positive_prefix.upper()):
        return "positive"
    if stem.startswith(args.negative_prefix.upper()):
        return "negative"
    return None


@torch.no_grad()
def cache_detections(model, video_path: Path, device, args, image_size: int) -> dict:
    """Run the model over one video, keeping every detection above CACHE_FLOOR."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {"error": "could not open"}

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240 or fps != fps:
        fps = 25.0
    max_frames = int(args.max_seconds * fps) if args.max_seconds else 0

    frames, boxes, scores, labels, scene = [], [], [], [], []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok or (max_frames and index >= max_frames):
            break
        if index % args.stride == 0:
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            canvas, (ratio, pad_left, pad_top) = letterbox(rgb, image_size)
            tensor = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float().div_(255.0).to(device)
            glow = torch.from_numpy(glow_features(canvas)[None, :]).to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=args.amp and device.type == "cuda"):
                detections, probs = model([tensor], glow=glow)

            det = detections[0]
            b = undo_letterbox(det["boxes"].float().cpu().numpy(), ratio, pad_left, pad_top)
            s = det["scores"].float().cpu().numpy()
            l = det["labels"].cpu().numpy()

            keep = s >= CACHE_FLOOR
            b, s, l = b[keep], s[keep], l[keep]
            if len(b):
                b[:, [0, 2]] = b[:, [0, 2]].clip(0, width)
                b[:, [1, 3]] = b[:, [1, 3]].clip(0, height)
                good = ((b[:, 2] - b[:, 0]) >= args.min_box_size) & ((b[:, 3] - b[:, 1]) >= args.min_box_size)
                b, s, l = b[good], s[good], l[good]

            frames.append(index)
            boxes.append(b.astype(np.float32))
            scores.append(s.astype(np.float32))
            labels.append(l.astype(np.int16))
            scene.append(probs[0].float().cpu().numpy())
        index += 1

    capture.release()
    return {
        "fps": float(fps),
        "source_frames": index,
        "duration_s": index / fps,
        "processed_frames": len(frames),
        "frames": np.array(frames, np.int32),
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "scene": np.array(scene, np.float32) if scene else np.zeros((0, 2), np.float32),
    }


def replay(record: dict, conf: float, enter_hits: int, args) -> dict:
    """Replay the alarm logic over cached detections. No inference."""
    confirmer = TemporalConfirmer(
        class_names={1: "smoke", 2: "fire"},
        window=args.window,
        enter_hits=enter_hits,
        exit_hits=args.exit_hits,
        enter_conf=conf,
        exit_conf=max(conf - 0.15, 0.05),
        iou_match=args.iou_match,
        max_age=args.max_age,
    )

    effective_fps = record["fps"] / args.stride
    first_alarm_frame = None
    alarm_frames = 0
    for i in range(record["processed_frames"]):
        alarms = confirmer.update(record["boxes"][i], record["scores"][i], record["labels"][i])
        if alarms:
            alarm_frames += 1
            if first_alarm_frame is None:
                first_alarm_frame = i

    return {
        "alarmed": first_alarm_frame is not None,
        "time_to_alarm_s": None if first_alarm_frame is None else first_alarm_frame / max(effective_fps, 1e-6),
        "alarm_seconds": alarm_frames / max(effective_fps, 1e-6),
        "duration_s": record["duration_s"],
    }


def format_grid(rows: list[dict]) -> str:
    header = (f"{'conf':>5} {'hits':>5} {'FP vids':>8} {'video FPR':>10} "
              f"{'false alarm s/h':>16} {'VP vids':>8} {'detect rate':>12} {'median TTA':>11}")
    lines = [header, "-" * len(header)]
    for r in rows:
        tta = "-" if r["median_time_to_alarm_s"] is None else f"{r['median_time_to_alarm_s']:.1f}s"
        lines.append(
            f"{r['conf']:>5.2f} {r['enter_hits']:>5d} "
            f"{r['negatives_alarmed']:>3d}/{r['negatives']:<4d} {r['video_fpr']:>10.4f} "
            f"{r['false_alarm_seconds_per_hour']:>16.1f} "
            f"{r['positives_alarmed']:>3d}/{r['positives']:<4d} {r['detect_rate']:>12.4f} {tta:>11}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    video_root = Path(args.videos)
    videos = sorted(p for p in video_root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise SystemExit(f"No videos found under {video_root}")

    out_dir = Path(args.output) if args.output else Path(args.weights).parent.parent / "eval_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else out_dir / "detections.npz"

    overrides = json.loads(Path(args.labels).read_text(encoding="utf-8")) if args.labels else {}
    def key_of(v: Path) -> str:
        return v.relative_to(video_root).as_posix()

    labelled = {key_of(v): label_for(v, video_root, args, overrides) for v in videos}
    if len(labelled) != len(videos):
        raise SystemExit("internal error: video keys collided")
    n_pos = sum(1 for x in labelled.values() if x == "positive")
    n_neg = sum(1 for x in labelled.values() if x == "negative")
    n_unknown = sum(1 for x in labelled.values() if x is None)

    print(f"videos    : {len(videos)} under {video_root}")
    print(f"labelled  : {n_pos} with fire, {n_neg} without"
          + (f", {n_unknown} unlabelled (ignored in the summary)" if n_unknown else ""))
    if n_pos == 0 and n_neg == 0:
        print("warning: nothing was labelled. Put clips under positive/ and negative/ folders,")
        print("         or pass --labels, --positive-prefix / --negative-prefix.")

    # ---- pass 1: inference ----
    if args.reuse_cache and cache_path.exists():
        print(f"reusing cache: {cache_path}")
        blob = np.load(cache_path, allow_pickle=True)
        records = blob["records"].item()
    else:
        model, checkpoint = load_detector(args.weights, device)
        image_size = model.config["image_size"]
        print(f"weights   : {args.weights} (epoch {checkpoint.get('epoch', '?')})")
        print(f"device    : {device}, image size {image_size}, stride {args.stride}\n")

        records = {}
        started = time.time()
        for video in tqdm(videos, desc="videos"):
            record = cache_detections(model, video, device, args, image_size)
            if "error" in record:
                print(f"  skipped {key_of(video)}: {record['error']}")
                continue
            records[key_of(video)] = record
        total_frames = sum(r["processed_frames"] for r in records.values())
        elapsed = time.time() - started
        print(f"\ninference: {total_frames} frames in {elapsed/60:.1f} min "
              f"({total_frames/max(elapsed, 1e-6):.1f} fps)")
        np.savez_compressed(cache_path, records=np.array(records, dtype=object))
        print(f"cached detections -> {cache_path}")

    # ---- pass 2: replay the alarm logic over the grid ----
    conf_grid = [float(x) for x in args.conf_grid.split(",")]
    hits_grid = [int(x) for x in args.hits_grid.split(",")]

    rows = []
    per_video_best = {}
    for conf in conf_grid:
        for hits in hits_grid:
            neg_alarmed = pos_alarmed = 0
            neg_total = pos_total = 0
            false_alarm_seconds = 0.0
            negative_hours = 0.0
            times_to_alarm = []
            details = {}

            for name, record in records.items():
                kind = labelled.get(name)
                result = replay(record, conf, hits, args)
                details[name] = result
                if kind == "negative":
                    neg_total += 1
                    negative_hours += result["duration_s"] / 3600.0
                    if result["alarmed"]:
                        neg_alarmed += 1
                        false_alarm_seconds += result["alarm_seconds"]
                elif kind == "positive":
                    pos_total += 1
                    if result["alarmed"]:
                        pos_alarmed += 1
                        times_to_alarm.append(result["time_to_alarm_s"])

            row = {
                "conf": conf,
                "enter_hits": hits,
                "negatives": neg_total,
                "negatives_alarmed": neg_alarmed,
                "video_fpr": neg_alarmed / max(neg_total, 1),
                "false_alarm_seconds_per_hour": false_alarm_seconds / max(negative_hours, 1e-6),
                "positives": pos_total,
                "positives_alarmed": pos_alarmed,
                "detect_rate": pos_alarmed / max(pos_total, 1),
                "median_time_to_alarm_s": float(np.median(times_to_alarm)) if times_to_alarm else None,
            }
            rows.append(row)
            per_video_best[f"conf{conf}_hits{hits}"] = details

    print("\n" + format_grid(rows))

    # A setting that never alarms trivially satisfies any FPR budget, so it must
    # not be recommended -- require it to actually detect something.
    feasible = [
        r for r in rows
        if r["video_fpr"] <= args.target_video_fpr and r["positives"] > 0 and r["detect_rate"] > 0
    ]
    best = max(feasible, key=lambda r: (r["detect_rate"], -r["video_fpr"])) if feasible else None
    print()
    if not any(r["detect_rate"] > 0 for r in rows):
        print("The model raised no alarm on any fire video at any setting in the grid.")
        print("Lower --conf-grid / --hits-grid, or check that the positive videos really contain fire.")
    elif best:
        tta = "n/a" if best["median_time_to_alarm_s"] is None else f"{best['median_time_to_alarm_s']:.1f}s"
        print(f"Best setting within a {args.target_video_fpr:.0%} video-FPR budget: "
              f"conf {best['conf']:.2f}, {best['enter_hits']}-of-{args.window}")
        print(f"  detects {best['detect_rate']:.1%} of fire videos, median time to alarm {tta}")
        print(f"  {best['negatives_alarmed']}/{best['negatives']} no-fire videos alarmed "
              f"({best['false_alarm_seconds_per_hour']:.1f} false-alarm seconds per hour of footage)")
    else:
        print(f"No setting met the {args.target_video_fpr:.0%} video-FPR budget. "
              "Raise --conf-grid, raise --hits-grid, or use an ignore-mask on the offenders.")

    report = {
        "weights": str(args.weights),
        "videos": str(video_root),
        "num_videos": len(records),
        "labels": labelled,
        "stride": args.stride,
        "window": args.window,
        "grid": rows,
        "recommended": best,
    }
    (out_dir / "video_eval.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "video_eval.md").write_text(
        "\n".join([
            "# Video evaluation - detector + temporal confirmation",
            "",
            f"- weights: `{args.weights}`",
            f"- videos: {len(records)} ({n_pos} with fire, {n_neg} without)",
            f"- frame stride: {args.stride}, confirmation window: {args.window} processed frames",
            "",
            "`video FPR` is the fraction of no-fire videos that raised **any** alarm.",
            "`false alarm s/h` is how many seconds per hour of no-fire footage the system",
            "spends in the alarm state - the number an operator actually experiences.",
            "`median TTA` is the median time from the start of a fire video to the alarm.",
            "",
            "```",
            format_grid(rows),
            "```",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"\nwrote {out_dir / 'video_eval.md'}")


if __name__ == "__main__":
    main()
