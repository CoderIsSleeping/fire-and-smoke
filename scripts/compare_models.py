"""Run two (or more) models on the same video, side by side, for a demo.

Built to show a project's history honestly: "this was the first model; we
changed it to this; here is what changed". Every model sees exactly the same
frames, with the same alarm settings, and each panel shows what that model
detected, whether its alarm is up, and how long it took per frame.

    python scripts/compare_models.py \
        --model "Model 1: ViT-S + Faster R-CNN=runs/stage2/best.pt" \
        --model "Model 2: ViT-Ti + FCOS (light)=runs/light/best.pt" \
        --video site.mov --output compare.mp4 --max-seconds 120

Each model gets its own alarm threshold via LABEL=WEIGHTS@CONF. This matters:
the FCOS head scores sqrt(classification x centerness), so its confidences
cluster far lower than Faster R-CNN's -- the trained light model never scores
above ~0.75, and at a shared threshold of 0.85 it would simply never alarm.
Compare models at their own operating points (matched false-alarm rate on the
test split), not at a shared number.

Latency is measured per model on the machine running this script. On a CPU the
absolute numbers are slow; the ratio between the models is what carries over.

Inference runs every --stride frames; the frames in between reuse each model's
last result, so the output plays smoothly at the source frame rate.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from fire_smoke import CLASS_NAMES
from fire_smoke.dataset import letterbox, undo_letterbox
from fire_smoke.glow import glow_features
from fire_smoke.model import load_detector
from fire_smoke.temporal import TemporalConfirmer
from predict_video_dinov3 import LatestFrameReader, clean_boxes, draw_label, open_source, resolve_fps

BOX_COLOURS = {1: (200, 200, 200), 2: (0, 140, 255)}
ALARM_COLOUR = (0, 0, 235)
PANEL_TINTS = [(120, 70, 20), (20, 110, 40), (110, 30, 110), (30, 90, 140)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Side-by-side comparison of models on one video.")
    p.add_argument("--model", action="append", required=True, metavar="LABEL=WEIGHTS[@CONF]",
                   help="Repeat for each model, in the order to show them. LABEL is optional. "
                        "@CONF sets that model's own alarm threshold -- use each model's operating "
                        "point from its test evaluation, because different heads score on different "
                        "scales (see module docstring).")
    p.add_argument("--video", required=True, help="Video file, or a camera index such as 0 for the webcam.")
    p.add_argument("--show", action="store_true", help="Show the comparison live in a window (q to quit).")
    p.add_argument("--no-save", action="store_true", help="Do not write an output video.")
    p.add_argument("--output", default="model_comparison.mp4")
    p.add_argument("--device", default="auto")
    p.add_argument("--stride", type=int, default=5, help="Run the models every Nth frame.")
    p.add_argument("--max-seconds", type=float, default=0, help="Stop after this much video (0 = all).")
    p.add_argument("--start-seconds", type=float, default=0, help="Skip this much video first.")
    p.add_argument("--panel-width", type=int, default=960)
    p.add_argument("--layout", choices=["row", "column"], default="row")

    alarm = p.add_argument_group("alarm settings (defaults; give each model its own threshold with @CONF)")
    alarm.add_argument("--conf", type=float, default=0.85)
    alarm.add_argument("--exit-conf", type=float, default=0.70)
    alarm.add_argument("--window", type=int, default=15)
    alarm.add_argument("--enter-hits", type=int, default=6)
    alarm.add_argument("--exit-hits", type=int, default=2)
    alarm.add_argument("--show-below", type=float, default=0.30,
                       help="Draw raw detections down to this score (thin boxes), for context.")
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested if torch.cuda.is_available() or requested == "cpu" else "cpu")


def describe(model) -> str:
    """One-line architecture summary read from the checkpoint, not typed by hand."""
    cfg = model.config
    name = cfg.get("backbone_name", "")
    trunk = ("ViT-Ti" if "tiny" in name else "ViT-S+" if "small_plus" in name
             else "ViT-S" if "small" in name else "ViT-B" if "base" in name else name.split(".")[0])
    head = "FCOS" if cfg.get("head") == "fcos" else "Faster R-CNN"
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return f"DINOv3 {trunk} + {head}  |  {params:.1f}M params  |  {cfg.get('image_size', '?')}px"


class Runner:
    """One model plus its own alarm state and running statistics."""

    def __init__(self, label: str, weights: str, device, args, conf: float | None = None) -> None:
        self.label = label
        self.conf = args.conf if conf is None else conf
        self.model, checkpoint = load_detector(weights, device)
        self.model.eval()
        self.device = device
        self.size = self.model.config["image_size"]
        self.summary = describe(self.model)
        self.confirmer = TemporalConfirmer(
            class_names={1: "smoke", 2: "fire"}, window=args.window, enter_hits=args.enter_hits,
            exit_hits=args.exit_hits, enter_conf=self.conf,
            exit_conf=min(args.exit_conf, self.conf - 0.15) if conf is not None else args.exit_conf,
        )
        self.latencies: list[float] = []
        self.detections = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))
        self.alarms: list = []
        self.alarm_events = 0
        self._active: set[int] = set()
        self.epoch = checkpoint.get("epoch")

    @torch.no_grad()
    def step(self, frame_bgr: np.ndarray) -> None:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        canvas, (ratio, pad_left, pad_top) = letterbox(rgb, self.size)
        tensor = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float().div_(255.0).to(self.device)
        glow = torch.from_numpy(glow_features(canvas)[None, :]).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        detections, _ = self.model([tensor], glow=glow)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.latencies.append(time.perf_counter() - started)

        det = detections[0]
        boxes = undo_letterbox(det["boxes"].float().cpu().numpy(), ratio, pad_left, pad_top)
        scores = det["scores"].float().cpu().numpy()
        labels = det["labels"].cpu().numpy()
        boxes, scores, labels = clean_boxes(boxes, scores, labels, width, height, 4.0)
        self.detections = (boxes, scores, labels)

        self.alarms = self.confirmer.update(boxes, scores, labels)
        current = {a.track_id for a in self.alarms}
        self.alarm_events += len(current - self._active)
        self._active = current

    @property
    def mean_ms(self) -> float:
        recent = self.latencies[-30:]
        return 1000.0 * sum(recent) / len(recent) if recent else float("nan")


def render_panel(frame, runner: Runner, index: int, tint, args, panel_w: int) -> np.ndarray:
    panel = frame.copy()
    boxes, scores, labels = runner.detections
    for box, score, label in zip(boxes, scores, labels):
        if score < args.show_below:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        colour = BOX_COLOURS.get(int(label), (0, 255, 0))
        cv2.rectangle(panel, (x1, y1), (x2, y2), colour, 2)
        draw_label(panel, f"{CLASS_NAMES[int(label)]} {score:.2f}", (x1, y1 - 4), colour, 0.6)
    for alarm in runner.alarms:
        x1, y1, x2, y2 = (int(v) for v in alarm.box)
        cv2.rectangle(panel, (x1, y1), (x2, y2), ALARM_COLOUR, 5)

    scale = panel_w / panel.shape[1]
    panel = cv2.resize(panel, (panel_w, int(panel.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    h, w = panel.shape[:2]

    # Header: which model this is, read from its checkpoint.
    cv2.rectangle(panel, (0, 0), (w, 58), tint, -1)
    cv2.putText(panel, runner.label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"{runner.summary}  |  alarm at conf >= {runner.conf:.2f}", (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)

    # Footer: live speed and alarm count.
    cv2.rectangle(panel, (0, h - 34), (w, h), (0, 0, 0), -1)
    footer = (f"{runner.mean_ms:6.0f} ms/frame   "
              f"alarms so far: {runner.alarm_events}   raw boxes >= {args.show_below:.2f}: "
              f"{int((runner.detections[1] >= args.show_below).sum())}")
    cv2.putText(panel, footer, (12, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if runner.alarms:
        kinds = sorted({a.class_name.upper() for a in runner.alarms})
        text = " + ".join(kinds) + " ALARM"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        x = (w - tw) // 2
        cv2.rectangle(panel, (x - 12, 66), (x + tw + 12, 66 + th + 18), ALARM_COLOUR, -1)
        cv2.putText(panel, text, (x, 66 + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    return panel


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    runners = []
    for position, spec in enumerate(args.model, start=1):
        label, _, rest = spec.rpartition("=")
        label = label or f"Model {position}"
        conf = None
        if "@" in rest and not Path(rest).exists():
            rest, _, conf_text = rest.rpartition("@")
            conf = float(conf_text)
        weights = rest
        if not Path(weights).exists():
            raise SystemExit(f"weights not found for {label!r}: {weights}")
        runner = Runner(label, weights, device, args, conf)
        runners.append(runner)
        print(f"[{position}] {label}\n    {runner.summary}  alarm conf >= {runner.conf:.2f}  (checkpoint epoch {runner.epoch})")

    is_live = args.video.isdigit()
    capture = open_source(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = resolve_fps(capture, is_live=is_live)
    if is_live:
        # A webcam produces frames faster than two models can process them;
        # always work on the newest frame rather than falling behind.
        reader = LatestFrameReader(capture)
        start, stop = 0, float("inf")
        if args.max_seconds:
            stop = int(args.max_seconds * fps)
    else:
        reader = capture
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        start = int(args.start_seconds * fps)
        stop = start + int(args.max_seconds * fps) if args.max_seconds else total
        # Skip ahead by reading, not seeking: CCTV exports use long GOPs and a
        # seek lands on frames the decoder cannot reconstruct.
        for _ in range(start):
            if not capture.grab():
                break
    wall_start = time.time()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    index = start
    progress = tqdm(total=None if is_live else (max(0, stop - start) or None), desc="frames")

    while index < stop:
        ok, frame = reader.read()
        if not ok:
            break
        if is_live or (index - start) % args.stride == 0:
            for runner in runners:
                runner.step(frame)

        panels = [render_panel(frame, r, index, PANEL_TINTS[i % len(PANEL_TINTS)], args, args.panel_width)
                  for i, r in enumerate(runners)]
        sheet = np.hstack(panels) if args.layout == "row" else np.vstack(panels)
        if writer is None and not args.no_save:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     2.0 if is_live else fps, (sheet.shape[1], sheet.shape[0]))
        if writer is not None:
            writer.write(sheet)
        if args.show:
            cv2.imshow("Model comparison  -  press q to quit", sheet)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        index += 1
        progress.update(1)

    progress.close()
    if is_live:
        reader.release()
    else:
        capture.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    seconds = (time.time() - wall_start) if is_live else (index - start) / fps
    if not runners[0].latencies:
        raise SystemExit("no frames were processed")
    print(f"\ncompared on {seconds:.1f}s of {Path(args.video).name}, "
          f"model run every {args.stride} frames, alarm {args.enter_hits}-of-{args.window} at each model's own conf")
    header = f"{'model':<44} {'ms/frame':>9} {'alarms':>7} {'alarms/hour':>12}"
    print(header)
    print("-" * len(header))
    rows = []
    base = None
    for runner in runners:
        ms = 1000.0 * float(np.mean(runner.latencies)) if runner.latencies else float("nan")
        base = base or ms
        per_hour = runner.alarm_events / max(seconds / 3600.0, 1e-9)
        print(f"{runner.label[:44]:<44} {ms:>9.0f} {runner.alarm_events:>7d} {per_hour:>12.1f}"
              + (f"   ({base / ms:.2f}x vs first)" if ms and runner is not runners[0] else ""))
        rows.append({"model": runner.label, "architecture": runner.summary, "alarm_conf": runner.conf,
                     "ms_per_frame": round(ms, 1),
                     "alarm_events": runner.alarm_events, "alarms_per_hour": round(per_hour, 2),
                     "seconds_compared": round(seconds, 1)})

    summary_path = out_path.with_suffix(".summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    if writer is not None:
        print(f"\nwrote {out_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
