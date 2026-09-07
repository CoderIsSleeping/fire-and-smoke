"""Run the detector over a video with deployment-style alarm logic.

This is the part that makes the model usable on a camera that never turns off.
The raw per-frame detector is only the first stage; on top of it:

  * every detection must survive N-of-M temporal confirmation, linked frame to
    frame by IoU, before it becomes an alarm (see `fire_smoke/temporal.py`),
  * an optional ignore-mask suppresses regions that are legitimately hot or
    bright all day - a furnace mouth, a welding bay, a flare stack,
  * the image-level scene head runs alongside the boxes, and when it insists on
    fire while the detector sees no flame, the glow prior is used to point at
    the brightest warm region and raise a distinct, lower-severity
    "possible occluded fire" cue rather than a confirmed flame alarm.

Every state change is written to a CSV event log, which is what an operator or
a report actually wants - not 40,000 annotated frames.

Example:
    python scripts/predict_video_dinov3.py \
        --weights runs/fire_smoke/dinov3_fire_smoke/weights/best.pt \
        --source /kaggle/input/my-video/site.mp4 \
        --output /kaggle/working/site_annotated.mp4 --conf 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from fire_smoke import CLASS_NAMES
from fire_smoke.dataset import letterbox, undo_letterbox
from fire_smoke.glow import glow_features, glow_hotspot
from fire_smoke.model import load_detector
from fire_smoke.temporal import ScalarConfirmer, TemporalConfirmer

COLORS = {1: (200, 200, 200), 2: (0, 140, 255)}  # BGR
GLOW_COLOR = (0, 215, 255)
ALARM_COLOR = (0, 0, 235)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fire/smoke inference on video with temporal confirmation.")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="Video file, image folder, or camera index.")
    p.add_argument("--output", default="video_annotated.mp4")
    p.add_argument("--events", default=None, help="CSV event log (defaults next to --output).")
    p.add_argument("--imgsz", type=int, default=0, help="Override the checkpoint image size.")
    p.add_argument("--device", default="auto")
    p.add_argument("--stride", type=int, default=1, help="Run the model every Nth frame; others reuse the last result.")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--no-amp", dest="amp", action="store_false", default=True)

    alarm = p.add_argument_group("alarm logic")
    alarm.add_argument("--conf", type=float, default=0.50, help="Confidence needed to count as evidence.")
    alarm.add_argument("--exit-conf", type=float, default=0.35, help="Confidence below which evidence is ignored.")
    alarm.add_argument("--window", type=int, default=15, help="Sliding window length, in processed frames.")
    alarm.add_argument("--enter-hits", type=int, default=6, help="Hits inside the window needed to raise an alarm.")
    alarm.add_argument("--exit-hits", type=int, default=2, help="Hits needed to keep an alarm up (hysteresis).")
    alarm.add_argument("--iou-match", type=float, default=0.20, help="IoU for linking a detection to a track.")
    alarm.add_argument("--max-age", type=int, default=20, help="Frames a track survives without any evidence.")

    occluded = p.add_argument_group("occluded-fire cue")
    occluded.add_argument("--scene-conf", type=float, default=0.60, help="Scene-head probability to trust.")
    occluded.add_argument("--glow-min", type=float, default=0.35, help="Minimum glow response to localise a cue.")
    occluded.add_argument("--no-occluded-cue", dest="occluded_cue", action="store_false", default=True)

    p.add_argument("--roi-mask", default=None, help="JSON file of polygons to ignore (see the README).")
    p.add_argument("--min-box-size", type=float, default=4.0,
                   help="Discard detections thinner than this many pixels in the source frame.")
    p.add_argument("--no-boxes", action="store_true", help="Only draw confirmed alarms, not raw detections.")
    p.add_argument("--show", action="store_true",
                   help="Display a live annotated window (press q to stop). Needed for webcam testing.")
    p.add_argument("--no-save", action="store_true", help="Do not write an output video (use with --show).")
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_ignore_mask(path: str | None, width: int, height: int) -> np.ndarray | None:
    """Build a boolean ignore-mask from a polygon JSON file.

    Format:
        {"normalized": true, "ignore_polygons": [[[0.1,0.2],[0.4,0.2],[0.4,0.6]]]}

    Coordinates are either normalised to 0..1 (default) or absolute pixels.
    Anything whose detection centre lands inside a polygon is discarded, which
    is the cheapest way to live with a permanently-lit furnace in frame.
    """
    if not path:
        return None
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    polygons = spec.get("ignore_polygons", [])
    if not polygons:
        return None
    normalized = spec.get("normalized", True)

    mask = np.zeros((height, width), np.uint8)
    for polygon in polygons:
        points = np.array(polygon, np.float32)
        if normalized:
            points[:, 0] *= width
            points[:, 1] *= height
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    print(f"ignore mask: {len(polygons)} polygon(s), {100.0 * mask.mean():.1f}% of the frame suppressed")
    return mask.astype(bool)


def clean_boxes(boxes, scores, labels, width, height, min_side: float):
    """Clip detections to the frame and drop degenerate ones.

    An uncertain box head can emit sub-pixel boxes pinned to an image edge.
    They are invisible to an operator and their IoU behaviour is meaningless,
    so they must not reach the tracker and start accumulating hits.
    """
    if len(boxes) == 0:
        return boxes, scores, labels
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    keep = (boxes[:, 2] - boxes[:, 0] >= min_side) & (boxes[:, 3] - boxes[:, 1] >= min_side)
    return boxes[keep], scores[keep], labels[keep]


def filter_by_mask(boxes, scores, labels, mask):
    if mask is None or len(boxes) == 0:
        return boxes, scores, labels
    height, width = mask.shape
    cx = np.clip(((boxes[:, 0] + boxes[:, 2]) / 2).astype(int), 0, width - 1)
    cy = np.clip(((boxes[:, 1] + boxes[:, 3]) / 2).astype(int), 0, height - 1)
    keep = ~mask[cy, cx]
    return boxes[keep], scores[keep], labels[keep]


def draw_dashed_rect(image, box, color, thickness=2, dash=14):
    x1, y1, x2, y2 = (int(v) for v in box)
    for x in range(x1, x2, dash * 2):
        cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_label(image, text, origin, color, scale=0.55):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(origin[0]), int(max(th + 4, origin[1]))
    cv2.rectangle(image, (x, y - th - 4), (x + tw + 6, y + baseline), color, -1)
    cv2.putText(image, text, (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(frame, status_lines, banner):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 26 + 20 * len(status_lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, (text, color) in enumerate(status_lines):
        cv2.putText(frame, text, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    if banner:
        text, color = banner
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        x = (frame.shape[1] - tw) // 2
        cv2.rectangle(frame, (x - 14, frame.shape[0] - th - 34), (x + tw + 14, frame.shape[0] - 12), color, -1)
        cv2.putText(frame, text, (x, frame.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)


def open_source(source: str):
    """Open a file path or a camera index.

    On Windows the default backend takes several seconds to open a webcam and
    sometimes fails outright, so use DirectShow for camera indices.
    """
    if source.isdigit():
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        capture = cv2.VideoCapture(int(source), backend)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Ask the driver not to queue frames. Not all backends honour it, which
        # is why LatestFrameReader exists as well.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture
    return cv2.VideoCapture(source)


def resolve_fps(capture, is_live: bool) -> float:
    """Frame rate, with a sane fallback.

    Webcams routinely report -1 or 0 for CAP_PROP_FPS through DirectShow. A
    bare `or 25.0` does not catch -1 (it is truthy), which silently produced
    negative timestamps in the event log.
    """
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or fps > 240 or fps != fps:  # last test catches NaN
        if not is_live:
            print(f"warning: source reported fps={fps}; assuming 25.")
        return 25.0
    return float(fps)


class LatestFrameReader:
    """Background reader that always hands back the newest frame.

    Inference runs at roughly 1 fps on CPU while the camera produces 30. A
    plain read() then returns progressively staler queued frames and the
    window drifts seconds behind reality -- it looks like lag, but it is the
    detector being shown the past. This thread drains the camera continuously
    and keeps only the most recent frame, so every inference is on 'now'.
    """

    def __init__(self, capture) -> None:
        self.capture = capture
        self._lock = threading.Lock()
        self._frame = None
        self._alive = True
        self._stop = False
        self._dropped = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop:
            ok, frame = self.capture.read()
            if not ok:
                self._alive = False
                return
            with self._lock:
                if self._frame is not None:
                    self._dropped += 1
                self._frame = frame

    def read(self, timeout: float = 5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    frame, self._frame = self._frame, None
                    return True, frame
            if not self._alive:
                return False, None
            time.sleep(0.005)
        return False, None

    @property
    def dropped(self) -> int:
        return self._dropped

    def release(self) -> None:
        self._stop = True
        self._thread.join(timeout=1.0)
        self.capture.release()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    overrides = {"image_size": args.imgsz} if args.imgsz else {}
    model, _ = load_detector(args.weights, device, **overrides)
    image_size = model.config["image_size"]
    use_amp = args.amp and device.type == "cuda"

    is_live = args.source.isdigit()
    capture = open_source(args.source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    fps = resolve_fps(capture, is_live)
    if is_live:
        # Take one frame to learn the true resolution before the reader thread
        # starts; driver-reported width/height are not always what you get.
        ok, probe = capture.read()
        if not ok:
            raise SystemExit(f"Opened camera {args.source} but could not read a frame from it.")
        height, width = probe.shape[:2]
    else:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reader = LatestFrameReader(capture) if is_live else capture

    total_frames = 0 if is_live else int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames) if total_frames else args.max_frames

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = (
        None if args.no_save
        else cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    )

    events_path = Path(args.events) if args.events else output_path.with_suffix(".events.csv")
    events_file = events_path.open("w", newline="", encoding="utf-8")
    events = csv.writer(events_file)
    events.writerow(["frame", "time_s", "event", "class", "score", "x1", "y1", "x2", "y2", "detail"])

    ignore_mask = load_ignore_mask(args.roi_mask, width, height)

    confirmer = TemporalConfirmer(
        class_names={1: "smoke", 2: "fire"},
        window=args.window,
        enter_hits=args.enter_hits,
        exit_hits=args.exit_hits,
        enter_conf=args.conf,
        exit_conf=args.exit_conf,
        iou_match=args.iou_match,
        max_age=args.max_age,
    )
    scene_confirmers = {
        "smoke": ScalarConfirmer(args.window, args.enter_hits + 2, args.exit_hits + 1, args.scene_conf, args.scene_conf - 0.15),
        "fire": ScalarConfirmer(args.window, args.enter_hits + 2, args.exit_hits + 1, args.scene_conf, args.scene_conf - 0.15),
    }
    occluded_confirmer = ScalarConfirmer(args.window, args.enter_hits + 2, args.exit_hits + 1, 0.5, 0.5)

    print(f"weights   : {args.weights}")
    print(f"source    : {args.source}  ({width}x{height} @ {fps:.1f} fps)")
    print(f"alarm     : conf>={args.conf}, {args.enter_hits}-of-{args.window} confirmation, "
          f"stride {args.stride} ({args.window * args.stride / fps:.1f}s window)")
    print(f"output    : {output_path}")

    active_alarms: set[int] = set()
    active_occluded = False
    last_boxes = np.zeros((0, 4), np.float32)
    last_scores = np.zeros(0, np.float32)
    last_labels = np.zeros(0, np.int64)
    last_alarms: list = []
    last_scene = np.zeros(2, np.float32)
    last_glow: tuple | None = None

    frame_index = 0
    started_at = time.time()
    progress = tqdm(total=total_frames or None, desc="frames")

    while True:
        ok, frame = reader.read()
        if not ok:
            break
        if args.max_frames and frame_index >= args.max_frames:
            break
        # For a live camera, wall-clock time is the truth: frames are dropped
        # to stay current, so frame_index / fps would drift badly.
        timestamp = (time.time() - started_at) if is_live else frame_index / fps

        if frame_index % args.stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            canvas, (ratio, pad_left, pad_top) = letterbox(rgb, image_size)
            tensor = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float().div_(255.0).to(device)

            glow_vector = torch.from_numpy(glow_features(canvas)[None, :]).to(device)

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                detections, scene_probs = model([tensor], glow=glow_vector)

            detection = detections[0]
            boxes = detection["boxes"].float().cpu().numpy()
            scores = detection["scores"].float().cpu().numpy()
            labels = detection["labels"].cpu().numpy()
            boxes = undo_letterbox(boxes, ratio, pad_left, pad_top)
            boxes, scores, labels = clean_boxes(boxes, scores, labels, width, height, args.min_box_size)
            boxes, scores, labels = filter_by_mask(boxes, scores, labels, ignore_mask)

            last_boxes, last_scores, last_labels = boxes, scores, labels
            last_scene = scene_probs[0].float().cpu().numpy()
            last_alarms = confirmer.update(boxes, scores, labels)

            scene_state = {
                name: confirmer_.update(float(last_scene[i]))
                for i, (name, confirmer_) in enumerate(scene_confirmers.items())
            }

            # Occluded-fire cue: the scene head is convinced, the detector is
            # not, and there is a warm glow somewhere to point at.
            fire_boxes_confirmed = any(a.label == 2 for a in last_alarms)
            hotspot = None
            if args.occluded_cue and scene_state["fire"] and not fire_boxes_confirmed:
                hotspot = glow_hotspot(rgb, args.glow_min)
            occluded_now = occluded_confirmer.update(1.0 if hotspot else 0.0)
            if hotspot:
                last_glow = hotspot
            elif not occluded_now:
                last_glow = None

            current = {a.track_id for a in last_alarms}
            for alarm in last_alarms:
                if alarm.track_id not in active_alarms:
                    events.writerow([frame_index, f"{timestamp:.2f}", "ALARM_ON", alarm.class_name,
                                     f"{alarm.score:.3f}", *[int(v) for v in alarm.box],
                                     f"{alarm.hits}/{alarm.window} hits"])
            for track_id in active_alarms - current:
                events.writerow([frame_index, f"{timestamp:.2f}", "ALARM_OFF", "", "", "", "", "", "",
                                 f"track {track_id}"])
            active_alarms = current

            if occluded_now and not active_occluded:
                box = last_glow[0] if last_glow else (0, 0, 0, 0)
                events.writerow([frame_index, f"{timestamp:.2f}", "OCCLUDED_FIRE_CUE", "fire",
                                 f"{last_scene[1]:.3f}", *[int(v) for v in box],
                                 "scene head + glow prior, no visible flame"])
            elif active_occluded and not occluded_now:
                events.writerow([frame_index, f"{timestamp:.2f}", "OCCLUDED_FIRE_CLEAR", "fire", "",
                                 "", "", "", "", ""])
            active_occluded = occluded_now

        # ---- draw ----
        if not args.no_boxes:
            for box, score, label in zip(last_boxes, last_scores, last_labels):
                if score < args.exit_conf:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box)
                color = COLORS.get(int(label), (0, 255, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                draw_label(frame, f"{CLASS_NAMES[int(label)]} {score:.2f}", (x1, y1 - 4), color, 0.45)

        for alarm in last_alarms:
            x1, y1, x2, y2 = (int(v) for v in alarm.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), ALARM_COLOR, 3)
            draw_label(frame, f"CONFIRMED {alarm.class_name.upper()} {alarm.score:.2f} "
                              f"[{alarm.hits}/{alarm.window}]", (x1, y1 - 6), ALARM_COLOR, 0.6)

        if active_occluded and last_glow:
            draw_dashed_rect(frame, last_glow[0], GLOW_COLOR, 2)
            draw_label(frame, f"POSSIBLE OCCLUDED FIRE (glow {last_glow[1]:.2f})",
                       (last_glow[0][0], last_glow[0][1] - 6), GLOW_COLOR, 0.55)

        status = [
            (f"t={timestamp:6.1f}s  frame {frame_index}"
             + (f"  {frame_index / max(timestamp, 1e-6):4.1f} fps"
                f"  dropped {reader.dropped}" if is_live else ""), (255, 255, 255)),
            (f"scene  smoke {last_scene[0]:.2f}   fire {last_scene[1]:.2f}", (200, 255, 200)),
            (f"tracks {len(confirmer.tracks)}   confirmed {len(last_alarms)}", (200, 220, 255)),
        ]
        banner = None
        if any(a.label == 2 for a in last_alarms):
            banner = ("FIRE ALARM", (0, 0, 200))
        elif active_occluded:
            banner = ("POSSIBLE OCCLUDED FIRE", (0, 140, 220))
        elif any(a.label == 1 for a in last_alarms):
            banner = ("SMOKE ALARM", (90, 90, 90))
        draw_hud(frame, status, banner)

        if writer is not None:
            writer.write(frame)
        if args.show:
            cv2.imshow("fire / smoke  --  press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("stopped by user")
                break
        frame_index += 1
        progress.update(1)

    progress.close()
    reader.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    events_file.close()

    if writer is not None:
        print(f"\nwrote video : {output_path}")
    print(f"wrote events: {events_path}")


if __name__ == "__main__":
    main()
