"""Score a model on video clips with known fires: per-fire confidence, time to
alarm, and false alarms everywhere else.

The D-Fire test split measures still photos. The failure that matters on site
is different: a small or distant flame on a fixed CCTV camera that the model
sees but never scores high enough to confirm. This benchmark measures exactly
that, per fire, so a change (new weights, --scales, a threshold) can be judged
with numbers instead of by watching the video.

Ground truth is a JSON file:

    {"clips": [{"video": "path.mp4",
                "fires": [{"id": "...", "t0": 2.0, "t1": 6.0, "box": [x1, y1, x2, y2]}]}]}

with boxes in original-frame pixels. A detection belongs to a fire if its
centre lies inside the fire's box grown by `--match-margin` times its size
(edited-in flames flicker, so the box moves). An alarm is a false alarm if it
starts outside every fire's time window and region.

    python scripts/eval_fire_clips.py --weights models/model1_vitS_fasterrcnn.pt \\
        --gt Industry/test/fire_clips_gt.json --conf 0.80 --scales 640,960
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import torch
from tqdm import tqdm

from fire_smoke.model import load_detector
from fire_smoke.temporal import TemporalConfirmer
from predict_video_dinov3 import clean_boxes, detect_multiscale


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-fire scores and time-to-alarm on labelled video clips.")
    p.add_argument("--weights", required=True)
    p.add_argument("--gt", required=True, help="Ground-truth JSON (see module docstring).")
    p.add_argument("--scales", default=None, help="Comma-separated input sizes, e.g. 640,960.")
    p.add_argument("--conf", default="0.80", help="Alarm threshold(s), comma-separated: each gets its own "
                                                  "6-of-15 confirmation, from one pass over the video.")
    p.add_argument("--window", type=int, default=15)
    p.add_argument("--enter-hits", type=int, default=6)
    p.add_argument("--stride", type=int, default=5, help="Run the model every Nth frame.")
    p.add_argument("--match-margin", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None, help="Write the full result as JSON here.")
    return p.parse_args()


def inside(fire: dict, box: np.ndarray, margin: float) -> bool:
    x1, y1, x2, y2 = fire["box"]
    grow = margin * max(x2 - x1, y2 - y1)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return x1 - grow <= cx <= x2 + grow and y1 - grow <= cy <= y2 + grow


def run_clip(model, clip: dict, args, device, scales) -> dict:
    cap = cv2.VideoCapture(clip["video"])
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip['video']}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    confs = args.confs
    confirmers = {c: TemporalConfirmer({1: "smoke", 2: "fire"}, window=args.window, enter_hits=args.enter_hits,
                                       exit_hits=2, enter_conf=c, exit_conf=c - 0.15) for c in confs}
    fires = clip["fires"]
    per_fire = {f["id"]: {"scores": [], "alarm_t": {c: None for c in confs}} for f in fires}
    false_alarms = {c: [] for c in confs}
    active = {c: set() for c in confs}

    index = 0
    progress = tqdm(total=total, desc=Path(clip["video"]).name, leave=False)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % args.stride == 0:
            t = index / fps
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, scores, labels, _ = detect_multiscale(model, rgb, scales, device)
            boxes, scores, labels = clean_boxes(boxes, scores, labels, w, h, 4.0)

            for f in fires:
                if f["t0"] <= t <= f["t1"]:
                    near = [s for b, s, l in zip(boxes, scores, labels) if l == 2 and inside(f, b, args.match_margin)]
                    per_fire[f["id"]]["scores"].append(max(near) if near else 0.0)

            for c, confirmer in confirmers.items():
                for alarm in confirmer.update(boxes, scores, labels):
                    if alarm.track_id in active[c]:
                        continue
                    active[c].add(alarm.track_id)
                    owner = next((f for f in fires if f["t0"] - 1.0 <= t <= f["t1"] + 2.0
                                  and inside(f, alarm.box, args.match_margin)), None)
                    if owner is None:
                        false_alarms[c].append({"t": round(t, 2), "class": alarm.class_name,
                                                "box": [int(v) for v in alarm.box], "score": round(alarm.score, 3)})
                    elif per_fire[owner["id"]]["alarm_t"][c] is None:
                        per_fire[owner["id"]]["alarm_t"][c] = round(t, 2)
                active[c] &= {a.track_id for a in confirmer.tracks if a.confirmed}
        index += 1
        progress.update(1)
    progress.close()

    rows = []
    for f in fires:
        s = np.array(per_fire[f["id"]]["scores"])
        side = int(round(np.sqrt((f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))))
        rows.append({
            "id": f["id"], "size_px": side, "t0": f["t0"], "t1": f["t1"],
            "frames": int(len(s)), "seen": int((s >= 0.3).sum()),
            "median_score": float(np.median(s)) if len(s) else 0.0, "max_score": float(s.max()) if len(s) else 0.0,
            "alarm_t": {str(c): t for c, t in per_fire[f["id"]]["alarm_t"].items()},
        })
    return {"video": clip["video"], "fires": rows,
            "false_alarms": {str(c): v for c, v in false_alarms.items()}, "duration_s": round(total / fps, 1)}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model, _ = load_detector(args.weights, device)
    scales = [int(v) for v in args.scales.split(",")] if args.scales else [model.config["image_size"]]
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    args.confs = [float(v) for v in str(args.conf).split(",")]

    print(f"weights {args.weights} | scales {scales} | alarm {args.enter_hits}-of-{args.window} at "
          f"conf >= {args.confs} | every {args.stride} frames")
    results = [run_clip(model, clip, args, device, scales) for clip in gt["clips"]]

    conf_cols = "".join(f" {'alarm@' + str(c):>12s}" for c in args.confs)
    header = f"{'fire':24s} {'size':>6s} {'window':>13s} {'median':>7s} {'max':>6s}{conf_cols}"
    caught = {c: 0 for c in args.confs}
    total = 0
    for res in results:
        print(f"\n{res['video']}  ({res['duration_s']} s)")
        print(header + "\n" + "-" * len(header))
        for r in res["fires"]:
            total += 1
            cells = ""
            for c in args.confs:
                t = r["alarm_t"][str(c)]
                caught[c] += t is not None
                cell = f"+{t - r['t0']:.1f}s" if t is not None else "MISSED"
                cells += f" {cell:>12s}"
            print(f"{r['id']:24s} {r['size_px']:>4d}px {r['t0']:5.1f}-{r['t1']:5.1f}s {r['median_score']:7.2f} "
                  f"{r['max_score']:6.2f}{cells}")
        for c in args.confs:
            for fa in res["false_alarms"][str(c)]:
                print(f"  FALSE ALARM @conf {c}: {fa['t']:.1f}s {fa['class']} {fa['box']} score {fa['score']}")
    print()
    for c in args.confs:
        n_false = sum(len(r["false_alarms"][str(c)]) for r in results)
        print(f"conf {c}: fires alarmed {caught[c]}/{total}   false alarms {n_false}")

    if args.output:
        Path(args.output).write_text(json.dumps({"args": vars(args), "results": results}, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
