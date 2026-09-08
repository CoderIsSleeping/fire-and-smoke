"""Measure inference throughput and work out how many cameras a box can carry.

The deployment target is 70 cameras at 10 fps -- 700 frames per second. That is
a different engineering problem from "does the model detect fire", and it has to
be answered with measurements on the actual hardware, not estimates.

This script sweeps image size, batch size and RPN proposal count, reports frames
per second, and converts that directly into camera capacity under two
architectures:

  every-frame   every camera's every frame goes through the detector
  cascade       a cheap CPU gate watches all cameras and only wakes the
                detector on frames that look interesting, plus a guaranteed
                periodic sweep so nothing can hide behind the gate

Run it on the GPU you intend to buy before you buy 70 cameras' worth of them.

    python scripts/benchmark_throughput.py --weights best.pt --device cuda
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

from fire_smoke.dataset import letterbox
from fire_smoke.glow import glow_features
from fire_smoke.model import load_detector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Throughput benchmark and camera-capacity planner.")
    p.add_argument("--weights", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--sizes", default="640,512,448", help="Image sizes to sweep (multiples of 64).")
    p.add_argument("--batches", default="1,4,8", help="Batch sizes to sweep.")
    p.add_argument("--proposals", default="1000,300", help="RPN post-NMS proposals at test time.")
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--no-amp", dest="amp", action="store_false", default=True)

    plan = p.add_argument_group("deployment plan")
    plan.add_argument("--cameras", type=int, default=70)
    plan.add_argument("--camera-fps", type=float, default=10.0, help="Ingest frame rate per camera.")
    plan.add_argument("--gate-pass-rate", type=float, default=0.02,
                      help="Fraction of frames the cheap gate is expected to pass through.")
    plan.add_argument("--sweep-fps", type=float, default=0.5,
                      help="Guaranteed detector rate per camera regardless of the gate.")
    p.add_argument("--output", default=None)
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("warning: CUDA unavailable, benchmarking on CPU. These numbers will not")
        print("         reflect a GPU deployment - rerun this on the target hardware.")
        return torch.device("cpu")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure(model, device, size, batch, iters, warmup, use_amp) -> float:
    """Frames per second for one configuration."""
    frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    canvas, _ = letterbox(frame, size)
    tensor = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float().div_(255.0).to(device)
    images = [tensor] * batch
    glow = torch.from_numpy(np.tile(glow_features(canvas), (batch, 1))).to(device)

    amp_on = use_amp and device.type == "cuda"
    for _ in range(warmup):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_on):
            model(images, glow=glow)
    synchronize(device)

    started = time.perf_counter()
    for _ in range(iters):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_on):
            model(images, glow=glow)
    synchronize(device)
    elapsed = time.perf_counter() - started
    return batch * iters / elapsed


def measure_gate() -> dict:
    """Cost of the cheap CPU pre-filter that would watch every camera."""
    frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    small = cv2.resize(frame, (320, 180))
    previous = cv2.resize(frame, (320, 180))

    def frame_difference():
        delta = cv2.absdiff(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY))
        return float((delta > 15).mean())

    results = {}
    for name, fn in (("frame_difference_320x180", frame_difference),
                     ("glow_features_full_frame", lambda: glow_features(frame))):
        for _ in range(5):
            fn()
        started = time.perf_counter()
        for _ in range(50):
            fn()
        per_call = (time.perf_counter() - started) / 50
        results[name] = {"ms": per_call * 1000, "fps_per_core": 1.0 / per_call}
    return results


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    sizes = [int(x) for x in args.sizes.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    proposals = [int(x) for x in args.proposals.split(",")]

    print(f"device : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else f" ({torch.get_num_threads()} threads)"))
    print(f"amp    : {args.amp and device.type == 'cuda'}")
    print(f"target : {args.cameras} cameras x {args.camera_fps:g} fps = "
          f"{args.cameras * args.camera_fps:g} frames/s\n")

    header = f"{'size':>5} {'batch':>6} {'props':>6} {'fps':>8} {'ms/frame':>9} {'cameras @10fps':>15}"
    print(header)
    print("-" * len(header))

    rows = []
    best = None
    for size in sizes:
        for post_nms in proposals:
            model, _ = load_detector(args.weights, device, image_size=size)
            model.eval()
            model.detector.rpn._post_nms_top_n = {"training": 2000, "testing": post_nms}
            model.detector.rpn._pre_nms_top_n = {"training": 2000, "testing": min(2000, post_nms * 4)}
            model.detector.roi_heads.detections_per_img = 20 if post_nms < 1000 else 50
            for batch in batches:
                try:
                    fps = measure(model, device, size, batch, args.iters, args.warmup, args.amp)
                except RuntimeError as exc:
                    print(f"{size:>5} {batch:>6} {post_nms:>6}  failed: {str(exc)[:40]}")
                    continue
                row = {"size": size, "batch": batch, "proposals": post_nms,
                       "fps": fps, "ms_per_frame": 1000.0 / fps,
                       "cameras_at_target_fps": fps / args.camera_fps}
                rows.append(row)
                print(f"{size:>5} {batch:>6} {post_nms:>6} {fps:>8.1f} {1000/fps:>9.1f} "
                      f"{fps/args.camera_fps:>15.1f}")
                if best is None or fps > best["fps"]:
                    best = row

    gate = measure_gate()
    print("\ncheap CPU gate (single core):")
    for name, entry in gate.items():
        print(f"  {name:<28} {entry['ms']:6.2f} ms  -> {entry['fps_per_core']:8.0f} fps/core")

    if best:
        needed = args.cameras * args.camera_fps
        # Cascade: a guaranteed periodic sweep of every camera, plus whatever the
        # gate lets through. The sweep is the safety net -- without it, anything
        # the gate misses is never seen by the detector at all.
        sweep_load = args.cameras * args.sweep_fps
        gated_load = needed * args.gate_pass_rate
        cascade_load = sweep_load + gated_load

        print(f"\n{'='*64}")
        print(f"CAPACITY  (best config: {best['size']}px, batch {best['batch']}, "
              f"{best['proposals']} proposals, {best['fps']:.1f} fps)")
        print(f"{'='*64}")
        print(f"  every-frame architecture")
        print(f"    required        {needed:8.0f} frames/s")
        print(f"    this device     {best['fps']:8.1f} frames/s")
        print(f"    devices needed  {needed / best['fps']:8.1f}")
        print()
        print(f"  cascade architecture (cheap gate + guaranteed sweep)")
        print(f"    periodic sweep  {sweep_load:8.1f} frames/s "
              f"({args.cameras} cameras x {args.sweep_fps:g} fps, never skipped)")
        print(f"    gate passes     {gated_load:8.1f} frames/s "
              f"({args.gate_pass_rate:.0%} of {needed:.0f})")
        print(f"    total detector  {cascade_load:8.1f} frames/s")
        print(f"    devices needed  {cascade_load / best['fps']:8.2f}")
        print()
        print(f"  the gate itself runs at {min(g['fps_per_core'] for g in gate.values()):.0f}+ fps on one CPU core,")
        print(f"  so all {args.cameras} cameras at {args.camera_fps:g} fps ({needed:.0f} fps) need roughly "
              f"{needed / min(g['fps_per_core'] for g in gate.values()):.1f} cores.")

        if device.type != "cuda":
            print("\n  NOTE: measured on CPU. Re-run on the target GPU before sizing hardware.")

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"device": str(device), "rows": rows, "gate": gate,
             "cameras": args.cameras, "camera_fps": args.camera_fps}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
