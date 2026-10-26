"""Multi-camera load test: can one GPU keep N camera streams current?

Each simulated camera is a thread that produces frames at --camera-fps and keeps
only its newest frame (the same latest-frame rule as the live pipeline, so a
slow detector never works on old frames). One inference worker owns the single
shared model: it takes the newest unseen frame from every camera, runs them as
one batch, and goes round again. This is the deployment design -- per-camera
workers, one shared model -- not N model copies.

For each camera count it reports, after a warm-up:

  analysed fps / camera   how often each camera actually gets checked
  latency p50 / p95       age of a frame when its result comes back
                          (capture -> detections), in milliseconds
  coverage                analysed / produced frames, per camera

A camera count "keeps up" when every camera is analysed at least
--min-camera-fps times per second and p95 latency stays under --max-latency-ms.
Fire develops over seconds, so 2 checks per camera per second is the default
bar; full 10 fps analysis per camera is reported but not required.

Sources (all simulated cameras loop their source):
  --videos a.mp4 b.mp4 ...   video files, assigned to cameras round-robin
  --image-folder DIR         still images played as a stream (no video needed;
                             on Kaggle the D-Fire test images work)

    python scripts/multicam_stress.py --weights best.pt \\
        --image-folder /kaggle/input/.../images/test --cameras 2,3,4,5,6,7,8,9,10

Adaptive mode (--adaptive) is the 70-camera design: a quiet camera is checked
only every 1/--quiet-fps seconds; as soon as any detection reaches --wake-conf
it becomes active and is checked at --active-fps until it has been clear for
--cooldown seconds. --active-cameras of the simulated cameras show fire images,
the rest show a few fixed fire-free images each (a near-static scene, like a
real camera), so a quiet camera that wakes up is a false wake-up. It keeps up
when quiet cameras are never left longer than 1.5 / --quiet-fps seconds, fire
cameras get --active-fps checks per second, and p95 latency stays in budget.

    python scripts/multicam_stress.py --weights best.pt --adaptive \
        --image-folder .../images/test --cameras 10,20,30,40,50,60,70

Plant footage is confidential: on Kaggle or any cloud GPU use public images or
videos only.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch

from fire_smoke.dataset import letterbox
from fire_smoke.glow import glow_features
from fire_smoke.model import load_detector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--videos", nargs="+")
    src.add_argument("--image-folder")
    p.add_argument("--cameras", default="2,3,4,5,6,7,8,9,10", help="Camera counts to test, in order.")
    p.add_argument("--camera-fps", type=float, default=10.0, help="Frame rate each simulated camera produces.")
    p.add_argument("--frame-size", default="1920x1080", help="Frame size of the simulated cameras (WxH).")
    p.add_argument("--frames-per-camera", type=int, default=40,
                   help="Distinct frames held in memory per camera when using --image-folder.")
    p.add_argument("--imgsz", type=int, default=0, help="Model input size (default: the checkpoint's).")
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--prep-workers", type=int, default=4, help="CPU threads preparing frames.")
    p.add_argument("--no-prefetch", dest="prefetch", action="store_false", default=True,
                   help="Prepare each batch only after the previous one finished (the old, serial loop).")
    p.add_argument("--duration", type=float, default=40.0, help="Measured seconds per camera count.")
    p.add_argument("--warmup", type=float, default=8.0, help="Unmeasured seconds before each measurement.")
    p.add_argument("--min-camera-fps", type=float, default=2.0)
    p.add_argument("--max-latency-ms", type=float, default=1000.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    p.add_argument("--output", default=None, help="Write the results as JSON here.")
    ad = p.add_argument_group("adaptive check rate")
    ad.add_argument("--adaptive", action="store_true", help="Quiet / active scheduling instead of round-robin.")
    ad.add_argument("--quiet-fps", type=float, default=0.5, help="Checks per second for a quiet camera.")
    ad.add_argument("--active-fps", type=float, default=2.0, help="Checks per second for an active camera.")
    ad.add_argument("--wake-conf", type=float, default=0.5, help="Detection score that makes a camera active.")
    ad.add_argument("--cooldown", type=float, default=5.0, help="Seconds without a detection before going quiet.")
    ad.add_argument("--active-cameras", type=int, default=2, help="Simulated cameras that show fire.")
    ad.add_argument("--frames-per-quiet-camera", type=int, default=3)
    return p.parse_args()


class SimCamera(threading.Thread):
    """Produces frames at a fixed rate and keeps only the newest one."""

    def __init__(self, cam_id: int, fps: float, frames: list[np.ndarray] | None = None,
                 video: str | None = None, size: tuple[int, int] = (1920, 1080)) -> None:
        super().__init__(daemon=True)
        self.cam_id, self.period, self.frames, self.video, self.size = cam_id, 1.0 / fps, frames, video, size
        self.lock = threading.Lock()
        self.latest: tuple[int, float, np.ndarray] | None = None   # (seq, capture time, frame)
        self.produced = 0
        self.running = True

    def _video_frames(self):
        while self.running:
            cap = cv2.VideoCapture(self.video)
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"cannot read {self.video}")
            while ok and self.running:
                if (frame.shape[1], frame.shape[0]) != self.size:
                    frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
                yield frame
                ok, frame = cap.read()
            cap.release()

    def _image_frames(self):
        i = self.cam_id
        while self.running:
            yield self.frames[i % len(self.frames)]
            i += 1

    def run(self) -> None:
        source = self._video_frames() if self.video else self._image_frames()
        next_t = time.perf_counter()
        for frame in source:
            if not self.running:
                break
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += self.period
            with self.lock:
                self.produced += 1
                self.latest = (self.produced, time.perf_counter(), frame)

    def take(self, last_seq: int):
        with self.lock:
            if self.latest is None or self.latest[0] == last_seq:
                return None
            return self.latest


def load_image_frames(folder: str, n_cams: int, per_cam: int, size: tuple[int, int]) -> list[list[np.ndarray]]:
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    need = n_cams * per_cam
    if len(files) < per_cam:
        raise SystemExit(f"only {len(files)} images in {folder}")
    step = max(1, len(files) // need)
    picked = files[::step][:need]
    frames = []
    for path in picked:
        img = cv2.imread(str(path))
        if img is not None:
            frames.append(cv2.resize(img, size, interpolation=cv2.INTER_LINEAR))
    return [frames[c::n_cams] or frames for c in range(n_cams)]


def prepare(frame: np.ndarray, size: int):
    """CPU side of one frame: colour conversion, letterbox, glow statistics."""
    canvas, _ = letterbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size)
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)), glow_features(canvas)


def run_batch(model, device, use_amp, batch, prepared):
    items = [f.result() for f in prepared]
    tensors = [torch.from_numpy(chw).to(device, non_blocking=True).float().div_(255.0) for chw, _ in items]
    glow = torch.from_numpy(np.stack([g for _, g in items])).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        detections, _ = model(tensors, glow=glow)
    for det in detections:
        det["boxes"].cpu()   # results are back on the host before the clock stops
    if device.type == "cuda":
        torch.cuda.synchronize()
    return detections


def load_adaptive_pools(folder: str, n_quiet: int, per_quiet: int, n_fire: int, size: tuple[int, int]):
    """Fire-free and fire frames from a YOLO image folder (labels next to images/)."""
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    neg, fire = [], []
    for f in files:
        lab = Path(str(f.parent).replace("images", "labels")) / (f.stem + ".txt")
        lines = lab.read_text().split() if lab.exists() else []
        (fire if lines[:1] and any(v in ("1", "1.0") for v in lines[0::5]) else neg if not lines else []).append(f)
    rng = np.random.default_rng(0)
    rng.shuffle(neg)
    rng.shuffle(fire)
    need_neg, need_fire = n_quiet * per_quiet, max(n_fire, 1) * 20
    if len(neg) < need_neg or len(fire) < 20:
        raise SystemExit(f"need {need_neg} fire-free and 20 fire images, found {len(neg)} and {len(fire)}")

    def load(paths):
        return [cv2.resize(cv2.imread(str(x)), size, interpolation=cv2.INTER_LINEAR) for x in paths]

    quiet = load(neg[:need_neg])
    return [quiet[k * per_quiet:(k + 1) * per_quiet] for k in range(n_quiet)], load(fire[:need_fire])


def run_count_adaptive(model, device, use_amp, size, n, args, pools):
    quiet_sets, fire_frames = pools
    n_fire = min(args.active_cameras, n)
    transform = model.detector.transform
    transform.min_size, transform.max_size = (size,), size
    cams = []
    for i in range(n):
        frames = fire_frames[(i * 20) % len(fire_frames):][:20] or fire_frames if i < n_fire else quiet_sets[i - n_fire]
        cams.append(SimCamera(i, args.camera_fps, size=args.frame_size_wh, frames=frames))
    for c in cams:
        c.start()
    while any(c.latest is None for c in cams):
        time.sleep(0.01)

    quiet_period, active_period = 1.0 / args.quiet_fps, 1.0 / args.active_fps
    t0 = time.perf_counter()
    t_measure, t_end = t0 + args.warmup, t0 + args.warmup + args.duration
    # stagger quiet cameras so they do not all fall due at once
    next_due = [t0 + (i / n) * quiet_period for i in range(n)]
    active = [False] * n
    last_hit = [-1e9] * n
    last_check = [None] * n
    gaps_quiet, latencies, batch_sizes = [], [], []
    checks = [0] * n
    false_wakes = set()
    pending = None
    pool = ThreadPoolExecutor(max_workers=args.prep_workers)
    while True:
        now = time.perf_counter()
        if now >= t_end:
            break
        measuring = now >= t_measure
        due = sorted((next_due[i], i) for i in range(n) if now >= next_due[i])[:args.max_batch]
        batch = []
        for _, i in due:
            seq, captured, frame = cams[i].latest
            batch.append((i, (seq, captured, frame)))
            next_due[i] = now + (active_period if active[i] else quiet_period)
        if not batch and pending is None:
            time.sleep(0.002)
            continue
        prepared = [pool.submit(prepare, fr, size) for _, (_, _, fr) in batch] if batch else None
        if pending is not None:
            dets = run_batch(model, device, use_amp, *pending)
            done = time.perf_counter()
            for (i, (_, captured, _)), det in zip(pending[0], dets):
                hit = bool((det["scores"] >= args.wake_conf).any())
                if hit:
                    last_hit[i] = done
                    if not active[i]:
                        active[i] = True
                        next_due[i] = min(next_due[i], done + active_period)
                        if i >= n_fire and done >= t_measure:
                            false_wakes.add(i)
                elif active[i] and done - last_hit[i] > args.cooldown:
                    active[i] = False
                if done >= t_measure:
                    checks[i] += 1
                    latencies.append((done - captured) * 1000.0)
                    if last_check[i] is not None and not active[i] and i >= n_fire:
                        gaps_quiet.append(done - last_check[i])
                last_check[i] = done
            if done >= t_measure:
                batch_sizes.append(len(pending[0]))
        pending = (batch, prepared) if batch else None
    pool.shutdown(wait=True)
    for c in cams:
        c.running = False
    for c in cams:
        c.join(timeout=2)

    lat = np.array(latencies) if latencies else np.array([float("nan")])
    fire_rate = [checks[i] / args.duration for i in range(n_fire)]
    quiet_gap_max = max(gaps_quiet) if gaps_quiet else float("nan")
    result = {
        "cameras": n,
        "fire_cameras": n_fire,
        "total_fps": sum(checks) / args.duration,
        "fire_cam_fps_min": min(fire_rate) if fire_rate else float("nan"),
        "quiet_gap_p95_s": float(np.percentile(gaps_quiet, 95)) if gaps_quiet else float("nan"),
        "quiet_gap_max_s": quiet_gap_max,
        "false_wakes": len(false_wakes),
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "mean_batch": float(np.mean(batch_sizes)) if batch_sizes else 0.0,
        "gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0,
    }
    result["keeps_up"] = (result["fire_cam_fps_min"] >= 0.9 * args.active_fps
                          and result["quiet_gap_p95_s"] <= 1.5 * quiet_period
                          and result["latency_p95_ms"] <= args.max_latency_ms)
    return result


def run_count(model, device, use_amp, size, n, args, frame_sets):
    transform = model.detector.transform
    transform.min_size, transform.max_size = (size,), size
    cams = [SimCamera(i, args.camera_fps, size=args.frame_size_wh,
                      frames=frame_sets[i] if frame_sets else None,
                      video=args.videos[i % len(args.videos)] if args.videos else None) for i in range(n)]
    for c in cams:
        c.start()

    last_seq = [0] * n
    analysed = [0] * n
    latencies: list[float] = []
    batch_sizes: list[int] = []
    measuring = False
    t_start = time.perf_counter()
    t_measure = t_start + args.warmup
    t_end = t_measure + args.duration
    produced_at_measure = [0] * n
    next_cam = 0
    pending = None
    pool = ThreadPoolExecutor(max_workers=args.prep_workers)
    while True:
        now = time.perf_counter()
        if now >= t_end:
            break
        if not measuring and now >= t_measure:
            measuring = True
            produced_at_measure = [c.produced for c in cams]
            analysed = [0] * n
            latencies.clear()
            batch_sizes.clear()
        # newest unseen frame from each camera, round-robin start so no camera is favoured
        batch = []
        for k in range(n):
            i = (next_cam + k) % n
            got = cams[i].take(last_seq[i])
            if got is not None:
                last_seq[i] = got[0]
                batch.append((i, got))
            if len(batch) >= args.max_batch:
                break
        next_cam = (next_cam + 1) % n
        if not batch and pending is None:
            time.sleep(0.001)
            continue
        # CPU preparation of this batch runs in the pool while the GPU works on the previous one
        prepared = [pool.submit(prepare, frame, size) for _, (_, _, frame) in batch] if batch else None
        if not args.prefetch and batch:
            pending, prepared, batch = (batch, prepared), None, []
        if pending is not None:
            done_batch, n_done = run_batch(model, device, use_amp, *pending), len(pending[0])
            done = time.perf_counter()
            if measuring:
                for i, (_, captured, _) in pending[0]:
                    analysed[i] += 1
                    latencies.append((done - captured) * 1000.0)
                batch_sizes.append(n_done)
        pending = (batch, prepared) if batch else None

    pool.shutdown(wait=True)
    for c in cams:
        c.running = False
    for c in cams:
        c.join(timeout=2)
    elapsed = args.duration
    per_cam_fps = [a / elapsed for a in analysed]
    produced = [c.produced - p for c, p in zip(cams, produced_at_measure)]
    lat = np.array(latencies) if latencies else np.array([float("nan")])
    result = {
        "cameras": n,
        "total_fps": sum(analysed) / elapsed,
        "min_camera_fps": min(per_cam_fps),
        "mean_camera_fps": float(np.mean(per_cam_fps)),
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "coverage": float(sum(analysed) / max(1, sum(produced))),
        "mean_batch": float(np.mean(batch_sizes)) if batch_sizes else 0.0,
        "gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0,
    }
    result["keeps_up"] = (result["min_camera_fps"] >= args.min_camera_fps
                          and result["latency_p95_ms"] <= args.max_latency_ms)
    return result


def main() -> None:
    args = parse_args()
    w, h = (int(v) for v in args.frame_size.lower().split("x"))
    args.frame_size_wh = (w, h)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    use_amp = args.amp and device.type == "cuda"
    overrides = {"image_size": args.imgsz} if args.imgsz else {}
    model, _ = load_detector(args.weights, device, **overrides)
    model.eval()
    size = model.config["image_size"]
    counts = [int(v) for v in args.cameras.split(",")]

    if args.adaptive:
        if not args.image_folder:
            raise SystemExit("--adaptive needs --image-folder (a YOLO folder with labels)")
        pools = load_adaptive_pools(args.image_folder, max(counts) - min(args.active_cameras, min(counts)),
                                    args.frames_per_quiet_camera, args.active_cameras, args.frame_size_wh)
        run_adaptive(model, device, use_amp, size, counts, args, pools)
        return

    frame_sets = None
    if args.image_folder:
        frame_sets = load_image_frames(args.image_folder, max(counts), args.frames_per_camera, args.frame_size_wh)

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"device {gpu} | input {size}px | amp {use_amp} | cameras at {args.camera_fps:g} fps, {w}x{h} | "
          f"max batch {args.max_batch} | prefetch {args.prefetch} ({args.prep_workers} CPU threads) | "
          f"{args.duration:g}s per count after {args.warmup:g}s warm-up")
    print(f"keeps up = every camera analysed >= {args.min_camera_fps:g}/s and p95 latency <= {args.max_latency_ms:g} ms\n")
    header = (f"{'cams':>4} {'total fps':>9} {'min fps/cam':>11} {'mean fps/cam':>12} {'p50 ms':>7} {'p95 ms':>7} "
              f"{'coverage':>8} {'batch':>5} {'GPU GB':>6}  result")
    print(header)
    print("-" * len(header))
    results = []
    for n in counts:
        r = run_count(model, device, use_amp, size, n, args, frame_sets)
        results.append(r)
        print(f"{n:>4} {r['total_fps']:>9.1f} {r['min_camera_fps']:>11.2f} {r['mean_camera_fps']:>12.2f} "
              f"{r['latency_p50_ms']:>7.0f} {r['latency_p95_ms']:>7.0f} {r['coverage']:>8.0%} {r['mean_batch']:>5.1f} "
              f"{r['gpu_mem_gb']:>6.2f}  {'KEEPS UP' if r['keeps_up'] else 'FALLS BEHIND'}", flush=True)
    ok = [r["cameras"] for r in results if r["keeps_up"]]
    print(f"\nlargest camera count that keeps up: {max(ok) if ok else 'none'}"
          f"  (bar: {args.min_camera_fps:g} checks per camera per second)")
    if args.output:
        Path(args.output).write_text(json.dumps({"args": {k: v for k, v in vars(args).items() if k != "frame_size_wh"},
                                                 "device": gpu, "image_size": size, "results": results}, indent=2))
        print("wrote", args.output)


def run_adaptive(model, device, use_amp, size, counts, args, pools) -> None:
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"device {gpu} | input {size}px | ADAPTIVE: quiet {args.quiet_fps:g}/s, active {args.active_fps:g}/s, "
          f"wake at {args.wake_conf:g}, cooldown {args.cooldown:g}s | {args.active_cameras} cameras show fire | "
          f"{args.duration:g}s per count after {args.warmup:g}s warm-up")
    print(f"keeps up = fire cameras >= {0.9 * args.active_fps:g} checks/s, quiet cameras checked at least every "
          f"{1.5 / args.quiet_fps:g}s (p95), p95 latency <= {args.max_latency_ms:g} ms\n")
    header = (f"{'cams':>4} {'total fps':>9} {'fire cam fps':>12} {'quiet gap p95':>13} {'max':>6} "
              f"{'false wakes':>11} {'p95 ms':>7} {'batch':>5}  result")
    print(header + "\n" + "-" * len(header))
    results = []
    for n in counts:
        r = run_count_adaptive(model, device, use_amp, size, n, args, pools)
        results.append(r)
        print(f"{n:>4} {r['total_fps']:>9.1f} {r['fire_cam_fps_min']:>12.2f} {r['quiet_gap_p95_s']:>12.1f}s "
              f"{r['quiet_gap_max_s']:>5.1f}s {r['false_wakes']:>4d} of {n - r['fire_cameras']:<4d} "
              f"{r['latency_p95_ms']:>7.0f} {r['mean_batch']:>5.1f}  {'KEEPS UP' if r['keeps_up'] else 'FALLS BEHIND'}",
              flush=True)
    ok = [r["cameras"] for r in results if r["keeps_up"]]
    print(f"\nlargest camera count that keeps up: {max(ok) if ok else 'none'}")
    if args.output:
        Path(args.output).write_text(json.dumps({"args": {k: v for k, v in vars(args).items() if k != "frame_size_wh"},
                                                 "device": gpu, "image_size": size, "results": results}, indent=2))
        print("wrote", args.output)


if __name__ == "__main__":
    main()
