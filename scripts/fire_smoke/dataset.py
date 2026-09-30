"""Dataset, letterboxing and augmentation for the D-Fire detection data.

Label format on disk is unchanged from the YOLO stage (one `.txt` per image,
`class cx cy w h` normalised, class 0 = smoke, 1 = fire, empty file = a verified
negative). Detector label ids are shifted by +1 because torchvision reserves 0
for background.

The augmentation pipeline is where most of the deployment requirements are
addressed, because D-Fire is overwhelmingly daytime, unoccluded, web-sourced
imagery and the target is a fixed industrial camera running around the clock:

  * `night`      -- simulates a camera whose auto-exposure has closed down:
                    everything goes dark but highlights survive, plus sensor
                    noise from the raised gain. This is the low-light domain.
  * `gray`       -- simulates IR / night-mode cameras that drop colour entirely,
                    forcing the model to use brightness and shape, not just the
                    orange colour cue.
  * `occlude`    -- covers most of a *fire* box with a synthetic obstacle while
                    keeping the label. Trains the model to infer fire from the
                    surrounding illumination rather than from the flame itself.
  * `crop`/`flip`/`jitter`/`blur` -- ordinary scale, viewpoint and sensor
                    variation.
  * `mosaic`     -- tiles 2x2 or 3x3 training images into one frame, so every
                    object appears at 1/2 or 1/3 of its usual size in a busy,
                    wide scene. `crop` only ever zooms *in*; this zooms out.
  * `paste_fire` -- cuts real flames out of training fire boxes and blends them
                    into the image at 8-40 px (in a 640 px input). Distant
                    flames on a fixed CCTV camera were measured to score 0.6-0.9
                    at that size; this manufactures thousands of them, in scenes
                    D-Fire never shows them in, with exact boxes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .glow import GLOW_FEATURE_DIM, glow_features

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PAD_VALUE = 114


@dataclass
class AugmentConfig:
    """Probabilities for the training-time augmentation pipeline."""

    hflip: float = 0.5
    crop: float = 0.8
    crop_min_scale: float = 0.45
    jitter: float = 0.8
    night: float = 0.35
    gray: float = 0.10
    blur: float = 0.10
    occlude_fire: float = 0.25
    # Small / distant fire augmentation. Off by default so earlier runs are
    # reproduced exactly; switched on with --mosaic-prob / --paste-prob.
    mosaic: float = 0.0
    paste_fire: float = 0.0
    paste_min: float = 8.0   # pasted flame size, sqrt(area) in px of the final model input
    paste_max: float = 40.0
    paste_max_count: int = 3

    @staticmethod
    def disabled() -> "AugmentConfig":
        return AugmentConfig(hflip=0, crop=0, jitter=0, night=0, gray=0, blur=0, occlude_fire=0,
                             mosaic=0, paste_fire=0)


def load_data_config(data_yaml: str | Path) -> tuple[Path, dict]:
    """Read a YOLO-style data.yaml and resolve its root directory.

    The `path:` field is often an absolute path from whichever machine prepared
    the dataset, so it is useless on Kaggle. If it does not exist we fall back
    to the directory containing the yaml, which is where the split folders
    actually live.
    """
    import yaml

    path = Path(data_yaml)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    declared = cfg.get("path")
    root = Path(declared) if declared else path.parent
    if not root.exists():
        root = path.parent
    return root.resolve(), cfg


def find_data_yaml(preferred: str | Path | None) -> Path:
    """Locate data.yaml, searching the usual Kaggle input mount as a fallback."""
    if preferred:
        candidate = Path(preferred)
        if candidate.exists():
            return candidate
    for search_root in (Path("/kaggle/input"), Path("datasets/processed")):
        if search_root.exists():
            found = sorted(search_root.rglob("data.yaml"))
            if found:
                return found[0]
    raise FileNotFoundError(
        "No data.yaml found. Pass --data, or attach the prepared detection "
        "dataset as a Kaggle input."
    )


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def letterbox(image: np.ndarray, size: int, pad_value: int = PAD_VALUE):
    """Resize preserving aspect ratio and pad to a square `size` x `size`.

    Returns the canvas plus (ratio, pad_left, pad_top) so predictions made on
    the canvas can be mapped back to original pixel coordinates.
    """
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    new_h, new_w = max(1, int(round(h * ratio))), max(1, int(round(w * ratio)))
    interp = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    canvas = np.full((size, size, 3), pad_value, dtype=image.dtype)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas, (ratio, pad_left, pad_top)


def apply_letterbox_to_boxes(boxes: np.ndarray, ratio: float, pad_left: int, pad_top: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    boxes = boxes.astype(np.float32) * ratio
    boxes[:, [0, 2]] += pad_left
    boxes[:, [1, 3]] += pad_top
    return boxes


def undo_letterbox(boxes: np.ndarray, ratio: float, pad_left: int, pad_top: int) -> np.ndarray:
    """Map boxes from letterboxed canvas coordinates back to the source image."""
    if len(boxes) == 0:
        return boxes
    boxes = boxes.astype(np.float32).copy()
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    return boxes / max(ratio, 1e-9)


def read_yolo_label(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO label file into absolute xyxy boxes and 0/1 class ids."""
    if not path.exists():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

    boxes, labels = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:])
        except ValueError:
            continue
        if class_id not in (0, 1):
            continue
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        boxes.append([x1, y1, x2, y2])
        labels.append(class_id)

    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    return np.array(boxes, np.float32), np.array(labels, np.int64)


def clip_and_filter(boxes: np.ndarray, labels: np.ndarray, width: int, height: int, min_side: float = 2.0):
    if len(boxes) == 0:
        return boxes, labels
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    keep = (boxes[:, 2] - boxes[:, 0] >= min_side) & (boxes[:, 3] - boxes[:, 1] >= min_side)
    return boxes[keep], labels[keep]


# --------------------------------------------------------------------------
# photometric / geometric augmentation
# --------------------------------------------------------------------------


def random_crop(image, boxes, labels, rng: random.Random, min_scale: float):
    h, w = image.shape[:2]
    for _ in range(30):
        scale = rng.uniform(min_scale, 1.0)
        aspect = rng.uniform(0.75, 1.33)
        crop_w = min(w, max(32, int(w * scale * math.sqrt(aspect))))
        crop_h = min(h, max(32, int(h * scale / math.sqrt(aspect))))
        x0 = rng.randint(0, w - crop_w)
        y0 = rng.randint(0, h - crop_h)

        if len(boxes) == 0:
            return image[y0 : y0 + crop_h, x0 : x0 + crop_w], boxes, labels

        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        inside = (cx >= x0) & (cx < x0 + crop_w) & (cy >= y0) & (cy < y0 + crop_h)
        if not inside.any():
            continue

        kept = boxes[inside].copy()
        kept_labels = labels[inside].copy()
        original_area = (kept[:, 2] - kept[:, 0]) * (kept[:, 3] - kept[:, 1])
        kept[:, [0, 2]] = kept[:, [0, 2]].clip(x0, x0 + crop_w) - x0
        kept[:, [1, 3]] = kept[:, [1, 3]].clip(y0, y0 + crop_h) - y0
        new_area = (kept[:, 2] - kept[:, 0]) * (kept[:, 3] - kept[:, 1])
        # Drop boxes the crop mostly cut away, otherwise the label points at a
        # sliver of smoke and teaches nothing useful.
        good = new_area > 0.25 * np.maximum(original_area, 1e-6)
        if not good.any():
            continue
        return image[y0 : y0 + crop_h, x0 : x0 + crop_w], kept[good], kept_labels[good]

    return image, boxes, labels


def color_jitter(image: np.ndarray, rng: random.Random) -> np.ndarray:
    x = image.astype(np.float32)
    x *= rng.uniform(0.75, 1.30)  # brightness
    mean = x.mean()
    x = (x - mean) * rng.uniform(0.75, 1.30) + mean  # contrast
    x = np.clip(x, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180  # hue, kept small
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.35), 0, 255)  # saturation
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def night_augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Darken like a camera stopping down, while preserving hot highlights.

    A naive global multiply dims the flame too, which is exactly wrong: on a
    real camera at night the flame stays blown out and everything else sinks
    into the noise floor. So the darkening factor is relaxed where the pixel is
    already very bright.
    """
    x = image.astype(np.float32) / 255.0
    exposure = rng.uniform(0.12, 0.55)

    luma = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    t = np.clip((luma - 0.60) / 0.35, 0.0, 1.0)
    highlight_keep = t * t * (3.0 - 2.0 * t)  # smoothstep
    x *= (exposure + (1.0 - exposure) * highlight_keep)[..., None]

    if rng.random() < 0.5:  # cool cast typical of low-CCT night sensors
        x[..., 2] *= rng.uniform(1.00, 1.18)
        x[..., 0] *= rng.uniform(0.88, 1.00)

    # Raised gain means visible sensor noise.
    sigma = min(0.06, rng.uniform(0.004, 0.030) / math.sqrt(max(exposure, 0.12)))
    x += np.random.normal(0.0, sigma, x.shape).astype(np.float32)

    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def to_gray(image: np.ndarray) -> np.ndarray:
    """IR / night-mode simulation: no colour cue left, only luminance."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def motion_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    size = rng.choice([3, 5, 7, 9])
    kernel = np.zeros((size, size), np.float32)
    kernel[size // 2, :] = 1.0 / size
    angle = rng.uniform(0, 180)
    matrix = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    total = kernel.sum()
    if total <= 1e-6:
        return image
    return cv2.filter2D(image, -1, kernel / total)


def occlude_fire(image: np.ndarray, boxes: np.ndarray, labels: np.ndarray, rng: random.Random) -> np.ndarray:
    """Hide most of one flame behind a synthetic obstacle, keeping the label.

    D-Fire contains essentially no examples of "fire you cannot see but whose
    light you can". This manufactures them: the box stays, the flame does not,
    so the only remaining evidence is the illumination it casts on the rest of
    the scene.
    """
    fire_indices = np.where(labels == 1)[0]
    if len(fire_indices) == 0:
        return image

    h, w = image.shape[:2]
    index = int(rng.choice(list(fire_indices)))
    x1, y1, x2, y2 = boxes[index]
    box_w, box_h = x2 - x1, y2 - y1
    if box_w < 8 or box_h < 8:
        return image

    cover = rng.uniform(0.45, 0.85)
    side = rng.choice(["bottom", "top", "left", "right"])
    overhang_x, overhang_y = box_w * rng.uniform(0.0, 0.35), box_h * rng.uniform(0.0, 0.35)

    if side == "bottom":
        rect = (x1 - overhang_x, y2 - box_h * cover, x2 + overhang_x, y2 + overhang_y)
    elif side == "top":
        rect = (x1 - overhang_x, y1 - overhang_y, x2 + overhang_x, y1 + box_h * cover)
    elif side == "left":
        rect = (x1 - overhang_x, y1 - overhang_y, x1 + box_w * cover, y2 + overhang_y)
    else:
        rect = (x2 - box_w * cover, y1 - overhang_y, x2 + overhang_x, y2 + overhang_y)

    rx1, ry1, rx2, ry2 = (
        int(max(0, rect[0])),
        int(max(0, rect[1])),
        int(min(w, rect[2])),
        int(min(h, rect[3])),
    )
    if rx2 - rx1 < 4 or ry2 - ry1 < 4:
        return image

    # A plausible dull obstacle: a dark tone sampled from the scene, lightly
    # textured so it does not look like a flat mask the model can memorise.
    base = np.percentile(image.reshape(-1, 3), rng.uniform(5, 30), axis=0)
    patch_h, patch_w = ry2 - ry1, rx2 - rx1
    patch = np.tile(base.astype(np.float32), (patch_h, patch_w, 1))
    patch += np.random.normal(0, rng.uniform(2, 10), patch.shape).astype(np.float32)
    shade = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(1.0, 1.25), patch_h, dtype=np.float32)
    patch *= shade[:, None, None]

    image = image.copy()
    image[ry1:ry2, rx1:rx2] = np.clip(patch, 0, 255).astype(np.uint8)
    return image



def flame_alpha(crop: np.ndarray):
    """Soft mask of the flame inside a fire-box crop, or None if there is no clean flame.

    Brightness alone is not enough: a fire box against a bright sky would cut
    out the sky and teach the model that sky is fire. So pixels must be *warm*
    (red well above blue); the white-hot core, which is not warm, is added
    back only where it touches warm pixels. Crops with too little or too much
    flame are rejected, and edges fade with an elliptical window so no
    rectangular seam is left behind.
    """
    x = crop.astype(np.float32) / 255.0
    value = x.max(axis=2)
    warm = np.clip(x[..., 0] - x[..., 2], 0, 1)
    fire = warm * value  # dark but reddish pixels (burnt ground, embers' shadow) count for little
    if fire.max() < 0.2:
        return None
    lo, hi = np.percentile(fire, 40), np.percentile(fire, 95)
    alpha = np.clip((fire - lo) / max(hi - lo, 1e-3), 0, 1)
    near_flame = cv2.dilate((alpha > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    alpha = np.maximum(alpha, ((value > 0.85) & near_flame).astype(np.float32))
    solid = alpha > 0.5
    coverage = float(solid.mean())
    if not 0.04 <= coverage <= 0.75 or float(value[solid].mean()) < 0.55:
        return None
    h, w = alpha.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ellipse = ((xx - (w - 1) / 2) / (w / 2)) ** 2 + ((yy - (h - 1) / 2) / (h / 2)) ** 2
    alpha *= np.clip(1.25 - ellipse, 0, 1)
    return cv2.GaussianBlur(alpha, (0, 0), max(0.6, min(h, w) / 30))


def paste_small_fires(image, boxes, labels, sample_flame, rng: random.Random, cfg: AugmentConfig, image_size: int):
    """Blend 1..paste_max_count small real flames into the image, with labels."""
    h, w = image.shape[:2]
    ratio = image_size / max(h, w)  # letterbox scale applied later
    image = image.copy()
    new_boxes, new_labels = [boxes.reshape(-1, 4)], [labels.reshape(-1)]
    for _ in range(rng.randint(1, cfg.paste_max_count)):
        flame = sample_flame(rng)
        if flame is None:
            continue
        crop, alpha = flame
        target = rng.uniform(cfg.paste_min, cfg.paste_max) / ratio  # sqrt-area in this image's pixels
        scale = target / np.sqrt(crop.shape[0] * crop.shape[1])
        fw, fh = max(3, int(round(crop.shape[1] * scale))), max(3, int(round(crop.shape[0] * scale)))
        if fw >= w // 2 or fh >= h // 2:
            continue
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        crop_r = cv2.resize(crop, (fw, fh), interpolation=interp).astype(np.float32)
        alpha_r = cv2.resize(alpha, (fw, fh), interpolation=cv2.INTER_LINEAR)[..., None]
        existing = np.concatenate(new_boxes)
        placed = False
        for _attempt in range(10):
            x0, y0 = rng.randint(0, w - fw), rng.randint(0, h - fh)
            overlap = (
                (np.minimum(x0 + fw, existing[:, 2]) > np.maximum(x0, existing[:, 0]))
                & (np.minimum(y0 + fh, existing[:, 3]) > np.maximum(y0, existing[:, 1]))
            )
            if not overlap.any():
                placed = True
                break
        if not placed:
            continue
        ys, xs = np.where(alpha_r[..., 0] > 0.35)
        if len(xs) < 4:
            continue
        region = image[y0:y0 + fh, x0:x0 + fw].astype(np.float32)
        image[y0:y0 + fh, x0:x0 + fw] = np.clip(alpha_r * crop_r + (1 - alpha_r) * region, 0, 255).astype(np.uint8)
        new_boxes.append(np.array([[x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1]], np.float32))
        new_labels.append(np.array([1], np.int64))
    return image, np.concatenate(new_boxes).astype(np.float32), np.concatenate(new_labels).astype(np.int64)


def cover_resize(image, boxes, labels, size: int, rng: random.Random):
    """Resize to fill a size x size cell (random crop of the overflow), boxes adjusted."""
    h, w = image.shape[:2]
    scale = size / min(h, w)
    nw, nh = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x0, y0 = rng.randint(0, nw - size), rng.randint(0, nh - size)
    image = image[y0:y0 + size, x0:x0 + size]
    if len(boxes) == 0:
        return image, boxes, labels
    b = boxes * scale
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    b[:, [0, 2]] = (b[:, [0, 2]] - x0).clip(0, size)
    b[:, [1, 3]] = (b[:, [1, 3]] - y0).clip(0, size)
    kept = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) > 0.4 * np.maximum(area, 1e-6)
    return image, b[kept], labels[kept]


def augment(image, boxes, labels, cfg: AugmentConfig, rng: random.Random, paste=None, crop: bool = True):
    if crop and cfg.crop and rng.random() < cfg.crop:
        image, boxes, labels = random_crop(image, boxes, labels, rng, cfg.crop_min_scale)

    if cfg.hflip and rng.random() < cfg.hflip:
        image = np.ascontiguousarray(image[:, ::-1])
        if len(boxes):
            width = image.shape[1]
            boxes = boxes.copy()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

    if cfg.occlude_fire and rng.random() < cfg.occlude_fire:
        image = occlude_fire(image, boxes, labels, rng)

    # After geometry, before photometry: pasted flames then get the same
    # night / IR / blur treatment as the rest of the frame.
    if paste is not None and cfg.paste_fire and rng.random() < cfg.paste_fire:
        image, boxes, labels = paste(image, boxes, labels, rng)

    if cfg.jitter and rng.random() < cfg.jitter:
        image = color_jitter(image, rng)
    if cfg.night and rng.random() < cfg.night:
        image = night_augment(image, rng)
    if cfg.gray and rng.random() < cfg.gray:
        image = to_gray(image)
    if cfg.blur and rng.random() < cfg.blur:
        image = motion_blur(image, rng)

    return image, boxes, labels


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


class FireSmokeDataset(Dataset):
    """D-Fire style detection dataset returning detector + scene supervision."""

    def __init__(
        self,
        data_yaml: str | Path,
        split: str,
        image_size: int = 640,
        augment_cfg: AugmentConfig | None = None,
        max_images: int | None = None,
        seed: int = 0,
    ) -> None:
        self.root, cfg = load_data_config(data_yaml)
        rel = cfg.get(split, f"images/{split}")
        self.image_dir = (self.root / rel).resolve()
        self.label_dir = (self.root / "labels" / split).resolve()
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image directory for split '{split}': {self.image_dir}")

        self.images = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if max_images:
            step = max(1, len(self.images) // max_images)
            self.images = self.images[::step][:max_images]
        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")

        self.image_size = image_size
        self.cfg = augment_cfg or AugmentConfig.disabled()
        self.seed = seed
        self.split = split
        self.flame_bank = self._build_flame_bank() if self.cfg.paste_fire > 0 else []

    def _build_flame_bank(self) -> list[tuple[Path, float, float, float, float]]:
        """Fire boxes big enough to cut a clean flame from (normalised cx, cy, w, h)."""
        bank = []
        for image_path in self.images:
            label = self.label_path(image_path)
            if not label.exists():
                continue
            for line in label.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) == 5 and parts[0] in ("1", "1.0"):
                    cx, cy, bw, bh = (float(v) for v in parts[1:])
                    if 0.05 <= bw <= 0.6 and 0.05 <= bh <= 0.6:
                        bank.append((image_path, cx, cy, bw, bh))
        return bank

    def sample_flame(self, rng: random.Random):
        for _ in range(12):
            image_path, cx, cy, bw, bh = self.flame_bank[rng.randrange(len(self.flame_bank))]
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            x1, x2 = max(0, int((cx - bw / 2) * w)), min(w, int((cx + bw / 2) * w))
            y1, y2 = max(0, int((cy - bh / 2) * h)), min(h, int((cy + bh / 2) * h))
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            crop = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            if rng.random() < 0.5:
                crop = np.ascontiguousarray(crop[:, ::-1])
            alpha = flame_alpha(crop)
            if alpha is None:
                continue
            return crop, alpha
        return None

    def _load(self, index: int):
        image_path = self.images[index]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        boxes, labels = read_yolo_label(self.label_path(image_path), width, height)
        boxes, labels = clip_and_filter(boxes, labels, width, height)
        return image, boxes, labels

    def _mosaic(self, index: int, rng: random.Random):
        """2x2 (70%) or 3x3 (30%) grid of training images, this one included."""
        grid = 2 if rng.random() < 0.7 else 3
        cell = self.image_size
        indices = [index] + [rng.randrange(len(self.images)) for _ in range(grid * grid - 1)]
        rng.shuffle(indices)
        canvas = np.zeros((grid * cell, grid * cell, 3), np.uint8)
        all_boxes, all_labels = [], []
        for k, idx in enumerate(indices):
            image, boxes, labels = self._load(idx)
            if rng.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                if len(boxes):
                    boxes = boxes.copy()
                    boxes[:, [0, 2]] = image.shape[1] - boxes[:, [2, 0]]
            tile, boxes, labels = cover_resize(image, boxes, labels, cell, rng)
            r, c = divmod(k, grid)
            canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = tile
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] += c * cell
                boxes[:, [1, 3]] += r * cell
                all_boxes.append(boxes)
                all_labels.append(labels)
        if all_boxes:
            return canvas, np.concatenate(all_boxes).astype(np.float32), np.concatenate(all_labels).astype(np.int64)
        return canvas, np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

    def __len__(self) -> int:
        return len(self.images)

    def label_path(self, image_path: Path) -> Path:
        return self.label_dir / f"{image_path.stem}.txt"

    def __getitem__(self, index: int):
        # Per-sample RNG keyed on epoch-independent index + worker entropy so
        # DataLoader workers do not all draw the same augmentations.
        rng = random.Random((self.seed * 1_000_003 + index * 7919 + random.randrange(1 << 30)) & 0xFFFFFFFF)

        paste = None
        if self.flame_bank:
            def paste(img, bxs, lbls, r):
                return paste_small_fires(img, bxs, lbls, self.sample_flame, r, self.cfg, self.image_size)

        if self.cfg.mosaic and rng.random() < self.cfg.mosaic:
            image, boxes, labels = self._mosaic(index, rng)
            image, boxes, labels = augment(image, boxes, labels, self.cfg, rng, paste=paste, crop=False)
        else:
            image, boxes, labels = self._load(index)
            image, boxes, labels = augment(image, boxes, labels, self.cfg, rng, paste=paste)

        canvas, (ratio, pad_left, pad_top) = letterbox(image, self.image_size)
        boxes = apply_letterbox_to_boxes(boxes, ratio, pad_left, pad_top)
        boxes, labels = clip_and_filter(boxes, labels, self.image_size, self.image_size)

        glow = glow_features(canvas)

        # Scene labels are recomputed from the surviving boxes, so a crop that
        # removed the only flame also flips the image-level label to negative.
        scene = np.zeros(2, np.float32)
        if len(labels):
            scene[0] = float((labels == 0).any())  # smoke
            scene[1] = float((labels == 1).any())  # fire

        tensor = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float().div_(255.0)
        boxes_t = torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32)).reshape(-1, 4)
        labels_t = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64)).reshape(-1) + 1

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]),
            "iscrowd": torch.zeros((len(boxes_t),), dtype=torch.int64),
        }
        return tensor, target, torch.from_numpy(scene), torch.from_numpy(glow)


def collate_fn(batch):
    images, targets, scenes, glows = zip(*batch)
    return list(images), list(targets), torch.stack(scenes), torch.stack(glows)


def describe_split(dataset: FireSmokeDataset) -> dict:
    """Cheap label-only pass, used for the dataset summary printed at startup."""
    counts = {"images": len(dataset), "negatives": 0, "smoke_boxes": 0, "fire_boxes": 0,
              "images_with_smoke": 0, "images_with_fire": 0}
    for image_path in dataset.images:
        boxes, labels = read_yolo_label(dataset.label_path(image_path), 1, 1)
        if len(labels) == 0:
            counts["negatives"] += 1
            continue
        counts["smoke_boxes"] += int((labels == 0).sum())
        counts["fire_boxes"] += int((labels == 1).sum())
        counts["images_with_smoke"] += int((labels == 0).any())
        counts["images_with_fire"] += int((labels == 1).any())
    return counts


GLOW_DIM = GLOW_FEATURE_DIM
