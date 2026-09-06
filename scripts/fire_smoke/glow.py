"""Hand-crafted fire-glow prior.

Requirement 3 of the project: a fire may be hidden behind machinery or a wall
and never appear as a flame-shaped object, but it still throws warm, flickering
light onto whatever is around it. A box detector trained on visible flames has
nothing to latch onto in that case.

So we compute a cheap, purely classical "glow" response:

    glow = warm_tint * local_brightness_excess

`warm_tint` is high where a pixel is redder than it is blue/green, which is what
firelight does to most surfaces. `local_brightness_excess` is high where a pixel
is brighter than its own neighbourhood, which is what a light *source* spilling
onto a surface looks like -- as opposed to a uniformly bright daylight scene.

Two things use this:
  * a 7-number summary is fed to the scene classifier as extra input, so the
    network can learn when the glow statistics actually mean fire and when they
    are just a sunset or a sodium lamp,
  * the map itself localises the "possible occluded fire" marker at inference,
    *after* the learned scene head has already decided that there is a fire.

## Measured behaviour (D-Fire val, 100 fire vs 100 verified-negative)

This prior is **not** a fire detector, and the numbers say so plainly.
ROC AUC, fire versus verified-negative, per `scripts/analyze_glow_prior.py`:

    feature             as-is   night   night+IR
    warm_cast           0.725   0.830     0.500
    glow_p99            0.145   0.169     0.500
    glow_mean           0.145   0.202     0.500
    frac_fire_chroma    0.189   0.493     0.500
    frac_highlight      0.534   0.627     0.627

Three things follow, and all three matter:

1. `warm_cast` is the only strong positive cue, and it gets *stronger* in the
   dark (0.725 -> 0.830) -- exactly the condition where the box detector has
   least to work with. This is the feature that earns the prior its place.
2. The `glow_*` statistics are **inverted**: D-Fire negatives (sunsets, warm
   indoor and street lighting) out-glow real fires. They are kept because an
   inverted-but-informative feature is still usable by a learned classifier,
   not because a high glow response means fire. Do not threshold them directly.
3. Under IR / grayscale every colour-derived feature collapses to exactly
   chance, because there is no colour left. On a mono night camera this prior
   contributes nothing but `frac_highlight`, and detection rests entirely on
   the DINOv3 features. That is a real limitation, not a tuning problem.

## Features deliberately NOT included

Global exposure statistics -- mean luminance, luminance std, dark-pixel fraction
-- are by far the strongest discriminators on D-Fire (AUC 0.82-0.84). They are
also a **dataset artifact**: fire photographs are disproportionately taken at
night, so "dark image" predicts "fire" in this data. On a camera that is dark
every single night, a model leaning on that would alarm on darkness itself,
which is the exact failure mode requirement 2 forbids. They are excluded here,
and `night` augmentation attacks the same shortcut from the data side.

Note this does not eliminate the shortcut -- the DINOv3 tokens encode scene
brightness too -- it only removes the most direct path to it. Re-check with
`scripts/analyze_glow_prior.py`.
"""

from __future__ import annotations

import cv2
import numpy as np

GLOW_FEATURE_DIM = 7
GLOW_FEATURE_NAMES = (
    "frac_highlight",
    "frac_fire_chroma",
    "glow_mean",
    "glow_p99",
    "frac_glow_strong",
    "largest_glow_blob",
    "warm_cast",
)
# Excluded on purpose - see the module docstring.
EXCLUDED_EXPOSURE_FEATURES = ("mean_luma", "std_luma", "frac_dark")
# Everything is computed on a downscaled copy; the prior is low-frequency by
# construction and this keeps it cheap enough for 24/7 video.
WORK_SIZE = 256


def _downscale(rgb: np.ndarray, max_side: int = WORK_SIZE) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return rgb
    return cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def glow_map(rgb: np.ndarray, downscale: bool = True) -> np.ndarray:
    """Return a float32 glow response in [0, 1] for an RGB uint8 image."""
    small = _downscale(rgb) if downscale else rgb
    img = small.astype(np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b

    # Warm tint: redder than blue, and not less than green.
    warm = np.clip((r - b) / 0.35, 0.0, 1.0) * np.clip((r - g) / 0.20 + 0.2, 0.0, 1.0)

    # Local brightness excess relative to a wide blur of the luminance.
    sigma = max(3.0, max(small.shape[:2]) / 16.0)
    background = cv2.GaussianBlur(luma, (0, 0), sigma)
    excess = np.clip((luma - background) / 0.15, 0.0, 1.0)

    return (warm * excess).astype(np.float32)


def fire_chroma_mask(rgb: np.ndarray) -> np.ndarray:
    """Classic R > G > B flame-colour rule, kept as a separate statistic."""
    img = rgb.astype(np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    r_threshold = 140.0
    sat_rule = saturation >= ((255.0 - r) * 0.35 / r_threshold)
    return ((r > r_threshold) & (r > g) & (g > b) & sat_rule).astype(np.float32)


def glow_features(rgb: np.ndarray) -> np.ndarray:
    """Return the fixed-length glow descriptor for an RGB uint8 image."""
    small = _downscale(rgb)
    img = small.astype(np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b

    gmap = glow_map(small, downscale=False)
    chroma = fire_chroma_mask(small)

    strong = (gmap > 0.40).astype(np.uint8)
    if strong.any():
        count, _, stats, _ = cv2.connectedComponentsWithStats(strong, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.zeros(1)
        largest = float(areas.max()) / float(strong.size)
    else:
        largest = 0.0

    features = np.array(
        [
            float((luma > 0.90).mean()),  # blown-out highlights: flame cores
            float(chroma.mean()),
            float(gmap.mean()),
            float(np.percentile(gmap, 99)),
            float((gmap > 0.40).mean()),
            largest,
            # Warm colour cast: the strongest measured cue, and it strengthens
            # in low light, which is exactly where the box detector struggles.
            float(np.clip((r - b).mean() * 4.0 + 0.5, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)


def glow_hotspot(rgb: np.ndarray, min_response: float = 0.35) -> tuple[tuple[int, int, int, int], float] | None:
    """Locate the strongest glow blob, in full-resolution xyxy pixel coords.

    Used at inference to put a box around a "possible occluded fire" cue when
    the scene head fires but the detector found no flame.
    """
    h, w = rgb.shape[:2]
    small = _downscale(rgb)
    sh, sw = small.shape[:2]
    gmap = glow_map(small, downscale=False)

    mask = (gmap > min_response).astype(np.uint8)
    if not mask.any():
        return None
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = (
        stats[best, cv2.CC_STAT_LEFT],
        stats[best, cv2.CC_STAT_TOP],
        stats[best, cv2.CC_STAT_WIDTH],
        stats[best, cv2.CC_STAT_HEIGHT],
    )
    strength = float(gmap[labels == best].mean())

    scale_x, scale_y = w / sw, h / sh
    box = (
        int(x * scale_x),
        int(y * scale_y),
        int((x + bw) * scale_x),
        int((y + bh) * scale_y),
    )
    return box, strength
