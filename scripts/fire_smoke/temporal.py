"""Temporal confirmation: turning per-frame detections into an alarm.

A single-frame detector is the wrong unit of decision for a camera that runs
24/7. Even at a 1% per-frame false positive rate, 5 fps produces roughly 4300
false boxes a day, and nobody will keep watching a system like that.

Real fire and real smoke are *persistent and spatially stable*: they stay in
roughly the same place and grow. Glare on a windscreen, a passing forklift
headlight, a welding flash and a bird are not. So the alarm requires:

  1. `enter_hits` detections inside a sliding window of `window` frames,
  2. all matched to the same track by IoU, so scattered flicker never adds up,
  3. and once raised, the alarm only clears when the evidence drops below
     `exit_hits` (hysteresis, so a momentarily obscured flame does not toggle).

Everything here is intentionally simple and inspectable -- a supervisor can be
told "six confirmed detections of the same thing within three seconds" and
understand what the alarm means.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


def iou_1_to_n(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, np.float32)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[:, 0] * wh[:, 1]
    area_a = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    area_b = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    return (inter / np.maximum(area_a + area_b - inter, 1e-9)).astype(np.float32)


@dataclass
class Track:
    track_id: int
    label: int
    box: np.ndarray
    last_frame: int
    hits: deque = field(default_factory=deque)
    score: float = 0.0
    confirmed: bool = False

    def recent_hits(self, frame_index: int, window: int) -> int:
        while self.hits and self.hits[0] <= frame_index - window:
            self.hits.popleft()
        return len(self.hits)


@dataclass
class Alarm:
    track_id: int
    label: int
    class_name: str
    box: np.ndarray
    score: float
    hits: int
    window: int


class TemporalConfirmer:
    """Sliding-window, IoU-linked N-of-M confirmation with hysteresis."""

    def __init__(
        self,
        class_names: dict[int, str],
        window: int = 15,
        enter_hits: int = 6,
        exit_hits: int = 2,
        enter_conf: float = 0.50,
        exit_conf: float = 0.35,
        iou_match: float = 0.20,
        max_age: int = 20,
        box_smoothing: float = 0.6,
    ) -> None:
        self.class_names = class_names
        self.window = window
        self.enter_hits = enter_hits
        self.exit_hits = exit_hits
        self.enter_conf = enter_conf
        self.exit_conf = exit_conf
        self.iou_match = iou_match
        self.max_age = max_age
        self.box_smoothing = box_smoothing

        self.tracks: list[Track] = []
        self.frame_index = -1
        self._next_id = 1

    def update(self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> list[Alarm]:
        """Feed one frame of raw detections, get back the confirmed alarms."""
        self.frame_index += 1
        frame = self.frame_index

        order = np.argsort(-scores) if len(scores) else np.zeros(0, int)
        claimed: set[int] = set()

        for index in order:
            score = float(scores[index])
            # A detection below exit_conf is not evidence of anything.
            if score < self.exit_conf:
                continue
            box = np.asarray(boxes[index], np.float32)
            label = int(labels[index])

            candidates = [
                t for t in self.tracks
                if t.label == label and id(t) not in claimed and t.last_frame > frame - self.max_age
            ]
            best_track, best_iou = None, 0.0
            for track in candidates:
                overlap = float(iou_1_to_n(box, track.box[None, :])[0])
                if overlap > best_iou:
                    best_track, best_iou = track, overlap

            if best_track is not None and best_iou >= self.iou_match:
                alpha = self.box_smoothing
                best_track.box = alpha * best_track.box + (1 - alpha) * box
                best_track.score = max(score, 0.7 * best_track.score)
                best_track.last_frame = frame
                if score >= self.enter_conf:
                    best_track.hits.append(frame)
                claimed.add(id(best_track))
            elif score >= self.enter_conf:
                track = Track(self._next_id, label, box, frame, deque([frame]), score)
                self._next_id += 1
                self.tracks.append(track)
                claimed.add(id(track))

        self.tracks = [t for t in self.tracks if t.last_frame > frame - self.max_age]

        alarms = []
        for track in self.tracks:
            hits = track.recent_hits(frame, self.window)
            if track.confirmed:
                track.confirmed = hits >= self.exit_hits
            else:
                track.confirmed = hits >= self.enter_hits
            if track.confirmed:
                alarms.append(
                    Alarm(
                        track_id=track.track_id,
                        label=track.label,
                        class_name=self.class_names.get(track.label, str(track.label)),
                        box=track.box.copy(),
                        score=track.score,
                        hits=hits,
                        window=self.window,
                    )
                )
        return alarms

    def reset(self) -> None:
        self.tracks.clear()
        self.frame_index = -1


class ScalarConfirmer:
    """Same N-of-M rule for a signal with no box (the scene head, the glow cue)."""

    def __init__(self, window: int = 15, enter_hits: int = 8, exit_hits: int = 3,
                 enter_conf: float = 0.6, exit_conf: float = 0.45) -> None:
        self.window = window
        self.enter_hits = enter_hits
        self.exit_hits = exit_hits
        self.enter_conf = enter_conf
        self.exit_conf = exit_conf
        self.history: deque = deque(maxlen=window)
        self.active = False

    def update(self, value: float) -> bool:
        self.history.append(value >= self.enter_conf)
        hits = sum(self.history)
        if self.active:
            self.active = sum(1 for v in self.history if v) >= self.exit_hits
        else:
            self.active = hits >= self.enter_hits
        return self.active

    def reset(self) -> None:
        self.history.clear()
        self.active = False
