"""
trajectory.py
-------------
Extracts the historical-motion feature set described in the write-up:
velocity, direction, trajectory shape, bounding-box position, aspect
ratio, and the temporal change of the aspect ratio / box size.

`Trajectory` accumulates per-track history frame by frame; `TrajectoryFeatures`
is the compact numeric summary handed to candidate_gating.py and
association.py.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np

from tracking.geometry import aspect_ratio, center, box_area


@dataclass
class TrajectoryFeatures:
    velocity: np.ndarray          # (vx, vy) pixels/frame
    speed: float                  # magnitude of velocity
    direction: float               # radians
    position: np.ndarray           # (cx, cy) current center
    aspect_ratio: float
    aspect_ratio_delta: float      # change vs previous frame
    height: float
    height_delta: float
    area_delta_ratio: float        # (area_t - area_{t-1}) / area_{t-1}


class Trajectory:
    """
    Rolling history of a single track's bounding boxes, used to derive
    smoothed motion/geometry features that are more robust than a
    single-frame velocity estimate (e.g. under brief occlusion / detector
    jitter).
    """

    def __init__(self, max_history: int = 30, smoothing_window: int = 5):
        self.boxes: Deque[np.ndarray] = deque(maxlen=max_history)
        self.frame_ids: Deque[int] = deque(maxlen=max_history)
        self.smoothing_window = smoothing_window

    def update(self, box_xyxy: np.ndarray, frame_id: int) -> None:
        self.boxes.append(np.asarray(box_xyxy, dtype=np.float32))
        self.frame_ids.append(frame_id)

    def __len__(self) -> int:
        return len(self.boxes)

    def _smoothed_velocity(self) -> np.ndarray:
        if len(self.boxes) < 2:
            return np.zeros(2, dtype=np.float32)

        window = min(self.smoothing_window, len(self.boxes) - 1)
        centers = [center(b) for b in list(self.boxes)[-(window + 1):]]
        deltas = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        return np.mean(deltas, axis=0).astype(np.float32)

    def extract_features(self) -> Optional[TrajectoryFeatures]:
        if len(self.boxes) == 0:
            return None

        current = self.boxes[-1]
        previous = self.boxes[-2] if len(self.boxes) >= 2 else current

        velocity = self._smoothed_velocity()
        speed = float(np.linalg.norm(velocity))
        direction = float(np.arctan2(velocity[1], velocity[0])) if speed > 1e-6 else 0.0

        cur_ar, prev_ar = aspect_ratio(current), aspect_ratio(previous)
        cur_h = current[3] - current[1]
        prev_h = previous[3] - previous[1]

        cur_area = box_area(current)
        prev_area = box_area(previous)
        area_delta_ratio = (cur_area - prev_area) / max(prev_area, 1e-6)

        return TrajectoryFeatures(
            velocity=velocity,
            speed=speed,
            direction=direction,
            position=center(current),
            aspect_ratio=cur_ar,
            aspect_ratio_delta=cur_ar - prev_ar,
            height=cur_h,
            height_delta=cur_h - prev_h,
            area_delta_ratio=float(area_delta_ratio),
        )

    def predict_next_position(self) -> np.ndarray:
        """Simple linear extrapolation of the center point one frame ahead,
        used as a cheap prior before the Kalman filter's own predict step."""
        feats = self.extract_features()
        if feats is None:
            return np.zeros(2, dtype=np.float32)
        return feats.position + feats.velocity

    def recent_boxes(self, n: int = 5) -> List[np.ndarray]:
        return list(self.boxes)[-n:]
