"""
track.py
--------
Represents a single tracked identity across time: its Kalman motion
state, trajectory history, and a rolling gallery of ReID embeddings used
for historical (multi-frame) appearance matching rather than a single
noisy per-frame embedding.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Deque, List, Optional

import numpy as np

from tracking.motion_model import KalmanBoxTracker
from tracking.trajectory import Trajectory, TrajectoryFeatures


class TrackState(Enum):
    TENTATIVE = "tentative"   # not yet confirmed (needs N consecutive hits)
    CONFIRMED = "confirmed"
    LOST = "lost"             # missed detections, kept alive for re-association
    REMOVED = "removed"       # deleted from the active track list


class Track:
    _next_id = 1

    def __init__(
        self,
        box_xyxy: np.ndarray,
        frame_id: int,
        embedding: Optional[np.ndarray] = None,
        embedding_gallery_size: int = 30,
        confirm_hits: int = 3,
        max_age: int = 30,
    ):
        self.track_id = Track._next_id
        Track._next_id += 1

        self.kalman = KalmanBoxTracker(box_xyxy)
        self.trajectory = Trajectory()
        self.trajectory.update(box_xyxy, frame_id)

        self.embedding_gallery: Deque[np.ndarray] = deque(maxlen=embedding_gallery_size)
        if embedding is not None:
            self.embedding_gallery.append(embedding)

        self.state = TrackState.TENTATIVE
        self.confirm_hits = confirm_hits
        self.max_age = max_age

        self.start_frame = frame_id
        self.last_update_frame = frame_id
        self.hits = 1
        self.misses = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def predict(self) -> np.ndarray:
        """Advance the motion model one frame; returns predicted xyxy box."""
        predicted_box = self.kalman.predict()
        return predicted_box

    def update(self, box_xyxy: np.ndarray, frame_id: int, embedding: Optional[np.ndarray] = None) -> None:
        self.kalman.update(box_xyxy)
        self.trajectory.update(box_xyxy, frame_id)

        if embedding is not None:
            self.embedding_gallery.append(embedding)

        self.hits += 1
        self.misses = 0
        self.last_update_frame = frame_id

        if self.state == TrackState.TENTATIVE and self.hits >= self.confirm_hits:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        self.misses += 1
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.REMOVED
        elif self.misses > self.max_age:
            self.state = TrackState.REMOVED
        else:
            self.state = TrackState.LOST

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    def is_removed(self) -> bool:
        return self.state == TrackState.REMOVED

    # ------------------------------------------------------------------ #
    # Feature accessors used by candidate_gating.py / association.py
    # ------------------------------------------------------------------ #
    @property
    def box_xyxy(self) -> np.ndarray:
        return self.kalman.box_xyxy

    def trajectory_features(self) -> Optional[TrajectoryFeatures]:
        return self.trajectory.extract_features()

    def mean_embedding(self) -> Optional[np.ndarray]:
        """Average gallery embedding, re-normalized -- more robust than a
        single embedding to transient occlusion/pose noise."""
        if not self.embedding_gallery:
            return None
        mean_vec = np.mean(np.stack(list(self.embedding_gallery)), axis=0)
        norm = np.linalg.norm(mean_vec)
        return mean_vec / norm if norm > 1e-6 else mean_vec

    def best_match_embedding_distance(self, query_embedding: np.ndarray) -> float:
        """
        Historical ReID matching: minimum cosine distance between the
        query embedding and *any* embedding in this track's gallery
        (more forgiving than only comparing to the mean, useful when
        appearance legitimately varies, e.g. the person turned around).
        """
        if not self.embedding_gallery:
            return 1.0
        gallery = np.stack(list(self.embedding_gallery))
        cosine_sims = gallery @ query_embedding
        return float(1.0 - np.max(cosine_sims))

    def __repr__(self) -> str:  # pragma: no cover
        return f"Track(id={self.track_id}, state={self.state.value}, hits={self.hits})"
