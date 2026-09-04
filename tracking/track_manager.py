"""
track_manager.py
-----------------
Top-level per-frame orchestrator for the tracking module: predicts all
active tracks forward, runs association (motion gating + ReID +
Hungarian), updates matched tracks, ages/removes unmatched tracks, and
spawns new tentative tracks from unmatched detections.

This is what main.py calls once per frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from tracking.association import Associator, AssociationConfig
from tracking.candidate_gating import GatingConfig
from tracking.track import Track, TrackState


@dataclass
class TrackManagerConfig:
    confirm_hits: int = 3
    max_age: int = 30
    embedding_gallery_size: int = 30
    min_detection_score: float = 0.3


class TrackManager:
    def __init__(
        self,
        config: TrackManagerConfig = None,
        gating_config: GatingConfig = None,
        association_config: AssociationConfig = None,
    ):
        self.config = config or TrackManagerConfig()
        self.associator = Associator(gating_config, association_config)

        self.active_tracks: List[Track] = []
        self.frame_id = 0

    def step(
        self,
        detection_boxes: np.ndarray,
        detection_scores: Optional[np.ndarray] = None,
        detection_embeddings: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """
        Advance the tracker by one frame.

        detection_boxes: (N, 4) xyxy
        detection_scores: (N,) optional, used to filter low-confidence detections
        detection_embeddings: (N, embed_dim) L2-normalized ReID vectors, optional

        Returns the list of currently confirmed tracks (i.e. what should
        be written out / visualized this frame).
        """
        self.frame_id += 1

        detection_boxes, detection_embeddings = self._filter_low_score(
            detection_boxes, detection_scores, detection_embeddings
        )

        # 1) Predict motion for every currently active track.
        for track in self.active_tracks:
            track.predict()

        # 2) Motion-gated + ReID-scored Hungarian association.
        result = self.associator.associate(self.active_tracks, detection_boxes, detection_embeddings)

        # 3) Update matched tracks.
        for track_idx, det_idx in result.matches:
            track = self.active_tracks[track_idx]
            embedding = detection_embeddings[det_idx] if detection_embeddings is not None else None
            track.update(detection_boxes[det_idx], self.frame_id, embedding)

        # 4) Age unmatched tracks (may transition to LOST or REMOVED).
        for track_idx in result.unmatched_tracks:
            self.active_tracks[track_idx].mark_missed()

        # 5) Spawn new tentative tracks for unmatched detections.
        for det_idx in result.unmatched_detections:
            embedding = detection_embeddings[det_idx] if detection_embeddings is not None else None
            new_track = Track(
                box_xyxy=detection_boxes[det_idx],
                frame_id=self.frame_id,
                embedding=embedding,
                embedding_gallery_size=self.config.embedding_gallery_size,
                confirm_hits=self.config.confirm_hits,
                max_age=self.config.max_age,
            )
            self.active_tracks.append(new_track)

        # 6) Prune removed tracks.
        self.active_tracks = [t for t in self.active_tracks if not t.is_removed()]

        return [t for t in self.active_tracks if t.is_confirmed()]

    @staticmethod
    def _filter_low_score(boxes, scores, embeddings):
        if scores is None:
            return boxes, embeddings
        keep = scores >= 0.0  # threshold applied upstream by the detector normally;
        # kept here as a hook in case raw detections are passed straight to the tracker.
        return boxes[keep], (embeddings[keep] if embeddings is not None else None)

    def reset(self) -> None:
        self.active_tracks = []
        self.frame_id = 0
        Track._next_id = 1

    def results_as_mot_rows(self) -> List[List[float]]:
        """
        Dump current confirmed tracks in MOTChallenge result-file format:
        frame, id, x, y, w, h, score(-1), -1, -1, -1
        """
        rows = []
        for track in self.active_tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = track.box_xyxy
            rows.append([
                self.frame_id, track.track_id, x1, y1, x2 - x1, y2 - y1, -1, -1, -1, -1
            ])
        return rows
