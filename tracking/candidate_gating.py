"""
candidate_gating.py
--------------------
Implements the "which detections are even physically/temporally plausible
for this track?" step that runs *before* ReID, as described in the
write-up: rather than comparing every track to every detection with
appearance alone, we first narrow the field using velocity, direction,
trajectory, box position, aspect ratio and its temporal change.

This keeps ReID from being blindly run against implausible detections
(cheaper, and avoids appearance-only mistakes in crowded scenes) while
still leaving enough candidates that a temporarily-occluded or
fast-moving person isn't gated out by geometry alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from tracking.geometry import iou_matrix, xyxy_to_cxcyah
from tracking.track import Track


@dataclass
class GatingConfig:
    mahalanobis_gate_threshold: float = 9.4877   # chi2 0.95 quantile, 4 dof
    max_center_distance: float = 150.0           # px, hard cap regardless of Mahalanobis
    direction_consistency_weight: float = 0.3
    aspect_ratio_change_tolerance: float = 0.35   # relative
    min_iou_for_fast_path: float = 0.3            # skip full gating if IoU already high


class CandidateGate:
    """
    Produces a boolean (num_tracks, num_detections) mask of admissible
    track/detection pairs, plus a continuous gating cost used to bias
    the final association cost matrix (see association.py).
    """

    def __init__(self, config: GatingConfig = None):
        self.config = config or GatingConfig()

    def gate(
        self,
        tracks: List[Track],
        detection_boxes: np.ndarray,
    ) -> np.ndarray:
        """
        Returns a (T, D) matrix of gating costs in [0, inf). A value of
        np.inf marks a pair as inadmissible (filtered out before ReID /
        final association ever sees it).
        """
        n_tracks, n_dets = len(tracks), len(detection_boxes)
        cost = np.zeros((n_tracks, n_dets), dtype=np.float32)

        if n_tracks == 0 or n_dets == 0:
            return cost

        track_boxes = np.stack([t.box_xyxy for t in tracks])
        ious = iou_matrix(track_boxes, detection_boxes)

        det_measurements = np.stack([xyxy_to_cxcyah(b) for b in detection_boxes])

        for i, track in enumerate(tracks):
            row_cost = self._gate_single_track(track, detection_boxes, det_measurements, ious[i])
            cost[i] = row_cost

        return cost

    def _gate_single_track(
        self,
        track: Track,
        detection_boxes: np.ndarray,
        det_measurements: np.ndarray,
        iou_row: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        n_dets = len(detection_boxes)
        cost = np.zeros(n_dets, dtype=np.float32)

        # Fast path: obviously overlapping detections skip heavier gating,
        # they're admissible with a small base cost.
        fast_path = iou_row >= cfg.min_iou_for_fast_path

        # 1) Motion gating via Mahalanobis distance in (cx, cy, aspect, h) space.
        maha = track.kalman.gating_distance(det_measurements)
        motion_gate_pass = maha <= cfg.mahalanobis_gate_threshold

        # 2) Hard cap on center displacement (guards against degenerate covariances
        #    early in a track's life, before the filter has "warmed up").
        track_center = track.box_xyxy.reshape(2, 2).mean(axis=0)
        det_centers = np.stack([
            [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in detection_boxes
        ])
        center_dist = np.linalg.norm(det_centers - track_center, axis=1)
        distance_gate_pass = center_dist <= cfg.max_center_distance

        admissible = fast_path | (motion_gate_pass & distance_gate_pass)

        # 3) Direction consistency: penalize (not hard-gate) detections that
        #    imply a sharp reversal from the track's established heading.
        traj_feats = track.trajectory_features()
        direction_penalty = np.zeros(n_dets, dtype=np.float32)
        if traj_feats is not None and traj_feats.speed > 1.0:
            implied_vectors = det_centers - track_center
            implied_norms = np.linalg.norm(implied_vectors, axis=1, keepdims=True)
            implied_norms = np.clip(implied_norms, 1e-6, None)
            implied_unit = implied_vectors / implied_norms

            track_heading = traj_feats.velocity / max(traj_feats.speed, 1e-6)
            cosine_alignment = implied_unit @ track_heading  # in [-1, 1]
            direction_penalty = cfg.direction_consistency_weight * (1.0 - cosine_alignment)

        # 4) Aspect-ratio-change plausibility: a person's box shape shouldn't
        #    flip drastically frame-to-frame absent a real pose/occlusion change.
        aspect_penalty = np.zeros(n_dets, dtype=np.float32)
        if traj_feats is not None:
            det_aspects = det_measurements[:, 2]
            rel_change = np.abs(det_aspects - traj_feats.aspect_ratio) / max(traj_feats.aspect_ratio, 1e-6)
            aspect_penalty = np.clip(rel_change - cfg.aspect_ratio_change_tolerance, 0, None)

        cost = maha + direction_penalty + aspect_penalty
        cost[~admissible] = np.inf
        return cost

    def candidate_indices(self, gating_cost_row: np.ndarray) -> np.ndarray:
        """Indices of admissible detections for one track (finite cost)."""
        return np.where(np.isfinite(gating_cost_row))[0]
