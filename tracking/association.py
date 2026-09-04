"""
association.py
---------------
Final data-association step: combines the motion/geometry candidate
gating (candidate_gating.py) with historical ReID appearance matching
(track.py's embedding gallery) into one cost matrix, solved with the
Hungarian algorithm.

Pipeline per frame, matching the write-up's diagram:

    previous tracks -> candidate gating (motion/geometry) -> plausible
    detections -> ReID cost among plausible pairs -> Hungarian assignment
    -> final identities

Detections that survive gating for a track are scored by appearance;
detections that were gated out never get an (expensive, and potentially
misleading in a crowd) ReID comparison at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

from tracking.candidate_gating import CandidateGate, GatingConfig
from tracking.geometry import iou_matrix
from tracking.track import Track


@dataclass
class AssociationConfig:
    max_reid_distance: float = 0.45     # cosine distance; above this, appearance rejects the match
    reid_weight: float = 0.7            # blend between normalized motion gate cost and ReID cost
    motion_weight: float = 0.3
    unmatched_cost: float = 1e5         # effective "infinity" fed to Hungarian for gated-out pairs


@dataclass
class AssociationResult:
    matches: List[Tuple[int, int]]        # (track_idx, detection_idx)
    unmatched_tracks: List[int]
    unmatched_detections: List[int]


class Associator:
    """Runs candidate gating + appearance ReID + Hungarian assignment for one frame."""

    def __init__(
        self,
        gating_config: GatingConfig = None,
        association_config: AssociationConfig = None,
    ):
        self.gate = CandidateGate(gating_config)
        self.config = association_config or AssociationConfig()

    def associate(
        self,
        tracks: List[Track],
        detection_boxes: np.ndarray,
        detection_embeddings: Optional[np.ndarray] = None,
    ) -> AssociationResult:
        n_tracks, n_dets = len(tracks), len(detection_boxes)

        if n_tracks == 0 or n_dets == 0:
            return AssociationResult(
                matches=[],
                unmatched_tracks=list(range(n_tracks)),
                unmatched_detections=list(range(n_dets)),
            )

        gating_cost = self.gate.gate(tracks, detection_boxes)  # (T, D), inf where inadmissible
        final_cost = self._build_final_cost(tracks, detection_boxes, detection_embeddings, gating_cost)

        row_idx, col_idx = self._solve(final_cost)

        matches, unmatched_tracks, unmatched_dets = self._filter_assignment(
            row_idx, col_idx, final_cost, n_tracks, n_dets
        )
        return AssociationResult(matches, unmatched_tracks, unmatched_dets)

    # ------------------------------------------------------------------ #
    def _build_final_cost(
        self,
        tracks: List[Track],
        detection_boxes: np.ndarray,
        detection_embeddings: Optional[np.ndarray],
        gating_cost: np.ndarray,
    ) -> np.ndarray:
        n_tracks, n_dets = gating_cost.shape
        cost = np.full((n_tracks, n_dets), self.config.unmatched_cost, dtype=np.float32)

        # Normalize gating cost (finite entries only) into [0, 1] for blending with ReID distance.
        finite_mask = np.isfinite(gating_cost)
        if finite_mask.any():
            finite_vals = gating_cost[finite_mask]
            g_min, g_max = finite_vals.min(), finite_vals.max()
            norm_gating = np.zeros_like(gating_cost)
            norm_gating[finite_mask] = (
                (gating_cost[finite_mask] - g_min) / max(g_max - g_min, 1e-6)
            )
        else:
            norm_gating = np.zeros_like(gating_cost)

        for i, track in enumerate(tracks):
            candidate_js = self.gate.candidate_indices(gating_cost[i])
            if len(candidate_js) == 0:
                continue

            for j in candidate_js:
                motion_component = norm_gating[i, j]

                if detection_embeddings is not None:
                    reid_distance = track.best_match_embedding_distance(detection_embeddings[j])
                    if reid_distance > self.config.max_reid_distance:
                        continue  # appearance vetoes this candidate
                else:
                    reid_distance = 0.0

                blended = (
                    self.config.motion_weight * motion_component
                    + self.config.reid_weight * reid_distance
                )
                cost[i, j] = blended

        return cost

    @staticmethod
    def _solve(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if _HAS_SCIPY:
            return linear_sum_assignment(cost)
        return Associator._greedy_assignment(cost)

    @staticmethod
    def _greedy_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback if scipy is unavailable: greedy nearest-cost matching."""
        cost = cost.copy()
        n_tracks, n_dets = cost.shape
        rows, cols = [], []
        used_rows, used_cols = set(), set()

        flat_indices = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for r, c in flat_indices:
            if r in used_rows or c in used_cols:
                continue
            used_rows.add(r)
            used_cols.add(c)
            rows.append(r)
            cols.append(c)
        return np.array(rows), np.array(cols)

    def _filter_assignment(
        self,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        cost: np.ndarray,
        n_tracks: int,
        n_dets: int,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        matches = []
        matched_tracks, matched_dets = set(), set()

        for r, c in zip(row_idx, col_idx):
            if cost[r, c] >= self.config.unmatched_cost:
                continue  # Hungarian still returns a pairing even for "infinite" cost cells
            matches.append((int(r), int(c)))
            matched_tracks.add(int(r))
            matched_dets.add(int(c))

        unmatched_tracks = [t for t in range(n_tracks) if t not in matched_tracks]
        unmatched_dets = [d for d in range(n_dets) if d not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets


def iou_fallback_association(
    tracks: List[Track], detection_boxes: np.ndarray, iou_threshold: float = 0.3
) -> AssociationResult:
    """
    Plain IoU-based Hungarian matching, kept only as a baseline/ablation
    reference to compare against the full motion+ReID pipeline above --
    this is deliberately the "traditional approach" described in the
    write-up, not used by the main tracker.
    """
    n_tracks, n_dets = len(tracks), len(detection_boxes)
    if n_tracks == 0 or n_dets == 0:
        return AssociationResult([], list(range(n_tracks)), list(range(n_dets)))

    track_boxes = np.stack([t.box_xyxy for t in tracks])
    ious = iou_matrix(track_boxes, detection_boxes)
    cost = 1.0 - ious

    if _HAS_SCIPY:
        row_idx, col_idx = linear_sum_assignment(cost)
    else:
        row_idx, col_idx = Associator._greedy_assignment(cost)

    matches, matched_t, matched_d = [], set(), set()
    for r, c in zip(row_idx, col_idx):
        if ious[r, c] < iou_threshold:
            continue
        matches.append((int(r), int(c)))
        matched_t.add(int(r))
        matched_d.add(int(c))

    unmatched_tracks = [t for t in range(n_tracks) if t not in matched_t]
    unmatched_dets = [d for d in range(n_dets) if d not in matched_d]
    return AssociationResult(matches, unmatched_tracks, unmatched_dets)
