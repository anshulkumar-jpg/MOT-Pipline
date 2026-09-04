"""
motion_model.py
----------------
Constant-velocity Kalman filter over the (cx, cy, aspect, h) bounding-box
representation, extended with explicit velocity/acceleration bookkeeping
that trajectory.py and candidate_gating.py read out as motion features
(speed, direction, etc.).

State vector (8-dim):
    [cx, cy, aspect, h, vx, vy, v_aspect, vh]

This mirrors the SORT/DeepSORT Kalman design but is kept standalone
(no external filterpy dependency) so the whole repo only needs numpy.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.linalg import solve_triangular as _solve_triangular
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

from tracking.geometry import xyxy_to_cxcyah, cxcyah_to_xyxy


class KalmanBoxTracker:
    """Constant-velocity Kalman filter for a single track's bounding box."""

    def __init__(self, initial_box_xyxy: np.ndarray):
        ndim, dt = 4, 1.0

        # Motion model: x_{t+1} = F x_t
        self._F = np.eye(2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._F[i, ndim + i] = dt

        # Observation model: we observe (cx, cy, aspect, h) directly.
        self._H = np.eye(ndim, 2 * ndim, dtype=np.float32)

        # Motion / observation uncertainty weights (tuned like DeepSORT).
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        measurement = xyxy_to_cxcyah(initial_box_xyxy)
        self.mean = np.concatenate([measurement, np.zeros(ndim, dtype=np.float32)])

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        self.covariance = np.diag(np.square(std)).astype(np.float32)

        self.age = 0
        self.hits = 1
        self.time_since_update = 0

    # ------------------------------------------------------------------ #
    # Kalman predict / update
    # ------------------------------------------------------------------ #
    def predict(self) -> np.ndarray:
        std_pos = [
            self._std_weight_position * self.mean[3],
            self._std_weight_position * self.mean[3],
            1e-2,
            self._std_weight_position * self.mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * self.mean[3],
            self._std_weight_velocity * self.mean[3],
            1e-5,
            self._std_weight_velocity * self.mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)

        self.mean = self._F @ self.mean
        self.covariance = self._F @ self.covariance @ self._F.T + motion_cov

        self.age += 1
        self.time_since_update += 1
        return cxcyah_to_xyxy(self.mean[:4])

    def update(self, box_xyxy: np.ndarray) -> None:
        measurement = xyxy_to_cxcyah(box_xyxy)

        std = [
            self._std_weight_position * self.mean[3],
            self._std_weight_position * self.mean[3],
            1e-1,
            self._std_weight_position * self.mean[3],
        ]
        innovation_cov = np.diag(np.square(std)).astype(np.float32)

        projected_mean = self._H @ self.mean
        projected_cov = self._H @ self.covariance @ self._H.T + innovation_cov

        kalman_gain = (
            self.covariance @ self._H.T @ np.linalg.inv(projected_cov)
        )
        innovation = measurement - projected_mean

        self.mean = self.mean + kalman_gain @ innovation
        self.covariance = self.covariance - kalman_gain @ self._H @ self.covariance

        self.hits += 1
        self.time_since_update = 0

    # ------------------------------------------------------------------ #
    # Convenience accessors used by trajectory / gating logic
    # ------------------------------------------------------------------ #
    @property
    def box_xyxy(self) -> np.ndarray:
        return cxcyah_to_xyxy(self.mean[:4])

    @property
    def velocity(self) -> np.ndarray:
        """(vx, vy) in pixels/frame, in center-point space."""
        return self.mean[4:6].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def direction(self) -> float:
        """Heading angle in radians, 0 = pointing along +x axis."""
        vx, vy = self.velocity
        if abs(vx) < 1e-6 and abs(vy) < 1e-6:
            return 0.0
        return float(np.arctan2(vy, vx))

    @property
    def aspect_velocity(self) -> float:
        return float(self.mean[6])

    @property
    def height_velocity(self) -> float:
        return float(self.mean[7])

    def gating_distance(self, measurements: np.ndarray) -> np.ndarray:
        """
        Squared Mahalanobis distance between this track's predicted
        (cx, cy, aspect, h) distribution and a set of candidate
        measurements (N, 4) in the same representation. Used by
        candidate_gating.py to reject physically-implausible matches.
        """
        projected_mean = self._H @ self.mean
        projected_cov = self._H @ self.covariance @ self._H.T

        diff = measurements - projected_mean
        cholesky_factor = np.linalg.cholesky(projected_cov)
        if _HAS_SCIPY:
            z = _solve_triangular(cholesky_factor, diff.T, lower=True, check_finite=False)
        else:
            z = np.linalg.lstsq(cholesky_factor, diff.T, rcond=None)[0]
        squared_maha = np.sum(z ** 2, axis=0)
        return squared_maha
