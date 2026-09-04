"""
geometry.py
-----------
Bounding-box geometry utilities shared across the tracking module:
format conversions, IoU, aspect ratio, and simple box-shape descriptors.

These are deliberately dependency-free (pure numpy) since they sit on
the hot path of per-frame association.
"""

from __future__ import annotations

import numpy as np


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def xyxy_to_cxcyah(box: np.ndarray) -> np.ndarray:
    """Center-x, center-y, aspect-ratio (w/h), height -- the classic
    Kalman-filter state representation used in SORT/DeepSORT-style trackers."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    aspect = w / max(h, 1e-6)
    return np.array([cx, cy, aspect, h], dtype=np.float32)


def cxcyah_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, aspect, h = state
    w = aspect * h
    x1, y1 = cx - w / 2.0, cy - h / 2.0
    x2, y2 = cx + w / 2.0, cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection-over-Union of two xyxy boxes."""
    xx1 = max(box_a[0], box_b[0])
    yy1 = max(box_a[1], box_b[1])
    xx2 = min(box_a[2], box_b[2])
    yy2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, xx2 - xx1)
    inter_h = max(0.0, yy2 - yy1)
    inter = inter_w * inter_h

    union = box_area(box_a) + box_area(box_b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized pairwise IoU. boxes_a: (N, 4), boxes_b: (M, 4) -> (N, M)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    xx1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    yy1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    xx2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    yy2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    union = np.clip(union, 1e-6, None)
    return inter / union


def aspect_ratio(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return w / max(h, 1e-6)


def center(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
