"""
detector.py
-----------
Object (person) detector wrapper used as the first stage of the
Identity-Aware Multi-Object Tracking pipeline.

The detector is intentionally decoupled from any specific architecture:
by default it wraps a torchvision Faster R-CNN / RetinaNet style model,
but any detector that can produce (boxes, scores, classes) per frame can
be plugged in by implementing `BaseDetector`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
import os
from typing import List, Optional

import numpy as np

try:
    import torch
    import torchvision
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - environment without torch
    _HAS_TORCH = False


@dataclass
class Detection:
    """A single detection in a frame, in xyxy pixel coordinates."""
    box: np.ndarray          # shape (4,) -> x1, y1, x2, y2
    score: float
    class_id: int = 0        # 0 == person by convention in this pipeline

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.box
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.box
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


class BaseDetector(abc.ABC):
    """Interface every detector backend must implement."""

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR/RGB frame (H, W, 3)."""
        raise NotImplementedError


class TorchvisionDetector(BaseDetector):
    """
    Thin wrapper around a torchvision detection model
    (Faster R-CNN / RetinaNet / FCOS all share this calling convention).

    weights_path: optional path to fine-tuned weights (state_dict).
    If not provided, COCO-pretrained weights are used and only the
    "person" class (COCO id 1) is kept.
    """

    COCO_PERSON_CLASS_ID = 1

    def __init__(
        self,
        weights_path: Optional[str] = None,
        score_threshold: float = 0.5,
        device: Optional[str] = None,
        person_only: bool = True,
    ):
        if not _HAS_TORCH:
            raise RuntimeError(
                "torch/torchvision are required for TorchvisionDetector"
            )
        # "auto" (or None) picks GPU only if one is actually present; on a
        # CPU-only machine this resolves to "cpu" with no extra config needed.
        # Passing "cpu" explicitly always works and never touches torch.cuda.
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.score_threshold = score_threshold
        self.person_only = person_only

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT" if weights_path is None else None
        )
        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)

        self.model.eval().to(self.device)

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> List[Detection]:
        rgb_frame = frame[:, :, ::-1].copy() if frame.ndim == 3 and frame.shape[2] == 3 else frame
        tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device).unsqueeze(0)

        output = self.model(tensor)[0]
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()

        detections: List[Detection] = []
        for box, score, label in zip(boxes, scores, labels):
            if score < self.score_threshold:
                continue
            if self.person_only and label != self.COCO_PERSON_CLASS_ID:
                continue
            detections.append(Detection(box=box.astype(np.float32), score=float(score),
                                         class_id=int(label)))
        return detections


class PrecomputedDetector(BaseDetector):
    """
    Detector backend for offline evaluation on datasets (e.g. MOT17/MOT20)
    that ship public detections. Loads a MOTChallenge-format det.txt:

        frame, id(-1), x, y, w, h, score, class, visibility
    """

    def __init__(self, det_file: str, score_threshold: float = 0.0):
        self.score_threshold = score_threshold
        self._by_frame = self._load(det_file)

    @staticmethod
    def _load(det_file: str) -> dict:
        import os
        by_frame: dict = {}
        if not os.path.exists(det_file) or os.path.getsize(det_file) == 0:
            return by_frame
        data = np.loadtxt(det_file, delimiter=",")
        if data.size == 0:
            return by_frame
        if data.ndim == 1:
            data = data[None, :]
        for row in data:
            frame_id = int(row[0])
            x, y, w, h = row[2:6]
            score = float(row[6]) if row.shape[0] > 6 else 1.0
            box = np.array([x, y, x + w, y + h], dtype=np.float32)
            by_frame.setdefault(frame_id, []).append(Detection(box=box, score=score))
        return by_frame

    def detect_frame_id(self, frame_id: int) -> List[Detection]:
        dets = self._by_frame.get(frame_id, [])
        return [d for d in dets if d.score >= self.score_threshold]

    def detect(self, frame: np.ndarray) -> List[Detection]:  # pragma: no cover
        raise RuntimeError(
            "PrecomputedDetector is frame-id indexed; use detect_frame_id() "
            "when iterating a MOTChallenge sequence."
        )


class YOLODetector(BaseDetector):
    """
    Detector backend using Ultralytics YOLO/VOLO models (e.g. volo26n, yolov8n, etc.).
    Supports high-resolution inference (`imgsz`), lower confidence thresholds,
    and optional tiled/sliced grid inference for dense crowd detection.
    """

    COCO_PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "volo26n.pt",
        weights_path: Optional[str] = None,
        score_threshold: float = 0.10,
        imgsz: Optional[int] = 352,
        tiling: bool = True,
        grid_split_3x3: bool = True,
        include_full_frame: bool = False,
        tile_size: int = 352,
        tile_overlap: float = 0.15,
        device: Optional[str] = None,
        person_only: bool = True,
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "ultralytics package is required for YOLODetector. Install via `pip install ultralytics`."
            )

        if device is None or device == "auto":
            self.device = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self.score_threshold = score_threshold
        self.imgsz = imgsz
        self.tiling = tiling
        self.grid_split_3x3 = grid_split_3x3
        self.include_full_frame = include_full_frame
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.person_only = person_only

        target = weights_path or model_name
        if not os.path.exists(target):
            if os.path.exists(target.replace("volo", "yolo")):
                target = target.replace("volo", "yolo")
            elif target in ("volo26n.pt", "volo26n", "yolo26n.pt", "yolo26n", "volo26m.pt", "volo26m", "yolo26m.pt", "yolo26m"):
                if os.path.exists("yolov8n.pt"):
                    target = "yolov8n.pt"
                else:
                    target = "yolov8n.pt"
        self.model = YOLO(target)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.tiling:
            return self._detect_single(frame)
        return self._detect_tiled(frame)

    def _detect_single(self, frame: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> List[Detection]:
        kwargs = {"verbose": False, "device": self.device, "conf": self.score_threshold}
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        results = self.model(frame, **kwargs)[0]

        detections: List[Detection] = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, score, class_id in zip(boxes, scores, cls_ids):
                if score < self.score_threshold:
                    continue
                if self.person_only and class_id != self.COCO_PERSON_CLASS_ID:
                    continue
                adjusted_box = np.array(
                    [box[0] + offset_x, box[1] + offset_y, box[2] + offset_x, box[3] + offset_y],
                    dtype=np.float32,
                )
                detections.append(Detection(box=adjusted_box, score=float(score), class_id=int(class_id)))
        return detections

    def _detect_tiled(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        all_dets: List[Detection] = []

        # 1. Full frame pass (ONLY if include_full_frame is True)
        if self.include_full_frame:
            all_dets.extend(self._detect_single(frame))

        # 2. 3x3 Grid Split (9 parts)
        if self.grid_split_3x3:
            overlap = self.tile_overlap
            tile_h = h / 3.0
            tile_w = w / 3.0

            pad_h = int(tile_h * overlap)
            pad_w = int(tile_w * overlap)

            y_ranges = [
                (0, min(h, int(tile_h + pad_h))),
                (max(0, int(tile_h - pad_h)), min(h, int(2 * tile_h + pad_h))),
                (max(0, int(2 * tile_h - pad_h)), h),
            ]

            x_ranges = [
                (0, min(w, int(tile_w + pad_w))),
                (max(0, int(tile_w - pad_w)), min(w, int(2 * tile_w + pad_w))),
                (max(0, int(2 * tile_w - pad_w)), w),
            ]

            for y1, y2 in y_ranges:
                for x1, x2 in x_ranges:
                    tile = frame[y1:y2, x1:x2]
                    if tile.shape[0] < 32 or tile.shape[1] < 32:
                        continue
                    tile_dets = self._detect_single(tile, offset_x=x1, offset_y=y1)
                    all_dets.extend(tile_dets)
        else:
            stride = int(self.tile_size * (1.0 - self.tile_overlap))
            y_starts = list(range(0, h, stride))
            x_starts = list(range(0, w, stride))

            if y_starts[-1] + self.tile_size < h:
                y_starts.append(max(0, h - self.tile_size))
            if x_starts[-1] + self.tile_size < w:
                x_starts.append(max(0, w - self.tile_size))

            for y in y_starts:
                for x in x_starts:
                    tile = frame[y : y + self.tile_size, x : x + self.tile_size]
                    if tile.shape[0] < 32 or tile.shape[1] < 32:
                        continue
                    tile_dets = self._detect_single(tile, offset_x=x, offset_y=y)
                    all_dets.extend(tile_dets)

        if not all_dets:
            return []

        boxes = np.stack([d.box for d in all_dets])
        scores = np.array([d.score for d in all_dets], dtype=np.float32)

        keep_indices = self._nms(boxes, scores, iou_threshold=0.45)
        return [all_dets[i] for i in keep_indices]

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
        if _HAS_TORCH:
            import torchvision
            boxes_t = torch.from_numpy(boxes).float()
            scores_t = torch.from_numpy(scores).float()
            keep = torchvision.ops.nms(boxes_t, scores_t, iou_threshold)
            return keep.cpu().numpy()
        import cv2
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), score_threshold=0.0, nms_threshold=iou_threshold
        )
        return np.array(indices).flatten() if len(indices) > 0 else np.empty((0,), dtype=int)


