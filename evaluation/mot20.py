"""
mot20.py
--------
MOT20-specific evaluation driver. MOT20 sequences are extremely dense
crowd scenes -- this is precisely the regime where naive IoU association
breaks down (see tracking/candidate_gating.py / association.py docstrings)
and where the motion+ReID gating in this pipeline is expected to matter
most. Structurally this mirrors mot17.py; MOT20 differs mainly in that
it ships a single detection set per sequence (no DPM/FRCNN/SDP variants).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from detector.detector import PrecomputedDetector, Detection
from evaluation.metrics import compute_sequence_metrics, aggregate_metrics, format_report, SequenceResult
from tracking.track_manager import TrackManager, TrackManagerConfig
from tracking.candidate_gating import GatingConfig
from tracking.association import AssociationConfig

MOT20_TRAIN_SEQUENCES = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]


@dataclass
class MOT20EvalConfig:
    dataset_root: str
    output_root: str
    sequences: List[str] = field(default_factory=lambda: list(MOT20_TRAIN_SEQUENCES))
    split: str = "train"
    detection_score_threshold: float = 0.3
    # MOT20 crowds warrant a tighter spatial gate and a somewhat stronger
    # appearance weight than the MOT17 defaults, since box overlap alone
    # is far less reliable at this density.
    gating_config: GatingConfig = field(default_factory=lambda: GatingConfig(
        max_center_distance=100.0,
    ))
    association_config: AssociationConfig = field(default_factory=lambda: AssociationConfig(
        reid_weight=0.8,
        motion_weight=0.2,
        max_reid_distance=0.4,
    ))


class MOT20Evaluator:
    def __init__(self, config: MOT20EvalConfig, reid_model=None):
        self.config = config
        self.reid_model = reid_model
        os.makedirs(config.output_root, exist_ok=True)

    def _sequence_paths(self, sequence: str):
        seq_dir = os.path.join(self.config.dataset_root, self.config.split, sequence)
        det_file = os.path.join(seq_dir, "det", "det.txt")
        gt_file = os.path.join(seq_dir, "gt", "gt.txt")
        img_dir = os.path.join(seq_dir, "img1")
        return seq_dir, det_file, gt_file, img_dir

    def run_sequence(self, sequence: str) -> str:
        seq_dir, det_file, gt_file, img_dir = self._sequence_paths(sequence)

        detector = PrecomputedDetector(det_file, score_threshold=self.config.detection_score_threshold)
        manager = TrackManager(
            TrackManagerConfig(),
            gating_config=self.config.gating_config,
            association_config=self.config.association_config,
        )

        num_frames = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else \
            int(max(detector._by_frame.keys())) if detector._by_frame else 0

        result_rows = []
        for frame_id in range(1, num_frames + 1):
            dets = detector.detect_frame_id(frame_id)
            boxes = np.stack([d.box for d in dets]) if dets else np.empty((0, 4), dtype=np.float32)
            scores = np.array([d.score for d in dets], dtype=np.float32)

            embeddings = self._embed_detections(img_dir, frame_id, dets) if self.reid_model else None

            manager.step(boxes, scores, embeddings)
            result_rows.extend(manager.results_as_mot_rows())

        result_path = os.path.join(self.config.output_root, f"{sequence}.txt")
        self._write_results(result_rows, result_path)
        return result_path

    def _embed_detections(self, img_dir, frame_id, detections: List[Detection]) -> Optional[np.ndarray]:
        if not detections:
            return np.empty((0, self.reid_model.embed_dim), dtype=np.float32)

        import cv2
        from reid.embedding import preprocess_crop
        import torch

        frame_path = os.path.join(img_dir, f"{frame_id:06d}.jpg")
        frame = cv2.imread(frame_path)
        if frame is None:
            return np.zeros((len(detections), self.reid_model.embed_dim), dtype=np.float32)

        crops = []
        for det in detections:
            x1, y1, x2, y2 = det.box.astype(int)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
            crops.append(preprocess_crop(crop))

        batch = torch.stack(crops)
        return self.reid_model.embed(batch)

    @staticmethod
    def _write_results(rows: List[List[float]], path: str) -> None:
        with open(path, "w") as f:
            for row in rows:
                f.write(",".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row) + "\n")

    def evaluate(self) -> Optional[str]:
        results: List[SequenceResult] = []

        for sequence in self.config.sequences:
            result_path = self.run_sequence(sequence)

            if self.config.split == "train":
                _, _, gt_file, _ = self._sequence_paths(sequence)
                if os.path.exists(gt_file):
                    seq_result = compute_sequence_metrics(gt_file, result_path, sequence_name=sequence)
                    results.append(seq_result)

        if not results:
            return None

        agg = aggregate_metrics(results)
        return format_report(results, agg)
