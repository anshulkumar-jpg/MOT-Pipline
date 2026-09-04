"""
mot17.py
--------
MOT17-specific evaluation driver: iterates the standard MOT17 train/val
sequence layout, runs the tracker (via track_manager.py, fed by a
detector + ReID model) on each sequence, writes MOTChallenge-format
result files, and reports metrics via evaluation/metrics.py.

MOT17 ships three detector sets (DPM, FRCNN, SDP) per sequence; by
convention we evaluate against the provided public detections unless a
custom detector is supplied.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from detector.detector import PrecomputedDetector, Detection
from evaluation.metrics import compute_sequence_metrics, aggregate_metrics, format_report, SequenceResult
from tracking.track_manager import TrackManager, TrackManagerConfig

MOT17_TRAIN_SEQUENCES = [
    "MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09",
    "MOT17-10", "MOT17-11", "MOT17-13",
]

MOT17_DETECTOR_SUFFIXES = ["DPM", "FRCNN", "SDP"]


@dataclass
class MOT17EvalConfig:
    dataset_root: str                      # path to MOT17 root (contains train/, test/)
    output_root: str                        # where result .txt files are written
    sequences: List[str] = field(default_factory=lambda: list(MOT17_TRAIN_SEQUENCES))
    detector_suffix: str = "FRCNN"
    split: str = "train"                    # "train" (has GT) or "test" (no GT)
    detection_score_threshold: float = 0.3


class MOT17Evaluator:
    def __init__(self, config: MOT17EvalConfig, reid_model=None):
        self.config = config
        self.reid_model = reid_model
        os.makedirs(config.output_root, exist_ok=True)

    def _sequence_paths(self, sequence: str):
        seq_dir_name = f"{sequence}-{self.config.detector_suffix}"
        seq_dir = os.path.join(self.config.dataset_root, self.config.split, seq_dir_name)
        det_file = os.path.join(seq_dir, "det", "det.txt")
        gt_file = os.path.join(seq_dir, "gt", "gt.txt")
        img_dir = os.path.join(seq_dir, "img1")
        return seq_dir, det_file, gt_file, img_dir

    def run_sequence(self, sequence: str) -> str:
        """Runs the full tracker on one sequence, writes results, returns result file path."""
        seq_dir, det_file, gt_file, img_dir = self._sequence_paths(sequence)

        detector = PrecomputedDetector(det_file, score_threshold=self.config.detection_score_threshold)
        manager = TrackManager(TrackManagerConfig())

        num_frames = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else \
            int(max(detector._by_frame.keys())) if detector._by_frame else 0

        result_rows = []
        for frame_id in range(1, num_frames + 1):
            dets = detector.detect_frame_id(frame_id)
            boxes = np.stack([d.box for d in dets]) if dets else np.empty((0, 4), dtype=np.float32)
            scores = np.array([d.score for d in dets], dtype=np.float32)

            embeddings = self._embed_detections(seq_dir, img_dir, frame_id, dets) if self.reid_model else None

            manager.step(boxes, scores, embeddings)
            result_rows.extend(manager.results_as_mot_rows())

        result_path = os.path.join(self.config.output_root, f"{sequence}-{self.config.detector_suffix}.txt")
        self._write_results(result_rows, result_path)
        return result_path

    def _embed_detections(self, seq_dir, img_dir, frame_id, detections: List[Detection]) -> Optional[np.ndarray]:
        """Extracts and embeds person crops for the given frame's detections.
        Left as a hook: wiring in actual image loading (cv2.imread) + the
        ReID model's `embed()` happens here in a full training/eval run."""
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
        """Runs tracking + (if split == 'train') metric computation over all configured sequences."""
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
