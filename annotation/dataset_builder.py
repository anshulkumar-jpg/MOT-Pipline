"""
dataset_builder.py
-------------------
Turns raw tracking output (video + per-frame boxes/track-ids) plus the
auto-generated attribute pseudo-labels (auto_attribute.py) into an
on-disk dataset directory ready for training MultiBranchReID:

    dataset_root/
        crops/
            <identity_id>/
                <track_id>_<frame_id>.jpg
        annotations.csv   # crop_path, identity_id, track_id, frame_id, <attr columns...>
        splits/
            train.txt
            val.txt

This is the "automatic data annotation pipeline" referenced in the
write-up: it removes the need to hand-label every person crop with
attributes, while still producing a standard identity-labeled ReID
dataset layout (compatible with Market-1501-style loaders) for the
cross-domain evaluation step.
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False


@dataclass
class CropRecord:
    crop: np.ndarray            # HxWx3 uint8
    identity_id: int            # ground-truth or track-derived pseudo identity
    track_id: int
    frame_id: int
    camera_id: int = 0
    attributes: Dict[str, int] = field(default_factory=dict)


class DatasetBuilder:
    def __init__(self, dataset_root: str, val_ratio: float = 0.1, seed: int = 42):
        self.dataset_root = dataset_root
        self.crops_dir = os.path.join(dataset_root, "crops")
        self.splits_dir = os.path.join(dataset_root, "splits")
        self.val_ratio = val_ratio
        self.rng = random.Random(seed)

        os.makedirs(self.crops_dir, exist_ok=True)
        os.makedirs(self.splits_dir, exist_ok=True)

        self._records: List[Tuple[str, CropRecord]] = []

    @staticmethod
    def extract_crop(frame: np.ndarray, box_xyxy: np.ndarray, padding: float = 0.0) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box_xyxy
        bw, bh = x2 - x1, y2 - y1
        x1 -= bw * padding
        x2 += bw * padding
        y1 -= bh * padding
        y2 += bh * padding

        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        return frame[y1:y2, x1:x2].copy()

    def add(self, record: CropRecord) -> str:
        identity_dir = os.path.join(self.crops_dir, str(record.identity_id))
        os.makedirs(identity_dir, exist_ok=True)

        filename = f"{record.track_id}_{record.frame_id}.jpg"
        path = os.path.join(identity_dir, filename)

        if _HAS_CV2:
            cv2.imwrite(path, record.crop)
        else:  # pragma: no cover - fallback without opencv
            np.save(path.replace(".jpg", ".npy"), record.crop)
            path = path.replace(".jpg", ".npy")

        self._records.append((path, record))
        return path

    def build_from_tracks(
        self,
        frames: List[np.ndarray],
        frame_track_boxes: List[List[Tuple[int, np.ndarray]]],
        attribute_labels: Optional[Dict[str, np.ndarray]] = None,
        camera_id: int = 0,
        padding: float = 0.1,
    ) -> None:
        """
        frames: list of raw video frames, indexed by frame_id
        frame_track_boxes[frame_id]: list of (track_id, box_xyxy) for that frame
        attribute_labels: optional, output of AutoAttributeAnnotator.annotate_dataset,
            aligned crop-by-crop in the same order records are added here.
        """
        crop_index = 0
        for frame_id, (frame, track_boxes) in enumerate(zip(frames, frame_track_boxes)):
            for track_id, box in track_boxes:
                crop = self.extract_crop(frame, box, padding=padding)
                if crop.size == 0:
                    continue

                attrs = {}
                if attribute_labels is not None:
                    for attr_name, values in attribute_labels.items():
                        attrs[attr_name] = int(values[crop_index])

                record = CropRecord(
                    crop=crop,
                    identity_id=track_id,   # track_id used as pseudo-identity within one sequence
                    track_id=track_id,
                    frame_id=frame_id,
                    camera_id=camera_id,
                    attributes=attrs,
                )
                self.add(record)
                crop_index += 1

    def write_annotations_csv(self, extra_attribute_names: Optional[List[str]] = None) -> str:
        csv_path = os.path.join(self.dataset_root, "annotations.csv")
        attr_names = extra_attribute_names or sorted({
            name for _, rec in self._records for name in rec.attributes.keys()
        })

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["crop_path", "identity_id", "track_id", "frame_id", "camera_id", *attr_names])
            for path, rec in self._records:
                row = [path, rec.identity_id, rec.track_id, rec.frame_id, rec.camera_id]
                row += [rec.attributes.get(name, -1) for name in attr_names]
                writer.writerow(row)

        return csv_path

    def write_splits(self) -> Tuple[str, str]:
        """Identity-disjoint train/val split: an identity's crops are
        entirely in one split, so val never leaks appearance of a training identity."""
        identities = sorted({rec.identity_id for _, rec in self._records})
        self.rng.shuffle(identities)

        n_val = max(1, int(len(identities) * self.val_ratio))
        val_identities = set(identities[:n_val])

        train_path = os.path.join(self.splits_dir, "train.txt")
        val_path = os.path.join(self.splits_dir, "val.txt")

        with open(train_path, "w") as f_train, open(val_path, "w") as f_val:
            for path, rec in self._records:
                target = f_val if rec.identity_id in val_identities else f_train
                target.write(f"{path}\n")

        return train_path, val_path
