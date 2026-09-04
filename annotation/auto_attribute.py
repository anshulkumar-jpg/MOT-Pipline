"""
auto_attribute.py
------------------
Automatic data-annotation pipeline: generates pseudo-labels for person
attributes (clothing color/type, accessories, etc.) at scale, so the
multi-branch ReID model can be trained jointly on identity + attributes
without requiring exhaustive manual attribute annotation.

Design:
  * A pretrained attribute-tagging model (can be the AttributeBranch of
    a previously-trained MultiBranchReID, or any standalone tagger) is
    run over large unlabeled person-crop collections.
  * Predictions above a confidence threshold are kept as pseudo-labels;
    everything else is marked "unknown" (-1) so the joint loss can
    correctly ignore it per-sample (see reid/losses.py AttributeLoss).
  * A light temporal-consistency pass optionally smooths predictions
    across crops belonging to the same track, since a person's clothing
    attributes shouldn't flip frame-to-frame within one track.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from reid.attribute_branch import DEFAULT_ATTRIBUTE_SCHEMA


@dataclass
class AttributeAnnotationConfig:
    confidence_threshold: float = 0.75      # below this -> label = unknown (-1)
    binary_confidence_margin: float = 0.5   # |p - 0.5| must exceed this for binary attrs
    enforce_track_consistency: bool = True
    track_majority_ratio: float = 0.6       # fraction of crops that must agree to keep a track-level label


class AutoAttributeAnnotator:
    """
    Wraps a trained attribute model (e.g. `MultiBranchReID.attribute_branch`)
    and turns raw predictions into a pseudo-label dataset.
    """

    def __init__(
        self,
        attribute_model,
        schema: List[Dict] = None,
        config: AttributeAnnotationConfig = None,
        device: str = "cpu",
    ):
        self.model = attribute_model
        self.schema = schema or DEFAULT_ATTRIBUTE_SCHEMA
        self.config = config or AttributeAnnotationConfig()
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

    @torch.no_grad()
    def annotate_batch(self, crops: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        crops: (B, 3, H, W) preprocessed person crops.
        Returns dict[attribute_name] -> (B,) int array of pseudo-labels,
        with -1 marking "below confidence threshold / unknown".
        """
        self.model.eval()
        crops = crops.to(self.device)
        if hasattr(self.model, "backbone"):
            feature_map = self.model.backbone(crops)
            attr_head = getattr(self.model, "attribute_branch", self.model)
            predictions = attr_head.predict(feature_map) if hasattr(attr_head, "predict") else attr_head(feature_map)
        elif hasattr(self.model, "predict"):
            predictions = self.model.predict(crops)
        else:
            predictions = self.model(crops)


        labels: Dict[str, np.ndarray] = {}
        schema_by_name = {a["name"]: a for a in self.schema}

        for name, probs in predictions.items():
            probs_np = probs.detach().cpu().numpy()
            attr_type = schema_by_name[name]["type"]

            if attr_type == "binary":
                confident = np.abs(probs_np - 0.5) >= self.config.binary_confidence_margin
                pseudo = (probs_np >= 0.5).astype(np.int64)
                pseudo[~confident] = -1
            else:
                max_prob = probs_np.max(axis=-1)
                pseudo = probs_np.argmax(axis=-1).astype(np.int64)
                pseudo[max_prob < self.config.confidence_threshold] = -1

            labels[name] = pseudo

        return labels

    def annotate_dataset(
        self,
        crop_loader,
        track_ids: Optional[List[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Runs annotate_batch over a full DataLoader-like iterable of crop
        batches and concatenates results. If `track_ids` (parallel array,
        one id per crop across the whole dataset) is provided and
        enforce_track_consistency is set, per-track majority voting is
        applied as a smoothing pass -- see `_apply_track_consistency`.
        """
        all_labels: Dict[str, List[np.ndarray]] = {a["name"]: [] for a in self.schema}

        for batch in crop_loader:
            batch_labels = self.annotate_batch(batch)
            for name, arr in batch_labels.items():
                all_labels[name].append(arr)

        concatenated = {name: np.concatenate(arrs) for name, arrs in all_labels.items()}

        if self.config.enforce_track_consistency and track_ids is not None:
            concatenated = self._apply_track_consistency(concatenated, np.asarray(track_ids))

        return concatenated

    def _apply_track_consistency(
        self, labels: Dict[str, np.ndarray], track_ids: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        For each track, replace per-crop labels with the majority label
        across that track's crops, provided the majority clears
        `track_majority_ratio`; otherwise all crops in that track are
        marked unknown (-1) for that attribute rather than guessing.
        """
        smoothed = {name: arr.copy() for name, arr in labels.items()}
        unique_tracks = np.unique(track_ids)

        for name, arr in labels.items():
            for tid in unique_tracks:
                mask = track_ids == tid
                values = arr[mask]
                known = values[values >= 0]
                if len(known) == 0:
                    continue

                counts = np.bincount(known)
                majority_label = int(np.argmax(counts))
                majority_ratio = counts[majority_label] / len(known)

                if majority_ratio >= self.config.track_majority_ratio:
                    smoothed[name][mask] = majority_label
                else:
                    smoothed[name][mask] = -1

        return smoothed
