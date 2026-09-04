"""
embedding.py
------------
Top-level multi-branch ReID model: shared backbone + identity branch +
attribute branch(es). This is "the" model referenced elsewhere in the
pipeline as `MultiBranchReID`.

At inference time (tracking/association.py) only `embedding()` is used
to get a single L2-normalized appearance vector per detection crop;
the attribute outputs exist purely to shape that embedding through
joint training (see losses.py).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from reid.backbone import ReIDBackbone
from reid.identity_branch import IdentityBranch
from reid.attribute_branch import AttributeBranch, DEFAULT_ATTRIBUTE_SCHEMA


class MultiBranchReID(nn.Module):
    def __init__(
        self,
        num_identities: int = 0,
        embed_dim: int = 512,
        attribute_schema: Optional[List[Dict]] = None,
        pretrained_backbone: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.backbone = ReIDBackbone(pretrained=pretrained_backbone)
        self.identity_branch = IdentityBranch(
            in_channels=self.backbone.out_channels,
            embed_dim=embed_dim,
            num_identities=num_identities,
        )
        self.attribute_branch = AttributeBranch(
            in_channels=self.backbone.out_channels,
            schema=attribute_schema or DEFAULT_ATTRIBUTE_SCHEMA,
        )
        self.embed_dim = embed_dim

        # CPU-only environments are fully supported: default to "cpu" rather
        # than probing torch.cuda.is_available(), and keep the model + every
        # inference input pinned to self.device so nothing silently tries to
        # touch a GPU that may not exist.
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Dict:
        feature_map = self.backbone(x)
        identity_out = self.identity_branch(feature_map)
        attribute_logits = self.attribute_branch(feature_map)
        return {
            **identity_out,
            "attribute_logits": attribute_logits,
        }

    @torch.no_grad()
    def embed(self, crops: torch.Tensor) -> np.ndarray:
        """
        Inference-only helper: batch of pre-processed person crops
        (B, 3, H, W) -> (B, embed_dim) numpy array of L2-normalized
        embeddings, ready for cosine-distance ReID matching.
        """
        self.eval()
        crops = crops.to(self.device)
        out = self.forward(crops)
        return out["embedding"].detach().cpu().numpy()

    @torch.no_grad()
    def embed_and_attributes(self, crops: torch.Tensor) -> Dict[str, np.ndarray]:
        """Inference helper that also returns attribute predictions,
        used by the automatic-annotation pipeline (annotation/auto_attribute.py)."""
        self.eval()
        crops = crops.to(self.device)
        out = self.forward(crops)
        result = {"embedding": out["embedding"].cpu().numpy()}
        for name, logit in out["attribute_logits"].items():
            result[f"attr_{name}"] = logit.cpu().numpy()
        return result


def preprocess_crop(
    crop_bgr_or_rgb: np.ndarray,
    target_size: tuple = (256, 128),
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Resize + normalize a single HxWx3 uint8 crop into a model-ready tensor."""
    import cv2

    resized = cv2.resize(crop_bgr_or_rgb, (target_size[1], target_size[0]))
    img = resized.astype(np.float32) / 255.0
    img = (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    return tensor
