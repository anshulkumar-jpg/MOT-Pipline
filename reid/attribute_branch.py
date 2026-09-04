"""
attribute_branch.py
--------------------
Auxiliary head(s) that predict complementary, human-interpretable
attributes from the same shared backbone feature map used by the
identity branch: clothing color/type, accessories, body attributes, etc.

These branches are not meant to be used standalone for retrieval; their
purpose is to regularize the shared backbone during joint training so
that the resulting representation captures more than a single "identity"
signal, which is what makes it more robust to appearance change and
cross-camera/cross-domain shifts (see losses.py for how the branches are
combined into one joint loss).
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# Default attribute schema. Each entry is (attribute_group_name, num_classes).
# Binary attributes (e.g. "wears_backpack") use num_classes = 1 (sigmoid),
# multi-class groups (e.g. "upper_body_color") use num_classes = K (softmax).
DEFAULT_ATTRIBUTE_SCHEMA: List[Dict] = [
    {"name": "upper_body_color", "num_classes": 11, "type": "multiclass"},
    {"name": "lower_body_color", "num_classes": 11, "type": "multiclass"},
    {"name": "upper_body_type", "num_classes": 4, "type": "multiclass"},   # e.g. shirt/jacket/tshirt/other
    {"name": "lower_body_type", "num_classes": 4, "type": "multiclass"},   # pants/shorts/skirt/other
    {"name": "has_backpack", "num_classes": 1, "type": "binary"},
    {"name": "has_bag", "num_classes": 1, "type": "binary"},
    {"name": "has_hat", "num_classes": 1, "type": "binary"},
    {"name": "gender", "num_classes": 1, "type": "binary"},
    {"name": "age_group", "num_classes": 4, "type": "multiclass"},
]


class AttributeBranch(nn.Module):
    """
    feature map (B, C, H, W) -> dict of per-attribute-group logits.

    A lightweight shared FC trunk feeds independent small heads per
    attribute group, keeping the branch cheap relative to the identity
    branch while still letting each attribute specialize.
    """

    def __init__(self, in_channels: int, schema: List[Dict] = None, hidden_dim: int = 512):
        super().__init__()
        self.schema = schema or DEFAULT_ATTRIBUTE_SCHEMA
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.trunk = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.heads = nn.ModuleDict({
            attr["name"]: nn.Linear(hidden_dim, attr["num_classes"])
            for attr in self.schema
        })

    def forward(self, feature_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = self.pool(feature_map).flatten(1)
        shared = self.trunk(pooled)
        return {name: head(shared) for name, head in self.heads.items()}

    def predict(self, feature_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience inference helper returning class predictions / probabilities
        instead of raw logits, keyed by attribute-group name."""
        logits = self.forward(feature_map)
        out: Dict[str, torch.Tensor] = {}
        schema_by_name = {attr["name"]: attr for attr in self.schema}
        for name, logit in logits.items():
            attr_type = schema_by_name[name]["type"]
            if attr_type == "binary":
                out[name] = torch.sigmoid(logit).squeeze(-1)
            else:
                out[name] = F.softmax(logit, dim=-1)
        return out
