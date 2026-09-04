"""
identity_branch.py
-------------------
Head that turns the shared backbone feature map into a compact,
L2-normalized identity embedding, plus a classification logit head used
only during training (BNNeck-style ID loss).

This branch answers: "which identity is this?" It is trained jointly
with the attribute branch (see attribute_branch.py / losses.py) so that
the resulting embedding is informed by, but not limited to, raw
appearance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedMeanPooling(nn.Module):
    """GeM pooling: a learnable generalization of avg/max pooling that
    tends to work well for ReID retrieval embeddings."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / self.p)


class IdentityBranch(nn.Module):
    """
    feature map (B, C, H, W) -> embedding (B, embed_dim), logits (B, num_ids)

    Follows the common "BNNeck" trick: the triplet loss is applied to the
    pre-BN feature, while the classification (ID) loss is applied to the
    post-BN feature; at inference time only the post-BN embedding is used
    for retrieval, which empirically improves both metrics simultaneously.
    """

    def __init__(self, in_channels: int, embed_dim: int = 512, num_identities: int = 0):
        super().__init__()
        self.pool = GeneralizedMeanPooling()
        self.reduce = nn.Linear(in_channels, embed_dim)
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)  # no-bias BNNeck

        self.classifier: nn.Module
        if num_identities > 0:
            self.classifier = nn.Linear(embed_dim, num_identities, bias=False)
        else:
            self.classifier = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.reduce.weight, mode="fan_out")
        nn.init.constant_(self.reduce.bias, 0.0)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        if isinstance(self.classifier, nn.Linear):
            nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, feature_map: torch.Tensor):
        pooled = self.pool(feature_map).flatten(1)          # (B, C)
        triplet_feat = self.reduce(pooled)                  # pre-BN, for triplet loss
        embedding = self.bottleneck(triplet_feat)            # post-BN, for retrieval / ID loss

        logits = None
        if not isinstance(self.classifier, nn.Identity):
            logits = self.classifier(embedding)

        normalized_embedding = F.normalize(embedding, p=2, dim=1)
        return {
            "embedding": normalized_embedding,   # used at inference (cosine distance)
            "triplet_feat": triplet_feat,        # used for triplet loss during training
            "logits": logits,                    # used for ID cross-entropy loss
        }
