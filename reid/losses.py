"""
losses.py
---------
Joint loss combining:
  1. Identity classification loss (label-smoothed cross-entropy)
  2. Triplet loss with batch-hard mining (metric learning on the
     pre-BN embedding)
  3. Attribute losses (BCE for binary attributes, CE for multiclass
     attribute groups)

This is the "new loss function that jointly trains ReID and attribute
learning" referenced in the write-up: a single scalar loss is produced
by weighting and summing the three components, which is what lets one
backward pass update the shared backbone using both identity and
attribute supervision simultaneously.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, num_classes: int, epsilon: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.epsilon / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class BatchHardTripletLoss(nn.Module):
    """
    Standard batch-hard triplet loss (Hermans et al.): for every anchor
    in the batch, mine the hardest positive (largest distance, same ID)
    and hardest negative (smallest distance, different ID), then apply a
    margin ranking loss.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    @staticmethod
    def _pairwise_distance(embeddings: torch.Tensor) -> torch.Tensor:
        dot = embeddings @ embeddings.t()
        sq_norm = dot.diagonal()
        dist_sq = sq_norm.unsqueeze(0) - 2 * dot + sq_norm.unsqueeze(1)
        return dist_sq.clamp(min=1e-12).sqrt()

    def forward(self, embeddings: torch.Tensor, identities: torch.Tensor) -> torch.Tensor:
        dist_mat = self._pairwise_distance(embeddings)
        n = embeddings.size(0)

        same_id = identities.unsqueeze(0) == identities.unsqueeze(1)
        diff_id = ~same_id

        hardest_positive = torch.stack([
            dist_mat[i][same_id[i]].max() if same_id[i].sum() > 1
            else torch.tensor(0.0, device=embeddings.device)
            for i in range(n)
        ])
        hardest_negative = torch.stack([
            dist_mat[i][diff_id[i]].min() if diff_id[i].any()
            else torch.tensor(float(self.margin), device=embeddings.device)
            for i in range(n)
        ])

        y = torch.ones_like(hardest_positive)
        return self.ranking_loss(hardest_negative, hardest_positive, y)


class AttributeLoss(nn.Module):
    """Combines per-group BCE (binary attrs) / CE (multiclass attrs)
    losses into a single averaged attribute loss."""

    def __init__(self, schema: List[Dict]):
        super().__init__()
        self.schema = schema

    def forward(
        self,
        attribute_logits: Dict[str, torch.Tensor],
        attribute_targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        for attr in self.schema:
            name, attr_type = attr["name"], attr["type"]
            if name not in attribute_targets:
                continue  # allow partially-labeled batches
            logit = attribute_logits[name]
            target = attribute_targets[name]
            mask = target >= 0  # -1 marks "unknown/unlabeled" for this sample
            if mask.sum() == 0:
                continue
            if attr_type == "binary":
                loss = F.binary_cross_entropy_with_logits(
                    logit.squeeze(-1)[mask], target[mask].float()
                )
            else:
                loss = F.cross_entropy(logit[mask], target[mask].long())
            losses.append(loss)
        if not losses:
            return torch.tensor(0.0, device=next(iter(attribute_logits.values())).device)
        return torch.stack(losses).mean()


class JointReIDAttributeLoss(nn.Module):
    """
    Total loss = w_id * ID-CE + w_triplet * Triplet + w_attr * AttributeLoss

    This is the single joint objective used to train MultiBranchReID
    end-to-end (backbone + identity_branch + attribute_branch together).
    """

    def __init__(
        self,
        num_identities: int,
        attribute_schema: List[Dict],
        w_id: float = 1.0,
        w_triplet: float = 1.0,
        w_attr: float = 0.5,
        triplet_margin: float = 0.3,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.id_loss = LabelSmoothingCrossEntropy(num_identities, epsilon=label_smoothing)
        self.triplet_loss = BatchHardTripletLoss(margin=triplet_margin)
        self.attribute_loss = AttributeLoss(attribute_schema)
        self.w_id, self.w_triplet, self.w_attr = w_id, w_triplet, w_attr

    def forward(
        self,
        model_output: Dict,
        identity_targets: torch.Tensor,
        attribute_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        id_loss = self.id_loss(model_output["logits"], identity_targets)
        triplet_loss = self.triplet_loss(model_output["triplet_feat"], identity_targets)

        if attribute_targets:
            attr_loss = self.attribute_loss(model_output["attribute_logits"], attribute_targets)
        else:
            attr_loss = torch.tensor(0.0, device=id_loss.device)

        total = (
            self.w_id * id_loss
            + self.w_triplet * triplet_loss
            + self.w_attr * attr_loss
        )
        return {
            "total_loss": total,
            "id_loss": id_loss,
            "triplet_loss": triplet_loss,
            "attribute_loss": attr_loss,
        }
