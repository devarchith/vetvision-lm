"""
CLIP-style symmetric contrastive (InfoNCE) loss for VetVision-LM.

L_contrastive = (1/2) * (CE(sim / τ, labels) + CE(sim^T / τ, labels))

where sim[i, j] = vision_embed[i] · text_embed[j]
      τ          = learnable (or fixed) temperature
      labels      = arange(B)   (diagonal matches)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ContrastiveLoss(nn.Module):
    """
    Symmetric InfoNCE / CLIP contrastive loss.

    Args:
        temperature: Initial temperature τ.  Can be made learnable.
        learnable_temp: If True, τ is a learnable ``nn.Parameter``.
        min_temp: Minimum clamped temperature (prevents collapse).
        max_temp: Maximum clamped temperature.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temp: bool = False,
        min_temp: float = 0.01,
        max_temp: float = 0.5,
    ) -> None:
        super().__init__()
        self.min_temp = min_temp
        self.max_temp = max_temp

        if learnable_temp:
            # Initialise logit_scale = log(1/τ) following CLIP
            self.logit_scale = nn.Parameter(
                torch.tensor(1.0 / temperature).log()
            )
        else:
            self.register_buffer(
                "logit_scale",
                torch.tensor(1.0 / temperature).log(),
            )

    @property
    def temperature(self) -> float:
        return (1.0 / self.logit_scale.exp()).item()

    def forward(
        self,
        vision_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute symmetric contrastive loss.

        Args:
            vision_embed: (B, D) L2-normalised image embeddings.
            text_embed:   (B, D) L2-normalised text embeddings.

        Returns:
            Scalar loss tensor.
        """
        B = vision_embed.size(0)
        device = vision_embed.device

        # Clamp logit_scale for training stability
        logit_scale = self.logit_scale.clamp(
            max=torch.tensor(1.0 / self.min_temp).log().item(),
            min=torch.tensor(1.0 / self.max_temp).log().item(),
        )
        scale = logit_scale.exp()

        # Cosine similarity matrix (B, B)
        # Both embeds are already L2-normalised so dot-product = cosine sim
        logits_v2t = scale * vision_embed @ text_embed.T   # (B, B)
        logits_t2v = logits_v2t.T                          # (B, B)

        labels = torch.arange(B, device=device)

        loss_v2t = F.cross_entropy(logits_v2t, labels)
        loss_t2v = F.cross_entropy(logits_t2v, labels)

        loss = (loss_v2t + loss_t2v) / 2.0
        return loss


class HardNegativeContrastiveLoss(nn.Module):
    """
    InfoNCE loss with hard-negative mining within the batch.

    Hard negatives are identified as non-diagonal pairs with the highest
    cosine similarity.  The top-``k`` hardest negatives per anchor are
    up-weighted by a factor ``alpha`` in the loss.

    Args:
        temperature: Contrastive temperature τ.
        num_hard:    Number of hard negatives to up-weight per anchor.
        alpha:       Up-weighting factor for hard negatives.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        num_hard: int = 4,
        alpha: float = 2.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.num_hard = num_hard
        self.alpha = alpha

    def forward(
        self,
        vision_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        B = vision_embed.size(0)
        device = vision_embed.device

        sim = vision_embed @ text_embed.T / self.temperature  # (B, B)
        labels = torch.arange(B, device=device)

        # Build weight matrix: 1 for easy, alpha for hard negatives
        weights = torch.ones_like(sim)
        mask_diag = torch.eye(B, dtype=torch.bool, device=device)
        sim_off = sim.clone()
        sim_off[mask_diag] = -1e9

        _, hard_idx = sim_off.topk(min(self.num_hard, B - 1), dim=1)
        for i in range(B):
            weights[i, hard_idx[i]] = self.alpha
        weights[mask_diag] = 1.0

        # Weighted cross-entropy (v→t direction only, symmetric via transpose)
        exp_sim = (sim * weights).exp()
        prob = exp_sim / exp_sim.sum(dim=1, keepdim=True)
        loss_v2t = -prob.log()[torch.arange(B), labels].mean()

        sim_t = sim.T
        exp_sim_t = (sim_t * weights.T).exp()
        prob_t = exp_sim_t / exp_sim_t.sum(dim=1, keepdim=True)
        loss_t2v = -prob_t.log()[torch.arange(B), labels].mean()

        return (loss_v2t + loss_t2v) / 2.0
