"""
Projection heads for VetVision-LM.

Maps vision and text encoder outputs into a shared embedding space
suitable for contrastive learning.

Architecture per modality:
    Linear(input_dim → hidden_dim) → GELU → Linear(hidden_dim → embed_dim) → L2-normalise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head with L2 normalisation.

    Args:
        input_dim:  Dimension of the encoder output (e.g. 768 for ViT-B).
        hidden_dim: Hidden layer dimension.
        embed_dim:  Output (shared embedding) dimension.
        dropout:    Dropout probability between layers. Default 0.1.
        normalize:  If True, L2-normalise the output.  Default True.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        embed_dim: int = 512,
        dropout: float = 0.1,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.normalize = normalize

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature (B, input_dim).

        Returns:
            Projected embedding (B, embed_dim), L2-normalised if ``normalize=True``.
        """
        out = self.net(x)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


class DualProjectionHead(nn.Module):
    """
    Paired vision + text projection heads with a shared embedding space.

    Args:
        vision_input_dim: Dimension from the vision encoder.
        text_input_dim:   Dimension from the text encoder.
        hidden_dim:       Shared hidden dimension in each projection MLP.
        embed_dim:        Shared output embedding dimension.
        dropout:          Dropout probability.
    """

    def __init__(
        self,
        vision_input_dim: int = 768,
        text_input_dim: int = 768,
        hidden_dim: int = 512,
        embed_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.vision_proj = ProjectionHead(
            input_dim=vision_input_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.text_proj = ProjectionHead(
            input_dim=text_input_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.embed_dim = embed_dim

    def forward(
        self,
        vision_feat: torch.Tensor,
        text_feat: torch.Tensor,
    ) -> tuple:
        """
        Project vision and text features into the shared space.

        Args:
            vision_feat: (B, vision_input_dim)
            text_feat:   (B, text_input_dim)

        Returns:
            vision_embed: (B, embed_dim) L2-normalised.
            text_embed:   (B, embed_dim) L2-normalised.
        """
        vision_embed = self.vision_proj(vision_feat)
        text_embed = self.text_proj(text_feat)
        return vision_embed, text_embed
