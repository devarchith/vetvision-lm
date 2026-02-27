"""
Species-aware contrastive loss for VetVision-LM.

L_species encourages species-conditioned image embeddings to be:
  (a) close to same-species text embeddings, and
  (b) separated from cross-species embeddings by a margin γ.

L_species = max(0, - (pos_sim - neg_sim) + γ)   [triplet-style]

combined with standard InfoNCE within each species group.

Total loss:
    L = L_contrastive + λ * L_species
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpeciesContrastiveLoss(nn.Module):
    """
    Species-aware contrastive loss.

    Pulls same-species image–text pairs together and pushes
    cross-species pairs apart (with margin γ).

    Args:
        temperature: Temperature τ for InfoNCE within species groups.
        margin:      Margin γ for cross-species separation.
        use_triplet: If True, add a triplet margin loss on top of InfoNCE.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.2,
        use_triplet: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.use_triplet = use_triplet

    def forward(
        self,
        species_embed: torch.Tensor,
        text_embed: torch.Tensor,
        species_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute species-aware contrastive loss.

        Args:
            species_embed: (B, D) species-conditioned image embeddings (L2-normed).
            text_embed:    (B, D) text embeddings (L2-normed).
            species_ids:   (B,)  integer species labels.

        Returns:
            Scalar species loss.
        """
        B = species_embed.size(0)
        device = species_embed.device

        # ---- InfoNCE within same-species groups ----
        loss_infonce = self._intra_species_infonce(
            species_embed, text_embed, species_ids
        )

        if not self.use_triplet:
            return loss_infonce

        # ---- Triplet margin loss (cross-species separation) ----
        loss_triplet = self._cross_species_triplet(
            species_embed, text_embed, species_ids
        )

        return loss_infonce + loss_triplet

    # ------------------------------------------------------------------

    def _intra_species_infonce(
        self,
        vision: torch.Tensor,
        text: torch.Tensor,
        species_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Symmetric InfoNCE computed separately within each species group,
        then averaged.
        """
        unique_species = species_ids.unique()
        losses = []

        for s in unique_species:
            mask = species_ids == s
            if mask.sum() < 2:
                continue
            v_s = vision[mask]   # (n_s, D)
            t_s = text[mask]     # (n_s, D)
            n_s = v_s.size(0)
            labels = torch.arange(n_s, device=vision.device)

            logits_v2t = v_s @ t_s.T / self.temperature
            logits_t2v = logits_v2t.T

            l = (
                F.cross_entropy(logits_v2t, labels)
                + F.cross_entropy(logits_t2v, labels)
            ) / 2.0
            losses.append(l)

        if not losses:
            return torch.tensor(0.0, device=vision.device)
        return torch.stack(losses).mean()

    def _cross_species_triplet(
        self,
        vision: torch.Tensor,
        text: torch.Tensor,
        species_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Triplet margin loss:
            pos_sim  = mean same-species cosine similarity
            neg_sim  = mean cross-species cosine similarity
            loss     = max(0, neg_sim - pos_sim + γ)
        """
        B = vision.size(0)
        sim_matrix = vision @ text.T  # (B, B) cosine similarities

        triplet_losses = []
        for i in range(B):
            same_mask = species_ids == species_ids[i]
            same_mask[i] = False          # exclude self
            diff_mask = species_ids != species_ids[i]

            if same_mask.sum() == 0 or diff_mask.sum() == 0:
                continue

            pos_sim = sim_matrix[i, same_mask].mean()
            neg_sim = sim_matrix[i, diff_mask].mean()
            loss_i = F.relu(neg_sim - pos_sim + self.margin)
            triplet_losses.append(loss_i)

        if not triplet_losses:
            return torch.tensor(0.0, device=vision.device)
        return torch.stack(triplet_losses).mean()


class CombinedLoss(nn.Module):
    """
    L = L_contrastive + λ * L_species

    Args:
        contrastive_loss: Contrastive loss module (e.g. ContrastiveLoss).
        species_loss:     Species loss module (SpeciesContrastiveLoss).
        lambda_species:   Weighting factor λ.  Default 0.5.
    """

    def __init__(
        self,
        contrastive_loss: nn.Module,
        species_loss: nn.Module,
        lambda_species: float = 0.5,
    ) -> None:
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.species_loss = species_loss
        self.lambda_species = lambda_species

    def forward(
        self,
        vision_embed: torch.Tensor,
        text_embed: torch.Tensor,
        species_embed: Optional[torch.Tensor],
        species_ids: Optional[torch.Tensor],
    ) -> dict:
        """
        Compute total combined loss.

        Args:
            vision_embed:  (B, D) vision embeddings.
            text_embed:    (B, D) text embeddings.
            species_embed: (B, D) species-conditioned vision embeddings (or None).
            species_ids:   (B,)  species labels (or None).

        Returns:
            Dict with keys: ``total``, ``contrastive``, ``species``.
        """
        l_cont = self.contrastive_loss(vision_embed, text_embed)

        l_spec = torch.tensor(0.0, device=vision_embed.device)
        if species_embed is not None and species_ids is not None:
            l_spec = self.species_loss(species_embed, text_embed, species_ids)

        total = l_cont + self.lambda_species * l_spec
        return {"total": total, "contrastive": l_cont, "species": l_spec}
