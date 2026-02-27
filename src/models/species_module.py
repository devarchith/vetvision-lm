"""
Species-Adaptive Module for VetVision-LM.

Implements the species conditioning mechanism described in the paper:

    h' = MLP([h; e_s])

where:
    h   is the image or text CLS representation (B, d)
    e_s is the learnable species embedding for species s  (B, d_s)
    MLP is: Linear → LayerNorm → GELU → Linear → LayerNorm

The species embedding matrix E ∈ R^{S × d_s} is learned end-to-end.
"""

import torch
import torch.nn as nn
from typing import Optional


class SpeciesAdaptiveModule(nn.Module):
    """
    Species-conditioning MLP with learnable species embeddings.

    Architecture (two-layer MLP):
        input  : concat([h, e_s])  → dim = input_dim + species_embed_dim
        layer1 : Linear → LayerNorm → GELU
        layer2 : Linear → LayerNorm
        output : dim = output_dim

    Args:
        num_species:        Number of distinct species (S).  Default 2 (canine/feline).
        species_embed_dim:  Dimension of each learnable species vector (d_s).
        input_dim:          Dimension of the incoming feature h (d).
        output_dim:         Dimension of the output feature h' (d').
    """

    SPECIES_NAMES = {0: "canine", 1: "feline"}

    def __init__(
        self,
        num_species: int = 2,
        species_embed_dim: int = 64,
        input_dim: int = 512,
        output_dim: int = 512,
    ) -> None:
        super().__init__()

        self.num_species = num_species
        self.species_embed_dim = species_embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Learnable species embedding matrix E ∈ R^{S × d_s}
        self.species_embeddings = nn.Embedding(num_species, species_embed_dim)
        nn.init.normal_(self.species_embeddings.weight, std=0.02)

        concat_dim = input_dim + species_embed_dim

        # Layer 1: Linear → LayerNorm → GELU
        self.fc1 = nn.Linear(concat_dim, output_dim)
        self.ln1 = nn.LayerNorm(output_dim)
        self.act = nn.GELU()

        # Layer 2: Linear → LayerNorm
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    # ------------------------------------------------------------------

    def get_species_embedding(self, species_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up species embeddings for a batch of species IDs.

        Args:
            species_ids: (B,) integer tensor of species labels.

        Returns:
            (B, species_embed_dim) species embedding vectors.
        """
        return self.species_embeddings(species_ids)

    def forward(
        self,
        h: torch.Tensor,
        species_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply species-adaptive conditioning.

        Args:
            h:           Feature tensor (B, input_dim).
            species_ids: Species label tensor (B,) with values in [0, S).

        Returns:
            h_prime: Conditioned feature tensor (B, output_dim).
        """
        e_s = self.species_embeddings(species_ids)   # (B, d_s)
        x = torch.cat([h, e_s], dim=-1)              # (B, d + d_s)

        # Layer 1
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act(x)

        # Layer 2
        x = self.fc2(x)
        x = self.ln2(x)

        return x
