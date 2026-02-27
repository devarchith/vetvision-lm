"""
VetVision-LM: Full model assembly.

Combines:
    1. Vision Encoder  (ViT-Base/16, CLS token)
    2. Text Encoder    (PubMedBERT, CLS token)
    3. Species-Adaptive Module  (applied to image CLS after projection)
    4. Dual Projection Heads   (map both modalities to shared embedding space)

Forward pass output (VetVisionOutput):
    vision_embed   – (B, embed_dim)  L2-normalised image embedding
    text_embed     – (B, embed_dim)  L2-normalised text embedding
    species_embed  – (B, embed_dim)  species-conditioned image embedding (optional)
    vision_feat    – (B, D_v)  raw vision CLS feature
    text_feat      – (B, D_t)  raw text CLS feature
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_encoder import VisionEncoder
from .text_encoder import TextEncoder
from .species_module import SpeciesAdaptiveModule
from .projection import DualProjectionHead


@dataclass
class VetVisionOutput:
    vision_embed: torch.Tensor          # (B, embed_dim)
    text_embed: torch.Tensor            # (B, embed_dim)
    species_embed: Optional[torch.Tensor] = None  # (B, embed_dim)
    vision_feat: Optional[torch.Tensor] = None    # (B, D_v)
    text_feat: Optional[torch.Tensor] = None      # (B, D_t)
    patch_feats: Optional[torch.Tensor] = None    # (B, N, D_v)


class VetVisionLM(nn.Module):
    """
    VetVision-LM — Self-Supervised Vision-Language Model for Veterinary Radiology.

    Args:
        vision_cfg:   Dict / OmegaConf with vision encoder params.
        text_cfg:     Dict / OmegaConf with text encoder params.
        species_cfg:  Dict / OmegaConf with species module params.
        proj_cfg:     Dict / OmegaConf with projection head params.
        use_species_module:  Enable / disable species conditioning (ablation).
    """

    def __init__(
        self,
        vision_cfg: Optional[Dict] = None,
        text_cfg: Optional[Dict] = None,
        species_cfg: Optional[Dict] = None,
        proj_cfg: Optional[Dict] = None,
        use_species_module: bool = True,
    ) -> None:
        super().__init__()
        vision_cfg = vision_cfg or {}
        text_cfg = text_cfg or {}
        species_cfg = species_cfg or {}
        proj_cfg = proj_cfg or {}

        # ---- Vision encoder ----
        self.vision_encoder = VisionEncoder(
            model_name=vision_cfg.get("name", "vit_base_patch16_224"),
            pretrained=vision_cfg.get("pretrained", True),
            img_size=vision_cfg.get("img_size", 224),
            return_patches=True,
            freeze_layers=vision_cfg.get("freeze_layers", 0),
        )
        v_dim = self.vision_encoder.output_dim  # 768

        # ---- Text encoder ----
        self.text_encoder = TextEncoder(
            model_name=text_cfg.get(
                "name",
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            ),
            pretrained=text_cfg.get("pretrained", True),
            max_length=text_cfg.get("max_length", 128),
            freeze_layers=text_cfg.get("freeze_layers", 0),
        )
        t_dim = self.text_encoder.output_dim  # 768

        # ---- Projection heads ----
        embed_dim = proj_cfg.get("embed_dim", 512)
        hidden_dim = proj_cfg.get("hidden_dim", 512)
        self.projector = DualProjectionHead(
            vision_input_dim=proj_cfg.get("vision_input_dim", v_dim),
            text_input_dim=proj_cfg.get("text_input_dim", t_dim),
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )

        # ---- Species-Adaptive Module ----
        self.use_species_module = use_species_module
        if use_species_module:
            self.species_module = SpeciesAdaptiveModule(
                num_species=species_cfg.get("num_species", 2),
                species_embed_dim=species_cfg.get("species_embed_dim", 64),
                input_dim=embed_dim,
                output_dim=species_cfg.get("output_dim", embed_dim),
            )
            # Extra projection to shared space after species conditioning
            species_out_dim = species_cfg.get("output_dim", embed_dim)
            self.species_proj = nn.Sequential(
                nn.Linear(species_out_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.species_module = None
            self.species_proj = None

        self.embed_dim = embed_dim

    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg) -> "VetVisionLM":
        """
        Instantiate from an OmegaConf config (e.g. loaded from pretrain.yaml).
        """
        m_cfg = cfg.model
        return cls(
            vision_cfg=dict(m_cfg.vision_encoder),
            text_cfg=dict(m_cfg.text_encoder),
            species_cfg=dict(m_cfg.species_module),
            proj_cfg=dict(m_cfg.projection),
        )

    # ------------------------------------------------------------------

    def encode_image(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode a batch of images.

        Returns:
            embed:       (B, embed_dim) L2-normalised vision embedding.
            patch_feats: (B, N, D) patch features or None.
        """
        cls_feat, patch_feats = self.vision_encoder(images)
        embed, _ = self.projector(cls_feat, torch.zeros(cls_feat.size(0), self.text_encoder.output_dim, device=cls_feat.device))
        return embed, patch_feats

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of strings.

        Returns:
            embed: (B, embed_dim) L2-normalised text embedding.
        """
        device = next(self.parameters()).device
        cls_feat, _ = self.text_encoder(texts=texts)
        _, embed = self.projector(torch.zeros(cls_feat.size(0), self.vision_encoder.output_dim, device=device), cls_feat)
        return embed

    def forward(
        self,
        images: torch.Tensor,
        texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        species_ids: Optional[torch.Tensor] = None,
        return_raw_feats: bool = False,
    ) -> VetVisionOutput:
        """
        Full forward pass.

        Args:
            images:         (B, 3, H, W)
            texts:          List[str] of length B  (or None if using input_ids)
            input_ids:      (B, seq_len) pre-tokenised ids (alternative to texts)
            attention_mask: (B, seq_len)
            species_ids:    (B,) species label tensor for species-adaptive module.
                            Required when ``use_species_module=True`` during training.
            return_raw_feats: If True, include raw CLS features in output.

        Returns:
            VetVisionOutput dataclass.
        """
        # ---- Encode vision ----
        vision_cls, patch_feats = self.vision_encoder(images)

        # ---- Encode text ----
        if texts is not None:
            text_cls, _ = self.text_encoder(texts=texts)
        else:
            text_cls, _ = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # ---- Project to shared space ----
        vision_embed, text_embed = self.projector(vision_cls, text_cls)
        # Both are (B, embed_dim), L2-normalised

        # ---- Species-adaptive conditioning ----
        species_embed = None
        if self.use_species_module and species_ids is not None:
            sa_out = self.species_module(vision_embed, species_ids)
            species_embed = F.normalize(self.species_proj(sa_out), p=2, dim=-1)

        return VetVisionOutput(
            vision_embed=vision_embed,
            text_embed=text_embed,
            species_embed=species_embed,
            vision_feat=vision_cls if return_raw_feats else None,
            text_feat=text_cls if return_raw_feats else None,
            patch_feats=patch_feats if return_raw_feats else None,
        )

    # ------------------------------------------------------------------
    # Convenience: logit scale (temperature)
    # ------------------------------------------------------------------

    def get_logit_scale(self) -> float:
        """Return 1/τ where τ = 0.07 (fixed as in paper)."""
        return 1.0 / 0.07
