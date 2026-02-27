"""
Vision Encoder for VetVision-LM.

Wraps ViT-Base/16 (via timm) and exposes the CLS token as the global image
representation.  Optionally returns patch-level features for attention
visualisation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


class VisionEncoder(nn.Module):
    """
    ViT-Base/16 vision encoder.

    Takes a batch of images (B, 3, 224, 224) and returns:
        - ``cls_feat``:    CLS-token feature  (B, embed_dim)
        - ``patch_feats``: Patch-token features (B, N_patches, embed_dim) [optional]

    Args:
        model_name:       timm model identifier.  Default ``"vit_base_patch16_224"``.
        pretrained:       Load ImageNet-pretrained weights.
        img_size:         Input resolution (square). Default 224.
        return_patches:   If True, also return patch-level features.
        freeze_layers:    Number of ViT blocks to freeze (0 = none, 12 = all).
    """

    _EMBED_DIMS = {
        "vit_base_patch16_224": 768,
        "vit_small_patch16_224": 384,
        "vit_large_patch16_224": 1024,
    }

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        img_size: int = 224,
        return_patches: bool = False,
        freeze_layers: int = 0,
    ) -> None:
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError("timm is required. Install with: pip install timm")

        self.return_patches = return_patches
        self.embed_dim: int = self._EMBED_DIMS.get(model_name, 768)

        # Build backbone — keep full sequence output
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,          # remove classification head
            global_pool="",         # disable pooling so we get all tokens
        )

        # Freeze early blocks if requested
        if freeze_layers > 0:
            self._freeze_blocks(freeze_layers)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _freeze_blocks(self, n: int) -> None:
        """Freeze the first *n* transformer blocks and the patch embedding."""
        # Patch embedding
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False
        # Positional embedding & cls token
        if hasattr(self.backbone, "cls_token"):
            self.backbone.cls_token.requires_grad = False
        if hasattr(self.backbone, "pos_embed"):
            self.backbone.pos_embed.requires_grad = False
        # Transformer blocks
        blocks = getattr(self.backbone, "blocks", [])
        for i, block in enumerate(blocks):
            if i < n:
                for param in block.parameters():
                    param.requires_grad = False

    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Image tensor of shape (B, 3, H, W).

        Returns:
            cls_feat:    (B, embed_dim) — CLS-token representation.
            patch_feats: (B, N, embed_dim) or None.
        """
        # timm ViT with global_pool="" returns (B, N+1, D) where index 0 is CLS
        tokens = self.backbone.forward_features(x)   # (B, 1+N, D)

        cls_feat = tokens[:, 0, :]                   # (B, D)
        patch_feats = tokens[:, 1:, :] if self.return_patches else None

        return cls_feat, patch_feats

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract last-block attention maps for visualisation.

        Args:
            x: (B, 3, H, W) image tensor.

        Returns:
            Attention tensor of shape (B, num_heads, N+1, N+1).
        """
        attn_maps = []

        def _hook(module, inp, out):
            # out is the attention weight after softmax — shape (B, H, N, N)
            attn_maps.append(out.detach())

        # Register hook on last block's attention module
        last_block = self.backbone.blocks[-1]
        attn_module = last_block.attn
        handle = attn_module.register_forward_hook(_hook)

        with torch.no_grad():
            self.forward(x)

        handle.remove()
        if attn_maps:
            return attn_maps[0]
        return torch.zeros(x.size(0), 1, 1, 1)
