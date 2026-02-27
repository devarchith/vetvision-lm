"""
Visualisation utilities for VetVision-LM.

Provides:
    - t-SNE embedding plots (coloured by species / modality)
    - Attention map visualisation for ViT-Base
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[List[str]] = None,
    title: str = "t-SNE Embedding Visualisation",
    output_path: Optional[str] = None,
    perplexity: int = 30,
    n_iter: int = 1000,
    figsize: Tuple[int, int] = (10, 8),
    seed: int = 42,
) -> plt.Figure:
    """
    Run t-SNE on ``embeddings`` and plot coloured by ``labels``.

    Args:
        embeddings:   (N, D) float array.
        labels:       (N,) integer array of class labels.
        label_names:  Optional mapping from label int to display string.
        title:        Plot title.
        output_path:  If provided, save figure to this path.
        perplexity:   t-SNE perplexity.
        n_iter:       t-SNE iterations.
        figsize:      Figure size in inches.
        seed:         Random seed.

    Returns:
        ``matplotlib.figure.Figure`` object.
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=seed,
        init="pca",
    )
    reduced = tsne.fit_transform(embeddings)

    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    fig, ax = plt.subplots(figsize=figsize)
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = label_names[lbl] if label_names and lbl < len(label_names) else str(lbl)
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            c=[cmap(i)],
            label=name,
            alpha=0.7,
            s=20,
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=2, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_multimodal_tsne(
    vision_embeds: np.ndarray,
    text_embeds: np.ndarray,
    species_labels: np.ndarray,
    species_names: List[str] = ("canine", "feline"),
    output_path: Optional[str] = None,
    **tsne_kwargs,
) -> plt.Figure:
    """
    Joint t-SNE of vision + text embeddings, coloured by species and marked
    by modality (circle = image, triangle = text).
    """
    from sklearn.manifold import TSNE

    N = vision_embeds.shape[0]
    combined = np.concatenate([vision_embeds, text_embeds], axis=0)
    modality = np.array([0] * N + [1] * N)       # 0=vision, 1=text
    sp_labels = np.concatenate([species_labels, species_labels])

    tsne = TSNE(
        n_components=2,
        perplexity=tsne_kwargs.get("perplexity", 30),
        n_iter=tsne_kwargs.get("n_iter", 1000),
        random_state=tsne_kwargs.get("seed", 42),
        init="pca",
    )
    reduced = tsne.fit_transform(combined)

    colours = ["#1f77b4", "#ff7f0e"]   # blue=canine, orange=feline
    markers = ["o", "^"]               # circle=image, triangle=text
    modality_labels = ["Image", "Text"]

    fig, ax = plt.subplots(figsize=tsne_kwargs.get("figsize", (10, 8)))

    for s_idx, s_name in enumerate(species_names):
        for m_idx, m_name in zip([0, 1], modality_labels):
            mask = (sp_labels == s_idx) & (modality == m_idx)
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c=colours[s_idx],
                marker=markers[m_idx],
                alpha=0.6,
                s=25,
                label=f"{s_name} ({m_name})",
            )

    ax.set_title("VetVision-LM: Vision + Text Embeddings (t-SNE)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=1.5, loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Attention maps
# ---------------------------------------------------------------------------

def visualise_attention(
    image: torch.Tensor,
    attention: torch.Tensor,
    output_path: Optional[str] = None,
    patch_size: int = 16,
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
) -> plt.Figure:
    """
    Overlay ViT attention map on the input image.

    Uses Attention Rollout (Abnar & Zuidema, 2020).

    Args:
        image:         (3, H, W) image tensor (values in [0, 1] or [-∞, ∞]).
        attention:     (num_heads, N+1, N+1) last-block attention weights.
        output_path:   Save figure here if provided.
        patch_size:    ViT patch size.
        head_fusion:   How to aggregate heads: ``"mean"``, ``"max"``, ``"min"``.
        discard_ratio: Fraction of lowest-attention patches to zero out.

    Returns:
        ``matplotlib.figure.Figure``.
    """
    # Aggregate heads
    if head_fusion == "mean":
        att = attention.mean(dim=0)        # (N+1, N+1)
    elif head_fusion == "max":
        att = attention.max(dim=0).values
    else:
        att = attention.min(dim=0).values

    # Add residual and renormalise
    residual = torch.eye(att.size(0), device=att.device)
    att = att + residual
    att = att / att.sum(dim=-1, keepdim=True)

    # Attention from CLS to all patches
    att_cls = att[0, 1:]    # (N,)
    N = att_cls.size(0)
    H_patches = W_patches = int(N ** 0.5)
    att_map = att_cls.reshape(H_patches, W_patches).cpu().numpy()

    # Threshold low-attention patches
    flat = att_map.flatten()
    threshold = np.sort(flat)[int(discard_ratio * len(flat))]
    att_map = np.where(att_map >= threshold, att_map, 0.0)

    # Normalise
    if att_map.max() > 0:
        att_map = att_map / att_map.max()

    # Resize attention map to image size
    from PIL import Image as PILImage
    H_img = image.shape[-2]
    W_img = image.shape[-1]
    att_pil = PILImage.fromarray((att_map * 255).astype(np.uint8))
    att_resized = np.array(att_pil.resize((W_img, H_img), PILImage.BILINEAR)) / 255.0

    # Prepare image for display
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(att_resized, cmap="jet")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")

    axes[2].imshow(img_np, cmap="gray")
    axes[2].imshow(att_resized, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
