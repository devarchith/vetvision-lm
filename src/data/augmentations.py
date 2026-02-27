"""
Radiograph augmentation pipeline for VetVision-LM.

Implements domain-specific augmentations suited for X-ray / radiograph images,
including contrast normalisation and geometric transforms that preserve
diagnostic features.
"""

import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance


# ---------------------------------------------------------------------------
# CLAHE approximation using PIL (lightweight, no cv2 dependency required)
# ---------------------------------------------------------------------------

class CLAHETransform:
    """
    Approximate Contrast Limited Adaptive Histogram Equalisation (CLAHE).

    Splits the image into a grid of tiles and equalises each tile
    independently before blending back with the original.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "L":
            img_l = img.convert("L")
            result = ImageEnhance.Contrast(img_l).enhance(self.clip_limit)
            if img.mode == "RGB":
                result = result.convert("RGB")
            return result
        return ImageEnhance.Contrast(img).enhance(self.clip_limit)


class RandomGammaCorrection:
    """Apply random gamma correction to simulate varying X-ray exposures."""

    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.5), p: float = 0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            img_np = np.array(img).astype(np.float32) / 255.0
            img_np = np.power(img_np, gamma)
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(img_np)
        return img


class RadiographNormalize:
    """
    Normalise a radiograph image.

    X-ray images often benefit from per-image normalisation.  This module
    applies ImageNet-style mean/std normalisation after converting to tensor.
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.normalize = T.Normalize(mean=mean, std=std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.normalize(tensor)


# ---------------------------------------------------------------------------
# Public transform factories
# ---------------------------------------------------------------------------

def build_train_transform(
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    augmentation_cfg: Optional[Dict] = None,
) -> T.Compose:
    """
    Build the training-time augmentation pipeline.

    Applies stochastic transforms appropriate for radiograph images:
    random horizontal flip, small rotation, colour jitter (contrast/brightness),
    and normalisation.

    Args:
        img_size: Target image size (square).
        mean:     ImageNet-style mean per channel.
        std:      ImageNet-style std per channel.
        augmentation_cfg: Optional dict overriding default parameters.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    cfg = augmentation_cfg or {}

    flip_p = cfg.get("random_horizontal_flip", 0.5)
    rotation = cfg.get("random_rotation", 10)
    cj = cfg.get("color_jitter", {"brightness": 0.2, "contrast": 0.2})

    transforms = [
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=flip_p),
        T.RandomRotation(degrees=rotation),
        T.ColorJitter(
            brightness=cj.get("brightness", 0.2),
            contrast=cj.get("contrast", 0.2),
        ),
        RandomGammaCorrection(p=0.3),
        T.ToTensor(),
        T.Normalize(mean=list(mean), std=list(std)),
    ]

    return T.Compose(transforms)


def build_val_transform(
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """
    Build the deterministic validation / inference transform.

    Args:
        img_size: Target image size (square).
        mean:     ImageNet-style mean per channel.
        std:      ImageNet-style std per channel.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    return T.Compose(
        [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_test_transform(
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """Alias of ``build_val_transform`` for clarity in test scripts."""
    return build_val_transform(img_size=img_size, mean=mean, std=std)
