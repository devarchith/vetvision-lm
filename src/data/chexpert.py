"""
CheXpert dataset loader for VetVision-LM pretraining.

Loads image–report pairs from the Stanford CheXpert dataset.
Includes an auto-download helper that points users to the official form.

Dataset reference:
    Irvin et al., "CheXpert: A Large Chest Radiograph Dataset with
    Uncertainty Labels and Expert Comparison." AAAI 2019.
"""

import os
import subprocess
import textwrap
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Label columns present in CheXpert CSVs
# ---------------------------------------------------------------------------
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# Template used to synthesise a short textual description from labels
_LABEL_TEMPLATE = (
    "Chest radiograph. Findings: {findings}."
)


def _labels_to_text(row: pd.Series) -> str:
    """Convert a CheXpert label row into a pseudo-report string."""
    positives = [col for col in CHEXPERT_LABELS if row.get(col, 0) == 1]
    if not positives:
        return "Chest radiograph. No significant findings."
    findings = ", ".join(positives)
    return _LABEL_TEMPLATE.format(findings=findings)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class CheXpertDataset(Dataset):
    """
    PyTorch Dataset for CheXpert image–text pairs.

    Each item returns a dict with:
        ``image``  – Float tensor ``(3, H, W)`` after transform.
        ``text``   – Pseudo-report string derived from pathology labels.
        ``labels`` – Float tensor ``(14,)`` of binary labels (NaN → 0).
        ``path``   – Absolute image file path (str).

    Args:
        csv_path:     Path to ``train.csv`` or ``valid.csv``.
        data_root:    Root directory that contains CheXpert images.  When
                      ``None`` the parent of ``csv_path`` is used.
        transform:    Optional callable applied to each ``PIL.Image``.
        max_samples:  If set, truncate the dataset to this many samples.
        uncertain_as: How to handle uncertain labels (−1).
                      ``"negative"`` → 0, ``"positive"`` → 1, ``"ignore"`` keeps −1.
    """

    def __init__(
        self,
        csv_path: str,
        data_root: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        uncertain_as: str = "negative",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root) if data_root else self.csv_path.parent
        self.transform = transform
        self.uncertain_as = uncertain_as

        self.df = pd.read_csv(self.csv_path)
        if max_samples is not None:
            self.df = self.df.head(max_samples).reset_index(drop=True)

        # Handle uncertain labels
        for col in CHEXPERT_LABELS:
            if col in self.df.columns:
                if uncertain_as == "negative":
                    self.df[col] = self.df[col].replace(-1, 0)
                elif uncertain_as == "positive":
                    self.df[col] = self.df[col].replace(-1, 1)
                self.df[col] = self.df[col].fillna(0)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        # ---- image ----
        img_rel = str(row.get("Path", ""))
        # CheXpert CSVs use paths like "CheXpert-v1.0-small/train/…"
        # Try absolute first, then relative to data_root
        img_path = Path(img_rel)
        if not img_path.is_absolute():
            img_path = self.data_root / img_rel
        # Strip leading dataset prefix if necessary
        if not img_path.exists():
            # Try just the filename components after the dataset root name
            parts = Path(img_rel).parts
            for i, part in enumerate(parts):
                if "CheXpert" in part or "chexpert" in part.lower():
                    img_path = self.data_root / Path(*parts[i + 1 :])
                    break

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            import torchvision.transforms.functional as TF
            img = TF.to_tensor(img)

        # ---- text ----
        text = _labels_to_text(row)

        # ---- labels tensor ----
        label_vals = [float(row.get(col, 0)) for col in CHEXPERT_LABELS]
        labels = torch.tensor(label_vals, dtype=torch.float32)

        return {
            "image": img,
            "text": text,
            "labels": labels,
            "path": str(img_path),
        }


# ---------------------------------------------------------------------------
# Smoke-test synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticCheXpertDataset(Dataset):
    """
    Generates random tensors with CheXpert-compatible shapes.

    Used by ``--smoke-test`` to validate the full pipeline without
    requiring real data or a GPU.

    Args:
        num_samples: Number of synthetic items.
        img_size:    Spatial dimension for image tensors.
    """

    _REPORTS = [
        "Chest radiograph showing mild cardiomegaly.",
        "No acute cardiopulmonary process identified.",
        "Bilateral pleural effusions with pulmonary edema.",
        "Pneumothorax on the right side noted.",
        "Atelectasis in the left lower lobe.",
    ]

    def __init__(self, num_samples: int = 50, img_size: int = 224) -> None:
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        image = torch.randn(3, self.img_size, self.img_size)
        text = self._REPORTS[idx % len(self._REPORTS)]
        labels = torch.zeros(14, dtype=torch.float32)
        labels[idx % 14] = 1.0
        return {
            "image": image,
            "text": text,
            "labels": labels,
            "path": f"synthetic_{idx:05d}.png",
        }


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_chexpert(target_dir: str = "data") -> None:
    """
    Print instructions for downloading CheXpert.

    Stanford requires completion of a data-use form before download;
    therefore automated download is not possible.  This function prints
    step-by-step instructions and the official URL.

    Args:
        target_dir: Where the dataset should be placed after download.
    """
    instructions = textwrap.dedent(
        f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║            CheXpert Download Instructions                     ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  1. Visit https://stanfordaimi.azurewebsites.net/datasets/   ║
        ║     8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2                     ║
        ║  2. Complete the data use agreement form.                    ║
        ║  3. Download "CheXpert-v1.0-small" (~11 GB).                ║
        ║  4. Extract the archive to: {target_dir:<34s}║
        ║     Expected layout:                                         ║
        ║       {target_dir}/CheXpert-v1.0-small/                     ║
        ║           train.csv                                          ║
        ║           valid.csv                                          ║
        ║           train/patient*/study*/view*.jpg                    ║
        ║  5. Update configs/pretrain.yaml:                            ║
        ║       data.chexpert_root: "{target_dir}/CheXpert-v1.0-small" ║
        ╚══════════════════════════════════════════════════════════════╝
        """
    )
    print(instructions)


def build_chexpert_dataloaders(
    cfg,
    train_transform=None,
    val_transform=None,
    smoke_test: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build CheXpert train and validation DataLoaders.

    Args:
        cfg:             OmegaConf / dict config object with data and training sections.
        train_transform: Transform for training split.
        val_transform:   Transform for validation split.
        smoke_test:      If True, returns synthetic dataloaders (no real data needed).

    Returns:
        Tuple of (train_loader, val_loader).
    """
    if smoke_test:
        train_ds = SyntheticCheXpertDataset(num_samples=50)
        val_ds = SyntheticCheXpertDataset(num_samples=10)
    else:
        data_cfg = cfg.data
        train_ds = CheXpertDataset(
            csv_path=data_cfg.train_csv,
            data_root=data_cfg.chexpert_root,
            transform=train_transform,
        )
        val_ds = CheXpertDataset(
            csv_path=data_cfg.valid_csv,
            data_root=data_cfg.chexpert_root,
            transform=val_transform,
        )

    bs = cfg.training.batch_size if not smoke_test else 4
    nw = cfg.training.get("num_workers", 4) if not smoke_test else 0

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=not smoke_test,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=not smoke_test,
    )
    return train_loader, val_loader
