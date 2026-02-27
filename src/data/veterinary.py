"""
Veterinary dataset loader for VetVision-LM fine-tuning.

Uses the TUFTS Veterinary Chest X-Ray dataset (Kaggle: v7labs/vets-chest-xray-competition)
as the primary public proxy for multi-species veterinary radiographs.

Each item exposes:
    image        – Float tensor (3, H, W)
    text         – Textual report / description
    species_label – Integer {0: canine, 1: feline}
    path         – Absolute file path
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split

# ---------------------------------------------------------------------------
# Species mapping
# ---------------------------------------------------------------------------
SPECIES_MAP = {"canine": 0, "dog": 0, "feline": 1, "cat": 1}
SPECIES_LABELS = {0: "canine", 1: "feline"}

# Default report templates by species (used when no free-text is available)
_REPORT_TEMPLATES = {
    0: [
        "Lateral and VD chest radiograph of a canine patient.",
        "Canine thoracic radiograph. Cardiac silhouette within normal limits.",
        "Dog chest X-ray showing the thoracic cavity.",
    ],
    1: [
        "Lateral and VD chest radiograph of a feline patient.",
        "Feline thoracic radiograph. No evidence of pleural effusion.",
        "Cat chest X-ray showing the thoracic cavity.",
    ],
}


def _infer_species(row: pd.Series, img_path: Path) -> int:
    """
    Infer species label from row columns or file path heuristics.

    Looks for 'species', 'label', or 'category' columns; falls back to
    path components.  Returns 0 (canine) when ambiguous.
    """
    for col in ("species", "label", "category", "class"):
        val = row.get(col, None)
        if val is not None:
            val_lower = str(val).strip().lower()
            if val_lower in SPECIES_MAP:
                return SPECIES_MAP[val_lower]
            if "dog" in val_lower or "canine" in val_lower:
                return 0
            if "cat" in val_lower or "feline" in val_lower:
                return 1

    # Path heuristic
    path_str = str(img_path).lower()
    if "dog" in path_str or "canine" in path_str:
        return 0
    if "cat" in path_str or "feline" in path_str:
        return 1
    return 0  # default to canine


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class VeterinaryDataset(Dataset):
    """
    PyTorch Dataset for veterinary radiology image–text–species triples.

    Reads a CSV manifest with columns:
        ``image_path``   – Relative or absolute path to the image.
        ``report_text``  – Free-text radiological report (may be empty).
        ``species_label``– Integer label (0 = canine, 1 = feline).

    Args:
        manifest_csv: Path to the CSV manifest file.
        data_root:    Root directory for relative image paths.
        transform:    Optional image transform.
        max_samples:  If set, truncate the dataset.
    """

    def __init__(
        self,
        manifest_csv: str,
        data_root: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.manifest_path = Path(manifest_csv)
        self.data_root = Path(data_root) if data_root else self.manifest_path.parent
        self.transform = transform

        self.df = pd.read_csv(manifest_csv)
        required = {"image_path", "species_label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Manifest CSV is missing required columns: {missing}. "
                f"Found: {list(self.df.columns)}"
            )
        if "report_text" not in self.df.columns:
            self.df["report_text"] = ""

        if max_samples is not None:
            self.df = self.df.head(max_samples).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        # ---- resolve image path ----
        img_rel = str(row["image_path"])
        img_path = Path(img_rel)
        if not img_path.is_absolute():
            img_path = self.data_root / img_rel

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            import torchvision.transforms.functional as TF
            img = TF.to_tensor(img)

        # ---- text ----
        text = str(row["report_text"]).strip()
        species_label = int(row["species_label"])
        if not text or text.lower() in ("", "nan", "none"):
            templates = _REPORT_TEMPLATES.get(species_label, _REPORT_TEMPLATES[0])
            text = templates[idx % len(templates)]

        return {
            "image": img,
            "text": text,
            "species_label": torch.tensor(species_label, dtype=torch.long),
            "path": str(img_path),
        }


# ---------------------------------------------------------------------------
# Synthetic dataset for smoke tests
# ---------------------------------------------------------------------------

class SyntheticVetDataset(Dataset):
    """Random-tensor dataset mirroring VeterinaryDataset interface."""

    _TEXTS = [
        "Canine thoracic radiograph showing normal cardiac silhouette.",
        "Feline chest X-ray. Mild pleural effusion noted.",
        "Dog thorax – no acute findings.",
        "Cat radiograph – lungs clear bilaterally.",
        "Canine patient: cardiomegaly with pulmonary overcirculation.",
    ]

    def __init__(self, num_samples: int = 50, img_size: int = 224) -> None:
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        image = torch.randn(3, self.img_size, self.img_size)
        species_label = idx % 2
        text = self._TEXTS[idx % len(self._TEXTS)]
        return {
            "image": image,
            "text": text,
            "species_label": torch.tensor(species_label, dtype=torch.long),
            "path": f"synthetic_vet_{idx:05d}.png",
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_vet_dataloaders(
    cfg,
    train_transform=None,
    val_transform=None,
    smoke_test: bool = False,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """
    Build train / val / test DataLoaders for the veterinary dataset.

    Args:
        cfg:             Config with data and training sections.
        train_transform: Transform applied to training images.
        val_transform:   Transform applied to val/test images.
        smoke_test:      If True, use synthetic data.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    if smoke_test:
        full_ds = SyntheticVetDataset(num_samples=50)
        n_train = 40
        n_val = 5
        n_test = len(full_ds) - n_train - n_val
        train_ds, val_ds, test_ds = random_split(
            full_ds,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        data_cfg = cfg.data
        full_ds = VeterinaryDataset(
            manifest_csv=data_cfg.manifest_csv,
            data_root=data_cfg.vet_root,
        )
        n = len(full_ds)
        n_train = int(n * data_cfg.get("train_split", 0.8))
        n_val = int(n * data_cfg.get("val_split", 0.1))
        n_test = n - n_train - n_val
        train_ds, val_ds, test_ds = random_split(
            full_ds,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

    bs = cfg.training.batch_size if not smoke_test else 4
    nw = cfg.training.get("num_workers", 4) if not smoke_test else 0

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=nw
    )
    return train_loader, val_loader, test_loader
