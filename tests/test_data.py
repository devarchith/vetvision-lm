"""
Tests for VetVision-LM data pipeline.

Run with:
    pytest tests/test_data.py -v
"""

import sys
import tempfile
from pathlib import Path

import pytest
import pandas as pd
import torch
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------

class TestAugmentations:
    def test_import(self):
        from data.augmentations import build_train_transform, build_val_transform
        assert build_train_transform is not None

    def test_train_transform_output_shape(self):
        from data.augmentations import build_train_transform
        tf = build_train_transform(img_size=224)
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        tensor = tf(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transform_output_shape(self):
        from data.augmentations import build_val_transform
        tf = build_val_transform(img_size=224)
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        tensor = tf(img)
        assert tensor.shape == (3, 224, 224)

    def test_train_transform_output_type(self):
        from data.augmentations import build_train_transform
        tf = build_train_transform()
        img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128)
        tensor = tf(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32

    def test_custom_augmentation_cfg(self):
        from data.augmentations import build_train_transform
        cfg = {
            "random_horizontal_flip": 0.0,
            "random_rotation": 5,
            "color_jitter": {"brightness": 0.1, "contrast": 0.1},
        }
        tf = build_train_transform(img_size=128, augmentation_cfg=cfg)
        img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
        tensor = tf(img)
        assert tensor.shape == (3, 128, 128)


# ---------------------------------------------------------------------------
# CheXpert synthetic dataset tests
# ---------------------------------------------------------------------------

class TestSyntheticCheXpert:
    def test_import(self):
        from data.chexpert import SyntheticCheXpertDataset
        assert SyntheticCheXpertDataset is not None

    def test_length(self):
        from data.chexpert import SyntheticCheXpertDataset
        ds = SyntheticCheXpertDataset(num_samples=50)
        assert len(ds) == 50

    def test_item_keys(self):
        from data.chexpert import SyntheticCheXpertDataset
        ds = SyntheticCheXpertDataset(num_samples=10)
        item = ds[0]
        assert "image" in item
        assert "text" in item
        assert "labels" in item
        assert "path" in item

    def test_image_shape(self):
        from data.chexpert import SyntheticCheXpertDataset
        ds = SyntheticCheXpertDataset(num_samples=5, img_size=224)
        item = ds[0]
        assert item["image"].shape == (3, 224, 224)

    def test_labels_shape(self):
        from data.chexpert import SyntheticCheXpertDataset
        ds = SyntheticCheXpertDataset(num_samples=5)
        item = ds[0]
        assert item["labels"].shape == (14,)

    def test_dataloader(self):
        from data.chexpert import SyntheticCheXpertDataset
        from torch.utils.data import DataLoader
        ds = SyntheticCheXpertDataset(num_samples=8)
        loader = DataLoader(ds, batch_size=4)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 3, 224, 224)
        assert len(batch["text"]) == 4


# ---------------------------------------------------------------------------
# Veterinary synthetic dataset tests
# ---------------------------------------------------------------------------

class TestSyntheticVetDataset:
    def test_import(self):
        from data.veterinary import SyntheticVetDataset
        assert SyntheticVetDataset is not None

    def test_length(self):
        from data.veterinary import SyntheticVetDataset
        ds = SyntheticVetDataset(num_samples=50)
        assert len(ds) == 50

    def test_item_keys(self):
        from data.veterinary import SyntheticVetDataset
        ds = SyntheticVetDataset(num_samples=10)
        item = ds[0]
        assert "image" in item
        assert "text" in item
        assert "species_label" in item
        assert "path" in item

    def test_species_labels_binary(self):
        from data.veterinary import SyntheticVetDataset
        ds = SyntheticVetDataset(num_samples=20)
        for i in range(len(ds)):
            label = ds[i]["species_label"].item()
            assert label in (0, 1), f"species_label must be 0 or 1, got {label}"

    def test_image_shape(self):
        from data.veterinary import SyntheticVetDataset
        ds = SyntheticVetDataset(num_samples=5, img_size=224)
        item = ds[0]
        assert item["image"].shape == (3, 224, 224)


# ---------------------------------------------------------------------------
# CSV-based VeterinaryDataset tests
# ---------------------------------------------------------------------------

class TestVeterinaryDataset:
    @pytest.fixture
    def tmp_dataset(self, tmp_path):
        """Create a tiny fake dataset with real image files."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        records = []
        for i in range(6):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            fname = f"img_{i:03d}.jpg"
            img.save(img_dir / fname)
            records.append({
                "image_path": str(img_dir / fname),
                "report_text": f"Test report {i}.",
                "species_label": i % 2,
            })

        df = pd.DataFrame(records)
        csv_path = tmp_path / "manifest.csv"
        df.to_csv(csv_path, index=False)
        return csv_path, tmp_path

    def test_load_dataset(self, tmp_dataset):
        from data.veterinary import VeterinaryDataset
        csv_path, root = tmp_dataset
        ds = VeterinaryDataset(manifest_csv=str(csv_path), data_root=str(root))
        assert len(ds) == 6

    def test_item_types(self, tmp_dataset):
        from data.veterinary import VeterinaryDataset
        csv_path, root = tmp_dataset
        ds = VeterinaryDataset(manifest_csv=str(csv_path), data_root=str(root))
        item = ds[0]
        assert isinstance(item["image"], torch.Tensor)
        assert isinstance(item["text"], str)
        assert item["species_label"].dtype == torch.long

    def test_missing_column_raises(self, tmp_path):
        from data.veterinary import VeterinaryDataset
        df = pd.DataFrame({"image_path": ["x.jpg"]})
        csv_path = tmp_path / "bad.csv"
        df.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="species_label"):
            VeterinaryDataset(manifest_csv=str(csv_path))
