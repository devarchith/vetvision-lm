"""
Tests for VetVision-LM model components.

Run with:
    pytest tests/test_model.py -v
    pytest tests/test_model.py -v --smoke   (same, alias flag)
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def dummy_images():
    return torch.randn(2, 3, 224, 224)


@pytest.fixture(scope="module")
def dummy_texts():
    return [
        "Canine thoracic radiograph showing cardiomegaly.",
        "Feline chest X-ray: pleural effusion noted bilaterally.",
    ]


@pytest.fixture(scope="module")
def dummy_species():
    return torch.tensor([0, 1], dtype=torch.long)


# ---------------------------------------------------------------------------
# Vision encoder tests
# ---------------------------------------------------------------------------

class TestVisionEncoder:
    def test_import(self):
        from models.vision_encoder import VisionEncoder
        assert VisionEncoder is not None

    def test_init_no_pretrained(self):
        from models.vision_encoder import VisionEncoder
        enc = VisionEncoder(pretrained=False)
        assert enc.output_dim == 768

    def test_forward_shape(self, dummy_images):
        from models.vision_encoder import VisionEncoder
        enc = VisionEncoder(pretrained=False, return_patches=True)
        cls_feat, patch_feats = enc(dummy_images)
        B = dummy_images.size(0)
        assert cls_feat.shape == (B, 768), f"Expected ({B}, 768), got {cls_feat.shape}"
        assert patch_feats is not None
        assert patch_feats.shape[0] == B

    def test_forward_no_patches(self, dummy_images):
        from models.vision_encoder import VisionEncoder
        enc = VisionEncoder(pretrained=False, return_patches=False)
        cls_feat, patch_feats = enc(dummy_images)
        assert patch_feats is None


# ---------------------------------------------------------------------------
# Species-Adaptive Module tests
# ---------------------------------------------------------------------------

class TestSpeciesModule:
    def test_import(self):
        from models.species_module import SpeciesAdaptiveModule
        assert SpeciesAdaptiveModule is not None

    def test_forward(self):
        from models.species_module import SpeciesAdaptiveModule
        mod = SpeciesAdaptiveModule(
            num_species=2, species_embed_dim=64, input_dim=512, output_dim=512
        )
        h = torch.randn(4, 512)
        species_ids = torch.tensor([0, 1, 0, 1])
        out = mod(h, species_ids)
        assert out.shape == (4, 512)

    def test_species_embedding_shape(self):
        from models.species_module import SpeciesAdaptiveModule
        mod = SpeciesAdaptiveModule(num_species=3, species_embed_dim=32, input_dim=256, output_dim=256)
        assert mod.species_embeddings.weight.shape == (3, 32)

    def test_different_species_different_output(self):
        from models.species_module import SpeciesAdaptiveModule
        mod = SpeciesAdaptiveModule(num_species=2, species_embed_dim=64, input_dim=512, output_dim=512)
        h = torch.ones(1, 512)
        out0 = mod(h, torch.tensor([0]))
        out1 = mod(h, torch.tensor([1]))
        assert not torch.allclose(out0, out1), "Different species should produce different outputs"


# ---------------------------------------------------------------------------
# Projection Head tests
# ---------------------------------------------------------------------------

class TestProjectionHead:
    def test_import(self):
        from models.projection import ProjectionHead, DualProjectionHead
        assert ProjectionHead is not None

    def test_projection_shape(self):
        from models.projection import ProjectionHead
        head = ProjectionHead(input_dim=768, hidden_dim=512, embed_dim=512)
        x = torch.randn(4, 768)
        out = head(x)
        assert out.shape == (4, 512)

    def test_l2_normalised(self):
        from models.projection import ProjectionHead
        head = ProjectionHead(input_dim=768, hidden_dim=512, embed_dim=512, normalize=True)
        x = torch.randn(4, 768)
        out = head(x)
        norms = out.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5), "Output should be L2-normalised"

    def test_dual_projection(self):
        from models.projection import DualProjectionHead
        dual = DualProjectionHead(vision_input_dim=768, text_input_dim=768)
        v = torch.randn(4, 768)
        t = torch.randn(4, 768)
        v_emb, t_emb = dual(v, t)
        assert v_emb.shape == (4, 512)
        assert t_emb.shape == (4, 512)


# ---------------------------------------------------------------------------
# Full model tests
# ---------------------------------------------------------------------------

class TestVetVisionLM:
    @pytest.fixture
    def model(self):
        from models.vetvision import VetVisionLM
        return VetVisionLM(
            vision_cfg={"name": "vit_base_patch16_224", "pretrained": False},
            text_cfg={"name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", "pretrained": False},
            species_cfg={"num_species": 2, "species_embed_dim": 64, "output_dim": 512},
            proj_cfg={"embed_dim": 512, "hidden_dim": 512, "vision_input_dim": 768, "text_input_dim": 768},
        )

    def test_output_shapes(self, model, dummy_images, dummy_texts, dummy_species):
        out = model(images=dummy_images, texts=dummy_texts, species_ids=dummy_species)
        B = dummy_images.size(0)
        assert out.vision_embed.shape == (B, 512)
        assert out.text_embed.shape == (B, 512)
        assert out.species_embed is not None
        assert out.species_embed.shape == (B, 512)

    def test_embeddings_normalised(self, model, dummy_images, dummy_texts):
        out = model(images=dummy_images, texts=dummy_texts)
        v_norms = out.vision_embed.norm(p=2, dim=-1)
        t_norms = out.text_embed.norm(p=2, dim=-1)
        assert torch.allclose(v_norms, torch.ones_like(v_norms), atol=1e-5)
        assert torch.allclose(t_norms, torch.ones_like(t_norms), atol=1e-5)

    def test_no_species_module(self, dummy_images, dummy_texts):
        from models.vetvision import VetVisionLM
        model = VetVisionLM(
            vision_cfg={"name": "vit_base_patch16_224", "pretrained": False},
            text_cfg={"name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", "pretrained": False},
            use_species_module=False,
        )
        out = model(images=dummy_images, texts=dummy_texts)
        assert out.species_embed is None, "species_embed should be None when module disabled"

    def test_parameter_count(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 1_000_000, "Model should have at least 1M parameters"
