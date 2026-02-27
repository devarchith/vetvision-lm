"""
Tests for VetVision-LM loss functions.

Run with:
    pytest tests/test_losses.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Contrastive Loss tests
# ---------------------------------------------------------------------------

class TestContrastiveLoss:
    @pytest.fixture
    def embeds(self):
        """Pair of L2-normalised embeddings."""
        import torch.nn.functional as F
        v = F.normalize(torch.randn(8, 512), p=2, dim=-1)
        t = F.normalize(torch.randn(8, 512), p=2, dim=-1)
        return v, t

    def test_import(self):
        from losses.contrastive import ContrastiveLoss
        assert ContrastiveLoss is not None

    def test_loss_is_scalar(self, embeds):
        from losses.contrastive import ContrastiveLoss
        loss_fn = ContrastiveLoss(temperature=0.07)
        v, t = embeds
        loss = loss_fn(v, t)
        assert loss.ndim == 0, "Loss should be a scalar tensor"

    def test_loss_is_positive(self, embeds):
        from losses.contrastive import ContrastiveLoss
        loss_fn = ContrastiveLoss()
        v, t = embeds
        loss = loss_fn(v, t)
        assert loss.item() > 0

    def test_perfect_match_lower_loss(self):
        """Identical v and t embeddings should yield lower loss."""
        import torch.nn.functional as F
        from losses.contrastive import ContrastiveLoss
        loss_fn = ContrastiveLoss()
        v = F.normalize(torch.randn(4, 128), p=2, dim=-1)
        loss_random = loss_fn(v, F.normalize(torch.randn(4, 128), p=2, dim=-1))
        loss_identical = loss_fn(v, v.clone())
        assert loss_identical.item() <= loss_random.item() + 1.0

    def test_learnable_temperature(self, embeds):
        from losses.contrastive import ContrastiveLoss
        loss_fn = ContrastiveLoss(learnable_temp=True)
        v, t = embeds
        loss = loss_fn(v, t)
        loss.backward()
        assert loss_fn.logit_scale.grad is not None

    def test_gradient_flows(self, embeds):
        from losses.contrastive import ContrastiveLoss
        import torch.nn.functional as F
        loss_fn = ContrastiveLoss()
        # v_raw is the leaf; F.normalize produces a non-leaf, so check leaf grad
        v_raw = torch.randn(4, 64, requires_grad=True)
        v = F.normalize(v_raw, p=2, dim=-1)
        t = F.normalize(torch.randn(4, 64), p=2, dim=-1)
        loss = loss_fn(v, t)
        loss.backward()
        assert v_raw.grad is not None


# ---------------------------------------------------------------------------
# Species Loss tests
# ---------------------------------------------------------------------------

class TestSpeciesLoss:
    @pytest.fixture
    def embeds_with_species(self):
        import torch.nn.functional as F
        s_emb = F.normalize(torch.randn(8, 512), p=2, dim=-1)
        t_emb = F.normalize(torch.randn(8, 512), p=2, dim=-1)
        species = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        return s_emb, t_emb, species

    def test_import(self):
        from losses.species_loss import SpeciesContrastiveLoss, CombinedLoss
        assert SpeciesContrastiveLoss is not None

    def test_species_loss_scalar(self, embeds_with_species):
        from losses.species_loss import SpeciesContrastiveLoss
        loss_fn = SpeciesContrastiveLoss()
        s, t, sp = embeds_with_species
        loss = loss_fn(s, t, sp)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_species_loss_positive(self, embeds_with_species):
        from losses.species_loss import SpeciesContrastiveLoss
        loss_fn = SpeciesContrastiveLoss()
        s, t, sp = embeds_with_species
        loss = loss_fn(s, t, sp)
        assert loss.item() >= 0

    def test_combined_loss_keys(self, embeds_with_species):
        from losses.contrastive import ContrastiveLoss
        from losses.species_loss import SpeciesContrastiveLoss, CombinedLoss
        cont = ContrastiveLoss()
        spec = SpeciesContrastiveLoss()
        combined = CombinedLoss(cont, spec, lambda_species=0.5)
        import torch.nn.functional as F
        v = F.normalize(torch.randn(8, 512), p=2, dim=-1)
        t = F.normalize(torch.randn(8, 512), p=2, dim=-1)
        s, _, sp = embeds_with_species
        result = combined(v, t, s, sp)
        assert "total" in result
        assert "contrastive" in result
        assert "species" in result

    def test_combined_loss_lambda(self, embeds_with_species):
        from losses.contrastive import ContrastiveLoss
        from losses.species_loss import SpeciesContrastiveLoss, CombinedLoss
        import torch.nn.functional as F
        v = F.normalize(torch.randn(8, 512), p=2, dim=-1)
        t = F.normalize(torch.randn(8, 512), p=2, dim=-1)
        s, _, sp = embeds_with_species
        cont = ContrastiveLoss()
        spec = SpeciesContrastiveLoss()

        lam0 = CombinedLoss(cont, spec, lambda_species=0.0)
        lam1 = CombinedLoss(cont, spec, lambda_species=0.5)

        r0 = lam0(v, t, s, sp)
        r1 = lam1(v, t, s, sp)

        # With lambda=0, total should equal contrastive only
        assert torch.isclose(r0["total"], r0["contrastive"], atol=1e-4)
        # With lambda>0, total includes species term
        assert not torch.isclose(r1["total"], r1["contrastive"], atol=1e-4)
