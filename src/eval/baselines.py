"""
Baseline model evaluators for VetVision-LM.

Implements evaluation wrappers for:
    - CLIP (openai/clip-vit-base-patch16)
    - BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
    - ResNet-50 (image-only retrieval via feature matching)

All baselines are evaluated on the same veterinary test set for fair comparison.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.metrics import compute_retrieval_metrics, compute_similarity_matrix
from utils.logger import get_logger


# ---------------------------------------------------------------------------
# Generic baseline interface
# ---------------------------------------------------------------------------

class BaselineEvaluator:
    """
    Abstract baseline evaluator.

    Subclasses implement ``encode_image`` and ``encode_text``.
    """

    name: str = "baseline"

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, loader, device, smoke_test=False, k_values=(1, 5, 10)):
        self.device = device
        vision_embeds, text_embeds = [], []
        max_batches = 2 if smoke_test else len(loader)

        for step, batch in enumerate(tqdm(loader, desc=f"Eval {self.name}")):
            if step >= max_batches:
                break
            images = batch["image"].to(device)
            texts = batch["text"]
            v = self.encode_image(images)
            t = self.encode_text(texts)
            vision_embeds.append(F.normalize(v, p=2, dim=-1).cpu())
            text_embeds.append(F.normalize(t, p=2, dim=-1).cpu())

        vision_embeds = torch.cat(vision_embeds, dim=0)
        text_embeds = torch.cat(text_embeds, dim=0)
        sim_matrix = compute_similarity_matrix(vision_embeds, text_embeds)
        return compute_retrieval_metrics(sim_matrix, k_values=list(k_values))


# ---------------------------------------------------------------------------
# CLIP baseline
# ---------------------------------------------------------------------------

class CLIPBaseline(BaselineEvaluator):
    name = "CLIP"

    def __init__(self, model_id: str = "openai/clip-vit-base-patch16"):
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(model_id)
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP: {e}")

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        self.model = self.model.to(images.device)
        outputs = self.model.get_image_features(pixel_values=images)
        return outputs

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = self.model.get_text_features(**inputs)
        return outputs


# ---------------------------------------------------------------------------
# BiomedCLIP baseline
# ---------------------------------------------------------------------------

class BiomedCLIPBaseline(BaselineEvaluator):
    name = "BiomedCLIP"

    def __init__(self, model_id: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        try:
            from transformers import CLIPModel, AutoProcessor
            self.model = CLIPModel.from_pretrained(model_id, ignore_mismatched_sizes=True)
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load BiomedCLIP: {e}")

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        self.model = self.model.to(images.device)
        return self.model.get_image_features(pixel_values=images)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        return self.model.get_text_features(**inputs)


# ---------------------------------------------------------------------------
# ResNet-50 baseline (vision-only, uses image features for retrieval)
# ---------------------------------------------------------------------------

class ResNet50Baseline(BaselineEvaluator):
    """
    ResNet-50 image-only baseline.

    Since ResNet-50 does not have a text encoder, we use random Gaussian
    vectors for text embeddings.  This serves as a lower-bound reference.
    """
    name = "ResNet50"

    def __init__(self):
        import torchvision.models as tvm
        self.model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()  # Remove classification head
        self.model.eval()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        self.model = self.model.to(images.device)
        return self.model(images)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        # ResNet-50 has no text encoder — return random features (same dim)
        return torch.randn(len(texts), 2048)


# ---------------------------------------------------------------------------
# Smoke-test stub baseline
# ---------------------------------------------------------------------------

class RandomBaseline(BaselineEvaluator):
    """Random embedding baseline — used in smoke tests."""
    name = "Random"

    def __init__(self, embed_dim: int = 512):
        self.embed_dim = embed_dim

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return torch.randn(images.size(0), self.embed_dim)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        return torch.randn(len(texts), self.embed_dim)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

BASELINE_REGISTRY = {
    "CLIP": CLIPBaseline,
    "BiomedCLIP": BiomedCLIPBaseline,
    "ResNet50": ResNet50Baseline,
    "Random": RandomBaseline,
}


def run_all_baselines(
    loader,
    device: torch.device,
    baselines: List[str] = ("CLIP", "BiomedCLIP", "ResNet50"),
    k_values: List[int] = (1, 5, 10),
    smoke_test: bool = False,
    logger=None,
) -> Dict:
    """
    Evaluate all specified baselines.

    In smoke_test mode, uses RandomBaseline for all to avoid network downloads.
    """
    results = {}

    for bname in baselines:
        if logger:
            logger.info(f"\n--- Baseline: {bname} ---")
        try:
            if smoke_test:
                evaluator = RandomBaseline()
                evaluator.name = bname
            else:
                cls = BASELINE_REGISTRY.get(bname, RandomBaseline)
                evaluator = cls()

            metrics = evaluator.evaluate(loader, device, smoke_test=smoke_test, k_values=k_values)
            results[bname] = metrics

            if logger:
                for k, v in metrics.items():
                    logger.info(f"  {k}: {v:.4f}")
        except Exception as e:
            if logger:
                logger.warning(f"  Failed to evaluate {bname}: {e}")
            results[bname] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------

def main():
    import argparse
    from omegaconf import OmegaConf
    from data.veterinary import SyntheticVetDataset
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Baseline Evaluation")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    smoke = args.smoke_test
    logger = get_logger("baselines", log_dir="results")
    device = torch.device("cpu" if smoke else cfg.evaluation.get("device", "cuda"))

    if smoke:
        test_ds = SyntheticVetDataset(num_samples=20)
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    else:
        from data.veterinary import build_vet_dataloaders
        _, _, test_loader = build_vet_dataloaders(cfg, smoke_test=False)

    baseline_names = [b["name"] for b in cfg.get("baselines", [{"name": "Random"}])]
    results = run_all_baselines(
        loader=test_loader,
        device=device,
        baselines=baseline_names,
        k_values=cfg.retrieval.get("k_values", [1, 5, 10]),
        smoke_test=smoke,
        logger=logger,
    )

    out_dir = Path(cfg.output.get("results_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Baseline results saved to {out_dir / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
