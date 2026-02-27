"""
Zero-Shot Classification Evaluation for VetVision-LM.

Classifies veterinary radiographs by species (canine / feline) without
any classification-specific training, using text prompt templates.

Metrics: Accuracy, AUC-ROC, F1 per species.
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

from models.vetvision import VetVisionLM
from utils.metrics import classification_metrics
from utils.logger import get_logger


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DEFAULT_TEMPLATES = [
    "A radiograph of a {} patient.",
    "Chest X-ray of a {}.",
    "Veterinary thoracic radiograph: {}.",
    "{} chest radiograph findings.",
]

SPECIES_PROMPTS = {
    0: "canine",
    1: "feline",
}


def build_class_embeddings(
    model: VetVisionLM,
    device: torch.device,
    templates: List[str],
    species_map: Dict[int, str],
) -> torch.Tensor:
    """
    Build average text embedding for each species using prompt templates.

    Args:
        model:      VetVisionLM instance.
        device:     Torch device.
        templates:  List of template strings with one ``{}`` placeholder.
        species_map: Dict mapping class index to species name string.

    Returns:
        (num_classes, embed_dim) class prototype tensor.
    """
    model.eval()
    class_embeds = []

    with torch.no_grad():
        for cls_idx in sorted(species_map.keys()):
            species_name = species_map[cls_idx]
            texts = [t.format(species_name) for t in templates]
            text_cls, _ = model.text_encoder(texts=texts)
            _, text_embed = model.projector(
                torch.zeros(len(texts), model.vision_encoder.output_dim, device=device),
                text_cls,
            )
            # Average over templates
            class_embed = text_embed.mean(dim=0, keepdim=True)  # (1, D)
            class_embed = F.normalize(class_embed, p=2, dim=-1)
            class_embeds.append(class_embed)

    return torch.cat(class_embeds, dim=0)  # (C, D)


@torch.no_grad()
def zero_shot_predict(
    model: VetVisionLM,
    loader,
    class_prototypes: torch.Tensor,
    device: torch.device,
    smoke_test: bool = False,
):
    """
    Run zero-shot classification on a dataloader.

    Returns:
        y_true:  (N,) ground-truth labels.
        y_pred:  (N,) predicted labels.
        y_score: (N, C) softmax probability scores.
    """
    model.eval()
    all_true, all_pred, all_score = [], [], []
    max_batches = 2 if smoke_test else len(loader)

    for step, batch in enumerate(tqdm(loader, desc="Zero-shot classify")):
        if step >= max_batches:
            break

        images = batch["image"].to(device)
        gt_labels = batch["species_label"]

        # Encode images
        vision_cls, _ = model.vision_encoder(images)
        vision_embed, _ = model.projector(
            vision_cls,
            torch.zeros(vision_cls.size(0), model.text_encoder.output_dim, device=device),
        )  # (B, D)

        # Cosine similarity to class prototypes → logits
        logits = vision_embed @ class_prototypes.T / 0.07   # (B, C)
        scores = torch.softmax(logits, dim=-1)               # (B, C)
        preds = scores.argmax(dim=-1)                        # (B,)

        all_true.append(gt_labels.numpy())
        all_pred.append(preds.cpu().numpy())
        all_score.append(scores.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_score = np.concatenate(all_score, axis=0)
    return y_true, y_pred, y_score


def run_classification_evaluation(
    model,
    loader,
    device,
    templates=None,
    smoke_test=False,
    logger=None,
):
    """
    Full zero-shot classification pipeline.

    Returns:
        Dict of metrics.
    """
    templates = templates or DEFAULT_TEMPLATES
    prototypes = build_class_embeddings(model, device, templates, SPECIES_PROMPTS)
    y_true, y_pred, y_score = zero_shot_predict(
        model, loader, prototypes, device, smoke_test=smoke_test
    )
    metrics = classification_metrics(
        y_true, y_pred, y_score=y_score,
        class_names=list(SPECIES_PROMPTS.values()),
    )

    if logger:
        logger.info("=== Zero-Shot Classification ===")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics, y_true, y_pred, y_score


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from omegaconf import OmegaConf
    from data.veterinary import build_vet_dataloaders, SyntheticVetDataset

    parser = argparse.ArgumentParser(description="Zero-Shot Species Classification")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    smoke = args.smoke_test
    logger = get_logger("eval_classification", log_dir="results")
    device = torch.device("cpu" if smoke else cfg.evaluation.get("device", "cuda"))

    model_cfg_path = Path("configs/finetune.yaml")
    model_cfg = OmegaConf.load(model_cfg_path) if model_cfg_path.exists() else cfg
    model = VetVisionLM.from_config(model_cfg).to(device)

    ckpt_path = args.checkpoint or cfg.model.get("checkpoint", None)
    if ckpt_path and Path(ckpt_path).exists() and not smoke:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    if smoke:
        from torch.utils.data import DataLoader
        test_ds = SyntheticVetDataset(num_samples=20)
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    else:
        _, _, test_loader = build_vet_dataloaders(cfg, smoke_test=False)

    templates = list(cfg.evaluation.get(
        "zero_shot_templates", DEFAULT_TEMPLATES
    ))

    metrics, y_true, y_pred, y_score = run_classification_evaluation(
        model, test_loader, device,
        templates=templates, smoke_test=smoke, logger=logger,
    )

    # Paper reference
    paper = cfg.get("paper_results", {})
    out = {
        "reproduced": metrics,
        "paper_results_note": paper.get("note", "Reported in Paper (not reproduced)"),
        "paper_zero_shot_overall": paper.get("zero_shot_accuracy_overall", 77.3),
        "paper_zero_shot_canine": paper.get("zero_shot_accuracy_canine", 79.0),
        "paper_zero_shot_feline": paper.get("zero_shot_accuracy_feline", 75.5),
    }

    out_dir = Path(cfg.output.get("results_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "classification_results.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved classification results to {out_dir / 'classification_results.json'}")


if __name__ == "__main__":
    main()
