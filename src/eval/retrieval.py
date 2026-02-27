"""
Image-Text Retrieval Evaluation for VetVision-LM.

Computes:
    - Recall@1, Recall@5, Recall@10
    - Mean Reciprocal Rank (MRR)
    - Both image→text and text→image directions

Usage:
    python scripts/evaluate.py --config configs/eval.yaml --checkpoint path/to/best.pth
    python scripts/evaluate.py --smoke-test
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from models.vetvision import VetVisionLM
from data.veterinary import build_vet_dataloaders, SyntheticVetDataset
from utils.metrics import compute_retrieval_metrics, compute_similarity_matrix
from utils.logger import get_logger


# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, loader, device, smoke_test=False):
    """Extract vision and text embeddings from the test set."""
    model.eval()
    vision_embeds = []
    text_embeds = []
    species_list = []
    paths = []

    max_batches = 2 if smoke_test else len(loader)

    for step, batch in enumerate(tqdm(loader, desc="Extracting embeddings")):
        if step >= max_batches:
            break

        images = batch["image"].to(device)
        texts = batch["text"]
        species = batch.get("species_label", None)

        out = model(images=images, texts=texts)
        vision_embeds.append(out.vision_embed.cpu())
        text_embeds.append(out.text_embed.cpu())

        if species is not None:
            species_list.append(species.cpu())
        paths.extend(batch.get("path", [""] * images.size(0)))

    vision_embeds = torch.cat(vision_embeds, dim=0)   # (N, D)
    text_embeds = torch.cat(text_embeds, dim=0)        # (N, D)
    species_arr = torch.cat(species_list, dim=0).numpy() if species_list else None

    return vision_embeds, text_embeds, species_arr, paths


def run_retrieval_evaluation(
    model,
    loader,
    device,
    k_values=(1, 5, 10),
    smoke_test=False,
    logger=None,
):
    """
    Full retrieval evaluation pipeline.

    Returns:
        Dict of metrics and arrays for further analysis.
    """
    vision_embeds, text_embeds, species_arr, paths = extract_embeddings(
        model, loader, device, smoke_test=smoke_test
    )

    sim_matrix = compute_similarity_matrix(vision_embeds, text_embeds)
    metrics = compute_retrieval_metrics(sim_matrix, k_values=list(k_values))

    if logger:
        logger.info("=== Retrieval Metrics ===")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    # Per-species breakdown if available
    species_metrics = {}
    if species_arr is not None:
        for s_id, s_name in [(0, "canine"), (1, "feline")]:
            mask = species_arr == s_id
            if mask.sum() < 2:
                continue
            sub_sim = sim_matrix[np.ix_(mask, mask)]
            sub_m = compute_retrieval_metrics(sub_sim, k_values=list(k_values))
            species_metrics[s_name] = sub_m
            if logger:
                logger.info(f"  --- {s_name} ---")
                for k, v in sub_m.items():
                    logger.info(f"    {k}: {v:.4f}")

    return {
        "overall": metrics,
        "per_species": species_metrics,
        "similarity_matrix": sim_matrix,
        "vision_embeds": vision_embeds.numpy(),
        "text_embeds": text_embeds.numpy(),
        "species": species_arr,
        "paths": paths,
    }


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VetVision-LM Retrieval Evaluation")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    smoke = args.smoke_test

    logger = get_logger("eval_retrieval", log_dir=cfg.output.get("results_dir", "results"))
    device = torch.device("cpu" if smoke else cfg.evaluation.get("device", "cuda"))

    # Build model
    # Use finetune config for model structure
    model_cfg_path = Path("configs/finetune.yaml")
    if model_cfg_path.exists():
        model_cfg = OmegaConf.load(model_cfg_path)
        model = VetVisionLM.from_config(model_cfg).to(device)
    else:
        model = VetVisionLM().to(device)

    ckpt_path = args.checkpoint or cfg.model.get("checkpoint", None)
    if ckpt_path and Path(ckpt_path).exists() and not smoke:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        logger.info(f"Loaded checkpoint: {ckpt_path}")
    elif not smoke:
        logger.warning("No checkpoint provided — evaluating with random weights.")

    # Build test loader
    if smoke:
        from torch.utils.data import DataLoader
        test_ds = SyntheticVetDataset(num_samples=20)
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    else:
        _, _, test_loader = build_vet_dataloaders(cfg, smoke_test=False)

    # Run evaluation
    results = run_retrieval_evaluation(
        model=model,
        loader=test_loader,
        device=device,
        k_values=cfg.retrieval.get("k_values", [1, 5, 10]),
        smoke_test=smoke,
        logger=logger,
    )

    # Add paper results as reference
    paper = cfg.get("paper_results", {})
    results["paper_results_note"] = paper.get("note", "Reported in Paper (not reproduced)")
    results["paper_recall_at_1"] = paper.get("retrieval_recall_at_1", 55.1)

    # Save
    out_dir = Path(cfg.output.get("results_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "retrieval_results.json"
    serialisable = {
        "overall": results["overall"],
        "per_species": results["per_species"],
        "paper_results_note": results["paper_results_note"],
        "paper_recall_at_1": results["paper_recall_at_1"],
    }
    with open(report_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info(f"Results saved to {report_path}")


if __name__ == "__main__":
    main()
