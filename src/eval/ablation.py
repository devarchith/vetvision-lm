"""
Ablation Study Runner for VetVision-LM.

Systematically evaluates the contribution of each component:
    - Full model
    - Without species-adaptive module
    - Without species-aware loss
    - Without any species components

Paper-reported ablation results (NOT reproduced — reference only):
    Full model:               R@1 = 55.1%
    - species_module:         R@1 = 48.6%  (Δ = -6.5)
    - species_loss:           R@1 = 51.3%  (Δ = -3.8)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from omegaconf import OmegaConf

from models.vetvision import VetVisionLM
from losses.contrastive import ContrastiveLoss
from losses.species_loss import CombinedLoss, SpeciesContrastiveLoss
from eval.retrieval import run_retrieval_evaluation
from utils.logger import get_logger


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = [
    {
        "name": "full_model",
        "description": "Full VetVision-LM (vision + text + species module + species loss)",
        "use_species_module": True,
        "use_species_loss": True,
    },
    {
        "name": "no_species_module",
        "description": "Without species-adaptive module",
        "use_species_module": False,
        "use_species_loss": True,
    },
    {
        "name": "no_species_loss",
        "description": "Without species-aware contrastive loss",
        "use_species_module": True,
        "use_species_loss": False,
    },
    {
        "name": "no_species",
        "description": "Without any species-specific components",
        "use_species_module": False,
        "use_species_loss": False,
    },
]


def build_ablation_model(
    base_cfg,
    use_species_module: bool,
    device: torch.device,
    checkpoint_path: str = None,
) -> VetVisionLM:
    """Construct a VetVisionLM with specified ablation settings."""
    model = VetVisionLM(
        vision_cfg=dict(base_cfg.model.vision_encoder),
        text_cfg=dict(base_cfg.model.text_encoder),
        species_cfg=dict(base_cfg.model.species_module),
        proj_cfg=dict(base_cfg.model.projection),
        use_species_module=use_species_module,
    ).to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    return model


def run_ablation_study(
    cfg,
    test_loader,
    device: torch.device,
    checkpoint_path: str = None,
    smoke_test: bool = False,
    logger=None,
) -> Dict:
    """
    Run all ablation configurations and collect results.

    Returns:
        Dict mapping ablation name → retrieval metrics.
    """
    results = {}
    paper_deltas = {
        "no_species_module": -6.5,
        "no_species_loss": -3.8,
        "no_species": -10.3,
    }

    for abl_cfg in ABLATION_CONFIGS:
        name = abl_cfg["name"]
        desc = abl_cfg["description"]
        if logger:
            logger.info(f"\n{'='*60}")
            logger.info(f"Ablation: {name}")
            logger.info(f"Description: {desc}")

        model = build_ablation_model(
            base_cfg=cfg,
            use_species_module=abl_cfg["use_species_module"],
            device=device,
            checkpoint_path=checkpoint_path,
        )

        eval_results = run_retrieval_evaluation(
            model=model,
            loader=test_loader,
            device=device,
            k_values=[1, 5, 10],
            smoke_test=smoke_test,
            logger=logger,
        )

        results[name] = {
            "description": desc,
            "use_species_module": abl_cfg["use_species_module"],
            "use_species_loss": abl_cfg["use_species_loss"],
            "metrics": eval_results["overall"],
            "per_species": eval_results["per_species"],
        }

        if name in paper_deltas:
            results[name]["paper_delta_r1"] = paper_deltas[name]
            results[name]["paper_note"] = "Reported in Paper (not reproduced)"

        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ---------------------------------------------------------------------------

def main():
    import argparse
    from data.veterinary import build_vet_dataloaders, SyntheticVetDataset
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="VetVision-LM Ablation Study")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    eval_cfg = OmegaConf.load(args.config)
    model_cfg = OmegaConf.load("configs/finetune.yaml")
    smoke = args.smoke_test

    logger = get_logger("ablation", log_dir="results")
    device = torch.device("cpu" if smoke else eval_cfg.evaluation.get("device", "cuda"))

    if smoke:
        test_ds = SyntheticVetDataset(num_samples=20)
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    else:
        _, _, test_loader = build_vet_dataloaders(eval_cfg, smoke_test=False)

    ckpt_path = args.checkpoint or eval_cfg.model.get("checkpoint", None)

    results = run_ablation_study(
        cfg=model_cfg,
        test_loader=test_loader,
        device=device,
        checkpoint_path=ckpt_path,
        smoke_test=smoke,
        logger=logger,
    )

    out_dir = Path(eval_cfg.output.get("results_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ablation_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Ablation results saved to {out_file}")

    # Print summary table
    logger.info("\n=== ABLATION SUMMARY ===")
    logger.info(f"{'Config':<25} {'i2t_R@1':>10} {'t2i_R@1':>10}")
    logger.info("-" * 50)
    for name, res in results.items():
        m = res["metrics"]
        logger.info(
            f"{name:<25} {m.get('i2t_R@1', 0):.4f}     {m.get('t2i_R@1', 0):.4f}"
        )


if __name__ == "__main__":
    main()
