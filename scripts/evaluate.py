"""
CLI entry point for VetVision-LM evaluation.

Runs retrieval, zero-shot classification, ablation, and baselines.

Usage:
    python scripts/evaluate.py --smoke-test
    python scripts/evaluate.py --config configs/eval.yaml --checkpoint path/to/best.pth
    python scripts/evaluate.py --config configs/eval.yaml --checkpoint path/to/best.pth --mode retrieval
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from omegaconf import OmegaConf

from models.vetvision import VetVisionLM
from eval.retrieval import run_retrieval_evaluation
from eval.classification import run_classification_evaluation, DEFAULT_TEMPLATES
from eval.ablation import run_ablation_study
from eval.baselines import run_all_baselines
from utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser(description="VetVision-LM Evaluation")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "retrieval", "classification", "ablation", "baselines"],
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    smoke = args.smoke_test
    logger = get_logger("evaluate", log_dir=cfg.output.get("results_dir", "results"))
    device = torch.device("cpu" if smoke else cfg.evaluation.get("device", "cuda"))

    # Build test loader
    if smoke:
        from data.veterinary import SyntheticVetDataset
        from torch.utils.data import DataLoader
        test_ds = SyntheticVetDataset(num_samples=20)
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    else:
        from data.veterinary import build_vet_dataloaders
        _, _, test_loader = build_vet_dataloaders(cfg, smoke_test=False)

    # Build model
    model_cfg_path = Path("configs/finetune.yaml")
    model_cfg = OmegaConf.load(model_cfg_path) if model_cfg_path.exists() else cfg
    model = VetVisionLM.from_config(model_cfg).to(device)

    ckpt_path = args.checkpoint or cfg.model.get("checkpoint", None)
    if ckpt_path and Path(ckpt_path).exists() and not smoke:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        logger.info(f"Loaded checkpoint: {ckpt_path}")

    out_dir = Path(cfg.output.get("results_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    paper = cfg.get("paper_results", {})

    # ---- Retrieval ----
    if args.mode in ("all", "retrieval"):
        logger.info("\n===== RETRIEVAL EVALUATION =====")
        ret = run_retrieval_evaluation(
            model, test_loader, device,
            k_values=cfg.retrieval.get("k_values", [1, 5, 10]),
            smoke_test=smoke, logger=logger,
        )
        all_results["retrieval"] = ret["overall"]
        all_results["retrieval_per_species"] = ret["per_species"]

    # ---- Classification ----
    if args.mode in ("all", "classification"):
        logger.info("\n===== ZERO-SHOT CLASSIFICATION =====")
        cls_metrics, _, _, _ = run_classification_evaluation(
            model, test_loader, device,
            templates=DEFAULT_TEMPLATES, smoke_test=smoke, logger=logger,
        )
        all_results["classification"] = cls_metrics

    # ---- Ablation ----
    if args.mode in ("all", "ablation"):
        logger.info("\n===== ABLATION STUDY =====")
        abl = run_ablation_study(
            cfg=model_cfg, test_loader=test_loader, device=device,
            checkpoint_path=ckpt_path, smoke_test=smoke, logger=logger,
        )
        all_results["ablation"] = {k: v["metrics"] for k, v in abl.items()}

    # ---- Baselines ----
    if args.mode in ("all", "baselines"):
        logger.info("\n===== BASELINES =====")
        baseline_names = [b["name"] for b in cfg.get("baselines", [{"name": "Random"}])]
        bl = run_all_baselines(
            test_loader, device,
            baselines=baseline_names,
            k_values=cfg.retrieval.get("k_values", [1, 5, 10]),
            smoke_test=smoke, logger=logger,
        )
        all_results["baselines"] = bl

    # Paper reference
    all_results["paper_results"] = {
        "note": paper.get("note", "Reported in Paper (not reproduced)"),
        "i2t_R@1": paper.get("retrieval_recall_at_1", 55.1),
        "chexzero_baseline_i2t_R@1": paper.get("chexzero_baseline_recall_at_1", 42.8),
        "zero_shot_overall": paper.get("zero_shot_accuracy_overall", 77.3),
        "zero_shot_canine": paper.get("zero_shot_accuracy_canine", 79.0),
        "zero_shot_feline": paper.get("zero_shot_accuracy_feline", 75.5),
    }

    # Save combined report
    report_path = out_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        # Convert numpy/torch types for JSON serialisation
        def default_serial(obj):
            if hasattr(obj, "item"):
                return obj.item()
            return str(obj)
        json.dump(all_results, f, indent=2, default=default_serial)

    logger.info(f"\nFull evaluation report: {report_path}")


if __name__ == "__main__":
    main()
