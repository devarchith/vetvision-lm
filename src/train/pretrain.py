"""
CheXpert pretraining loop for VetVision-LM.

Trains vision and text encoders jointly with symmetric contrastive loss
on CheXpert image–report pairs.

Usage:
    python scripts/pretrain.py --config configs/pretrain.yaml
    python scripts/pretrain.py --smoke-test
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from omegaconf import OmegaConf

from data.chexpert import build_chexpert_dataloaders
from data.augmentations import build_train_transform, build_val_transform
from models.vetvision import VetVisionLM
from losses.contrastive import ContrastiveLoss
from losses.species_loss import CombinedLoss, SpeciesContrastiveLoss
from utils.logger import get_logger, MetricLogger


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: VetVisionLM,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLoss,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg,
    metric_logger: MetricLogger,
    smoke_test: bool = False,
) -> dict:
    model.train()
    total_loss = 0.0
    total_cont = 0.0
    total_spec = 0.0
    steps = 0

    max_steps = 2 if smoke_test else len(loader)

    for step, batch in enumerate(loader):
        if step >= max_steps:
            break

        images = batch["image"].to(device, non_blocking=True)
        texts = batch["text"]                          # list of strings
        species_ids = batch.get("species_label")
        if species_ids is not None:
            species_ids = species_ids.to(device)

        with autocast(enabled=(device.type == "cuda")):
            out = model(
                images=images,
                texts=texts,
                species_ids=species_ids,
            )
            loss_dict = criterion(
                vision_embed=out.vision_embed,
                text_embed=out.text_embed,
                species_embed=out.species_embed,
                species_ids=species_ids,
            )
            loss = loss_dict["total"]

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if cfg.training.get("gradient_clip", 0) > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.gradient_clip
            )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_cont += loss_dict["contrastive"].item()
        total_spec += loss_dict["species"].item()
        steps += 1

        if step % cfg.logging.get("log_every", 100) == 0:
            metric_logger.log(
                {
                    "epoch": epoch,
                    "step": step,
                    "loss": loss.item(),
                    "loss_contrastive": loss_dict["contrastive"].item(),
                    "loss_species": loss_dict["species"].item(),
                },
                step=epoch * len(loader) + step,
            )

    return {
        "loss": total_loss / max(steps, 1),
        "loss_contrastive": total_cont / max(steps, 1),
        "loss_species": total_spec / max(steps, 1),
    }


@torch.no_grad()
def evaluate(
    model: VetVisionLM,
    loader: torch.utils.data.DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    smoke_test: bool = False,
) -> dict:
    model.eval()
    total_loss = 0.0
    steps = 0
    max_steps = 2 if smoke_test else len(loader)

    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        images = batch["image"].to(device)
        texts = batch["text"]
        species_ids = batch.get("species_label")
        if species_ids is not None:
            species_ids = species_ids.to(device)

        out = model(images=images, texts=texts, species_ids=species_ids)
        loss_dict = criterion(
            out.vision_embed, out.text_embed, out.species_embed, species_ids
        )
        total_loss += loss_dict["total"].item()
        steps += 1

    return {"val_loss": total_loss / max(steps, 1)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VetVision-LM CheXpert Pretraining")
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run with synthetic data for 2 epochs (no GPU/data needed)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    smoke = args.smoke_test
    if smoke:
        # Override training params for smoke test
        cfg.training.num_epochs = 2
        cfg.training.batch_size = 4
        cfg.hardware.device = "cpu"
        cfg.logging.log_dir = "logs/smoke_pretrain"
        cfg.logging.checkpoint_dir = "checkpoints/smoke_pretrain"

    # Setup
    log_dir = cfg.logging.get("log_dir", "logs/pretrain")
    ckpt_dir = cfg.logging.get("checkpoint_dir", "checkpoints/pretrain")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    logger = get_logger("pretrain", log_dir=log_dir)
    metric_logger = MetricLogger(
        logger,
        use_wandb=cfg.logging.get("wandb", {}).get("enabled", False),
        wandb_config=dict(cfg.logging.get("wandb", {})),
    )

    torch.manual_seed(cfg.hardware.get("seed", 42))
    device = torch.device(cfg.hardware.get("device", "cpu") if not smoke else "cpu")
    logger.info(f"Device: {device}  |  smoke_test={smoke}")

    # Build data
    aug_cfg = dict(cfg.data.get("augmentation", {}))
    train_tf = build_train_transform(augmentation_cfg=aug_cfg) if not smoke else None
    val_tf = build_val_transform() if not smoke else None

    train_loader, val_loader = build_chexpert_dataloaders(
        cfg, train_transform=train_tf, val_transform=val_tf, smoke_test=smoke
    )
    logger.info(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # Build model
    model = VetVisionLM.from_config(cfg).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Losses
    cont_loss = ContrastiveLoss(
        temperature=cfg.loss.get("temperature", 0.07),
        learnable_temp=True,
    )
    spec_loss = SpeciesContrastiveLoss(
        temperature=cfg.loss.get("temperature", 0.07),
        margin=cfg.loss.get("margin", 0.2),
    )
    criterion = CombinedLoss(
        contrastive_loss=cont_loss,
        species_loss=spec_loss,
        lambda_species=cfg.loss.get("lambda_species", 0.5),
    )
    criterion = criterion.to(device)

    # Optimiser & LR schedule
    optimizer = AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg.training.get("learning_rate", 1e-4),
        weight_decay=cfg.training.get("weight_decay", 1e-4),
    )
    num_epochs = cfg.training.get("num_epochs", 50)
    warmup_epochs = cfg.training.get("warmup_epochs", 5)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Resume
    start_epoch = 0
    if args.resume and not smoke:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"--- Epoch {epoch+1}/{num_epochs} ---")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, epoch, cfg, metric_logger, smoke_test=smoke,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, smoke_test=smoke)
        scheduler.step()

        epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}
        metric_logger.log(epoch_metrics)

        # Checkpoint
        save_every = cfg.logging.get("save_every", 5)
        if (epoch + 1) % save_every == 0 or smoke:
            ckpt_path = Path(ckpt_dir) / f"epoch_{epoch+1:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": OmegaConf.to_container(cfg),
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(
                {"epoch": epoch, "model": model.state_dict()},
                Path(ckpt_dir) / "best.pth",
            )

    logger.info("Pretraining complete.")
    metric_logger.finish()


if __name__ == "__main__":
    main()
