"""
Veterinary fine-tuning loop for VetVision-LM.

Loads a pretrained VetVision-LM checkpoint and fine-tunes on the veterinary
species dataset with both contrastive and species-aware losses.

Usage:
    python scripts/finetune.py --config configs/finetune.yaml
    python scripts/finetune.py --smoke-test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from omegaconf import OmegaConf

from data.veterinary import build_vet_dataloaders
from data.augmentations import build_train_transform, build_val_transform
from models.vetvision import VetVisionLM
from losses.contrastive import ContrastiveLoss
from losses.species_loss import CombinedLoss, SpeciesContrastiveLoss
from utils.logger import get_logger, MetricLogger


# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, criterion, scaler, device, epoch, cfg,
    metric_logger, smoke_test=False,
):
    model.train()
    total_loss = total_cont = total_spec = 0.0
    steps = 0
    max_steps = 2 if smoke_test else len(loader)

    for step, batch in enumerate(loader):
        if step >= max_steps:
            break

        images = batch["image"].to(device)
        texts = batch["text"]
        species_ids = batch["species_label"].to(device)

        with autocast(enabled=(device.type == "cuda")):
            out = model(images=images, texts=texts, species_ids=species_ids)
            loss_dict = criterion(
                out.vision_embed, out.text_embed, out.species_embed, species_ids
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

        if step % cfg.logging.get("log_every", 50) == 0:
            metric_logger.log(
                {
                    "epoch": epoch,
                    "step": step,
                    "loss": loss.item(),
                    "loss_contrastive": loss_dict["contrastive"].item(),
                    "loss_species": loss_dict["species"].item(),
                }
            )

    n = max(steps, 1)
    return {"loss": total_loss / n, "loss_contrastive": total_cont / n, "loss_species": total_spec / n}


@torch.no_grad()
def evaluate(model, loader, criterion, device, smoke_test=False):
    model.eval()
    total_loss = 0.0
    steps = 0
    max_steps = 2 if smoke_test else len(loader)

    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        images = batch["image"].to(device)
        texts = batch["text"]
        species_ids = batch["species_label"].to(device)

        out = model(images=images, texts=texts, species_ids=species_ids)
        loss_dict = criterion(out.vision_embed, out.text_embed, out.species_embed, species_ids)
        total_loss += loss_dict["total"].item()
        steps += 1

    return {"val_loss": total_loss / max(steps, 1)}


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VetVision-LM Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path (defaults to cfg.model.checkpoint)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    smoke = args.smoke_test

    if smoke:
        cfg.training.num_epochs = 2
        cfg.training.batch_size = 4
        cfg.hardware.device = "cpu"
        cfg.logging.log_dir = "logs/smoke_finetune"
        cfg.logging.checkpoint_dir = "checkpoints/smoke_finetune"

    log_dir = cfg.logging.get("log_dir", "logs/finetune")
    ckpt_dir = cfg.logging.get("checkpoint_dir", "checkpoints/finetune")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    logger = get_logger("finetune", log_dir=log_dir)
    metric_logger = MetricLogger(
        logger,
        use_wandb=cfg.logging.get("wandb", {}).get("enabled", False),
    )

    torch.manual_seed(cfg.hardware.get("seed", 42))
    device = torch.device("cpu" if smoke else cfg.hardware.get("device", "cuda"))
    logger.info(f"Device: {device}  |  smoke_test={smoke}")

    # Build data
    aug_cfg = dict(cfg.data.get("augmentation", {}))
    train_tf = build_train_transform(augmentation_cfg=aug_cfg) if not smoke else None
    val_tf = build_val_transform() if not smoke else None

    train_loader, val_loader, test_loader = build_vet_dataloaders(
        cfg, train_transform=train_tf, val_transform=val_tf, smoke_test=smoke
    )
    logger.info(f"Train: {len(train_loader)}  Val: {len(val_loader)}  Test: {len(test_loader)}")

    # Build model
    model = VetVisionLM.from_config(cfg).to(device)

    # Load pretrained weights if available
    ckpt_path = args.checkpoint or cfg.model.get("checkpoint", None)
    if ckpt_path and Path(ckpt_path).exists() and not smoke:
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded pretrained weights from: {ckpt_path}")
    elif not smoke:
        logger.warning("No pretrained checkpoint found — fine-tuning from scratch.")

    # Losses
    cont_loss = ContrastiveLoss(
        temperature=cfg.loss.get("temperature", 0.07),
        learnable_temp=True,
    )
    spec_loss = SpeciesContrastiveLoss(
        temperature=cfg.loss.get("temperature", 0.07),
        margin=cfg.loss.get("margin", 0.2),
    )
    criterion = CombinedLoss(cont_loss, spec_loss, lambda_species=cfg.loss.get("lambda_species", 0.5))
    criterion = criterion.to(device)

    # Optimiser & schedule
    optimizer = AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg.training.get("learning_rate", 5e-5),
        weight_decay=cfg.training.get("weight_decay", 1e-4),
    )
    num_epochs = cfg.training.get("num_epochs", 30)
    warmup_epochs = cfg.training.get("warmup_epochs", 3)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        logger.info(f"--- Epoch {epoch+1}/{num_epochs} ---")
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, cfg, metric_logger, smoke)
        val_m = evaluate(model, val_loader, criterion, device, smoke)
        scheduler.step()

        metric_logger.log({**train_m, **val_m, "epoch": epoch + 1})

        if (epoch + 1) % cfg.logging.get("save_every", 5) == 0 or smoke:
            p = Path(ckpt_dir) / f"epoch_{epoch+1:03d}.pth"
            torch.save({"epoch": epoch, "model": model.state_dict()}, p)
            logger.info(f"Saved: {p}")

        if val_m["val_loss"] < best_val_loss:
            best_val_loss = val_m["val_loss"]
            torch.save({"epoch": epoch, "model": model.state_dict()}, Path(ckpt_dir) / "best.pth")

    logger.info("Fine-tuning complete.")
    metric_logger.finish()


if __name__ == "__main__":
    main()
