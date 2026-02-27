"""Logging utilities for VetVision-LM training and evaluation."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "vetvision",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a logger that writes to both stdout and (optionally) a log file.

    Args:
        name:    Logger name.
        log_dir: Directory for log file.  If None, only stdout handler added.
        level:   Logging level.

    Returns:
        Configured ``logging.Logger``.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # File handler
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"{name}_{timestamp}.log"
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class MetricLogger:
    """
    Lightweight metric logger that can write to stdout and optionally W&B.

    Args:
        logger:       Python logger instance.
        use_wandb:    Enable Weights & Biases logging.
        wandb_config: W&B init kwargs (project, entity, config, …).
    """

    def __init__(
        self,
        logger: logging.Logger,
        use_wandb: bool = False,
        wandb_config: Optional[dict] = None,
    ) -> None:
        self.logger = logger
        self.use_wandb = use_wandb
        self._wandb = None

        if use_wandb:
            try:
                import wandb
                wandb.init(**(wandb_config or {}))
                self._wandb = wandb
            except ImportError:
                self.logger.warning("wandb not installed — skipping W&B logging.")
                self.use_wandb = False

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log a dict of metric values."""
        msg = "  ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        if step is not None:
            msg = f"step={step}  " + msg
        self.logger.info(msg)

        if self.use_wandb and self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.use_wandb and self._wandb is not None:
            self._wandb.finish()
