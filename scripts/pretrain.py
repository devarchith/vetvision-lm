"""
CLI entry point for VetVision-LM pretraining.

Delegates to src/train/pretrain.py::main()

Usage:
    python scripts/pretrain.py --smoke-test
    python scripts/pretrain.py --config configs/pretrain.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train.pretrain import main

if __name__ == "__main__":
    main()
