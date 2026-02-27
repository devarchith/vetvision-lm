"""
CLI entry point for VetVision-LM fine-tuning.

Delegates to src/train/finetune.py::main()

Usage:
    python scripts/finetune.py --smoke-test
    python scripts/finetune.py --config configs/finetune.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train.finetune import main

if __name__ == "__main__":
    main()
