# VetVision-LM

[![CI — Smoke Tests](https://github.com/devarchith/vetvision-lm/actions/workflows/ci.yml/badge.svg)](https://github.com/devarchith/vetvision-lm/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<!-- Live CI badge screenshot (captured from passing run) -->
[![CI — Smoke Tests passing](assets/ci-passing-badge.png)](https://github.com/devarchith/vetvision-lm/actions/workflows/ci.yml)

**VetVision-LM** — *Self-Supervised Vision-Language Representation Learning for Multi-Species Veterinary Radiology*

A production-grade research implementation of a contrastive vision-language model tailored for veterinary chest radiography, supporting multi-species (canine / feline) representation learning via a learnable species-adaptive module.

**Author:** Devarchith Parashara Batchu · devarchithbatchu@gmail.com

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Demo](#demo)
- [Smoke Test Mode](#smoke-test-mode)
- [Citation](#citation)

---

## Overview

VetVision-LM learns joint image–text representations from veterinary radiographs and associated radiology reports using a **CLIP-style contrastive objective** augmented with a **species-aware contrastive loss**.

Key contributions:
- **Species-Adaptive Module** — learnable species embeddings condition image representations on patient species (canine / feline), yielding species-specific feature spaces.
- **Dual contrastive loss** — combines standard InfoNCE with an intra-species InfoNCE + cross-species triplet margin loss.
- **Pretraining → Fine-tuning pipeline** — pretrained on CheXpert human chest X-rays, fine-tuned on veterinary radiographs.

---

## Architecture

```
Input Image (224×224)
       │
  ┌────▼──────────────────┐
  │   Vision Encoder      │  ViT-Base/16 → CLS token (768-d)
  └────┬──────────────────┘
       │ vision_cls
  ┌────▼──────────────────┐
  │   Vision Projection   │  Linear(768→512) → GELU → Linear(512→512) → L2-norm
  └────┬──────────────────┘
       │ vision_embed (512-d)
  ┌────▼──────────────────┐
  │ Species-Adaptive Mod. │  h' = MLP([h ; e_s])
  │  E ∈ R^{S × d_s}     │  Linear→LN→GELU→Linear→LN
  └────┬──────────────────┘
       │ species_embed (512-d)
       │
       │           Input Text (radiology report)
       │                  │
       │        ┌─────────▼─────────┐
       │        │   Text Encoder    │  PubMedBERT → [CLS] (768-d)
       │        └─────────┬─────────┘
       │                  │ text_cls
       │        ┌─────────▼─────────┐
       │        │  Text Projection  │  Linear(768→512)→GELU→Linear→L2-norm
       │        └─────────┬─────────┘
       │                  │ text_embed (512-d)
       │                  │
  ─────┴──────────────────┴─────────
  L = L_contrastive(v,t) + λ · L_species(s,t)
  τ = 0.07,  λ = 0.5,  γ = 0.2
```

### Component Details

| Component | Architecture | Output Dim |
|---|---|---|
| Vision Encoder | ViT-Base/16, pretrained ImageNet | 768 |
| Text Encoder | PubMedBERT (uncased, abstract-fulltext) | 768 |
| Species Embedding | Learnable E ∈ R^{2 × 64} | 64 |
| Species Module MLP | Linear→LN→GELU→Linear→LN | 512 |
| Projection (vision) | Linear(768→512)→GELU→Linear(512→512) | 512 |
| Projection (text) | Linear(768→512)→GELU→Linear(512→512) | 512 |
| Shared Embedding | L2-normalised, dim = 512 | 512 |

### Loss Function

```
L = L_contrastive + λ · L_species

L_contrastive = (1/2)[CE(sim/τ, I) + CE(simᵀ/τ, I)]
              (symmetric InfoNCE, CLIP-style)

L_species     = L_intra-InfoNCE + L_triplet
L_intra-InfoNCE  = per-species symmetric InfoNCE
L_triplet        = max(0, neg_sim - pos_sim + γ)

Hyperparameters: τ=0.07, λ=0.5, γ=0.2
```

---

## Results

> **Scientific Integrity Note:**
> All numbers in this table are **reported in the paper and have NOT been reproduced**.
> Reproduced results will be added after training on the full datasets.

### Image-Text Retrieval (Recall@K)

| Model | i2t R@1 | i2t R@5 | i2t R@10 |
|---|---|---|---|
| CheXzero (baseline) | 42.8% | — | — |
| CLIP (baseline) | — | — | — |
| **VetVision-LM (ours)** | **55.1%** | — | — |

*Source: Reported in Paper (not reproduced)*

### Zero-Shot Species Classification

| Model | Overall Acc. | Canine Acc. | Feline Acc. |
|---|---|---|---|
| **VetVision-LM** | **77.3%** | **79.0%** | **75.5%** |

*Source: Reported in Paper (not reproduced)*

### Ablation Study

| Configuration | i2t R@1 | Δ vs Full |
|---|---|---|
| Full VetVision-LM | 55.1% | — |
| − Species-Adaptive Module | 48.6% | −6.5% |
| − Species-Aware Loss | 51.3% | −3.8% |
| − Both Species Components | ~44.8% | −10.3% |

*Source: Reported in Paper (not reproduced)*

---

## Repository Structure

```
vetvision-lm/
├── src/
│   ├── data/
│   │   ├── chexpert.py          # CheXpert loader + downloader
│   │   ├── veterinary.py        # Vet dataset loader + manifest generator
│   │   └── augmentations.py     # Radiograph augmentation pipeline
│   ├── models/
│   │   ├── vision_encoder.py    # ViT-Base/16 wrapper
│   │   ├── text_encoder.py      # PubMedBERT wrapper
│   │   ├── species_module.py    # Species-Adaptive Module
│   │   ├── projection.py        # Projection heads
│   │   └── vetvision.py         # Full VetVision-LM model
│   ├── losses/
│   │   ├── contrastive.py       # CLIP-style symmetric InfoNCE
│   │   └── species_loss.py      # Species-aware loss + CombinedLoss
│   ├── train/
│   │   ├── pretrain.py          # CheXpert pretraining loop
│   │   └── finetune.py          # Veterinary fine-tuning loop
│   ├── eval/
│   │   ├── retrieval.py         # R@1, R@5, R@10, MRR
│   │   ├── classification.py    # Zero-shot classification
│   │   ├── ablation.py          # Ablation study runner
│   │   └── baselines.py         # CLIP, BiomedCLIP, ResNet-50
│   └── utils/
│       ├── visualize.py         # t-SNE + attention maps
│       ├── metrics.py           # Shared metric utilities
│       └── logger.py            # Logging setup
├── configs/
│   ├── pretrain.yaml
│   ├── finetune.yaml
│   └── eval.yaml
├── scripts/
│   ├── pretrain.py              # CLI: python scripts/pretrain.py
│   ├── finetune.py              # CLI: python scripts/finetune.py
│   ├── evaluate.py              # CLI: python scripts/evaluate.py
│   ├── demo.py                  # Gradio demo
│   ├── generate_manifest.py     # Auto-generate vet dataset CSV
│   ├── download_chexpert.sh     # CheXpert download instructions
│   └── download_vet_dataset.sh  # Kaggle vet dataset download
├── notebooks/
│   ├── exploration.ipynb
│   └── results_visualization.ipynb
├── tests/
│   ├── test_model.py
│   ├── test_losses.py
│   └── test_data.py
├── .github/workflows/ci.yml    # GitHub Actions CI
├── Dockerfile
├── requirements.txt
├── setup.py
├── README.md
└── MODEL_CARD.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/devarchith/vetvision-lm.git
cd vetvision-lm
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

### 2. Smoke test — verify pipeline (no GPU, no data needed)

```bash
python scripts/pretrain.py --smoke-test
python scripts/finetune.py --smoke-test
python scripts/evaluate.py --smoke-test
```

### 3. Run tests

```bash
pytest tests/ -v
```

---

## Datasets

### CheXpert (pretraining)

CheXpert requires a data-use agreement. Run the download helper:

```bash
bash scripts/download_chexpert.sh data/
```

Follow the printed instructions to request access at the Stanford AIMI portal.

### TUFTS Veterinary Chest X-Ray (fine-tuning)

Requires Kaggle API credentials (`~/.kaggle/kaggle.json`):

```bash
bash scripts/download_vet_dataset.sh data/
python scripts/generate_manifest.py \
    --data-dir data/vet_dataset \
    --output data/vet_dataset/manifest.csv
```

---

## Training

### Pretraining on CheXpert

```bash
python scripts/pretrain.py --config configs/pretrain.yaml
```

### Fine-tuning on veterinary data

```bash
python scripts/finetune.py --config configs/finetune.yaml
```

### Training hyperparameters

| Hyperparameter | Pretrain | Finetune |
|---|---|---|
| Batch size | 32 | 32 |
| Learning rate | 1e-4 | 5e-5 |
| Epochs | 50 | 30 |
| LR schedule | Cosine | Cosine |
| Warmup epochs | 5 | 3 |
| Weight decay | 1e-4 | 1e-4 |
| Temperature τ | 0.07 | 0.07 |
| λ (species loss) | 0.5 | 0.5 |
| Margin γ | 0.2 | 0.2 |

---

## Evaluation

```bash
# Full evaluation suite
python scripts/evaluate.py \
    --config configs/eval.yaml \
    --checkpoint checkpoints/finetune/best.pth

# Individual modes
python scripts/evaluate.py --mode retrieval --checkpoint path/to/best.pth
python scripts/evaluate.py --mode classification --checkpoint path/to/best.pth
python scripts/evaluate.py --mode ablation --checkpoint path/to/best.pth
python scripts/evaluate.py --mode baselines --checkpoint path/to/best.pth
```

Results are saved to `results/evaluation_report.json`.

---

## Demo

```bash
python scripts/demo.py --checkpoint checkpoints/finetune/best.pth
# Opens Gradio UI at http://localhost:7860
```

---

## Smoke Test Mode

All training and evaluation scripts support `--smoke-test`:

```bash
python scripts/pretrain.py --smoke-test   # 50 synthetic samples, 2 epochs, CPU
python scripts/finetune.py --smoke-test
python scripts/evaluate.py --smoke-test
python scripts/demo.py --smoke-test
```

This verifies the full pipeline runs without real data or GPU.

---

## Docker

```bash
docker build -t vetvision-lm .
docker run --rm vetvision-lm python scripts/pretrain.py --smoke-test
docker run --rm -p 7860:7860 vetvision-lm python scripts/demo.py
```

---

## Citation

If you use this code, please cite:

```bibtex
@misc{batchu2024vetvision,
  title   = {VetVision-LM: Self-Supervised Vision-Language Representation
             Learning for Multi-Species Veterinary Radiology},
  author  = {Batchu, Devarchith Parashara},
  year    = {2024},
  url     = {https://github.com/devarchith/vetvision-lm},
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
