# Model Card — VetVision-LM

**Model name:** VetVision-LM
**Version:** 0.1.0
**Author:** Devarchith Parashara Batchu · devarchithbatchu@gmail.com
**Repository:** https://github.com/devarchith/vetvision-lm
**Date:** 2024

---

## Model Description

VetVision-LM is a self-supervised vision-language model designed for multi-species veterinary chest radiography. It jointly encodes radiograph images and associated free-text radiology reports into a shared 512-dimensional embedding space using contrastive learning.

### Model Architecture

| Component | Specification |
|---|---|
| Vision encoder | ViT-Base/16, 224×224 input, CLS token |
| Text encoder | PubMedBERT (uncased, abstract-fulltext), CLS token |
| Species embedding matrix | E ∈ R^{2 × 64}, learnable |
| Species-Adaptive Module | MLP: Linear→LN→GELU→Linear→LN |
| Projection dimension | 512 (shared embedding space) |
| Loss | L_contrastive + 0.5 · L_species |
| Temperature | τ = 0.07 (fixed) |
| Species margin | γ = 0.2 |

---

## Intended Uses

### Primary intended use
- Research on veterinary medical imaging AI
- Multi-species radiograph representation learning
- Image-text retrieval for radiology reports
- Zero-shot species classification from radiographs

### Out-of-scope uses
- **Not validated for clinical diagnosis or treatment decisions**
- Not validated for species beyond canine and feline
- Not intended for human medical imaging (use CheXzero or similar)
- Not a replacement for board-certified veterinary radiologists

---

## Benchmarks

> **Scientific Integrity Statement:**
> All benchmark results below are **reported in the associated paper and have NOT been independently reproduced** in this repository. They are included for reference only. Reproduction requires access to the full training datasets and significant compute. The expected vs. reproduced columns clearly distinguish paper-reported from independently verified values.

### Image-Text Retrieval

| Metric | Expected (Paper) | Reproduced |
|---|---|---|
| i2t Recall@1 | 55.1% | *Not yet reproduced* |
| CheXzero baseline R@1 | 42.8% | *Not yet reproduced* |
| CLIP baseline R@1 | — | *Not yet reproduced* |

*Source: Reported in Paper (not reproduced)*

### Zero-Shot Species Classification

| Metric | Expected (Paper) | Reproduced |
|---|---|---|
| Overall Accuracy | 77.3% | *Not yet reproduced* |
| Canine Accuracy | 79.0% | *Not yet reproduced* |
| Feline Accuracy | 75.5% | *Not yet reproduced* |

*Source: Reported in Paper (not reproduced)*

### Ablation Results

| Removed Component | Δ i2t R@1 (Paper) | Reproduced |
|---|---|---|
| Species-Adaptive Module | −6.5% | *Not yet reproduced* |
| Species-Aware Loss | −3.8% | *Not yet reproduced* |
| Both species components | ~−10.3% | *Not yet reproduced* |

*Source: Reported in Paper (not reproduced)*

---

## Training Data

### Pretraining
- **Dataset:** CheXpert (Stanford AIMI)
- **Domain:** Human chest radiographs with 14 pathology labels
- **Use:** Pretrain joint vision-language representations
- **Access:** Gated — requires data-use agreement at https://stanfordaimi.azurewebsites.net/

### Fine-tuning
- **Dataset:** TUFTS Veterinary Chest X-Ray (public proxy via Kaggle: v7labs/vets-chest-xray-competition)
- **Domain:** Canine and feline thoracic radiographs
- **Access:** Requires Kaggle account

---

## Training Procedure

| Hyperparameter | Pretraining | Fine-tuning |
|---|---|---|
| Batch size | 32 | 32 |
| Learning rate | 1e-4 | 5e-5 |
| LR schedule | Cosine with warmup | Cosine with warmup |
| Warmup epochs | 5 | 3 |
| Total epochs | 50 | 30 |
| Weight decay | 1e-4 | 1e-4 |
| Gradient clipping | 1.0 | 1.0 |
| Mixed precision | FP16 | FP16 |
| Optimiser | AdamW | AdamW |

---

## Evaluation

### Metrics used
- **Retrieval:** Recall@1, Recall@5, Recall@10, Mean Reciprocal Rank (MRR)
- **Classification:** Accuracy, AUC-ROC, F1 (macro, weighted, per-species)
- **Ablation:** Systematic component removal study

---

## Limitations

1. **Data limitations:** Trained on a limited veterinary radiograph dataset. Generalisation to rare breeds, exotic species, or unusual pathologies is not guaranteed.

2. **Species coverage:** Currently supports only canine (dog) and feline (cat) species. The species-adaptive module would need extension for additional species.

3. **Domain shift:** Pretraining on human CheXpert introduces a domain mismatch. Performance may degrade compared to a model pretrained on veterinary data from scratch.

4. **Report quality:** Synthetic report text (derived from pathology labels) is used during pretraining. Fine-tuning benefits from real free-text reports when available.

5. **Clinical validation:** This model has **not been validated for clinical use**. Do not use for diagnosis, treatment, or clinical decision support.

6. **Compute requirements:** Full training requires at least one GPU with 16 GB VRAM. Results may vary with different hardware.

7. **Unreproduced results:** Paper-reported benchmarks reflect results from the original research environment. Independent reproduction may yield different numbers due to dataset splits, random seeds, or implementation differences.

---

## Ethical Considerations

- **No patient data is included** in this repository. All data must be obtained through official channels with appropriate agreements.
- Training data (CheXpert) contains de-identified human radiographs under a research data-use agreement.
- This model must not be used to make or influence veterinary clinical decisions without validation by a licensed veterinary radiologist.
- Bias analysis across breeds, body conditions, and imaging equipment has not been performed.

---

## How to Use

```python
import torch
from omegaconf import OmegaConf
from src.models.vetvision import VetVisionLM
from src.data.augmentations import build_val_transform
from PIL import Image

# Load model
cfg = OmegaConf.load("configs/finetune.yaml")
model = VetVisionLM.from_config(cfg)
ckpt = torch.load("checkpoints/finetune/best.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

# Encode an image
transform = build_val_transform()
img = Image.open("path/to/xray.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    out = model(images=img_tensor, texts=["A chest radiograph of a canine patient."])
    print("Vision embed:", out.vision_embed.shape)  # (1, 512)
    print("Text embed:  ", out.text_embed.shape)    # (1, 512)
```

---

## Citation

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

MIT License. See [LICENSE](LICENSE) for full text.

**Important:** Data accessed through Stanford AIMI (CheXpert) and Kaggle are subject to their respective data-use agreements and must not be redistributed.
