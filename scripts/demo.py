"""
Gradio inference demo for VetVision-LM.

Launches an interactive web UI for:
  - Image-text retrieval (upload an X-ray, retrieve most similar reports)
  - Zero-shot species classification (canine vs feline)
  - Attention map visualisation

Usage:
    python scripts/demo.py
    python scripts/demo.py --checkpoint checkpoints/finetune/best.pth
    python scripts/demo.py --smoke-test   # random weights, no checkpoint needed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.vetvision import VetVisionLM
from data.augmentations import build_val_transform
from utils.visualize import visualise_attention

# ---------------------------------------------------------------------------
# Corpus of sample veterinary reports (used for retrieval demo)
# ---------------------------------------------------------------------------

SAMPLE_CORPUS = [
    "Canine thoracic radiograph. Cardiac silhouette within normal limits. "
    "Lung fields are clear bilaterally. No evidence of pleural effusion.",
    "Feline chest X-ray showing mild bilateral pleural effusion. "
    "Cardiac silhouette mildly enlarged. Suspect congestive heart failure.",
    "Dog thorax: moderate cardiomegaly with pulmonary overcirculation. "
    "Consistent with patent ductus arteriosus.",
    "Cat radiograph: hyperinflation of the lung fields. "
    "Possible feline asthma. No pleural fluid.",
    "Canine patient: right-sided cardiomegaly. "
    "Possible right ventricular hypertrophy.",
    "Feline patient: pleural effusion and mediastinal widening. "
    "Differential includes lymphoma or pyothorax.",
    "Dog chest X-ray: no acute findings. Thorax within normal limits.",
    "Cat thorax: tracheal deviation noted. Mass effect in cranial mediastinum.",
    "Canine: soft tissue opacity in right caudal lung lobe — pneumonia vs mass.",
    "Feline: miliary nodular pattern throughout lung fields. "
    "Differential includes metastatic disease or fungal infection.",
]

SPECIES_MAP = {0: "Canine", 1: "Feline"}


def load_model(checkpoint_path=None, device="cpu"):
    """Load VetVisionLM with optional checkpoint."""
    from omegaconf import OmegaConf
    cfg_path = Path("configs/finetune.yaml")
    if cfg_path.exists():
        cfg = OmegaConf.load(cfg_path)
        model = VetVisionLM.from_config(cfg).to(device)
    else:
        model = VetVisionLM().to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    else:
        print("[INFO] Using randomly initialised weights (no checkpoint provided).")

    model.eval()
    return model


@torch.no_grad()
def encode_corpus(model, corpus, device):
    """Pre-compute text embeddings for the report corpus."""
    text_cls, _ = model.text_encoder(texts=corpus)
    _, text_emb = model.projector(
        torch.zeros(len(corpus), model.vision_encoder.output_dim, device=device),
        text_cls,
    )
    return F.normalize(text_emb, p=2, dim=-1)


@torch.no_grad()
def predict(
    model,
    image: Image.Image,
    transform,
    corpus_embeds: torch.Tensor,
    device: str,
    top_k: int = 3,
):
    """Run retrieval and classification for a single image."""
    img_tensor = transform(image).unsqueeze(0).to(device)

    vision_cls, patch_feats = model.vision_encoder(img_tensor)
    vision_emb, _ = model.projector(
        vision_cls,
        torch.zeros(1, model.text_encoder.output_dim, device=device),
    )
    vision_emb = F.normalize(vision_emb, p=2, dim=-1)

    # Retrieval
    sims = (vision_emb @ corpus_embeds.T).squeeze(0)   # (N,)
    top_idx = sims.argsort(descending=True)[:top_k]
    retrieved = [(SAMPLE_CORPUS[i], float(sims[i])) for i in top_idx]

    # Species classification
    species_texts = [f"A radiograph of a {v.lower()} patient." for v in SPECIES_MAP.values()]
    text_cls, _ = model.text_encoder(texts=species_texts)
    _, species_emb = model.projector(
        torch.zeros(len(species_texts), model.vision_encoder.output_dim, device=device),
        text_cls,
    )
    species_emb = F.normalize(species_emb, p=2, dim=-1)
    cls_logits = vision_emb @ species_emb.T / 0.07
    cls_probs = torch.softmax(cls_logits, dim=-1).squeeze(0)
    species_pred = int(cls_probs.argmax().item())
    species_conf = float(cls_probs.max().item())

    # Attention map
    attn = model.vision_encoder.get_attention_maps(img_tensor)

    return retrieved, species_pred, species_conf, attn, img_tensor.squeeze(0)


def build_gradio_ui(model, corpus_embeds, transform, device):
    """Build and return the Gradio Blocks UI."""
    import gradio as gr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    def run_inference(image_pil, top_k):
        if image_pil is None:
            return "Please upload an X-ray image.", "", None

        retrieved, species_pred, species_conf, attn, img_tensor = predict(
            model, image_pil, transform, corpus_embeds, device, top_k=int(top_k)
        )

        # Format retrieval results
        ret_text = "### Top Retrieved Reports\n\n"
        for rank, (report, score) in enumerate(retrieved, 1):
            ret_text += f"**{rank}. (score: {score:.4f})**\n{report}\n\n"

        # Classification result
        cls_text = (
            f"**Predicted Species:** {SPECIES_MAP[species_pred]}  \n"
            f"**Confidence:** {species_conf*100:.1f}%"
        )

        # Attention map figure
        try:
            fig = visualise_attention(img_tensor.cpu(), attn[0].cpu())
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            plt.close(fig)
            attn_img = Image.open(buf)
        except Exception:
            attn_img = None

        return ret_text, cls_text, attn_img

    with gr.Blocks(title="VetVision-LM Demo") as demo:
        gr.Markdown(
            """
            # VetVision-LM — Veterinary Radiology Vision-Language Demo
            **Author:** Devarchith Parashara Batchu

            Upload a veterinary chest X-ray to:
            1. Retrieve the most similar radiology reports from a sample corpus.
            2. Classify the patient species (canine vs feline).
            3. Visualise ViT attention maps.

            > **Note:** This demo uses randomly initialised weights unless a trained
            > checkpoint is provided. Results are not clinically valid.
            """
        )
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label="Upload X-Ray Image")
                top_k_slider = gr.Slider(1, 5, value=3, step=1, label="Top-K reports")
                run_btn = gr.Button("Analyse", variant="primary")
            with gr.Column():
                cls_output = gr.Markdown(label="Species Classification")
                ret_output = gr.Markdown(label="Retrieved Reports")
                attn_output = gr.Image(label="Attention Map", type="pil")

        run_btn.click(
            fn=run_inference,
            inputs=[img_input, top_k_slider],
            outputs=[ret_output, cls_output, attn_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="VetVision-LM Gradio Demo")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Launch demo with random weights (no checkpoint/data needed)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = build_val_transform()

    print("[INFO] Loading model...")
    model = load_model(checkpoint_path=args.checkpoint, device=device)

    print("[INFO] Encoding report corpus...")
    corpus_embeds = encode_corpus(model, SAMPLE_CORPUS, device)

    if args.smoke_test:
        print("[SMOKE] Smoke test: verifying demo pipeline with a synthetic image.")
        dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        retrieved, species_pred, species_conf, _, _ = predict(
            model, dummy, transform, corpus_embeds, device, top_k=3
        )
        print(f"[SMOKE] Species: {SPECIES_MAP[species_pred]} ({species_conf*100:.1f}%)")
        print(f"[SMOKE] Top report: {retrieved[0][0][:60]}...")
        print("[SMOKE] Demo pipeline OK.")
        return

    print(f"[INFO] Launching Gradio on port {args.port}...")
    demo = build_gradio_ui(model, corpus_embeds, transform, device)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
