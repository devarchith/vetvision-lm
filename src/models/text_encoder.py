"""
Text Encoder for VetVision-LM.

Wraps PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
and exposes the [CLS] token as the global text representation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

try:
    from transformers import AutoModel, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


class TextEncoder(nn.Module):
    """
    PubMedBERT text encoder.

    Tokenises input strings on-the-fly and returns the [CLS] token
    embedding as the global sentence representation.

    Args:
        model_name:     HuggingFace model identifier.
        pretrained:     Load pretrained weights.
        max_length:     Token sequence length (pad / truncate).
        freeze_layers:  Number of BERT layers to freeze (0 = none, 12 = all).
        device:         Torch device for tokenisation.
    """

    _EMBED_DIMS = {
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": 768,
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract": 768,
        "bert-base-uncased": 768,
    }

    def __init__(
        self,
        model_name: str = (
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        ),
        pretrained: bool = True,
        max_length: int = 128,
        freeze_layers: int = 0,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )

        self.model_name = model_name
        self.max_length = max_length
        self.embed_dim: int = self._EMBED_DIMS.get(model_name, 768)
        self._device = device

        if pretrained:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = AutoModel.from_config(config)

        if freeze_layers > 0:
            self._freeze_blocks(freeze_layers)

    # ------------------------------------------------------------------

    def _freeze_blocks(self, n: int) -> None:
        """Freeze embeddings and first *n* encoder layers."""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    # ------------------------------------------------------------------

    def tokenize(
        self, texts: Union[str, List[str]], device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenise a string or list of strings.

        Returns a dict of tensors ready for ``self.bert(**encoding)``.
        """
        if isinstance(texts, str):
            texts = [texts]
        encoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if device is not None:
            encoding = {k: v.to(device) for k, v in encoding.items()}
        return encoding

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text to a fixed-size representation.

        Accepts either pre-tokenised tensors (``input_ids``, etc.) or raw
        ``texts`` strings (tokenised internally).

        Returns:
            cls_feat:      (B, embed_dim) CLS-token representation.
            token_feats:   (B, seq_len, embed_dim) all token representations.
        """
        if texts is not None:
            device = next(self.parameters()).device
            encoding = self.tokenize(texts, device=device)
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            token_type_ids = encoding.get("token_type_ids")

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # last_hidden_state: (B, seq_len, D)
        token_feats = outputs.last_hidden_state
        cls_feat = token_feats[:, 0, :]      # [CLS] token

        return cls_feat, token_feats
