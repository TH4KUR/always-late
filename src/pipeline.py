"""Model loading + residual-stream hooking + SAE encoding.

Deliberately uses plain HF `transformers` (AutoModelForCausalLM +
`register_forward_hook`) rather than TransformerLens: Gemma-2's
RMSNorm-folding / soft-capping handling in TransformerLens has been a
recurring source of numerical-mismatch bugs across versions, and we don't
need TL's other features (activation patching, etc.) for this project --
only the resid_post activation stream at one fixed layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.sae_models import JumpReLUSAE


@dataclass
class ModelConfig:
    model_id: str = "unsloth/gemma-2-2b"  # bf16 mirror of google/gemma-2-2b
    layer: int = 12
    dtype: torch.dtype = torch.bfloat16
    device: str = "cpu"


class ResidualCapture:
    """Registers a forward hook on `model.model.layers[layer]` and stores
    the residual-stream output (resid_post) of the most recent forward pass."""

    def __init__(self, model: torch.nn.Module, layer: int):
        self.activation: torch.Tensor | None = None
        target = model.model.layers[layer]
        self._handle = target.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.activation = output[0] if isinstance(output, tuple) else output

    def remove(self):
        self._handle.remove()


def load_model_and_tokenizer(cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=cfg.dtype, cache_dir="data/hf_cache"
    )
    model.to(cfg.device)
    model.eval()
    return model, tokenizer


def load_sae(cfg: ModelConfig, cache_dir: str = "data/hf_cache") -> JumpReLUSAE:
    sae = JumpReLUSAE.from_gemma_scope(layer=cfg.layer, cache_dir=cache_dir)
    return sae.to(cfg.device).to(cfg.dtype)


@torch.no_grad()
def encode_texts(
    texts: list[str],
    model,
    tokenizer,
    sae: JumpReLUSAE,
    capture: ResidualCapture,
    device: str,
    max_length: int = 128,
) -> torch.Tensor:
    """Returns concatenated per-token SAE feature activations for `texts`:
    shape [total_real_tokens, d_sae], padding tokens excluded."""
    all_feats = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        model(**enc)
        resid = capture.activation  # [1, seq_len, d_model]
        feats = sae.encode(resid[0])  # [seq_len, d_sae]
        all_feats.append(feats.float().cpu())
    return torch.cat(all_feats, dim=0)
