"""Minimal, dependency-free SAE loaders.

We deliberately avoid `sae_lens` / EleutherAI's `sae` package: both pull in
extra (and version-volatile) dependencies, and their loading APIs have
churned across releases. Each SAE architecture used in this project is a
few dozen lines of plain PyTorch, so we implement them directly against the
raw weight files released by each source. This also makes the "active
feature" definition fully explicit and auditable (see JumpReLUSAE.encode),
which matters for exactly reproducing Duan (2026)'s firing-rate protocol.
"""

from __future__ import annotations

import numpy as np
import torch
from huggingface_hub import hf_hub_download


class JumpReLUSAE(torch.nn.Module):
    """JumpReLU SAE, matching Gemma Scope's released format (Lieberum et al., 2024).

    A feature is "active" iff its pre-activation exceeds a learned,
    per-feature threshold: mask = (pre_acts > threshold). This is NOT the
    same as ReLU(x) > 0 -- using that definition instead would silently
    mismatch Duan (2026)'s reported active-feature counts.
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))

    @torch.no_grad()
    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        pre_acts = acts.to(self.W_enc.dtype) @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        return mask * torch.nn.functional.relu(pre_acts)

    @torch.no_grad()
    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        return feats @ self.W_dec + self.b_dec

    @classmethod
    def from_gemma_scope(
        cls,
        repo_id: str = "google/gemma-scope-2b-pt-res",
        layer: int = 12,
        width: str = "16k",
        l0: str = "average_l0_82",
        cache_dir: str | None = None,
    ) -> "JumpReLUSAE":
        filename = f"layer_{layer}/width_{width}/{l0}/params.npz"
        path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        raw = np.load(path)
        d_model, d_sae = raw["W_enc"].shape
        sae = cls(d_model=d_model, d_sae=d_sae)
        with torch.no_grad():
            sae.W_enc.copy_(torch.from_numpy(raw["W_enc"]))
            sae.W_dec.copy_(torch.from_numpy(raw["W_dec"]))
            sae.b_enc.copy_(torch.from_numpy(raw["b_enc"]))
            sae.b_dec.copy_(torch.from_numpy(raw["b_dec"]))
            sae.threshold.copy_(torch.from_numpy(raw["threshold"]))
        sae.eval()
        return sae


class TopKSAE(torch.nn.Module):
    """TopK SAE, matching the EleutherAI `sae` / seonglae FaithfulSAE release format.

    A feature is "active" iff it survives the top-k gate on a given token
    (i.e. its pre-activation is among the k largest for that token AND
    positive). Stretch-only in this project (Llama-3.2-1B extension); not
    required for the guaranteed core.
    """

    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_sae, d_model))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))
        self.k = k

    @torch.no_grad()
    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        pre_acts = torch.nn.functional.relu(
            (acts - self.b_dec).to(self.W_enc.dtype) @ self.W_enc + self.b_enc
        )
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        out = torch.zeros_like(pre_acts)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    @torch.no_grad()
    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        return feats @ self.W_dec + self.b_dec
