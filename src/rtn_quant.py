"""Per-channel round-to-nearest (RTN) fake quantization.

Implemented from scratch (not bitsandbytes/GPTQ) to match Duan (2026)'s
protocol exactly: bitsandbytes' NF4 is a non-uniform, non-linear codebook
(not INT quantization), and GPTQ requires a different calibration procedure.
Plain per-channel symmetric RTN is what the pinned protocol (Phase 0 gate,
Section 4.2 of anlp_revised.txt) specifies.

Modules quantized: attention and MLP/gate `nn.Linear` projections.
Modules excluded: layernorms, embeddings, and the (often tied) lm_head --
matching the module list in Duan (2026) / anlp_revised.txt Section 5.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def rtn_fake_quantize(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-output-channel RTN fake-quantize/dequantize.

    `weight` has shape [out_features, in_features] (standard nn.Linear
    layout); quantization is per row (per output channel).
    """
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1
    max_abs = weight.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / qmax
    q = torch.clamp(torch.round(weight / scale), qmin, qmax)
    return q * scale


def _is_quant_target(name: str, module: torch.nn.Module) -> bool:
    if not isinstance(module, torch.nn.Linear):
        return False
    lname = name.lower()
    if "norm" in lname or "embed" in lname or "lm_head" in lname:
        return False
    return any(p in lname for p in ("proj", "gate", "up", "down", "fc"))


@contextmanager
def quantized_model(model: torch.nn.Module, bits: int):
    """Context manager: quantize target Linear weights in-place for `bits`,
    restoring the original FP weights on exit.

    Original weights are staged on CPU (not cloned on-device): cloning all
    target tensors on GPU would nearly double resident weight memory (the
    quantized Linear projections are ~90% of a transformer's parameters),
    which OOM'd on an 8GB card holding the ~5.2GB bf16 model. The CPU<->GPU
    transfer cost is small next to a forward pass and paid once per entry/exit,
    not per token.

    Usage:
        with quantized_model(model, bits=7):
            out = model(**inputs)
    """
    saved: dict[str, torch.Tensor] = {}
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if _is_quant_target(name, module)
    ]
    try:
        for name, module in targets:
            saved[name] = module.weight.data.clone().cpu()
            module.weight.data.copy_(rtn_fake_quantize(module.weight.data, bits))
        yield [name for name, _ in targets]
    finally:
        for name, module in targets:
            if name in saved:
                module.weight.data.copy_(saved[name].to(module.weight.device))
