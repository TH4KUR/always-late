"""Quantization grids of the GPTQ extension (plan Task 3; audit sections 3 and 4). `rtn_matched` is the comparator of GPTQ: the authors' symmetric per-output-channel grid (their `Quantizer`, `sym=True`, no MSE search),
FP32 arithmetic, nearest rounding, no Hessian compensation, the result in the weight's own dtype. `legacy_scale` and `legacy_rtn` reproduce, per tensor, the project's legacy quantizer (third_party/duan2026
`quantize_rtn_per_output_channel_tl`: scale = max|w| / (2^(b-1) - 1), reduced over every dimension except the last of the TransformerLens layout, bf16 arithmetic) and exist to show how the grids differ; the legacy
experiments themselves keep using the vendored function."""
from functools import lru_cache

import torch

from . import upstream


@lru_cache(maxsize=1)
def _core():
    return upstream.load()


def authors_grid(w, bits):
    """(scale, zero), both float32 [rows, 1], of the authors' symmetric per-channel grid for the canonical [out, in] matrix `w`: levels scale * (k - zero), k = 0 .. 2^bits - 1, zero = 2^(bits-1)."""
    core = _core()
    q = core.Quantizer(); q.configure(bits, perchannel=True, sym=True, mse=False)
    q.find_params(w.detach().float(), weight=True)
    return q.scale, q.zero


def rtn_matched(w, bits, dead=None):
    """Nearest rounding on the authors' grid, computed in float32 and returned in `w.dtype`. `dead` (bool [in]) marks input columns without calibration signal: they are set to 0 after the scale is found, exactly as
    the unmodified upstream GPTQ does (`W[:, dead] = 0`), so that the two arms differ only by the error compensation."""
    core = _core()
    wf = w.detach().float()
    scale, zero = authors_grid(w, bits)
    if dead is not None:
        wf = wf.clone(); wf[:, dead] = 0
    return core.quantize(wf, scale, zero, torch.tensor(2 ** bits - 1)).to(w.dtype)


def legacy_scale(w, bits):
    """The legacy scale of a tensor: max|w| / (2^(bits-1) - 1) over every dimension except the last (a 1-D tensor: one scale)."""
    qmax = 2 ** (bits - 1) - 1
    return w.abs().max() / qmax if w.ndim == 1 else w.abs().amax(dim=tuple(range(w.ndim - 1)), keepdim=True) / qmax


def legacy_rtn(w, bits):
    """The legacy quantizer applied to one tensor (arithmetic in the tensor's own dtype, as upstream's function does)."""
    qmax, qmin = 2 ** (bits - 1) - 1, -(2 ** (bits - 1))
    scale = legacy_scale(w, bits)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (torch.round(w / scale).clamp(qmin, qmax) * scale).to(w.dtype)
