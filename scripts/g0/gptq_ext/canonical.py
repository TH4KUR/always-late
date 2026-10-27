"""TransformerLens parameters <-> canonical [out, in] linear matrices (plan Task 2; audit section 3). GPTQ and the matched RTN treat every output channel of a projection separately, so each target parameter is
viewed as the matrix of the corresponding linear layer: Q/K/V [h, m, d] -> [h*d, m] (row = head-major, then head coordinate), O [h, d, m] -> [m, h*d], MLP W_in / W_gate [m, f] -> [f, m] and W_out [f, m] -> [m, f]
(TransformerLens 3.9.0 `convert_gemma_weights`: `W_Q = rearrange(q_proj.weight, "(n h) m -> n m h")`, `W_O = rearrange(o_proj.weight, "m (n h) -> n h m")`, the MLP weights are the transposed Linear weights).
Grouped-query attention stores the key and value projections as `_W_K` and `_W_V` with the n_key_value_heads heads; they are used as stored, never through the expanded `W_K` / `W_V` views.
The target set is the legacy filter of the project (substring test on W_Q, W_K, W_V, W_O, W_in, W_gate, W_out, third_party/duan2026 `target_tl_weight`)."""
import torch

TARGET_NAMES = ("W_Q", "W_K", "W_V", "W_O", "W_in", "W_gate", "W_out")
_KIND = {"W_Q": "qkv", "W_K": "qkv", "W_V": "qkv", "W_O": "o", "W_in": "mlp", "W_gate": "mlp", "W_out": "mlp"}


def is_target(name):
    """The legacy filter: any target name occurs in the parameter name."""
    return any(t in name for t in TARGET_NAMES)


def kind(name):
    """'qkv', 'o' or 'mlp' from the last component of the parameter name (a leading underscore, as in `_W_K`, is ignored); ValueError for anything else."""
    leaf = name.split(".")[-1].lstrip("_")
    if leaf not in _KIND:
        raise ValueError(f"{name!r} is not a target projection")
    return _KIND[leaf]


def to_linear(name, w):
    """The canonical [out, in] matrix (a contiguous copy in the same dtype) of the TransformerLens tensor `w`."""
    k = kind(name)
    if k == "qkv":
        if w.ndim != 3:
            raise ValueError(f"{name}: expected [heads, d_model, d_head], got {tuple(w.shape)}")
        h, m, d = w.shape
        return w.permute(0, 2, 1).reshape(h * d, m).contiguous()
    if k == "o":
        if w.ndim != 3:
            raise ValueError(f"{name}: expected [heads, d_head, d_model], got {tuple(w.shape)}")
        h, d, m = w.shape
        return w.permute(2, 0, 1).reshape(m, h * d).contiguous()
    if w.ndim != 2:
        raise ValueError(f"{name}: expected [in, out], got {tuple(w.shape)}")
    return w.t().contiguous()


def from_linear(name, lin, like):
    """The inverse of `to_linear`: the TransformerLens-layout tensor with the shape, dtype and device of `like`, holding the values of the canonical matrix `lin`."""
    k = kind(name)
    if k == "qkv":
        h, m, d = like.shape
        want = (h * d, m)
    elif k == "o":
        h, d, m = like.shape
        want = (m, h * d)
    else:
        want = (like.shape[1], like.shape[0])
    if tuple(lin.shape) != want:
        raise ValueError(f"{name}: canonical matrix {tuple(lin.shape)} does not fit {tuple(like.shape)} (expected {want})")
    if k == "qkv":
        out = lin.reshape(h, d, m).permute(0, 2, 1)
    elif k == "o":
        out = lin.reshape(m, h, d).permute(1, 2, 0)
    else:
        out = lin.t()
    return out.contiguous().to(device=like.device, dtype=like.dtype)


def targets(model):
    """[(name, parameter)] of every target projection in `named_parameters()` order (the order of the legacy quantizer); every name must have a known kind."""
    out = [(n, p) for n, p in model.named_parameters() if is_target(n)]
    for n, _ in out:
        kind(n)
    return out
