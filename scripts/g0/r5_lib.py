"""R5 library: OUR pipeline on THEIR tokens, with factor toggles (see reports/audit_our_vs_duan.md).
Pure math/orchestration pieces with no transformers import, so they can be unit-tested on CPU (scripts/g0/tests_torch)."""
from contextlib import contextmanager

import torch

SEQ_LEN, TOKEN_BUDGET = 256, 500_000


def build_windows(token_ids, token_budget=TOKEN_BUDGET, seq_len=SEQ_LEN):
    """Their build_tokens_2d rule: first min(budget, stream) tokens, truncated to whole windows -> [n, seq_len]."""
    t = token_ids if torch.is_tensor(token_ids) else torch.tensor(list(token_ids), dtype=torch.long)
    usable = min(token_budget, t.shape[0]); n = usable // seq_len
    if n == 0:
        raise RuntimeError("not enough tokens for one window")
    return t[: n * seq_len].reshape(n, seq_len)


def apply_regime(windows, regime, bos_id):
    """none: their shipped input. mask/keep: BOS prepended to the IDENTICAL windows (rows become seq_len+1)."""
    if regime == "none":
        return windows
    if regime in ("mask", "keep"):
        return torch.cat([torch.full((windows.shape[0], 1), bos_id, dtype=torch.long), windows], dim=1)
    raise ValueError(f"unknown BOS regime {regime!r}")


def stats_start(regime):
    """First position that enters the feature statistics (mask drops exactly the BOS position)."""
    return 1 if regime == "mask" else 0


class PearsonStream:
    """Their streaming_feature_metrics math: float64 running sums, corr clamped to [-1, 1], denominator clamped at sqrt(1e-12)."""
    def __init__(self, d_sae, device):
        z = lambda: torch.zeros(d_sae, dtype=torch.float64, device=device)
        self.sx, self.sy, self.sx2, self.sy2, self.sxy, self.fire, self.maxx = z(), z(), z(), z(), z(), z(), z()
        self.n = 0

    def update(self, x, y):
        x, y = x.to(torch.float64), y.to(torch.float64)
        self.sx += x.sum(0); self.sy += y.sum(0); self.sx2 += (x ** 2).sum(0); self.sy2 += (y ** 2).sum(0); self.sxy += (x * y).sum(0)
        self.fire += (x > 0).sum(0); self.maxx = torch.maximum(self.maxx, x.max(0).values); self.n += x.shape[0]

    def result(self):
        n = self.n
        num = self.sxy - (self.sx * self.sy / n)
        dx, dy = self.sx2 - (self.sx ** 2 / n), self.sy2 - (self.sy ** 2 / n)
        den = torch.sqrt(torch.clamp(dx * dy, min=1e-12))
        return {"corr": torch.clamp(num / den, -1.0, 1.0).cpu(), "firing_rate": (self.fire / n).cpu(), "mean_activation": (self.sx / n).cpu(),
                "max_activation": self.maxx.cpu(), "n": n}


def summarize(corr, firing_rate, threshold=0.001):
    """Their summary over ACTIVE features (FP firing rate > threshold): survived r>0.9, degraded 0.5<r<=0.9, damaged r<0.5; torch (lower) median."""
    active = firing_rate > threshold
    ac = corr[active]
    pct = lambda m: float(m.double().mean().item() * 100)
    return {"active": active, "n_active": int(active.sum().item()), "mean_corr": float(ac.mean().item()), "median_corr": float(ac.median().item()),
            "survived_pct": pct(ac > 0.9), "degraded_pct": pct((ac > 0.5) & (ac <= 0.9)), "damaged_pct": pct(ac < 0.5)}


class SparseFeatureStore:
    """JumpReLU features are ~0.5% non-zero: keep the full-precision features of every batch sparsely (indices + values, CPU)."""
    def __init__(self):
        self._b = []

    def add(self, dense):
        r, c = dense.nonzero(as_tuple=True)
        self._b.append((tuple(dense.shape), dense.dtype, r.to(torch.int32).cpu(), c.to(torch.int32).cpu(), dense[r, c].cpu()))

    @property
    def n_batches(self):
        return len(self._b)

    def get(self, i, device):
        shape, dt, r, c, v = self._b[i]
        out = torch.zeros(shape, dtype=dt, device=device)
        out[r.to(device).long(), c.to(device).long()] = v.to(device)
        return out


def group_for(kind):
    """Their TransformerLens layout shares one scale per head-dim index across heads for q/k/v; every other module is per output row."""
    return "head_shared" if kind in ("q_proj", "k_proj", "v_proj") else "row"


def duan_fake_quantize(w, bits, group, head_dim=256):
    """Their RTN arithmetic (quantize_rtn_per_output_channel_tl), applied to an HF-layout [out, in] weight."""
    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))
    if group == "row":
        scale = w.abs().amax(dim=1, keepdim=True) / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.round(w / scale).clamp(qmin, qmax)
        return (q * scale).to(w.dtype)
    if group == "head_shared":
        out, inn = w.shape
        w3 = w.view(out // head_dim, head_dim, inn)
        scale = w3.abs().amax(dim=(0, 2), keepdim=True) / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.round(w3 / scale).clamp(qmin, qmax)
        return (q * scale).to(w.dtype).reshape(out, inn)
    raise ValueError(group)


@contextmanager
def quantized_model_duan(model, bits, head_dim=256):
    """Like src.rtn_quant.quantized_model (same target set, originals staged on CPU and restored) but with THEIR grouping/arithmetic."""
    from src.rtn_quant import _is_quant_target
    targets = [(n, m) for n, m in model.named_modules() if _is_quant_target(n, m)]
    saved = {}
    try:
        for name, module in targets:
            saved[name] = module.weight.data.clone().cpu()
            module.weight.data.copy_(duan_fake_quantize(module.weight.data, bits, group_for(name.split(".")[-1]), head_dim))
        yield [n for n, _ in targets]
    finally:
        for name, module in targets:
            if name in saved:
                module.weight.data.copy_(saved[name].to(module.weight.device))
