"""The sequential calibrator of the GPTQ extension (plan Task 5; audit section 6). It walks through a TransformerLens Gemma-2 model block by block and, inside a block, stage by stage: (1) Q, K and V together (input: the OUTPUT
of the `ln1` module, i.e. after the norm's learned scale), (2) O (input: `attn.hook_z`, heads flattened, computed with the already quantized Q/K/V), (3) input and gate together (input: the output of `ln2`), (4) the MLP output
(input: `mlp.hook_post`, computed with the quantized input and gate matrices). After every stage the block is run again on the stored residuals, so each stage sees the inputs produced by the already quantized earlier stages;
after the last stage the quantized block produces the residuals of the next block. One Hessian per stage input, accumulated with the pinned authors' `add_batch` (one call per calibration batch over its non-padding rows;
the real BOS row is included), shared by the matrices of the stage. The per-matrix rule is `gptq` (the authors' `fasterquant`, unmodified) or `rtn_matched` (the same grid, nearest rounding; dead input columns zeroed with the same
mask as GPTQ); `identity` returns the original weights (a control: every stage input must then equal the original model's activation). The model is left exactly as it was found; the artifact is returned as {parameter name:
tensor in the parameter's own TransformerLens shape and dtype}. A Hessian that is not positive definite raises `HessianError` naming the matrix (no silent retry: upstream has none). The report gives, per matrix, the share of the Hessian trace that comes from the BOS rows
(position 0 of every sequence), so that the BOS question (massive activations at the attention sink) is decided on a number."""
import contextlib
import io
import re
import time
from dataclasses import asdict, dataclass

import torch

from . import canonical, grids, upstream

METHODS = ("gptq", "rtn_matched", "identity")


class HessianError(RuntimeError):
    """The damped Hessian of a matrix is not positive definite (or not finite)."""


@dataclass(frozen=True)
class GptqConfig:
    """The primary setting of the audit (section 4): symmetric per-output-channel grid, blocksize 128, damping 1 % of the mean Hessian diagonal, no activation ordering, no input grouping, no clipping search."""
    bits: int
    blocksize: int = 128
    percdamp: float = 0.01
    actorder: bool = False
    groupsize: int = -1
    sym: bool = True
    mse: bool = False


def valid_rows(x, mask):
    """The rows of `x` [B, L, ...] at the positions where `mask` [B, L] is True (everything except padding): [N, ...]."""
    return x[mask]


class HessianAccumulator:
    """The authors' running Hessian (`GPTQ.add_batch`): every call counts as one sample, H = (2 / K) * sum_k X_k^T X_k over the K calls (a scalar multiple of the plain sum; the algorithm is invariant to it)."""
    def __init__(self, d, device):
        self.gp = upstream.load().GPTQ(torch.nn.Linear(d, 1, bias=False, device=device))
        self.calls = 0

    def add(self, rows):
        self.gp.add_batch(rows, None); self.calls += 1

    @property
    def H(self):
        return self.gp.H


def quantize_matrix(w, h, cfg, method, name=""):
    """-> (quantized matrix in w's dtype and device, info) for the canonical [out, in] matrix `w` and the Hessian `h` [in, in] (not modified)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    dead = torch.diag(h) == 0
    info = {"dead": int(dead.sum())}
    if method == "identity":
        return w.clone(), info
    if method == "rtn_matched":
        if cfg.groupsize != -1 or cfg.actorder or cfg.mse or not cfg.sym:
            raise ValueError("the matched RTN exists for the primary setting only (symmetric, per output channel, no grouping, no ordering, no clipping search)")
        return grids.rtn_matched(w, cfg.bits, dead=dead if bool(dead.any()) else None), info
    core = upstream.load()
    layer = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device=h.device)
    layer.weight.data = w.detach().to(h.device).clone()
    gp = core.GPTQ(layer); gp.quantizer = core.Quantizer(); gp.quantizer.configure(cfg.bits, perchannel=True, sym=cfg.sym, mse=cfg.mse)
    gp.H = h.clone()
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            gp.fasterquant(blocksize=cfg.blocksize, percdamp=cfg.percdamp, groupsize=cfg.groupsize, actorder=cfg.actorder)
    except torch.linalg.LinAlgError as e:
        raise HessianError(f"{name or 'matrix'}: the damped Hessian is not positive definite (dead columns {info['dead']}, mean diagonal {float(torch.diag(h).mean()):.4g}): {e}") from e
    m = re.search(r"error ([0-9.eE+-]+|nan|inf)", out.getvalue())
    if m:
        info["error"] = float(m.group(1))
    return layer.weight.data.to(w.device), info


STAGES = (("qkv", ("attn.W_Q", "attn._W_K", "attn._W_V")), ("o", ("attn.W_O",)), ("mlp_in", ("mlp.W_in", "mlp.W_gate")), ("mlp_out", ("mlp.W_out",)))


@contextlib.contextmanager
def _capture(block, stage, store):
    """Record the input of the matrices of `stage` in `store['x']` (first call wins) while the block runs."""
    handles, hp = [], None
    def keep(x):
        store.setdefault("x", x.detach())
    if stage == "qkv":
        handles.append(block.ln1.register_forward_hook(lambda m, i, o: keep(o)))
    elif stage == "mlp_in":
        handles.append(block.ln2.register_forward_hook(lambda m, i, o: keep(o)))
    elif stage == "o":
        hp = block.attn.hook_z; hp.add_hook(lambda t, hook: keep(t.reshape(*t.shape[:2], -1)))
    elif stage == "mlp_out":
        hp = block.mlp.hook_post; hp.add_hook(lambda t, hook: keep(t))
    else:
        raise ValueError(stage)
    try:
        yield
    finally:
        for h in handles:
            h.remove()
        if hp is not None:
            hp.remove_hooks()


def calibrate(model, batches, cfg, method, device, log=print, observer=None):
    """-> (tensors, report). `batches` come from `calibration.pool_batches` (tokens [B, L], mask [B, L], padding excluded from every Hessian). `observer(block, stage, name, rows)` (tests) receives the stage input rows."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    targets = canonical.targets(model); params = dict(targets)
    originals = {n: p.data.detach().clone() for n, p in targets}
    report = {"method": method, "bits": cfg.bits, "config": asdict(cfg), "n_batches": len(batches), "n_rows": sum(int(b["mask"].sum()) for b in batches), "targets": {}}
    t0 = time.time()
    try:
        with torch.no_grad():
            resid = [model(b["tokens"].to(device), stop_at_layer=0).detach().cpu() for b in batches]
            for l, block in enumerate(model.blocks):
                tb = time.time()
                for stage, leaves in STAGES:
                    names = [f"blocks.{l}.{leaf}" for leaf in leaves]
                    acc, kept, bos_e, all_e = None, [], 0.0, 0.0
                    for r, b in zip(resid, batches):
                        store = {}
                        with _capture(block, stage, store):
                            block(r.to(device))
                        rows = valid_rows(store["x"], b["mask"].to(device))
                        bos_e += float((store["x"][:, 0].float() ** 2).sum()); all_e += float((rows.float() ** 2).sum())      # the trace of X^T X is the sum of the squared row norms; position 0 is the BOS of every sequence
                        if acc is None:
                            acc = HessianAccumulator(rows.shape[1], device)
                        acc.add(rows)
                        if observer is not None:
                            kept.append(rows.float().cpu())
                    ts = time.time()
                    for name in names:
                        p = params[name]
                        q, info = quantize_matrix(canonical.to_linear(name, p.data), acc.H, cfg, method, name)
                        p.data.copy_(canonical.from_linear(name, q, p.data))
                        report["targets"][name] = {**info, "block": l, "stage": stage, "rows": report["n_rows"], "seconds": time.time() - ts, "bos_trace_share": bos_e / all_e if all_e else 0.0}
                        if observer is not None:
                            observer(l, stage, name, torch.cat(kept))
                    del acc, kept
                resid = [block(r.to(device)).detach().cpu() for r in resid]
                log(f"block {l + 1}/{len(model.blocks)} done in {time.time() - tb:.1f}s")
            tensors = {n: p.data.detach().cpu().clone() for n, p in targets}
    finally:
        with torch.no_grad():
            for n, p in targets:
                p.data.copy_(originals[n])
    report["dead_total"] = sum(v["dead"] for v in report["targets"].values())
    shares = {n: v["bos_trace_share"] for n, v in report["targets"].items()}
    report["bos_trace_share"] = {"max": max(shares.values()), "mean": sum(shares.values()) / len(shares), "max_target": max(shares, key=shares.get)}
    report["seconds"] = time.time() - t0
    report["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return tensors, report
