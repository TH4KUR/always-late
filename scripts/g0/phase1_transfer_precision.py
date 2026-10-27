"""Machinery of the paired arithmetic-precision audit (task 2 of docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md; protocol revision_R5_precision_audit.json; implementation plan
docs/superpowers/plans/2026-10-27-precision-audit-task2-implementation.md, step 3; POST HOC).

Two execution modes of the SUFFIX (the blocks above the SAE site and the final norm): the existing bf16 mode and an fp32 mode in which the SAME stored bf16-valued parameters are promoted exactly to float32 after
the weight state was set (the quantization is never redone: the order is demote the suffix, set the state with the unchanged `setup`, promote) and the configuration dtype is float32 for the duration of the suffix
call. This module has no dependency on TransformerLens: it works on any model with `.blocks` (a list of modules), `.ln_final` and `.cfg`, so that the dtype logic is testable without the library; the real-model
behaviour is checked by the dtype probe and the gates of the Ada job. The frozen runner is not changed.
"""
import hashlib
from contextlib import contextmanager

import torch

MODES = ("bf16", "fp32")


def _suffix_modules(model, layer):
    return [(f"blocks.{i}", model.blocks[i]) for i in range(layer + 1, len(model.blocks))] + [("ln_final", model.ln_final)]


def _tensors(model, layer):
    """(name, tensor) of every parameter and every floating-point buffer of the suffix blocks (above `layer`) and of the final norm, in a fixed order."""
    out = []
    for prefix, mod in _suffix_modules(model, layer):
        out += [(f"{prefix}.{n}", p) for n, p in mod.named_parameters()] + [(f"{prefix}.{n}", b) for n, b in mod.named_buffers() if b.is_floating_point()]
    return sorted(out, key=lambda x: x[0])


def snapshot(model, layer):
    """{name: dtype} of the suffix tensors as they are now (the stored precision to which `demote_suffix` returns)."""
    return {n: t.dtype for n, t in _tensors(model, layer)}


def _set(model, layer, fn):
    for prefix, mod in _suffix_modules(model, layer):
        for n, p in mod.named_parameters():
            p.data = fn(f"{prefix}.{n}", p.data)
        for n, b in list(mod.named_buffers()):
            if b.is_floating_point():
                owner = mod
                parts = n.split(".")
                for part in parts[:-1]:
                    owner = getattr(owner, part)
                owner._buffers[parts[-1]] = fn(f"{prefix}.{n}", b)


def promote_suffix(model, layer):
    """Every suffix tensor to float32, in place (exact: a lower-precision value is represented exactly in float32); the prefix is not touched."""
    _set(model, layer, lambda name, t: t.to(torch.float32))


def demote_suffix(model, layer, snap):
    """Every suffix tensor back to the dtype of the snapshot, in place (exact if the values are representable in it, which `roundtrip_equal` checks); a tensor missing from the snapshot is a KeyError."""
    missing = [n for n, _ in _tensors(model, layer) if n not in snap]
    if missing:
        raise KeyError(f"no recorded dtype for {missing[0]}")
    _set(model, layer, lambda name, t: t.to(snap[name]))


def suffix_hashes(model, layer, snap=None):
    """{name: sha256 of the bytes of the tensor}; with `snap` every tensor is first converted to its recorded dtype (so a promoted suffix hashes like the original one if the promotion was exact)."""
    out = {}
    for n, t in _tensors(model, layer):
        x = t.detach().to("cpu")
        if snap is not None:
            x = x.to(snap[n])
        out[n] = hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    return out


def roundtrip_equal(model, layer, snap, hashes):
    """Gate G1: the current (promoted) suffix converted back to the recorded dtypes has exactly the recorded hashes."""
    return suffix_hashes(model, layer, snap) == hashes


def fp32_setup(setup, model, layer, snap):
    """The state change of the fp32 mode: demote the suffix to its stored precision, run the unchanged `setup(label)` (restore and quantize exactly as in the main runs), promote again (also if `setup` fails)."""
    def wrapped(label):
        demote_suffix(model, layer, snap)
        try:
            return setup(label)
        finally:
            promote_suffix(model, layer)
    return wrapped


def _cfgs(model, layer):
    seen, out = set(), []
    for _, mod in _suffix_modules(model, layer):
        for sub in mod.modules():
            c = getattr(sub, "cfg", None)
            if c is not None and id(c) not in seen:
                seen.add(id(c)); out.append(c)
    return out


@contextmanager
def _dtype_of_cfgs(model, layer, dtype):
    cfgs = _cfgs(model, layer); old = [c.dtype for c in cfgs]
    for c in cfgs:
        c.dtype = dtype
    try:
        yield
    finally:
        for c, o in zip(cfgs, old):
            c.dtype = o


def mode_suffix(model, layer, fn, mode):
    """The suffix function of a mode: `fn` itself in the bf16 mode; in the fp32 mode `fn` called with the configuration dtype float32 of every configuration object of the suffix modules (restored afterwards, also on an
    exception). The parameters must already be promoted (`promote_suffix`)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}: expected one of {MODES}")
    if mode == "bf16":
        return fn

    def suffix(*a, **k):
        with _dtype_of_cfgs(model, layer, torch.float32):
            return fn(*a, **k)
    return suffix


def dtype_probe(model, paths, run):
    """Gate G2: {key: str(dtype)} of the tensor produced at each hooked submodule (`paths`: {key: dotted path of a submodule, e.g. a HookPoint}) during `run()`; the hooks are removed afterwards."""
    seen, handles = {}, []
    try:
        for key, path in paths.items():
            handles.append(model.get_submodule(path).register_forward_hook(lambda mod, inp, out, key=key: seen.__setitem__(key, str(out.dtype))))
        run()
    finally:
        for h in handles:
            h.remove()
    return seen


def assert_dtypes(observed, mode):
    """The fp32 mode must show float32 at every probed tensor (AssertionError naming the first that does not); the bf16 mode only records what it observes."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "fp32":
        bad = [k for k, v in observed.items() if v != "torch.float32"]
        assert not bad, f"the fp32 mode shows a lower precision at {bad[0]}: {observed[bad[0]]}"
