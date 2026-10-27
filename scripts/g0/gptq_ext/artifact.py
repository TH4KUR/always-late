"""Artifact identity and application (plan Task 6; audit sections 5 and 8). An artifact is {target parameter name: tensor in the parameter's own TransformerLens shape and dtype}: the dense bf16 dequantized weights
of one (method, bit width). Its identity is a sha256 over the sorted names, dtypes, shapes and raw bytes; every job that uses an artifact rebuilds it and refuses to continue unless the hash equals the frozen value.
Applying an artifact copies the values into the existing parameters and touches nothing else (the experiments restore the full-precision weights with their own `restore_model_work`)."""
import hashlib

import torch

from . import canonical


class ArtifactHashMismatch(Exception):
    """An artifact's sha256 differs from the frozen value (or there is none and recording was not requested)."""


def _raw(t):
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def artifact_sha256(tensors):
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name]
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode()); h.update(_raw(t))
    return h.hexdigest()


def apply(model, tensors):
    """Copy `tensors` into the target parameters of `model`. All names, shapes and dtypes are checked before the first copy (ValueError otherwise); non-target parameters are never touched."""
    targets = dict(canonical.targets(model))
    missing, extra = sorted(set(targets) - set(tensors)), sorted(set(tensors) - set(targets))
    if missing or extra:
        raise ValueError(f"artifact and model targets differ: missing {missing[:3]}{'...' if len(missing) > 3 else ''}, extra {extra[:3]}{'...' if len(extra) > 3 else ''}")
    for name, t in tensors.items():
        p = targets[name]
        if tuple(t.shape) != tuple(p.shape) or t.dtype != p.dtype:
            raise ValueError(f"{name}: artifact {tuple(t.shape)} {t.dtype} does not fit parameter {tuple(p.shape)} {p.dtype}")
    with torch.no_grad():
        for name, t in tensors.items():
            targets[name].data.copy_(t)
