"""The authors' GPTQ core (https://github.com/IST-DASLab/gptq, commit 2d65066e, Apache-2.0), vendored byte for byte in this directory and verified by sha256 at every load (reports/quantization_gptq_audit_2026-10-27/evidence.json records
the same hashes). The two files are never edited. They are not imported as modules (`gptq.py` imports `transformers` at the top, `from quant import *`, and calls `torch.cuda.synchronize()` unconditionally); instead the class and function
bodies this project needs are extracted by AST and executed in a controlled namespace, as the audit's CPU checks did (reports/quantization_gptq_audit_2026-10-27/check_compatibility.py). The only substitutions, none of which touches
the optimisation arithmetic: (1) `transformers.Conv1D` (used only in isinstance tests, never true for the nn.Linear proxies of this project) is a dummy class, because the installed Transformers 5.x does not export it at top level;
(2) `torch.cuda.synchronize()` is a no-op when CUDA is unavailable (the timing print of upstream is left in place); (3) the TF32 switches that upstream sets at import time are set to False the same way."""
import ast
import hashlib
import math
import time
import types
from functools import lru_cache
from pathlib import Path

import torch

UPSTREAM_DIR = Path(__file__).resolve().parent
PINNED_SHA256 = {"gptq.py": "bb47468cc2c788577562751de414d76eeb9968ff1c6098e21153557ef3d154c7", "quant.py": "211528568b962b868d20fc58b55200b615f41ebd9f36b4d2d7e4a196a095ea5a"}


class HashMismatch(Exception):
    """A vendored upstream file is missing or differs from the pinned commit."""


class _Conv1D:
    """Stand-in for transformers.Conv1D (upstream uses it only in isinstance tests)."""


class _CudaView:
    def __getattr__(self, name):
        if name == "synchronize" and not torch.cuda.is_available():
            return lambda *a, **k: None
        return getattr(torch.cuda, name)


class _TorchView:
    cuda = _CudaView()

    def __getattr__(self, name):
        return getattr(torch, name)


def _read_verified(directory):
    d = Path(directory)
    sources = {}
    for name, want in PINNED_SHA256.items():
        p = d / name
        if not p.is_file():
            raise HashMismatch(f"{p} is missing")
        body = p.read_bytes()
        got = hashlib.sha256(body).hexdigest()
        if got != want:
            raise HashMismatch(f"{p}: sha256 {got} differs from the pinned {want}")
        sources[name] = body.decode("utf-8")
    return sources


def _extract(source, names, namespace, filename):
    tree = ast.parse(source)
    for name in names:
        node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)


def load(directory=None):
    """-> SimpleNamespace(GPTQ, Quantizer, quantize, sha256) from the verified files of `directory` (default: this directory). Raises HashMismatch."""
    src = _read_verified(directory or UPSTREAM_DIR)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ns = {"math": math, "time": time, "torch": _TorchView(), "nn": torch.nn, "transformers": types.SimpleNamespace(Conv1D=_Conv1D), "DEBUG": False}
    _extract(src["quant.py"], ("quantize", "Quantizer"), ns, "upstream/quant.py")
    _extract(src["gptq.py"], ("GPTQ",), ns, "upstream/gptq.py")
    return types.SimpleNamespace(GPTQ=ns["GPTQ"], Quantizer=ns["Quantizer"], quantize=ns["quantize"], sha256=dict(PINNED_SHA256))
