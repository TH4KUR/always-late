"""Reproduce the GPTQ audit's mathematical checks; no model or GPU is used.

Run with the existing anlp-torch Python, optionally passing --out evidence.json.
Only hash-verified upstream class/function bodies are executed. This is an audit
utility, not a quantization adapter or a production experiment runner.
"""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import time
import types
import urllib.request

import torch


UPSTREAM = "2d65066eeb06a5c9ff5184d8cebdf33662c67faf"
HASHES = {
    "gptq.py": "bb47468cc2c788577562751de414d76eeb9968ff1c6098e21153557ef3d154c7",
    "quant.py": "211528568b962b868d20fc58b55200b615f41ebd9f36b4d2d7e4a196a095ea5a",
}


def extract(source, names, namespace, filename):
    tree = ast.parse(source)
    for name in names:
        node = next(
            n for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)


def check():
    namespace = {
        "torch": torch, "nn": torch.nn, "math": math, "time": time, "DEBUG": False,
        "transformers": types.SimpleNamespace(Conv1D=type("Conv1D", (torch.nn.Module,), {})),
    }
    with tempfile.TemporaryDirectory(prefix="always-late-gptq-audit-") as temporary:
        for name, symbols in (("quant.py", ("quantize", "Quantizer")), ("gptq.py", ("GPTQ",))):
            url = f"https://raw.githubusercontent.com/IST-DASLab/gptq/{UPSTREAM}/{name}"
            with urllib.request.urlopen(url, timeout=30) as response:
                source = response.read()
            if hashlib.sha256(source).hexdigest() != HASHES[name]:
                raise ValueError(f"upstream source hash differs: {name}")
            path = Path(temporary) / name
            path.write_bytes(source)
            extract(source, symbols, namespace, str(path))

    root = Path(__file__).resolve().parents[2]
    legacy_path = root / "third_party/duan2026/pristine/scripts/03_gemma_phase3_streaming_sweep.py"
    legacy_source = legacy_path.read_bytes()
    legacy = {"torch": torch}
    extract(legacy_source, ("target_tl_weight", "quantize_rtn_per_output_channel_tl"), legacy, str(legacy_path))

    torch.manual_seed(42)
    heads, model_width, head_width = 4, 12, 3
    weights = torch.randn(heads, model_width, head_width, dtype=torch.float64)
    inputs = torch.randn(7, model_width, dtype=torch.float64)
    linear = weights.permute(0, 2, 1).reshape(heads * head_width, model_width)
    projected = (inputs @ linear.T).reshape(7, heads, head_width)
    reference = torch.einsum("bm,hmd->bhd", inputs, weights)
    output_weights = torch.randn(heads, head_width, model_width, dtype=torch.float64)
    values = torch.randn(7, heads, head_width, dtype=torch.float64)
    output_linear = output_weights.permute(2, 0, 1).reshape(model_width, heads * head_width)
    checks = {
        "qkv_projection_mapping_max_abs": float((projected - reference).abs().max()),
        "qkv_roundtrip_exact": torch.equal(weights, linear.reshape(heads, head_width, model_width).permute(0, 2, 1)),
        "o_projection_mapping_max_abs": float((values.reshape(7, -1) @ output_linear.T - torch.einsum("bhd,hdm->bm", values, output_weights)).abs().max()),
        "o_roundtrip_exact": torch.equal(output_weights, output_linear.reshape(model_width, heads, head_width).permute(1, 2, 0)),
        "qkv_legacy_scale_count": weights.abs().amax((0, 1), keepdim=True).numel(),
        "qkv_standard_scale_count": linear.abs().amax(1, keepdim=True).numel(),
        "target_filter_includes_gqa_storage": all(legacy["target_tl_weight"](name) for name in ("blocks.0.attn._W_K", "blocks.0.attn._W_V")),
    }
    assert checks["qkv_projection_mapping_max_abs"] < 1e-12
    assert checks["o_projection_mapping_max_abs"] < 1e-12
    assert checks["qkv_roundtrip_exact"] and checks["o_roundtrip_exact"]
    assert checks["target_filter_includes_gqa_storage"]

    quantizer = namespace["Quantizer"]()
    quantizer.configure(4, perchannel=True, sym=True, mse=False)
    example = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])
    quantizer.find_params(example, weight=True)
    fake = torch.nn.Module()
    fake.register_parameter("W_in", torch.nn.Parameter(example.T.clone()))
    legacy["quantize_rtn_per_output_channel_tl"](fake, bits=4)
    checks.update({
        "author_sym_scale": quantizer.scale.flatten().tolist(),
        "legacy_scale": [1 / 7],
        "same_sym_flag_same_grid": torch.equal(quantizer.quantize(example), fake.W_in.T),
        "author_quantized_example": quantizer.quantize(example).tolist(),
        "legacy_quantized_example": fake.W_in.detach().T.tolist(),
    })
    assert not checks["same_sym_flag_same_grid"]

    algorithm_checks = []
    original_sync = torch.cuda.synchronize
    torch.cuda.synchronize = lambda: None  # Upstream timing only; no CPU arithmetic replacement.
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for bits in (4, 6):
                torch.manual_seed(73)
                layer = torch.nn.Linear(12, 8, bias=False, dtype=dtype)
                original = layer.weight.detach().clone()
                calibration = torch.eye(12, dtype=dtype).unsqueeze(0)
                algorithm = namespace["GPTQ"](layer)
                algorithm.quantizer = namespace["Quantizer"]()
                algorithm.quantizer.configure(bits, perchannel=True, sym=True, mse=False)
                algorithm.add_batch(calibration, layer(calibration))
                nearest = namespace["Quantizer"]()
                nearest.configure(bits, perchannel=True, sym=True, mse=False)
                nearest.find_params(original.float(), weight=True)
                expected = nearest.quantize(original.float()).to(dtype)
                with contextlib.redirect_stdout(io.StringIO()):
                    algorithm.fasterquant(blocksize=4, percdamp=0.01, groupsize=-1, actorder=False)
                assert torch.equal(layer.weight, expected)
                assert torch.isfinite(layer.weight).all() and layer.weight.dtype == dtype
                algorithm_checks.append({"dtype": str(dtype), "bits": bits, "identity_hessian_matches_same_grid_nearest_exactly": True, "output_dtype_preserved": True})
    finally:
        torch.cuda.synchronize = original_sync

    return {
        "scope": "Source audit and CPU mathematical checks only; no Gemma inference or scientific experiment",
        "audited_repository_commit": "3d5c6d7160907bfc6bf75459faf1cb836b64f7a8",
        "torch_version": torch.__version__,
        "upstream_commit": UPSTREAM,
        "upstream_sha256": HASHES,
        "legacy_script_sha256": hashlib.sha256(legacy_source).hexdigest(),
        "inspected_transformer_lens_release": "3.9.0",
        "inspected_gemma_conversion_sha256": "65e84ef124c324331486ee4317c3e12782133cf4c2ba0f10eb57963f7b1f1442",
        "upstream_algorithm_body_modified": False,
        "substitutions": "Unused Transformers Conv1D type stub; CUDA timing synchronize skipped on CPU; optimization arithmetic unchanged",
        "layout_and_grid_checks": checks,
        "upstream_core_checks": algorithm_checks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    result = json.dumps(check(), indent=2) + "\n"
    if arguments.out:
        arguments.out.write_text(result)
    print(result, end="")
