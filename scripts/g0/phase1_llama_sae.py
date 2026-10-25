"""TopK sparse autoencoder of `seonglae/Llama-3.2-1B-sae` (SAELens file format: `<folder>/cfg.json` and `<folder>/sae_weights.safetensors`) for the Llama-3.2-1B consistency check
(plan docs/superpowers/plans/2026-10-26-llama-consistency-check.md, D-087). Plain PyTorch, no sae_lens: the loader refuses every configuration it does not implement.

    encode(x) = top-k over the last axis of  relu((x - b_dec) @ W_enc + b_enc)   (the values of the k largest, zeros elsewhere; a token with fewer than k positive pre-activations keeps only those)
    decode(f) = f @ W_dec + b_dec

The interface is the one the frozen runners use of a Gemma Scope SAE: `encode`, `decode`, `W_dec` [d_sae, d_in], `cfg.d_in`, `cfg.d_sae`, `cfg.hook_name`. The frozen code reads the residual as
`blocks.<LAYER>.hook_resid_post`; the Llama SAE hook `blocks.12.hook_resid_pre` is the same tensor as `blocks.11.hook_resid_post`, so a `hook_alias` can replace the name the frozen code sees (the original
name stays in `cfg.hook_name_original`)."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch


class TopKSAE(torch.nn.Module):
    def __init__(self, W_enc, W_dec, b_enc, b_dec, k, cfg):
        super().__init__()
        self.W_enc = torch.nn.Parameter(W_enc, requires_grad=False); self.W_dec = torch.nn.Parameter(W_dec, requires_grad=False)
        self.b_enc = torch.nn.Parameter(b_enc, requires_grad=False); self.b_dec = torch.nn.Parameter(b_dec, requires_grad=False)
        self.k, self.cfg = int(k), cfg

    @torch.no_grad()
    def encode(self, x):
        pre = torch.relu((x.to(self.W_enc.device).float() - self.b_dec) @ self.W_enc + self.b_enc)
        vals, idx = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, vals)

    @torch.no_grad()
    def decode(self, f):
        return f.to(self.W_dec.device).float() @ self.W_dec + self.b_dec


def load_topk_sae(folder, expected_hook=None, hook_alias=None):
    """The SAE of one folder (cfg.json and sae_weights.safetensors), weights as float32, eval mode, no gradients."""
    from safetensors.torch import load_file
    folder = Path(folder)
    cfg_path, w_path = folder / "cfg.json", folder / "sae_weights.safetensors"
    if not cfg_path.exists() or not w_path.exists():
        raise FileNotFoundError(f"{folder}: cfg.json and sae_weights.safetensors are needed")
    c = json.loads(cfg_path.read_text())
    if c.get("architecture") != "topk":
        raise ValueError(f"{folder}: expected architecture 'topk', got {c.get('architecture')!r}")
    if not c.get("apply_b_dec_to_input", False):
        raise ValueError(f"{folder}: apply_b_dec_to_input=False is not supported (encode centres the input on the decoder bias)")
    if c.get("normalize_activations", "none") != "none":
        raise ValueError(f"{folder}: activation normalisation {c['normalize_activations']!r} is not supported")
    if not c.get("prepend_bos", False):
        raise ValueError(f"{folder}: the SAE was trained without a BOS token; this study prepends one (mask regime)")
    k = (c.get("activation_fn_kwargs") or {}).get("k")
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"{folder}: no valid k in activation_fn_kwargs ({c.get('activation_fn_kwargs')!r})")
    hook = c.get("hook_name")
    if expected_hook is not None and hook != expected_hook:
        raise ValueError(f"{folder}: the SAE hook is {hook!r}, expected {expected_hook!r}")
    state = load_file(str(w_path))
    missing = [n for n in ("W_enc", "W_dec", "b_enc", "b_dec") if n not in state]
    if missing:
        raise ValueError(f"{folder}: weights lack {missing}")
    d_in, d_sae = state["W_enc"].shape
    if (d_in, d_sae) != (c["d_in"], c["d_sae"]) or tuple(state["W_dec"].shape) != (d_sae, d_in) or tuple(state["b_enc"].shape) != (d_sae,) or tuple(state["b_dec"].shape) != (d_in,):
        raise ValueError(f"{folder}: weight shapes disagree with the config (d_in {c['d_in']}, d_sae {c['d_sae']})")
    if k > d_sae:
        raise ValueError(f"{folder}: k = {k} is larger than the dictionary ({d_sae})")
    cfg = SimpleNamespace(d_in=int(d_in), d_sae=int(d_sae), hook_name=hook_alias or hook, hook_name_original=hook, architecture="topk", k=k)
    sae = TopKSAE(state["W_enc"].float(), state["W_dec"].float(), state["b_enc"].float(), state["b_dec"].float(), k, cfg)
    return sae.eval()


def load_sae_compat_llama(release, sae_id, device, hook_alias=None, expected_hook=None):
    """The signature of the frozen runners' `load_sae_compat(release, sae_id, device)`: `release` is a local directory that holds the folder `sae_id`, or a Hugging Face repository id (the files come
    from the local cache when the job is offline)."""
    if Path(release).is_dir():
        folder = Path(release) / sae_id
    else:
        from huggingface_hub import hf_hub_download
        cfg_file = hf_hub_download(release, f"{sae_id}/cfg.json"); hf_hub_download(release, f"{sae_id}/sae_weights.safetensors")
        folder = Path(cfg_file).parent
    sae = load_topk_sae(folder, expected_hook=expected_hook, hook_alias=hook_alias)
    return sae.to(device)
