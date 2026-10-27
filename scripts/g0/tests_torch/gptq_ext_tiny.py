"""Shared helper of the GPTQ extension tests: a tiny random-weight Gemma-2 HookedTransformer (the architecture flags of TransformerLens 3.9.0's `google/gemma-2-2b` entry: grouped-query attention, alternating local
and global layers, gated MLP, RMS norm before and after, soft caps), or None when TransformerLens is not installed (the local `~/.venvs/anlp-torch` has no TransformerLens; `~/.venvs/anlp-tl` has the version of Ada)."""
import warnings

import torch

try:
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    HAVE_TL = True
except Exception:          # pragma: no cover - environment dependent
    HAVE_TL = False

DIMS = {"d_model": 32, "d_head": 8, "n_heads": 4, "n_kv": 2, "d_mlp": 64, "d_vocab": 100}


def tiny_gemma2(n_layers=2, dtype=torch.float32, seed=0, scale=0.3):
    if not HAVE_TL:
        return None
    cfg = HookedTransformerConfig(d_model=DIMS["d_model"], d_head=DIMS["d_head"], n_heads=DIMS["n_heads"], d_mlp=DIMS["d_mlp"], n_layers=n_layers, n_ctx=64, eps=1e-6, d_vocab=DIMS["d_vocab"],
                                  act_fn="gelu_pytorch_tanh", normalization_type="RMS", rotary_base=10000.0, positional_embedding_type="rotary", use_attn_scale=True, n_key_value_heads=DIMS["n_kv"],
                                  window_size=16, use_local_attn=True, attn_types=(["local", "global"] * n_layers)[:n_layers], attn_scores_soft_cap=50.0, output_logits_soft_cap=30.0, gated_mlp=True,
                                  final_rms=True, use_normalization_before_and_after=True, original_architecture="Gemma2ForCausalLM", dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = HookedTransformer(cfg)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith((".w",)):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            elif ".b_" in name or "._b_" in name or name.endswith(("b_U",)):
                p.zero_()
            else:
                p.copy_((scale * torch.randn(p.shape, generator=g)).to(p.dtype))
    return model.eval()


def tokens(batch=3, length=10, seed=0, bos=1):
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(2, DIMS["d_vocab"], (batch, length), generator=g)
    t[:, 0] = bos
    return t
