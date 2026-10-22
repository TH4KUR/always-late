"""Check of the TransformerLens semantics that the F5 runner relies on, on a tiny random Gemma-2-style HookedTransformer (CPU, no weights needed; run it in the job environment, for example on the login node of Ada):
`model(resid, start_at_layer=L + 1, stop_at_layer=n_layers)` returns the residual before the final norm, a float32 residual passes through bf16 blocks, and `make_logits_fn` (float32 final norm, unembedding in vocabulary slices, soft cap)
reproduces the model's own logits (exactly for a float32 model, within bf16 noise for a bf16 model).

    OMP_NUM_THREADS=2 <env>/bin/python -B scripts/g0/check_f5_tl_semantics.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import phase1_func_f5_run as r5
torch.manual_seed(0); torch.set_grad_enabled(False)
for dtype in (torch.bfloat16, torch.float32):
    cfg = HookedTransformerConfig(n_layers=4, d_model=32, n_ctx=64, d_head=8, n_heads=4, d_mlp=64, d_vocab=200, act_fn="gelu_pytorch_tanh", normalization_type="RMS",
                                  positional_embedding_type="rotary", rotary_dim=8, use_normalization_before_and_after=True, gated_mlp=True, final_rms=True, output_logits_soft_cap=30.0,
                                  attn_scores_soft_cap=50.0, attn_types=["global", "local", "global", "local"], window_size=8, use_attn_scale=True, original_architecture="Gemma2ForCausalLM",
                                  dtype=dtype, device="cpu", eps=1e-6)
    model = HookedTransformer(cfg); model.eval()
    for p in model.parameters():
        p.data = (torch.randn_like(p.data.float()) * (0.5 if p.ndim > 1 else 0.2) + (1.0 if p.ndim == 1 else 0.0)).to(dtype)
    tokens = torch.randint(2, 200, (3, 14)); L = 1
    resid = model(tokens, stop_at_layer=L + 1)
    native = model(resid, start_at_layer=L + 1)
    up = r5.make_upper_final(model, L); fn = r5.make_logits_fn(model, vocab_chunk=64)
    h = up(resid); h32 = up(resid.float())
    print(dtype, "resid", resid.dtype, tuple(resid.shape), "hidden", h.dtype, tuple(h.shape), "hidden(float32 in)", h32.dtype, tuple(h32.shape))
    z = fn(h.reshape(-1, h.shape[-1])); z32 = fn(h32.reshape(-1, h32.shape[-1])); ref = native.reshape(-1, native.shape[-1]).float()
    print("  logits dtype", z.dtype, "agreement native-resid", r5.logit_agreement(ref, z, 0.95, 2.0), "float32-resid", r5.logit_agreement(ref, z32, 0.95, 2.0))
    print("  soft cap", cfg.output_logits_soft_cap, "max |logit|", float(z.abs().max()))
