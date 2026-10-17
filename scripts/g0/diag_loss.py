"""DIAGNOSTIC (not a replication): where does the 0.72% perplexity gap between our Phase 1 runner and their 3a_test smoke come from (anchor S5, anomaly A-024)?
Computes, for each of their 78 windows (their build_tokens_2d, their model loader, batch 1), the loss as (a) their call `model(x, return_type="loss")`, (b) the same
formula by hand on bf16 logits, (c) the token-sum in float64 on bf16 log-softmax, (d) float32 log-softmax. Prints the means and the per-window differences."""
import gc, sys
import numpy as np
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from datasets import load_dataset
import their_code as tc

torch.set_grad_enabled(False)
env = {"torch": torch, "np": np, "tqdm": tqdm, "gc": gc, "HookedTransformer": HookedTransformer, "load_dataset": load_dataset, "DEVICE": "cuda", "WORK_DEVICE": "cuda", "BATCH_SIZE": 1}
tc.build_namespace(tc.extract_functions(tc.PRISTINE, ["load_gemma_tl_model", "build_tokens_2d"]), env)
model, _ = env["load_gemma_tl_model"](["gemma-2-2b", "google/gemma-2-2b"]); toks = env["build_tokens_2d"](model.tokenizer, 20000, 256)
rows = []
for w in toks:
    x = w[None]
    a = float(model(x, return_type="loss").item()); logits = model(x)
    g = lambda lg: lg[:, :-1].gather(-1, x[:, 1:, None])[..., 0]
    lp = g(torch.log_softmax(logits, -1)); lp32 = g(torch.log_softmax(logits.float(), -1))
    rows.append((a, float((-lp.mean()).item()), float(-lp.double().sum() / lp.numel()), float(-lp32.double().sum() / lp32.numel())))
r = np.array(rows); names = ["their call (return_type=loss)", "hand bf16 mean", "bf16 log-softmax, float64 sum", "float32 log-softmax, float64 sum"]
for k, n in enumerate(names):
    print(f"{n:34s} mean loss {r[:, k].mean():.6f}  ppl {np.exp(r[:, k].mean()):.3f}", flush=True)
print("max |their call - hand bf16 mean|:", np.abs(r[:, 0] - r[:, 1]).max())
print("mean (their call - float64 sum on bf16 log-softmax):", (r[:, 0] - r[:, 2]).mean(), " mean (bf16 - f32 log-softmax):", (r[:, 2] - r[:, 3]).mean())
print("per-window quantisation steps of their call (unique diffs of sorted values, first 5):", np.unique(np.round(np.diff(np.sort(r[:, 0])), 6))[:5])
