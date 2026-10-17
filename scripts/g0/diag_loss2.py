"""DIAGNOSTIC (not a replication), follow-up to diag_loss.py: their call gives a mean loss 0.0073 higher than the same formula applied by hand to the logits of the same
windows. Isolate it: are the logits of model(x, return_type="loss"/"both") equal to those of model(x)? Does TL's own lm_cross_entropy_loss on model(x)'s logits give their value?
Also: is the first window's loss reproduced by a second identical call (determinism)?"""
import gc
import numpy as np
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from transformer_lens import utils as tlu
from datasets import load_dataset
import their_code as tc

torch.set_grad_enabled(False)
env = {"torch": torch, "np": np, "tqdm": tqdm, "gc": gc, "HookedTransformer": HookedTransformer, "load_dataset": load_dataset, "DEVICE": "cuda", "WORK_DEVICE": "cuda", "BATCH_SIZE": 1}
tc.build_namespace(tc.extract_functions(tc.PRISTINE, ["load_gemma_tl_model", "build_tokens_2d"]), env)
model, _ = env["load_gemma_tl_model"](["gemma-2-2b", "google/gemma-2-2b"]); toks = env["build_tokens_2d"](model.tokenizer, 20000, 256)
rows, diffs = [], []
for w in toks:
    x = w[None]
    a = float(model(x, return_type="loss").item()); a2 = float(model(x, return_type="loss").item())
    lg = model(x); both_lg, both_loss = model(x, return_type="both")
    tl_on_logits = float(tlu.lm_cross_entropy_loss(lg, x).item()); tl_on_both = float(tlu.lm_cross_entropy_loss(both_lg, x).item())
    rows.append((a, a2, float(both_loss.item()), tl_on_logits, tl_on_both)); diffs.append(float((lg.float() - both_lg.float()).abs().max()))
r = np.array(rows); names = ["their call", "their call again", "return_type=both loss", "TL fn on model(x) logits", "TL fn on both-logits"]
for k, n in enumerate(names):
    print(f"{n:28s} mean loss {r[:, k].mean():.6f}", flush=True)
print("max |logits(model(x)) - logits(both)|:", max(diffs), " | logits dtype:", lg.dtype, " | loss dtype of their call:", model(x, return_type="loss").dtype)
print("same-window determinism: max |their call - their call again|:", np.abs(r[:, 0] - r[:, 1]).max())
