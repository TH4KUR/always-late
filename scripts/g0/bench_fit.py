"""Plan-verification probe for running Duan's script 03 on Ada (2 x 11 GB GPUs, Turing).
Uses the UNGATED unsloth mirror via plain HF, so it tests memory fit, bf16-on-Turing speed, SAE loading with the
installed sae_lens, and the CPU float64 accumulation cost. It does NOT test TransformerLens' gated-config load."""
import os, time, json, sys
os.environ.setdefault("HF_HOME", "/scratch/karma-hf-bench")
import numpy as np, torch
out = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "n_gpu": torch.cuda.device_count()}
for i in range(torch.cuda.device_count()):
    out[f"gpu{i}"] = dict(name=torch.cuda.get_device_name(i), cc=torch.cuda.get_device_capability(i),
                          total_gib=round(torch.cuda.get_device_properties(i).total_memory/2**30, 2))
try: out["is_bf16_supported"] = torch.cuda.is_bf16_supported()
except Exception as e: out["is_bf16_supported"] = repr(e)
def bench(dt):
    a = torch.randn(256, 2304, device="cuda:0", dtype=dt); b = torch.randn(2304, 9216, device="cuda:0", dtype=dt)
    for _ in range(5): a @ b
    torch.cuda.synchronize(); t = time.time()
    for _ in range(50): a @ b
    torch.cuda.synchronize(); return round((time.time()-t)/50*1000, 3)
out["matmul_ms_256x2304x9216"] = {str(dt).split(".")[-1]: bench(dt) for dt in (torch.float32, torch.float16, torch.bfloat16)}
print(json.dumps(out, indent=1), flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("unsloth/gemma-2-2b")
t0 = time.time()
m_ref = AutoModelForCausalLM.from_pretrained("unsloth/gemma-2-2b", dtype=torch.bfloat16).to("cuda:0").eval()
m_work = AutoModelForCausalLM.from_pretrained("unsloth/gemma-2-2b", dtype=torch.bfloat16).to("cuda:1").eval()
out["load_two_models_s"] = round(time.time()-t0, 1)
out["attn_impl"] = getattr(m_ref.config, "_attn_implementation", "?")
# SAE via the installed sae_lens, exactly as Duan's script loads it
import sae_lens
out["sae_lens"] = sae_lens.__version__ if hasattr(sae_lens, "__version__") else "?"
from sae_lens import SAE
try:
    sae = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical", sae_id="layer_12/width_16k/canonical", device="cuda:0")
except TypeError:
    sae = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical", sae_id="layer_12/width_16k/canonical")
sae = sae[0] if isinstance(sae, tuple) else sae
sae.eval()
out["sae_cfg"] = dict(d_in=sae.cfg.d_in, d_sae=sae.cfg.d_sae, hook_name=getattr(sae.cfg, "hook_name", None) or getattr(getattr(sae.cfg, "metadata", None), "hook_name", None))
out["sae_dtype"] = str(next(sae.parameters()).dtype)
# real WikiText-2 train tokens, as in Duan's build_tokens_2d
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
text = "\n\n".join(x for x in ds["text"] if x.strip())
ids = tok.encode(text, add_special_tokens=False)
out["wikitext_train_tokens"] = len(ids); out["seqs_of_256_in_500k"] = min(500_000, len(ids))//256
toks = torch.tensor(ids[:256*8], dtype=torch.long).reshape(8, 256)
# one streaming step, as Duan's loop does it: two forwards + SAE encode + float64 CPU accumulation
d_sae = 16384; sums = [torch.zeros(d_sae, dtype=torch.float64) for _ in range(5)]
def fwd_acts(m, dev, batch):
    store = {}
    h = m.model.layers[12].register_forward_hook(lambda mod, i, o: store.__setitem__("a", (o[0] if isinstance(o, tuple) else o).detach()))
    with torch.no_grad(): m(batch.to(dev))
    h.remove(); return store["a"].reshape(-1, store["a"].shape[-1])
def step(batch):
    a = fwd_acts(m_ref, "cuda:0", batch); b = fwd_acts(m_work, "cuda:1", batch)
    with torch.no_grad():
        x = sae.encode(a.to("cuda:0").float()).cpu().to(torch.float64); y = sae.encode(b.to("cuda:0").float()).cpu().to(torch.float64)
    sums[0] += x.sum(0); sums[1] += y.sum(0); sums[2] += (x**2).sum(0); sums[3] += (y**2).sum(0); sums[4] += (x*y).sum(0)
for i in range(2): step(toks[i:i+1])
torch.cuda.synchronize(); t = time.time()
for i in range(8): step(toks[i:i+1])
torch.cuda.synchronize(); per = (time.time()-t)/8
out["streaming_step_s(2 fwd + 2 SAE + cpu f64)"] = round(per, 3)
t = time.time()
with torch.no_grad():
    for i in range(8): m_ref(toks[i:i+1].to("cuda:0"))
torch.cuda.synchronize(); out["single_fwd_s_256tok"] = round((time.time()-t)/8, 3)
out["max_mem_gib"] = {f"cuda:{i}": round(torch.cuda.max_memory_allocated(i)/2**30, 2) for i in range(2)}
# check the two bf16 models are bit-identical in output for the same input (needed: FP-vs-FP must give r==1)
with torch.no_grad():
    a = m_ref(toks[:1].to("cuda:0")).logits.float().cpu(); b = m_work(toks[:1].to("cuda:1")).logits.float().cpu()
out["identical_logits_across_gpus"] = bool(torch.equal(a, b)); out["max_abs_logit_diff_across_gpus"] = float((a-b).abs().max())
tot = out["streaming_step_s(2 fwd + 2 SAE + cpu f64)"]
out["projection_G0_core_hours(23.4k fwd ~ 7.8k streaming steps + 7.8k ppl fwds)"] = round((1953*4*tot + 1953*4*out["single_fwd_s_256tok"])/3600, 2)
print(json.dumps(out, indent=1), flush=True)
json.dump(out, open("/home2/karma/duan-verify/bench_fit_result.json", "w"), indent=1)
