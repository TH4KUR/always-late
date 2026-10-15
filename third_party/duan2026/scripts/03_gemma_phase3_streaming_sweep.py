# Auto-exported from 03_gemma_phase3_streaming_sweep.ipynb
# Some notebooks are intended for interactive/XXXX-13 use.


# %% [markdown]
# # QDM Phase 3: Gemma-2-2B Gemma Scope SAE Sweep
# 
# This notebook runs:

# %% Cell 1
# Install dependencies if needed.
# Run this only if imports fail.
# !pip install -q -U transformer-lens sae-lens datasets tqdm pandas matplotlib scikit-learn huggingface_hub

# %% Cell 2
import os

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import torch
import pandas as pd
import transformer_lens
import sae_lens
from datasets import load_dataset

print("Imports OK")
print("CUDA available:", torch.cuda.is_available())

# %% Cell 3
import torch
import pandas as pd
import transformer_lens
import sae_lens
from datasets import load_dataset

print("Imports OK")
print("CUDA available:", torch.cuda.is_available())

# %% Cell 4
# Optional Hugging Face login.
# Gemma models may require accepting Google's license on Hugging Face and logging in.

from huggingface_hub import login
import getpass, os

DO_HF_LOGIN = False  # set True if model download fails or asks for authentication

if DO_HF_LOGIN:
    hf_token = getpass.getpass("Paste your Hugging Face token: ")
    login(token=hf_token)
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    print("Hugging Face login complete.")
else:
    print("Skipping HF login.")

# %% Cell 5
import os
import gc
import json
import math
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

torch.set_grad_enabled(False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORK_DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE  # [G0-EDIT 2] second bf16 model on a second card when one exists (11 GB cards); identical to the shipped behaviour on one large GPU
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GB:", torch.cuda.get_device_properties(0).total_memory / 1e9)

# %% Cell 6
# =========================
# Config
# =========================

# Choose one:
#   "3a_test"  = 20k tokens, FP16 + INT8 + INT6
#   "3b_full"  = 500k tokens, FP16 + INT8/7/6/5/4 + pruning
RUN_MODE = "3b_full"

MODEL_CANDIDATES = [
    "gemma-2-2b",
    "google/gemma-2-2b",
]

# Gemma Scope residual stream canonical SAE for Gemma-2-2B.
# If this fails, run the SAE diagnostic cell below to list available releases.
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"

LAYER = 12
SEQ_LEN = 256

if RUN_MODE == "3a_test":
    TOKEN_BUDGET = 20_000
    BITS_TO_TEST = [8, 6]
    BATCH_SIZE = 1
    INCLUDE_INT4 = False
    RUN_PRUNING = False
elif RUN_MODE == "3b_full":
    TOKEN_BUDGET = 500_000
    BITS_TO_TEST = [8, 7, 6, 5, 4]
    BATCH_SIZE = 1
    INCLUDE_INT4 = True
    RUN_PRUNING = False  # [G0-EDIT 1] their own flag: their pruning stage is not needed for this study
else:
    raise ValueError("RUN_MODE must be '3a_test' or '3b_full'.")

FIRING_THRESHOLD = 0.001
PRUNE_MATCH_BITS = 6
PRUNE_SEARCH_TOKEN_BUDGET = 100_000
PRUNE_SEARCH_STEPS = 6

OUTPUT_DIR = Path(f"phase3_gemma_outputs_{RUN_MODE}_{TOKEN_BUDGET//1000}k_L{LAYER}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("RUN_MODE:", RUN_MODE)
print("TOKEN_BUDGET:", TOKEN_BUDGET)
print("BITS_TO_TEST:", BITS_TO_TEST)
print("OUTPUT_DIR:", OUTPUT_DIR)

# %% Cell 7
# Load Gemma model
# =========================

def load_gemma_tl_model(candidates, device=DEVICE):
    last_err = None

    for name in candidates:
        try:
            print(f"Trying model: {name}")

            model = HookedTransformer.from_pretrained_no_processing(
                name,
                device=device,
                dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,  # [G0-EDIT 3] was: device == "cuda" (would silently give float32 for "cuda:1")
            )

            model.eval()

            print("Loaded:", name)
            print("Layers:", model.cfg.n_layers, "d_model:", model.cfg.d_model)

            return model, name

        except Exception as e:
            print("Failed:", repr(e)[:500])
            last_err = e

    raise RuntimeError(f"Could not load any Gemma candidate. Last error: {last_err}")


model_ref, MODEL_NAME_USED = load_gemma_tl_model(MODEL_CANDIDATES)
model_work, _ = load_gemma_tl_model([MODEL_NAME_USED], device=WORK_DEVICE)  # [G0-EDIT 4]

print("Reference and work models loaded.")

# %% Cell 8
# =========================
# Load Gemma Scope SAE
# =========================

def load_sae_compat(release, sae_id, device=DEVICE):
    # SAE Lens v6+ usually returns a single SAE from SAE.from_pretrained().
    # Older versions may return tuples.
    try:
        out = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    except TypeError:
        out = SAE.from_pretrained(release=release, sae_id=sae_id)
        if hasattr(out, "to"):
            out = out.to(device)

    if isinstance(out, tuple):
        sae = out[0]
    else:
        sae = out

    sae.eval()
    return sae

sae = load_sae_compat(SAE_RELEASE, SAE_ID, DEVICE)

print("SAE loaded")
print("SAE release:", SAE_RELEASE)
print("SAE id:", SAE_ID)
print("SAE d_in:", sae.cfg.d_in)
print("SAE d_sae:", sae.cfg.d_sae)

HOOK_NAME = getattr(sae.cfg, "hook_name", None)
if HOOK_NAME is None:
    HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"

print("Hook name:", HOOK_NAME)

assert sae.cfg.d_in == model_ref.cfg.d_model, (
    f"Shape mismatch: SAE d_in={sae.cfg.d_in}, model d_model={model_ref.cfg.d_model}. "
    "You probably loaded the wrong SAE for this model/site."
)

# %% Cell 9
# Optional diagnostic if SAE loading failed above.
# Uncomment and run to inspect Pythia/Gemma releases known to SAE Lens.

# import sae_lens
# import pkgutil
# print("sae_lens version:", sae_lens.__version__)
# print("Top-level modules:", [m.name for m in pkgutil.iter_modules(sae_lens.__path__)])
#
# try:
#     from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
#     directory = get_pretrained_saes_directory()
#     gemma_releases = [k for k in directory.keys() if "gemma" in k.lower()]
#     print("Gemma releases:")
#     for r in gemma_releases:
#         print(" ", r)
# except Exception as e:
#     print("Could not inspect pretrained SAE directory:", repr(e))

# %% Cell 10
# =========================
# Build token dataset
# =========================

def build_tokens_2d(tokenizer, token_budget, seq_len):
    # Train split is safer for 500k because WikiText-2 test may not have enough usable tokens.
    split = "test" if token_budget <= 20_000 else "train"
    print(f"Using WikiText-2 split: {split}")

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)

    # Keep all non-empty lines.
    full_text = "\n\n".join(x for x in ds["text"] if x.strip())
    print(f"Total characters: {len(full_text):,}")

    # TransformerLens tokenizer supports encode.
    token_ids = tokenizer.encode(full_text, add_special_tokens=False)
    tokens = torch.tensor(token_ids, dtype=torch.long)

    print(f"Total available tokens: {tokens.shape[0]:,}")

    usable_tokens = min(token_budget, tokens.shape[0])
    n_seqs = usable_tokens // seq_len
    usable_tokens = n_seqs * seq_len

    if usable_tokens == 0:
        raise RuntimeError("Not enough tokens for one sequence.")

    tokens_2d = tokens[:usable_tokens].reshape(n_seqs, seq_len).to(DEVICE)

    print(f"Using tokens: {usable_tokens:,}")
    print(f"tokens_2d shape: {tuple(tokens_2d.shape)}")

    return tokens_2d

tokens_2d = build_tokens_2d(model_ref.tokenizer, TOKEN_BUDGET, SEQ_LEN)

# Smaller token subset for pruning search.
prune_search_tokens_2d = tokens_2d[: max(1, min(tokens_2d.shape[0], PRUNE_SEARCH_TOKEN_BUDGET // SEQ_LEN))]
print("prune_search_tokens_2d shape:", tuple(prune_search_tokens_2d.shape))

# %% Cell 11
# =========================
# Save CPU copy of model_work original state
# =========================

def clone_state_to_cpu(model):
    print("Cloning original state to CPU...")
    state = {}
    for k, v in tqdm(model.state_dict().items(), desc="Cloning state"):
        state[k] = v.detach().cpu().clone()
    return state

original_state_cpu = clone_state_to_cpu(model_work)

def restore_model_work():
    model_work.load_state_dict(original_state_cpu, strict=True)
    model_work.to(WORK_DEVICE)  # [G0-EDIT 5]
    model_work.eval()
    torch.cuda.empty_cache()
    gc.collect()

print("Original CPU state saved.")

# %% Cell 12
# =========================
# Quantization and pruning utilities
# =========================

def target_tl_weight(name):
    # Includes Gemma gated MLP weights.
    target_names = ["W_Q", "W_K", "W_V", "W_O", "W_in", "W_gate", "W_out"]
    return any(t in name for t in target_names)

def quantize_rtn_per_output_channel_tl(model, bits=8):
    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))

    n_tensors = 0
    n_params = 0

    for name, param in model.named_parameters():
        if target_tl_weight(name):
            w = param.data
            if w.ndim == 1:
                scale = w.abs().max() / qmax
            else:
                # Per-output-channel scaling: reduce all dims except last.
                scale = w.abs().amax(
                    dim=tuple(range(w.ndim - 1)),
                    keepdim=True
                ) / qmax

            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            q = torch.round(w / scale).clamp(qmin, qmax)
            param.data = (q * scale).to(w.dtype)

            n_tensors += 1
            n_params += w.numel()

    return n_tensors, n_params

def magnitude_prune_tl(model, sparsity):
    n_tensors = 0
    n_params = 0
    n_zeroed = 0

    for name, param in model.named_parameters():
        if target_tl_weight(name):
            w = param.data
            flat = w.abs().flatten()
            if flat.numel() == 0:
                continue

            k = int(sparsity * flat.numel())
            if k <= 0:
                continue
            if k >= flat.numel():
                threshold = flat.max() + 1
            else:
                threshold = torch.kthvalue(flat, k).values

            mask = w.abs() <= threshold
            n_zeroed += mask.sum().item()
            n_params += w.numel()
            w[mask] = 0

            n_tensors += 1

    return n_tensors, n_params, n_zeroed

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# %% Cell 13
# =========================
# Model evaluation and activation helpers
# =========================

def compute_perplexity(model, tokens_2d, batch_size=BATCH_SIZE, desc="Perplexity"):
    losses = []

    for i in tqdm(range(0, tokens_2d.shape[0], batch_size), desc=desc):
        batch = tokens_2d[i:i + batch_size]
        with torch.no_grad():
            loss = model(batch, return_type="loss")
        losses.append(float(loss.item()))

    avg_loss = float(np.mean(losses))
    ppl = float(np.exp(avg_loss))
    return ppl, avg_loss

def get_hook_acts(model, batch, hook_name):
    with torch.no_grad():
        _, cache = model.run_with_cache(
            batch,
            names_filter=[hook_name],
        )
    acts = cache[hook_name].detach()
    acts = acts.reshape(-1, acts.shape[-1])
    del cache
    return acts

def encode_sae(sae, acts):
    with torch.no_grad():
        feats = sae.encode(acts.to(DEVICE).float())
    return feats

# %% Cell 14
# =========================
# Streaming feature metrics
# =========================

def streaming_feature_metrics(
    model_a,
    model_b,
    sae,
    tokens_2d,
    hook_name,
    condition_name,
    layer,
    batch_size=BATCH_SIZE,
    firing_threshold=FIRING_THRESHOLD,
):
    d_sae = sae.cfg.d_sae

    sum_x = torch.zeros(d_sae, dtype=torch.float64)
    sum_y = torch.zeros(d_sae, dtype=torch.float64)
    sum_x2 = torch.zeros(d_sae, dtype=torch.float64)
    sum_y2 = torch.zeros(d_sae, dtype=torch.float64)
    sum_xy = torch.zeros(d_sae, dtype=torch.float64)

    fire_count = torch.zeros(d_sae, dtype=torch.float64)
    sum_x_activation = torch.zeros(d_sae, dtype=torch.float64)
    max_x_activation = torch.zeros(d_sae, dtype=torch.float64)

    total_positions = 0

    for i in tqdm(range(0, tokens_2d.shape[0], batch_size), desc=f"Streaming {condition_name}"):
        batch = tokens_2d[i:i + batch_size]

        acts_a = get_hook_acts(model_a, batch, hook_name)
        feats_a = encode_sae(sae, acts_a).detach().cpu().to(torch.float64)

        acts_b = get_hook_acts(model_b, batch, hook_name)
        feats_b = encode_sae(sae, acts_b).detach().cpu().to(torch.float64)

        x = feats_a
        y = feats_b

        sum_x += x.sum(dim=0)
        sum_y += y.sum(dim=0)
        sum_x2 += (x ** 2).sum(dim=0)
        sum_y2 += (y ** 2).sum(dim=0)
        sum_xy += (x * y).sum(dim=0)

        fire_count += (x > 0).sum(dim=0)
        sum_x_activation += x.sum(dim=0)
        max_x_activation = torch.maximum(max_x_activation, x.max(dim=0).values)

        total_positions += x.shape[0]

        del acts_a, acts_b, feats_a, feats_b, x, y
        free_memory()

    n = total_positions
    numerator = sum_xy - (sum_x * sum_y / n)
    denom_x = sum_x2 - (sum_x ** 2 / n)
    denom_y = sum_y2 - (sum_y ** 2 / n)
    denominator = torch.sqrt(torch.clamp(denom_x * denom_y, min=1e-12))

    corr = torch.clamp(numerator / denominator, -1.0, 1.0)
    firing_rate = fire_count / n
    mean_activation = sum_x_activation / n

    active_mask = firing_rate > firing_threshold
    active_corrs = corr[active_mask]

    if active_corrs.numel() == 0:
        raise RuntimeError("No active features. Lower FIRING_THRESHOLD.")

    survived = (active_corrs > 0.9).double().mean().item() * 100
    degraded = ((active_corrs > 0.5) & (active_corrs <= 0.9)).double().mean().item() * 100
    damaged = (active_corrs < 0.5).double().mean().item() * 100

    summary = {
        "condition": condition_name,
        "n_tokens": int(n),
        "n_total_features": int(d_sae),
        "n_active_features": int(active_mask.sum().item()),
        "mean_corr": float(active_corrs.mean().item()),
        "median_corr": float(active_corrs.median().item()),
        "survived_>0.9_pct": float(survived),
        "degraded_0.5_0.9_pct": float(degraded),
        "damaged_<0.5_pct": float(damaged),
        "layer": int(layer),
    }

    per_feature = pd.DataFrame({
        "feature_id": np.arange(d_sae),
        "condition": condition_name,
        "layer": int(layer),
        "corr": corr.numpy(),
        "firing_rate": firing_rate.numpy(),
        "mean_activation": mean_activation.numpy(),
        "max_activation": max_x_activation.numpy(),
        "active": active_mask.numpy(),
        "survived_>0.9": ((corr > 0.9) & active_mask).numpy(),
        "damaged_<0.5": ((corr < 0.5) & active_mask).numpy(),
    })

    return summary, per_feature

# %% Cell 15
# =========================
# Run one condition and save
# =========================

all_summaries = []
per_feature_files = {}

def save_condition_outputs(condition_slug, summary, per_feature):
    summary_path = OUTPUT_DIR / f"{condition_slug}_summary.csv"
    pf_path = OUTPUT_DIR / f"{condition_slug}_per_feature.csv"

    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    per_feature.to_csv(pf_path, index=False)

    per_feature_files[condition_slug] = str(pf_path)

    with open(OUTPUT_DIR / "phase3_per_feature_files.json", "w") as f:
        json.dump(per_feature_files, f, indent=2)

    pd.DataFrame(all_summaries).to_csv(OUTPUT_DIR / "phase3_summary_partial.csv", index=False)

    print("Saved:", summary_path)
    print("Saved:", pf_path)

def add_summary(summary):
    all_summaries.append(summary)
    print("\nCurrent summary:")
    display(pd.DataFrame(all_summaries))

# %% Cell 16
# =========================
# FP16 baseline
# =========================

print(f"=== Layer {LAYER}: FP16 perplexity ===")
ppl_fp16, loss_fp16 = compute_perplexity(model_ref, tokens_2d, desc="FP16 perplexity")
print(f"FP16 ppl={ppl_fp16:.3f}, loss={loss_fp16:.4f}")

print(f"=== Layer {LAYER}: FP16 feature baseline ===")
summary, pf = streaming_feature_metrics(
    model_a=model_ref,
    model_b=model_ref,
    sae=sae,
    tokens_2d=tokens_2d,
    hook_name=HOOK_NAME,
    condition_name="FP16 baseline",
    layer=LAYER,
)

summary.update({
    "bits": 16,
    "perplexity": ppl_fp16,
    "loss": loss_fp16,
    "ppl_delta_pct": 0.0,
    "method": "FP16",
})

add_summary(summary)
save_condition_outputs("FP16_baseline", summary, pf)

del pf
free_memory()

# %% Cell 17
# =========================
# RTN bitwidth sweep
# =========================

rtn_ppl_by_bits = {}

for bits in BITS_TO_TEST:
    condition = f"RTN INT{bits}"
    slug = f"RTN_INT{bits}"

    print(f"\n=== Layer {LAYER}: {condition} ===")

    restore_model_work()
    n_tensors, n_params = quantize_rtn_per_output_channel_tl(model_work, bits=bits)
    print(f"Quantized tensors={n_tensors}, params={n_params:,}")

    ppl_q, loss_q = compute_perplexity(model_work, tokens_2d, desc=f"{condition} perplexity")
    ppl_delta = (ppl_q / ppl_fp16 - 1) * 100
    rtn_ppl_by_bits[bits] = ppl_q

    print(f"{condition} ppl={ppl_q:.3f}, delta={ppl_delta:+.2f}%")

    summary, pf = streaming_feature_metrics(
        model_a=model_ref,
        model_b=model_work,
        sae=sae,
        tokens_2d=tokens_2d,
        hook_name=HOOK_NAME,
        condition_name=condition,
        layer=LAYER,
    )

    summary.update({
        "bits": bits,
        "perplexity": ppl_q,
        "loss": loss_q,
        "ppl_delta_pct": ppl_delta,
        "method": "RTN",
    })

    add_summary(summary)
    save_condition_outputs(slug, summary, pf)

    del pf
    restore_model_work()
    free_memory()

# %% Cell 18
# =========================
# Matched-perplexity pruning
# =========================

def pruning_search_match_target(target_ppl, search_tokens_2d, steps=PRUNE_SEARCH_STEPS):
    records = []
    lo, hi = 0.0, 0.8

    best = None

    for step in range(steps):
        sparsity = (lo + hi) / 2

        restore_model_work()
        n_tensors, n_params, n_zeroed = magnitude_prune_tl(model_work, sparsity=sparsity)

        ppl, loss = compute_perplexity(
            model_work,
            search_tokens_2d,
            desc=f"Prune search {step+1}/{steps}, sparsity={sparsity:.3f}",
        )

        rec = {
            "step": step,
            "sparsity": sparsity,
            "ppl": ppl,
            "loss": loss,
            "target_ppl": target_ppl,
            "abs_error": abs(ppl - target_ppl),
            "n_tensors": n_tensors,
            "n_params": n_params,
            "n_zeroed": n_zeroed,
        }
        records.append(rec)

        if best is None or rec["abs_error"] < best["abs_error"]:
            best = rec

        # If pruned ppl is too low, prune more.
        if ppl < target_ppl:
            lo = sparsity
        else:
            hi = sparsity

        restore_model_work()
        free_memory()

    search_df = pd.DataFrame(records)
    search_df.to_csv(OUTPUT_DIR / "pruning_match_search.csv", index=False)
    return best, search_df

if RUN_PRUNING:
    print("\n=== Matched-perplexity pruning ===")

    # Use the same calibration subset to find target INT6 calibration perplexity.
    restore_model_work()
    quantize_rtn_per_output_channel_tl(model_work, bits=PRUNE_MATCH_BITS)
    target_calib_ppl, target_calib_loss = compute_perplexity(
        model_work,
        prune_search_tokens_2d,
        desc=f"INT{PRUNE_MATCH_BITS} calibration ppl",
    )
    restore_model_work()

    print(f"Target calibration ppl from INT{PRUNE_MATCH_BITS}: {target_calib_ppl:.3f}")

    best, search_df = pruning_search_match_target(
        target_ppl=target_calib_ppl,
        search_tokens_2d=prune_search_tokens_2d,
    )

    print("Best pruning match:")
    display(pd.DataFrame([best]))

    # Apply best sparsity and evaluate on the full token budget.
    restore_model_work()
    n_tensors, n_params, n_zeroed = magnitude_prune_tl(model_work, sparsity=best["sparsity"])

    condition = f"Magnitude pruning matched INT{PRUNE_MATCH_BITS}"
    slug = f"Magnitude_pruning_matched_INT{PRUNE_MATCH_BITS}"

    ppl_prune, loss_prune = compute_perplexity(model_work, tokens_2d, desc="Pruned full perplexity")
    ppl_delta = (ppl_prune / ppl_fp16 - 1) * 100

    summary, pf = streaming_feature_metrics(
        model_a=model_ref,
        model_b=model_work,
        sae=sae,
        tokens_2d=tokens_2d,
        hook_name=HOOK_NAME,
        condition_name=condition,
        layer=LAYER,
    )

    summary.update({
        "bits": np.nan,
        "perplexity": ppl_prune,
        "loss": loss_prune,
        "ppl_delta_pct": ppl_delta,
        "method": "pruning",
        "matched_to_bits": PRUNE_MATCH_BITS,
        "sparsity": best["sparsity"],
        "n_zeroed": n_zeroed,
    })

    add_summary(summary)
    save_condition_outputs(slug, summary, pf)

    del pf
    restore_model_work()
    free_memory()
else:
    print("Skipping pruning because RUN_PRUNING=False.")

# %% Cell 19
# =========================
# Final summary and plots
# =========================

summary_df = pd.DataFrame(all_summaries)

preferred_cols = [
    "layer", "condition", "method", "bits",
    "perplexity", "ppl_delta_pct",
    "n_active_features", "mean_corr", "median_corr",
    "survived_>0.9_pct", "degraded_0.5_0.9_pct", "damaged_<0.5_pct",
    "sparsity", "matched_to_bits"
]

cols = [c for c in preferred_cols if c in summary_df.columns] + [c for c in summary_df.columns if c not in preferred_cols]
summary_df = summary_df[cols]

final_path = OUTPUT_DIR / "phase3_summary_final.csv"
summary_df.to_csv(final_path, index=False)

print("Saved final summary:", final_path)
pd.set_option("display.float_format", "{:.3f}".format)
display(summary_df)

# RTN sweep plot
rtn_df = summary_df[summary_df["method"].eq("RTN")].copy()
if len(rtn_df):
    rtn_df = rtn_df.sort_values("bits", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.plot(rtn_df["bits"], rtn_df["survived_>0.9_pct"], marker="o", label="Survived >0.9")
    plt.plot(rtn_df["bits"], rtn_df["damaged_<0.5_pct"], marker="o", label="Damaged <0.5")
    plt.gca().invert_xaxis()
    plt.xlabel("Bitwidth")
    plt.ylabel("Feature percentage")
    plt.title(f"Gemma Phase 3 RTN sweep, layer {LAYER}, {TOKEN_BUDGET:,} tokens")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase3_rtn_sweep.png", dpi=160, bbox_inches="tight")
    plt.show()

# All-conditions bar plot
plot_df = summary_df[summary_df["condition"] != "FP16 baseline"].copy()
if len(plot_df):
    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(max(10, len(plot_df) * 1.2), 5))
    plt.bar(x - width/2, plot_df["survived_>0.9_pct"], width, label="Survived >0.9")
    plt.bar(x + width/2, plot_df["damaged_<0.5_pct"], width, label="Damaged <0.5")
    plt.xticks(x, plot_df["condition"], rotation=35, ha="right")
    plt.ylabel("Feature percentage")
    plt.title(f"Gemma Phase 3 all conditions, layer {LAYER}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase3_all_conditions_bar.png", dpi=160, bbox_inches="tight")
    plt.show()

# %% Cell 20
# =========================
# Inspect top disrupted features
# =========================

# Choose the first condition with damaged features, else choose lowest survival RTN condition.
summary_df = pd.read_csv(OUTPUT_DIR / "phase3_summary_final.csv")

candidate = summary_df[
    (summary_df["condition"] != "FP16 baseline") &
    (summary_df["damaged_<0.5_pct"] > 0)
]

if len(candidate):
    target_condition = candidate.sort_values("damaged_<0.5_pct", ascending=False).iloc[0]["condition"]
else:
    target_condition = summary_df[summary_df["condition"] != "FP16 baseline"].sort_values("survived_>0.9_pct").iloc[0]["condition"]

# Find matching per-feature file.
with open(OUTPUT_DIR / "phase3_per_feature_files.json", "r") as f:
    pf_files = json.load(f)

slug_guess = target_condition.replace(" ", "_").replace("/", "_")
print("Target condition:", target_condition)
print("Available per-feature files:")
for k, v in pf_files.items():
    print(" ", k, "->", v)

# Load the file by matching condition in contents if needed.
pf_path = None
for k, v in pf_files.items():
    try:
        tmp = pd.read_csv(v, nrows=5)
        if "condition" in tmp.columns and tmp["condition"].iloc[0] == target_condition:
            pf_path = v
            break
    except Exception:
        pass

if pf_path is None:
    print("Could not auto-find per-feature file. Pick manually from list above.")
else:
    pf = pd.read_csv(pf_path)
    top = pf[pf["active"] == True].sort_values("corr").head(30)
    display(top[[
        "feature_id", "corr", "firing_rate", "mean_activation",
        "max_activation", "survived_>0.9", "damaged_<0.5"
    ]])
    top.to_csv(OUTPUT_DIR / f"top_disrupted_{slug_guess}.csv", index=False)
    print("Saved top disrupted:", OUTPUT_DIR / f"top_disrupted_{slug_guess}.csv")

# %% Cell 21
import pandas as pd
import glob, os

for f in sorted(glob.glob("phase3_gemma_outputs_3b_full_500k_L12/*summary*.csv")):
    print(f, os.path.getsize(f) / 1e6, "MB")

