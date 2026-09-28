"""Reduced-scale quantization-survival pipeline validation on Gemma-2-2B.

This is NOT the formal Phase-0 replication gate (Section 4.2): that requires
Duan (2026)'s exact corpus, token budget, and pinned numeric bands, and is
scheduled for W2 of the project timeline. What this script does instead is
exercise the *identical* algorithmic pipeline Phase 0/1 need -- RTN
quantization + per-feature Pearson survival + the firing-rate activity rule
-- end-to-end on real data (WikiText-2, public/ungated), at a reduced token
budget suited to CPU-only execution. It answers a narrower, still-real
question: does our from-scratch RTN + survival code reproduce the
*qualitative* direction Duan (2026) report (feature damage increases as
bit-width decreases), before we invest in the full-budget formal replication?
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from datasets import load_dataset

from src.pipeline import ModelConfig, ResidualCapture, load_model_and_tokenizer, load_sae
from src.rtn_quant import quantized_model
from src.survival import active_feature_mask, classify_survival, pearson_corr_per_feature

BIT_WIDTHS = [8, 7, 6, 5, 4]
TARGET_TOKENS = 3000
CHUNK_LEN = 256  # matches the proposal's chunked protocol (L=256)


def load_wikitext_chunks(tokenizer, target_tokens: int, chunk_len: int) -> list[str]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    chunks, buf, n_tokens = [], [], 0
    for row in ds:
        text = row["text"].strip()
        if len(text) < 20:  # skip blank lines / section headers
            continue
        buf.append(text)
        n_tokens += len(tokenizer.encode(text))
        if n_tokens >= chunk_len:
            chunks.append(" ".join(buf))
            buf, n_tokens = [], 0
        if sum(len(tokenizer.encode(c)) for c in chunks) >= target_tokens:
            break
    return chunks


@torch.no_grad()
def encode_chunks(chunks, model, tokenizer, sae, capture, device) -> torch.Tensor:
    feats = []
    for c in chunks:
        enc = tokenizer(c, return_tensors="pt", truncation=True, max_length=CHUNK_LEN)
        enc = {k: v.to(device) for k, v in enc.items()}
        model(**enc)
        resid = capture.activation[0]
        feats.append(sae.encode(resid).float().cpu())
    return torch.cat(feats, dim=0)


def main():
    log = {"stage": "quantization_survival_demo", "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    cfg = ModelConfig(device=device, dtype=dtype)
    log["device"] = device

    print("[1/4] Loading model, tokenizer, SAE...")
    model, tokenizer = load_model_and_tokenizer(cfg)
    sae = load_sae(cfg)
    capture = ResidualCapture(model, cfg.layer)

    print(f"[2/4] Building a ~{TARGET_TOKENS}-token WikiText-2 sample...")
    chunks = load_wikitext_chunks(tokenizer, TARGET_TOKENS, CHUNK_LEN)
    log["n_chunks"] = len(chunks)

    print(f"[3/4] FP-baseline encode ({len(chunks)} chunks)...")
    t0 = time.time()
    baseline = encode_chunks(chunks, model, tokenizer, sae, capture, device).numpy()
    log["baseline_encode_seconds"] = round(time.time() - t0, 1)
    log["n_tokens"] = int(baseline.shape[0])

    active_mask = active_feature_mask(baseline, threshold=0.001)
    log["n_active_features_baseline"] = int(active_mask.sum())
    print(f"    active features (firing rate > 0.001): {active_mask.sum()} / {baseline.shape[1]}")

    print("[4/4] RTN sweep across bit-widths...")
    results = {}
    for bits in BIT_WIDTHS:
        t0 = time.time()
        with quantized_model(model, bits) as targets:
            quant_acts = encode_chunks(chunks, model, tokenizer, sae, capture, device).numpy()
        r = pearson_corr_per_feature(baseline[:, active_mask], quant_acts[:, active_mask])
        labels = classify_survival(r)
        survived = int((labels == "survived").sum())
        damaged = int((labels == "damaged").sum())
        n_active = int(active_mask.sum())
        results[f"INT{bits}"] = {
            "n_quantized_linear_modules": len(targets),
            "seconds": round(time.time() - t0, 1),
            "n_active_features": n_active,
            "survived_pct": round(100 * survived / n_active, 2) if n_active else None,
            "damaged_pct": round(100 * damaged / n_active, 2) if n_active else None,
            "median_pearson_r": float(np.nanmedian(r)),
        }
        print(f"    INT{bits}: survived={results[f'INT{bits}']['survived_pct']}% "
              f"damaged={results[f'INT{bits}']['damaged_pct']}% "
              f"median_r={results[f'INT{bits}']['median_pearson_r']:.3f}")

    log["results_by_bitwidth"] = results
    log["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    out_path = REPO_ROOT / "reports" / "quantization_survival_demo_result.json"
    out_path.write_text(json.dumps(log, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
