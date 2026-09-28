"""W0 feasibility spike #3 (anlp_revised.txt, Section 8, W0 item 3):

'60-minute sanity run: pinned Gemma Scope SAE on FLORES Kannada vs English
dev sentences; verify a nonzero population of supported language-selective
features exists at layer 12.'

This does NOT compute the full pre-registered selectivity score S(f) with
bootstrap stability (that needs the full 27-language panel, Section 4.1) --
it is a cheap, bounded precondition check: do *any* layer-12 SAE features
fire preferentially on Kannada vs. English at all, on a real model + real
SAE + real FLORES-200 data. A "no" here would be a serious problem for the
whole project; a "yes" licenses moving on to the full panel.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.pipeline import ModelConfig, ResidualCapture, encode_texts, load_model_and_tokenizer, load_sae
from src.survival import firing_rate

FLORES_DEV = REPO_ROOT / "data" / "flores200_dataset" / "dev"
N_SENTENCES = 40  # small on purpose: CPU fallback, this is a precondition check not a full run


def load_dev_sentences(lang_file: str, n: int) -> list[str]:
    lines = (FLORES_DEV / lang_file).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()][:n]


def main():
    log = {"stage": "W0_sanity_check", "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 even on CPU: fp32 would need ~10.4GB just for Gemma-2-2B's weights,
    # which OOM-killed an earlier run on this 15GB-RAM machine (~8GB free).
    dtype = torch.bfloat16
    cfg = ModelConfig(device=device, dtype=dtype)
    log["device"] = device
    log["dtype"] = str(dtype)

    print(f"[1/4] Loading model + tokenizer ({cfg.model_id}) on {device}...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(cfg)
    log["model_load_seconds"] = round(time.time() - t0, 1)

    print(f"[2/4] Loading Gemma Scope SAE (layer {cfg.layer})...")
    t0 = time.time()
    sae = load_sae(cfg)
    capture = ResidualCapture(model, cfg.layer)
    log["sae_load_seconds"] = round(time.time() - t0, 1)
    log["d_model"] = sae.W_enc.shape[0]
    log["d_sae"] = sae.W_enc.shape[1]

    print(f"[3/4] Encoding {N_SENTENCES} Kannada + {N_SENTENCES} English FLORES dev sentences...")
    kan_sents = load_dev_sentences("kan_Knda.dev", N_SENTENCES)
    eng_sents = load_dev_sentences("eng_Latn.dev", N_SENTENCES)
    log["n_kan_sentences"] = len(kan_sents)
    log["n_eng_sentences"] = len(eng_sents)

    t0 = time.time()
    kan_feats = encode_texts(kan_sents, model, tokenizer, sae, capture, device)
    eng_feats = encode_texts(eng_sents, model, tokenizer, sae, capture, device)
    log["encode_seconds"] = round(time.time() - t0, 1)
    log["n_kan_tokens"] = kan_feats.shape[0]
    log["n_eng_tokens"] = eng_feats.shape[0]

    print("[4/4] Computing per-language firing rates and selectivity...")
    kan_rate = firing_rate(kan_feats.numpy())
    eng_rate = firing_rate(eng_feats.numpy())

    # A feature counts as "Kannada-selective" for this bounded check if it
    # fires materially more on Kannada tokens than English tokens (>=5x,
    # with an absolute floor so we don't count noise on near-zero rates).
    floor = 1e-3
    kan_selective = (kan_rate > floor) & (kan_rate > 5 * (eng_rate + 1e-6))
    eng_selective = (eng_rate > floor) & (eng_rate > 5 * (kan_rate + 1e-6))

    log["n_features_total"] = int(kan_rate.shape[0])
    log["n_kannada_selective_features"] = int(kan_selective.sum())
    log["n_english_selective_features"] = int(eng_selective.sum())
    log["top5_kannada_selective_feature_ids"] = [
        int(i) for i in kan_rate.argsort()[::-1] if kan_selective[i]
    ][:5]

    log["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    log["verdict"] = (
        "PASS: nonzero language-selective feature population found at layer 12"
        if log["n_kannada_selective_features"] > 0
        else "FAIL: no Kannada-selective features found -- investigate before proceeding"
    )

    out_path = REPO_ROOT / "reports" / "w0_sanity_check_result.json"
    out_path.write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
