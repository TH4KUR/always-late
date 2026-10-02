"""Associational Behavioral Link: Internal Feature Damage vs. Language-Level Cross-Entropy Loss Degradation.

Implements STEP 6 of the Mid-Submission priority plan:
- Evaluates on held-out FLORES-200 devtest parallel sentences.
- Computes per-language cross-entropy loss:
    Delta L_l = L_compressed,l - L_FP16,l
- Gathers per-language feature-damage metrics:
    - selective_feature_damage: fraction of language-selective features damaged (r <= 0.5)
    - overall_feature_damage: fraction of all active features damaged
    - token_fertility: ratio of subword tokens to whitespace words
- Fits ordinary least squares / rank correlations:
    Delta L_l ~ selective_damage + overall_damage + token_fertility
- Tests whether language-selective feature damage explains language degradation
  better than overall feature damage alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
import torch

from src.data_utils import PANEL_27, build_equalized_token_chunks, load_sentences
from src.pipeline import ModelConfig, load_model_and_tokenizer
from src.rtn_quant import quantized_model


def compute_cross_entropy_loss(model, tokenizer, texts: list[str], device: str, max_length: int = 128) -> float:
    """Computes mean next-token cross-entropy loss over a collection of text chunks."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = enc["input_ids"].clone()
            outputs = model(**enc, labels=labels)
            loss = outputs.loss
            n_tok = enc["input_ids"].numel()
            total_loss += float(loss.item()) * n_tok
            total_tokens += n_tok

    return float(total_loss / max(total_tokens, 1))


def compute_token_fertility(tokenizer, sentences: list[str]) -> float:
    """Computes subword fertility: tokens / words."""
    total_words = 0
    total_tokens = 0
    for s in sentences:
        words = len(s.strip().split())
        tokens = len(tokenizer.encode(s, add_special_tokens=False))
        total_words += words
        total_tokens += tokens
    return float(total_tokens / max(total_words, 1))


def run_behavioral_link(
    survival_report_path: str = "reports/phase1_survival_results.json",
    bitwidths: list[int] | None = None,
    tokens_per_lang: int = 1000,
    device: str | None = None,
    output_path: str = "reports/behavioral_loss_link_results.json",
):
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if bitwidths is None:
        bitwidths = [7, 6]

    print(f"=== Running Associational Behavioral Loss Link ===")
    print(f"Device: {device} | Bitwidths: {bitwidths} | Split: FLORES devtest")

    cfg = ModelConfig(device=device, dtype=torch.bfloat16)
    print("\n[1/4] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Load survival report if available to extract selective/overall damage
    surv_path = REPO_ROOT / survival_report_path
    surv_data = json.loads(surv_path.read_text(encoding="utf-8"))
    surv_by_bw = surv_data["results_by_bitwidth"]

    # Same 27 languages as the survival report (PANEL_27 also holds a cmn_Hans alias).
    test_langs = list(surv_by_bw[f"INT{bitwidths[0]}"]["by_language"].keys())

    print(f"\n[2/4] Measuring FP16 Baseline Loss and Token Fertility across {len(test_langs)} languages...")
    fp16_losses = {}
    fertility_map = {}
    eval_chunks = {}

    for code in test_langs:
        sentences = load_sentences(code, split="devtest")
        chunks = build_equalized_token_chunks(sentences, tokenizer, target_tokens=tokens_per_lang)
        eval_chunks[code] = chunks
        fp16_loss = compute_cross_entropy_loss(model, tokenizer, chunks, device)
        fertility = compute_token_fertility(tokenizer, sentences)
        fp16_losses[code] = fp16_loss
        fertility_map[code] = fertility
        print(f"  {code:10s} | FP16 Loss: {fp16_loss:.4f} | Fertility: {fertility:.2f}")

    print("\n[3/4] Measuring Compressed Loss (Delta L) across Bitwidths...")
    delta_loss_by_bw = {f"INT{b}": {} for b in bitwidths}

    for b in bitwidths:
        print(f"--- Quantizing to INT{b} ---")
        with quantized_model(model, bits=b):
            for code in test_langs:
                quant_loss = compute_cross_entropy_loss(model, tokenizer, eval_chunks[code], device)
                delta_l = quant_loss - fp16_losses[code]
                delta_loss_by_bw[f"INT{b}"][code] = delta_l
                print(f"  {code:10s} | INT{b} Loss: {quant_loss:.4f} | Delta L: {delta_l:+.4f}")

    print("\n[4/4] Associational Analysis: Feature Damage vs Delta Loss...")
    records = []
    for code in test_langs:
        info = PANEL_27[code]
        rec = {
            "language": code,
            "resource_tier": info.tier,
            "script": info.script,
            "token_fertility": round(fertility_map[code], 3),
            "fp16_loss": round(fp16_losses[code], 4),
        }
        for b in bitwidths:
            bw_key = f"INT{b}"
            rec[f"{bw_key}_delta_loss"] = round(delta_loss_by_bw[bw_key][code], 4)
            # Measured feature damage from the Phase-1 survival report (same devtest split).
            lang_surv = surv_by_bw[bw_key]["by_language"][code]
            rec[f"{bw_key}_selective_damage"] = lang_surv["selective_damaged_frac"]
            rec[f"{bw_key}_overall_damage"] = lang_surv["overall_damaged_frac"]
            rec[f"{bw_key}_selective_median_r"] = lang_surv["selective_median_r"]
            rec[f"{bw_key}_overall_median_r"] = lang_surv["overall_median_r"]
        records.append(rec)

    df_behavior = pd.DataFrame(records)
    csv_path = REPO_ROOT / "reports" / "language_behavioral_dataset.csv"
    df_behavior.to_csv(csv_path, index=False)
    print(f"\nSaved language behavioral dataset to {csv_path}")

    # Spearman correlations
    correlations = {}
    for b in bitwidths:
        bw_key = f"INT{b}"
        dl = df_behavior[f"{bw_key}_delta_loss"]
        # Damaged fractions (r <= 0.5) are near-zero at INT7/INT6, so continuous
        # drift (1 - median Pearson r) is the informative predictor.
        predictors = {
            "selective_damage": df_behavior[f"{bw_key}_selective_damage"],
            "overall_damage": df_behavior[f"{bw_key}_overall_damage"],
            "selective_drift": 1 - df_behavior[f"{bw_key}_selective_median_r"],
            "overall_drift": 1 - df_behavior[f"{bw_key}_overall_median_r"],
            "fertility": df_behavior["token_fertility"],
        }
        correlations[bw_key] = {}
        for name, x in predictors.items():
            rho, p = stats.spearmanr(x, dl, nan_policy="omit")
            rho = float(rho) if np.isfinite(rho) else None
            p = float(p) if np.isfinite(p) else None
            correlations[bw_key][f"spearman_{name}_vs_delta_loss"] = {
                "rho": round(rho, 4) if rho is not None else None,
                "p_val": round(p, 4) if p is not None else None,
            }
            print(f"{bw_key} Spearman: {name:17s} vs Delta Loss: rho={rho} (p={p})")

    out_file = REPO_ROOT / output_path
    final_output = {
        "stage": "Behavioral_Loss_Link",
        "started_at": start_time,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_languages": len(test_langs),
        "correlations": correlations,
        "data": records,
    }
    out_file.write_text(json.dumps(final_output, indent=2))
    print(f"Results written to: {out_file}")
    return final_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Associational Behavioral Loss Link")
    parser.add_argument("--report", type=str, default="reports/phase1_survival_results_27lang_gpu.json")
    parser.add_argument("--tokens", type=int, default=1000)
    parser.add_argument("--bitwidths", type=str, default="7,6")
    parser.add_argument("--output", type=str, default="reports/behavioral_loss_link_results.json")
    args = parser.parse_args()

    bits = [int(b.strip()) for b in args.bitwidths.split(",") if b.strip()]
    run_behavioral_link(survival_report_path=args.report, tokens_per_lang=args.tokens, bitwidths=bits,
                        output_path=args.output)
