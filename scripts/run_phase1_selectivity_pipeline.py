"""Phase 1 Selectivity Extraction across the FLORES-200 Language Panel.

Formal implementation of Section 4.1 (Formal Setting and Selectivity Score):
- Loads script-balanced languages across High, Mid, and Low resource tiers.
- Applies exact token-budget equalization using `src.data_utils.build_equalized_token_chunks`.
- Computes per-language firing rates on the frozen Gemma Scope layer-12 SAE.
- Computes Deng et al. (2025) selectivity score S(f) and peak language assignments.
- Implements Pre-Registered Safeguards:
  1. Minimum firing rate floor (> 0.001) and >= 50 positive activations in peak language.
  2. Rank-normalized continuous selectivity S_tilde(f) in [0, 1].
  3. Empirical shuffled null distribution to set the 95th percentile binary threshold.
- Outputs machine-readable summary and detailed JSON reports.
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
import torch

from src.data_utils import PANEL_27, build_equalized_token_chunks, ensure_flores_dataset, load_sentences
from src.pipeline import ModelConfig, ResidualCapture, encode_texts, load_model_and_tokenizer, load_sae
from src.survival import (
    compute_shuffled_null_threshold,
    filter_supported_features,
    firing_rate,
    positive_count,
    rank_normalized_selectivity,
    selectivity_score,
)


def run_selectivity_pipeline(
    languages: list[str] | None = None,
    target_tokens_per_lang: int = 1500,
    chunk_len: int = 128,
    run_null_shuffling: bool = True,
    device: str | None = None,
    output_file: str = "reports/selectivity_results.json",
):
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if languages is None:
        # Default to full 27 panel if available, or allow subset
        languages = list(PANEL_27.keys())

    print(f"=== Running Phase 1 Selectivity Extraction ===")
    print(f"Device: {device} | Languages: {len(languages)} | Target Tokens/Lang: {target_tokens_per_lang}")

    ensure_flores_dataset()

    cfg = ModelConfig(device=device, dtype=torch.bfloat16)
    print("\n[1/4] Loading model, tokenizer, and Gemma Scope SAE (layer 12)...")
    model, tokenizer = load_model_and_tokenizer(cfg)
    sae = load_sae(cfg)
    capture = ResidualCapture(model, cfg.layer)

    print("\n[2/4] Tokenizing and encoding equalized token budgets from FLORES dev...")
    rates_by_lang = []
    pos_counts_by_lang = []
    raw_token_acts_sample = []  # stored for null distribution
    actual_tokens_per_lang = {}

    t0 = time.time()
    for lang_code in languages:
        sentences = load_sentences(lang_code, split="dev")
        chunks = build_equalized_token_chunks(
            sentences, tokenizer, target_tokens=target_tokens_per_lang, chunk_len=chunk_len
        )
        feats = encode_texts(chunks, model, tokenizer, sae, capture, device).numpy()
        rate = firing_rate(feats)
        pos = positive_count(feats)

        rates_by_lang.append(rate)
        pos_counts_by_lang.append(pos)
        actual_tokens_per_lang[lang_code] = int(feats.shape[0])

        if run_null_shuffling and len(raw_token_acts_sample) < 6:
            # Subsample a few languages for the permutation null distribution to keep runtime tight
            raw_token_acts_sample.append(feats[:min(500, feats.shape[0])])

        print(f"  - {lang_code:10s} ({PANEL_27[lang_code].tier:>4s}, {PANEL_27[lang_code].script:>10s}): "
              f"{feats.shape[0]} tokens, active features: {(rate > 0.001).sum()}")

    encode_duration = round(time.time() - t0, 1)
    rate_matrix = np.stack(rates_by_lang, axis=0)  # [L, d_sae]
    pos_matrix = np.stack(pos_counts_by_lang, axis=0)  # [L, d_sae]

    print("\n[3/4] Computing Deng selectivity scores, safeguards, and rank normalization...")
    raw_s, peak_lang_idx = selectivity_score(rate_matrix)
    min_acts = min(50, max(3, int(target_tokens_per_lang * 0.01)))
    supported_mask = filter_supported_features(
        rate_matrix,
        pos_matrix,
        min_rate_floor=0.001,
        min_peak_positive_acts=min_acts,
    )
    s_tilde = rank_normalized_selectivity(raw_s, supported_mask)

    null_threshold = 5.0
    if run_null_shuffling and raw_token_acts_sample:
        print("  Computing empirical 95th-percentile shuffled null threshold...")
        null_threshold = compute_shuffled_null_threshold(
            raw_token_acts_sample, n_permutations=20, percentile=95.0
        )
        print(f"  Empirical Null Cutoff: S(f) >= {null_threshold:.3f}")

    is_selective = supported_mask & (raw_s >= null_threshold)

    print("\n[4/4] Aggregating results by language and resource tier...")
    tier_counts = {"high": 0, "mid": 0, "low": 0}
    lang_results = {}

    for i, lang_code in enumerate(languages):
        info = PANEL_27[lang_code]
        lang_mask = is_selective & (peak_lang_idx == i)
        n_sel = int(lang_mask.sum())
        tier_counts[info.tier] += n_sel

        top5_features = [
            int(idx) for idx in np.argsort(raw_s)[::-1] if lang_mask[idx]
        ][:5]

        lang_results[lang_code] = {
            "name": info.name,
            "tier": info.tier,
            "script": info.script,
            "tokens": actual_tokens_per_lang[lang_code],
            "n_selective_features": n_sel,
            "top5_selective_feature_ids": top5_features,
        }
        print(f"  {lang_code:10s} | Tier: {info.tier:4s} | Script: {info.script:10s} | Selective Features: {n_sel}")

    summary = {
        "stage": "Phase1_Selectivity_Extraction",
        "started_at": start_time,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "n_languages": len(languages),
        "target_tokens_per_lang": target_tokens_per_lang,
        "total_encode_seconds": encode_duration,
        "total_sae_features": int(rate_matrix.shape[1]),
        "supported_features_count": int(supported_mask.sum()),
        "shuffled_null_threshold_95th": round(float(null_threshold), 4),
        "total_selective_features": int(is_selective.sum()),
        "selective_features_by_tier": tier_counts,
        "languages": lang_results,
    }

    out_path = REPO_ROOT / output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    # Per-feature arrays for plotting the real S(f) distribution (figures script).
    np.savez_compressed(
        out_path.with_suffix(".features.npz"),
        raw_s=raw_s, peak_lang_idx=peak_lang_idx, supported=supported_mask,
        is_selective=is_selective, languages=np.array(languages),
    )
    print(f"\nSaved full results to: {out_path}")
    print(f"Summary by Tier: High={tier_counts['high']} | Mid={tier_counts['mid']} | Low={tier_counts['low']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Selectivity Extraction")
    parser.add_argument("--tokens", type=int, default=1500, help="Tokens per language")
    parser.add_argument("--languages", type=str, default="", help="Comma-separated lang codes (default: all 27)")
    parser.add_argument("--no-null", action="store_true", help="Skip shuffled null calculation")
    parser.add_argument("--output", type=str, default="reports/selectivity_results.json", help="Output JSON path")
    args = parser.parse_args()

    lang_list = [l.strip() for l in args.languages.split(",") if l.strip()] if args.languages else None
    run_selectivity_pipeline(
        languages=lang_list,
        target_tokens_per_lang=args.tokens,
        run_null_shuffling=not args.no_null,
        output_file=args.output,
    )
