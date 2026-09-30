"""Phase 1 Core Quantization Survival with Quintile Matching (RQ1 - RQ2).

Formal implementation of Section 4.3 (Phase 1: Compression-Survival by Selectivity and Resource Level):
- Evaluates on held-out FLORES-200 `devtest` tokens (strict train/test separation from `dev` tagging).
- Applies RTN fake quantization across bit-widths (focusing on INT7 and INT6).
- Computes per-feature Pearson survival against FP16 baseline on tokens from the feature's peak language.
- Critical Control: Matches selective features against universal features by firing-rate quintile
  to isolate language-selectivity from general feature rarity (Borobia et al., 2026).
- Aggregates survival across Resource Tiers (High, Mid, Low) and Script Families.
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

from src.data_utils import PANEL_27, build_equalized_token_chunks, load_sentences
from src.pipeline import ModelConfig, ResidualCapture, encode_texts, load_model_and_tokenizer, load_sae
from src.rtn_quant import quantized_model
from src.survival import (
    active_feature_mask,
    classify_survival,
    firing_rate,
    pearson_corr_per_feature,
    stratify_by_firing_rate_quintiles,
)


def run_survival_pipeline(
    selectivity_report_path: str = "reports/selectivity_results.json",
    bitwidths: list[int] | None = None,
    tokens_per_lang: int = 1500,
    chunk_len: int = 128,
    device: str | None = None,
    output_file: str = "reports/phase1_survival_results.json",
):
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if bitwidths is None:
        bitwidths = [7, 6]  # Pre-registered inferential bitwidths

    sel_path = REPO_ROOT / selectivity_report_path
    if not sel_path.exists():
        raise FileNotFoundError(
            f"Selectivity report not found at {sel_path}. Run scripts/run_phase1_selectivity_pipeline.py first."
        )

    sel_data = json.loads(sel_path.read_text(encoding="utf-8"))
    languages = list(sel_data["languages"].keys())

    print(f"=== Running Phase 1 Survival with Quintile Matching ===")
    print(f"Device: {device} | Languages: {len(languages)} | Bitwidths: {bitwidths} | Held-out: devtest")

    cfg = ModelConfig(device=device, dtype=torch.bfloat16)
    print("\n[1/4] Loading model, tokenizer, and Gemma Scope SAE...")
    model, tokenizer = load_model_and_tokenizer(cfg)
    sae = load_sae(cfg)
    capture = ResidualCapture(model, cfg.layer)

    print("\n[2/4] Encoding FP16 baseline on held-out FLORES devtest...")
    devtest_fp_acts = {}
    for lang_code in languages:
        sentences = load_sentences(lang_code, split="devtest")
        chunks = build_equalized_token_chunks(
            sentences, tokenizer, target_tokens=tokens_per_lang, chunk_len=chunk_len
        )
        feats = encode_texts(chunks, model, tokenizer, sae, capture, device).numpy()
        devtest_fp_acts[lang_code] = feats
        print(f"  - {lang_code}: {feats.shape[0]} tokens encoded (FP16)")

    # Pool overall firing rates to identify active features and compute quintiles
    all_fp_tokens = np.concatenate(list(devtest_fp_acts.values()), axis=0)
    baseline_rates = firing_rate(all_fp_tokens)
    is_active = baseline_rates > 0.001
    quintiles = stratify_by_firing_rate_quintiles(baseline_rates, is_active)

    # Classify features from selectivity report
    selective_feature_ids = set()
    feature_to_peak_lang = {}
    feature_to_tier = {}
    for lang_code, info in sel_data["languages"].items():
        for fid in info.get("top5_selective_feature_ids", []):
            selective_feature_ids.add(fid)
            feature_to_peak_lang[fid] = lang_code
            feature_to_tier[fid] = info["tier"]

    # Also build universal feature candidate pool (active, but not selective)
    universal_candidate_ids = [
        f for f in range(len(baseline_rates))
        if is_active[f] and f not in selective_feature_ids
    ]

    print(f"\n[3/4] Active features: {is_active.sum()} | Selective features tracked: {len(selective_feature_ids)}")

    print("\n[4/4] Evaluating RTN Fake Quantization across bit-widths...")
    survival_by_bitwidth = {}

    for b in bitwidths:
        t0 = time.time()
        print(f"\n--- Quantizing to INT{b} ---")
        devtest_quant_acts = {}
        with quantized_model(model, bits=b):
            for lang_code in languages:
                sentences = load_sentences(lang_code, split="devtest")
                chunks = build_equalized_token_chunks(
                    sentences, tokenizer, target_tokens=tokens_per_lang, chunk_len=chunk_len
                )
                feats = encode_texts(chunks, model, tokenizer, sae, capture, device).numpy()
                devtest_quant_acts[lang_code] = feats

        # Compute per-feature Pearson correlation on its designated peak language
        sel_corrs = []
        sel_q_indices = []
        sel_tiers = []

        for fid in sorted(selective_feature_ids):
            lang = feature_to_peak_lang[fid]
            fp_vec = devtest_fp_acts[lang][:, fid:fid+1]
            quant_vec = devtest_quant_acts[lang][:, fid:fid+1]
            r = float(pearson_corr_per_feature(fp_vec, quant_vec)[0])
            sel_corrs.append(r)
            sel_q_indices.append(quintiles[fid])
            sel_tiers.append(feature_to_tier[fid])

        # Sample matched universal features within identical quintiles
        matched_univ_corrs = []
        rng = np.random.default_rng(42)
        for q in sel_q_indices:
            q_candidates = [
                f for f in universal_candidate_ids if quintiles[f] == q
            ]
            if q_candidates:
                matched_f = rng.choice(q_candidates)
                # For universal features, evaluate on pooled language tokens
                fp_vec = all_fp_tokens[:, matched_f:matched_f+1]
                quant_pooled = np.concatenate(list(devtest_quant_acts.values()), axis=0)[:, matched_f:matched_f+1]
                r_univ = float(pearson_corr_per_feature(fp_vec, quant_pooled)[0])
                matched_univ_corrs.append(r_univ)

        sel_arr = np.array(sel_corrs)
        univ_arr = np.array(matched_univ_corrs) if matched_univ_corrs else np.array([])

        sel_labels = classify_survival(sel_arr)
        univ_labels = classify_survival(univ_arr) if len(univ_arr) else np.array([])

        # Tier breakdowns
        tier_survival = {}
        for tier in ["high", "mid", "low"]:
            tier_mask = np.array([t == tier for t in sel_tiers])
            if tier_mask.any():
                sub_r = sel_arr[tier_mask]
                sub_labels = classify_survival(sub_r)
                tier_survival[tier] = {
                    "count": int(tier_mask.sum()),
                    "survived_pct": round(float((sub_labels == "survived").mean() * 100), 2),
                    "damaged_pct": round(float((sub_labels == "damaged").mean() * 100), 2),
                    "median_pearson_r": round(float(np.nanmedian(sub_r)), 4),
                }

        survival_by_bitwidth[f"INT{b}"] = {
            "elapsed_seconds": round(time.time() - t0, 1),
            "selective_features": {
                "count": len(sel_arr),
                "survived_pct": round(float((sel_labels == "survived").mean() * 100), 2),
                "damaged_pct": round(float((sel_labels == "damaged").mean() * 100), 2),
                "median_pearson_r": round(float(np.nanmedian(sel_arr)), 4),
            },
            "quintile_matched_universal": {
                "count": len(univ_arr),
                "survived_pct": round(float((univ_labels == "survived").mean() * 100), 2) if len(univ_arr) else None,
                "damaged_pct": round(float((univ_labels == "damaged").mean() * 100), 2) if len(univ_arr) else None,
                "median_pearson_r": round(float(np.nanmedian(univ_arr)), 4) if len(univ_arr) else None,
            },
            "by_resource_tier": tier_survival,
        }

        print(f"INT{b} Results:")
        print(f"  Selective: Survived={survival_by_bitwidth[f'INT{b}']['selective_features']['survived_pct']}% | "
              f"Damaged={survival_by_bitwidth[f'INT{b}']['selective_features']['damaged_pct']}% | "
              f"Median r={survival_by_bitwidth[f'INT{b}']['selective_features']['median_pearson_r']}")
        if len(univ_arr):
            print(f"  Universal (Matched): Survived={survival_by_bitwidth[f'INT{b}']['quintile_matched_universal']['survived_pct']}% | "
                  f"Damaged={survival_by_bitwidth[f'INT{b}']['quintile_matched_universal']['damaged_pct']}% | "
                  f"Median r={survival_by_bitwidth[f'INT{b}']['quintile_matched_universal']['median_pearson_r']}")

    report = {
        "stage": "Phase1_Survival_Quintile_Matching",
        "started_at": start_time,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "n_languages": len(languages),
        "held_out_split": "devtest",
        "results_by_bitwidth": survival_by_bitwidth,
    }

    out_path = REPO_ROOT / output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWritten Phase 1 Survival Results to {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Survival Evaluation")
    parser.add_argument("--report", type=str, default="reports/selectivity_results.json")
    parser.add_argument("--bitwidths", type=str, default="7,6")
    parser.add_argument("--tokens", type=int, default=1500)
    parser.add_argument("--output", type=str, default="reports/phase1_survival_results.json")
    args = parser.parse_args()

    bits = [int(b.strip()) for b in args.bitwidths.split(",") if b.strip()]
    run_survival_pipeline(
        selectivity_report_path=args.report,
        bitwidths=bits,
        tokens_per_lang=args.tokens,
        output_file=args.output,
    )
