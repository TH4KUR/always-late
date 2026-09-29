"""Extended precondition check: language-selective features across a small,
tier-representative language set (not the full 27-language panel -- this is
still the bounded, coarse check from Section 8's W0 item 3, just widened
from a single Kannada/English pair now that GPU makes it cheap).

One language per resource tier, plus English as the high-resource anchor,
matching examples from the proposal's own tier table (Section 5):
  high: eng (anchor), mid: hin, low: kan, low: tel

This is NOT the pre-registered selectivity score S(f) (Section 4.1), which
needs the full 27-language equal-budget panel and 100-draw bootstrap
stability check. It answers a narrower question: does the same >=5x
firing-rate-ratio precondition check (that passed for Kannada vs. English
alone) hold up across a slightly broader, tier-spanning set?
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.pipeline import ModelConfig, ResidualCapture, encode_texts, load_model_and_tokenizer, load_sae
from src.survival import firing_rate

REPO_ROOT = Path(__file__).resolve().parent.parent
FLORES_DEV = REPO_ROOT / "data" / "flores200_dataset" / "dev"
N_SENTENCES = 80
LANGS = {
    "eng_Latn": "high",
    "hin_Deva": "mid",
    "kan_Knda": "low",
    "tel_Telu": "low",
}


def load_dev_sentences(lang_file: str, n: int) -> list[str]:
    lines = (FLORES_DEV / lang_file).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()][:n]


def main():
    log = {"stage": "multilang_sanity_check", "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(device=device, dtype=torch.bfloat16)
    log["device"] = device
    log["languages"] = LANGS

    print(f"[1/3] Loading model + SAE on {device}...")
    model, tokenizer = load_model_and_tokenizer(cfg)
    sae = load_sae(cfg)
    capture = ResidualCapture(model, cfg.layer)

    print(f"[2/3] Encoding {N_SENTENCES} FLORES dev sentences per language ({list(LANGS)})...")
    rates = {}
    n_tokens = {}
    t0 = time.time()
    for lang_file in LANGS:
        sents = load_dev_sentences(f"{lang_file}.dev", N_SENTENCES)
        feats = encode_texts(sents, model, tokenizer, sae, capture, device).numpy()
        rates[lang_file] = firing_rate(feats)
        n_tokens[lang_file] = int(feats.shape[0])
    log["encode_seconds"] = round(time.time() - t0, 1)
    log["n_tokens_per_language"] = n_tokens

    print("[3/3] Pairwise selectivity check (>=5x firing-rate ratio, floor 1e-3)...")
    floor = 1e-3
    pairwise = {}
    for a, b in combinations(LANGS, 2):
        ra, rb = rates[a], rates[b]
        a_sel = int(((ra > floor) & (ra > 5 * (rb + 1e-6))).sum())
        b_sel = int(((rb > floor) & (rb > 5 * (ra + 1e-6))).sum())
        pairwise[f"{a}_vs_{b}"] = {f"{a}_selective": a_sel, f"{b}_selective": b_sel}
        print(f"    {a} vs {b}: {a}-selective={a_sel}  {b}-selective={b_sel}")

    log["pairwise_selective_feature_counts"] = pairwise
    log["n_features_total"] = int(next(iter(rates.values())).shape[0])
    all_pass = all(
        any(v > 0 for v in pair.values()) for pair in pairwise.values()
    )
    log["verdict"] = (
        "PASS: every language pair shows a nonzero selective-feature population"
        if all_pass
        else "PARTIAL/FAIL: at least one language pair showed no selective features either direction"
    )
    log["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    out_path = REPO_ROOT / "reports" / "multilang_sanity_check_result.json"
    out_path.write_text(json.dumps(log, indent=2))
    print(f"\n{log['verdict']}\nWritten to {out_path}")


if __name__ == "__main__":
    main()
