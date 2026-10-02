"""GPU-resident Phase-1 survival benchmark (all bit-widths in one shot, wandb-logged).

Same protocol as run_phase1_survival_quintile_matching.py (FLORES devtest, equalized
token budget, RTN per-channel quantization, Pearson survival r>=0.9 / damaged r<=0.5,
firing-rate-quintile-matched universal controls, rng seed 42), but:
  * batched forward passes (right-padded; causal LM so real-token activations are unchanged),
  * activations stay on the GPU as bf16 (lossless: the SAE emits bf16) and Pearson r is
    computed on the GPU only for the columns that are needed,
  * every bit-width (default 8,7,6,5,4) runs in one process; INT7/INT6 stay the
    confirmatory contrasts, the rest are descriptive,
  * tracks ALL selective features from the selectivity report (`--tracked all`, using
    the saved .features.npz) for power; `--tracked top5` reproduces the earlier design.
    Both are written to the output (`selective_features` = chosen set,
    `selective_features_top5` = earlier design), so the paper can cite either.
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

from src.data_utils import build_equalized_token_chunks, load_sentences
from src.pipeline import ModelConfig, ResidualCapture, load_model_and_tokenizer, load_sae
from src.rtn_quant import quantized_model
from src.survival import classify_survival, stratify_by_firing_rate_quintiles


@torch.no_grad()
def encode_batched(chunks, model, tokenizer, sae, capture, device, batch_size, max_length=128):
    """[total_real_tokens, d_sae] bf16 on `device`, same tokens/order as encode_texts."""
    out = []
    tokenizer.padding_side = "right"
    for i in range(0, len(chunks), batch_size):
        enc = tokenizer(chunks[i:i + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        model(**enc)
        feats = sae.encode(capture.activation)  # [B, T, d_sae] bf16
        out.append(feats[enc["attention_mask"].bool()])
    return torch.cat(out, dim=0)


def pearson_cols(a: torch.Tensor, b: torch.Tensor, col_chunk: int = 2048) -> np.ndarray:
    """Per-column Pearson r between a and b ([N, K] each); NaN if zero variance."""
    res = []
    for s in range(0, a.shape[1], col_chunk):
        x = a[:, s:s + col_chunk].float()
        y = b[:, s:s + col_chunk].float()
        x = x - x.mean(0, keepdim=True)
        y = y - y.mean(0, keepdim=True)
        den = ((x * x).sum(0) * (y * y).sum(0)).sqrt()
        r = (x * y).sum(0) / den
        r[den == 0] = float("nan")
        res.append(r.cpu())
    return torch.cat(res).numpy() if res else np.zeros(0)


def summarize(r: np.ndarray) -> dict:
    lab = classify_survival(r)
    return {
        "count": int(len(r)),
        "survived_pct": round(float((lab == "survived").mean() * 100), 2) if len(r) else None,
        "damaged_pct": round(float((lab == "damaged").mean() * 100), 2) if len(r) else None,
        "median_pearson_r": round(float(np.nanmedian(r)), 4) if len(r) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/selectivity_results_27lang.json")
    ap.add_argument("--bitwidths", default="8,7,6,5,4")
    ap.add_argument("--tokens", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--tracked", choices=["all", "top5"], default="all")
    ap.add_argument("--output", default="reports/phase1_survival_results_27lang_gpu.json")
    ap.add_argument("--wandb-project", default="always-late")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    bits = [int(b) for b in args.bitwidths.split(",")]
    sel_path = REPO_ROOT / args.report
    sel = json.loads(sel_path.read_text())
    languages = list(sel["languages"])

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=f"survival-gpu-{args.tokens}tok-{args.tracked}",
                         job_type="survival", config={**vars(args), "gpu": torch.cuda.get_device_name(0),
                                                      "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)})

    def log(d):
        d = {**d, "gpu/mem_alloc_gb": torch.cuda.memory_allocated() / 2**30,
             "gpu/mem_peak_gb": torch.cuda.max_memory_allocated() / 2**30}
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}, flush=True)
        if run:
            run.log(d)

    # Tracked selective features: id -> (peak language, tier)
    top5_ids = {}
    for lang, info in sel["languages"].items():
        for fid in info.get("top5_selective_feature_ids", []):
            top5_ids[fid] = lang
    if args.tracked == "all":
        npz = np.load(sel_path.with_suffix(".features.npz"))
        langs_npz = [str(x) for x in npz["languages"]]
        tracked = {int(f): langs_npz[int(npz["peak_lang_idx"][f])] for f in np.flatnonzero(npz["is_selective"])}
    else:
        tracked = dict(top5_ids)
    tier_of = {l: sel["languages"][l]["tier"] for l in languages}
    sel_ids = sorted(tracked)
    sel_lang = [tracked[f] for f in sel_ids]
    sel_tier = np.array([tier_of[l] for l in sel_lang])
    is_top5 = np.array([f in top5_ids for f in sel_ids])

    cfg = ModelConfig(device=device, dtype=torch.bfloat16)
    model, tokenizer = load_model_and_tokenizer(cfg)
    sae = load_sae(cfg)
    capture = ResidualCapture(model, cfg.layer)

    chunks = {}
    for l in languages:
        chunks[l] = build_equalized_token_chunks(load_sentences(l, "devtest"), tokenizer, args.tokens, 128)

    t0 = time.time()
    fp = {l: encode_batched(chunks[l], model, tokenizer, sae, capture, device, args.batch_size) for l in languages}
    sizes = [fp[l].shape[0] for l in languages]
    fp_pool = torch.cat([fp[l] for l in languages])
    for l in languages:
        del fp[l]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    fp = {l: fp_pool[offs[i]:offs[i + 1]] for i, l in enumerate(languages)}
    rates = (fp_pool > 0).float().mean(0).cpu().numpy()
    is_active = rates > 0.001
    active_ids = np.flatnonzero(is_active)
    quint = stratify_by_firing_rate_quintiles(rates, is_active)
    sel_set = set(sel_ids)
    univ_cand = [f for f in active_ids if f not in sel_set]
    rng = np.random.default_rng(42)
    matched = []
    for f in sel_ids:
        cands = [u for u in univ_cand if quint[u] == quint[f]]
        matched.append(int(rng.choice(cands)) if cands else -1)
    matched_ok = np.array([m >= 0 for m in matched])
    matched_ids = [m for m in matched if m >= 0]
    log({"fp/encode_s": time.time() - t0, "fp/tokens": int(fp_pool.shape[0]), "fp/active_features": int(is_active.sum()),
         "tracked/selective": len(sel_ids), "tracked/matched_universal": len(matched_ids)})

    lang_sel_idx = {l: [i for i, x in enumerate(sel_lang) if x == l] for l in languages}
    results = {}
    for b in bits:
        t1 = time.time()
        with quantized_model(model, bits=b):
            q = {l: encode_batched(chunks[l], model, tokenizer, sae, capture, device, args.batch_size) for l in languages}
        q_pool = torch.cat([q[l] for l in languages])
        del q
        q = {l: q_pool[offs[i]:offs[i + 1]] for i, l in enumerate(languages)}

        sel_r = np.full(len(sel_ids), np.nan)
        for l in languages:
            idx = lang_sel_idx[l]
            if idx:
                cols = torch.tensor([sel_ids[i] for i in idx], device=device)
                sel_r[idx] = pearson_cols(fp[l][:, cols], q[l][:, cols])
        univ_r = pearson_cols(fp_pool[:, torch.tensor(matched_ids, device=device)],
                              q_pool[:, torch.tensor(matched_ids, device=device)])

        act_cols = torch.tensor(active_ids, device=device)
        by_language = {}
        for l in languages:
            r_all = pearson_cols(fp[l][:, act_cols], q[l][:, act_cols])
            r_all = r_all[~np.isnan(r_all)]
            r_l = sel_r[lang_sel_idx[l]]
            r_l = r_l[~np.isnan(r_l)]
            by_language[l] = {
                "n_selective": int(len(r_l)),
                "selective_damaged_frac": round(float((r_l <= 0.5).mean()), 4) if len(r_l) else None,
                "selective_survived_frac": round(float((r_l >= 0.9).mean()), 4) if len(r_l) else None,
                "selective_median_r": round(float(np.median(r_l)), 4) if len(r_l) else None,
                "n_active_defined": int(len(r_all)),
                "overall_damaged_frac": round(float((r_all <= 0.5).mean()), 4),
                "overall_survived_frac": round(float((r_all >= 0.9).mean()), 4),
                "overall_median_r": round(float(np.median(r_all)), 4),
            }
        tier = {}
        for t in ["high", "mid", "low"]:
            m = sel_tier == t
            if m.any():
                tier[t] = summarize(sel_r[m])
        res = {
            "elapsed_seconds": round(time.time() - t1, 1),
            "selective_features": summarize(sel_r),
            "quintile_matched_universal": summarize(univ_r),
            "selective_features_top5": summarize(sel_r[is_top5]),
            "quintile_matched_universal_top5": summarize(univ_r[is_top5[matched_ok]]),
            "by_resource_tier": tier,
            "by_language": by_language,
            "selective_feature_pearson_r": {str(f): round(float(r), 6) for f, r in zip(sel_ids, sel_r)},
        }
        results[f"INT{b}"] = res
        s, u = res["selective_features"], res["quintile_matched_universal"]
        log({"bits": b, f"INT{b}/elapsed_s": res["elapsed_seconds"],
             f"INT{b}/selective_survived_pct": s["survived_pct"], f"INT{b}/universal_survived_pct": u["survived_pct"],
             f"INT{b}/selective_median_r": s["median_pearson_r"], f"INT{b}/universal_median_r": u["median_pearson_r"],
             **{f"INT{b}/tier_{t}_survived_pct": d["survived_pct"] for t, d in tier.items()}})
        del q, q_pool
        torch.cuda.empty_cache()

    report = {
        "stage": "Phase1_Survival_GPU",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(0),
        "n_languages": len(languages),
        "held_out_split": "devtest",
        "tokens_per_language": args.tokens,
        "tracked_selective_set": args.tracked,
        "results_by_bitwidth": results,
    }
    out = REPO_ROOT / args.output
    out.write_text(json.dumps(report, indent=2))
    print("Wrote", out)
    if run:
        art = __import__("wandb").Artifact("survival-report-gpu", type="results")
        art.add_file(str(out))
        run.log_artifact(art)
        run.finish()


if __name__ == "__main__":
    main()
