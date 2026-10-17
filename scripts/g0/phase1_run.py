"""Phase 1 orchestration (labelled: modified / new code on top of Duan et al.'s setup, never called a replication).
run_tagging / run_survival take the model forward, the SAE encoder, the quantizer and the restore as injectable callables: in production they are
THEIR functions (extracted verbatim from the vendored pristine script, see load_their_functions); in the tests they are a fake causal model.
Decisions: D-001 (regimes), D-003 (selectivity), D-004 (this design), D-006 (ladder), D-010 (all sentences)."""
import math

import torch

import phase1_lib as p1


def _offsets(data):
    offsets, lang_of, c = [], [], 0
    for l in sorted(data):
        offsets.append(c); lang_of += [l] * len(data[l]); c += len(data[l])
    return offsets, lang_of


def run_tagging(forward, encode, data, regime, bos_id, pad_id, n_feat, device="cpu", batch_size=32, max_tokens=None, max_len=256):
    """Full-precision pass over the dev split: per-sentence positive-token counts, activation sums and token counts of every SAE feature."""
    offsets, lang_of = _offsets(data)
    tag = p1.TagStats(len(lang_of), n_feat)
    with torch.no_grad():
        for b in p1.make_batches(data, bos_id, regime, pad_id, max_len, batch_size, max_tokens):
            resid, _ = forward(b["tokens"].to(device))
            feats = encode(resid.reshape(-1, resid.shape[-1]))
            v = b["valid"].reshape(-1)
            gsent = (offsets[b["lang"]] + b["sent"])[:, None].expand_as(b["valid"]).reshape(-1)[v]
            tag.update(feats[v.to(feats.device)], gsent.to(feats.device))
    return tag, offsets, torch.tensor(lang_of)


class _Loss:
    def __init__(self, n_lang):
        self.s, self.n = [0.0] * n_lang, [0] * n_lang

    def add(self, b, logits):
        if logits is None:
            return
        s, n = p1.chunked_nll(logits, b["tokens"].to(logits.device), b["valid"].to(logits.device), chunk=4)   # 4 rows x 256k vocabulary in float32 ~ 1 GB at 256 tokens
        self.s[b["lang"]] += float(s.sum()); self.n[b["lang"]] += int(n.sum())


def _pack(lp, loss):
    n_lang = lp.n_lang
    rs = [lp.result(l) for l in range(n_lang)]
    stack = lambda k: torch.stack([r[k].float() for r in rs])
    return {"corr": stack("corr"), "firing_rate": stack("firing_rate"), "mean_activation": stack("mean_activation"), "max_activation": stack("max_activation"),
            "n": [r["n"] for r in rs], "loss_sum": list(loss.s), "loss_n": list(loss.n), "pooled": lp.pooled()}


def run_survival(forward, encode, quantize, restore, data, regime, bos_id, pad_id, n_feat, bits_list, device="cpu", batch_size=32, max_tokens=None, max_len=256):
    """Devtest pass: the full-precision features of every batch are cached sparsely on the host, then each rung quantizes the single model in place,
    recomputes the features and updates the per-language streaming Pearson against the cached ones; bits == 0 is the restore-only control (INT0).
    Returns {"FP": result, bits: result, ...}; every result holds per-language corr / firing_rate / mean_activation / max_activation [n_lang, n_feat],
    token counts n and loss sums / counts per language."""
    n_lang = len(data)
    batches = p1.make_batches(data, bos_id, regime, pad_id, max_len, batch_size, max_tokens)
    cache, results = p1.SparseRows(), {}
    with torch.no_grad():
        restore()
        lp, loss = p1.LangPearson(n_lang, n_feat, device), _Loss(n_lang)
        for b in batches:
            resid, logits = forward(b["tokens"].to(device))
            feats = encode(resid.reshape(-1, resid.shape[-1]))
            x = feats[b["valid"].reshape(-1).to(feats.device)]
            cache.add(x); lp.update(b["lang"], x, x); loss.add(b, logits)
        results["FP"] = _pack(lp, loss)
        for bits in bits_list:
            restore()
            if bits != 0:
                quantize(bits)
            lp, loss = p1.LangPearson(n_lang, n_feat, device), _Loss(n_lang)
            for i, b in enumerate(batches):
                resid, logits = forward(b["tokens"].to(device))
                feats = encode(resid.reshape(-1, resid.shape[-1]))
                y = feats[b["valid"].reshape(-1).to(feats.device)]
                lp.update(b["lang"], cache.get(i, feats.device), y); loss.add(b, logits)
            results[bits] = _pack(lp, loss)
        restore()
    return results


def survival_summary(corr, firing_rate, threshold=0.001):
    """Their summary definitions (streaming_feature_metrics): active = full-precision firing rate > threshold; survived r > 0.9; degraded 0.5 < r <= 0.9;
    damaged r < 0.5 (so r == 0.5 falls in no bin, as in their code); median is torch's lower-middle value."""
    active = firing_rate > threshold
    c = corr[active].double()
    if c.numel() == 0:
        raise RuntimeError("No active features. Lower FIRING_THRESHOLD.")
    return {"n_active_features": int(active.sum()), "mean_corr": float(c.mean()), "median_corr": float(c.median()),
            "survived_>0.9_pct": float((c > 0.9).double().mean() * 100), "degraded_0.5_0.9_pct": float(((c > 0.5) & (c <= 0.9)).double().mean() * 100),
            "damaged_<0.5_pct": float((c < 0.5).double().mean() * 100)}


def _loss(r):
    return sum(r["loss_sum"]) / sum(r["loss_n"])


def condition_rows(results, bits_list, n_feat, layer=12, threshold=0.001):
    """One row per condition over ALL languages pooled, in the column order of their phase3_summary_final.csv. FP baseline has bits 32 (wandb schema v1);
    bits == 0 is the restore-only control."""
    fp = results["FP"]; fp_loss = _loss(fp); fp_ppl = math.exp(fp_loss); rate = fp["pooled"]["firing_rate"]
    rows = []
    for key in ["FP"] + list(bits_list):
        r = results[key]; loss = _loss(r); ppl = math.exp(loss)
        name, method, bits = ("FP16 baseline", "FP16", 32) if key == "FP" else ((f"RTN INT{key}", "RTN", key) if key else ("restore-only control (INT0)", "control", 0))
        rows.append({"layer": layer, "condition": name, "method": method, "bits": bits, "perplexity": ppl, "ppl_delta_pct": (ppl / fp_ppl - 1) * 100,
                     **survival_summary(r["pooled"]["corr"], rate, threshold), "n_tokens": int(sum(r["n"])), "n_total_features": int(n_feat), "loss": loss})
    return rows


def language_rows(results, bits_list, lang_names, threshold=0.001):
    """Per language and rung: the same survival summary with that language's own full-precision active set (a sanity table; the Phase 1 analysis proper is phase1_analysis.py)."""
    fp, rows = results["FP"], []
    for li, name in enumerate(lang_names):
        for bits in bits_list:
            r = results[bits]
            rows.append({"language": name, "bits": bits, "n_tokens": int(fp["n"][li]), "perplexity": math.exp(r["loss_sum"][li] / r["loss_n"][li]),
                         **survival_summary(r["corr"][li].double(), fp["firing_rate"][li].double(), threshold)})
    return rows


def selectivity_report(tag, lang_of, n_lang, min_panel_rate=1e-5, min_pos=50, n_boot=100, frac=0.9, min_stable=80, n_perm=50, q=0.95, n_strata=5, seed=0):
    """Proposal section 4.1 on the tagging statistics of the dev split: S(f) = max_l r_l / mean_l r_l with its three safeguards (support rule; bootstrap stability
    of the peak language in >= min_stable of n_boot draws; the binary split at the q-quantile of S after shuffling the sentence-language assignment), the
    rank-normalised score S~ within supported features (empirical CDF, ties share the largest rank), Deng et al.'s nu (decision D-003), and the
    firing-rate-quintile-matched universal controls (non-selective supported features drawn without replacement per stratum of the panel mean firing rate)."""
    rates = p1.firing_rates(tag.counts, tag.ntok, lang_of, n_lang); means = p1.mean_activations(tag.sums, tag.ntok, lang_of, n_lang)
    totals = p1.language_totals(tag.counts, lang_of, n_lang)
    S, peak = p1.selectivity_S(rates), p1.peak_language(rates)
    supported = p1.supported_mask(rates, totals, min_panel_rate, min_pos)
    stable = p1.bootstrap_peak_stability(tag.counts, tag.ntok, lang_of, n_lang, peak, n_boot, frac, seed)
    cutoff = p1.shuffled_null_cutoff(tag.counts, tag.ntok, lang_of, n_lang, supported, n_perm, q, seed)
    selective = supported & (stable >= min_stable) & (S > cutoff)
    idx = supported.nonzero().squeeze(1)
    S_rank = torch.zeros_like(S); strata = torch.full((S.shape[0],), -1, dtype=torch.long)
    if len(idx):
        _, inv, cnt = torch.unique(S[idx], return_inverse=True, return_counts=True)
        S_rank[idx] = (cnt.cumsum(0)[inv]).to(S.dtype) / len(idx)
        strata[idx] = p1.strata_ids(rates.mean(0)[idx], n_strata)
    controls, short = p1.matched_sample(strata, selective, supported & ~selective, seed)
    return {"rates": rates, "mean_activation": means, "S": S, "S_rank": S_rank, "peak": peak, "supported": supported, "stable_count": stable, "null_cutoff": cutoff,
            "selective": selective, "nu": p1.deng_nu(means), "strata": strata, "controls": controls, "control_strata_short": short}


def save_outputs(out_dir, meta, results, bits_list, lang_names, n_feat, threshold=0.001, report=None):
    """Write the files the analysis reads: summary.csv (their columns, pooled over languages), per_language.csv, survival.pt (per-language correlations of
    every rung, FP firing statistics, token counts, loss sums), selectivity.pt (if a report is given) and meta.json (provenance)."""
    import csv, json
    from pathlib import Path
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary.csv", condition_rows(results, bits_list, n_feat, threshold=threshold)),
                       ("per_language.csv", language_rows(results, bits_list, lang_names, threshold=threshold))):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    keys = ["FP"] + list(bits_list)
    torch.save({"languages": list(lang_names), "bits": list(bits_list), "n": results["FP"]["n"],
                "corr": {b: results[b]["corr"].float() for b in bits_list},
                "firing_rate": results["FP"]["firing_rate"].float(), "mean_activation": results["FP"]["mean_activation"].float(), "max_activation": results["FP"]["max_activation"].float(),
                "loss_sum": {k: results[k]["loss_sum"] for k in keys}, "loss_n": {k: results[k]["loss_n"] for k in keys},
                "pooled_corr": {b: results[b]["pooled"]["corr"].float() for b in bits_list}, "pooled_firing_rate": results["FP"]["pooled"]["firing_rate"].float()}, out / "survival.pt")
    if report is not None:
        keep = ("rates", "mean_activation", "S", "S_rank", "nu")
        torch.save({**{k: (v.float() if k in keep else v) for k, v in report.items()}, "languages": list(lang_names)}, out / "selectivity.pt")
    from their_code import MANIFEST
    manifest = json.loads(MANIFEST.read_text())
    (out / "meta.json").write_text(json.dumps({**meta, "languages": list(lang_names), "bits": list(bits_list), "n_feat": int(n_feat), "threshold": threshold,
                                               "their_functions_sha256": manifest["file_sha256"], "their_functions": manifest["functions"]}, indent=1, default=str) + "\n")


def package_versions(names):
    """Installed distribution versions from package metadata ({name: version or 'not installed'}); transformer_lens has no __version__ attribute."""
    from importlib import metadata
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = "not installed"
    return out


def check_forward_paths(full, short, tokens):
    """The tagging pass stops after the hooked block (no logits); its activations must equal those of the full forward pass exactly."""
    a, b = full(tokens)[0], short(tokens)[0]
    assert torch.equal(a, b), f"stop-at-layer activations differ from the full forward pass (max abs diff {float((a.float() - b.float()).abs().max())})"


SAE_RELEASE, SAE_ID, LAYER, SEQ_LEN, FIRING_THRESHOLD = "gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical", 12, 256, 0.001     # their config, copied
MODEL_CANDIDATES = ["gemma-2-2b", "google/gemma-2-2b"]
DEFAULT_BITS = [8, 7, 6, 5, 4, 16, 12, 0]       # decision D-006: ladder INT8/7/6/5/4, then the INT16/INT12 noise-floor controls and INT0 (restore only)


def main(argv=None):
    import argparse, json, subprocess, time
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Phase 1 run on Duan et al.'s setup: one BOS regime, one GPU, in-place RTN quantize / restore of a single model.")
    ap.add_argument("--corpus", choices=["flores", "wikitext"], required=True); ap.add_argument("--regime", choices=p1.REGIMES, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, nargs="*", default=DEFAULT_BITS); ap.add_argument("--budget", type=int, default=20000, help="wikitext only: token budget (their 3a_test = 20000)")
    ap.add_argument("--flores-root"); ap.add_argument("--roster", default=str(ROOT / "reports/g0/phase1_languages.json"))
    ap.add_argument("--languages", nargs="*", help="subset of FLORES codes (smoke)"); ap.add_argument("--max-sentences", type=int, help="first N sentences per language and split (smoke)")
    ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--skip-tagging", action="store_true"); ap.add_argument("--n-boot", type=int, default=100); ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--commit", default="unknown")
    a = ap.parse_args(argv)
    t0 = time.time(); log = lambda *m: print(f"[{time.time() - t0:7.0f}s]", *m, flush=True)

    import gc
    import numpy as np
    from tqdm.auto import tqdm
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    import their_code as tc
    import phase1_data as pd_
    torch.set_grad_enabled(False)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    env = {"torch": torch, "np": np, "tqdm": tqdm, "gc": gc, "HookedTransformer": HookedTransformer, "SAE": SAE, "DEVICE": DEVICE, "WORK_DEVICE": DEVICE,
           "BATCH_SIZE": 1, "FIRING_THRESHOLD": FIRING_THRESHOLD}
    if a.corpus == "wikitext":
        from datasets import load_dataset
        env["load_dataset"] = load_dataset
    ex = tc.extract_functions(tc.PRISTINE, tc.THEIR_FUNCTIONS); tc.verify_hashes(ex, json.loads(tc.MANIFEST.read_text())["functions"]); tc.build_namespace(ex, env)
    log("their functions verified against the frozen manifest:", sorted(ex))

    model, model_name = env["load_gemma_tl_model"](MODEL_CANDIDATES)
    sae = env["load_sae_compat"](SAE_RELEASE, SAE_ID, DEVICE)
    hook = getattr(sae.cfg, "hook_name", None) or f"blocks.{LAYER}.hook_resid_post"
    assert sae.cfg.d_in == model.cfg.d_model, "SAE / model width mismatch"
    n_feat = int(sae.cfg.d_sae)
    env["model_work"] = model; env["original_state_cpu"] = env["clone_state_to_cpu"](model)
    tok = model.tokenizer; bos = tok.bos_token_id; pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    log("model", model_name, "hook", hook, "d_sae", n_feat, "bos", bos, "pad", pad)

    def forward(tokens):
        logits, cache = model.run_with_cache(tokens, names_filter=[hook]); return cache[hook], logits
    def forward_acts(tokens):                                                    # tagging needs no logits: stop after the hooked block (same activations)
        _, cache = model.run_with_cache(tokens, names_filter=[hook], stop_at_layer=LAYER + 1); return cache[hook], None
    encode = lambda acts: env["encode_sae"](sae, acts)
    quantize = lambda bits: env["quantize_rtn_per_output_channel_tl"](model, bits=bits)
    restore = env["restore_model_work"]

    meta = {"corpus": a.corpus, "regime": a.regime, "bits": a.bits, "seed": a.seed, "commit": a.commit, "model": model_name, "sae_release": SAE_RELEASE, "sae_id": SAE_ID,
            "hook": hook, "bos_id": bos, "pad_id": pad, "batch_size": a.batch_size, "max_tokens": a.max_tokens, "device": DEVICE,
            "versions": package_versions(["torch", "transformer-lens", "sae-lens", "transformers", "datasets"]), "argv": list(argv) if argv is not None else None}
    if a.corpus == "flores":
        roster = pd_.load_roster(a.roster)
        if a.languages:
            roster = [r for r in roster if r["code"] in set(a.languages)]
            meta["subset"] = True
        dev, rep_dev = pd_.build_corpus(a.flores_root, roster, "dev", tok, None if a.languages else 997)
        devtest, rep_test = pd_.build_corpus(a.flores_root, roster, "devtest", tok, None if a.languages else 1012)
        if a.max_sentences:
            dev = {k: v[:a.max_sentences] for k, v in dev.items()}; devtest = {k: v[:a.max_sentences] for k, v in devtest.items()}
        names = [r["code"] for r in roster]; meta["corpus_report"] = {"dev": rep_dev, "devtest": rep_test}
    else:
        toks = env["build_tokens_2d"](tok, a.budget, SEQ_LEN)
        dev, devtest, names = None, {0: [row.tolist() for row in toks.cpu()]}, ["wikitext"]
    log("languages:", len(names), "devtest sentences:", sum(len(v) for v in devtest.values()))

    report = None
    if dev is not None and not a.skip_tagging:
        probe = p1.make_batches({0: dev[0][:8]}, bos, a.regime, pad, 256, 8)[0]["tokens"].to(DEVICE)
        check_forward_paths(forward, forward_acts, probe); log("stop-at-layer tagging forward equals the full forward pass on a probe batch")
        tag, offsets, lang_of = run_tagging(forward_acts, encode, dev, a.regime, bos, pad, n_feat, device=DEVICE, batch_size=a.batch_size, max_tokens=a.max_tokens)
        log("tagging pass done; sentences", len(lang_of))
        report = selectivity_report(tag, lang_of, len(names), n_boot=a.n_boot, n_perm=a.n_perm, seed=a.seed)
        log("supported", int(report["supported"].sum()), "selective", int(report["selective"].sum()), "controls", len(report["controls"]), "null cutoff", report["null_cutoff"])
        del tag

    results = run_survival(forward, encode, quantize, restore, devtest, a.regime, bos, pad, n_feat, a.bits, device=DEVICE, batch_size=a.batch_size, max_tokens=a.max_tokens)
    log("survival pass done; peak GPU memory GB:", torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0)
    meta["seconds"] = time.time() - t0; meta["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    save_outputs(a.out, meta, results, a.bits, names, n_feat, FIRING_THRESHOLD, report)
    log("saved", a.out)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
