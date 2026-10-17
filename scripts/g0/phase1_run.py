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
        s, n = p1.chunked_nll(logits, b["tokens"].to(logits.device), b["valid"].to(logits.device))
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
