"""The runner of the F5 study (plan Task 13; design docs/superpowers/specs/2026-10-23-feature-function-equivalence-design.md sections 3 and 11): does an SAE feature do the same thing in a quantized model as in the
full-precision (FP) model? This file holds the model-independent core (callables lower / upper_final / logits_fn, tested on a toy causal model); the Ada set-up and `main` follow in later sub-steps.

Callables: lower(tokens) -> residual [B, L, d] at the SAE layer; upper_final(residual) -> final hidden state [B, L, d_h] (before the final norm); logits_fn(hidden rows [n, d_h]) -> FLOAT32 logits [n, V]
(final norm, unembedding and soft cap computed in float32: bf16 logits have a spacing of 0.125 near 20 and would drown the effect of a single feature); encode(rows) -> SAE activations [n, F].
Positions p are indices into the valid (content) rows of a sentence, the padded tensor position is p + 1 (the `mask` regime: BOS at position 0, never edited, never a statistic).

Single-feature ablation in state m at the positions T_f where the FULL-PRECISION activation is positive: x' = x - a_f^m(t) d_f (a no-op where the feature is inactive in m; the SAE reconstruction error is kept, as in the
frozen interventions). Fingerprint D_m(f) = mean over the ablated positions of the CENTRED logit change (z' - z) - mean_v (z' - z); the baseline and the ablated rows of a feature are evaluated in ONE batch so that
both pass through identical kernels (a sham of amplitude zero then gives exactly zero). dnll = mean per predicted token of the loss change over all predictions of the context sentences."""
from collections import defaultdict

import torch

import phase1_func_intervene as fi


def select_contexts(max_act, M, min_ctx):
    """max_act [N_sent, n_f]: the maximum full-precision activation of each sample feature in each candidate sentence. Per feature the M sentences with the largest activation among those where it is positive (ties:
    the smaller sentence index first). Returns contexts [n_f, M] (-1 padded), n_ctx, n_active (sentences where the feature is active) and excluded (n_active < min_ctx: too rare to test on the write side)."""
    n_s, n_f = max_act.shape
    ctx = torch.full((n_f, M), -1, dtype=torch.long); n_ctx = torch.zeros(n_f, dtype=torch.long)
    for f in range(n_f):
        v = max_act[:, f]; act = torch.nonzero(v > 0, as_tuple=True)[0]
        if not len(act):
            continue
        top = act[torch.sort(-v[act], stable=True).indices][:M]
        ctx[f, :len(top)] = top; n_ctx[f] = len(top)
    n_active = (max_act > 0).sum(0)
    return {"contexts": ctx, "n_ctx": n_ctx, "n_active": n_active, "excluded": n_active < min_ctx}


def plan_contexts(contexts, excluded, dev_sentences):
    """The unique context sentences (in order of first use) and, per non-excluded feature k, the local indices of its sentences: {"sentences": [token lists], "dev_index": [index in the candidate list], "ctx": {k: [local]}}."""
    local, sentences, dev_index, ctx = {}, [], [], {}
    for k in range(contexts.shape[0]):
        if bool(excluded[k]):
            continue
        ids = [int(x) for x in contexts[k] if int(x) >= 0]
        for i in ids:
            if i not in local:
                local[i] = len(sentences); sentences.append(dev_sentences[i]); dev_index.append(i)
        ctx[k] = [local[i] for i in ids]
    return {"sentences": sentences, "dev_index": dev_index, "ctx": ctx}


def _rows(b, device):
    tokens, valid = b["tokens"], b["valid"]
    return (tokens.to(device), valid.to(device)) if device is not None else (tokens, valid)


@torch.no_grad()
def max_activation_per_sentence(lower, encode, batches, cols, n_sent, device=None):
    """[n_sent, len(cols)]: the maximum over the valid positions of each sentence of the activation of the columns `cols` (feature indices)."""
    cols = torch.as_tensor(cols, dtype=torch.long); out = torch.zeros(n_sent, len(cols))
    for b in batches:
        tokens, valid = _rows(b, device)
        a = fi.encode_valid(lower(tokens), valid, encode); a = a[:, cols.to(a.device)]
        off = 0
        for r in range(tokens.shape[0]):
            n = int(valid[r].sum()); s = int(b["sent"][r]); rows = a[off:off + n]; off += n
            if n:
                out[s] = rows.max(0).values.cpu()
    return out


def _pairs_by_sentence(plan):
    by = defaultdict(list)
    for k, ss in plan["ctx"].items():
        for s in ss:
            by[s].append(k)
    return by


@torch.no_grad()
def fp_active_positions(lower, encode, batches, plan, feat_ids, device=None):
    """{(k, s): LongTensor of positions p of context sentence s where the full-precision activation of feature feat_ids[k] is positive}; one entry per (feature, context sentence). Run with the FP weights."""
    by = _pairs_by_sentence(plan); pos = {}
    for b in batches:
        tokens, valid = _rows(b, device)
        a = fi.encode_valid(lower(tokens), valid, encode); off = 0
        for r in range(tokens.shape[0]):
            n = int(valid[r].sum()); s = int(b["sent"][r]); rows = a[off:off + n]; off += n
            for k in by.get(s, ()):
                pos[(k, s)] = torch.nonzero(rows[:, int(feat_ids[k])] > 0, as_tuple=True)[0].cpu()
    return pos


@torch.no_grad()
def cache_state(lower, encode, batches, plan, feat_ids, positions, device=None):
    """For the CURRENT model state: the residual rows of every context sentence (BOS row included, padding trimmed; host) and, per (feature, sentence), the activation of the feature in this state at the
    full-precision positions. -> {"resid": {s: [n + 1, d]}, "act": {(k, s): [len(positions)]}}."""
    by = _pairs_by_sentence(plan); resid_c, act_c = {}, {}
    for b in batches:
        tokens, valid = _rows(b, device)
        resid = lower(tokens); a = fi.encode_valid(resid, valid, encode); off = 0
        for r in range(tokens.shape[0]):
            n = int(valid[r].sum()); s = int(b["sent"][r]); rows = a[off:off + n]; off += n
            resid_c[s] = resid[r, :n + 1].detach().cpu()
            for k in by.get(s, ()):
                act_c[(k, s)] = rows[positions[(k, s)].to(rows.device), int(feat_ids[k])].float().cpu()
    return {"resid": resid_c, "act": act_c}


@torch.no_grad()
def fingerprint_feature(k, plan, cache, positions, d_f, upper_final, logits_fn, V, device, chunk=64, sham=False, reverse=False, extra_pad=0):
    """The effect fingerprint of feature k in the state of `cache` (see the module docstring). -> {"D": [V] float32 on `device`, "n_pos": ablated positions (the full-precision active ones), "n_active_q": of
    these, the positions where the feature is active in this state, "dnll": mean per predicted token of the loss change, "n_ctx"}. reverse and extra_pad present the contexts in another order and with more padding
    (the same weights, another batching: the numerical-floor pseudo-state)."""
    sents = plan["ctx"][k][::-1] if reverse else plan["ctx"][k]; rows = [cache["resid"][s] for s in sents]; M = len(rows); L = max(r.shape[0] for r in rows) + extra_pad
    base = torch.zeros(M, L, rows[0].shape[1], dtype=rows[0].dtype)
    for j, r in enumerate(rows):
        base[j, :r.shape[0]] = r
    abl = base.clone(); df = d_f.float().cpu(); n_pos = n_active = 0
    for j, s in enumerate(sents):
        p = positions[(k, s)]; a = cache["act"][(k, s)]
        n_pos += len(p); n_active += int((a > 0).sum())
        if len(p) and not sham:
            abl[j, p + 1] = (abl[j, p + 1].float() - a[:, None] * df[None, :]).to(base.dtype)
    hidden = upper_final(torch.cat([base, abl], 0).to(device))
    hb, ha, tg, has_t, flag = [], [], [], [], []
    for j, s in enumerate(sents):
        n = len(plan["sentences"][s]); is_abl = torch.zeros(n + 1, dtype=torch.bool); is_abl[positions[(k, s)] + 1] = True            # content position p is the padded row p + 1
        t = torch.zeros(n + 1, dtype=torch.long); t[:n] = torch.tensor(plan["sentences"][s], dtype=torch.long)                             # row r predicts content token r; the last row has no target
        ht = torch.zeros(n + 1, dtype=torch.bool); ht[:n] = True
        hb.append(hidden[j, :n + 1]); ha.append(hidden[M + j, :n + 1]); tg.append(t); has_t.append(ht); flag.append(is_abl)
    hb, ha, tg, has_t, flag = torch.cat(hb), torch.cat(ha), torch.cat(tg).to(device), torch.cat(has_t).to(device), torch.cat(flag).to(device)
    total = torch.zeros(V, dtype=torch.float32, device=device); d_nll = 0.0
    for i in range(0, hb.shape[0], chunk):
        zb, za = logits_fn(hb[i:i + chunk]).float(), logits_fn(ha[i:i + chunk]).float(); t = tg[i:i + chunk]; ar = torch.arange(len(t), device=device); m = has_t[i:i + chunk]
        d_nll += float(((-torch.log_softmax(za, -1)[ar, t] + torch.log_softmax(zb, -1)[ar, t]) * m).sum())
        f = flag[i:i + chunk]
        if bool(f.any()):
            d = (za - zb)[f]; total += (d - d.mean(1, keepdim=True)).sum(0)
    return {"D": total / n_pos if n_pos else total, "n_pos": n_pos, "n_active_q": n_active, "dnll": d_nll / int(has_t.sum()), "n_ctx": M}


def _row_norms(X, chunk=256):
    return torch.cat([X[i:i + chunk].float().norm(dim=1) for i in range(0, X.shape[0], chunk)])


@torch.no_grad()
def similarity_matrix(Dfp, Dq, device="cpu", chunk=256):
    """Cosine similarity S[i, j] of the full-precision fingerprint of feature i and the quantized fingerprint of feature j, computed in chunks (the inputs may be float16 on the host). A zero full-precision row gives
    a NaN row (nothing to compare with); a zero quantized row gives similarity 0 to every query."""
    nfp, nq = _row_norms(Dfp, chunk), _row_norms(Dq, chunk); S = torch.zeros(Dfp.shape[0], Dq.shape[0])
    for i in range(0, Dfp.shape[0], chunk):
        A = Dfp[i:i + chunk].to(device).float(); A = A / nfp[i:i + chunk].to(device).clamp_min(1e-30)[:, None]
        for j in range(0, Dq.shape[0], chunk):
            B = Dq[j:j + chunk].to(device).float(); B = B / nq[j:j + chunk].to(device).clamp_min(1e-30)[:, None]
            S[i:i + chunk, j:j + chunk] = (A @ B.T).cpu()
    S[:, nq == 0] = 0.0; S[nfp == 0] = float("nan")
    return S


def topk_indices(X, k, chunk=256):
    """The indices of the k largest entries (the most promoted tokens) of every row of X."""
    return torch.cat([torch.topk(X[i:i + chunk].float(), k, dim=1).indices for i in range(0, X.shape[0], chunk)])


def topk_overlap_from_indices(Iu, Iv, k):
    """Share of the k promoted tokens of Iu that are also in Iv, per row (zero fingerprints are the caller's business: the metrics module marks them undefined)."""
    return (Iu[:, :, None] == Iv[:, None, :]).any(2).sum(1).float() / k


@torch.no_grad()
def collect_read(lower, encode, batches, cols, device=None):
    """The activations of the columns `cols` over all valid rows of the batches (float32, host, one array per batch): the full-precision reference of the read side."""
    cols = torch.as_tensor(cols, dtype=torch.long); out = []
    for b in batches:
        tokens, valid = _rows(b, device)
        a = fi.encode_valid(lower(tokens), valid, encode); out.append(a[:, cols.to(a.device)].float().cpu())
    return out


@torch.no_grad()
def accumulate_read(lower, encode, batches, cols, fp_list, acc, device=None):
    """Feed the streaming read-side statistics of the CURRENT state against the stored full-precision activations of the same batches."""
    cols = torch.as_tensor(cols, dtype=torch.long)
    for b, x in zip(batches, fp_list):
        tokens, valid = _rows(b, device)
        a = fi.encode_valid(lower(tokens), valid, encode)
        acc.add(x.numpy(), a[:, cols.to(a.device)].float().cpu().numpy())


def compare_state(D_fp, D_q, stats, topk, groups=None):
    """Compact comparison of a state's fingerprints (float16, host, one row per tested feature) with the full-precision ones: the similarity matrix (float16), its diagonal (the feature against itself: no effect
    in the quantized state is 0 and flagged, no effect in full precision is NaN), the promoted-token overlap, the magnitude ratio, the retrieval of the feature's own identity, and the counts of the state."""
    import numpy as np
    import phase1_func_f5_metrics as met
    S = similarity_matrix(D_fp, D_q); nfp, nq = _row_norms(D_fp).numpy(), _row_norms(D_q).numpy()
    zero_fp, zero_q = nfp == 0, nq == 0
    ov = topk_overlap_from_indices(topk_indices(D_fp, topk), topk_indices(D_q, topk), topk).numpy().astype(float); ov[zero_fp | zero_q] = np.nan
    ratio = np.where(nfp > 0, nq / np.where(nfp > 0, nfp, 1.0), np.nan)
    Sn = S.numpy()
    return {"S": S.half(), "cos": np.diagonal(Sn).astype(float).copy(), "zero_fp": zero_fp, "zero_q": zero_q, "topk_overlap": ov, "magnitude_ratio": ratio,
            "n_pos": np.array(stats["n_pos"]), "n_active_q": np.array(stats["n_active_q"]), "dnll": np.array(stats["dnll"]), "retrieval": met.retrieval(Sn, groups=groups)}


def run_f5(states, setup, lower, upper_final, logits_fn, encode, W_dec, feat_ids, eval_batches, dev_sentences, batcher, M, min_ctx, V, device, topk=50, groups=None, extra_pad=3, log=lambda *a: None):
    """The F5 measurements for the model states `states` (labels; "FP" first; "FP_rebatched" is the numerical-floor pseudo-state: the same weights, the contexts reversed and padded). setup(label) puts the model in
    that state (restores the full-precision weights, quantizes) and is called before the state is measured; batcher(sentences) -> batches (the format of phase1_lib.make_batches).
    Stage A (full precision): the contexts of every sample feature from the candidate sentences, the full-precision active positions, the full-precision read-side reference. Per state: the read side over the evaluation
    batches, then the write side (fingerprints) and its comparison with the full-precision fingerprints. -> {"stage_a": {...}, "states": {label: {"read": {...}, "write": {...}}}}."""
    import numpy as np
    import phase1_func_f5_metrics as met
    setup("FP")
    ma = max_activation_per_sentence(lower, encode, batcher(dev_sentences), feat_ids, len(dev_sentences), device)
    sel = select_contexts(ma, M, min_ctx); plan = plan_contexts(sel["contexts"], sel["excluded"], dev_sentences)
    rows = sorted(plan["ctx"]); write_features = [int(feat_ids[k]) for k in rows]; ctx_batches = batcher(plan["sentences"])
    positions = fp_active_positions(lower, encode, ctx_batches, plan, feat_ids, device); fp_read = collect_read(lower, encode, eval_batches, feat_ids, device)
    log("stage A:", len(rows), "features with contexts,", int(sel["excluded"].sum()), "excluded,", len(plan["sentences"]), "context sentences")
    out = {"stage_a": {"write_features": write_features, "write_rows": rows, "excluded": sel["excluded"].numpy(), "n_active": sel["n_active"].numpy(), "n_ctx": sel["n_ctx"].numpy()}, "states": {}}
    D_fp, g = None, (None if groups is None else np.asarray(groups)[rows])
    for label in states:
        setup(label); acc = met.ReadAccumulator(len(feat_ids)); accumulate_read(lower, encode, eval_batches, feat_ids, fp_read, acc, device)
        cache = cache_state(lower, encode, ctx_batches, plan, feat_ids, positions, device); rebatched = label == "FP_rebatched"
        Ds, st = [], {"n_pos": [], "n_active_q": [], "dnll": []}
        for k in rows:
            o = fingerprint_feature(k, plan, cache, positions, W_dec[int(feat_ids[k])], upper_final, logits_fn, V, device, reverse=rebatched, extra_pad=extra_pad if rebatched else 0)
            Ds.append(o["D"].half().cpu()); st["n_pos"].append(o["n_pos"]); st["n_active_q"].append(o["n_active_q"]); st["dnll"].append(o["dnll"])
        D = torch.stack(Ds)
        if label == "FP":
            D_fp = D
        elif D_fp is None:
            raise ValueError("the full-precision state must come first")
        out["states"][label] = {"read": acc.finish(), "write": compare_state(D_fp, D, st, topk, g)}
        log("state", label, "done; median cos", float(np.nanmedian(out["states"][label]["write"]["cos"])))
    return out
