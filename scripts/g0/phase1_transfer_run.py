"""Runner pieces of the intervention-transfer study (D-088; ledger reports/intervention_transfer/STATUS.md; plan Task 2). This file holds the parts that do not need a GPU and are tested with small fake suffixes:
the five directions of a feature, the targets of a context, the chunking and the resumable chunk files, and `process_chunk`, which computes the five cells of the crossed design for every (feature, context,
direction) item of a chunk and stores only the 5 x 5 Gram matrix of the cell vectors, the signed loss changes, the edited-row counts and the pooled (averaged-fingerprint) Gram of every (feature, direction).
The model-facing glue (loading Gemma-2-2B and the SAE, the prefix caches under FP and under a quantized state, the logit gates, the loop over states and chunks, the instrument job) is added below `main` once the
tests of these pieces pass; the ledger records the design question that this file answers: the Gram needs vectors from two suffix states, so a chunk is processed as  FP suffix (cells FP/FP and Q/FP)  ->  switch
the weights to the quantized state  ->  Q suffix (cells FP/Q, Q/Q, Q/Q-native), holding the FP-suffix vectors of the chunk on the host in between (about 2 MB per item, so chunks of about 25 features).

Conventions (as in the F5 runner): a context is one dev sentence; its cached residual has n + 1 rows (row 0 is BOS); the content position p of the sentence is row p + 1; row r predicts content token r and the
last row has no target; the edit at a row subtracts the activation of the feature (FP activation in the fixed cells, the quantized state's own activation in the native cell) times the direction."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

import phase1_transfer_core as tc

DIRECTION_KINDS = ("real", "matched_swapped", "random1", "random2", "random3")
RANDOM_SEEDS = [1, 2, 3]
GRAD_SCALES = []            # amendment A7 (finite-difference cells of the gradient baseline) is WITHDRAWN: at t = 0.05 the edit is at the bf16 resolution of the residual and the effect is rounding noise (instrument job 40300); the gradient baseline is amendment A9 (`grad_fn`)
# Instrument gates of the model-facing part (not part of the frozen spec). The float32 readout is compared with the model's own bfloat16 logits on the rows of a probe batch: top-1 agreement and the largest absolute
# logit difference. Amendment A6 (ledger, 2026-10-26 14:35): the top-1 threshold of the F5 runner (0.95) was written for ITS probe of four evaluation sentences (62 rows, 59 of 62 needed); the first instrument job
# (Slurm 40070) agreed on 58 of 62 rows of its probe with a largest difference of 0.59, so the threshold is 0.90 on a larger probe (8 sentences, at most 512 rows): the gate catches a wrong split, layer, dtype or
# readout, not near ties between the bfloat16 and the float32 logits. The largest-difference limit is unchanged.
GATE_MIN_TOP1, GATE_MAX_ABS, GATE_PROBE_SENTENCES, GATE_ROWS = 0.90, 2.0, 8, 512


def random_direction(direction, seed, feature_id):
    """A direction with the norm of `direction`, orthogonal to it, reproducible from (seed, feature id): auxiliary calibration directions, not independent samples."""
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(feature_id))
    d = direction.double(); r = torch.randn(d.shape[0], generator=g, dtype=torch.float64)
    r = r - (r @ d) / (d @ d) * d
    return (r * (d.norm() / r.norm())).to(direction.dtype)


def build_directions(W_dec, feature_id, matched_id, seeds):
    """{kind: [d] vector} for one feature: its own decoder direction, the decoder direction of the matched different feature, three random directions."""
    if int(matched_id) == int(feature_id):
        raise ValueError(f"the matched direction of feature {feature_id} is the feature itself")
    if len(seeds) != 3:
        raise ValueError("three random-direction seeds are required")
    real = W_dec[int(feature_id)]
    out = {"real": real, "matched_swapped": W_dec[int(matched_id)]}
    for i, s in enumerate(seeds, 1):
        out[f"random{i}"] = random_direction(real, s, feature_id)
    return out


def context_targets(tokens):
    """Targets of the rows of a cached context: row r predicts content token r, the last row (the last content token's own position) has none (-1)."""
    t = torch.as_tensor(tokens, dtype=torch.long)
    return torch.cat([t, torch.tensor([-1], dtype=torch.long)])


def plan_chunks(feature_ids, per_chunk):
    if per_chunk < 1:
        raise ValueError("a chunk needs at least one feature")
    ids = list(feature_ids)
    return [ids[i:i + per_chunk] for i in range(0, len(ids), per_chunk)]


def chunk_key(freeze_sha, state, feature_ids):
    return hashlib.sha256(json.dumps([freeze_sha, state, [int(f) for f in feature_ids]]).encode()).hexdigest()


def atomic_save(obj, path, key):
    """Write the chunk to a temporary file and move it into place: a chunk file is either complete or absent."""
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    torch.save({"key": key, "payload": obj}, tmp); os.replace(tmp, path)


def load_chunk(path, key):
    """None if there is no completed chunk file; the payload if its key matches; ValueError for a stale file (another freeze, state or feature list)."""
    path = Path(path)
    if not path.exists():
        return None
    d = torch.load(path, weights_only=False)
    if d["key"] != key:
        raise ValueError(f"{path} is a stale chunk (its key differs from the requested one)")
    return d["payload"]


def _batches(jobs, max_rows, per_item=2):
    """Group the items by context ("ctx"; an item without one is its own context) and cut each group into sub-batches whose largest suffix call (the quantized suffix with the fixed AND the native edits of the quantized
    prefix: 1 + per_item K sequences of L rows, per_item = 2 + the number of gradient scales) stays within `max_rows` rows (at least one item per sub-batch). Items of one context must share the residuals and the targets."""
    groups = {}
    for it in jobs:
        groups.setdefault(it.get("ctx", ("single", it["key"])), []).append(it)
    out = []
    for g in groups.values():
        first = g[0]
        if any(it["h_fp"] is not first["h_fp"] or it["h_q"] is not first["h_q"] or it["targets"] is not first["targets"] for it in g[1:]):
            raise ValueError(f"items of the context {first.get('ctx', first['key'])} must share the residuals and the targets")
        kmax = max(1, ((max_rows // first["h_fp"].shape[0]) - 1) // per_item)
        out += [g[i:i + kmax] for i in range(0, len(g), kmax)]
    return out


def process_items(items, suffixes, switches, center=True, max_rows=768, observer=None, grad_scales=(), grad_fn=None):
    """The five cells of the crossed design for a list of items, each with its OWN edits (the feature-level run builds them from one feature's amplitude times a direction, the group-level run from the sum over a
    feature group). item: {"key" (unique), "rows" (edited rows of the cached context), "h_fp", "h_q" ([L, d] float32 prefix residuals), "edit_fp" (the FP edit at those rows, used in the cells FP/FP, Q/FP, FP/Q,
    Q/Q), "edit_native" (the quantized state's own edit, cell Q/Q-native), "targets" ([L], -1 = none), "pool" (key of the pool the item belongs to, or None), "ctx" (items of one context: the same residual and
    target objects; they share the unedited run in ONE suffix call per cell, see `_batches`), "n_active_q" (optional count), "extra" (optional dict of scalars stored with the item)}.
    grad_scales: scales t of finite-difference cells 'Q/Q-grad@t' (amendment A7, withdrawn: kept only for the instrument check that showed them to be rounding noise). grad_fn(h [L, d] float32, targets) -> [L, d]: the
    gradient of the mean NLL over the rows with a target with respect to the cached prefix residual under the CURRENT weights (amendment A9); it is called once per context batch under the FP weights (h_fp) and once under
    the Q weights (h_q), and each item stores the first-order predictions of its loss change: g_pred_fp = -sum over the edited rows of <gradient_FP, edit_fp>, g_pred_q_fixed = -<gradient_Q, edit_fp>,
    g_pred_q_native = -<gradient_Q, edit_native>, and the mean row norms of the two gradients at the edited rows.
    Order of work: switches["FP"](), the FP-suffix cells of every context batch (vectors kept on the host), switches["Q"](), the Q-suffix cells (the cells Q/Q and Q/Q-native of a batch share one call and one
    unedited run), then the Gram matrices. observer: see `phase1_transfer_core.effect_many` (the tag is (cell, [item keys]); for the call 'Q/Q+Q/Q-native' the first K edits are Q/Q, the next K Q/Q-native).
    -> {"items": {key: {"gram" (5 x 5 float64), "dloss", "loss_base" (five values each), "n_rows", "n_target_rows", "n_active_q", "n_positions", "edit_norm_fp", "edit_norm_native", "resid_norm_fp", "resid_norm_q" and the item's optional "extra" fields}}, "pooled": {pool: {"gram", "n_rows", "n_contexts"}}, "skipped": [(key, reason)]}."""
    keys = [it["key"] for it in items]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate item keys: {sorted({k for k in keys if keys.count(k) > 1}, key=str)[:3]}")
    jobs, skipped = [], []
    for it in items:
        if len(it["rows"]) == 0:
            skipped.append((it["key"], "no_edited_row"))
        else:
            jobs.append(it)
    names = tuple(tc.CELLS) + tuple(tc.grad_cell_name(t) for t in grad_scales); batches = _batches(jobs, max_rows, 2 + len(grad_scales)); fp_vectors = {}; g_fp = {}
    switches["FP"]()
    for b in batches:
        f0, ks = b[0], [it["key"] for it in b]; edits = [(it["rows"], it["edit_fp"]) for it in b]
        try:
            ff = tc.effect_many(f0["h_fp"], edits, suffixes["FP"], f0["targets"], center, observer, ("FP/FP", ks)); qf = tc.effect_many(f0["h_q"], edits, suffixes["FP"], f0["targets"], center, observer, ("Q/FP", ks))
        except ValueError as e:
            raise ValueError(f"context {ks[0]} (batch of {len(ks)}): {e}") from None
        for it, x, y in zip(b, ff, qf):
            fp_vectors[it["key"]] = {"FP/FP": _cpu(x), "Q/FP": _cpu(y)}
        if grad_fn is not None:
            g_fp[id(b)] = grad_fn(f0["h_fp"], f0["targets"]).detach().cpu()
    out, sums, nrows, counts = {}, {}, {}, {}
    switches["Q"]()
    for b in batches:
        f0, ks = b[0], [it["key"] for it in b]; edits = [(it["rows"], it["edit_fp"]) for it in b]; native = [(it["rows"], it["edit_native"]) for it in b]
        try:
            fq = tc.effect_many(f0["h_fp"], edits, suffixes["Q"], f0["targets"], center, observer, ("FP/Q", ks))
            grads = [(it["rows"], t * it["edit_fp"]) for t in grad_scales for it in b]
            qq = tc.effect_many(f0["h_q"], edits + native + grads, suffixes["Q"], f0["targets"], center, observer, ("Q/Q+Q/Q-native" + "".join("+" + tc.grad_cell_name(t) for t in grad_scales), ks))
        except ValueError as e:
            raise ValueError(f"context {ks[0]} (batch of {len(ks)}): {e}") from None
        gF = g_fp.pop(id(b), None); gQ = grad_fn(f0["h_q"], f0["targets"]).detach().cpu() if grad_fn is not None else None
        for i, it in enumerate(b):
            key = it["key"]; cells = dict(fp_vectors.pop(key)); cells["FP/Q"] = _cpu(fq[i]); cells["Q/Q"] = _cpu(qq[i]); cells["Q/Q-native"] = _cpu(qq[len(b) + i])
            for j, t in enumerate(grad_scales):
                cells[tc.grad_cell_name(t)] = _cpu(qq[(2 + j) * len(b) + i])
            n = cells["FP/FP"]["n_rows"]
            out[key] = {"gram": tc.gram(cells, names).numpy().astype(np.float64), "cells": list(names), "dloss": [cells[c]["dloss"] for c in names], "loss_base": [cells[c]["loss_base"] for c in names], "n_rows": n,
                        "n_target_rows": int((it["targets"] >= 0).sum()), "n_active_q": it.get("n_active_q"), "n_positions": int(len(it["rows"])),
                        "edit_norm_fp": float(it["edit_fp"].norm(dim=-1).mean()), "edit_norm_native": float(it["edit_native"].norm(dim=-1).mean()),
                        "resid_norm_fp": float(it["h_fp"][it["rows"]].norm(dim=-1).mean()), "resid_norm_q": float(it["h_q"][it["rows"]].norm(dim=-1).mean()), **it.get("extra", {})}
            if gF is not None:
                r = it["rows"]; out[key].update({"g_pred_fp": -float((gF[r] * it["edit_fp"]).sum()), "g_pred_q_fixed": -float((gQ[r] * it["edit_fp"]).sum()), "g_pred_q_native": -float((gQ[r] * it["edit_native"]).sum()),
                                                 "g_norm_fp": float(gF[r].norm(dim=-1).mean()), "g_norm_q": float(gQ[r].norm(dim=-1).mean())})
            pool = it.get("pool")
            if pool is not None:
                acc = sums.setdefault(pool, {c: 0 for c in names}); nrows[pool] = nrows.get(pool, 0) + n; counts[pool] = counts.get(pool, 0) + 1
                for c in names:
                    acc[c] = acc[c] + n * cells[c]["vector"].double()
    pooled = {}
    for pool, acc in sums.items():
        avg = {c: {"vector": acc[c] / nrows[pool]} for c in names}
        pooled[pool] = {"gram": tc.gram(avg, names).numpy().astype(np.float64), "n_rows": nrows[pool], "n_contexts": counts[pool]}
    return {"items": out, "pooled": pooled, "skipped": skipped}


def process_chunk(features, contexts, caches, positions, amplitudes, dirs, tokens, suffixes, switches, center=True, max_rows=768, observer=None, grad_scales=(), grad_fn=None):
    """The items of one chunk of the feature-level run: for every feature, context (dev sentence) and direction kind the edit is the activation of the feature at its FP-active positions times the direction (FP
    activation in the fixed cells, the quantized state's own activation in the native cell).
    features: feature ids; contexts: {feature: [dev sentence index]}; caches: {"FP": {sentence: [n+1, d] residual}, "Q": {...}} (any float dtype; converted to float32);
    positions: {(feature, sentence): content positions where the feature is active in FP}; amplitudes: {"FP": {(feature, sentence): [len(positions)]}, "Q": {...}}; dirs: {feature: {kind: [d] vector}};
    tokens: {sentence: content token ids}; suffixes: {"FP": f, "Q": f}; switches: {"FP": f, "Q": f} (called once each, FP first).
    -> process_items' result with the keys (feature, sentence, kind), the pools (feature, kind) and, in "skipped", (feature, sentence, reason) for contexts without an FP-active position."""
    items, skipped = [], []
    for f in features:
        for s in contexts[f]:
            p = positions[(f, s)]
            if len(p) == 0:
                skipped.append((f, s, "no_fp_active_position")); continue
            for state in ("FP", "Q"):
                if s not in caches[state]:
                    raise KeyError(f"no cached residual of state {state} for context sentence {s}")
                if len(amplitudes[state][(f, s)]) != len(p):
                    raise ValueError(f"feature {f}, context {s}: {len(amplitudes[state][(f, s)])} {state} amplitudes for {len(p)} positions")
            rows, tg, h_fp, h_q = p + 1, context_targets(tokens[s]), caches["FP"][s].float(), caches["Q"][s].float(); n_act = int((amplitudes["Q"][(f, s)] > 0).sum())
            for kind in DIRECTION_KINDS:
                d = dirs[f][kind].float()
                items.append({"key": (f, s, kind), "rows": rows, "h_fp": h_fp, "h_q": h_q, "edit_fp": amplitudes["FP"][(f, s)].float()[:, None] * d[None, :], "edit_native": amplitudes["Q"][(f, s)].float()[:, None] * d[None, :],
                              "targets": tg, "pool": (f, kind), "ctx": (f, s), "n_active_q": n_act,
                              "extra": {"amp_fp_mean": float(amplitudes["FP"][(f, s)].float().mean()), "amp_q_mean": float(amplitudes["Q"][(f, s)].float().mean()), "direction_norm": float(d.norm()),
                                        "amp_fp": [float(x) for x in amplitudes["FP"][(f, s)].float()], "amp_q": [float(x) for x in amplitudes["Q"][(f, s)].float()]}})
    res = process_items(items, suffixes, switches, center, max_rows, observer, grad_scales, grad_fn)
    return {"items": res["items"], "pooled": res["pooled"], "skipped": skipped}


def group_edit(act_fp, act_q, W_dec, idx):
    """The zero-ablation edit of a feature group on one context. act_fp, act_q: [n, F] SAE activations of the content rows in the FP and in the quantized state; W_dec [F, d]; idx: the group's feature ids.
    -> (rows (content row + 1, the BOS row), edit_fp [m, d], edit_native [m, d]): the rows where some group feature is active in FP; the FP edit is the sum over the group of activation times decoder row, the
    native edit the same with the quantized state's own activations at the same rows."""
    idx = torch.as_tensor(idx, dtype=torch.long); a_fp, a_q = act_fp[:, idx].float(), act_q[:, idx].float(); W = W_dec[idx].float()
    active = torch.nonzero((a_fp > 0).any(1), as_tuple=True)[0]
    return active + 1, a_fp[active] @ W, a_q[active] @ W


def group_items(code, sentence, h_fp, h_q, act_fp, act_q, W_dec, sets, targets):
    """The items of one evaluation sentence for the group and its matched control draws: sets {kind: feature ids} (kind 'group', 'control0', ...). -> (items, skipped); a set with no active feature in FP on this
    sentence is skipped with its key. The pool of an item is (language code, kind): the loss changes of the sentences are pooled by token count in the analysis."""
    items, skipped = [], []
    for kind, idx in sets.items():
        rows, e_fp, e_q = group_edit(act_fp, act_q, W_dec, idx)
        if len(rows) == 0:
            skipped.append(((code, sentence, kind), "no_active_group_feature")); continue
        n_act = int((act_q[:, torch.as_tensor(idx, dtype=torch.long)][rows - 1] > 0).any(1).sum())
        items.append({"key": (code, sentence, kind), "rows": rows, "h_fp": h_fp, "h_q": h_q, "edit_fp": e_fp, "edit_native": e_q, "targets": targets, "pool": (code, kind), "ctx": (code, sentence), "n_active_q": n_act})
    return items, skipped


def _cpu(e):
    return {**e, "vector": e["vector"].detach().cpu()}


def make_suffix(upper_final, logits_fn, device):
    """suffix([B, L, d] float32 residual) -> [B, L, V] float32 logits: the layers above the SAE site, then the float32 readout. The readout of the F5 runner (`make_logits_fn`) takes ROWS [n, d] (it concatenates the
    vocabulary slices along dimension 1), so batch and length are flattened for it and restored afterwards (the first real run failed here: a [B, L, d] input made the slices disagree)."""
    def suffix(x):
        h = upper_final(x.to(device)); B, L, d = h.shape
        return logits_fn(h.reshape(B * L, d)).reshape(B, L, -1)
    return suffix


def subset_core(core, max_features, max_contexts):
    """The smoke subset of the instrument job: evenly spaced features (first and last included), the first `max_contexts` contexts of each (the full core when both are None)."""
    n = len(core["feature_ids"]); idx = list(range(n))
    if max_features is not None and max_features < n:
        idx = sorted(set(np.linspace(0, n - 1, max_features).round().astype(int).tolist()))
    out = {k: [core[k][i] for i in idx] for k in ("feature_ids", "direction_ids", "feature_language_index", "contexts")}
    if max_contexts is not None:
        out["contexts"] = [c[:max_contexts] for c in out["contexts"]]
    return out


def run_transfer(states, setup, lower, suffix, encode, W_dec, core, dev_tokens, batcher, out_dir, freeze_sha, per_chunk, device="cpu", log=lambda *a: None, max_features=None, max_contexts=None, max_rows=768, grad_scales=(), grad_fn=None):
    """The feature-level core of the study for the quantized states `states` (labels, FP excluded: the FP prefix cache is built once). setup(label) puts the model weights in that state ('FP' or a quantized label);
    lower(tokens) -> [B, L, d] prefix residual; suffix([B, L, d] float32) -> [B, L, V] float32 logits (upper layers, final norm, unembedding, soft cap); encode(rows) -> SAE activations;
    W_dec [F, d]; core: the committed core sample (feature ids, matched direction ids, contexts as dev-sentence indices); dev_tokens: token ids of every dev sentence; batcher(sentences) -> batches (as the F5 runner).
    Writes one resumable file per (state, chunk of features) and returns the metadata (counts, gates, timings)."""
    import time
    import phase1_func_f5_run as fr5
    t0 = time.time(); sub = subset_core(core, max_features, max_contexts)
    feats = [int(f) for f in sub["feature_ids"]]; matched = {f: int(m) for f, m in zip(feats, sub["direction_ids"])}; ctxs = {f: [int(s) for s in c] for f, c in zip(feats, sub["contexts"])}
    uniq = sorted({s for f in feats for s in ctxs[f]}); local = {s: i for i, s in enumerate(uniq)}
    plan = {"sentences": [dev_tokens[s] for s in uniq], "dev_index": uniq, "ctx": {k: [local[s] for s in ctxs[f]] for k, f in enumerate(feats)}}
    fid = torch.tensor(feats, dtype=torch.long); batches = batcher(plan["sentences"]); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); W_cpu = W_dec.detach().float().cpu()
    keyed = lambda cache: ({uniq[l]: r for l, r in cache["resid"].items()}, {(feats[k], uniq[l]): a for (k, l), a in cache["act"].items()})
    gates = {"sham_exact": True, "sham_max_abs": 0.0}

    def sham(resid_by_sentence):
        for (f, s), p in list(pos_f.items())[:2]:
            e = tc.effect(resid_by_sentence[s].float(), p + 1, torch.zeros(len(p), resid_by_sentence[s].shape[1]), suffix, context_targets(dev_tokens[s]))
            worst = max(float(e["vector"].abs().max()), abs(e["dloss"])); gates["sham_max_abs"] = max(gates["sham_max_abs"], worst); gates["sham_exact"] = gates["sham_exact"] and worst == 0.0

    setup("FP")
    positions = fr5.fp_active_positions(lower, encode, batches, plan, fid, device); pos_f = {(feats[k], uniq[l]): p for (k, l), p in positions.items()}
    cache_fp = fr5.cache_state(lower, encode, batches, plan, fid, positions, device); resid_fp, act_fp = keyed(cache_fp); sham(resid_fp)
    log("FP caches:", len(uniq), "context sentences,", len(pos_f), "feature-context pairs,", sum(len(p) for p in pos_f.values()), "edited positions")
    meta = {"n_features": len(feats), "n_context_sentences": len(uniq), "n_pairs": len(pos_f), "n_positions": int(sum(len(p) for p in pos_f.values())), "states": {}, "gates": gates}
    for label in states:
        setup(label); cache_q = fr5.cache_state(lower, encode, batches, plan, fid, positions, device); resid_q, act_q = keyed(cache_q); sham(resid_q)
        caches = {"FP": resid_fp, "Q": resid_q}; amps = {"FP": act_fp, "Q": act_q}; chunks = plan_chunks(feats, per_chunk)
        st = {"chunks": len(chunks), "resumed": 0, "items": 0, "skipped": 0, "seconds": 0.0}; meta["states"][label] = st
        for ci, chunk in enumerate(chunks):
            key = chunk_key(freeze_sha, label, chunk); path = out / f"transfer_{label}_c{ci:03d}.pt"
            if load_chunk(path, key) is not None:
                st["resumed"] += 1; continue
            t1 = time.time()
            res = process_chunk(chunk, {f: ctxs[f] for f in chunk}, caches, pos_f, amps, {f: build_directions(W_cpu, f, matched[f], RANDOM_SEEDS) for f in chunk},
                                {s: dev_tokens[s] for f in chunk for s in ctxs[f]}, {"FP": suffix, "Q": suffix}, {"FP": lambda: setup("FP"), "Q": lambda l=label: setup(l)}, max_rows=max_rows, grad_scales=grad_scales, grad_fn=grad_fn)
            atomic_save({**res, "features": chunk}, path, key); st["items"] += len(res["items"]); st["skipped"] += len(res["skipped"]); st["seconds"] += time.time() - t1
            log(f"state {label} chunk {ci + 1}/{len(chunks)}: {len(res['items'])} items, {time.time() - t1:.0f} s")
    setup("FP"); meta["seconds"] = time.time() - t0
    return meta


def run_group(states, setup, lower, suffix, encode, W_dec, man, languages, eval_tokens, batcher, out_dir, freeze_sha, per_chunk, device="cpu", log=lambda *a: None, label="K64", n_controls=3, max_sentences=None, max_rows=768, grad_fn=None):
    """The group-level panel of the study for the quantized states `states`: the zero-ablation of each language's feature group (`label`, K64: the first K features of the manifest's list) and of its first
    `n_controls` matched control draws on the language's own evaluation sentences, in the cells of the crossed design. man: the frozen groups manifest; languages: codes in the order of the roster;
    eval_tokens: {code: [token ids per evaluation sentence]}; the other arguments as in `run_transfer`. The SAE activations of a sentence are those of its CACHED residual in each state (the SAE does not depend on the
    model weights), so the edits are exact functions of the caches. A chunk is `per_chunk` sentences (all their items), written to `group_<state>_c<NNN>.pt` and resumable. Returns the metadata."""
    import time
    t0 = time.time(); languages = list(languages); sets = {}
    for code in languages:
        if code not in man["groups"] or code not in man["controls"] or code not in eval_tokens:
            raise ValueError(f"language {code} is not in the manifest or in the evaluation sentences")
        draws = man["controls"][code][label][:n_controls]
        if len(draws) < n_controls:
            raise ValueError(f"{code} has {len(draws)} control draws of size {label}, {n_controls} asked")
        sets[code] = {"group": man["groups"][code]["all"][:man["sizes"][code][label]], **{f"control{r}": d for r, d in enumerate(draws)}}
    cut = (lambda xs: xs[:max_sentences]) if max_sentences is not None else (lambda xs: xs)
    sents = [(code, j) for code in languages for j in range(len(cut(eval_tokens[code])))]; toks = [eval_tokens[c][j] for c, j in sents]
    batches = batcher(toks); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); W_cpu = W_dec.detach().float().cpu(); gates = {"sham_exact": True, "sham_max_abs": 0.0}

    def cache():
        res = {}
        for b in batches:
            tokens, valid = b["tokens"].to(device), b["valid"].to(device); r = lower(tokens)
            for i in range(tokens.shape[0]):
                res[int(b["sent"][i])] = r[i, :int(valid[i].sum()) + 1].detach().cpu()
        return res

    def sham(resid):
        for g in range(min(2, len(sents))):
            e = tc.effect(resid[g].float(), torch.tensor([1]), torch.zeros(1, resid[g].shape[1]), suffix, context_targets(toks[g]))
            worst = max(float(e["vector"].abs().max()), abs(e["dloss"])); gates["sham_max_abs"] = max(gates["sham_max_abs"], worst); gates["sham_exact"] = gates["sham_exact"] and worst == 0.0

    acts = lambda h: encode(h[1:].to(device)).float().cpu()
    setup("FP"); cache_fp = cache(); sham(cache_fp)
    meta = {"n_sentences": len(sents), "n_languages": len(languages), "label": label, "n_controls": n_controls, "states": {}, "gates": gates}
    log("FP caches:", len(sents), "evaluation sentences of", len(languages), "languages")
    for state in states:
        setup(state); cache_q = cache(); sham(cache_q); chunks = plan_chunks(list(range(len(sents))), per_chunk)
        st = {"chunks": len(chunks), "resumed": 0, "items": 0, "skipped": 0, "seconds": 0.0}; meta["states"][state] = st
        for ci, chunk in enumerate(chunks):
            key = chunk_key(freeze_sha, f"group-{label}-{state}", chunk); path = out / f"group_{state}_c{ci:03d}.pt"
            if load_chunk(path, key) is not None:
                st["resumed"] += 1; continue
            t1 = time.time(); items, skipped = [], []
            for g in chunk:
                code, j = sents[g]; h_fp, h_q = cache_fp[g].float(), cache_q[g].float()
                its, sk = group_items(code, j, h_fp, h_q, acts(h_fp), acts(h_q), W_cpu, sets[code], context_targets(toks[g])); items += its; skipped += sk
            res = process_items(items, {"FP": suffix, "Q": suffix}, {"FP": lambda: setup("FP"), "Q": lambda l=state: setup(l)}, max_rows=max_rows, grad_fn=grad_fn)
            res["skipped"] = skipped + res["skipped"]
            atomic_save({**res, "sentences": chunk}, path, key); st["items"] += len(res["items"]); st["skipped"] += len(res["skipped"]); st["seconds"] += time.time() - t1
            log(f"group state {state} chunk {ci + 1}/{len(chunks)}: {len(res['items'])} items, {time.time() - t1:.0f} s")
    setup("FP"); meta["seconds"] = time.time() - t0
    return meta


class GateError(RuntimeError):
    pass


def check_core_against_freeze(freeze, core):
    """The core sample the job reads and the spec in the code must be the frozen ones (the hashes of the feature ids, the matched directions and the contexts, and the whole spec)."""
    import phase1_transfer_freeze as tf
    now, was = tf._core_block(core), freeze["core"]; problems = []
    for key, label in (("feature_ids_sha256", "feature_ids"), ("direction_ids_sha256", "direction_ids"), ("contexts_sha256", "contexts"), ("n_features", "n_features"), ("n_feature_context_pairs", "n_pairs")):
        if now[key] != was.get(key):
            problems.append(label)
    if json.loads(json.dumps(tf.SPEC)) != freeze["spec"]:
        problems.append("spec")
    if problems:
        raise ValueError("the core sample or the spec of the code differ from the freeze in: " + ", ".join(problems))


def make_grad_fn(upper_final, model, device, vocab_chunk=65536):
    """grad_fn(h [L, d], targets [L]) -> [L, d] float32 on the host: the gradient of the mean NLL over the rows with a target with respect to the prefix residual, through the suffix of the CURRENT weights (amendment A9).
    torch.autograd.grad asks for the gradient of the residual only, so no parameter gradient is formed; the global no-grad mode of the run is lifted inside. The readout is the float32 one of the F5 runner (final RMS
    norm, unembedding, soft cap; `make_logits_fn`) but its gradient is formed in vocabulary chunks: autograd through the whole readout kept four float32 copies of the unembedding slices (2.4 GB) and ran out of
    memory next to the model on an 11 GB card (jobs 40330 and 40331). Pass 1 (no graph): the log-sum-exp of every row over the vocabulary chunks. Pass 2, chunk by chunk: the loss sum_v softmax_v(detached) z_v -
    z_target (weighted 1 / number of rows with a target), whose gradient with respect to the normalized hidden state is exactly the gradient of the NLL; then one backward pass through the final norm and the layers above."""
    def grad_fn(h, targets):
        with torch.enable_grad():
            x = h.to(device).float().unsqueeze(0).requires_grad_(True)
            hf = upper_final(x)[0].float(); n = hf / torch.sqrt(hf.pow(2).mean(-1, keepdim=True) + model.cfg.eps) * model.ln_final.w.float()
            valid = (targets >= 0).to(device); tg = targets.clamp(min=0).to(device); rows = torch.arange(len(tg), device=device); weight = valid.float() / valid.sum().clamp(min=1)
            W, b, cap = model.unembed.W_U, model.unembed.b_U.float(), model.cfg.output_logits_soft_cap; starts = range(0, W.shape[1], vocab_chunk)

            def z_chunk(i, nn):
                z = nn @ W[:, i:i + vocab_chunk].float() + b[i:i + vocab_chunk]
                return cap * torch.tanh(z / cap) if cap else z

            with torch.no_grad():
                lse = torch.logsumexp(torch.stack([torch.logsumexp(z_chunk(i, n.detach()), -1) for i in starts]), 0)
            leaf = n.detach().clone().requires_grad_(True); dn = torch.zeros_like(n)
            for i in starts:
                z = z_chunk(i, leaf); p = torch.exp(z.detach() - lse[:, None]); loss = ((p * z).sum(-1) * weight).sum()
                inside = (tg >= i) & (tg < i + vocab_chunk)
                if bool(inside.any()):
                    loss = loss - (z[rows[inside], tg[inside] - i] * weight[inside]).sum()
                dn = dn + torch.autograd.grad(loss, leaf)[0]                                                                # inputs=leaf only: if the weights require grad, no parameter gradient is formed
            (g,) = torch.autograd.grad(n, x, grad_outputs=dn)
        return g[0].detach().cpu()
    return grad_fn


def check_inputs_against_freeze(freeze, paths):
    """The input files of the group panel (the groups manifest, the evaluation freeze) must be the files whose hashes the freeze recorded. paths: {name: path} with names as in freeze["files"]."""
    problems = []
    for name, path in paths.items():
        if name not in freeze["files"]:
            problems.append(f"{name} is not in the freeze"); continue
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != freeze["files"][name]["sha256"]:
            problems.append(f"{name} differs from the freeze")
    if problems:
        raise ValueError("; ".join(problems))


def main(argv=None):
    import argparse, gc, time
    ROOT = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Intervention-transfer study (D-088): the feature-level core, Gemma-2-2B layer 12, states FP and the quantized ones (instrument job or full run).")
    ap.add_argument("--out", required=True); ap.add_argument("--flores-root", required=True); ap.add_argument("--roster", default=str(ROOT / "reports/g0/phase1_languages.json"))
    ap.add_argument("--freeze", default=str(ROOT / "reports/intervention_transfer/freeze.json")); ap.add_argument("--core", default=str(ROOT / "reports/intervention_transfer/core_sample.json"))
    ap.add_argument("--states", nargs="+", default=["16", "6", "4"], help="quantized states (bit widths; FP is always the reference); 16 = the rounding-only floor")
    ap.add_argument("--task", choices=["feature", "group"], default="feature", help="feature: the feature-level core (404 features); group: the group-level panel (K64 group and control draws on the 27 x 300 evaluation sentences)")
    ap.add_argument("--manifest", default=str(ROOT / "reports/functional_validation/groups_manifest.json")); ap.add_argument("--eval-freeze", default=str(ROOT / "reports/functional_validation/evaluation_freeze_lm.json"))
    ap.add_argument("--chunk-sentences", type=int, default=60); ap.add_argument("--max-sentences", type=int, help="first n evaluation sentences of each language (group smoke)"); ap.add_argument("--max-languages", type=int, help="first n languages (group smoke)")
    ap.add_argument("--grad-scales", type=float, nargs="*", default=GRAD_SCALES, help="scales of the first-order cells of the gradient baseline (feature task only; amendment A7: 0.05; the instrument job may add 0.025 to check the finite difference)")
    ap.add_argument("--max-rows", type=int, default=768, help="largest suffix call in rows (sequences x length); a context batch is cut to stay within it"); ap.add_argument("--chunk-features", type=int, default=25); ap.add_argument("--max-features", type=int, help="evenly spaced subset of the core (instrument job)"); ap.add_argument("--max-contexts", type=int, help="first contexts of each feature (instrument job)")
    ap.add_argument("--dev-per-language", type=int, default=997); ap.add_argument("--max-len", type=int, default=256); ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--commit", default="unknown")
    a = ap.parse_args(argv)
    t0 = time.time(); log = lambda *m: print(f"[{time.time() - t0:7.0f}s]", *m, flush=True)
    freeze_text = Path(a.freeze).read_text(); freeze = json.loads(freeze_text); core = json.loads(Path(a.core).read_text()); check_core_against_freeze(freeze, core)
    freeze_sha = hashlib.sha256(freeze_text.encode()).hexdigest()
    if a.task == "group":
        check_inputs_against_freeze(freeze, {"groups_manifest": a.manifest, "evaluation_freeze_lm": a.eval_freeze})

    from tqdm.auto import tqdm
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    import their_code as their
    import phase1_data as pd_
    import phase1_func_run as fr
    import phase1_func_f5_run as fr5
    import phase1_lib as p1
    torch.set_grad_enabled(False)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LAYER = fr.LAYER
    env = {"torch": torch, "np": np, "tqdm": tqdm, "gc": gc, "HookedTransformer": HookedTransformer, "SAE": SAE, "DEVICE": DEVICE, "WORK_DEVICE": DEVICE, "BATCH_SIZE": 1, "FIRING_THRESHOLD": 0.001}
    ex = their.extract_functions(their.PRISTINE, their.THEIR_FUNCTIONS); their.verify_hashes(ex, json.loads(their.MANIFEST.read_text())["functions"]); their.build_namespace(ex, env)
    log("their functions verified against the frozen manifest")
    import phase1_run as pr
    model, model_name = env["load_gemma_tl_model"](pr.MODEL_CANDIDATES)
    sae = env["load_sae_compat"](pr.SAE_RELEASE, pr.SAE_ID, DEVICE)
    hook = getattr(sae.cfg, "hook_name", None) or f"blocks.{LAYER}.hook_resid_post"
    assert hook == f"blocks.{LAYER}.hook_resid_post", f"unexpected SAE hook {hook}"
    assert sae.cfg.d_in == model.cfg.d_model
    env["model_work"] = model; env["original_state_cpu"] = env["clone_state_to_cpu"](model)
    tok = model.tokenizer; bos = tok.bos_token_id; pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    lower = lambda tokens: model(tokens, stop_at_layer=LAYER + 1)
    upper = lambda resid: model(resid, start_at_layer=LAYER + 1)
    upper_final, logits_fn = fr5.make_upper_final(model, LAYER), fr5.make_logits_fn(model)
    encode = lambda x: env["encode_sae"](sae, x)
    restore = env["restore_model_work"]
    suffix = make_suffix(upper_final, logits_fn, DEVICE)
    grad_fn = make_grad_fn(upper_final, model, DEVICE)
    log("model", model_name, "hook", hook, "d_sae", int(sae.cfg.d_sae), "V", int(model.cfg.d_vocab), "soft cap", model.cfg.output_logits_soft_cap)

    labels = [fr5.state_label(x) for x in a.states]
    if "FP" in labels or "FP_rebatched" in labels:
        raise SystemExit("--states are the quantized states; FP is the reference and is always run")
    setup_seconds = {}

    def setup(label):
        s = time.time(); restore()
        if fr5.state_bits(label) is not None:
            env["quantize_rtn_per_output_channel_tl"](model, bits=fr5.state_bits(label))
        d = setup_seconds.setdefault(label, {"calls": 0, "seconds": 0.0}); d["calls"] += 1; d["seconds"] += time.time() - s

    names = [r["code"] for r in pd_.load_roster(a.roster)]
    if list(core["languages"]) != names:
        raise ValueError("the core sample was built for another language order")
    dev_tokens, truncated = [], 0
    for code in names:
        for s in pd_.encode_sentences(tok, pd_.read_flores_split(a.flores_root, code, "dev", 997)[:a.dev_per_language]):
            truncated += len(s) > a.max_len; dev_tokens.append(s[:a.max_len])
    if len(dev_tokens) != len(names) * a.dev_per_language:
        raise ValueError(f"{len(dev_tokens)} dev sentences, expected {len(names) * a.dev_per_language}: the dev pool is not the one of the core sample")
    batcher = lambda sents: p1.make_batches({0: sents}, bos, "mask", pad, a.max_len, a.batch_size, a.max_tokens)
    log("dev sentences", len(dev_tokens), "truncated", truncated, "features in the core", len(core["feature_ids"]), "states", labels, "task", a.task)

    setup("FP")
    probe = batcher(dev_tokens[:GATE_PROBE_SENTENCES])[0]; tokens, valid = probe["tokens"][:GATE_PROBE_SENTENCES].to(DEVICE), probe["valid"][:GATE_PROBE_SENTENCES].to(DEVICE)
    resid = lower(tokens); z_ref = upper(resid)
    gates = {"split_equals_full": bool(torch.equal(model(tokens), z_ref))}
    z_ref = z_ref[valid][:GATE_ROWS].float()
    gates["logits_native_residual"] = fr5.logit_agreement(z_ref, logits_fn(upper_final(resid)[valid][:GATE_ROWS]), min_top1=GATE_MIN_TOP1, max_abs=GATE_MAX_ABS)
    gates["logits_float32_residual"] = fr5.logit_agreement(z_ref, logits_fn(upper_final(resid.float())[valid][:GATE_ROWS]), min_top1=GATE_MIN_TOP1, max_abs=GATE_MAX_ABS)
    gates["probe_rows"] = int(min(int(valid.sum()), GATE_ROWS))
    del resid, z_ref; gc.collect(); torch.cuda.empty_cache()
    log("gates:", gates)
    if not gates["split_equals_full"] or not gates["logits_native_residual"]["ok"] or not gates["logits_float32_residual"]["ok"]:
        raise GateError(f"the split or the float32 logits do not reproduce the model: {gates}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = {"task": "transfer-" + a.task, "commit": a.commit, "model": model_name, "hook": hook, "states": labels, "chunk_features": a.chunk_features, "max_rows": a.max_rows, "grad_scales": a.grad_scales if a.task == "feature" else [], "gradient": "A9 (autograd gradient scalars per item)", "max_features": a.max_features, "max_contexts": a.max_contexts,
            "freeze_sha256": freeze_sha, "core_sha256": hashlib.sha256(Path(a.core).read_bytes()).hexdigest(), "gates": gates, "dev_sentences": len(dev_tokens), "truncated": truncated,
            "argv": list(argv) if argv is not None else None, "versions": fr.p1_package_versions(), "complete": False}
    (out / "meta_transfer.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    if a.task == "feature":
        res = run_transfer(labels, setup, lower, suffix, encode, sae.W_dec, core, dev_tokens, batcher, out, freeze_sha, a.chunk_features, device=DEVICE, log=log, max_features=a.max_features, max_contexts=a.max_contexts, max_rows=a.max_rows, grad_scales=a.grad_scales, grad_fn=grad_fn)
    else:
        man = json.loads(Path(a.manifest).read_text()); eval_ids = fr5.frozen_eval_ids(json.loads(Path(a.eval_freeze).read_text())); langs = names[:a.max_languages] if a.max_languages else names; eval_tokens = {}
        for code in langs:
            lines = pd_.read_flores_split(a.flores_root, code, "devtest", 1012)
            eval_tokens[code] = [s[:a.max_len] for s in pd_.encode_sentences(tok, [lines[j] for j in eval_ids])]
        res = run_group(labels, setup, lower, suffix, encode, sae.W_dec, man, langs, eval_tokens, batcher, out, freeze_sha, a.chunk_sentences, device=DEVICE, log=log, max_sentences=a.max_sentences, max_rows=a.max_rows, grad_fn=grad_fn)
    if not res["gates"]["sham_exact"]:
        raise GateError(f"the sham edit is not exactly zero (max abs {res['gates']['sham_max_abs']}): the unedited and the edited rows do not take identical computations")
    setup("FP")
    meta.update({"run": res, "setup_seconds": setup_seconds, "complete": True, "seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0})
    (out / "meta_transfer.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    log("done; items per state", {k: v["items"] for k, v in res["states"].items()}, "peak GPU GB", meta["peak_gpu_gb"])


if __name__ == "__main__":
    main()
