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


def process_chunk(features, contexts, caches, positions, amplitudes, dirs, tokens, suffixes, switches, center=True):
    """The items of one chunk.
    features: feature ids; contexts: {feature: [dev sentence index]}; caches: {"FP": {sentence: [n+1, d] residual}, "Q": {...}} (any float dtype; converted to float32);
    positions: {(feature, sentence): content positions where the feature is active in FP}; amplitudes: {"FP": {(feature, sentence): [len(positions)]}, "Q": {...}} (the activations of the feature at those positions
    in the FP and in the quantized state); dirs: {feature: {kind: [d] vector}}; tokens: {sentence: content token ids}; suffixes: {"FP": f, "Q": f} (callables [B, L, d] -> [B, L, V]);
    switches: {"FP": f, "Q": f} (put the model weights in that state; called once each, FP first).
    -> {"items": {(feature, sentence, kind): {"gram", "dloss", "loss_base", "n_rows", "n_active_q", "n_positions"}}, "pooled": {(feature, kind): {"gram", "n_rows", "n_contexts"}}, "skipped": [(feature, sentence, reason)]}."""
    jobs, skipped = [], []
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
            jobs.append((f, s))
    prep = {}
    for f, s in jobs:
        rows = positions[(f, s)] + 1; tg = context_targets(tokens[s]); prep[(f, s)] = (rows, tg, caches["FP"][s].float(), caches["Q"][s].float())
    fp_vectors = {}
    switches["FP"]()
    for f, s in jobs:
        rows, tg, h_fp, h_q = prep[(f, s)]
        for kind in DIRECTION_KINDS:
            edit = amplitudes["FP"][(f, s)].float()[:, None] * dirs[f][kind].float()[None, :]
            try:
                fp_vectors[(f, s, kind)] = {"FP/FP": _cpu(tc.effect(h_fp, rows, edit, suffixes["FP"], tg, center)), "Q/FP": _cpu(tc.effect(h_q, rows, edit, suffixes["FP"], tg, center))}
            except ValueError as e:
                raise ValueError(f"context {(f, s, kind)}: {e}") from None
    items, sums, nrows = {}, {}, {}
    switches["Q"]()
    for f, s in jobs:
        rows, tg, h_fp, h_q = prep[(f, s)]
        for kind in DIRECTION_KINDS:
            d = dirs[f][kind].float()
            edit = amplitudes["FP"][(f, s)].float()[:, None] * d[None, :]; native = amplitudes["Q"][(f, s)].float()[:, None] * d[None, :]
            try:
                cells = dict(fp_vectors.pop((f, s, kind)))
                cells["FP/Q"] = _cpu(tc.effect(h_fp, rows, edit, suffixes["Q"], tg, center)); cells["Q/Q"] = _cpu(tc.effect(h_q, rows, edit, suffixes["Q"], tg, center))
                cells["Q/Q-native"] = _cpu(tc.effect(h_q, rows, native, suffixes["Q"], tg, center))
            except ValueError as e:
                raise ValueError(f"context {(f, s, kind)}: {e}") from None
            n = cells["FP/FP"]["n_rows"]
            items[(f, s, kind)] = {"gram": tc.gram(cells).numpy().astype(np.float64), "dloss": [cells[c]["dloss"] for c in tc.CELLS], "loss_base": [cells[c]["loss_base"] for c in tc.CELLS], "n_rows": n,
                                   "n_active_q": int((amplitudes["Q"][(f, s)] > 0).sum()), "n_positions": int(len(positions[(f, s)]))}
            acc = sums.setdefault((f, kind), {c: 0 for c in tc.CELLS}); nrows[(f, kind)] = nrows.get((f, kind), 0) + n
            for c in tc.CELLS:
                acc[c] = acc[c] + n * cells[c]["vector"].double()
    pooled = {}
    for key, acc in sums.items():
        n = nrows[key]; avg = {c: {"vector": acc[c] / n} for c in tc.CELLS}
        pooled[key] = {"gram": tc.gram(avg).numpy().astype(np.float64), "n_rows": n, "n_contexts": sum(1 for (f, s, k) in items if (f, k) == key)}
    return {"items": items, "pooled": pooled, "skipped": skipped}


def _cpu(e):
    return {**e, "vector": e["vector"].detach().cpu()}


def subset_core(core, max_features, max_contexts):
    """The smoke subset of the instrument job: evenly spaced features (first and last included), the first `max_contexts` contexts of each (the full core when both are None)."""
    n = len(core["feature_ids"]); idx = list(range(n))
    if max_features is not None and max_features < n:
        idx = sorted(set(np.linspace(0, n - 1, max_features).round().astype(int).tolist()))
    out = {k: [core[k][i] for i in idx] for k in ("feature_ids", "direction_ids", "feature_language_index", "contexts")}
    if max_contexts is not None:
        out["contexts"] = [c[:max_contexts] for c in out["contexts"]]
    return out


def run_transfer(states, setup, lower, suffix, encode, W_dec, core, dev_tokens, batcher, out_dir, freeze_sha, per_chunk, device="cpu", log=lambda *a: None, max_features=None, max_contexts=None):
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
                                {s: dev_tokens[s] for f in chunk for s in ctxs[f]}, {"FP": suffix, "Q": suffix}, {"FP": lambda: setup("FP"), "Q": lambda l=label: setup(l)})
            atomic_save({**res, "features": chunk}, path, key); st["items"] += len(res["items"]); st["skipped"] += len(res["skipped"]); st["seconds"] += time.time() - t1
            log(f"state {label} chunk {ci + 1}/{len(chunks)}: {len(res['items'])} items, {time.time() - t1:.0f} s")
    setup("FP"); meta["seconds"] = time.time() - t0
    return meta
