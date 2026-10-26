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
