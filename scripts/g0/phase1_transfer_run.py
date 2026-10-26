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


def process_items(items, suffixes, switches, center=True):
    """The five cells of the crossed design for a list of items, each with its OWN edits (the feature-level run builds them from one feature's amplitude times a direction, the group-level run from the sum over a
    feature group). item: {"key" (unique), "rows" (edited rows of the cached context), "h_fp", "h_q" ([L, d] float32 prefix residuals), "edit_fp" (the FP edit at those rows, used in the cells FP/FP, Q/FP, FP/Q,
    Q/Q), "edit_native" (the quantized state's own edit, cell Q/Q-native), "targets" ([L], -1 = none), "pool" (key of the pool the item belongs to, or None), "n_active_q" (optional count)}.
    Order of work: switches["FP"](), the FP-suffix cells of every item (vectors kept on the host), switches["Q"](), the Q-suffix cells, then the Gram matrices.
    -> {"items": {key: {"gram" (5 x 5 float64), "dloss", "loss_base" (five values each), "n_rows", "n_target_rows", "n_active_q", "n_positions"}}, "pooled": {pool: {"gram", "n_rows", "n_contexts"}}, "skipped": [(key, reason)]}."""
    keys = [it["key"] for it in items]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate item keys: {sorted({k for k in keys if keys.count(k) > 1}, key=str)[:3]}")
    jobs, skipped = [], []
    for it in items:
        if len(it["rows"]) == 0:
            skipped.append((it["key"], "no_edited_row"))
        else:
            jobs.append(it)
    fp_vectors = {}
    switches["FP"]()
    for it in jobs:
        try:
            fp_vectors[it["key"]] = {"FP/FP": _cpu(tc.effect(it["h_fp"], it["rows"], it["edit_fp"], suffixes["FP"], it["targets"], center)),
                                     "Q/FP": _cpu(tc.effect(it["h_q"], it["rows"], it["edit_fp"], suffixes["FP"], it["targets"], center))}
        except ValueError as e:
            raise ValueError(f"context {it['key']}: {e}") from None
    out, sums, nrows, counts = {}, {}, {}, {}
    switches["Q"]()
    for it in jobs:
        key = it["key"]
        try:
            cells = dict(fp_vectors.pop(key))
            cells["FP/Q"] = _cpu(tc.effect(it["h_fp"], it["rows"], it["edit_fp"], suffixes["Q"], it["targets"], center)); cells["Q/Q"] = _cpu(tc.effect(it["h_q"], it["rows"], it["edit_fp"], suffixes["Q"], it["targets"], center))
            cells["Q/Q-native"] = _cpu(tc.effect(it["h_q"], it["rows"], it["edit_native"], suffixes["Q"], it["targets"], center))
        except ValueError as e:
            raise ValueError(f"context {key}: {e}") from None
        n = cells["FP/FP"]["n_rows"]
        out[key] = {"gram": tc.gram(cells).numpy().astype(np.float64), "dloss": [cells[c]["dloss"] for c in tc.CELLS], "loss_base": [cells[c]["loss_base"] for c in tc.CELLS], "n_rows": n,
                    "n_target_rows": int((it["targets"] >= 0).sum()), "n_active_q": it.get("n_active_q"), "n_positions": int(len(it["rows"]))}
        pool = it.get("pool")
        if pool is not None:
            acc = sums.setdefault(pool, {c: 0 for c in tc.CELLS}); nrows[pool] = nrows.get(pool, 0) + n; counts[pool] = counts.get(pool, 0) + 1
            for c in tc.CELLS:
                acc[c] = acc[c] + n * cells[c]["vector"].double()
    pooled = {}
    for pool, acc in sums.items():
        avg = {c: {"vector": acc[c] / nrows[pool]} for c in tc.CELLS}
        pooled[pool] = {"gram": tc.gram(avg).numpy().astype(np.float64), "n_rows": nrows[pool], "n_contexts": counts[pool]}
    return {"items": out, "pooled": pooled, "skipped": skipped}


def process_chunk(features, contexts, caches, positions, amplitudes, dirs, tokens, suffixes, switches, center=True):
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
                              "targets": tg, "pool": (f, kind), "n_active_q": n_act})
    res = process_items(items, suffixes, switches, center)
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
        items.append({"key": (code, sentence, kind), "rows": rows, "h_fp": h_fp, "h_q": h_q, "edit_fp": e_fp, "edit_native": e_q, "targets": targets, "pool": (code, kind), "n_active_q": n_act})
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


def main(argv=None):
    import argparse, gc, time
    ROOT = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Intervention-transfer study (D-088): the feature-level core, Gemma-2-2B layer 12, states FP and the quantized ones (instrument job or full run).")
    ap.add_argument("--out", required=True); ap.add_argument("--flores-root", required=True); ap.add_argument("--roster", default=str(ROOT / "reports/g0/phase1_languages.json"))
    ap.add_argument("--freeze", default=str(ROOT / "reports/intervention_transfer/freeze.json")); ap.add_argument("--core", default=str(ROOT / "reports/intervention_transfer/core_sample.json"))
    ap.add_argument("--states", nargs="+", default=["16", "6", "4"], help="quantized states (bit widths; FP is always the reference); 16 = the rounding-only floor")
    ap.add_argument("--chunk-features", type=int, default=25); ap.add_argument("--max-features", type=int, help="evenly spaced subset of the core (instrument job)"); ap.add_argument("--max-contexts", type=int, help="first contexts of each feature (instrument job)")
    ap.add_argument("--dev-per-language", type=int, default=997); ap.add_argument("--max-len", type=int, default=256); ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--commit", default="unknown")
    a = ap.parse_args(argv)
    t0 = time.time(); log = lambda *m: print(f"[{time.time() - t0:7.0f}s]", *m, flush=True)
    freeze_text = Path(a.freeze).read_text(); freeze = json.loads(freeze_text); core = json.loads(Path(a.core).read_text()); check_core_against_freeze(freeze, core)
    freeze_sha = hashlib.sha256(freeze_text.encode()).hexdigest()

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
    log("dev sentences", len(dev_tokens), "truncated", truncated, "features in the core", len(core["feature_ids"]), "states", labels)

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
    meta = {"task": "transfer", "commit": a.commit, "model": model_name, "hook": hook, "states": labels, "chunk_features": a.chunk_features, "max_features": a.max_features, "max_contexts": a.max_contexts,
            "freeze_sha256": freeze_sha, "core_sha256": hashlib.sha256(Path(a.core).read_bytes()).hexdigest(), "gates": gates, "dev_sentences": len(dev_tokens), "truncated": truncated,
            "argv": list(argv) if argv is not None else None, "versions": fr.p1_package_versions(), "complete": False}
    (out / "meta_transfer.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    res = run_transfer(labels, setup, lower, suffix, encode, sae.W_dec, core, dev_tokens, batcher, out, freeze_sha, a.chunk_features, device=DEVICE, log=log, max_features=a.max_features, max_contexts=a.max_contexts)
    if not res["gates"]["sham_exact"]:
        raise GateError(f"the sham edit is not exactly zero (max abs {res['gates']['sham_max_abs']}): the unedited and the edited rows do not take identical computations")
    setup("FP")
    meta.update({"run": res, "setup_seconds": setup_seconds, "complete": True, "seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0})
    (out / "meta_transfer.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    log("done; items per state", {k: v["items"] for k, v in res["states"].items()}, "peak GPU GB", meta["peak_gpu_gb"])


if __name__ == "__main__":
    main()
