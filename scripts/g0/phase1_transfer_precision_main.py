"""Ada driver of the paired arithmetic-precision audit (protocol revision_R5_precision_audit.json; implementation plan docs/superpowers/plans/2026-10-27-precision-audit-task2-implementation.md, step 5b-2; POST HOC).

The pure helpers (aux file, skipped keys, direction kinds, probe paths, arguments) are tested locally. `main` needs TransformerLens, the SAE and a GPU and can only run on Ada: it is the integration test of the
audit (the instrument job, `--instrument`, runs ONE chunk of 16 features: memory, throughput, gates and the stop rules). It loads the model exactly as `phase1_transfer_run.main` does, builds the bf16-prefix caches
as `phase1_transfer_run.run_transfer` does (shared by both modes), and runs `phase1_transfer_precision_chunks.run_chunks` with a `process` built from `phase1_transfer_run.process_chunk` and a
`VectorObserver`. The frozen runner is not changed.

    python phase1_transfer_precision_main.py --out <dir> --flores-root <dir> [--instrument]
    python phase1_transfer_precision_main.py --build-aux --out <aux.json>      (local, once: the cluster map and the old floors for the 512 pairs)
"""
import argparse
import csv
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

KINDS = ("real", "matched_swapped")
LAYER_PROBES = (13, 25)
ROOT = Path(__file__).resolve().parents[2]


def expand_skipped(entries, kinds=KINDS):
    """The skipped (feature, sentence, kind) keys from the skipped entries of `process_chunk` ((feature, sentence, reason): a context without an FP-active position, all audited kinds) and `process_items` (((feature,
    sentence, kind), reason)); kinds that are not audited are dropped, repeats removed, order kept."""
    out = []
    for e in entries:
        if len(e) == 3:
            keys = [(e[0], e[1], k) for k in kinds]
        elif len(e) == 2 and len(e[0]) == 3:
            keys = [tuple(e[0])] if e[0][2] in kinds else []
        else:
            raise ValueError(f"unexpected skipped entry {e!r}")
        for k in keys:
            if k not in out:
                out.append(k)
    return out


@contextmanager
def direction_kinds(module, kinds):
    """Restrict `module.DIRECTION_KINDS` to `kinds` inside the block (the frozen runner builds items for every kind it lists) and restore it afterwards, also on an exception."""
    unknown = [k for k in kinds if k not in module.DIRECTION_KINDS]
    if unknown:
        raise ValueError(f"unknown direction kind {unknown[0]}")
    old = module.DIRECTION_KINDS; module.DIRECTION_KINDS = tuple(kinds)
    try:
        yield
    finally:
        module.DIRECTION_KINDS = old


def probe_paths(layers=LAYER_PROBES):
    """Gate G2: the hook points whose tensor dtype is recorded (the normalized input, q, the attention pattern and the MLP post-activation of the given suffix blocks)."""
    out = {}
    for n in layers:
        out.update({f"{n}.normalized": f"blocks.{n}.ln1.hook_normalized", f"{n}.q": f"blocks.{n}.attn.hook_q", f"{n}.pattern": f"blocks.{n}.attn.hook_pattern", f"{n}.mlp_post": f"blocks.{n}.mlp.hook_post"})
    return out


def build_aux(freeze, per_context_csv, floors, kinds=KINDS):
    """The small file shipped to Ada with the audit freeze: the cluster (source-sentence cluster of the main study) of every selected (feature, sentence) pair, read from the REAL rows of the main per-context table, and
    the old fixed floors of the audited kinds. A selected pair without a cluster, or with two different clusters, and a missing floor are errors."""
    missing = [k for k in kinds if k not in floors]
    if missing:
        raise ValueError(f"no floor for the kind {missing[0]}")
    want = {(e["feature"], s) for e in freeze["features"] for s in e["contexts"]}; got = {}
    with open(per_context_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] != "real":
                continue
            k = (int(r["feature"]), int(r["sentence"]))
            if k in want:
                c = int(float(r["cluster"]))
                if k in got and got[k] != c:
                    raise ValueError(f"the pair {k} has two clusters ({got[k]} and {c})")
                got[k] = c
    lost = sorted(want - set(got))
    if lost:
        raise ValueError(f"no cluster for the selected pair {lost[0]} in {per_context_csv}")
    return {"schema": 1, "kind": "intervention-transfer-precision-audit-aux", "selection_sha256": freeze.get("selection_sha256"), "clusters": {f"{f}|{s}": c for (f, s), c in sorted(got.items())},
            "old_floors": {k: float(floors[k]) for k in kinds}, "inputs_sha256": {"per_context_table": hashlib.sha256(Path(per_context_csv).read_bytes()).hexdigest()},
            "note": "clusters from the real rows of the main per-context table (transfer_runs/analysis_main/per_context_INT16.csv); old floors from results/main/analysis_summary.json (the FLORES floors of the main study, INT16 calibration items)"}


def load_aux(path):
    """-> ({(feature, sentence): cluster}, {kind: old floor})."""
    d = json.loads(Path(path).read_text()); return {tuple(int(x) for x in k.split("|")): int(v) for k, v in d["clusters"].items()}, {k: float(v) for k, v in d["old_floors"].items()}


def parse_args(argv=None):
    import phase1_transfer_run as tr
    ap = argparse.ArgumentParser(description="Paired arithmetic-precision audit of the suffix (revision R5): bf16 suffix against fp32 suffix, Gemma-2-2B layer 12, 512 selected pairs.")
    ap.add_argument("--out", required=True); ap.add_argument("--flores-root", required=False); ap.add_argument("--roster", default=str(ROOT / "reports/g0/phase1_languages.json"))
    ap.add_argument("--freeze", default=str(ROOT / "reports/intervention_transfer/freeze.json")); ap.add_argument("--core", default=str(ROOT / "reports/intervention_transfer/core_sample.json"))
    ap.add_argument("--audit-freeze", default=str(ROOT / "reports/intervention_transfer/precision_audit_freeze.json")); ap.add_argument("--aux", default=str(ROOT / "reports/intervention_transfer/precision_audit_aux.json"))
    ap.add_argument("--commit", default="unknown"); ap.add_argument("--instrument", action="store_true", help="run ONE chunk (16 features) and report the stop-rule extrapolation: the instrument job")
    ap.add_argument("--chunk-features", type=int, default=16); ap.add_argument("--dev-per-language", type=int, default=997); ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--max-tokens", type=int, default=1024); ap.add_argument("--gate-max-abs", type=float, default=tr.GATE_MAX_ABS)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=20261026); ap.add_argument("--build-aux", action="store_true", help="local: write the aux file from --table and the main analysis summary to --out")
    ap.add_argument("--table", default=None, help="with --build-aux: transfer_runs/analysis_main/per_context_INT16.csv"); ap.add_argument("--summary", default=str(ROOT / "reports/intervention_transfer/results/main/analysis_summary.json"))
    a = ap.parse_args(argv)
    if not a.build_aux and not a.flores_root:
        ap.error("--flores-root is required")
    return a


def prepare(states, setup, lower, encode, batcher, W_dec, sub_core, dev_tokens, device):
    """The caches of the audit, built exactly as `phase1_transfer_run.run_transfer` builds them (bf16 prefix; the FP caches and the cache of every state; positions and amplitudes keyed by (feature, dev
    sentence)); shared by both execution modes. -> dict with feats, ctxs, matched, pos_f, caches {state: {"FP", "Q"}}, amps {state: {"FP", "Q"}}, W_cpu."""
    import torch
    import phase1_func_f5_run as fr5
    feats = [int(f) for f in sub_core["feature_ids"]]; matched = {f: int(m) for f, m in zip(feats, sub_core["direction_ids"])}; ctxs = {f: [int(s) for s in c] for f, c in zip(feats, sub_core["contexts"])}
    uniq = sorted({s for f in feats for s in ctxs[f]}); local = {s: i for i, s in enumerate(uniq)}
    plan = {"sentences": [dev_tokens[s] for s in uniq], "dev_index": uniq, "ctx": {k: [local[s] for s in ctxs[f]] for k, f in enumerate(feats)}}
    fid = torch.tensor(feats, dtype=torch.long); batches = batcher(plan["sentences"]); W_cpu = W_dec.detach().float().cpu()
    keyed = lambda cache: ({uniq[l]: r for l, r in cache["resid"].items()}, {(feats[k], uniq[l]): a for (k, l), a in cache["act"].items()})
    setup("FP"); positions = fr5.fp_active_positions(lower, encode, batches, plan, fid, device); pos_f = {(feats[k], uniq[l]): p for (k, l), p in positions.items()}
    resid_fp, act_fp = keyed(fr5.cache_state(lower, encode, batches, plan, fid, positions, device)); caches, amps = {}, {}
    for label in states:
        setup(label); resid_q, act_q = keyed(fr5.cache_state(lower, encode, batches, plan, fid, positions, device)); caches[label] = {"FP": resid_fp, "Q": resid_q}; amps[label] = {"FP": act_fp, "Q": act_q}
    setup("FP")
    return {"feats": feats, "ctxs": ctxs, "matched": matched, "pos_f": pos_f, "caches": caches, "amps": amps, "W_cpu": W_cpu}


def main(argv=None):
    a = parse_args(argv)
    if a.build_aux:
        freeze = json.loads(Path(a.audit_freeze).read_text()); summ = json.loads(Path(a.summary).read_text()); floors = summ["feature_level"]["floors"]
        Path(a.out).write_text(json.dumps(build_aux(freeze, a.table, floors), indent=1)); print("aux written", a.out); return
    import gc, time
    import numpy as np
    import torch
    import phase1_transfer_run as tr
    import phase1_transfer_core as tc
    import phase1_transfer_precision as pp
    import phase1_transfer_precision_cross as px
    import phase1_transfer_precision_run as pr
    import phase1_transfer_precision_chunks as ch
    t0 = time.time(); log = lambda *m: print(f"[{time.time() - t0:7.0f}s]", *m, flush=True)
    freeze_text = Path(a.freeze).read_text(); freeze = json.loads(freeze_text); core = json.loads(Path(a.core).read_text()); tr.check_core_against_freeze(freeze, core); freeze_sha = hashlib.sha256(freeze_text.encode()).hexdigest()
    audit = json.loads(Path(a.audit_freeze).read_text()); core_sha = hashlib.sha256(Path(a.core).read_bytes()).hexdigest(); plan = pr.plan_audit(audit, core, core_sha, a.chunk_features); clusters, old_floors = load_aux(a.aux)
    audit_sha = hashlib.sha256(Path(a.audit_freeze).read_bytes()).hexdigest(); log("audit plan:", plan["n_features"], "features,", plan["n_pairs"], "pairs,", len(plan["chunks"]), "chunks of", a.chunk_features)

    from tqdm.auto import tqdm
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    import their_code as their
    import phase1_data as pd_
    import phase1_func_run as fr
    import phase1_func_f5_run as fr5
    import phase1_lib as p1
    import phase1_run as p1run
    torch.set_grad_enabled(False)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE != "cuda":
        raise SystemExit("no GPU visible: the audit refuses to run on the CPU")
    LAYER = fr.LAYER
    env = {"torch": torch, "np": np, "tqdm": tqdm, "gc": gc, "HookedTransformer": HookedTransformer, "SAE": SAE, "DEVICE": DEVICE, "WORK_DEVICE": DEVICE, "BATCH_SIZE": 1, "FIRING_THRESHOLD": 0.001}
    ex = their.extract_functions(their.PRISTINE, their.THEIR_FUNCTIONS); their.verify_hashes(ex, json.loads(their.MANIFEST.read_text())["functions"]); their.build_namespace(ex, env)
    log("their functions verified against the frozen manifest")
    model, model_name = env["load_gemma_tl_model"](p1run.MODEL_CANDIDATES); sae = env["load_sae_compat"](p1run.SAE_RELEASE, p1run.SAE_ID, DEVICE)
    hook = getattr(sae.cfg, "hook_name", None) or f"blocks.{LAYER}.hook_resid_post"
    assert hook == f"blocks.{LAYER}.hook_resid_post", f"unexpected SAE hook {hook}"
    assert sae.cfg.d_in == model.cfg.d_model
    env["model_work"] = model; env["original_state_cpu"] = env["clone_state_to_cpu"](model); tok = model.tokenizer; bos = tok.bos_token_id; pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    lower = lambda tokens: model(tokens, stop_at_layer=LAYER + 1); upper = lambda resid: model(resid, start_at_layer=LAYER + 1); upper_final, logits_fn = fr5.make_upper_final(model, LAYER), fr5.make_logits_fn(model)
    encode = lambda x: env["encode_sae"](sae, x); restore = env["restore_model_work"]; suffix_base = tr.make_suffix(upper_final, logits_fn, DEVICE)
    snap = pp.snapshot(model, LAYER)                                                                      # the stored precision of every suffix tensor, taken BEFORE any promotion
    log("model", model_name, "hook", hook, "d_sae", int(sae.cfg.d_sae), "suffix tensors", len(snap), "soft cap", model.cfg.output_logits_soft_cap, "dtype", model.cfg.dtype)

    def setup(label):
        restore()
        if fr5.state_bits(label) is not None:
            env["quantize_rtn_per_output_channel_tl"](model, bits=fr5.state_bits(label))
    names = [r["code"] for r in pd_.load_roster(a.roster)]
    if list(core["languages"]) != names:
        raise ValueError("the core sample was built for another language order")
    dev_tokens, truncated = [], 0
    for code in names:
        for s in pd_.encode_sentences(tok, pd_.read_flores_split(a.flores_root, code, "dev", 997)[:a.dev_per_language]):
            truncated += len(s) > a.max_len; dev_tokens.append(s[:a.max_len])
    if len(dev_tokens) != len(names) * a.dev_per_language:
        raise ValueError(f"{len(dev_tokens)} dev sentences, expected {len(names) * a.dev_per_language}")
    batcher = lambda sents: p1.make_batches({0: sents}, bos, "mask", pad, a.max_len, a.batch_size, a.max_tokens)
    # the gates of the main run (the split reproduces the model; the float32 readout reproduces the model's logits) in the bf16 mode with the FP weights
    setup("FP"); probe = batcher(dev_tokens[:tr.GATE_PROBE_SENTENCES])[0]; tokens, valid = probe["tokens"][:tr.GATE_PROBE_SENTENCES].to(DEVICE), probe["valid"][:tr.GATE_PROBE_SENTENCES].to(DEVICE)
    resid = lower(tokens); z_ref = upper(resid); main_gates = {"split_equals_full": bool(torch.equal(model(tokens), z_ref))}; z_ref = z_ref[valid][:tr.GATE_ROWS].float()
    main_gates["logits_float32_residual"] = fr5.logit_agreement(z_ref, logits_fn(upper_final(resid.float())[valid][:tr.GATE_ROWS]), min_top1=tr.GATE_MIN_TOP1, max_abs=a.gate_max_abs)
    log("main gates:", main_gates)
    if not main_gates["split_equals_full"] or not main_gates["logits_float32_residual"]["ok"]:
        raise tr.GateError(f"the split or the float32 logits do not reproduce the model: {main_gates}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); states = ["INT16", "INT4"]
    meta = {"task": "precision-audit", "commit": a.commit, "model": model_name, "hook": hook, "layer": LAYER, "freeze_sha256": freeze_sha, "audit_freeze_sha256": audit_sha, "selection_sha256": audit["selection_sha256"], "core_sha256": core_sha,
            "instrument": a.instrument, "chunk_features": a.chunk_features, "n_chunks": len(plan["chunks"]), "n_pairs": plan["n_pairs"], "states": states, "main_gates": main_gates, "versions": fr.p1_package_versions(), "complete": False, "argv": list(argv) if argv is not None else None}
    (out / "meta_running.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    prep = prepare(states, setup, lower, encode, batcher, sae.W_dec, plan["core"], dev_tokens, DEVICE); log("caches:", {s: len(prep["caches"][s]["Q"]) for s in states}, "contexts;", len(prep["pos_f"]), "feature-context pairs")
    g1 = []; gates = {"G1": g1}

    def mode_suffix_fn(mode):
        return pp.mode_suffix(model, LAYER, suffix_base, mode)

    # G1 pre-flight: the promotion round trip at the state changes of the fp32 mode (INT16 and INT4)
    pp.promote_suffix(model, LAYER); wrapped = pp.fp32_setup(setup, model, LAYER, snap, record=g1)
    try:
        wrapped("INT16"); wrapped("INT4")
    finally:
        pp.demote_suffix(model, LAYER, snap); setup("FP")
    # G2: dtypes at the probe points; G3: the zero-edit sham; G4: logit agreement of the two modes (FP weights, 512 rows)
    h_probe = resid.float()
    def run_probe(mode):
        if mode == "fp32":
            pp.promote_suffix(model, LAYER)
        try:
            return pp.dtype_probe(model, probe_paths(), lambda: mode_suffix_fn(mode)(h_probe[:1]))
        finally:
            if mode == "fp32":
                pp.demote_suffix(model, LAYER, snap)
    gates["G2"] = {m: run_probe(m) for m in px.MODES}; first_f = prep["feats"][0]; first_s = prep["ctxs"][first_f][0]; h0 = prep["caches"]["INT4"]["FP"][first_s].float(); rows0 = prep["pos_f"][(first_f, first_s)] + 1; tg0 = tr.context_targets(dev_tokens[first_s])
    gates["G3"] = {}
    for m in px.MODES:
        if m == "fp32":
            pp.promote_suffix(model, LAYER)
        try:
            e = tc.effect(h0, rows0, torch.zeros(len(rows0), h0.shape[1]), mode_suffix_fn(m), tg0); gates["G3"][m] = bool(float(e["vector"].abs().max()) == 0.0 and abs(e["dloss"]) == 0.0)
        finally:
            if m == "fp32":
                pp.demote_suffix(model, LAYER, snap)
    z_bf = logits_fn(upper_final(h_probe)[valid][:tr.GATE_ROWS]).float(); pp.promote_suffix(model, LAYER)
    try:
        z_32 = logits_fn(pp.mode_suffix(model, LAYER, upper_final, "fp32")(h_probe)[valid][:tr.GATE_ROWS]).float()
    finally:
        pp.demote_suffix(model, LAYER, snap); setup("FP")
    gates["G4"] = fr5.logit_agreement(z_bf, z_32, min_top1=0.90, max_abs=5.0); log("gates:", {k: (v if k != "G1" else f"{len(v)} state changes, ok {all(r['ok'] for r in v)}") for k, v in gates.items()}); verdict = pr.gates_ok(gates)
    (out / "gates_preflight.json").write_text(json.dumps(gates, indent=1, default=str))
    if not verdict["ok"]:
        raise tr.GateError(f"the audit gates failed: {verdict['failed']}")
    del z_bf, z_32; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    quartiles = pr.quartile_of(audit); dev = dev_tokens

    def process(chunk, state, mode, max_rows):
        ob = px.VectorObserver(); promoted = mode == "fp32"
        if promoted:
            pp.promote_suffix(model, LAYER)
        sw = pp.fp32_setup(setup, model, LAYER, snap, record=g1) if promoted else setup; suffix = mode_suffix_fn(mode)
        try:
            with direction_kinds(tr, KINDS):
                res = tr.process_chunk(chunk, {f: prep["ctxs"][f] for f in chunk}, prep["caches"][state], prep["pos_f"], prep["amps"][state], {f: tr.build_directions(prep["W_cpu"], f, prep["matched"][f], tr.RANDOM_SEEDS) for f in chunk},
                                       {s: dev[s] for f in chunk for s in prep["ctxs"][f]}, {"FP": suffix, "Q": suffix}, {"FP": lambda: sw("FP"), "Q": lambda: sw(state)}, max_rows=max_rows, observer=ob)
        finally:
            if promoted:
                pp.demote_suffix(model, LAYER, snap)
            setup("FP")
        return res["items"], ob.vectors, expand_skipped(res["skipped"])
    peak = lambda: torch.cuda.max_memory_allocated() / 1e9
    result = ch.run_chunks(plan, process, out / "chunks", audit_sha, quartiles, clusters, n_chunks_limit=1 if a.instrument else None, peak_gb=peak, log=log)
    gates["G1"] = g1; final = pr.gates_ok(gates); meta.update({"complete": True, "seconds": time.time() - t0, "peak_gpu_gb": peak(), "decision": result["decision"], "gates_ok": final, "chunk_seconds": result["chunk_seconds"], "n_rows": len(result["rows"]), "n_skipped": len(result["skipped"])})
    ch.write_results(result, old_floors, out / "results", a.n_boot, a.seed, gates, meta); log("done;", len(result["rows"]), "rows; decision", result["decision"], "gates", final)
    if not final["ok"]:
        raise tr.GateError(f"gates failed after the run: {final['failed']}")


if __name__ == "__main__":
    main()
