"""Chunk loop of the paired arithmetic-precision audit (protocol revision_R5_precision_audit.json; implementation plan step 5b-1; POST HOC). The loop is independent of the model: it calls a `process` callable
(`process(chunk features, state, mode, max_rows) -> (outputs {item key: {"dloss": [...]}}, vectors {(item key, cell): tensor}, skipped keys)`), which on Ada is built from `phase1_transfer_run.process_chunk` with a
`VectorObserver`, so that everything around it (which states, modes and executions run, the rows, the repeated-batching control, the checkpoints and the resume, the stop rules, the written results) is tested
locally with a fake. Per chunk: every state in `states` in every mode, and the INT16 state a second time per mode with a smaller `max_rows` (control C1: different batch composition).
"""
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import phase1_transfer_precision_analysis as pa
import phase1_transfer_precision_cross as px
import phase1_transfer_precision_run as pr

KINDS = ("real", "matched_swapped")
STATES = ("INT16", "INT4")
MAX_ROWS = (256, 192)
CHECKPOINT_VERSION = "audit-chunk-v1"


class StopAudit(RuntimeError):
    """The stop rules of the protocol fired (peak memory or extrapolated cost): the run is reported infeasible, never shrunk. `.decision` holds `stop_decision`."""
    def __init__(self, decision):
        super().__init__("stop rules: " + decision["why"]); self.decision = decision


def _jsonable_key(key):
    return "|".join(str(x) for x in key)


def _control_to_json(control):
    return {m: {_jsonable_key(k): v for k, v in d.items()} for m, d in control.items()}


def _control_from_json(raw):
    out = {}
    for m, d in raw.items():
        out[m] = {}
        for k, v in d.items():
            f, s, kind = k.split("|"); out[m][(int(f), int(s), kind)] = v
    return out


def _atomic_json(obj, path):
    fd, tmp = tempfile.mkstemp(dir=str(Path(path).parent), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, default=float)
    os.replace(tmp, path)


def chunk_key(freeze_sha, chunk, contexts, kinds, states, modes, max_rows):
    """The key of a chunk checkpoint: the freeze, the features and their contexts, the protocol parameters and the version; a checkpoint with another key is never taken for this chunk."""
    payload = json.dumps([CHECKPOINT_VERSION, freeze_sha, list(chunk), [list(contexts[f]) for f in chunk], list(kinds), list(states), list(modes), list(max_rows)], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _empty_control(modes):
    return {m: {} for m in modes}


def process_one_chunk(chunk, process, ctxs, quartiles, clusters, kinds, states, modes, max_rows):
    """The rows, the control entries and the skipped keys of one chunk: per state every mode is processed, the INT16 state once more with `max_rows[1]`."""
    expected = [(f, s, k) for f in chunk for s in ctxs[f] for k in kinds]; rows, skipped, control = [], [], _empty_control(modes)
    for state in states:
        vecs, outs = {}, {}
        for mode in modes:
            out, vec, sk = process(chunk, state, mode, max_rows[0]); outs[mode], vecs[mode] = out, vec
            for k in sk:
                if k not in skipped:
                    skipped.append(k)
            if state == "INT16":
                out2, vec2, _ = process(chunk, state, mode, max_rows[1]); present = [k for k in expected if k in out and k in out2]; control[mode].update(px.batching_difference(vec, vec2, present)); del out2, vec2
        r, _ = pr.rows_from_chunk(expected, vecs, outs, state, quartiles, clusters); rows += r; del vecs, outs
    return rows, control, skipped


def run_chunks(plan, process, out_dir, freeze_sha, quartiles, clusters, kinds=KINDS, states=STATES, modes=px.MODES, max_rows=MAX_ROWS, n_chunks_limit=None, clock=time.time, peak_gb=lambda: 0.0, log=print):
    """Run (or resume) the chunks of the plan. A checkpoint `audit_chunk_NNN.json` per chunk holds the rows, the control entries and the skipped keys. After the first chunk PROCESSED in this call the stop rules
    are evaluated (`pr.stop_decision` with the seconds of that chunk, the number of chunks of the plan and `peak_gb()`): a full run raises `StopAudit` if they fail, a run limited by `n_chunks_limit` (the instrument
    run) only reports the decision. -> {"rows", "control", "skipped", "chunk_seconds", "decision"}."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); ctxs = dict(zip(plan["core"]["feature_ids"], plan["core"]["contexts"])); rows, skipped, control, seconds, decision = [], [], _empty_control(modes), [], None; n_total = len(plan["chunks"])
    for ci, chunk in enumerate(plan["chunks"]):
        if n_chunks_limit is not None and ci >= n_chunks_limit:
            break
        path = out / f"audit_chunk_{ci:03d}.json"; key = chunk_key(freeze_sha, chunk, ctxs, kinds, states, modes, max_rows); loaded = None
        if path.exists():
            raw = json.loads(path.read_text())
            if raw.get("key") == key:
                loaded = raw
        if loaded is None:
            t0 = clock(); r, c, sk = process_one_chunk(chunk, process, ctxs, quartiles, clusters, kinds, states, modes, max_rows); dt = clock() - t0; gb = peak_gb()
            loaded = json.loads(json.dumps({"key": key, "rows": r, "control": _control_to_json(c), "skipped": [list(k) for k in sk], "seconds": dt, "peak_gb": gb}, default=float)); _atomic_json(loaded, path); seconds.append(dt)
            log(f"chunk {ci + 1}/{n_total}: {len(r)} rows, {dt:.0f} s, peak GPU {gb:.2f} GB")
            if decision is None:
                decision = pr.stop_decision(dt, n_total, gb)
                if not decision["proceed"] and n_chunks_limit is None:
                    raise StopAudit(decision)
        rows += loaded["rows"]; skipped += [tuple(k) for k in loaded["skipped"] if tuple(k) not in skipped]
        for m, d in _control_from_json(loaded["control"]).items():
            control[m].update(d)
    return {"rows": rows, "control": control, "skipped": skipped, "chunk_seconds": seconds, "decision": decision}


def write_results(result, old_floors, out_dir, n_boot, seed, gates, meta):
    """Assemble (`pr.assemble_results`) and write `per_item.csv`, `summary.csv`, `readings.json` (readings, control summary, coverage, stop decision, skipped items), `gates.json` and `meta.json`. -> the summary."""
    res = pr.assemble_results(result["rows"], result["control"], old_floors, n_boot, seed); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); pa.write_tables(res["rows"], res["summary"], out)
    cov = {"|".join(k): v for k, v in res["coverage"].items()}
    (out / "readings.json").write_text(json.dumps({"readings": res["readings"], "controls": res["controls"], "coverage": cov, "decision": result["decision"], "skipped": [list(k) for k in result["skipped"]], "n_boot": n_boot, "seed": seed}, indent=1, default=float))
    (out / "gates.json").write_text(json.dumps(gates, indent=1, default=str)); (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return res["summary"]
