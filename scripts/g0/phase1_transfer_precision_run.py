"""Pure logic of the driver of the paired arithmetic-precision audit (protocol revision_R5_precision_audit.json; implementation plan docs/superpowers/plans/2026-10-27-precision-audit-task2-implementation.md, step 5a;
POST HOC): the plan of the audit from the frozen selection, the per-item rows from the captured vectors, the stop rules, the gate summary and the assembly of the results. No model, no GPU and no TransformerLens
are needed here, so all of it is tested locally; the glue that runs the model on Ada (step 5b) imports it.

Order of execution: the 128 selected features are processed in ROUND-ROBIN order over the FP-effect-norm quartiles (rank inside the quartile, then quartile), so that every chunk of 16 features holds four per
quartile and the instrument chunk is representative; the order only decides which features run first, it does not change the selection (amendment note in revision_R5_precision_audit.json).
"""
import phase1_transfer_precision as pp
import phase1_transfer_precision_analysis as pa
import phase1_transfer_precision_cross as px

CHUNK_FEATURES = 16
MEMORY_LIMIT_GB = 10.6
BUDGET_HOURS = 8.0
CELL_INDEX = {c: i for i, c in enumerate(px.CELLS)}


def quartile_of(freeze):
    """{feature: quartile} of the frozen selection."""
    return {e["feature"]: e["quartile"] for e in freeze["features"]}


def audit_order(freeze):
    """The selected features in round-robin order over the quartiles: sorted by (rank inside its quartile in the frozen order, quartile)."""
    seen, out = {}, []
    for e in freeze["features"]:
        rank = seen.get(e["quartile"], 0); seen[e["quartile"]] = rank + 1; out.append((rank, e["quartile"], e))
    return [e for _, _, e in sorted(out, key=lambda x: (x[0], x[1]))]


def plan_audit(freeze, core, core_sha256, chunk_features=CHUNK_FEATURES):
    """The sub-core of the selected features (the fields `phase1_transfer_run.run_transfer` reads, in audit order, with the frozen four contexts each) and the chunks of feature ids. The core must be the one of the
    freeze (sha256), contain every selected feature with the same direction, language index and leading contexts."""
    if freeze["inputs_sha256"]["core"] != core_sha256:
        raise ValueError(f"the core file ({core_sha256[:12]}) is not the core of the freeze ({freeze['inputs_sha256']['core'][:12]})")
    index = {int(f): i for i, f in enumerate(core["feature_ids"])}; order = audit_order(freeze); sub = {"feature_ids": [], "direction_ids": [], "feature_language_index": [], "contexts": []}
    for e in order:
        i = index.get(e["feature"])
        if i is None:
            raise ValueError(f"feature {e['feature']} of the freeze is not in the core")
        if int(core["direction_ids"][i]) != e["direction_id"]:
            raise ValueError(f"feature {e['feature']}: the direction of the core ({core['direction_ids'][i]}) differs from the freeze ({e['direction_id']})")
        if int(core["feature_language_index"][i]) != e["language_index"]:
            raise ValueError(f"feature {e['feature']}: the language index of the core differs from the freeze")
        if [int(s) for s in core["contexts"][i][:len(e["contexts"])]] != [int(s) for s in e["contexts"]]:
            raise ValueError(f"feature {e['feature']}: the leading contexts of the core differ from the freeze contexts")
        sub["feature_ids"].append(e["feature"]); sub["direction_ids"].append(e["direction_id"]); sub["feature_language_index"].append(e["language_index"]); sub["contexts"].append([int(s) for s in e["contexts"]])
    ids = sub["feature_ids"]
    return {"core": sub, "chunks": [ids[i:i + chunk_features] for i in range(0, len(ids), chunk_features)], "n_features": len(ids), "n_pairs": sum(len(c) for c in sub["contexts"])}


def rows_from_chunk(keys, vectors, outputs, state, quartiles, clusters):
    """The per-item rows of one chunk and state. keys: (feature, sentence, kind); vectors: {"bf16": observer.vectors, "fp32": ...}; outputs: {"bf16": process_items items, "fp32": ...} (the signed loss changes `dloss`
    in the order of the cells). An item in neither mode (a skipped context) is returned in the second value; an item in one mode only is a ValueError; a missing quartile or cluster is a KeyError."""
    rows, skipped = [], []
    for key in keys:
        have = [key in outputs[m] for m in px.MODES]
        if not any(have):
            skipped.append(key); continue
        if not all(have):
            raise ValueError(f"item {key} was produced in one mode only")
        f, s, kind = key; m = px.item_metrics(px.cross_gram(vectors, key)); row = {"feature": f, "sentence": s, "kind": kind, "state": state, "quartile": quartiles[f], "cluster": clusters[(f, s)], **m}
        for mode in px.MODES:
            d = outputs[mode][key]["dloss"]; row[f"dloss_FF_{mode}"] = d[CELL_INDEX["FP/FP"]]; row[f"dloss_QQ_{mode}"] = d[CELL_INDEX["Q/Q"]]
        rows.append(row)
    return rows, skipped


def stop_decision(first_chunk_seconds, n_chunks, peak_gb, memory_limit_gb=MEMORY_LIMIT_GB, budget_hours=BUDGET_HOURS):
    """The stop rules of the protocol on the instrument chunk: the cost of the first chunk (both modes, all states and executions) extrapolated to all chunks against the allowance in GPU hours, and the peak GPU
    memory against the limit. -> {"extrapolated_hours", "peak_gb", "memory_ok", "time_ok", "proceed", "why"}; infeasible runs are reported, never shrunk."""
    hours = first_chunk_seconds * n_chunks / 3600.0; mem_ok, time_ok = peak_gb <= memory_limit_gb, hours <= budget_hours; why = []
    if not mem_ok:
        why.append(f"memory: peak {peak_gb:.2f} GB above {memory_limit_gb} GB")
    if not time_ok:
        why.append(f"time: extrapolated {hours:.2f} GPU hours above {budget_hours}")
    return {"extrapolated_hours": hours, "peak_gb": peak_gb, "memory_ok": mem_ok, "time_ok": time_ok, "proceed": mem_ok and time_ok, "why": "; ".join(why)}


def gates_ok(gates):
    """G1: the hash round trip passed at every state change of the fp32 mode (and there was at least one); G2: the dtype probe of the fp32 mode shows float32 everywhere (both modes recorded); G3: the sham edit is
    exactly zero in both modes; G4: the logit agreement of the two modes passed. A missing gate counts as failed. -> {"ok", "failed": [names in order]}."""
    failed = []
    g1 = gates.get("G1")
    if not g1 or not all(r.get("ok") for r in g1):
        failed.append("G1")
    g2 = gates.get("G2")
    try:
        if not g2 or set(g2) != set(pp.MODES):
            raise AssertionError("modes missing")
        pp.assert_dtypes(g2["fp32"], "fp32")
    except AssertionError:
        failed.append("G2")
    g3 = gates.get("G3")
    if not g3 or not all(g3.get(m) is True for m in pp.MODES):
        failed.append("G3")
    g4 = gates.get("G4")
    if not g4 or g4.get("ok") is not True:
        failed.append("G4")
    return {"ok": not failed, "failed": failed}


def assemble_results(rows, control, old_floors, n_boot, seed):
    """From the per-item rows of all chunks (INT16 and INT4) and the batching control: the finalized rows (mode floors from the INT16 rows), the summaries, the readings A to D for the primary and the eligible
    population, the coverage and the control summary."""
    fin = pa.attach_batching(pa.finalize_rows(rows, old_floors), control); summary = pa.summarize(fin, n_boot, seed)
    return {"rows": fin, "summary": summary, "readings": {pop: pa.readings(summary, population=pop) for pop in ("all_real", "eligible_real_old")}, "coverage": pa.coverage(fin), "controls": pa.controls_summary(fin)}
