"""D-035 Task 1 (plan: docs/superpowers/plans/2026-10-21-language-association-retention.md): bounded inventory of what the saved Phase 1 files can support.
Decides from the SCHEMA of saved files (keys, shapes, dtypes, language order, sha256; never values) which quantities of the association-retention study are
available / derivable on the CPU / incomplete / need a new collection, and rejects inconsistent provenance (another language order, feature count, array shape).
The pure logic works on file records (dictionaries); `describe_pt` / `build_inventory` build the records from real files and need torch (imported lazily).
Sidecar contract (written by the Task 2 collector, checked here): profiles.pt = {"schema_version": 1, "languages": [codes], "feature_ids": int64 [F],
"conditions": {cond: {"n_tokens": int64 [L], "positive_counts": int64 [L, F], "activation_sums": float64 [L, F], "activation_max": float64 [L, F]}}}."""
import hashlib, json
from pathlib import Path

SCHEMA_VERSION = 1
REQUIRED_CONDITIONS = ["FP", "INT8", "INT7", "INT6", "INT5", "INT4", "INT16", "INT12", "INT0"]       # plan: the study ladder plus FP and the numerical controls
PROFILE_FIELDS = ("n_tokens", "positive_counts", "activation_sums", "activation_max")
QUANTITIES = ["fp_language_profiles", "quantized_language_profiles", "all_language_correlations", "fp_tagging_cohort", "quantized_peak_and_selectivity",
              "sentence_level_summaries", "s_vs_nu_cohort_comparison", "fp_baseline_on_devtest"]
SELECTIVITY_KEYS = ("rates", "mean_activation", "S", "S_rank", "peak", "supported", "selective", "controls", "nu", "stable_count", "strata")
PROFILE_KEYS = ("firing_rate", "mean_activation", "max_activation")


class InventoryError(ValueError):
    pass


# ------------------------------------------------------------------ describing files (schema only)
def _describe(obj, depth=0):
    """Schema of a loaded object: tensors -> shape/dtype, dicts/lists/scalars structurally; array values are never copied."""
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return {"type": "tensor", "shape": [int(s) for s in obj.shape], "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {"type": "dict", "items": {str(k): _describe(v, depth + 1) for k, v in obj.items()}}
    if isinstance(obj, (list, tuple)):
        kinds = {type(x).__name__ for x in obj}; elem = kinds.pop() if len(kinds) == 1 else ("mixed" if kinds else "empty")
        d = {"type": "list", "len": len(obj), "elem": elem}
        if elem == "str":
            d["values"] = [str(x) for x in obj]
        return d
    if isinstance(obj, (bool, int, float, str)):
        d = {"type": "scalar", "kind": type(obj).__name__}
        if isinstance(obj, (bool, int)) or (isinstance(obj, str) and len(obj) <= 64):
            d["value"] = obj
        return d
    return {"type": type(obj).__name__}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe_pt(path):
    """Record of a .pt file (needs torch; weights_only=True, so no pickled code runs): sha256, size and the schema of its top-level keys."""
    import torch
    path = Path(path)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:                       # pickled code or a corrupt file: refuse, never fall back to an unsafe load
        raise InventoryError(f"{path.name}: cannot be loaded as plain tensors and containers ({type(e).__name__})") from e
    if not isinstance(obj, dict):
        raise InventoryError(f"{path.name}: expected a dictionary at top level")
    return {"path": str(path), "name": path.name, "sha256": sha256_file(path), "size": path.stat().st_size, "top": {str(k): _describe(v) for k, v in obj.items()}}


def describe_json(path):
    path = Path(path); obj = json.loads(path.read_text())
    return {"path": str(path), "name": path.name, "sha256": sha256_file(path), "size": path.stat().st_size,
            "top": {str(k): _describe(v) for k, v in obj.items()} if isinstance(obj, dict) else {}}


def build_inventory(paths, expected_languages=None, required_conditions=REQUIRED_CONDITIONS):
    recs = [describe_pt(p) if str(p).endswith(".pt") else describe_json(p) for p in paths]
    return inventory_from_records(recs, expected_languages, required_conditions)


# ------------------------------------------------------------------ pure logic on records
def _items(desc):
    return desc.get("items", {}) if desc and desc.get("type") == "dict" else {}


def _shape(desc):
    return tuple(desc["shape"]) if desc and desc.get("type") == "tensor" else None


def _kind(top):
    if {"corr", "loss_sum", "pooled_corr"} <= set(top):
        return "legacy_survival"
    if {"nu", "selective", "peak"} <= set(top):
        return "legacy_selectivity"
    if {"schema_version", "conditions"} <= set(top):
        return "sidecar_profiles"
    return "unknown"


def _languages(rec):
    d = rec["top"].get("languages")
    return list(d.get("values", [])) if d and d.get("type") == "list" else None


def _check_arrays(rec, kind, langs, errors):
    """Collect the feature count of every [L, F] / [F] array of a record and reject arrays whose language dimension disagrees with the language list."""
    n_l, feats, top, name = len(langs) if langs else None, set(), rec["top"], rec["name"]
    def lf(label, desc):
        s = _shape(desc)
        if s is None:
            return
        if len(s) != 2 or (n_l is not None and s[0] != n_l):
            errors.append(f"{name}: {label} has shape {list(s)}; expected [{n_l}, F] (shape of the language dimension)")
        else:
            feats.add(s[1])
    def f1(label, desc):
        s = _shape(desc)
        if s is not None:
            if len(s) != 1:
                errors.append(f"{name}: {label} has shape {list(s)}; expected [F] (shape)")
            else:
                feats.add(s[0])
    if kind == "legacy_survival":
        for b, d in _items(top.get("corr")).items():
            lf(f"corr[{b}]", d)
        for k in PROFILE_KEYS:
            lf(k, top.get(k))
        f1("pooled_firing_rate", top.get("pooled_firing_rate"))
    elif kind == "legacy_selectivity":
        for k in ("rates", "mean_activation", "nu"):
            lf(k, top.get(k))
        for k in ("S", "S_rank", "peak", "supported", "selective", "stable_count", "strata"):
            f1(k, top.get(k))
    elif kind == "sidecar_profiles":
        if top["schema_version"].get("value") != SCHEMA_VERSION:
            errors.append(f"{name}: unsupported schema version {top['schema_version'].get('value')!r}; expected {SCHEMA_VERSION}")
        f1("feature_ids", top.get("feature_ids"))
        for c, cd in _items(top["conditions"]).items():
            for k, d in _items(cd).items():
                if k == "n_tokens":
                    s = _shape(d)
                    if s is not None and (len(s) != 1 or s[0] != n_l):
                        errors.append(f"{name}: {c}.n_tokens has shape {list(s)}; expected [{n_l}] (shape)")
                else:
                    lf(f"{c}.{k}", d)
    return feats


def _complete_profile_conditions(rec, required):
    """-> (complete conditions, {condition: missing fields}) of a sidecar profiles record."""
    conds = _items(rec["top"]["conditions"]); complete, missing = [], {}
    for c in required:
        fields = set(_items(conds.get(c))) if c in conds else set(PROFILE_FIELDS)
        absent = [f for f in PROFILE_FIELDS if f not in fields] if c in conds else list(PROFILE_FIELDS)
        (missing.__setitem__(c, absent) if absent else complete.append(c))
    return complete, missing


def inventory_from_records(records, expected_languages=None, required_conditions=REQUIRED_CONDITIONS):
    recs = [{**r, "kind": _kind(r["top"])} for r in records]; errors, langs_seen, feats = [], {}, {}
    for r in recs:
        langs = _languages(r)
        if langs is not None:
            if len(set(langs)) != len(langs):
                raise InventoryError(f"{r['name']}: duplicate language codes")
            langs_seen[r["name"]] = langs
        r["_feats"] = _check_arrays(r, r["kind"], langs, errors)
        if r["_feats"]:
            feats[r["name"]] = r["_feats"]
    if errors:
        raise InventoryError("; ".join(errors))
    languages = next(iter(langs_seen.values()), None)
    for name, l in langs_seen.items():
        if l != languages:
            raise InventoryError(f"language order or roster of {name} differs from the other files ({l} against {languages})")
    if expected_languages is not None and languages is not None and list(expected_languages) != languages:
        raise InventoryError(f"languages {languages} differ from the expected roster {list(expected_languages)}")
    all_feats = set().union(*feats.values()) if feats else set()
    if len(all_feats) > 1:
        raise InventoryError(f"feature count differs between files or arrays (array shape mismatch): {sorted(all_feats)}")
    n_features = next(iter(all_feats), None)
    by_kind = {}
    for r in recs:
        by_kind.setdefault(r["kind"], []).append(r)
    surv, sel, side = (by_kind.get(k, [None])[0] for k in ("legacy_survival", "legacy_selectivity", "sidecar_profiles"))
    q = {}

    def say(name, verdict, evidence):
        q[name] = {"verdict": verdict, "evidence": evidence}

    # response profiles of the full-precision features
    side_complete, side_missing = _complete_profile_conditions(side, required_conditions) if side else ([], {})
    if side and "FP" in side_complete:
        say("fp_language_profiles", "available", f"{side['name']} condition FP (counts, sums, maxima per language and feature)")
    elif surv and all(k in surv["top"] for k in PROFILE_KEYS):
        say("fp_language_profiles", "available", f"{surv['name']}: firing_rate, mean_activation, max_activation (full-precision side only)")
    else:
        say("fp_language_profiles", "new_collection_required", "no file holds full-precision per-language firing, mean and maximum arrays")
    # response profiles of every quantized condition
    if side and not side_missing:
        say("quantized_language_profiles", "available", f"{side['name']}: all {len(required_conditions)} conditions with {', '.join(PROFILE_FIELDS)}")
    elif side:
        say("quantized_language_profiles", "incomplete", f"{side['name']} lacks " + "; ".join(f"{c}: {', '.join(m)}" for c, m in side_missing.items()))
    else:
        note = "; the saved per-rung arrays are correlations or full-precision-side statistics, and a correlation is not a response profile" if surv else ""
        say("quantized_language_profiles", "new_collection_required", "no sidecar profiles.pt with quantized counts, sums and maxima" + note)
    corr = _items(surv["top"].get("corr")) if surv else {}
    say("all_language_correlations", "available" if corr else "new_collection_required",
        f"{surv['name']}: corr for rungs {sorted(corr)} at every language (historical Pearson, not a response profile)" if corr else "no all-language correlation arrays")
    # frozen full-precision tagging cohort
    if sel:
        absent = [k for k in SELECTIVITY_KEYS if k not in sel["top"]]
        say("fp_tagging_cohort", "incomplete" if absent else "available",
            f"{sel['name']} lacks {', '.join(absent)}" if absent else f"{sel['name']}: {', '.join(SELECTIVITY_KEYS)}")
    else:
        say("fp_tagging_cohort", "new_collection_required", "no selectivity.pt")
    qs = q["quantized_language_profiles"]["verdict"]
    say("quantized_peak_and_selectivity", "derivable" if qs == "available" else ("incomplete" if qs == "incomplete" else "new_collection_required"),
        "computed on the CPU from the quantized profiles" if qs == "available" else "needs the quantized profiles first")
    say("sentence_level_summaries", "new_collection_required", "paired uncertainty needs per-sentence counts and sums; no saved file holds them (Task 2 sidecar shards)")
    can_nu = sel and all(k in sel["top"] for k in ("S", "selective", "peak")) and ("nu" in sel["top"] or "mean_activation" in sel["top"])
    say("s_vs_nu_cohort_comparison", "derivable" if can_nu else "new_collection_required",
        f"{sel['name']}: S, selective, peak and nu (or mean activations) on the full-precision tagging" if can_nu else "needs S, selective, peak and nu in selectivity.pt")
    can_base = sel and surv and all(k in sel["top"] for k in ("selective", "peak")) and all(k in surv["top"] for k in ("firing_rate", "n"))
    say("fp_baseline_on_devtest", "derivable" if can_base else "new_collection_required",
        "dev peak labels (selectivity.pt) against the full-precision devtest firing rates and token counts (survival.pt)" if can_base else "needs both selectivity.pt and survival.pt")
    return {"schema_version": SCHEMA_VERSION, "languages": languages, "n_features": n_features, "required_conditions": list(required_conditions),
            "files": [{k: v for k, v in r.items() if k in ("path", "name", "sha256", "size", "kind")} for r in recs], "quantities": q}


def _file_record(path):
    return {"sha256": sha256_file(path), "size": Path(path).stat().st_size}


def frozen_manifest(run_dir, min_pos=50, min_panel_rate=1e-5):
    """Provenance and cohort facts of an original run directory (mask/eq: meta.json, selectivity.pt, survival.pt), computed on the CPU from the saved files. Holds hashes, counts
    and run settings, never feature-id lists (the ids stay in selectivity.pt, identified by its hash). The baseline block applies the plan's establishment conditions to the
    full-precision devtest profile: the dev peak language is the unique devtest peak, has at least `min_pos` positive tokens (positives = rate * n, ROUNDED: a float32 rate of
    exactly 50 positives times n can fall just below 50) and the panel-mean rate is at least `min_panel_rate`."""
    import torch
    d = Path(run_dir); meta = json.loads((d / "meta.json").read_text())
    try:
        sel = torch.load(d / "selectivity.pt", map_location="cpu", weights_only=True); sur = torch.load(d / "survival.pt", map_location="cpu", weights_only=True)
    except Exception as e:
        raise InventoryError(f"cannot load the saved run files as plain tensors ({type(e).__name__})") from e
    langs = list(sur["languages"])
    if list(sel["languages"]) != langs:
        raise InventoryError(f"language order differs between selectivity.pt {list(sel['languages'])} and survival.pt {langs}")
    n = torch.tensor(sur["n"], dtype=torch.float64)
    budget = (meta.get("equal_budget") or {}).get("devtest_tokens_per_language")
    if budget is not None and not bool((n == budget).all()):
        raise InventoryError(f"devtest token counts {sur['n']} disagree with the recorded equal budget {budget}")
    selective, supported, peak = sel["selective"].bool(), sel["supported"].bool(), sel["peak"].long()
    controls = sel["controls"].long(); n_feat = int(selective.shape[0])
    union = selective.clone(); union[controls] = True
    sidx = selective.nonzero().squeeze(1); l0 = peak[sidx]
    per_lang = torch.bincount(l0, minlength=len(langs))
    dev = sel["rates"].double()[:, sidx]; dev_ties = int(((dev == dev.max(0).values[None, :]).sum(0) > 1).sum())
    counts = torch.round(sur["firing_rate"].double()[:, sidx] * n[:, None]).long()                     # positives per language, rounded
    mx = counts.max(0).values; n_max = (counts == mx[None, :]).sum(0); ar = torch.arange(len(sidx))
    at_l0 = counts[l0, ar]; unique_l0 = (n_max == 1) & (at_l0 == mx)
    panel_ok = sur["firing_rate"].double()[:, sidx].mean(0) >= min_panel_rate
    est = unique_l0 & (at_l0 >= min_pos) & panel_ok
    per_lang_est = {langs[i]: [int((est & (l0 == i)).sum()), int((l0 == i).sum())] for i in range(len(langs))}
    keys = ("corpus", "regime", "stream", "seed", "commit", "model", "sae_release", "sae_id", "hook", "bos_id", "pad_id", "batch_size", "max_tokens", "bits", "n_feat", "equal_budget")
    return {"schema_version": SCHEMA_VERSION, "run_dir": str(d), "files": {name: _file_record(d / name) for name in ("meta.json", "selectivity.pt", "survival.pt")},
            "provenance": {k: meta.get(k) for k in keys}, "languages": langs, "n_features": n_feat,
            "cohort": {"supported": int(supported.sum()), "selective": int(selective.sum()), "controls": int(controls.numel()), "union": int(union.sum()),
                       "selective_per_language": {langs[i]: int(per_lang[i]) for i in range(len(langs))}},
            "dev_peak_ties_among_selective": dev_ties,
            "devtest_tokens_per_language": {langs[i]: int(n[i]) for i in range(len(langs))},
            "fp_baseline_on_devtest": {"selective": int(len(sidx)), "unique_peak_at_dev_language": int(unique_l0.sum()), "tied_at_maximum": int((n_max > 1).sum()),
                                       "positives_at_dev_language_ge_min": int((at_l0 >= min_pos).sum()), "panel_rate_ge_floor": int(panel_ok.sum()), "established": int(est.sum()),
                                       "min_positives": min_pos, "min_panel_rate": min_panel_rate, "per_language": per_lang_est}}


def render_markdown(result):
    lines = ["| quantity | verdict | evidence |", "|---|---|---|"]
    for name in QUANTITIES:
        v = result["quantities"][name]
        lines.append(f"| `{name}` | **{v['verdict']}** | {v['evidence']} |")
    files = ["", "| file | kind | size (bytes) | sha256 (first 12) |", "|---|---|---|---|"]
    files += [f"| `{f['name']}` | {f['kind']} | {f['size']} | `{f['sha256'][:12]}` |" for f in result["files"]]
    return "\n".join(lines + files) + "\n"


def render_report(result, manifest, inspected, not_inspected):
    b = manifest["fp_baseline_on_devtest"]; c = manifest["cohort"]
    lines = ["# Artifact inventory and frozen cohort manifest (D-035 Task 1)", "",
             "Generated by `scripts/g0/phase1_association_inventory.py` from the schema of the saved files (keys, shapes, dtypes, hashes; no values were copied). This is a bounded inspection, **not a search of every cache**.", "",
             "## Locations inspected", ""] + [f"- {x}" for x in inspected] + ["", "## Locations not inspected", ""] + [f"- {x}" for x in not_inspected] + ["",
             f"## What the saved files support ({len(result['languages'])} languages, {result['n_features']} features)", "", render_markdown(result),
             "## Frozen cohort facts (original `mask/eq` selection, full-precision devtest evaluation)", "",
             f"- Cohort: {c['selective']} selective features, {c['controls']} matched controls, {c['supported']} supported; selective plus controls cover {c['union']} features.",
             f"- Selective features per original language: min {min(c['selective_per_language'].values())}, max {max(c['selective_per_language'].values())}; languages with none: "
             f"{[k for k, v in c['selective_per_language'].items() if v == 0] or 'none'}.",
             f"- Dev ties at the peak among selective features (the frozen label is the first maximum): {manifest['dev_peak_ties_among_selective']}.",
             f"- Baseline on devtest full precision (equal-budget stream, {min(manifest['devtest_tokens_per_language'].values())} to {max(manifest['devtest_tokens_per_language'].values())} tokens per language): "
             f"the dev peak language is the unique devtest peak for {b['unique_peak_at_dev_language']} of {b['selective']}; {b['tied_at_maximum']} are tied at the maximum; "
             f"{b['positives_at_dev_language_ge_min']} have at least {b['min_positives']} positives there; **{b['established']} meet every condition** "
             f"(panel mean rate at least {b['min_panel_rate']:g}: {b['panel_rate_ge_floor']}). Positives are rate times n, rounded.",
             f"- Settings: {json.dumps({k: v for k, v in manifest['provenance'].items() if k in ('regime', 'stream', 'seed', 'model', 'sae_id', 'bos_id', 'batch_size', 'max_tokens', 'commit')})}.",
             "", "The corpus token ids, sentence ids and selected-position masks are reconstructed from the corpus files, the tokenizer and these settings when the new run starts; their hashes are written by that run (plan Task 1).", ""]
    return "\n".join(lines)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="original run directory (mask/eq): meta.json, selectivity.pt, survival.pt"); ap.add_argument("--out", required=True)
    ap.add_argument("--expected-languages", help="comma-separated language codes in the expected order"); ap.add_argument("--inspected", action="append", default=[]); ap.add_argument("--not-inspected", action="append", default=[])
    a = ap.parse_args(argv)
    d = Path(a.run); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    expected = a.expected_languages.split(",") if a.expected_languages else None
    result = build_inventory([d / "survival.pt", d / "selectivity.pt"], expected_languages=expected); manifest = frozen_manifest(d)
    (out / "frozen_cohort_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    (out / "artifact_inventory.md").write_text(render_report(result, manifest, a.inspected, a.not_inspected))
    print("wrote", out / "artifact_inventory.md", "and", out / "frozen_cohort_manifest.json")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
