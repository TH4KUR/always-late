"""Freeze of the intervention-transfer study (D-088; ledger reports/intervention_transfer/STATUS.md section 4). Written and committed BEFORE any quantized intervention output of this study is read. It records, with hashes
that `verify` recomputes, the feature core (the 404 features of the F5 control run, their matched swapped directions and their 16 FP-active dev contexts, taken from the stage-A file of that run), the deterministic
calibration/held-out split of the features, the hashes of the inputs (F5 sample, evaluation freeze, groups manifest, plan, spec, verification), and every scientific choice of the ledger (the SPEC below: states, cells,
directions, floor and eligibility rule, stored statistics, baselines, endpoint and utility rule, bootstrap, budget, hypotheses). A deviation after the freeze is a dated amendment in the ledger; `verify` makes it visible.
The external panel (a public corpus) is recorded here as 'pending' with its rules; its passage ids and hashes are frozen by a later amendment, before any outcome on them is read.

  ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_transfer_freeze.py build  --sample ... --stage-a ... --eval-freeze ... --groups-manifest ... --plan ... --spec ... --verification ... --commit <hash> --out-dir reports/intervention_transfer
  ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_transfer_freeze.py verify --freeze reports/intervention_transfer/freeze.json (the same input options) --out-dir reports/intervention_transfer"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

SPLIT_SALT = "intervention-transfer-split-v1"

# 50 high-frequency characters that occur only in Traditional Chinese text (their Simplified forms differ): a zho_Hans passage containing any of them is dropped (ledger item X).
TRADITIONAL_ONLY = list("這個們來說為與對會時學國將後開長關點從還麼種現發裡當問應頭見無動經讓樣實體電機車東書氣門馬歲親邊運進")

SPEC = {
    "models": ["gemma-2-2b"],
    "model": {"hf": "google/gemma-2-2b", "sae_release": "gemma-scope-2b-pt-res-canonical", "sae_id": "layer_12/width_16k/canonical", "hook": "blocks.12.hook_resid_post", "weights_dtype": "bfloat16", "edit_and_readout_dtype": "float32"},
    "states": ["FP", "INT16", "INT6", "INT4"], "quantized_states": ["INT6", "INT4"], "floor_state": "INT16",
    "cells": ["FP/FP", "Q/FP", "FP/Q", "Q/Q", "Q/Q-native"],
    "cell_meaning": "prefix state / suffix state of the cached layer-12 residual and the layers above; the edit is the fixed FP decoder contribution at the FP-active positions; Q/Q-native subtracts the quantized state's own activation; every cell subtracts its own unedited baseline",
    "decomposition": {"C": "D[Q,FP]-D[FP,FP]", "W": "D[FP,Q]-D[FP,FP]", "I": "D[Q,Q]-D[Q,FP]-D[FP,Q]+D[FP,FP]", "identity": "D[Q,Q]-D[FP,FP]=C+W+I before aggregation"},
    "direction_kinds": ["real", "matched_swapped", "random1", "random2", "random3"], "random_direction_seeds": [1, 2, 3],
    "random_directions": "norm-matched to the real decoder direction and orthogonal to it; auxiliary calibration directions, not independent samples",
    "feature_subset": {"per_cell": 3, "subset_seed": 0, "M": 16, "min_ctx": 4, "dev_per_language": 997},
    "contexts": "the F5 rule: per feature the M dev sentences with the largest FP activation among those where it is positive (any language of the dev pool); a context is one dev sentence",
    "group_panel": {"languages": 27, "evaluation_sentences": 300, "group_size": "K64", "control_draws": 3},
    "stored_statistics": {"gram_cells": 5, "per": "feature, direction, context and quantized state", "contents": "5x5 Gram of the centered-logit effect vectors accumulated in float64 from float32 logits; signed loss change of each cell; edited positions; positions where the feature is active in the state; unedited hybrid loss of each cell; per feature also the Gram of the sum over contexts (averaged fingerprint)"},
    "floor": {"percentile": 95, "eligibility_multiple": 5},
    "floor_definition": "95th percentile over the calibration contexts of the norm of D[INT16] minus D[FP] at the same features, directions and contexts (same batch regime and readout); the repeated-batch pseudo-state of F5 if that is exactly zero; eligible: FP effect norm above 5 times the floor; fixed from calibration contexts only",
    "primary_endpoint": "normalized per-context transfer error ||D[Q,Q]-D[FP,FP]|| / (||D[FP,FP]|| + floor); also unnormalized error, norm ratio, cosine and signed loss change",
    "split": {"rule": "per language by SHA-256 of 'salt:feature id', the first ceil(n/2) features are calibration, the rest held out; the FLORES source sentence (dev index modulo 997) is the cluster", "salt": SPLIT_SALT, "cluster": "dev index modulo dev_per_language"},
    "baselines": {"A": "activation correlation, log FP activation, firing rate, log residual norm", "B": "A plus matched and random-direction sensitivity measured with independent directions",
                  "gradient": "first-order (forward-mode) prediction of the effect of the FP edit at the native Q context and Q suffix (arXiv 2610.15850), mandatory"},
    "ridge_strengths": [0.1, 1, 10], "ridge_selection": "feature-group folds inside the calibration half; standardization fitted on calibration only; never pooled across INT4 and INT6",
    "utility_rule": {"relative_mae_reduction": 0.10, "baseline": "the strongest prespecified baseline", "interval": "paired cluster bootstrap excluding zero improvement", "scope": "Gemma-2-2B"},
    "bootstrap": {"replicates": 2000, "seed": 20261026}, "bootstrap_unit": "source-sentence clusters resampled jointly across languages, features crossed",
    "budget": {"elapsed_hours": 48, "gpu_hours": 32, "concurrent_jobs": 2},
    "hypotheses": {
        "H1": "native effect error requires more than activation drift or a scalar change of perturbation size (tested against magnitude, residual norm and generic-direction baselines)",
        "H2": "the crossed background and suffix contrasts explain a reproducible part of the error, with the interaction measured",
        "H3": "generic sensitivity predicts some transfer error and feature- or context-specific variation may remain",
        "H4": "per-context transfer differs from similarity of averaged fingerprints (same feature-context pairs)",
        "A3": "expectation after Chang et al. (EMNLP 2025, sections 6.1 and 6.2): the median over languages of |C| > |W| at INT4 and at INT6 in the group-level effect; confirming it alone is not a contribution",
    },
    "external_panel": {"status": "pending", "languages": ["eng_Latn", "spa_Latn", "rus_Cyrl", "hin_Deva", "zho_Hans", "yor_Latn"], "calibration_passages": 100, "evaluation_passages": 300, "max_tokens": 128,
                       "corpus": {"dataset": "wikimedia/wikipedia", "config_date": "20231101", "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"},
                       "rules": "first parquet shard per language, distinct articles by SHA-256 order of the article id, passages of 64 to 128 tokens cut at sentence ends, exact and normalized duplicates of FLORES lines removed; zho_Hans passages with any filter character are dropped; a language with too few passages is recorded as missing, never replaced",
                       "zho_Hans_filter_characters": "".join(TRADITIONAL_ONLY)},
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_cluster(dev_index, per_language):
    """The FLORES source sentence of a dev-pool index: the pool is language-major with `per_language` sentences per language, so the position inside the language is the parallel sentence id."""
    return int(dev_index) % int(per_language)


def derive_core(sample, stage_a, per_cell, subset_seed, dev_per_language, M, min_ctx):
    """The feature core of the study from the F5 sample and the stage-A file of the run that used the same subset rule. Everything is checked against the sample, nothing is taken on trust."""
    import phase1_func_f5_run as fr
    ids, groups = fr.sample_arrays(sample, per_cell=per_cell, seed=subset_seed); ids = [int(i) for i in ids.tolist()]; groups = [int(g) for g in groups.tolist()]
    if [int(i) for i in stage_a["feat_ids"]] != ids:
        raise ValueError("the stage file has other features than the subset rule (per_cell, subset_seed) gives on this sample")
    if [int(g) for g in stage_a["feat_lang"]] != groups:
        raise ValueError("the stage file gives other languages for the features than the sample")
    n_lang = len(sample["languages"]); dev_lang = np.asarray(stage_a["dev_lang"])
    if len(dev_lang) != n_lang * dev_per_language or not np.array_equal(dev_lang, np.arange(len(dev_lang)) // dev_per_language):
        raise ValueError(f"the dev pool is not language-major with {dev_per_language} sentences per language: the source-sentence cluster would be wrong")
    contexts = np.asarray(stage_a["contexts"]); excluded = np.asarray(stage_a["excluded"], dtype=bool); direction = [int(d) for d in stage_a["direction_ids"]]
    if contexts.shape != (len(ids), M):
        raise ValueError(f"the stage file has contexts of shape {contexts.shape}, expected ({len(ids)}, {M})")
    feats = sample["features"]; out = {"feature_ids": [], "feature_language_index": [], "direction_ids": [], "contexts": [], "context_clusters": [], "excluded_features": []}
    for k, f in enumerate(ids):
        if bool(excluded[k]):
            out["excluded_features"].append(f); continue
        row = [int(x) for x in contexts[k]]; valid = [x for x in row if x >= 0]
        if row[:len(valid)] != valid:
            raise ValueError(f"feature {f}: a padding entry precedes a valid context")
        if len(valid) < min_ctx:
            raise ValueError(f"feature {f} has {len(valid)} contexts, fewer than min_ctx {min_ctx}, and is not marked excluded")
        if any(x >= len(dev_lang) for x in valid) or len(set(valid)) != len(valid):
            raise ValueError(f"feature {f}: contexts outside the dev pool or repeated")
        d = direction[k]
        if d == f or str(d) not in feats or feats[str(d)]["language"] != feats[str(f)]["language"]:
            raise ValueError(f"feature {f}: the matched direction {d} is not another feature of the same language")
        out["feature_ids"].append(f); out["feature_language_index"].append(groups[k]); out["direction_ids"].append(d); out["contexts"].append(valid)
        out["context_clusters"].append([source_cluster(x, dev_per_language) for x in valid])
    out.update({"schema": 1, "kind": "transfer-core-sample", "languages": list(sample["languages"]),
                "parameters": {"per_cell": per_cell, "subset_seed": subset_seed, "M": M, "min_ctx": min_ctx, "dev_per_language": dev_per_language}})
    return out


def split_features(feature_ids, languages, salt):
    """Calibration and held-out features: per language the features in the order of SHA-256('salt:id'); the first ceil(n/2) are calibration, the rest held out."""
    by_lang = {}
    for f, l in zip(feature_ids, languages):
        by_lang.setdefault(int(l), []).append(int(f))
    cal, held = [], []
    for l in sorted(by_lang):
        members = sorted(by_lang[l], key=lambda i: hashlib.sha256(f"{salt}:{i}".encode()).hexdigest())
        if len(members) < 2:
            raise ValueError(f"language index {l} has {len(members)} feature(s): it cannot have both a calibration and a held-out half")
        half = (len(members) + 1) // 2
        cal += members[:half]; held += members[half:]
    return {"calibration": sorted(cal), "heldout": sorted(held)}


def _entries(files, root):
    out = {}
    for name, p in files.items():
        p = Path(p)
        out[name] = {"path": str(p.relative_to(root)) if root and p.is_relative_to(root) else str(p), "sha256": sha256_file(p)}
    return out


def _core_block(core):
    return {"n_features": len(core["feature_ids"]), "n_feature_context_pairs": sum(len(c) for c in core["contexts"]), "feature_ids_sha256": sha256_obj(core["feature_ids"]),
            "direction_ids_sha256": sha256_obj(core["direction_ids"]), "contexts_sha256": sha256_obj(core["contexts"]), "excluded_features": list(core["excluded_features"]),
            "parameters": core["parameters"] if "parameters" in core else None, "source": core.get("source")}


def build_freeze(files, core, commit, created, root=None):
    split = split_features(core["feature_ids"], core["feature_language_index"], SPLIT_SALT)
    return {"schema": 1, "kind": "intervention-transfer-freeze", "created": created, "commit": commit, "files": _entries(files, Path(root) if root else None), "core": _core_block(core),
            "split": {"salt": SPLIT_SALT, **split}, "spec": json.loads(json.dumps(SPEC))}


def verify_freeze(freeze, files, core):
    """True, or ValueError listing every difference between the freeze and the current state of the inputs, the core and the spec."""
    problems = []
    cur, frozen = set(files), set(freeze["files"])
    for n in sorted(cur - frozen):
        problems.append(f"file {n} is not in the freeze")
    for n in sorted(frozen - cur):
        problems.append(f"file {n} of the freeze is not given")
    for n in sorted(cur & frozen):
        if sha256_file(files[n]) != freeze["files"][n]["sha256"]:
            problems.append(f"file {n} changed since the freeze")
    now = _core_block(core); was = freeze["core"]
    for key, label in (("feature_ids_sha256", "feature ids"), ("direction_ids_sha256", "matched directions (direction ids)"), ("contexts_sha256", "contexts"), ("n_features", "feature count"),
                       ("n_feature_context_pairs", "feature-context pair count"), ("excluded_features", "excluded features")):
        if now[key] != was.get(key):
            problems.append(f"the core's {label} differ from the freeze")
    expected = split_features(core["feature_ids"], core["feature_language_index"], SPLIT_SALT); sp = freeze["split"]
    if sp.get("salt") != SPLIT_SALT or sp.get("calibration") != expected["calibration"] or sp.get("heldout") != expected["heldout"] or set(sp.get("calibration", [])) & set(sp.get("heldout", [])):
        problems.append("the frozen split differs from its recomputation or its halves overlap")
    if json.loads(json.dumps(SPEC)) != freeze["spec"]:
        diff = sorted(k for k in set(SPEC) | set(freeze["spec"]) if json.loads(json.dumps(SPEC)).get(k) != freeze["spec"].get(k))
        problems.append(f"the spec differs from the freeze in {diff}")
    if problems:
        raise ValueError("; ".join(problems))
    return True


def _common_args(ap):
    ap.add_argument("--sample"); ap.add_argument("--stage-a"); ap.add_argument("--eval-freeze"); ap.add_argument("--groups-manifest"); ap.add_argument("--plan"); ap.add_argument("--spec"); ap.add_argument("--verification")
    ap.add_argument("--commit", default="unknown"); ap.add_argument("--dev-per-language", type=int, default=SPEC["feature_subset"]["dev_per_language"]); ap.add_argument("--M", type=int, default=SPEC["feature_subset"]["M"])
    ap.add_argument("--min-ctx", type=int, default=SPEC["feature_subset"]["min_ctx"]); ap.add_argument("--per-cell", type=int, default=SPEC["feature_subset"]["per_cell"], help="0: all features of the sample")
    ap.add_argument("--subset-seed", type=int, default=SPEC["feature_subset"]["subset_seed"]); ap.add_argument("--out-dir", default="reports/intervention_transfer")


def _inputs(a):
    return {"f5_sample": a.sample, "evaluation_freeze_lm": a.eval_freeze, "groups_manifest": a.groups_manifest, "plan": a.plan, "spec": a.spec, "verification": a.verification}


def _derive(a):
    import torch
    sample = json.loads(Path(a.sample).read_text()); stage = torch.load(a.stage_a, weights_only=False)
    stage = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in stage.items()}
    core = derive_core(sample, stage, a.per_cell or None, a.subset_seed, a.dev_per_language, a.M, a.min_ctx)
    core["source"] = {"stage_a_sha256": sha256_file(a.stage_a), "stage_a_origin": "Ada /home2/karma/duan-verify/out/phase1/f5-dir2/f5_stage_a.pt (the 404-feature direction-swap run of the F5 study)", "sample_sha256": sha256_file(a.sample)}
    return core


def main(argv=None):
    ap = argparse.ArgumentParser(description="Freeze of the intervention-transfer study.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _common_args(sub.add_parser("build"))
    v = sub.add_parser("verify"); v.add_argument("--freeze", required=True); _common_args(v)
    a = ap.parse_args(argv); out = Path(a.out_dir)
    if a.cmd == "build":
        core = _derive(a); out.mkdir(parents=True, exist_ok=True); core_path = out / "core_sample.json"
        core_path.write_text(json.dumps(core, indent=1, sort_keys=True) + "\n")
        files = {**_inputs(a), "core_sample": core_path}
        freeze = build_freeze(files, core, a.commit, datetime.datetime.now().strftime("%Y-%m-%d %H:%M IST"), root=Path.cwd())
        (out / "freeze.json").write_text(json.dumps(freeze, indent=1, sort_keys=True) + "\n")
        print(f"wrote {out / 'freeze.json'} and {core_path}: {freeze['core']['n_features']} features, {freeze['core']['n_feature_context_pairs']} feature-context pairs, calibration {len(freeze['split']['calibration'])}, held out {len(freeze['split']['heldout'])}")
    else:
        core_path = out / "core_sample.json"; core = json.loads(core_path.read_text())
        files = {**_inputs(a), "core_sample": core_path}
        freeze = json.loads(Path(a.freeze).read_text()); verify_freeze(freeze, files, core)
        if a.stage_a and Path(a.stage_a).exists():
            fresh = _derive(a)
            if {k: fresh[k] for k in ("feature_ids", "direction_ids", "contexts", "excluded_features")} != {k: core[k] for k in ("feature_ids", "direction_ids", "contexts", "excluded_features")}:
                raise ValueError("the committed core differs from a fresh derivation from the stage file")
        print(f"verified {a.freeze}: {len(core['feature_ids'])} features, spec and inputs unchanged")


if __name__ == "__main__":
    main()
