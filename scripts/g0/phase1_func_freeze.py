"""Freeze file of the functional-validation study (design section 7, D-042). Written and committed BEFORE any evaluation run on a quantized model (T-LM part) or any Belebele scoring (T-MC part, a second
file). It records, with hashes that `verify` recomputes, everything that defines the evaluation: the cohort and group manifests, the evaluation code, the evaluation sentence ids, the configuration presets,
the precision map and the analysis constants. A deviation after the freeze is a D-entry; `verify` makes it visible.

  python3 scripts/g0/phase1_func_freeze.py build  --out reports/functional_validation/evaluation_freeze_lm.json
  python3 scripts/g0/phase1_func_freeze.py verify --file reports/functional_validation/evaluation_freeze_lm.json"""
import hashlib
import json
import subprocess
from pathlib import Path

import phase1_func_tasks as ft


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_list(xs):
    return hashlib.sha256(json.dumps(list(xs)).encode()).hexdigest()


def ids_block(ids):
    """The evaluation sentence ids derived from (n_total, n_pilot, n_eval, seed), with hashes of the evaluation and pilot lists."""
    sets = ft.lm_ids(ids["n_total"], ids["n_pilot"], ids["n_eval"], ids["seed"])
    if set(sets["eval"]) & set(sets["pilot"]):
        raise ValueError("evaluation ids overlap the pilot ids")
    return {**{k: ids[k] for k in ("n_total", "n_pilot", "n_eval", "seed")}, "eval_ids": sets["eval"], "eval_sha256": sha256_list(sets["eval"]), "pilot_sha256": sha256_list(sets["pilot"])}


def _entries(files, root):
    out = {}
    for name, p in files.items():
        p = Path(p)
        out[name] = {"path": str(p.relative_to(root)) if root else str(p), "sha256": sha256_file(p)}
    return out


def build_freeze(files, presets, precisions, ids, analysis, commit, created, root=None, part="T-LM"):
    unknown = sorted({v for v in precisions.values() if v not in presets})
    if unknown:
        raise ValueError(f"the precision map uses unknown presets {unknown}")
    return {"schema": 1, "kind": "functional-validation-freeze", "part": part, "created": created, "commit": commit, "files": _entries(files, Path(root) if root else None),
            "presets": json.loads(json.dumps(presets)), "precisions": dict(precisions), "ids": ids_block(ids), "analysis": json.loads(json.dumps(analysis))}


def verify_freeze(freeze, files, presets, ids, analysis):
    """True, or ValueError listing every difference between the freeze and the current repository state."""
    problems = []
    cur = set(files); frozen = set(freeze["files"])
    for n in sorted(cur - frozen):
        problems.append(f"file {n} is not in the freeze")
    for n in sorted(frozen - cur):
        problems.append(f"file {n} of the freeze is not given")
    for n in sorted(cur & frozen):
        if sha256_file(files[n]) != freeze["files"][n]["sha256"]:
            problems.append(f"file {n} changed since the freeze")
    if json.loads(json.dumps(presets)) != freeze["presets"]:
        diff = sorted(k for k in set(presets) | set(freeze["presets"]) if json.loads(json.dumps(presets)).get(k) != freeze["presets"].get(k))
        problems.append(f"preset(s) {diff} differ from the freeze")
    for k in ("n_total", "n_pilot", "n_eval", "seed"):
        if ids.get(k) != freeze["ids"].get(k):
            problems.append(f"ids: {k} is {ids.get(k)} but the freeze has {freeze['ids'].get(k)}")
    try:
        fresh = ids_block({k: freeze["ids"][k] for k in ("n_total", "n_pilot", "n_eval", "seed")})
        if fresh["eval_ids"] != freeze["ids"]["eval_ids"] or fresh["eval_sha256"] != freeze["ids"]["eval_sha256"] or sha256_list(freeze["ids"]["eval_ids"]) != freeze["ids"]["eval_sha256"]:
            problems.append("ids: the frozen evaluation id list differs from its derivation or its hash")
        if fresh["pilot_sha256"] != freeze["ids"]["pilot_sha256"]:
            problems.append("ids: the pilot id hash differs")
    except ValueError as e:
        problems.append(f"ids: {e}")
    if json.loads(json.dumps(analysis)) != freeze["analysis"]:
        problems.append("analysis constants differ from the freeze")
    unknown = sorted({v for v in freeze["precisions"].values() if v not in freeze["presets"]})
    if unknown:
        problems.append(f"the frozen precision map uses unknown presets {unknown}")
    if problems:
        raise ValueError("; ".join(problems))
    return True


def repo_state(root):
    """Default inputs of the T-LM freeze from the repository at `root`."""
    import phase1_func_analysis as fa
    import phase1_func_run as fr
    root = Path(root)
    files = {"cohort_manifest": root / "reports/association_retention/frozen_cohort_manifest.json", "groups_manifest": root / "reports/functional_validation/groups_manifest.json",
             "their_functions": root / "reports/g0/their_functions.json", "code_run": root / "scripts/g0/phase1_func_run.py", "code_intervene": root / "scripts/g0/phase1_func_intervene.py",
             "code_groups": root / "scripts/g0/phase1_func_groups.py", "code_tasks": root / "scripts/g0/phase1_func_tasks.py", "code_lib": root / "scripts/g0/phase1_lib.py"}
    presets = {k: fr.PRESETS[k] for k in ("eval", "floor")}
    precisions = {"FP": "eval", "INT8": "eval", "INT7": "eval", "INT6": "eval", "INT5": "eval", "INT4": "eval", "INT16": "floor", "INT12": "floor", "INT0": "floor"}
    ids = {"n_total": 1012, "n_pilot": 150, "n_eval": 300, "seed": 0}
    analysis = {"bands": list(fa.BANDS), "macro_rule": "D-041: mean over the languages with a clearly positive full-precision E of the per-language ratios E_l(b)/E_l(FP); median ratio and pooled ratio are secondaries",
                "n_boot": 2000, "seed": 0, "primary_K": "K64", "sensitivity_K": ["K16", "all"], "covered": "all 27 languages", "panel": ["eng_Latn", "hin_Deva"], "off_target_label": "K64",
                "max_tokens": 1024, "batch_size": 32, "pool_sentences": 120, "pool_rows": 300, "analysis_module": "scripts/g0/phase1_func_analysis.py (commit recorded, not hashed)"}
    return files, presets, precisions, ids, analysis


def main(argv=None):
    import argparse
    ROOT = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Build or verify the T-LM evaluation freeze file.")
    ap.add_argument("command", choices=["build", "verify"]); ap.add_argument("--out"); ap.add_argument("--file"); ap.add_argument("--created")
    a = ap.parse_args(argv)
    files, presets, precisions, ids, analysis = repo_state(ROOT)
    if a.command == "build":
        if subprocess.run(["git", "status", "--porcelain", "--", *[str(p.relative_to(ROOT)) for p in files.values()]], cwd=ROOT, capture_output=True, text=True).stdout.strip():
            raise SystemExit("uncommitted changes in frozen files: commit first")
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        fz = build_freeze(files, presets, precisions, ids, analysis, commit, a.created or "unspecified", root=ROOT)
        Path(a.out).write_text(json.dumps(fz, indent=1) + "\n")
        print("wrote", a.out, "commit", commit, "eval ids", len(fz["ids"]["eval_ids"]))
    else:
        fz = json.loads(Path(a.file).read_text())
        verify_freeze(fz, files, presets, ids, analysis)
        print("ok: the repository state equals the freeze", a.file, "(freeze commit", fz["commit"] + ")")


if __name__ == "__main__":
    main()
