"""Freeze file of the functional-validation study (design section 7, D-042). Written and committed BEFORE any evaluation run on a quantized model (T-LM part) or any Belebele scoring (T-MC part, a second
file). It records, with hashes that `verify` recomputes, everything that defines the evaluation: the cohort and group manifests, the evaluation code, the evaluation sentence ids, the configuration presets,
the precision map and the analysis constants. A deviation after the freeze is a D-entry; `verify` makes it visible.

  python3 scripts/g0/phase1_func_freeze.py build  --out reports/functional_validation/evaluation_freeze_lm.json
  python3 scripts/g0/phase1_func_freeze.py verify --file reports/functional_validation/evaluation_freeze_lm.json"""
import hashlib
import json
import re
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
        out[name] = {"path": str(p.relative_to(root)) if root and p.is_relative_to(root) else str(p), "sha256": sha256_file(p)}      # a file outside the repository (git-ignored data) is recorded by its absolute path
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


# ---- the T-MC part (plan Task 7b): a second file, written after the Belebele calibration and before any Belebele evaluation item is scored -------------------------------------------------------

MC_PRECISIONS = ["FP", "INT8", "INT7", "INT6", "INT5", "INT4"]


def mc_analysis_constants(covered, fmt, bands):
    """The analysis constants of the confirmatory family: the statements of the Q19 default (a) and the settings of phase1_func_mc_analysis.py."""
    statements = [{"id": "S0", "precision": "FP", "claim": "the macro effect E over the covered languages is greater than 0", "sidedness": "one-sided", "interval_level": 0.95,
                   "decision": "the lower bound of the percentile interval of the paired passage bootstrap is above 0"}]
    for i, prec in enumerate(["INT8", "INT7", "INT6", "INT5", "INT4"], start=1):
        statements.append({"id": f"S{i}", "precision": prec, "claim": "the macro retention of E (mean of per-language ratios E_l(b) / E_l(FP), D-041) is classified preserved, attenuated, amplified or inconclusive against the bands",
                           "sidedness": "two-sided", "interval_level": 0.99, "correction": "Bonferroni over the five precision statements (Q19 default a)",
                           "decision": "preserved if the interval lies inside the bands, attenuated if entirely below, amplified if entirely above, inconclusive otherwise"})
    return {"bands": list(bands), "macro_rule": "D-041: mean over the covered languages with a clearly positive full-precision E of the per-language ratios E_l(b)/E_l(FP); median ratio and pooled ratio are secondaries",
            "n_boot": 2000, "seed": 0, "primary_K": "K64", "cluster": "passage", "cluster_note": "the passage is the link of the item key; one multinomial draw matrix serves every language and precision",
            "harm": "-(log p_correct(edit) - log p_correct(none)), p_correct = softmax over the four option scores; E = zero-ablation harm beyond the mean of the control draws",
            "covered": list(covered), "format": fmt, "statements": statements, "everything_else": "T-LM, per-language effects, rescue, F4, subgroups and doses are exploratory and labelled",
            "q19": "default (a) applied; Q19 stays open in DECISIONS_FOR_REVIEW.md until the user answers (a change before the build, never after)"}


def items_block(eval_keys, cal_keys=None):
    """Per language the number and the (order-sensitive) hash of the evaluation item keys [(passage_key, question_number)]; duplicates and an overlap with the calibration items are errors."""
    out = {}
    for code, keys in sorted(eval_keys.items()):
        ks = [list(k) for k in keys]
        tup = {tuple(k) for k in ks}
        if len(tup) != len(ks):
            raise ValueError(f"{code}: duplicate evaluation item keys")
        if cal_keys and tup & {tuple(k) for k in cal_keys.get(code, [])}:
            raise ValueError(f"{code}: the evaluation items overlap the calibration items")
        out[code] = {"n_items": len(ks), "keys_sha256": sha256_list(ks)}
    return out


def _shared_code_problems(files, lm_freeze, deviations):
    """Files that the T-MC and the T-LM freeze both hash must be equal, unless a decision-register entry is recorded for the file."""
    out = []
    for name in sorted(set(files) & set(lm_freeze["files"])):
        if sha256_file(files[name]) != lm_freeze["files"][name]["sha256"] and name not in deviations:
            out.append(f"file {name} differs from the T-LM freeze and no deviation (D-entry) is recorded for it")
    return out


def build_mc_freeze(files, lm_freeze, lm_freeze_sha256, calibration, mc, eval_keys, cal_keys, precisions, analysis, commit, created, root=None, deviations=None):
    deviations = dict(deviations or {})
    for name, did in deviations.items():
        if not re.fullmatch(r"D-\d{3}", str(did)):
            raise ValueError(f"deviation for {name}: {did!r} is not a decision-register id (D-nnn)")
        if name not in files:
            raise ValueError(f"deviation for {name}: not a frozen file")
    if not calibration.get("covered"):
        raise ValueError("no covered language in the calibration record: there is nothing to freeze for T-MC")
    if calibration.get("format") not in ("letter", "cloze"):
        raise ValueError(f"the calibration record has no chosen format ({calibration.get('format')!r})")
    if mc.get("format") != calibration["format"]:
        raise ValueError(f"the run format {mc.get('format')!r} is not the calibrated format {calibration['format']!r}")
    if set(eval_keys) != set(calibration["covered"]):
        raise ValueError(f"the evaluation items must be those of exactly the covered languages: {sorted(set(eval_keys) ^ set(calibration['covered']))} differ")
    bad = sorted(k for k, v in precisions.items() if v != mc.get("preset"))
    if bad:
        raise ValueError(f"the precision map must use the frozen preset {mc.get('preset')!r}: {bad} do not")
    problems = _shared_code_problems(files, lm_freeze, deviations)
    if problems:
        raise ValueError("; ".join(problems))
    return {"schema": 1, "kind": "functional-validation-freeze", "part": "T-MC", "created": created, "commit": commit, "lm_freeze_sha256": lm_freeze_sha256,
            "files": _entries(files, Path(root) if root else None), "calibration": {"format": calibration["format"], "covered": list(calibration["covered"])},
            "mc": json.loads(json.dumps(mc)), "items": items_block(eval_keys, cal_keys), "precisions": dict(precisions), "analysis": json.loads(json.dumps(analysis)), "deviations": deviations}


def verify_mc_freeze(freeze, files, lm_freeze, lm_freeze_sha256, calibration, mc, eval_keys, analysis):
    """True, or ValueError listing every difference between the T-MC freeze and the current state."""
    problems = []
    cur, frozen = set(files), set(freeze["files"])
    for n in sorted(cur - frozen):
        problems.append(f"file {n} is not in the freeze")
    for n in sorted(frozen - cur):
        problems.append(f"file {n} of the freeze is not given")
    for n in sorted(cur & frozen):
        if sha256_file(files[n]) != freeze["files"][n]["sha256"]:
            problems.append(f"file {n} changed since the freeze")
    if lm_freeze_sha256 != freeze["lm_freeze_sha256"]:
        problems.append("the T-LM freeze file changed (its sha256 differs from the recorded one)")
    problems += _shared_code_problems(files, lm_freeze, freeze.get("deviations", {}))
    if {"format": calibration.get("format"), "covered": list(calibration.get("covered", []))} != freeze["calibration"]:
        problems.append("calibration record: the chosen format or the covered languages differ from the freeze")
    if json.loads(json.dumps(mc)) != freeze["mc"]:
        problems.append("mc settings differ from the freeze (preset, definition, caps, seed or limits)")
    try:
        fresh = items_block(eval_keys)
        diff = sorted(c for c in set(fresh) | set(freeze["items"]) if fresh.get(c) != freeze["items"].get(c))
        if diff:
            problems.append(f"items: the evaluation item selection differs for {diff}")
    except ValueError as e:
        problems.append(f"items: {e}")
    if json.loads(json.dumps(analysis)) != freeze["analysis"]:
        problems.append("analysis constants (statements, bands, bootstrap settings) differ from the freeze")
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


def repo_state_mc(root, calibration_path, belebele_root, origin_file, n_items=200, item_seed=0):
    """Inputs of the T-MC freeze from the repository, the calibration record and the Belebele files: the evaluation items are chosen by the runner's own `mc_items_for` (devtest-origin passages), the
    calibration items (dev-origin, the whole split) are listed to prove that the two sets are disjoint."""
    import phase1_func_analysis as fa
    import phase1_func_run as fr
    root = Path(root); belebele_root = Path(belebele_root)
    calibration = json.loads(Path(calibration_path).read_text())
    origin = json.loads(Path(origin_file).read_text())["origin"]
    files = {"cohort_manifest": root / "reports/association_retention/frozen_cohort_manifest.json", "groups_manifest": root / "reports/functional_validation/groups_manifest.json",
             "their_functions": root / "reports/g0/their_functions.json", "code_run": root / "scripts/g0/phase1_func_run.py", "code_intervene": root / "scripts/g0/phase1_func_intervene.py",
             "code_groups": root / "scripts/g0/phase1_func_groups.py", "code_tasks": root / "scripts/g0/phase1_func_tasks.py", "code_lib": root / "scripts/g0/phase1_lib.py",
             "code_mc_analysis": root / "scripts/g0/phase1_func_mc_analysis.py", "code_analysis": root / "scripts/g0/phase1_func_analysis.py", "code_calibration": root / "scripts/g0/phase1_func_mc_calibration.py",
             "belebele_manifest": belebele_root / "MANIFEST.json", "passage_origin": Path(origin_file), "calibration_record": Path(calibration_path)}
    mc = {"preset": "mc_eval", "preset_definition": fr.MC_PRESETS["mc_eval"], "split": "eval", "format": calibration.get("format"), "n_items": n_items, "n_items_cloze": n_items, "item_seed": item_seed,
          "max_len": 1024, "max_tokens": 1024, "batch_size": 16}
    key = lambda it: (it["passage_key"], it["question_number"])
    eval_keys, cal_keys = {}, {}
    for code in calibration.get("covered", []):
        items = ft.load_belebele(belebele_root, code, expect=900)
        eval_keys[code] = [key(it) for it in fr.mc_items_for(items, origin, "eval", n_items, item_seed)]
        cal_keys[code] = [key(it) for it in fr.mc_items_for(items, origin, "cal", 0, item_seed)]
    precisions = {p: "mc_eval" for p in MC_PRECISIONS}
    analysis = mc_analysis_constants(calibration.get("covered", []), calibration.get("format"), fa.BANDS)
    return {"files": files, "calibration": calibration, "mc": mc, "eval_keys": eval_keys, "cal_keys": cal_keys, "precisions": precisions, "analysis": analysis}


def main(argv=None):
    import argparse
    ROOT = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Build or verify the T-LM (build, verify) or T-MC (build-mc, verify-mc) evaluation freeze file.")
    ap.add_argument("command", choices=["build", "verify", "build-mc", "verify-mc"]); ap.add_argument("--out"); ap.add_argument("--file"); ap.add_argument("--created")
    ap.add_argument("--lm-freeze", default=str(ROOT / "reports/functional_validation/evaluation_freeze_lm.json")); ap.add_argument("--calibration", default=str(ROOT / "reports/functional_validation/belebele_calibration.json"))
    ap.add_argument("--belebele-root", default=str(ROOT / "data/belebele")); ap.add_argument("--origin-file", default=str(ROOT / "data/belebele/passage_origin.json"))
    ap.add_argument("--n-items", type=int, default=200); ap.add_argument("--deviation", nargs="*", default=[], help="file=D-nnn for a shared code file changed after the T-LM freeze")
    a = ap.parse_args(argv)
    if a.command in ("build-mc", "verify-mc"):
        lm_path = Path(a.lm_freeze); lm = json.loads(lm_path.read_text()); lm_sha = sha256_file(lm_path)
        if a.command == "build-mc":
            st = repo_state_mc(ROOT, a.calibration, a.belebele_root, a.origin_file, a.n_items)
            dirty = subprocess.run(["git", "status", "--porcelain", "--", *[str(Path(p).relative_to(ROOT)) for p in st["files"].values() if Path(p).is_relative_to(ROOT)]], cwd=ROOT, capture_output=True, text=True).stdout.strip()
            if dirty:
                raise SystemExit("uncommitted changes in frozen files: commit first")
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
            dev = dict(x.split("=", 1) for x in a.deviation)
            fz = build_mc_freeze(st["files"], lm, lm_sha, st["calibration"], st["mc"], st["eval_keys"], st["cal_keys"], st["precisions"], st["analysis"], commit, a.created or "unspecified", root=ROOT, deviations=dev)
            Path(a.out).write_text(json.dumps(fz, indent=1) + "\n")
            print("wrote", a.out, "commit", commit, "covered", fz["calibration"]["covered"], "items", {c: v["n_items"] for c, v in fz["items"].items()})
        else:
            fz = json.loads(Path(a.file).read_text())
            st = repo_state_mc(ROOT, a.calibration, a.belebele_root, a.origin_file, fz["mc"]["n_items"])
            verify_mc_freeze(fz, st["files"], lm, lm_sha, st["calibration"], st["mc"], st["eval_keys"], st["analysis"])
            print("ok: the repository state equals the T-MC freeze", a.file, "(freeze commit", fz["commit"] + ")")
        return
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
