"""Validation of a finished functional-study job of the GPTQ extension (loss task E4, rescue E5, Belebele E7; the entry points `phase1_func_run:main` and `main_mc` write `meta_lm_eval.json` or `meta_mc_eval.json`): sacct COMPLETED
0:0; `GPU 0:` in the log head and no CUDA warning; the meta file complete, its gates true, nothing skipped for time, the planned precision(s), preset, split and sentence count, the groups manifest and the number of
configurations per state equal to the OLD job's (from its metadata); the expected output files present; every file copied from Ada with a sha256 equal to Ada's; the weights manifest against the recorded hashes.

  ~/.venvs/anlp-torch/bin/python scripts/g0/gptq_ext/validate_func_job.py --job 43862 --jobname gptq-fe4 --name gptq-func-eval-int4 --method gptq --bits 4 --precisions 4 --preset eval --configs INT4=351"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_transfer_validate as tv  # noqa: E402
from gptq_ext import validate_job as vj  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path("/home/karma/Documents/ANLP/project/transfer_runs/func_inputs/jobs")


def check_meta(meta, expected):
    why = []
    if not meta.get("complete"):
        why.append("the meta file is not complete")
    gates = meta.get("gates", {})
    if not gates or not all(v is True for v in gates.values()):
        why.append(f"gates not all true: {gates}")
    if meta.get("skipped_for_time"):
        why.append(f"skipped for time: {meta['skipped_for_time']}")
    for key, name in (("precisions", "precisions"), ("preset", "preset"), ("split", "split"), ("n_sentences", "sentences")):
        if expected.get(key) is not None and meta.get(key) != expected[key]:
            why.append(f"{name} {meta.get(key)!r} instead of {expected[key]!r}")
    if expected.get("manifest_source_sha256") is not None and meta.get("manifest_source_sha256") != expected["manifest_source_sha256"]:
        why.append("manifest_source_sha256 differs from the old jobs'")
    for state, n in (expected.get("configs") or {}).items():
        got = meta.get("summary", {}).get(state, {}).get("configs")
        if got != n:
            why.append(f"{state}: {got} configs instead of {n}")
    return {"ok": not why, "why": why, "peak_gpu_gb": meta.get("peak_gpu_gb"), "seconds": meta.get("seconds")}


F5_SAME = ("task", "model", "hook", "n_features", "per_cell", "subset_seed", "M", "min_ctx", "topk", "extra_pad", "dev_sentences_per_language", "n_dev", "n_eval_per_language", "work_dtype", "languages",
           "sample_sha256", "freeze_sha256")


def check_meta_f5(meta, expected):
    """The F5 runner's `meta_f5.json`: complete, every gate (a dict with `ok`) and the in-run sham gate true, the planned states each timed, the variant flags off (the entry point has two optional variants since
    2026-10-23, the old main jobs ran before them), and every design field equal to the OLD job's meta (`expected["reference"]`)."""
    why = []
    if not meta.get("complete"):
        why.append("the meta file is not complete")
    gates = meta.get("gates", {})
    if not gates or not all((g.get("ok") if isinstance(g, dict) else g) is True for g in gates.values()):
        why.append(f"gate not all true: {gates}")
    sham = meta.get("sham_gate")
    if not sham or not all(x.get("exact") is True for x in sham):
        why.append(f"sham gate missing or not exact: {sham}")
    if meta.get("states") != expected["states"]:
        why.append(f"states {meta.get('states')!r} instead of {expected['states']!r}")
    if sorted(meta.get("state_seconds", {})) != sorted(expected["states"]):
        why.append(f"state_seconds {sorted(meta.get('state_seconds', {}))} do not cover the planned states")
    for key, word in (("direction_control", "direction control"), ("fixed_amplitude", "fixed amplitude")):
        if meta.get(key) is not False:
            why.append(f"{word} variant flag is {meta.get(key)!r}, not False")
    for key in F5_SAME:
        if key not in meta or meta[key] != expected["reference"].get(key):
            why.append(f"{key} {meta.get(key)!r} differs from the old job's {expected['reference'].get(key)!r}")
    return {"ok": not why, "why": why, "peak_gpu_gb": meta.get("peak_gpu_gb"), "seconds": meta.get("seconds")}


def check_files(local, files, meta_name):
    present = {p.name for p in Path(local).iterdir()}
    missing = sorted((set(files) | {meta_name, "weights_manifest.json"}) - present)
    return {"ok": not missing, "missing": missing, "files": sorted(present)}


def validate(job, jobname, name, files, meta_name, expected, method, bits, recorded, remote_exec, fetch, local_root=LOCAL, task="lm"):
    new = f"{jobname}-{job}.out"
    remote = remote_exec
    checks = {}
    sacct = remote(f"sacct -j {job} --format=JobID,JobName,State,Elapsed,NodeList,ExitCode -X -n -P").strip()
    checks["sacct"] = tv.check_sacct(sacct.splitlines()[0] if sacct else "")
    log = f"{tv.REMOTE_LOGS}/{new}"
    checks["log"] = tv.check_log_head(remote(f"head -12 {log}"), int((remote(f"grep -c 'CUDA initialization' {log} || true").strip() or "0")))
    local = Path(local_root) / name; local.mkdir(parents=True, exist_ok=True); fetch(f"{tv.REMOTE_OUT}/{name}", str(local))
    local_sums = subprocess.run("sha256sum *", shell=True, cwd=local, capture_output=True, text=True).stdout
    checks["checksums"] = tv.compare_checksums(remote(f"cd {tv.REMOTE_OUT}/{name} && sha256sum *"), local_sums)
    mp = local / meta_name
    checks["files"] = check_files(local, files, meta_name)
    meta_check = check_meta_f5 if task == "f5" else check_meta
    checks["meta"] = meta_check(json.loads(mp.read_text()), expected) if mp.exists() else {"ok": False, "why": [f"no {meta_name}"]}
    wp = local / "weights_manifest.json"
    checks["weights"] = vj.check_weights_manifest(json.loads(wp.read_text()), method, bits, recorded) if wp.exists() else {"ok": False, "why": ["no weights_manifest.json"]}
    failed = [k for k, v in checks.items() if not v.get("ok")]
    return {"ok": not failed, "failed": failed, "checks": checks}


def expected_from_args(a):
    """The expectations of a job from the command line; a negative --n-sentences or an empty --manifest-sha256 means 'do not check' (the Belebele meta file has neither field)."""
    return {"precisions": a.precisions, "preset": a.preset, "split": a.split, "n_sentences": a.n_sentences if a.n_sentences >= 0 else None, "manifest_source_sha256": a.manifest_sha256 or None,
            "configs": {k: int(v) for k, v in (x.split("=") for x in a.configs)}}


def expected_f5_from_args(a, reference):
    """The expectations of an F5 job: the state list of the command line and the old job's meta as the reference of every design field."""
    return {"states": [state_name(st) for st in a.states], "reference": reference}


def state_name(st):
    """`--states 6` is the runner's `INT6`; `FP` stays."""
    return f"INT{st}" if st.isdigit() else st


def default_files(task, bits, states):
    """The files a job must have written: one `func_lm_eval_INT<b>.pt` per bit width, or for F5 the stage-A file and one `f5_state_<S>.pt` per state."""
    if task == "f5":
        return ["f5_stage_a.pt"] + [f"f5_state_{state_name(st)}.pt" for st in states]
    return [f"func_lm_eval_INT{b}.pt" for b in bits]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--job", required=True); ap.add_argument("--jobname", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--method", choices=["gptq", "rtn-matched"], required=True); ap.add_argument("--bits", nargs="+", type=int, required=True)
    ap.add_argument("--task", choices=["lm", "f5"], default="lm", help="f5: the F5 runner's meta_f5.json and files; needs --states and --reference-meta")
    ap.add_argument("--meta", help="default: meta_lm_eval.json, or meta_f5.json for --task f5"); ap.add_argument("--files", nargs="+", help="default: func_lm_eval_INT<bits>.pt, or the F5 files")
    ap.add_argument("--precisions", nargs="+"); ap.add_argument("--preset"); ap.add_argument("--split", default="eval"); ap.add_argument("--n-sentences", type=int, default=300)
    ap.add_argument("--states", nargs="+", help="F5: the planned states, e.g. FP 6 4"); ap.add_argument("--reference-meta", help="F5: the old job's meta_f5.json (every design field must equal it)")
    ap.add_argument("--manifest-sha256", default="2f0422c2c4c4fdeae9024b8afa9c0b68a2baca926f0df5a16aac838e5a0db3f6", help="sha256 of groups_manifest.json as recorded by the old jobs")
    ap.add_argument("--configs", nargs="*", default=[], metavar="STATE=N"); ap.add_argument("--recorded", default=str(ROOT / "reports/gptq_extension/artifact_hashes.json"))
    a = ap.parse_args(argv)
    if a.task == "f5" and not (a.states and a.reference_meta):
        ap.error("--task f5 needs --states and --reference-meta")
    if a.task == "lm" and not a.precisions:
        ap.error("--task lm needs --precisions")
    files = a.files or default_files(a.task, a.bits, a.states)
    meta_name = a.meta or ("meta_f5.json" if a.task == "f5" else "meta_lm_eval.json")
    expected = expected_f5_from_args(a, json.loads(Path(a.reference_meta).read_text())) if a.task == "f5" else expected_from_args(a)
    recorded = json.loads(Path(a.recorded).read_text())["artifact_sha256"]
    remote = lambda cmd: subprocess.run(tv.ADA + [cmd], capture_output=True, text=True).stdout
    fetch = lambda r, l: subprocess.run(["scp", "-q", "-o", "HostKeyAlias=ada-gw1", f"karma@10.4.24.30:{r}/*", l], check=True)
    rep = validate(a.job, a.jobname, a.name, files, meta_name, expected, a.method.replace("-", "_"), a.bits, recorded, remote, fetch, task=a.task)
    print(json.dumps(rep, indent=1, default=str)); return 0 if rep["ok"] else 1

if __name__ == "__main__":
    sys.exit(main())
