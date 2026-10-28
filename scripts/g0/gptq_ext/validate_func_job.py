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


def check_files(local, files, meta_name):
    present = {p.name for p in Path(local).iterdir()}
    missing = sorted((set(files) | {meta_name, "weights_manifest.json"}) - present)
    return {"ok": not missing, "missing": missing, "files": sorted(present)}


def validate(job, jobname, name, files, meta_name, expected, method, bits, recorded, remote_exec, fetch, local_root=LOCAL):
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
    checks["meta"] = check_meta(json.loads(mp.read_text()), expected) if mp.exists() else {"ok": False, "why": [f"no {meta_name}"]}
    wp = local / "weights_manifest.json"
    checks["weights"] = vj.check_weights_manifest(json.loads(wp.read_text()), method, bits, recorded) if wp.exists() else {"ok": False, "why": ["no weights_manifest.json"]}
    failed = [k for k, v in checks.items() if not v.get("ok")]
    return {"ok": not failed, "failed": failed, "checks": checks}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--job", required=True); ap.add_argument("--jobname", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--method", choices=["gptq", "rtn-matched"], required=True); ap.add_argument("--bits", nargs="+", type=int, required=True)
    ap.add_argument("--meta", default="meta_lm_eval.json"); ap.add_argument("--files", nargs="+", help="default: func_lm_eval_INT<bits>.pt")
    ap.add_argument("--precisions", nargs="+", required=True); ap.add_argument("--preset"); ap.add_argument("--split", default="eval"); ap.add_argument("--n-sentences", type=int, default=300)
    ap.add_argument("--manifest-sha256", default="2f0422c2c4c4fdeae9024b8afa9c0b68a2baca926f0df5a16aac838e5a0db3f6", help="sha256 of groups_manifest.json as recorded by the old jobs")
    ap.add_argument("--configs", nargs="*", default=[], metavar="STATE=N"); ap.add_argument("--recorded", default=str(ROOT / "reports/gptq_extension/artifact_hashes.json"))
    a = ap.parse_args(argv)
    files = a.files or [f"func_lm_eval_INT{b}.pt" for b in a.bits]
    expected = {"precisions": a.precisions, "preset": a.preset, "split": a.split, "n_sentences": a.n_sentences, "manifest_source_sha256": a.manifest_sha256, "configs": {k: int(v) for k, v in (x.split("=") for x in a.configs)}}
    recorded = json.loads(Path(a.recorded).read_text())["artifact_sha256"]
    remote = lambda cmd: subprocess.run(tv.ADA + [cmd], capture_output=True, text=True).stdout
    fetch = lambda r, l: subprocess.run(["scp", "-q", "-o", "HostKeyAlias=ada-gw1", f"karma@10.4.24.30:{r}/*", l], check=True)
    rep = validate(a.job, a.jobname, a.name, files, a.meta, expected, a.method.replace("-", "_"), a.bits, recorded, remote, fetch)
    print(json.dumps(rep, indent=1, default=str)); return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
