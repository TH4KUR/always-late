"""Validation of a finished intervention-transfer job on Ada (the checklist of NEXT STEP item 1 in reports/functional_validation/STATUS.md), the same for every job: sacct COMPLETED with exit 0:0; the log head lists
`GPU 0:` and holds no `CUDA initialization` warning (a node without a GPU runs the model on the CPU and passes every gate: job 40402); the meta file is complete, the gates passed, the sham exact, the gate limit the
expected one, and items + skipped equal the planned count of every state; every file of the output directory copied to the local run folder with a sha256 equal to Ada's. Prints a JSON report and exits 1 if any check fails.

    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_transfer_validate.py --job 40399 --name transfer-feature-int4 --task feature --states INT4 --core reports/intervention_transfer/core_sample.json
    ~/.venvs/anlp-torch/bin/python scripts/g0/phase1_transfer_validate.py --job 40403 --name transfer-group-int6 --task group --states INT6 --languages 27 --sentences 300"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ADA = ["ssh", "-o", "HostKeyAlias=ada-gw1", "karma@10.4.24.30"]
REMOTE_OUT = "/home2/karma/duan-verify/out/phase1"
REMOTE_LOGS = "/home2/karma/duan-verify/logs"
LOCAL_RUNS = Path("/home/karma/Documents/ANLP/project/transfer_runs")


def check_sacct(line):
    """One line of `sacct -X -n -P` (JobID|JobName|State|Elapsed|NodeList|ExitCode): COMPLETED with exit 0:0."""
    f = line.strip().split("|")
    if len(f) < 6:
        return {"ok": False, "why": f"no sacct line: {line!r}"}
    ok = f[2] == "COMPLETED" and f[5] == "0:0"
    return {"ok": ok, "state": f[2], "elapsed": f[3], "node": f[4], "exit": f[5], **({} if ok else {"why": f"state {f[2]}, exit {f[5]}"})}


def check_log_head(text, n_cuda_warnings):
    """The log head must list a GPU (`GPU 0:`), no `No devices found`, and the whole log no `CUDA initialization` warning."""
    lines = text.splitlines(); gpu = next((l for l in lines if l.startswith("GPU 0:")), None)
    if gpu is None or "No devices found" in text:
        return {"ok": False, "why": "no `GPU 0:` line or `No devices found`: the job may have run on the CPU"}
    if n_cuda_warnings:
        return {"ok": False, "why": f"{n_cuda_warnings} `CUDA initialization` warning(s)"}
    return {"ok": True, "gpu": gpu.split("(")[0].strip()}


def planned_feature_items(core):
    """Items of a feature-level run: every context of every feature in five directions."""
    return sum(len(c) for c in core["contexts"]) * 5


def planned_group_items(languages, sentences, controls):
    """Items of a group-level run: every sentence of every language for the group and for each control draw."""
    return languages * sentences * (1 + controls)


def check_meta(meta, planned, gate_max_abs):
    """The meta file of a run: complete; gates of the instrument passed (split equals full, both logit checks, an exact sham); the gate limit the expected one (a meta file from before amendment A12 has none: 2.0);
    and for each state items + skipped equal the planned count (states in the file and in `planned` must be the same). Resumed chunks and skipped items are reported in `notes`."""
    why, notes = [], []
    if not meta.get("complete"):
        why.append("the meta file is not complete")
    if meta.get("gate_max_abs", 2.0) != gate_max_abs:
        why.append(f"gate limit {meta.get('gate_max_abs', 2.0)} instead of {gate_max_abs}")
    g = meta.get("gates", {})
    if not g.get("split_equals_full"):
        why.append("split_equals_full is false")
    for k in ("logits_native_residual", "logits_float32_residual"):
        if not g.get(k, {}).get("ok"):
            why.append(f"gate {k} not ok")
    if not meta.get("run", {}).get("gates", {}).get("sham_exact"):
        why.append("the sham edit is not exact")
    st = meta.get("run", {}).get("states", {})
    if set(st) != set(planned):
        why.append(f"states {sorted(st)} instead of {sorted(planned)}")
    for s, n in planned.items():
        if s in st:
            got = st[s]["items"] + st[s]["skipped"]
            if got != n:
                why.append(f"state {s}: {st[s]['items']} items + {st[s]['skipped']} skipped = {got}, planned {n}")
            if st[s].get("resumed"):
                notes.append(f"state {s}: {st[s]['resumed']} chunk(s) resumed from an earlier run")
            if st[s]["skipped"]:
                notes.append(f"state {s}: {st[s]['skipped']} item(s) skipped by the runner (contexts or sets without an active feature)")
    return {"ok": not why, "why": why, "notes": notes, "peak_gpu_gb": meta.get("peak_gpu_gb"), "seconds": meta.get("seconds")}


def compare_checksums(a, b):
    """Two `sha256sum` outputs (hash, two spaces, file): every file must be in both with the same hash."""
    parse = lambda t: {l.split("  ", 1)[1]: l.split("  ", 1)[0] for l in t.strip().splitlines() if "  " in l}
    x, y = parse(a), parse(b); diff = sorted(f for f in set(x) | set(y) if x.get(f) != y.get(f))
    return {"ok": not diff and bool(x), "differences": diff, "files": len(x)}


def summarize(checks):
    failed = [k for k, v in checks.items() if not v.get("ok")]
    return {"ok": not failed, "failed": failed, "checks": checks}


def validate_job(job, name, planned, gate_max_abs, remote_exec, fetch, local_root=LOCAL_RUNS):
    """Runs the checklist for one job. remote_exec(cmd) -> output text of a command on Ada; fetch(remote_dir, local_dir) copies the directory; both are injected (the command line uses ssh and scp)."""
    sacct = remote_exec(f"sacct -j {job} --format=JobID,JobName,State,Elapsed,NodeList,ExitCode -X -n -P").strip(); checks = {"sacct": check_sacct(sacct.splitlines()[0] if sacct else "")}
    log = f"{REMOTE_LOGS}/transfer-{job}.out"
    checks["log"] = check_log_head(remote_exec(f"head -12 {log}"), int((remote_exec(f"grep -c 'CUDA initialization' {log} || true").strip() or "0")))
    local = Path(local_root) / name; local.mkdir(parents=True, exist_ok=True); fetch(f"{REMOTE_OUT}/{name}", str(local))
    local_sums = subprocess.run("sha256sum *", shell=True, cwd=local, capture_output=True, text=True).stdout
    checks["checksums"] = compare_checksums(remote_exec(f"cd {REMOTE_OUT}/{name} && sha256sum *"), local_sums)
    meta_path = local / "meta_transfer.json"
    checks["meta"] = check_meta(json.loads(meta_path.read_text()), planned, gate_max_abs) if meta_path.exists() else {"ok": False, "why": ["no meta_transfer.json in the output"]}
    return summarize(checks)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--job", required=True); ap.add_argument("--name", required=True); ap.add_argument("--task", choices=["feature", "group"], required=True); ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--core", help="core file of a feature-level run (planned items = contexts x 5)"); ap.add_argument("--languages", type=int); ap.add_argument("--sentences", type=int); ap.add_argument("--controls", type=int, default=3)
    ap.add_argument("--gate-max-abs", type=float, default=2.0)
    a = ap.parse_args(argv)
    if a.task == "feature":
        n = planned_feature_items(json.loads(Path(a.core).read_text()))
    else:
        n = planned_group_items(a.languages, a.sentences, a.controls)
    remote = lambda cmd: subprocess.run(ADA + [cmd], capture_output=True, text=True).stdout
    fetch = lambda r, l: subprocess.run(["scp", "-q", "-o", "HostKeyAlias=ada-gw1", f"karma@10.4.24.30:{r}/*", l], check=True)
    rep = validate_job(a.job, a.name, {s: n for s in a.states}, a.gate_max_abs, remote, fetch)
    print(json.dumps(rep, indent=1, default=str)); return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
