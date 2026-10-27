"""Validation of a finished transfer job of the GPTQ extension: the project's own checklist (`phase1_transfer_validate.validate_job`, unchanged: sacct COMPLETED 0:0, `GPU 0:` in the log head and no CUDA warning, complete meta
with gates passed and exact sham, items + skipped = planned, every output file copied with a sha256 equal to Ada's) with the extension's log name (`<job name>-<id>.out` instead of `transfer-<id>.out`), plus a check of the
job's `weights_manifest.json`: the method, the artifact bit widths, and every artifact hash against the recorded one (`reports/gptq_extension/artifact_hashes.json`; bit widths without a recorded hash are listed).

  ~/.venvs/anlp-torch/bin/python scripts/g0/gptq_ext/validate_job.py --job 42954 --jobname rtnm-tf4 --name rtnm-transfer-feature-int4 --task feature --states INT4 --method rtn-matched --bits 4 --core reports/intervention_transfer/core_sample.json"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_transfer_validate as tv  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def check_weights_manifest(man, method, bits, recorded):
    why, unrecorded = [], []
    if man.get("method") != method:
        why.append(f"method {man.get('method')!r} instead of {method!r}")
    got = sorted(man.get("artifact_sha256", {}), key=int); want = sorted((str(b) for b in bits), key=int)
    if got != want:
        why.append(f"artifact bit widths {got} instead of {want}")
    for b, h in man.get("artifact_sha256", {}).items():
        rec = recorded.get(method, {}).get(b)
        if rec is None:
            unrecorded.append(b)
        elif rec != h:
            why.append(f"{b} bits: sha256 {h[:12]}... differs from the recorded {rec[:12]}...")
    return {"ok": not why, "why": why, "unrecorded": unrecorded, "hashes": {b: h[:16] for b, h in man.get("artifact_sha256", {}).items()}}


def validate(job, jobname, name, planned, gate_max_abs, method, bits, recorded, remote_exec, fetch, local_root=tv.LOCAL_RUNS):
    old, new = f"transfer-{job}.out", f"{jobname}-{job}.out"
    rep = tv.validate_job(job, name, planned, gate_max_abs, lambda cmd: remote_exec(cmd.replace(old, new)), fetch, local_root)
    mp = Path(local_root) / name / "weights_manifest.json"
    rep["checks"]["weights"] = check_weights_manifest(json.loads(mp.read_text()), method, bits, recorded) if mp.exists() else {"ok": False, "why": ["no weights_manifest.json in the output"]}
    failed = [k for k, v in rep["checks"].items() if not v.get("ok")]
    return {"ok": not failed, "failed": failed, "checks": rep["checks"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--job", required=True); ap.add_argument("--jobname", required=True, help="the Slurm job name (-J), part of the log file name"); ap.add_argument("--name", required=True, help="the output directory name")
    ap.add_argument("--task", choices=["feature", "group"], required=True); ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--method", choices=["gptq", "rtn-matched"], required=True); ap.add_argument("--bits", nargs="+", type=int, required=True)
    ap.add_argument("--core", help="core file of a feature-level run"); ap.add_argument("--languages", type=int, default=27); ap.add_argument("--sentences", type=int, default=300); ap.add_argument("--controls", type=int, default=3)
    ap.add_argument("--gate-max-abs", type=float, default=2.0); ap.add_argument("--recorded", default=str(ROOT / "reports/gptq_extension/artifact_hashes.json"))
    a = ap.parse_args(argv)
    n = tv.planned_feature_items(json.loads(Path(a.core).read_text())) if a.task == "feature" else tv.planned_group_items(a.languages, a.sentences, a.controls)
    recorded = json.loads(Path(a.recorded).read_text())["artifact_sha256"]
    remote = lambda cmd: subprocess.run(tv.ADA + [cmd], capture_output=True, text=True).stdout
    fetch = lambda r, l: subprocess.run(["scp", "-q", "-o", "HostKeyAlias=ada-gw1", f"karma@10.4.24.30:{r}/*", l], check=True)
    rep = validate(a.job, a.jobname, a.name, {s: n for s in a.states}, a.gate_max_abs, a.method.replace("-", "_"), a.bits, recorded, remote, fetch)
    print(json.dumps(rep, indent=1, default=str)); return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
