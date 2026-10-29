"""Validation of a finished survival-sidecar job (D-117 Stage 2) on Ada, in the style of gptq_ext/validate_job.py: the project's own tested checks (sacct COMPLETED 0:0, `GPU 0:` in the log head and no `CUDA initialization`
warning, every output file copied with a sha256 equal to Ada's), then the sidecar's own: meta.json (gates G1, G2(i), G3, G4, G6 present and passed, no gate failure, conditions, pre-activation source recorded, full run
over all languages), the saved sums (`survival_sidecar.validate_dir`), the weights manifest of the matched-RTN and GPTQ arms (method, artifact bit widths, every hash against the recorded one) and, for a full run,
the committed comparison G2(ii) / G4 (`check_committed`), which is REPORTED but never makes the job fail: a disagreement is investigated, not tolerated silently (design note, update of 05:30).

  ~/.venvs/anlp-torch/bin/python scripts/g0/survival_validate.py --job 43001 --jobname sc-gptq --name sc-gptq --arm gptq --bits 6 4 0 --committed-run transfer_runs/gptq-p1-mask/eq"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_transfer_validate as tv  # noqa: E402
import survival_sidecar as sc  # noqa: E402
from gptq_ext import states  # noqa: E402
from gptq_ext.validate_job import check_weights_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_GATES = ("G1", "G2_internal", "G3", "G4", "G6")
METHOD = {"rtn-matched": "rtn_matched", "gptq": "gptq"}


def merge_recorded(recorded, stage1_manifest):
    """The recorded artifact hashes (artifact_hashes.json holds INT4 only) completed with the hashes of the stage-1 job of the same arm (`weights_manifest.json` of transfer_runs/{rtnm,gptq}-p1-mask): the freeze file wins;
    widths it does not record come from the stage-1 manifest, so a job's INT6 artifact is compared with the one the committed stage-1 run used. -> (merged copy, list of conflicts)."""
    method = stage1_manifest.get("method"); merged = {k: dict(v) for k, v in recorded.items()}; conflicts = []
    for b, h in stage1_manifest.get("artifact_sha256", {}).items():
        have = merged.setdefault(method, {}).get(b)
        if have is None:
            merged[method][b] = h
        elif have != h:
            conflicts.append(f"{method} {b} bits: freeze file {have[:8]}... differs from the stage-1 manifest {h[:8]}...")
    return merged, conflicts


def check_meta(meta, bits, full, n_lang=27, n_feat=16384, arm=None):
    why = []
    if arm is not None and sc._norm(meta.get("weights")) != sc._norm({"legacy": "rtn"}.get(arm, arm)):
        why.append(f"weights {meta.get('weights')!r} recorded in the job, arm {arm!r} expected")
    if not str(meta.get("study", "")).startswith("D-117"):
        why.append(f"unexpected study {meta.get('study')!r}")
    if meta.get("gate_failures"):
        why.append(f"gate failures {meta['gate_failures']}")
    gates = meta.get("gates", {})
    for g in REQUIRED_GATES:
        if g not in gates:
            why.append(f"gate {g} missing")
        elif not gates[g].get("passed"):
            why.append(f"gate {g} did not pass: {gates[g].get('why')}")
    if "G3" in gates and gates["G3"].get("max_abs_diff_theta_vs_csv") is None:
        why.append("G3 did not compare the thresholds with the committed table")
    want = [f"INT{b}" for b in bits if b != 0]
    if meta.get("saved_conditions") != want:
        why.append(f"saved conditions {meta.get('saved_conditions')} instead of {want}")
    if 0 in bits and "G1" in gates and gates["G1"].get("note"):
        why.append("INT0 was requested but G1 did not run")
    if meta.get("regime") != "mask" or meta.get("stream") != "eq":
        why.append(f"regime/stream {meta.get('regime')}/{meta.get('stream')} instead of mask/eq")
    if meta.get("pre_activation_source") not in ("sae.hook_sae_acts_pre", "x@W_enc+b_enc") or meta.get("pre_activation_dtype") != "torch.float32":
        why.append(f"pre-activation source/dtype {meta.get('pre_activation_source')}/{meta.get('pre_activation_dtype')}")
    if full and (meta.get("subset") or len(meta.get("languages", [])) != n_lang or meta.get("n_feat") != n_feat or meta.get("max_sentences")):
        why.append("not a full run (subset, language count, feature count or sentence cap)")
    return {"ok": not why, "why": why, "pre_activation_source": meta.get("pre_activation_source"), "seconds": meta.get("seconds"), "peak_gpu_gb": meta.get("peak_gpu_gb")}


def validate(job, jobname, name, arm, bits, remote_exec, fetch, local_root=tv.LOCAL_RUNS, full=True, committed_run=None, recorded=None, n_lang=27, n_feat=16384):
    sacct = remote_exec(f"sacct -j {job} --format=JobID,JobName,State,Elapsed,NodeList,ExitCode -X -n -P").strip()
    checks = {"sacct": tv.check_sacct(sacct.splitlines()[0] if sacct else "")}
    log = f"{tv.REMOTE_LOGS}/{jobname}-{job}.out"
    checks["log"] = tv.check_log_head(remote_exec(f"head -12 {log}"), int((remote_exec(f"grep -c 'CUDA initialization' {log} || true").strip() or "0")))
    local = Path(local_root) / name; local.mkdir(parents=True, exist_ok=True); fetch(f"{tv.REMOTE_OUT}/{name}", str(local))
    local_sums = subprocess.run("sha256sum *", shell=True, cwd=local, capture_output=True, text=True).stdout
    checks["checksums"] = tv.compare_checksums(remote_exec(f"cd {tv.REMOTE_OUT}/{name} && sha256sum *"), local_sums)
    meta_path = local / "meta.json"
    if not meta_path.exists():
        checks["meta"] = {"ok": False, "why": ["no meta.json in the output"]}
        return tv.summarize(checks)
    meta = json.loads(meta_path.read_text()); checks["meta"] = check_meta(meta, bits, full, n_lang, n_feat, arm)
    try:
        v = sc.validate_dir(local, meta.get("saved_conditions", []), n_lang=len(meta.get("languages", [])) or n_lang, n_feat=meta.get("n_feat", n_feat))
        checks["sums"] = {"ok": True, "conditions": v["conditions"], "tokens_per_language_min_max": [min(v["n"]), max(v["n"])] if v["n"] else None}
    except (sc.SidecarError, OSError, KeyError) as e:
        checks["sums"] = {"ok": False, "why": [str(e)]}
    if arm != "legacy":
        mp = local / "weights_manifest.json"
        built = [b for b in bits if b in states.ARTIFACT_BITS]
        checks["weights"] = check_weights_manifest(json.loads(mp.read_text()), METHOD[arm], built, recorded or {}) if mp.exists() else {"ok": False, "why": ["no weights_manifest.json in the output"]}
    rep = tv.summarize(checks)
    if committed_run is not None and full and checks["sums"]["ok"]:
        rep["committed"] = sc.check_committed(local, committed_run)                         # reported, never a failure of the job (investigate)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--job", required=True); ap.add_argument("--jobname", required=True); ap.add_argument("--name", required=True, help="the output directory name under /home2/karma/duan-verify/out/phase1")
    ap.add_argument("--arm", choices=["legacy", "rtn-matched", "gptq"], required=True); ap.add_argument("--bits", nargs="+", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="a subset run: no full-run or committed checks"); ap.add_argument("--committed-run"); ap.add_argument("--local-root", default=str(tv.LOCAL_RUNS))
    ap.add_argument("--recorded", default=str(ROOT / "reports/gptq_extension/artifact_hashes.json")); ap.add_argument("--stage1-manifest", help="default: transfer_runs/{rtnm,gptq}-p1-mask/weights_manifest.json of the arm")
    a = ap.parse_args(argv)
    recorded = json.loads(Path(a.recorded).read_text())["artifact_sha256"]
    default = {"rtn-matched": "rtnm-p1-mask", "gptq": "gptq-p1-mask"}.get(a.arm)
    manifest = a.stage1_manifest or (str(Path(a.local_root) / default / "weights_manifest.json") if default else None)
    if manifest and Path(manifest).exists():
        recorded, conflicts = merge_recorded(recorded, json.loads(Path(manifest).read_text()))
        if conflicts:
            print("WARNING: " + "; ".join(conflicts), file=sys.stderr)
    remote = lambda cmd: subprocess.run(tv.ADA + [cmd], capture_output=True, text=True).stdout
    fetch = lambda r, l: subprocess.run(["scp", "-q", "-o", "HostKeyAlias=ada-gw1", f"karma@10.4.24.30:{r}/*", l], check=True)
    rep = validate(a.job, a.jobname, a.name, a.arm, a.bits, remote, fetch, Path(a.local_root), full=not a.smoke, committed_run=a.committed_run, recorded=recorded)
    print(json.dumps(rep, indent=1, default=str)); return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
