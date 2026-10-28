"""Turn the weights manifests of build-and-gate jobs into the frozen hash file of the Gemma-2-9B jobs (plan docs/superpowers/plans/2026-10-28-gemma9b-three-arm-confirmation.md, Task 3). The schema is that of
`reports/gptq_extension/artifact_hashes.json`, which `gptq_ext.states.build_artifacts` reads through `--gptq-freeze`: `artifact_sha256: {method: {bits: sha}}` and `calibration: {freeze_file_sha256, pool_sha256}`.
A gate directory holds `<method>/weights_manifest.json` (what `gate_run` writes) or `weights_manifest_<method>.json` (the committed copies); methods `gptq` and `rtn_matched`. Several directories are merged;
two that disagree about a hash or about the pool are an error. Never overwrites an existing file. Stdlib only.

  python -m gptq_ext.record_hashes --gate-dir <dir> [--gate-dir <dir2>] --out reports/functional_validation/gemma9b/artifact_hashes.json"""
import argparse
import json
import sys
from pathlib import Path

METHODS = ("gptq", "rtn_matched")


def _manifests(directory):
    d = Path(directory)
    found = {}
    for method in METHODS:
        for p in (d / method / "weights_manifest.json", d / f"weights_manifest_{method}.json"):
            if p.exists():
                found[method] = json.loads(p.read_text())
                break
    if not found:
        raise FileNotFoundError(f"no weights manifest of {METHODS} in {d} (looked for <method>/weights_manifest.json and weights_manifest_<method>.json)")
    return found


def collect(gate_dirs):
    """-> {"artifact_sha256": {method: {bits: sha}}, "calibration": {"freeze_file_sha256", "pool_sha256"}} merged over the gate directories."""
    sha, pool, freeze = {}, None, None
    for d in gate_dirs:
        for method, m in _manifests(d).items():
            if pool is not None and m["pool_sha256"] != pool:
                raise ValueError(f"{d}: the calibration pool hash {m['pool_sha256']} differs from {pool} of an earlier directory")
            if freeze is not None and m["calibration_freeze_sha256"] != freeze:
                raise ValueError(f"{d}: the calibration freeze hash differs from that of an earlier directory")
            pool, freeze = m["pool_sha256"], m["calibration_freeze_sha256"]
            for bits, h in m["artifact_sha256"].items():
                old = sha.setdefault(method, {}).setdefault(str(bits), h)
                if old != h:
                    raise ValueError(f"{d}: {method} at {bits} bits has sha256 {h} but an earlier directory has {old}")
    return {"artifact_sha256": {m: dict(sorted(v.items())) for m, v in sorted(sha.items())}, "calibration": {"freeze_file_sha256": freeze, "pool_sha256": pool}}


def write(gate_dirs, out_path):
    """Write the collected hashes to `out_path` (sorted keys); FileExistsError when it exists. -> the dict."""
    out = Path(out_path)
    if out.exists():
        raise FileExistsError(f"{out} exists: a frozen hash file is never overwritten")
    data = collect(gate_dirs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description="Record the artifact hashes of gate jobs as the frozen hash file.")
    ap.add_argument("--gate-dir", action="append", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    data = write(a.gate_dir, a.out)
    print("wrote", a.out, {m: sorted(v) for m, v in data["artifact_sha256"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
