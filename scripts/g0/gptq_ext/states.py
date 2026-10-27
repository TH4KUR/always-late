"""The weight-state shim of the GPTQ extension (plan Task 6): the experiment entry points obtain their quantized states through the injected callable `env["quantize_rtn_per_output_channel_tl"](model, bits=...)`. A shim
replaces it, for `--weights gptq` or `--weights rtn-matched`, by a callable of the same signature that applies the artifact of that method and bit width; bits 12 and 16 stay the legacy rounding in every arm (audit
section 7), every other bit width raises. The default `--weights rtn` changes nothing. Two forms: `install` (eager: builds the artifacts of the given bit widths at once; used by the gate job and the tests) and
`LazyQuantizer` (builds the artifact of a bit width the first time it is requested, from the full-precision model, then reuses it; used by `run_with_weights`, which leaves the old entry points byte-identical because
some of them are hash-pinned in frozen evaluation freezes). Either way an artifact is built from the frozen calibration pool, at most once per job, and is refused unless its sha256 equals the value in the extension
freeze (`--record-weights-hashes` is for the build and gate jobs that create those values). A `weights_manifest.json` in the output directory binds the run to its artifacts: a directory whose manifest has another
method, pool or artifact hash is refused (no resume across weight sets); bit widths built later are merged into it."""
import hashlib
import json
from pathlib import Path

import torch

from . import artifact, calibrate as cb, calibration as cal, canonical, upstream

ARTIFACT_BITS = (4, 5, 6, 7, 8)
LEGACY_BITS = (12, 16)


def add_weights_args(ap):
    g = ap.add_argument_group("weights (GPTQ extension, D-101)")
    g.add_argument("--weights", choices=["rtn", "rtn-matched", "gptq"], default="rtn", help="how the quantized states are obtained: the legacy RTN (default, unchanged), the matched RTN, or GPTQ")
    g.add_argument("--gptq-freeze", help="reports/gptq_extension/freeze.json: the frozen artifact hashes")
    g.add_argument("--calibration-freeze", help="calibration_freeze.json of the pool")
    g.add_argument("--calibration-text", help="directory with the pool's token files")
    g.add_argument("--calibration-batch-size", type=int, default=16)
    g.add_argument("--record-weights-hashes", action="store_true", help="accept artifacts without a frozen hash (build and gate jobs only)")


def bits_in(labels):
    """The distinct integer bit widths among state labels (8, '8', 'INT8'); 'FP' and 'FP_rebatched' and anything else are ignored."""
    out = set()
    for x in labels or []:
        s = str(x).upper()
        s = s[3:] if s.startswith("INT") else s
        if s.isdigit():
            out.add(int(s))
    return sorted(out)


def code_hashes():
    d = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.py"))} | {f"upstream/{k}": v for k, v in upstream.PINNED_SHA256.items()}


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_artifacts(model, args, log, bits, bos_id, pad_id):
    """Build (and hash-check) the artifacts of `args.weights` for the artifact bit widths among `bits`. -> ({bits: tensors}, manifest part). `model` must be in its full-precision state (it is left unchanged)."""
    method = args.weights.replace("-", "_")
    wanted = sorted(b for b in bits_in(bits) if b in ARTIFACT_BITS)
    freeze = json.loads(Path(args.gptq_freeze).read_text()) if args.gptq_freeze else {}
    calfreeze = json.loads(Path(args.calibration_freeze).read_text())
    expect_cal = freeze.get("calibration", {})
    if expect_cal.get("freeze_file_sha256") and expect_cal["freeze_file_sha256"] != _sha_file(args.calibration_freeze):
        raise ValueError("the calibration freeze file differs from the extension freeze")
    pool = cal.load_pool(calfreeze, args.calibration_text)
    pool_sha = cal.pool_sha256(pool)
    if expect_cal.get("pool_sha256") and expect_cal["pool_sha256"] != pool_sha:
        raise ValueError("the calibration pool differs from the extension freeze")
    batches = cal.pool_batches(pool, bos_id, pad_id, batch_size=args.calibration_batch_size)
    device = next(model.parameters()).device
    artifacts, hashes, reports = {}, {}, {}
    for b in wanted:
        tensors, report = cb.calibrate(model, batches, cb.GptqConfig(bits=b), method, device, log=log)
        sha = artifact.artifact_sha256(tensors)
        expected = freeze.get("artifact_sha256", {}).get(method, {}).get(str(b))
        if expected is None and not args.record_weights_hashes:
            raise artifact.ArtifactHashMismatch(f"no frozen sha256 for {method} at {b} bits (built {sha}); the build and gate jobs record it with --record-weights-hashes")
        if expected is not None and expected != sha:
            raise artifact.ArtifactHashMismatch(f"{method} at {b} bits: built sha256 {sha} differs from the frozen {expected}")
        artifacts[b], hashes[str(b)] = tensors, sha
        reports[str(b)] = {k: report[k] for k in ("dead_total", "seconds", "peak_gpu_gb", "n_rows", "bos_trace_share")}
        log(f"{method} {b} bits: artifact sha256 {sha[:16]}..., dead columns {report['dead_total']}, {report['seconds']:.0f}s")
    part = {"method": method, "bits": wanted, "config": cb.GptqConfig(bits=0).__dict__ | {"bits": "per state"}, "pool_sha256": pool_sha, "calibration_freeze_sha256": _sha_file(args.calibration_freeze),
            "gptq_freeze_sha256": _sha_file(args.gptq_freeze) if args.gptq_freeze else None, "code_sha256": code_hashes(), "artifact_sha256": hashes, "calibration_report": reports}
    return artifacts, part


def write_manifest(out_dir, part):
    """Write or merge `weights_manifest.json` in `out_dir`; ValueError if it exists with another method, pool or artifact hash for a bit width in common. -> the manifest."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); fp = out / "weights_manifest.json"
    if not fp.exists():
        fp.write_text(json.dumps(part, indent=1, sort_keys=True) + "\n"); return part
    old = json.loads(fp.read_text())
    clash = [k for k in ("method", "pool_sha256") if old.get(k) != part[k]] + [f"artifact_sha256[{b}]" for b in set(old["artifact_sha256"]) & set(part["artifact_sha256"]) if old["artifact_sha256"][b] != part["artifact_sha256"][b]]
    if clash:
        raise ValueError(f"{out} already holds the weights manifest of another weight set ({', '.join(clash)}); use a new output directory")
    merged = {**old, "bits": sorted(set(old["bits"]) | set(part["bits"])), "artifact_sha256": {**old["artifact_sha256"], **part["artifact_sha256"]}, "calibration_report": {**old["calibration_report"], **part["calibration_report"]}}
    if merged != old:
        fp.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    return merged


class _Applier:
    """Shared behaviour of the two shims: floors stay legacy rounding, artifact bits apply their artifact, everything else raises."""
    def __init__(self, legacy, weights):
        self.legacy, self.weights = legacy, weights

    def _have(self, model, bits):
        raise NotImplementedError

    def __call__(self, model, bits=8):
        if bits in LEGACY_BITS:
            return self.legacy(model, bits=bits)
        tensors = self._have(model, bits)
        artifact.apply(model, tensors)
        return len(tensors), sum(t.numel() for t in tensors.values())


class _Eager(_Applier):
    def __init__(self, legacy, weights, artifacts):
        super().__init__(legacy, weights); self.artifacts = artifacts

    def _have(self, model, bits):
        if bits not in self.artifacts:
            raise ValueError(f"no weight state for {bits} bits under --weights {self.weights} (built: {sorted(self.artifacts)}; legacy rounding floors: {LEGACY_BITS})")
        return self.artifacts[bits]


class LazyQuantizer(_Applier):
    """Builds the artifact of a bit width on its first request (the model must then be in its full-precision state, checked against `env['original_state_cpu']`) and reuses it afterwards."""
    def __init__(self, env, args, log, out_dir):
        super().__init__(env["quantize_rtn_per_output_channel_tl"], args.weights)
        self.env, self.args, self.log, self.out_dir, self.artifacts = env, args, log, out_dir, {}

    def _assert_full_precision(self, model):
        orig = self.env.get("original_state_cpu")
        if orig is None:
            raise RuntimeError("cannot build an artifact: the entry point did not provide env['original_state_cpu'] to verify the full-precision state")
        for name, p in canonical.targets(model):
            if not torch.equal(p.data.cpu(), orig[name]):
                raise RuntimeError(f"cannot build an artifact: {name} is not in its full-precision state (restore before the first request of a bit width)")

    def _have(self, model, bits):
        if bits not in ARTIFACT_BITS:
            raise ValueError(f"no weight state for {bits} bits under --weights {self.weights} (artifact bit widths: {ARTIFACT_BITS}; legacy rounding floors: {LEGACY_BITS})")
        if bits not in self.artifacts:
            self._assert_full_precision(model)
            tok = model.tokenizer; bos = tok.bos_token_id; pad = tok.pad_token_id if tok.pad_token_id is not None else 0
            arts, part = build_artifacts(model, self.args, self.log, [bits], bos, pad)
            self.artifacts.update(arts)
            if self.out_dir is not None:
                write_manifest(self.out_dir, part)
        return self.artifacts[bits]


def install(env, model, args, log, bits, out_dir=None, bos_id=None, pad_id=None):
    """Eager form: build the artifacts of `bits` now and replace the quantizer of `env` according to `args.weights`; -> the weights manifest (a dict), or None for the legacy RTN."""
    if args.weights == "rtn":
        return None
    artifacts, part = build_artifacts(model, args, log, bits, bos_id, pad_id)
    manifest = write_manifest(out_dir, part) if out_dir is not None else part
    env["quantize_rtn_per_output_channel_tl"] = _Eager(env["quantize_rtn_per_output_channel_tl"], args.weights, artifacts)
    return manifest
