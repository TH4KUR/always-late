"""The build-and-gate job of the GPTQ extension (plan Task 8; audit section 8, gates 3 to 6): builds the artifacts of the requested methods and bit widths on the real model, records their hashes, and checks on the same model
that (a) applying an artifact changes exactly the target tensors and nothing else, (b) restoring the full-precision weights is exact after every state, (c) the full-precision logits are bit-identical before and after all
switches, (d) a second build of the first artifact gives the same hash (determinism within the job; the cross-node check compares the hashes of two jobs), and reports the next-token loss of a small probe of FLORES dev
sentences in the states FP, legacy RTN, matched RTN and GPTQ (a sanity diagnostic, not an experiment endpoint: the dev split is not the evaluation split of E1). `run_gates` is model-agnostic and tested on a tiny model;
`main` loads Gemma-2-2B the way the transfer entry point does (their functions verified against the frozen manifest, mask-regime batches).

  sbatch ... scripts/g0/run_gptq_build.sbatch gptq-gate --methods gptq rtn-matched --bits 4 6"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_lib as p1  # noqa: E402

from . import artifact, calibrate as cb, calibration as cal, canonical, states  # noqa: E402


def probe_nll(model, batches, device):
    """-> {batch language index: [sum of next-token NLL (float64), token count]} over the mask-regime batches (BOS excluded from the statistics)."""
    out = {}
    with torch.no_grad():
        for b in batches:
            logits = model(b["tokens"].to(device))
            sums, counts = p1.chunked_nll(logits, b["tokens"].to(device), b["valid"].to(device))
            s = out.setdefault(b["lang"], [0.0, 0]); s[0] += float(sums.sum()); s[1] += int(counts.sum())
    return out


def _mean(nll):
    tot, n = sum(v[0] for v in nll.values()), sum(v[1] for v in nll.values())
    return {"per_language": {k: v[0] / v[1] for k, v in nll.items() if v[1]}, "mean": tot / n, "tokens": n}


def _state_equal(model, original):
    return all(torch.equal(v, original[k]) for k, v in model.state_dict().items()) and set(model.state_dict()) == set(original)


def run_gates(env, model, probe, bos, pad, methods, bits, install_args, log, out_dir=None, max_len=64, batch_size=16):
    """probe: {language index: [token-id lists]}. -> the gate report (a dict); never raises on a failed gate (the caller decides), raises on build errors."""
    device = next(model.parameters()).device
    restore = env["restore_model_work"]; legacy = env["quantize_rtn_per_output_channel_tl"]
    batches = p1.make_batches(probe, bos, "mask", pad, max_len, batch_size)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    t0 = time.time(); report = {"methods": methods, "bits": bits, "nll": {}, "gates": {}, "manifests": {}, "rebuild": None}
    restore(); fp_logits = model(batches[0]["tokens"].to(device)).detach().clone(); report["nll"]["FP"] = _mean(probe_nll(model, batches, device))
    for b in bits:
        restore(); legacy(model, bits=b); report["nll"][f"legacy_rtn_{b}"] = _mean(probe_nll(model, batches, device))
    restore()
    for method in methods:
        env2 = dict(env)
        manifest = states.install(env2, model, install_args_for(install_args, method), log, bits=bits, out_dir=None if out_dir is None else Path(out_dir) / method.replace("-", "_"), bos_id=bos, pad_id=pad)
        report["manifests"][method] = manifest
        for b in bits:
            restore(); env2["quantize_rtn_per_output_channel_tl"](model, bits=b)
            sd = model.state_dict(); tgt = {n for n, _ in canonical.targets(model)}
            report["gates"][f"{method}_{b}"] = {"non_targets_unchanged": all(torch.equal(sd[k], original[k]) for k in sd if k not in tgt),
                                               "targets_changed": all(not torch.equal(sd[k], original[k]) for k in tgt)}
            report["nll"][f"{method}_{b}"] = _mean(probe_nll(model, batches, device))
            restore(); report["gates"][f"{method}_{b}"]["restore_exact"] = _state_equal(model, original)
        if report["rebuild"] is None and manifest["bits"]:
            b0 = manifest["bits"][0]; tensors, rep = cb.calibrate(model, cal.pool_batches(cal.load_pool(json.loads(Path(install_args.calibration_freeze).read_text()), install_args.calibration_text), bos, pad, batch_size=install_args.calibration_batch_size),
                                                                  cb.GptqConfig(bits=b0), method.replace("-", "_"), device, log=log)
            report["rebuild"] = {"method": method, "bits": b0, "first": manifest["artifact_sha256"][str(b0)], "second": artifact.artifact_sha256(tensors), "equal": manifest["artifact_sha256"][str(b0)] == artifact.artifact_sha256(tensors)}
    restore()
    report["gates"]["fp_logits_bit_identical_after_all_switches"] = bool(torch.equal(fp_logits, model(batches[0]["tokens"].to(device))))
    report["gates"]["fp_state_exact_at_the_end"] = _state_equal(model, original)
    report["seconds"] = time.time() - t0
    report["all_gates_ok"] = all(v if isinstance(v, bool) else all(v.values()) for v in report["gates"].values()) and (report["rebuild"] is None or report["rebuild"]["equal"])
    return report


def install_args_for(a, method):
    b = argparse.Namespace(**vars(a)); b.weights = method
    return b


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the artifacts of the GPTQ extension on the real model and run the gates (Task 8).")
    ap.add_argument("--out", required=True); ap.add_argument("--flores-root", required=True); ap.add_argument("--roster", required=True)
    ap.add_argument("--methods", nargs="+", default=["gptq", "rtn-matched"], choices=["gptq", "rtn-matched"]); ap.add_argument("--bits", nargs="+", type=int, default=[4])
    ap.add_argument("--n-probe", type=int, default=32, help="FLORES dev sentences per language for the loss probe"); ap.add_argument("--commit", default="unknown")
    states.add_weights_args(ap)
    a = ap.parse_args(argv)
    t0 = time.time(); log = lambda *m: print(f"[{time.time() - t0:7.0f}s]", *m, flush=True)
    import gc
    import json as _json
    import numpy as np
    from tqdm.auto import tqdm
    from transformer_lens import HookedTransformer
    import their_code as their
    import phase1_data as pd_
    import phase1_run as pr
    torch.set_grad_enabled(False)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE != "cuda":
        raise SystemExit("no GPU visible: refusing to run on the CPU")
    env = {"torch": torch, "np": np, "tqdm": tqdm, "gc": gc, "HookedTransformer": HookedTransformer, "DEVICE": DEVICE, "WORK_DEVICE": DEVICE, "BATCH_SIZE": 1, "FIRING_THRESHOLD": 0.001}
    ex = their.extract_functions(their.PRISTINE, their.THEIR_FUNCTIONS); their.verify_hashes(ex, _json.loads(their.MANIFEST.read_text())["functions"]); their.build_namespace(ex, env)
    log("their functions verified against the frozen manifest")
    model, model_name = env["load_gemma_tl_model"](pr.MODEL_CANDIDATES)
    env["model_work"] = model; env["original_state_cpu"] = env["clone_state_to_cpu"](model)
    tok = model.tokenizer; bos = tok.bos_token_id; pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    roster = pd_.load_roster(a.roster)
    dev, _ = pd_.build_corpus(a.flores_root, roster, "dev", tok, 997)
    probe = {i: s[:a.n_probe] for i, s in dev.items()}
    log("model", model_name, "probe sentences", sum(len(v) for v in probe.values()), "methods", a.methods, "bits", a.bits)
    report = run_gates(env, model, probe, bos, pad, a.methods, a.bits, a, log, out_dir=a.out)
    report.update({"commit": a.commit, "model": model_name, "languages": [r["code"] for r in roster], "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9, "device": torch.cuda.get_device_name(0)})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "gate_report.json").write_text(_json.dumps(report, indent=1, default=str) + "\n")
    log("gates ok:", report["all_gates_ok"], "mean NLL", {k: round(v["mean"], 4) for k, v in report["nll"].items()})
    if not report["all_gates_ok"]:
        raise SystemExit("a gate failed: see gate_report.json")


if __name__ == "__main__":
    main()
