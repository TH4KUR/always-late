"""N11 step 1 (decision D-025): the JumpReLU thresholds theta_f of the SAE we use (Gemma Scope, Lieberum et al. 2024, arXiv:2408.05147: a feature is active when its pre-activation
exceeds a learned threshold) written to a small CSV with provenance, so that the margin of an activation above its feature's threshold can be used as a covariate (handoff 007).
The SAE is loaded with THEIR function (`load_sae_compat`, taken verbatim by AST, hash verified against the frozen manifest) on the CPU; nothing else of their script runs.
  python3 extract_sae_thresholds.py --out sae_thresholds.csv       (needs sae_lens and the offline Hugging Face cache; run once on the Ada login node, no GPU, no Slurm job)"""
import hashlib
import sys
from pathlib import Path

import numpy as np

RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"


def thresholds_from_sae(sae, n_feat=None):
    """The `threshold` vector of a JumpReLU SAE as float32, in feature order; ValueError if it is absent, not one-dimensional, of the wrong length, or not positive and finite."""
    thr = getattr(sae, "threshold", None)
    if thr is None:
        raise ValueError("the SAE has no 'threshold' attribute (not a JumpReLU SAE?)")
    t = np.asarray(thr.detach().float().cpu().numpy() if hasattr(thr, "detach") else thr, dtype=np.float32)
    if t.ndim != 1:
        raise ValueError(f"expected a one-dimensional threshold vector, got shape {t.shape}")
    if n_feat is not None and len(t) != n_feat:
        raise ValueError(f"{len(t)} thresholds for {n_feat} features")
    if not (np.isfinite(t).all() and (t > 0).all()):
        raise ValueError("thresholds must be positive and finite")
    return t


def write_thresholds_csv(path, thresholds, meta):
    t = np.asarray(thresholds, dtype=np.float32); lines = [f"# {k}: {v}" for k, v in meta.items()]
    lines += [f"# d_sae: {len(t)}", f"# threshold_sha256: {hashlib.sha256(t.tobytes()).hexdigest()}", "feature,threshold"]
    lines += [f"{i},{np.format_float_positional(x, unique=True, trim='0')}" for i, x in enumerate(t)]
    Path(path).write_text("\n".join(lines) + "\n")


def read_thresholds(path, n_feat=None):
    rows = [l for l in Path(path).read_text().splitlines() if l and not l.startswith("#")]
    if not rows or rows[0] != "feature,threshold":
        raise ValueError(f"{path}: the header line 'feature,threshold' is missing")
    vals = []
    for k, l in enumerate(rows[1:]):
        i, v = l.split(",")
        if int(i) != k:
            raise ValueError(f"{path}: feature order broken at line {k + 1} (found {i})")
        vals.append(np.float32(v))
    t = np.array(vals, dtype=np.float32)
    if n_feat is not None and len(t) != n_feat:
        raise ValueError(f"{path}: expected {n_feat} features, found {len(t)}")
    if not (np.isfinite(t).all() and (t > 0).all()):
        raise ValueError(f"{path}: thresholds must be positive and finite")
    return t


def extract(load_sae, release, sae_id, out_path):
    """`load_sae(release, sae_id, device)` is their loader signature; the SAE is loaded on the CPU. Nothing is written if the thresholds are not valid."""
    sae = load_sae(release, sae_id, "cpu"); t = thresholds_from_sae(sae, getattr(getattr(sae, "cfg", None), "d_sae", None))
    try:
        from importlib.metadata import version
        sl = version("sae-lens")
    except Exception:
        sl = "unknown"
    write_thresholds_csv(out_path, t, {"release": release, "sae_id": sae_id, "sae_lens": sl, "source": "SAE.from_pretrained via their load_sae_compat, CPU; the SAE 'threshold' parameter"}); return t


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--out", required=True)
    ap.add_argument("--release", default=RELEASE); ap.add_argument("--sae-id", default=SAE_ID); a = ap.parse_args(argv)
    import torch
    from sae_lens import SAE
    import their_code as tc
    torch.set_grad_enabled(False)
    env = {"torch": torch, "SAE": SAE, "DEVICE": "cpu"}
    ex = tc.extract_functions(tc.PRISTINE, ["load_sae_compat"]); tc.verify_hashes(ex, json.loads(tc.MANIFEST.read_text())["functions"]); tc.build_namespace(ex, env)
    t = extract(env["load_sae_compat"], a.release, a.sae_id, a.out)
    print(f"wrote {a.out}: {len(t)} thresholds, min {t.min():.4g}, median {np.median(t):.4g}, max {t.max():.4g}")


if __name__ == "__main__":
    sys.exit(main())
