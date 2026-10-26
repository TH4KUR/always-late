"""Adapter that runs the frozen Gemma functional-validation code on Gemma-2-9B with Gemma Scope `layer_20/width_16k/canonical` (D-089, plan docs/superpowers/plans/2026-10-26-gemma9b-confirmation.md, handoff 040).
No frozen file is edited: for the duration of a run the constants of the frozen runners are replaced (model name, SAE release and id, layer). The SAE loads through the frozen `load_sae_compat` (same family as the 2B
study), so unlike the Llama adapter there is no loader replacement and no hook alias: the hook is natively `blocks.20.hook_resid_post`.

    python phase1_gemma9b_run.py tag --out <dir> --flores-root <dir>                    dev tagging pass -> <dir>/eq/selectivity.pt and meta.json
    python phase1_gemma9b_run.py manifest --selectivity <dir>/eq/selectivity.pt --manifest <file> [--families k]
                                                                                       the frozen group builder (families K16, K64 and `all`; `--families k` drops `all` if the control pool is too small, which must be reported)
    python phase1_gemma9b_run.py lm --out <dir> --flores-root <dir> --selectivity ... --manifest ... --precisions FP 8 7 6 5 4 --split eval --n-eval 300 --panel ...
                                                                                       the frozen T-LM runner with the Study G main preset (the 2B ratios, three control draws)"""
import contextlib
import os
from pathlib import Path

import phase1_func_run as fr
import phase1_gain_run as gr
import phase1_lib as p1
import phase1_llama_run as lr
import phase1_run as pr
import their_code as tc

MODEL = "google/gemma-2-9b"
SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
SAE_ID = "layer_20/width_16k/canonical"
LAYER = 20
HOOK = f"blocks.{LAYER}.hook_resid_post"
STUDY_RATIOS = lr.STUDY_RATIOS                   # the Study G decision (D-071): the 2B ratios, no recalibration on 9B
FAMILIES = {"full": ((16, 64, None), (5, 5, 3)), "k": ((16, 64), (5, 5))}
N_DEVICES_ENV = "GEMMA9B_N_DEVICES"              # GPUs the model is split over (default 1: a card of >= 40 GB); 4 on Ada's 11 GB RTX 2080 Ti nodes


def default_n_devices():
    return int(os.environ.get(N_DEVICES_ENV, "1"))


def first_device_outputs(model, device):
    """Wraps `model.forward` so that a tensor output (logits, or the residual of `stop_at_layer`) is returned on `device`, the device of the tokens and the SAE.
    TransformerLens moves the residual between the blocks of a multi-device model, but returns the output on the device of the last block it ran; the frozen code indexes it with masks and
    reads it against tokens on the first device, which needs it there. Nothing else changes (hooks and `run_with_cache` still see the activations where they are computed)."""
    import torch
    inner = model.forward

    def forward(*args, **kwargs):
        out = inner(*args, **kwargs)
        return out.to(device) if isinstance(out, torch.Tensor) else out
    model.forward = forward
    return model


def make_multi_device_loader(n_devices, hooked_transformer, device="cuda"):
    """A replacement of the frozen `load_gemma_tl_model` that splits the model over `n_devices` GPUs (bf16, no processing: the same load call as the frozen function plus `n_devices`)."""
    import torch
    if n_devices < 2:
        raise ValueError("a multi-device loader needs n_devices >= 2")

    def load(candidates, device=device):
        last = None
        for name in candidates:
            try:
                print(f"Trying model: {name} on {n_devices} devices", flush=True)
                model = hooked_transformer.from_pretrained_no_processing(name, device=device, dtype=torch.bfloat16, n_devices=n_devices)
                model.eval()
                first_device_outputs(model, device)
                print("Loaded:", name, "Layers:", model.cfg.n_layers, "d_model:", model.cfg.d_model, "n_devices:", n_devices, flush=True)
                return model, name
            except Exception as e:                                   # the frozen loader tries every candidate too
                print("Failed:", repr(e)[:500], flush=True)
                last = e
        raise RuntimeError(f"no model candidate could be loaded: {last!r}")
    return load


@contextlib.contextmanager
def gemma9b_setting(n_devices=1):
    """Inside the block the frozen runners load Gemma-2-9B and the layer-20 Gemma Scope SAE; everything is restored on exit.
    With n_devices > 1 the namespace builder is wrapped so that the frozen model loader is replaced by the multi-device one (the SAE loader and every other frozen function stay)."""
    saved = (pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER, tc.build_namespace)
    orig_build = tc.build_namespace

    def build_namespace(ex, env):
        orig_build(ex, env)
        env["load_gemma_tl_model"] = make_multi_device_loader(n_devices, env["HookedTransformer"], env.get("DEVICE", "cuda"))

    pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER = [MODEL], SAE_RELEASE, SAE_ID, LAYER, LAYER
    if n_devices > 1:
        tc.build_namespace = build_namespace
    try:
        yield
    finally:
        pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER, tc.build_namespace = saved


def build_manifest(selectivity_path, out_path, families="full", seed=0):
    """The frozen group and control builder. families="full": K16, K64 and `all` (as the 2B manifest); "k": K16 and K64 only. A K64 group larger than the control pool is still refused by the frozen code."""
    import json
    import phase1_func_groups as fg
    if families not in FAMILIES:
        raise ValueError(f"unknown families {families!r}; choose from {sorted(FAMILIES)}")
    Ks, Rs = FAMILIES[families]
    man = fg.build_manifest(selectivity_path, Ks=Ks, Rs=Rs, seed=seed)
    Path(out_path).write_text(json.dumps(man, separators=(",", ":")) + "\n")
    return man


def check_sae_fits(hook, d_in, d_model):
    """The runner's hook and width assertions as a function (tested with fakes)."""
    if hook != HOOK:
        raise AssertionError(f"SAE hook {hook!r} is not {HOOK!r}")
    if d_in != d_model:
        raise AssertionError(f"SAE width d_in={d_in} does not fit d_model={d_model}")


def tag_main(argv):
    """The dev tagging pass (setup lines mirror `phase1_llama_run.tag_main`, which was exercised on the cluster)."""
    import argparse, gc, json, time
    ROOT = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Gemma-2-9B dev tagging pass: per-language feature statistics and the selectivity report (equal-budget stream).")
    ap.add_argument("--out", required=True); ap.add_argument("--flores-root", required=True); ap.add_argument("--roster", default=str(ROOT / "reports/g0/phase1_languages.json"))
    ap.add_argument("--languages", nargs="*"); ap.add_argument("--max-sentences", type=int); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--n-perm", type=int, default=50); ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--max-tokens", type=int, default=1024); ap.add_argument("--commit", default="unknown")
    a = ap.parse_args(argv)
    t0 = time.time(); log = lambda *m: print(f"[{time.time() - t0:7.0f}s]", *m, flush=True)
    import numpy as np, torch
    from tqdm.auto import tqdm
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    import phase1_data as pd_
    torch.set_grad_enabled(False)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    env = {"torch": torch, "np": np, "tqdm": tqdm, "gc": gc, "HookedTransformer": HookedTransformer, "SAE": SAE, "DEVICE": DEVICE, "WORK_DEVICE": DEVICE, "BATCH_SIZE": 1, "FIRING_THRESHOLD": 0.001}
    ex = tc.extract_functions(tc.PRISTINE, tc.THEIR_FUNCTIONS); tc.verify_hashes(ex, json.loads(tc.MANIFEST.read_text())["functions"]); tc.build_namespace(ex, env)
    log("their functions verified against the frozen manifest")
    model, model_name = env["load_gemma_tl_model"](pr.MODEL_CANDIDATES)
    sae = env["load_sae_compat"](pr.SAE_RELEASE, pr.SAE_ID, DEVICE)
    hook = getattr(sae.cfg, "hook_name", None) or HOOK
    check_sae_fits(hook, sae.cfg.d_in, model.cfg.d_model)
    tok = model.tokenizer; bos = tok.bos_token_id; pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    log("model", model_name, "layers", model.cfg.n_layers, "d_model", model.cfg.d_model, "SAE", pr.SAE_ID, "hook", hook, "d_sae", sae.cfg.d_sae, "bos", bos, "pad", pad)

    def full_acts(tokens):
        _, cache = model.run_with_cache(tokens, names_filter=[hook]); return cache[hook].to(DEVICE), None
    def forward_acts(tokens):                                                     # no logits needed: stop after the hooked block (the same activations)
        _, cache = model.run_with_cache(tokens, names_filter=[hook], stop_at_layer=LAYER + 1); return cache[hook].to(DEVICE), None
    encode = lambda acts: env["encode_sae"](sae, acts)
    roster = pd_.load_roster(a.roster)
    if a.languages:
        roster = [r for r in roster if r["code"] in set(a.languages)]
    dev, rep_dev = pd_.build_corpus(a.flores_root, roster, "dev", tok, None if a.languages else 997)
    if a.max_sentences:
        dev = {k: v[:a.max_sentences] for k, v in dev.items()}
    names = [r["code"] for r in roster]
    probe = p1.make_batches({0: dev[0][:8]}, bos, "mask", pad, 256, 8)[0]["tokens"].to(DEVICE)
    pr.check_forward_paths(full_acts, forward_acts, probe); log("stop-at-layer tagging forward equals the full forward pass on a probe batch")
    report, info = lr.tag_pipeline(forward_acts, encode, dev, bos, pad, int(sae.cfg.d_sae), a.seed, a.n_boot, a.n_perm, device=DEVICE, batch_size=a.batch_size, max_tokens=a.max_tokens)
    log("[equal budget] supported", int(report["supported"].sum()), "selective", int(report["selective"].sum()), "controls", len(report["controls"]), "null cutoff", report["null_cutoff"])
    out = Path(a.out) / "eq"
    path = lr.save_selectivity(out, report, names)
    own = {n: int(((report["peak"] == i) & report["selective"]).sum()) for i, n in enumerate(names)}
    (out / "meta.json").write_text(json.dumps({**info, "model": model_name, "sae_repo": pr.SAE_RELEASE, "sae_id": pr.SAE_ID, "hook": hook, "layer": LAYER, "bos_id": bos, "pad_id": pad,
                                              "languages": names, "selective_per_language": own, "commit": a.commit, "corpus_report": {"dev": rep_dev}, "device": DEVICE,
                                              "versions": pr.package_versions(["torch", "transformer-lens", "transformers", "sae-lens", "safetensors", "huggingface_hub"]), "seconds": time.time() - t0, "argv": list(argv)}, indent=1, default=str) + "\n")
    log("saved", path, "selective features per language:", own)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Gemma-2-9B confirmation (D-089): the frozen runners with the 9B setting (the remaining arguments go to the frozen code).", allow_abbrev=False)
    ap.add_argument("task", choices=["tag", "lm", "manifest"])
    ap.add_argument("--n-devices", type=int, default=default_n_devices(), help=f"GPUs to split the model over (default ${N_DEVICES_ENV} or 1)")
    a, rest = ap.parse_known_args(argv)
    if a.task == "manifest":
        mp = argparse.ArgumentParser(); mp.add_argument("--selectivity", required=True); mp.add_argument("--manifest", required=True); mp.add_argument("--seed", type=int, default=0)
        mp.add_argument("--families", choices=sorted(FAMILIES), default="full")
        m = mp.parse_args(rest)
        man = build_manifest(m.selectivity, m.manifest, families=m.families, seed=m.seed)
        print("wrote", m.manifest, {n: man["sizes"][n] for n in man["languages"]})
    elif a.task == "lm":
        if "--ratios" in rest or "--preset" in rest:
            ap.error("--ratios and --preset are fixed by this wrapper (the Study G down ratios, three control draws)")
        with gemma9b_setting(a.n_devices):
            gr.main(["--ratios", *[str(r) for r in STUDY_RATIOS], "--n-controls", "3", "--label", "K64"] + rest)
    else:
        with gemma9b_setting(a.n_devices):
            tag_main(rest)


if __name__ == "__main__":
    main()
