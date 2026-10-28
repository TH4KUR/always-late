"""Adapter that runs the frozen Gemma functional-validation code on Gemma-2-9B with the Gemma Scope SAE (plan docs/superpowers/plans/2026-10-26-gemma9b-confirmation.md, D-089).
No frozen file is edited: for the duration of a run the constants of the frozen runners are replaced (model name, SAE repository and folder, layer); everything else (their quantizer, batching, selectivity report,
edit path, evaluation, item-file format) is the frozen code.

The SAE is Gemma Scope JumpReLU at layer 20 (width 16k, canonical), loaded by the frozen `load_sae_compat` -- no custom loader, no hook alias.  The hook is natively `blocks.20.hook_resid_post`,
matching the frozen code's expectation of `blocks.<LAYER>.hook_resid_post`.

    python phase1_gemma9b_run.py tag --out <dir> --flores-root <dir>                                  dev tagging pass -> <dir>/eq/selectivity.pt (equal-budget stream) and meta.json
    python phase1_gemma9b_run.py manifest --selectivity <dir>/eq/selectivity.pt --manifest <file>     the groups and controls (the frozen builder, K16/K64/all families)
    python phase1_gemma9b_run.py lm --out <dir> --flores-root <dir> --selectivity ... --manifest ... --precisions FP 8 6 5 4 --split eval --n-eval 300 --panel ...
                                                                                                       the frozen T-LM runner with the Study G main preset (zero-ablation, three control draws, the eight
                                                                                                       ratios of the group and of the controls, the sham)"""
import contextlib
from pathlib import Path

import torch

import phase1_func_run as fr
import phase1_gain_run as gr
import phase1_lib as p1
import phase1_run as pr
import their_code as tc

MODEL = "google/gemma-2-9b"
SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
SAE_ID = "layer_20/width_16k/canonical"
LAYER = 20
STUDY_RATIOS = (0.3535533905932738, 0.1767766952966369, 0.08838834764831845, 2.82842712474619, 5.65685424949238, 11.31370849898476, 1.1, 1.2)    # the Study G decision (D-071): down, up, fine; no recalibration on 9B


@contextlib.contextmanager
def gemma9b_setting():
    """Inside the block the frozen runners load Gemma-2-9B and the Gemma Scope SAE at layer 20 (resid_post); everything is restored on exit."""
    saved = (pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER)
    pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER = [MODEL], SAE_RELEASE, SAE_ID, LAYER, LAYER
    try:
        yield
    finally:
        pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER = saved


def tag_pipeline(forward_acts, encode, dev, bos, pad, n_feat, seed=0, n_boot=100, n_perm=50, device="cpu", batch_size=32, max_tokens=None):
    """The frozen Phase 1 tagging pass and selectivity report on the equal-token-budget stream (D-015) of the dev split, mask regime. Returns (report, info); report has the fields of selectivity.pt."""
    budget = p1.budget_for(dev)
    keep = p1.keep_masks(dev, budget, seed, split_offset=1)
    tag, offsets, lang_of = pr.run_tagging(forward_acts, encode, dev, "mask", bos, pad, n_feat, device=device, batch_size=batch_size, max_tokens=max_tokens, keep=keep)
    report = pr.selectivity_report(tag.eq, lang_of, len(dev), n_boot=n_boot, n_perm=n_perm, seed=seed)
    return report, {"stream": "eq", "budget_tokens_per_language": int(budget), "n_languages": len(dev), "seed": seed, "n_boot": n_boot, "n_perm": n_perm}


def save_selectivity(out_dir, report, names):
    """selectivity.pt in the format of `phase1_run.save_outputs` (floats as float32, languages added); returns the path."""
    if len(names) != report["rates"].shape[0]:
        raise ValueError(f"{len(names)} language names for {report['rates'].shape[0]} languages")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    keep = ("rates", "mean_activation", "S", "S_rank", "nu")
    torch.save({**{k: (v.float() if k in keep else v) for k, v in report.items()}, "languages": list(names)}, out / "selectivity.pt")
    return out / "selectivity.pt"


def build_gemma9b_manifest(selectivity_path, out_path, Ks=(16, 64, None), Rs=(5, 5, 5), seed=0):
    """The frozen group and control builder with K16, K64 and `all` families (the 9B SAE has enough controls for the whole selective set, unlike the Llama SAE)."""
    import json
    import phase1_func_groups as fg
    man = fg.build_manifest(selectivity_path, Ks=Ks, Rs=Rs, seed=seed)
    Path(out_path).write_text(json.dumps(man, separators=(",", ":")) + "\n")
    return man


def tag_main(argv, _sae_key=None):
    """The dev tagging pass of the Gemma-2-9B SAE (setup lines mirror `phase1_run.main`)."""
    import argparse, gc, json, time
    ROOT = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Gemma-2-9B dev tagging pass: per-language feature statistics and the selectivity report (equal-budget stream).")
    ap.add_argument("--out", required=True); ap.add_argument("--flores-root", required=True); ap.add_argument("--roster", default=str(ROOT / "reports/g0/phase1_languages.json"))
    ap.add_argument("--languages", nargs="*"); ap.add_argument("--max-sentences", type=int); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--n-perm", type=int, default=50); ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--max-tokens", type=int, default=1024); ap.add_argument("--commit", default="unknown")
    a = ap.parse_args(argv)
    t0 = time.time(); log = lambda *m: print(f"[{time.time() - t0:7.0f}s]", *m, flush=True)
    import numpy as np
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
    hook = getattr(sae.cfg, "hook_name", None) or f"blocks.{LAYER}.hook_resid_post"
    d_in = getattr(sae.cfg, "d_in", None) or sae.W_dec.shape[1]
    assert hook == f"blocks.{LAYER}.hook_resid_post" and d_in == model.cfg.d_model, f"SAE hook or width does not fit the model (hook={hook}, d_in={d_in}, d_model={model.cfg.d_model})"
    tok = model.tokenizer; bos = tok.bos_token_id; pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    hook_orig = getattr(sae.cfg, "hook_name_original", hook); d_sae = getattr(sae.cfg, "d_sae", None) or sae.W_dec.shape[0]; k = getattr(sae, "k", None)
    log("model", model_name, "layers", model.cfg.n_layers, "SAE", pr.SAE_ID, "hook", hook_orig, "read as", hook, "d_sae", d_sae, "k", k, "bos", bos, "pad", pad)

    def full_acts(tokens):
        _, cache = model.run_with_cache(tokens, names_filter=[hook]); return cache[hook], None
    def forward_acts(tokens):
        _, cache = model.run_with_cache(tokens, names_filter=[hook], stop_at_layer=LAYER + 1); return cache[hook], None
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
    report, info = tag_pipeline(forward_acts, encode, dev, bos, pad, int(d_sae), a.seed, a.n_boot, a.n_perm, device=DEVICE, batch_size=a.batch_size, max_tokens=a.max_tokens)
    log("[equal budget] supported", int(report["supported"].sum()), "selective", int(report["selective"].sum()), "controls", len(report["controls"]), "null cutoff", report["null_cutoff"])
    out = Path(a.out) / "eq"
    path = save_selectivity(out, report, names)
    own = {n: int(((report["peak"] == i) & report["selective"]).sum()) for i, n in enumerate(names)}
    (out / "meta.json").write_text(json.dumps({**info, "model": model_name, "sae_repo": pr.SAE_RELEASE, "sae_id": pr.SAE_ID, "hook": hook_orig, "layer_alias": hook, "bos_id": bos, "pad_id": pad,
                                              "languages": names, "selective_per_language": own, "commit": a.commit, "corpus_report": {"dev": rep_dev}, "device": DEVICE,
                                              "versions": pr.package_versions(["torch", "transformer-lens", "transformers", "safetensors", "huggingface_hub"]), "seconds": time.time() - t0, "argv": list(argv)}, indent=1, default=str) + "\n")
    log("saved", path, "selective features per language:", own)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Gemma-2-9B confirmation test (D-089): the frozen runners with the 9B setting (the remaining arguments go to the frozen code).", allow_abbrev=False)
    ap.add_argument("task", choices=["tag", "lm", "manifest"])
    a, rest = ap.parse_known_args(argv)
    if a.task == "manifest":
        mp = argparse.ArgumentParser(); mp.add_argument("--selectivity", required=True); mp.add_argument("--manifest", required=True); mp.add_argument("--seed", type=int, default=0)
        m = mp.parse_args(rest)
        man = build_gemma9b_manifest(m.selectivity, m.manifest, seed=m.seed)
        print("wrote", m.manifest, {n: man["sizes"][n] for n in man["languages"]})
    elif a.task == "lm":
        if "--ratios" in rest or "--preset" in rest:
            ap.error("--ratios and --preset are fixed by this wrapper (the Study G down ratios, three control draws)")
        with gemma9b_setting():
            gr.main(["--ratios", *[str(r) for r in STUDY_RATIOS], "--n-controls", "3", "--label", "K64"] + rest)
    else:
        with gemma9b_setting():
            tag_main(rest)


if __name__ == "__main__":
    main()
