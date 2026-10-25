"""Adapter that runs the frozen Gemma functional-validation code on Llama-3.2-1B with the TopK SAE of `seonglae/Llama-3.2-1B-sae` (plan docs/superpowers/plans/2026-10-26-llama-consistency-check.md, D-087).
No frozen file is edited: for the duration of a run the constants of the frozen runners are replaced (model name, SAE repository and folder, layer) and the SAE loader of their namespace is replaced by
`phase1_llama_sae.load_sae_compat_llama`; everything else (their quantizer, batching, selectivity report, edit path, evaluation, item-file format) is the frozen code.

The SAE hook `blocks.12.hook_resid_pre` is the residual that `blocks.11.hook_resid_post` holds; the frozen code reads `blocks.<LAYER>.hook_resid_post` and starts the upper half at `LAYER + 1`, so
LAYER is 11 and the loaded SAE reports the alias as its hook (its original hook is checked at load time).

    python phase1_llama_run.py tag --out <dir> --flores-root <dir> [--sae fineweb42|faithful42]     dev tagging pass -> <dir>/eq/selectivity.pt (equal-budget stream) and meta.json
    python phase1_func_groups.py build --selectivity <dir>/eq/selectivity.pt --manifest <file>      the groups and controls (frozen, model free)
    python phase1_llama_run.py lm --out <dir> --flores-root <dir> --selectivity ... --manifest ... --precisions FP 8 6 5 4 --split eval --n-eval 300 --panel ...
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

MODEL = "meta-llama/Llama-3.2-1B"
SAE_REPO = "seonglae/Llama-3.2-1B-sae"
SAE_IDS = {"fineweb42": "Llama-3.2-1B_blocks.12.hook_resid_pre_14336_topk_48_0.0002_42_fineweb_512",
           "faithful42": "Llama-3.2-1B_blocks.12.hook_resid_pre_14336_topk_48_0.0002_42_faithful-llama3.2-1b_512"}
HOOK = "blocks.12.hook_resid_pre"
HOOK_ALIAS = "blocks.11.hook_resid_post"
LAYER = 11
STUDY_RATIOS = (0.3535533905932738, 0.1767766952966369, 0.08838834764831845, 2.82842712474619, 5.65685424949238, 11.31370849898476, 1.1, 1.2)    # the Study G decision (D-071): down, up, fine; no new calibration on Llama


@contextlib.contextmanager
def llama_setting(sae_key="fineweb42"):
    """Inside the block the frozen runners load Llama-3.2-1B and the chosen SAE at layer 11 (resid_post, = resid_pre of block 12); everything is restored on exit."""
    if sae_key not in SAE_IDS:
        raise ValueError(f"unknown SAE {sae_key!r}; choose from {sorted(SAE_IDS)}")
    saved = (pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER, tc.build_namespace)
    orig_build = tc.build_namespace

    def build_namespace(ex, env):
        orig_build(ex, env)
        import phase1_llama_sae as ls
        env["load_sae_compat"] = lambda release, sae_id, device=None: ls.load_sae_compat_llama(release, sae_id, device or env.get("DEVICE", "cpu"), hook_alias=HOOK_ALIAS, expected_hook=HOOK)

    pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER, tc.build_namespace = [MODEL], SAE_REPO, SAE_IDS[sae_key], LAYER, LAYER, build_namespace
    try:
        yield
    finally:
        pr.MODEL_CANDIDATES, pr.SAE_RELEASE, pr.SAE_ID, pr.LAYER, fr.LAYER, tc.build_namespace = saved


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


def tag_main(argv, sae_key):
    """The dev tagging pass of the Llama SAE (setup lines mirror `phase1_run.main`, which was exercised on the cluster)."""
    import argparse, gc, json, time
    ROOT = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Llama-3.2-1B dev tagging pass: per-language feature statistics and the selectivity report (equal-budget stream).")
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
    hook = sae.cfg.hook_name
    assert hook == f"blocks.{LAYER}.hook_resid_post" and sae.cfg.d_in == model.cfg.d_model, "SAE hook or width does not fit the model"
    tok = model.tokenizer; bos = tok.bos_token_id; pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    log("model", model_name, "layers", model.cfg.n_layers, "SAE", pr.SAE_ID, "hook", sae.cfg.hook_name_original, "read as", hook, "d_sae", sae.cfg.d_sae, "k", sae.k, "bos", bos, "pad", pad)

    def full_acts(tokens):
        _, cache = model.run_with_cache(tokens, names_filter=[hook]); return cache[hook], None
    def forward_acts(tokens):                                                     # no logits needed: stop after the hooked block (the same activations)
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
    report, info = tag_pipeline(forward_acts, encode, dev, bos, pad, int(sae.cfg.d_sae), a.seed, a.n_boot, a.n_perm, device=DEVICE, batch_size=a.batch_size, max_tokens=a.max_tokens)
    log("[equal budget] supported", int(report["supported"].sum()), "selective", int(report["selective"].sum()), "controls", len(report["controls"]), "null cutoff", report["null_cutoff"])
    out = Path(a.out) / "eq"
    path = save_selectivity(out, report, names)
    own = {n: int(((report["peak"] == i) & report["selective"]).sum()) for i, n in enumerate(names)}
    (out / "meta.json").write_text(json.dumps({**info, "model": model_name, "sae_repo": pr.SAE_RELEASE, "sae_id": pr.SAE_ID, "hook": sae.cfg.hook_name_original, "layer_alias": hook, "bos_id": bos, "pad_id": pad,
                                              "languages": names, "selective_per_language": own, "commit": a.commit, "corpus_report": {"dev": rep_dev}, "device": DEVICE,
                                              "versions": pr.package_versions(["torch", "transformer-lens", "transformers", "safetensors", "huggingface_hub"]), "seconds": time.time() - t0, "argv": list(argv)}, indent=1, default=str) + "\n")
    log("saved", path, "selective features per language:", own)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Llama-3.2-1B consistency check (D-087): the frozen runners with the Llama setting (the remaining arguments go to the frozen code).", allow_abbrev=False)
    ap.add_argument("task", choices=["tag", "lm"]); ap.add_argument("--sae", choices=sorted(SAE_IDS), default="fineweb42")
    a, rest = ap.parse_known_args(argv)
    if a.task == "lm":
        if "--ratios" in rest or "--preset" in rest:
            ap.error("--ratios and --preset are fixed by this wrapper (the Study G down ratios, three control draws)")
        with llama_setting(a.sae):
            gr.main(["--ratios", *[str(r) for r in STUDY_RATIOS], "--n-controls", "3", "--label", "K64"] + rest)
    else:
        with llama_setting(a.sae):
            tag_main(rest, a.sae)


if __name__ == "__main__":
    main()
