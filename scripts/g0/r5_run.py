"""R5: OUR pipeline (HF transformers, one bf16 model quantized in place) run on THEIR tokens, with one factor flipped per cell.
Goal (reports/audit_our_vs_duan.md): find which implementation differences move INT7/INT6 survival from our 98%/94% to their 81%/51%.
Outputs per cell follow their schema (phase3_summary_final.csv, <condition>_per_feature.csv) so scripts/g0/baseline_report.py can compare them.

Usage on Ada (see run_r5.sbatch):  python r5_run.py --cells T,O --bits 8,7,6 --out /home2/karma/duan-verify/r5/<job>
"""
import argparse, copy, gc, hashlib, json, math, os, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if HERE.name == "g0":      # in the repo; on Ada src/ is deployed next to this file
    sys.path.insert(0, str(HERE.parents[1]))
import r5_lib as lib  # noqa: E402
from src.pipeline import ResidualCapture  # noqa: E402
from src.rtn_quant import quantized_model  # noqa: E402  (OUR quantizer: per output row)
from src.sae_models import JumpReLUSAE  # noqa: E402

FIRING_THRESHOLD = 0.001
SUMMARY_COLS = ["condition", "n_tokens", "n_total_features", "n_active_features", "mean_corr", "median_corr", "survived_>0.9_pct",
                "degraded_0.5_0.9_pct", "damaged_<0.5_pct", "layer", "bits", "perplexity", "loss", "ppl_delta_pct", "method"]


@dataclass(frozen=True)
class CellSpec:
    attn: str        # eager (applies Gemma-2 attention soft-capping, as the model Gemma Scope was trained on) | sdpa (HF default; silently skips it)
    sae_dtype: str   # fp32 (theirs, via sae_lens) | bf16 (ours, SAE cast to the model dtype)
    quant: str       # duan (their grouping: q/k/v scales shared across heads) | ours (per output row for every module)
    regime: str      # none (theirs) | mask | keep (BOS prepended to the identical windows)


_T = CellSpec(attn="eager", sae_dtype="fp32", quant="duan", regime="none")
CELLS = {
    "T": _T,                                                # "theirs-in-HF" anchor: should reproduce the vendored baseline if the factors are enumerated
    "O": CellSpec("sdpa", "bf16", "ours", "none"),          # our pipeline as it stands, no BOS, on their tokens
    "O_keep": CellSpec("sdpa", "bf16", "ours", "keep"),     # our pipeline as it stands INCLUDING BOS, on their tokens (links R5 back to our original numbers)
    "T_sdpa": CellSpec("sdpa", "fp32", "duan", "none"),
    "T_bf16sae": CellSpec("eager", "bf16", "duan", "none"),
    "T_rowquant": CellSpec("eager", "fp32", "ours", "none"),
    "T_mask": CellSpec("eager", "fp32", "duan", "mask"),
    "T_keep": CellSpec("eager", "fp32", "duan", "keep"),
}


# ------------------------------------------------------------------ losses
def tl_style_loss(logits, ids):
    """TransformerLens' lm_cross_entropy_loss: log_softmax in the logits dtype (bf16), gather, mean. This is what their perplexity uses."""
    lp = torch.log_softmax(logits, dim=-1)
    return float(-lp[..., :-1, :].gather(-1, ids[..., 1:, None])[..., 0].mean())


def fp32_loss(logits, ids):
    return float(torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)))


# ------------------------------------------------------------------ model helpers
def set_attention(model, impl):
    if impl not in ("eager", "sdpa"):
        raise ValueError(f"attention implementation {impl!r} not supported here")
    model.set_attn_implementation(impl)


def _forward(model, cap, ids):
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    return logits, cap.activation


def _quant_ctx(spec_quant, model):
    if spec_quant == "duan":
        return lambda bits: lib.quantized_model_duan(model, bits, head_dim=model.config.head_dim)
    if spec_quant == "ours":
        return lambda bits: quantized_model(model, bits)
    raise ValueError(spec_quant)


def _row(condition, bits, result, summ, losses_tl, losses_32, ppl_fp, method, cell, layer, d_sae):
    loss, loss32 = float(np.mean(losses_tl)), float(np.mean(losses_32))
    ppl, ppl32 = float(np.exp(loss)), float(np.exp(loss32))
    return {"condition": condition, "n_tokens": int(result["n"]), "n_total_features": int(d_sae), "n_active_features": summ["n_active"],
            "mean_corr": summ["mean_corr"], "median_corr": summ["median_corr"], "survived_>0.9_pct": summ["survived_pct"],
            "degraded_0.5_0.9_pct": summ["degraded_pct"], "damaged_<0.5_pct": summ["damaged_pct"], "layer": int(layer), "bits": bits,
            "perplexity": ppl, "loss": loss, "ppl_delta_pct": 0.0 if ppl_fp is None else (ppl / ppl_fp - 1) * 100, "method": method,
            "perplexity_fp32": ppl32, "loss_fp32": loss32, "cell": cell}


def _per_feature(condition, layer, result, summ):
    corr, active = result["corr"].numpy(), summ["active"].numpy()
    return pd.DataFrame({"feature_id": np.arange(len(corr)), "condition": condition, "layer": int(layer), "corr": corr,
                         "firing_rate": result["firing_rate"].numpy(), "mean_activation": result["mean_activation"].numpy(),
                         "max_activation": result["max_activation"].numpy(), "active": active,
                         "survived_>0.9": (corr > 0.9) & active, "damaged_<0.5": (corr < 0.5) & active})


def run_cell(name, spec, model, sae, windows, bos_id, bits, out_dir, layer, device, logger=None):
    """One cell: FP pass (features cached sparsely on CPU) then one pass per bit-width; feature statistics and perplexity share a forward pass."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    prev_attn = model.config._attn_implementation
    set_attention(model, spec.attn)
    sae = copy.deepcopy(sae).to(device)
    if spec.sae_dtype == "bf16":
        sae = sae.to(torch.bfloat16)
    elif spec.sae_dtype != "fp32":
        raise ValueError(spec.sae_dtype)
    tokens = lib.apply_regime(windows, spec.regime, bos_id)
    start = lib.stats_start(spec.regime)
    d_sae = sae.W_enc.shape[1]
    qctx = _quant_ctx(spec.quant, model)
    cap = ResidualCapture(model, layer)
    store, rows, files = lib.SparseFeatureStore(), [], {}
    n = tokens.shape[0]

    def one_pass(condition, bits_, fp):
        stream, losses_tl, losses_32, t0 = lib.PearsonStream(d_sae, device), [], [], time.time()
        for i in range(n):
            ids = tokens[i:i + 1].to(device)
            logits, resid = _forward(model, cap, ids)
            y = sae.encode(resid[0]).float()[start:]
            if fp:
                store.add(y); x = y
            else:
                x = store.get(i, device)
            stream.update(x, y)
            losses_tl.append(tl_style_loss(logits, ids)); losses_32.append(fp32_loss(logits, ids))
            if logger is not None and ((i + 1) % 50 == 0 or i + 1 == n):
                logger.heartbeat(condition, i + 1, n, time.time() - t0, {"cuda": torch.cuda.max_memory_allocated() / 2**30} if torch.cuda.is_available() else {})
            del logits, resid, y, x
        result = stream.result(); summ = lib.summarize(result["corr"], result["firing_rate"], FIRING_THRESHOLD)
        return result, summ, losses_tl, losses_32

    try:
        ppl_fp = None
        for bits_ in [16] + list(bits):
            fp = bits_ == 16
            condition, slug, method = ("FP16 baseline", "FP16_baseline", "FP16") if fp else (f"RTN INT{bits_}", f"RTN_INT{bits_}", "RTN")
            if fp:
                result, summ, ltl, l32 = one_pass(condition, bits_, True)
            else:
                with qctx(bits_):
                    result, summ, ltl, l32 = one_pass(condition, bits_, False)
            row = _row(condition, bits_, result, summ, ltl, l32, ppl_fp, method, name, layer, d_sae)
            if fp:
                ppl_fp = row["perplexity"]
            rows.append(row)
            _per_feature(condition, layer, result, summ).to_csv(out_dir / f"{slug}_per_feature.csv", index=False)
            files[slug] = str(out_dir / f"{slug}_per_feature.csv")
            extra = [c for c in rows[0] if c not in SUMMARY_COLS]
            pd.DataFrame(rows)[SUMMARY_COLS + extra].to_csv(out_dir / "phase3_summary_partial.csv", index=False)
            if logger is not None:
                logger.log_condition(row)
            print(f"[{name}] {condition}: active={row['n_active_features']} survived={row['survived_>0.9_pct']:.2f}% median_r={row['median_corr']:.4f} "
                  f"ppl={row['perplexity']:.3f} ({row['ppl_delta_pct']:+.2f}%)", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        cap.remove()
        set_attention(model, prev_attn)
    pd.DataFrame(rows)[SUMMARY_COLS + extra].to_csv(out_dir / "phase3_summary_final.csv", index=False)
    (out_dir / "phase3_per_feature_files.json").write_text(json.dumps(files, indent=2))
    (out_dir / "cell_done.json").write_text(json.dumps({"cell": name, "spec": asdict(spec), "bits": list(bits), "n_windows": int(n)}, indent=2))
    return rows


def cell_is_done(out_dir):
    return (Path(out_dir) / "cell_done.json").exists()


# ------------------------------------------------------------------ self-checks (quantify, do not assume)
def int0_identity(model, windows, device, bits=8, ctx=None):
    """Quantize then restore without using the quantized model: the FP logits must be bit-identical afterwards (restore integrity + determinism)."""
    ctx = ctx if ctx is not None else quantized_model
    ids = windows.to(device)

    def logits():
        with torch.no_grad():
            return torch.cat([model(input_ids=ids[i:i + 1]).logits for i in range(ids.shape[0])])
    before = logits()
    with ctx(model, bits):
        pass
    after = logits()
    return {"identical": bool(torch.equal(before, after)), "max_abs_diff": float((before.float() - after.float()).abs().max())}


def sdpa_vs_eager(model, windows, layer, device):
    """Size of the effect of HF's sdpa path skipping Gemma-2 attention soft-capping, on layer-`layer` resid_post, logits and loss.
    Window by window (a 256k-vocab logits tensor per window is too big to keep for all of them)."""
    prev, cap, ids = model.config._attn_implementation, ResidualCapture(model, layer), windows.to(device)
    sq_diff = sq_ref = 0.0; resid_max = logit_max = 0.0; le, ls = [], []
    try:
        for i in range(ids.shape[0]):
            out = {}
            for impl in ("eager", "sdpa"):
                set_attention(model, impl)
                logits, resid = _forward(model, cap, ids[i:i + 1])
                out[impl] = (resid.float().clone(), logits.float(), fp32_loss(logits, ids[i:i + 1]))
            (ra, la, a), (rb, lb, b) = out["eager"], out["sdpa"]
            sq_diff += float(((ra - rb) ** 2).sum()); sq_ref += float((ra ** 2).sum())
            resid_max = max(resid_max, float((ra - rb).abs().max())); logit_max = max(logit_max, float((la - lb).abs().max())); le.append(a); ls.append(b)
            del out, la, lb
    finally:
        cap.remove(); set_attention(model, prev)
    return {"resid_rel_l2": math.sqrt(sq_diff / sq_ref), "resid_max_abs": resid_max, "logit_max_abs": logit_max,
            "loss_eager": float(np.mean(le)), "loss_sdpa": float(np.mean(ls)), "loss_delta": float(np.mean(ls) - np.mean(le))}


def feature_agreement(a, b):
    """Compare two dense feature matrices [n, d_sae]: active-set overlap and value differences where both are active."""
    ma, mb = a > 0, b > 0; both = ma & mb
    union = int((ma | mb).sum())
    diff = (a.float() - b.float()).abs()[both]
    return {"n_active_a": int(ma.sum()), "n_active_b": int(mb.sum()), "n_both": int(both.sum()), "active_jaccard": (int(both.sum()) / union) if union else 1.0,
            "max_abs_diff_common": float(diff.max()) if diff.numel() else 0.0,
            "mean_rel_diff_common": float((diff / a.float()[both].abs().clamp(min=1e-12)).mean()) if diff.numel() else 0.0}


# ------------------------------------------------------------------ CLI
def build_tokens(tokenizer, budget):
    """Their build_tokens_2d, verbatim in effect: WikiText-2 (train for budgets > 20k), non-empty lines joined by blank lines, no special tokens."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test" if budget <= 20_000 else "train")
    return torch.tensor(tokenizer.encode("\n\n".join(x for x in ds["text"] if x.strip()), add_special_tokens=False), dtype=torch.long)


def selfchecks(model, sae, windows, args, device, out):
    k = min(args.selfcheck_windows, windows.shape[0]); w = windows[:k]
    res = {"tokens_sha256": hashlib.sha256(windows.numpy().tobytes()).hexdigest(), "windows": list(windows.shape), "default_attention": model.config._attn_implementation}
    res["int0_identity"] = int0_identity(model, w[:4], device)
    res["sdpa_vs_eager"] = sdpa_vs_eager(model, w, args.layer, device)
    # our fp32 JumpReLU vs the installed sae_lens SAE that their script uses, on real layer-12 activations (eager)
    set_attention(model, "eager"); cap = ResidualCapture(model, args.layer); acts = []
    for i in range(k):
        acts.append(_forward(model, cap, w[i:i + 1].to(device))[1][0].float())
    cap.remove(); acts = torch.cat(acts)
    ours = sae.to(device).encode(acts).float()
    try:
        from sae_lens import SAE
        ref = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical", sae_id="layer_12/width_16k/canonical", device=device)
        ref = ref[0] if isinstance(ref, tuple) else ref
        with torch.no_grad():
            res["ours_fp32_vs_sae_lens"] = feature_agreement(ref.encode(acts).float(), ours)
    except Exception as e:  # noqa: BLE001
        res["ours_fp32_vs_sae_lens"] = {"error": f"{type(e).__name__}: {e}"}
    bf = copy.deepcopy(sae).to(device).to(torch.bfloat16)
    res["ours_bf16_vs_fp32"] = feature_agreement(ours, bf.encode(acts.to(torch.bfloat16)).float())
    (out / "selfcheck.json").write_text(json.dumps(res, indent=2))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="T,O"); ap.add_argument("--bits", default="8,7,6"); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-2-2b"); ap.add_argument("--tokens", type=int, default=lib.TOKEN_BUDGET)
    ap.add_argument("--layer", type=int, default=12); ap.add_argument("--limit-windows", type=int, default=0)
    ap.add_argument("--selfcheck", choices=["run", "only", "skip"], default="run"); ap.add_argument("--selfcheck-windows", type=int, default=16)
    ap.add_argument("--wandb-name", default="r5"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    cells, bits, out, device = args.cells.split(","), [int(b) for b in args.bits.split(",")], Path(args.out), args.device
    unknown = [c for c in cells if c not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cells {unknown}; choose from {list(CELLS)}")
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    windows = lib.build_windows(build_tokens(tok, args.tokens), args.tokens)
    if args.limit_windows:
        windows = windows[: args.limit_windows]
    torch.save(windows, out / "tokens_2d.pt")           # no BOS: R2's tokens_2d.pt[:, 1:] must equal this
    print(f"windows {tuple(windows.shape)} = {windows.numel():,} tokens; bos_id={tok.bos_token_id}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    sae = JumpReLUSAE.from_gemma_scope(layer=args.layer)
    print("default attention:", model.config._attn_implementation, "| dtype:", next(model.parameters()).dtype, flush=True)
    if args.selfcheck != "skip":
        res = selfchecks(model, sae, windows, args, device, out)
        print("selfcheck:", json.dumps(res, indent=1), flush=True)
        if not res["int0_identity"]["identical"]:
            raise SystemExit("INT0 identity failed: quantize->restore changed the FP model; results would be meaningless")
    if args.selfcheck == "only":
        return
    from g0_wandb import G0Logger
    for name in cells:
        cdir = out / name
        if cell_is_done(cdir):
            print(f"cell {name} already done; skipping", flush=True); continue
        logger = G0Logger.start(project=os.environ.get("WANDB_PROJECT", "anlp-project"), entity=os.environ.get("WANDB_ENTITY"), name=f"{args.wandb_name}-{name}",
                                config={"cell": name, **asdict(CELLS[name]), "bits": bits, "tokens": int(windows.numel()), "model": args.model},
                                out_dir=cdir, tags=["r5", "audit"])
        run_cell(name, CELLS[name], model, sae, windows, tok.bos_token_id, bits, cdir, args.layer, device, logger=logger)
        logger.finish(cdir)


if __name__ == "__main__":
    main()
