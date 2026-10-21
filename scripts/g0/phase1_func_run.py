"""Engine of the functional-validation study (D-039/D-040; design section 5, plan Task 4): per precision, run the lower blocks once per batch, cache the layer-12 residual and the SAE
activations, then re-run only the upper blocks for every edit configuration and read out per-sentence next-token NLL. The model, SAE and quantizer are injected callables
(`lower`, `upper`, `encode`, `decode`, `W_dec`) so that the engine is tested on a toy causal model; `main` (added below) wires the real Gemma-2-2B and Gemma Scope.

Items are paired by construction: every configuration of a language is evaluated on exactly the batches of that language's sentences, and results are stored by sentence position."""
import hashlib

import torch

import phase1_func_intervene as fi
import phase1_func_tasks as ft
import phase1_lib as p1


class GateError(RuntimeError):
    """A correctness gate of the edit path failed; the run must stop."""


def _seed(*parts):
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def validate_configs(configs):
    seen = set()
    for c in configs:
        for k in ("name", "kind", "idx", "langs"):
            if k not in c:
                raise ValueError(f"configuration lacks {k!r}: {c}")
        if c["name"] == "none" or c["name"] in seen:
            raise ValueError(f"configuration name {c['name']!r} is reserved or duplicated")
        if c["kind"] not in fi.KINDS:
            raise ValueError(f"unknown intervention {c['kind']!r}")
        seen.add(c["name"])


class DonorCache:
    """The full-precision model's layer-12 residual rows (bf16, host) and SAE activations (sparse, host) of every batch, for FP-donor patching of a quantized recipient run on the SAME batches."""
    def __init__(self):
        self._x, self._a = [], p1.SparseRows()

    def add(self, x_rows, a_rows):
        self._x.append(x_rows.detach().cpu()); self._a.add(a_rows.detach().cpu())

    def __len__(self):
        return len(self._x)

    def rows(self, bi):
        return self._x[bi].shape[0]

    def get_x(self, bi, device):
        return self._x[bi].to(device)

    def get_a(self, bi, device):
        return self._a.get(bi, device)


def build_donor_cache(lower, encode, batches, device=None):
    """Run the (full-precision) model over the batches once and cache residual rows and activations of the valid positions."""
    cache = DonorCache()
    with torch.no_grad():
        for b in batches:
            tokens, valid = (b["tokens"], b["valid"]) if device is None else (b["tokens"].to(device), b["valid"].to(device))
            x = lower(tokens)[valid]
            cache.add(x, encode(x).float())
    return cache


def _donor_kwargs(donors, bi, kind, valid, device):
    if donors is None:
        raise ValueError("a donor configuration needs the donor activations")
    if isinstance(donors, dict):                                                        # plain activations per batch (kind donor only)
        return {"a_donor": donors[bi]}
    if donors.rows(bi) != int(valid.sum()):
        raise ValueError(f"the donor cache does not match batch {bi} ({donors.rows(bi)} rows against {int(valid.sum())} valid positions)")
    kw = {}
    if kind in ("donor", "donor_error"):
        kw["a_donor"] = donors.get_a(bi, device)
    if kind in ("donor_error", "donor_resid"):
        kw["x_donor"] = donors.get_x(bi, device)
    return kw


def lm_readout(chunk=4):
    """Per-row next-token NLL (float64 sum over the batch's `score` positions, default the valid positions) and the number of scored tokens: {"nll", "ntok"}."""
    def read(logits, b, tokens, valid):
        s, n = p1.chunked_nll(logits, tokens, b["score"].to(valid.device) if "score" in b else valid, chunk=chunk)
        return {"nll": s.double(), "ntok": n.long()}
    return read


def letter_readout(logits, b, tokens, valid):
    """Log-probabilities (over the whole vocabulary) of the four option-letter tokens at the last prompt position of every row: {"cand_logp": [B, 4]}."""
    rows = torch.arange(logits.shape[0], device=logits.device)
    lp = torch.log_softmax(logits[rows, b["last"].to(logits.device)].float(), dim=-1)
    return {"cand_logp": lp.gather(1, b["cand"].to(logits.device)).double()}


def evaluate_precision(lower, upper, encode, W_dec, batches, n_sent, configs, decode=None, donors=None, nll_chunk=4, device=None, readout=None):
    """Run every configuration over the batches of the languages it applies to (`c["langs"]`), plus the unintervened baseline `none` on every batch.
    batches: p1.make_batches / make_mc_batches output (single-language, right-padded, `valid` = content positions); n_sent: {language: number of rows}; configs: dicts with name, kind, idx (feature
    group), langs (text languages evaluated), optional kw (alpha, pool and seed for resample, mean_vals); donors: {batch index: donor activations [N, F]} for kind `donor`; readout(logits, batch,
    tokens, valid) -> {key: tensor [B, ...]} (default: next-token NLL; letter_readout for multiple choice).
    Returns {"items": {(lang, name): {key: array [n_rows, ...]}}, "stats": {(lang, name): sums of the perturbation norm ratio, active group features, rows}, "fidelity": {lang: sums of
    reconstruction error, residual norm, L0, rows}} (fidelity only when `decode` is given)."""
    validate_configs(configs)
    if isinstance(donors, DonorCache) and len(donors) != len(batches):
        raise ValueError(f"the donor cache holds {len(donors)} batches, the run has {len(batches)}")
    readout = readout or lm_readout(nll_chunk)
    items, stats, fid = {}, {}, {}

    def record(l, name, sent, out):
        slot = items.setdefault((l, name), {})
        for key, v in out.items():
            v = v.cpu()
            if key not in slot:
                slot[key] = torch.zeros((n_sent[l],) + tuple(v.shape[1:]), dtype=torch.float64 if v.is_floating_point() else torch.long)
            slot[key][sent] = v.to(slot[key].dtype)

    with torch.no_grad():
        for bi, b in enumerate(batches):
            l = b["lang"]
            tokens, valid = b["tokens"], b["valid"]
            if device is not None:
                tokens, valid = tokens.to(device), valid.to(device)
            resid = lower(tokens)
            a = fi.encode_valid(resid, valid, encode)
            record(l, "none", b["sent"], readout(upper(resid), b, tokens, valid))
            if decode is not None:
                x = resid[valid].float()
                err = x - decode(a)
                f = fid.setdefault(l, {"sum_err2": 0.0, "sum_x2": 0.0, "sum_l0": 0.0, "rows": 0.0})
                f["sum_err2"] += float((err.double() ** 2).sum()); f["sum_x2"] += float((x.double() ** 2).sum())
                f["sum_l0"] += float((a > 0).sum()); f["rows"] += float(x.shape[0])
            for c in configs:
                if l not in c["langs"]:
                    continue
                kw = dict(c.get("kw", {}))
                if c["kind"] in ("donor", "donor_error", "donor_resid"):
                    kw.update(_donor_kwargs(donors, bi, c["kind"], valid, resid.device))
                if c["kind"] == "resample":
                    kw["generator"] = torch.Generator().manual_seed(_seed(kw.pop("seed", 0), l, bi))
                resid2, st = fi.apply_edit(resid, valid, a, W_dec, c["idx"], c["kind"], **kw)
                record(l, c["name"], b["sent"], readout(upper(resid2), b, tokens, valid))
                acc = stats.setdefault((l, c["name"]), {"sum_norm_ratio": 0.0, "sum_active": 0.0, "rows": 0})
                for k in acc:
                    acc[k] += st[k]
    return {"items": items, "stats": stats, "fidelity": fid}


def mc_n_rows(data, fmt):
    """Rows per language of the multiple-choice batches: one per item (letter) or four per item (cloze: one row per option)."""
    return {l: len(v) * (1 if fmt == "letter" else 4) for l, v in data.items()}


def make_mc_batches(data, fmt, bos_id, pad_id=0, max_tokens=1024, batch_size=32):
    """Batches of multiple-choice rows from {language: [encode_mc output]}: single-language, length-sorted, right-padded after one BOS; `valid` = content positions (never BOS or padding, the
    positions that are edited). letter: one row per item with `last` (index of the last prompt token) and `cand` (the four option-letter ids); `sent` = item index. cloze: four rows per item
    (context plus option j, row id 4 * item + j) with `score` marking the option's token positions. A row longer than max_tokens gets a batch of its own (the loader drops overlong prompts)."""
    if fmt not in ("letter", "cloze"):
        raise ValueError(f"unknown format {fmt!r}")
    batches = []
    for lang in sorted(data):
        rows = []
        for i, e in enumerate(data[lang]):
            if fmt == "letter":
                rows.append((i, list(e["tokens"]), None, e["cand_ids"]))
            else:
                rows += [(4 * i + j, list(e["tokens"]) + list(o), len(e["tokens"]), None) for j, o in enumerate(e["option_ids"])]
        order = sorted(range(len(rows)), key=lambda k: (len(rows[k][1]), rows[k][0]))
        groups, cur = [], []
        for k in order:
            L = len(rows[k][1]) + 1
            if cur and (len(cur) + 1 > batch_size or (len(cur) + 1) * L > max_tokens):
                groups.append(cur); cur = []
            cur.append(k)
        if cur:
            groups.append(cur)
        for g in groups:
            L = max(len(rows[k][1]) for k in g) + 1
            tokens = torch.full((len(g), L), pad_id, dtype=torch.long)
            valid = torch.zeros((len(g), L), dtype=torch.bool); score = torch.zeros((len(g), L), dtype=torch.bool)
            last, cand = [], []
            for r, k in enumerate(g):
                rid, toks, ctx_len, cands = rows[k]
                tokens[r, 0] = bos_id; tokens[r, 1:1 + len(toks)] = torch.tensor(toks, dtype=torch.long); valid[r, 1:1 + len(toks)] = True
                if fmt == "letter":
                    last.append(len(toks)); cand.append(list(cands))
                else:
                    score[r, 1 + ctx_len:1 + len(toks)] = True
            b = {"lang": lang, "tokens": tokens, "valid": valid, "sent": torch.tensor([rows[k][0] for k in g], dtype=torch.long)}
            if fmt == "letter":
                b["last"] = torch.tensor(last, dtype=torch.long); b["cand"] = torch.tensor(cand, dtype=torch.long)
            else:
                b["score"] = score
            batches.append(b)
    return batches


def run_gates(full, lower, upper, encode, W_dec, tokens, valid, idx):
    """Correctness gates on a probe batch (unedited input): (1) the cached lower half followed by the upper half equals the full forward bit for bit; (2) identity edits (scale 1, a donor equal
    to the recipient, sham) leave the logits unchanged (bitwise for 16-bit residuals, to 1e-4 relative for float32, where the sham round trip has round-off). Raises GateError naming the failures."""
    with torch.no_grad():
        ref = full(tokens)
        resid = lower(tokens)
        res = {"split_equals_full": torch.equal(ref, upper(resid))}
        a = fi.encode_valid(resid, valid, encode)
        exact = resid.dtype in (torch.bfloat16, torch.float16)
        for name, kind, kw in (("scale_one_identity", "scale", {"alpha": 1.0}), ("donor_self_identity", "donor", {"a_donor": a}), ("sham_identity", "sham", {})):
            r2, _ = fi.apply_edit(resid, valid, a, W_dec, idx, kind, **kw)
            out = upper(r2)
            if kind == "sham" and not exact:
                res[name] = bool(torch.allclose(out, ref, rtol=1e-4, atol=1e-5))
            else:
                res[name] = torch.equal(out, ref)
    bad = [k for k, v in res.items() if not v]
    if bad:
        raise GateError("gate(s) failed: " + ", ".join(bad))
    return res


FAMILIES = ("zero", "controls", "n_controls", "sham", "scale", "mean", "resample", "rescue", "rescue_bounds")


def _size(label):
    return None if label == "all" else int(label[1:])


def build_configs(man, run_langs, targets, panel, families, off_target_label="K64", mean_vals_fn=None, pool_fn=None, n_features=None):
    """Edit configurations from the frozen groups manifest. run_langs: language codes of the run in batch-index order; targets: codes to build for; panel: codes whose text is used for the
    off-target check (limited to run_langs); families: {"zero": [labels], "controls": [labels] (every draw of the manifest), "sham": [labels], "scale": {label: [alphas]}, "mean": [labels],
    "resample": [labels], "n_controls": {label: number of control draws to use (cap; default all draws of the manifest)}, "rescue": [labels] (FP-donor patch of the group and of every control draw), "rescue_bounds": [True] (donor patches of all features, of the SAE error term and of the whole
    residual, on every language of the run; needs n_features)} with labels as in the manifest (K16, K64, all). Names are `<family>[draw]_<label>@<target>`. Only the zero-ablation of `off_target_label` also runs on the panel text
    (target excluded); every other configuration runs on its target language's text only. mean_vals_fn(code, idx) and pool_fn(code, idx) supply the mean-ablation values and the resample pool."""
    bad = [f for f in families if f not in FAMILIES]
    if bad:
        raise ValueError(f"unknown configuration families {bad}; choose from {FAMILIES}")
    index = {c: i for i, c in enumerate(run_langs)}
    missing = [t for t in targets if t not in index or t not in man["groups"]]
    if missing:
        raise ValueError(f"targets {missing} are not in the run or in the manifest")
    panel_idx = {index[c] for c in panel if c in index}
    cfgs = []

    def ids(code, label):
        if label not in man["sizes"][code]:
            raise ValueError(f"unknown group size {label!r} for {code}")
        return torch.tensor(man["groups"][code]["all"][:man["sizes"][code][label]], dtype=torch.long)

    caps = families.get("n_controls", {})
    if any(v < 1 for v in caps.values()):
        raise ValueError("n_controls caps must be at least 1")
    draws_of = lambda code, label: man["controls"][code][label][:caps.get(label)]

    for code in targets:
        i = index[code]
        for label in families.get("zero", []):
            langs = {i} | ({j for j in panel_idx if j != i} if label == off_target_label else set())
            cfgs.append({"name": f"zero_{label}@{code}", "kind": "ablate", "idx": ids(code, label), "langs": langs})
        for label in families.get("controls", []):
            if label not in man["controls"][code]:
                raise ValueError(f"no control draws of size {label!r} for {code}")
            for r, s in enumerate(draws_of(code, label)):
                cfgs.append({"name": f"ctrl{r}_{label}@{code}", "kind": "ablate", "idx": torch.tensor(s, dtype=torch.long), "langs": {i}})
        for label in families.get("sham", []):
            cfgs.append({"name": f"sham_{label}@{code}", "kind": "sham", "idx": ids(code, label), "langs": {i}})
        for label, alphas in families.get("scale", {}).items():
            for alpha in alphas:
                cfgs.append({"name": f"scale{alpha:g}_{label}@{code}", "kind": "scale", "idx": ids(code, label), "langs": {i}, "kw": {"alpha": float(alpha)}})
        for label in families.get("mean", []):
            if mean_vals_fn is None:
                raise ValueError("the mean family needs mean_vals_fn")
            idx = ids(code, label)
            cfgs.append({"name": f"mean_{label}@{code}", "kind": "mean", "idx": idx, "langs": {i}, "kw": {"mean_vals": mean_vals_fn(code, idx)}})
        for label in families.get("rescue", []):
            cfgs.append({"name": f"rescue_{label}@{code}", "kind": "donor", "idx": ids(code, label), "langs": {i}})
            for r, s in enumerate(draws_of(code, label)):
                cfgs.append({"name": f"rescuectrl{r}_{label}@{code}", "kind": "donor", "idx": torch.tensor(s, dtype=torch.long), "langs": {i}})
        for label in families.get("resample", []):
            if pool_fn is None:
                raise ValueError("the resample family needs pool_fn")
            idx = ids(code, label)
            cfgs.append({"name": f"res_{label}@{code}", "kind": "resample", "idx": idx, "langs": {i}, "kw": {"pool": pool_fn(code, idx), "seed": 0}})
    if families.get("rescue_bounds"):
        if n_features is None:
            raise ValueError("rescue_bounds needs n_features")
        every, langs = torch.arange(n_features, dtype=torch.long), set(range(len(run_langs)))
        for name, kind in (("rescue_all", "donor"), ("rescue_error", "donor_error"), ("rescue_full", "donor_resid")):
            cfgs.append({"name": name, "kind": kind, "idx": every, "langs": set(langs)})
    return cfgs


def other_language_means(sel, l, idx):
    """Dev mean activation (zeros included) of the features `idx`, averaged over all languages except l: the fixed replacement value of the mean ablation."""
    mu = sel["mean_activation"].double()
    return ((mu.sum(0) - mu[l]) / (mu.shape[0] - 1))[idx].float()


def collect_pool(lower, encode, batches, rows_per_lang, seed, device=None):
    """SAE activations (float16, host) of `rows_per_lang` random content positions of every language of `batches`, for resample ablation: ([P, F], language of each row [P]).
    The positions are drawn per language from a generator seeded by (seed, language) before the model runs; a language with fewer positions contributes all of them."""
    counts = {}
    for b in batches:
        counts[b["lang"]] = counts.get(b["lang"], 0) + int(b["valid"].sum())
    chosen = {}
    for l, n in counts.items():
        g = torch.Generator().manual_seed(_seed(seed, l))
        chosen[l] = set(torch.randperm(n, generator=g)[:min(rows_per_lang, n)].tolist())
    seen, rows, langs = {l: 0 for l in counts}, [], []
    with torch.no_grad():
        for b in batches:
            l = b["lang"]
            tokens, valid = (b["tokens"], b["valid"]) if device is None else (b["tokens"].to(device), b["valid"].to(device))
            n = int(valid.sum())
            take = [k - seen[l] for k in range(seen[l], seen[l] + n) if k in chosen[l]]
            seen[l] += n
            if not take:
                continue
            a = fi.encode_valid(lower(tokens), valid, encode)
            rows.append(a[torch.tensor(take, dtype=torch.long, device=a.device)].half().cpu()); langs += [l] * len(take)
    order = sorted(range(len(langs)), key=lambda i: langs[i])                       # stable: language ascending, stream order inside a language
    pool = torch.cat(rows)[torch.tensor(order, dtype=torch.long)] if rows else torch.zeros(0)
    return pool, torch.tensor([langs[i] for i in order], dtype=torch.long)


# --------------------------------------------------------------------------------------------- real-model wiring (exercised by the smoke job, not by unit tests)
PANEL = ("eng_Latn", "rus_Cyrl", "zho_Hans", "hin_Deva")
PRESETS = {
    "none": {},
    "smoke": {"zero": ["K64"], "controls": ["K64"], "sham": ["K64"], "scale": {"K64": [0.5, 2.0]}, "mean": ["K64"], "resample": ["K64"]},
    "primary": {"zero": ["K16", "K64", "all"], "controls": ["K16", "K64", "all"], "sham": ["K64"], "scale": {"K64": [0.5, 2.0]}, "mean": ["K64"], "resample": ["K64"]},
    # evaluation presets (reduction order of design section 8, applied to the pilot's 21 configurations per language: fewer control draws, no dose and mean-ablation families)
    "eval": {"zero": ["K16", "K64", "all"], "controls": ["K16", "K64", "all"], "n_controls": {"K16": 3, "K64": 3, "all": 2}, "sham": ["K64"], "resample": ["K64"]},
    "floor": {"zero": ["K64"], "controls": ["K64"], "n_controls": {"K64": 3}, "sham": ["K64"]},
}
LAYER = 12


def _precision_key(s):
    s = str(s)
    return "FP" if s.upper() == "FP" else int(s)


def main(argv=None):
    import argparse, gc, json, time
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Functional validation, task T-LM: feature-group interventions at the layer-12 residual of Gemma-2-2B in FP and RTN-quantized precisions (D-040).")
    ap.add_argument("--out", required=True); ap.add_argument("--precisions", nargs="*", default=["FP", "8", "7", "6", "5", "4", "16", "12", "0"])
    ap.add_argument("--flores-root", required=True); ap.add_argument("--roster", default=str(ROOT / "reports/g0/phase1_languages.json"))
    ap.add_argument("--manifest", default=str(ROOT / "reports/functional_validation/groups_manifest.json")); ap.add_argument("--selectivity", required=True)
    ap.add_argument("--split", choices=["pilot", "eval"], default="pilot"); ap.add_argument("--n-pilot", type=int, default=150); ap.add_argument("--n-eval", type=int, default=450)
    ap.add_argument("--lm-seed", type=int, default=0); ap.add_argument("--languages", nargs="*"); ap.add_argument("--max-sentences", type=int)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="smoke"); ap.add_argument("--off-target-label", default="K64"); ap.add_argument("--panel", nargs="*", default=list(PANEL), help="languages whose text is used for the off-target check")
    ap.add_argument("--rescue", help="group size label (K16, K64, all): add FP-donor patch configurations of that group, its controls and the bounds (all features, error term, whole residual) for every quantized precision")
    ap.add_argument("--max-tokens", type=int, default=1024); ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--pool-sentences", type=int, default=120); ap.add_argument("--pool-rows", type=int, default=300); ap.add_argument("--commit", default="unknown")
    a = ap.parse_args(argv)
    t0 = time.time(); log = lambda *m: print(f"[{time.time() - t0:7.0f}s]", *m, flush=True)

    import numpy as np
    from tqdm.auto import tqdm
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    import their_code as tc
    import phase1_data as pd_
    import phase1_func_groups as fg
    torch.set_grad_enabled(False)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    env = {"torch": torch, "np": np, "tqdm": tqdm, "gc": gc, "HookedTransformer": HookedTransformer, "SAE": SAE, "DEVICE": DEVICE, "WORK_DEVICE": DEVICE, "BATCH_SIZE": 1, "FIRING_THRESHOLD": 0.001}
    ex = tc.extract_functions(tc.PRISTINE, tc.THEIR_FUNCTIONS); tc.verify_hashes(ex, json.loads(tc.MANIFEST.read_text())["functions"]); tc.build_namespace(ex, env)
    log("their functions verified against the frozen manifest")
    import phase1_run as pr                                                  # SAE release/id, model candidates and package_versions as in the Phase 1 runs
    model, model_name = env["load_gemma_tl_model"](pr.MODEL_CANDIDATES)
    sae = env["load_sae_compat"](pr.SAE_RELEASE, pr.SAE_ID, DEVICE)
    hook = getattr(sae.cfg, "hook_name", None) or f"blocks.{LAYER}.hook_resid_post"
    assert hook == f"blocks.{LAYER}.hook_resid_post", f"unexpected SAE hook {hook}"
    assert sae.cfg.d_in == model.cfg.d_model
    env["model_work"] = model; env["original_state_cpu"] = env["clone_state_to_cpu"](model)
    tok = model.tokenizer; bos = tok.bos_token_id; pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    lower = lambda tokens: model(tokens, stop_at_layer=LAYER + 1)
    upper = lambda resid: model(resid, start_at_layer=LAYER + 1)
    full = lambda tokens: model(tokens)
    encode = lambda x: env["encode_sae"](sae, x)
    decode = lambda act: sae.decode(act)
    W_dec = sae.W_dec
    log("model", model_name, "hook", hook, "d_sae", int(sae.cfg.d_sae), "bos", bos, "pad", pad)

    roster = pd_.load_roster(a.roster)
    names = [r["code"] for r in roster]
    run_names = [c for c in names if not a.languages or c in set(a.languages)]
    man = json.loads(Path(a.manifest).read_text())
    assert man["languages"] == names, "the groups manifest was built for another language order"
    sel = fg.dev_view(torch.load(a.selectivity, weights_only=False)); assert list(sel["languages"]) == names
    ids = ft.lm_ids(1012, a.n_pilot, a.n_eval, a.lm_seed)[a.split]
    if a.max_sentences:
        ids = ids[:a.max_sentences]
    data, lang_ids = {}, {}
    for i, code in enumerate(run_names):
        lines = pd_.read_flores_split(a.flores_root, code, "devtest", 1012)
        data[i] = pd_.encode_sentences(tok, [lines[j] for j in ids]); lang_ids[code] = list(ids)
    batches = p1.make_batches(data, bos, "mask", pad, 256, a.batch_size, a.max_tokens)
    n_sent = {i: len(v) for i, v in data.items()}
    log("split", a.split, "sentences per language", len(ids), "languages", len(run_names), "batches", len(batches))

    probe = batches[0]
    gates = run_gates(full, lower, upper, encode, W_dec, probe["tokens"].to(DEVICE), probe["valid"].to(DEVICE), torch.tensor(man["groups"][run_names[0]]["all"][:64]))
    log("gates passed:", gates)

    families = PRESETS[a.preset]
    pool_batches = None
    if "resample" in families:
        dev = {}
        for i, code in enumerate(run_names):
            lines = pd_.read_flores_split(a.flores_root, code, "dev", 997)
            dev[i] = pd_.encode_sentences(tok, lines[:a.pool_sentences])
        pool_batches = p1.make_batches(dev, bos, "mask", pad, 256, a.batch_size, a.max_tokens)
    restore = env["restore_model_work"]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    donors = None
    if a.rescue:
        restore()                                                           # full-precision weights: the donor of every quantized run
        donors = build_donor_cache(lower, encode, batches, device=DEVICE)
        log("donor cache built:", len(donors), "batches")
    summary = {}
    for prec in [_precision_key(x) for x in a.precisions]:
        restore()
        if prec not in ("FP", 0):
            env["quantize_rtn_per_output_channel_tl"](model, bits=prec)
        pool_fn = None
        if pool_batches is not None:
            pool, pool_lang = collect_pool(lower, encode, pool_batches, a.pool_rows, 0, device=DEVICE)
            pool_fn = lambda code, idx: pool[pool_lang != run_names.index(code)][:, idx].float().to(DEVICE)
        mean_fn = lambda code, idx: other_language_means(sel, names.index(code), idx).to(DEVICE)
        fams = dict(families)
        if a.rescue and prec != "FP":
            fams["rescue"] = [a.rescue]; fams["rescue_bounds"] = [True]
        configs = build_configs(man, run_names, run_names, list(a.panel), fams, a.off_target_label, mean_vals_fn=mean_fn, pool_fn=pool_fn, n_features=int(sae.cfg.d_sae))
        passes = sum((1 + sum(1 for c in configs if b["lang"] in c["langs"])) * int(b["valid"].sum()) for b in batches)
        ts = time.time()
        out = evaluate_precision(lower, upper, encode, W_dec, batches, n_sent, configs, decode=decode, donors=donors, device=DEVICE)
        dt = time.time() - ts
        key = f"INT{prec}" if prec != "FP" else "FP"
        torch.save({"precision": key, "split": a.split, "ids": lang_ids, "languages": run_names, "preset": a.preset,
                    "items": {f"{run_names[l]}|{n}": v for (l, n), v in out["items"].items()},
                    "stats": {f"{run_names[l]}|{n}": v for (l, n), v in out["stats"].items()},
                    "fidelity": {run_names[l]: v for l, v in out["fidelity"].items()}}, Path(a.out) / f"func_lm_{a.split}_{key}.pt")
        summary[key] = {"seconds": dt, "token_passes": passes, "tokens_per_second": passes / dt, "configs": len(configs)}
        log(key, "done:", summary[key], "peak GPU GB", torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0)
        del out; gc.collect()
    restore()
    meta = {"task": "lm", "split": a.split, "preset": a.preset, "precisions": a.precisions, "languages": run_names, "n_sentences": len(ids), "lm_seed": a.lm_seed, "commit": a.commit,
            "model": model_name, "hook": hook, "gates": gates, "summary": summary, "manifest_source_sha256": man["source"]["sha256"], "argv": list(argv) if argv is not None else None,
            "versions": p1_package_versions(), "seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0}
    (Path(a.out) / f"meta_lm_{a.split}.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    log("saved", a.out)


MC_PRESETS = {"none": {}, "mc": {"zero": ["K64"], "controls": ["K64"], "sham": ["K64"], "mean": ["K64"], "resample": ["K64"]}}


def mc_items_for(items, origin, split, n_items, seed):
    """The items of a language for the calibration (dev-origin passages) or evaluation (devtest-origin) split, in an order that depends only on (seed, passage key, question number), so
    that every language takes the same subset; at most n_items of them."""
    sel = ft.split_items(items, origin)["cal" if split == "cal" else "eval"]
    order = sorted(sel, key=lambda it: (hashlib.sha256(f"{seed}:{it['passage_key']}:{it['question_number']}".encode()).hexdigest(), it["passage_key"], it["question_number"]))
    return order[:n_items] if n_items else order


def main_mc(argv=None):
    """Task T-MC: Belebele multiple choice (design section 3). Same model, SAE, quantizer and edit path as `main`; items split by the FLORES origin of the passage (calibration = dev, evaluation =
    devtest); letter and/or cloze scoring; optional edit configurations (preset `mc`). Setup lines mirror `main` on purpose (that function was already exercised on the cluster)."""
    import argparse, gc, json, time
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Functional validation, task T-MC (Belebele) in FP and RTN-quantized precisions (D-040).")
    ap.add_argument("--out", required=True); ap.add_argument("--precisions", nargs="*", default=["FP"])
    ap.add_argument("--belebele-root", required=True); ap.add_argument("--origin-file", required=True); ap.add_argument("--roster", default=str(ROOT / "reports/g0/phase1_languages.json"))
    ap.add_argument("--manifest", default=str(ROOT / "reports/functional_validation/groups_manifest.json")); ap.add_argument("--selectivity", required=True); ap.add_argument("--flores-root", required=True)
    ap.add_argument("--split", choices=["cal", "eval"], default="cal"); ap.add_argument("--formats", nargs="*", default=["letter"], choices=["letter", "cloze"])
    ap.add_argument("--n-items", type=int, default=0, help="cap per language (0 = all)"); ap.add_argument("--n-items-cloze", type=int, default=0, help="cap per language for the cloze format (0 = same as --n-items)")
    ap.add_argument("--item-seed", type=int, default=0); ap.add_argument("--languages", nargs="*"); ap.add_argument("--max-len", type=int, default=1024); ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--preset", choices=sorted(MC_PRESETS), default="none")
    ap.add_argument("--pool-sentences", type=int, default=120); ap.add_argument("--pool-rows", type=int, default=300); ap.add_argument("--commit", default="unknown")
    a = ap.parse_args(argv)
    t0 = time.time(); log = lambda *m: print(f"[{time.time() - t0:7.0f}s]", *m, flush=True)

    import numpy as np
    from tqdm.auto import tqdm
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    import their_code as tc
    import phase1_data as pd_
    import phase1_func_groups as fg
    torch.set_grad_enabled(False)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    env = {"torch": torch, "np": np, "tqdm": tqdm, "gc": gc, "HookedTransformer": HookedTransformer, "SAE": SAE, "DEVICE": DEVICE, "WORK_DEVICE": DEVICE, "BATCH_SIZE": 1, "FIRING_THRESHOLD": 0.001}
    ex = tc.extract_functions(tc.PRISTINE, tc.THEIR_FUNCTIONS); tc.verify_hashes(ex, json.loads(tc.MANIFEST.read_text())["functions"]); tc.build_namespace(ex, env)
    import phase1_run as pr
    model, model_name = env["load_gemma_tl_model"](pr.MODEL_CANDIDATES)
    sae = env["load_sae_compat"](pr.SAE_RELEASE, pr.SAE_ID, DEVICE)
    assert getattr(sae.cfg, "hook_name", None) in (None, f"blocks.{LAYER}.hook_resid_post") and sae.cfg.d_in == model.cfg.d_model
    env["model_work"] = model; env["original_state_cpu"] = env["clone_state_to_cpu"](model)
    tok = model.tokenizer; bos = tok.bos_token_id; pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    lower = lambda tokens: model(tokens, stop_at_layer=LAYER + 1)
    upper = lambda resid: model(resid, start_at_layer=LAYER + 1)
    full = lambda tokens: model(tokens)
    encode = lambda x: env["encode_sae"](sae, x)
    W_dec = sae.W_dec
    enc = lambda s: tok.encode(s, add_special_tokens=False)

    roster = pd_.load_roster(a.roster)
    names = [r["code"] for r in roster]
    run_names = [c for c in names if not a.languages or c in set(a.languages)]
    origin = json.loads(Path(a.origin_file).read_text())["origin"]
    per_lang = {}
    for code in run_names:
        items = ft.load_belebele(a.belebele_root, code, expect=900)
        per_lang[code] = mc_items_for(items, origin, a.split, a.n_items, a.item_seed)
    log("items per language:", {c: len(v) for c, v in per_lang.items()})
    man = sel = None
    families = MC_PRESETS[a.preset]
    if families:
        man = json.loads(Path(a.manifest).read_text()); assert man["languages"] == names
        sel = fg.dev_view(torch.load(a.selectivity, weights_only=False))
    restore = env["restore_model_work"]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    summary, gates = {}, None
    for fmt in a.formats:
        cap = (a.n_items_cloze or a.n_items) if fmt == "cloze" else a.n_items
        data, kept, dropped = {}, {}, {}
        for i, code in enumerate(run_names):
            encs, nd = ft.build_mc_items(enc, per_lang[code][:cap] if cap else per_lang[code], fmt, a.max_len)
            data[i] = encs; dropped[code] = nd
            kept[code] = [(e["passage_key"], e["question_number"]) for e in encs]
        batches = make_mc_batches(data, fmt, bos, pad, a.max_tokens, a.batch_size)
        n_rows = mc_n_rows(data, fmt)
        log(fmt, "batches", len(batches), "kept", {c: len(kept[c]) for c in run_names}, "dropped (too long)", dropped)
        if gates is None:
            probe = batches[0]
            idx0 = torch.tensor((man["groups"][run_names[0]]["all"] if man else list(range(8)))[:64])
            gates = run_gates(full, lower, upper, encode, W_dec, probe["tokens"].to(DEVICE), probe["valid"].to(DEVICE), idx0)
            log("gates passed:", gates)
        pool_batches = None
        if "resample" in families:
            dev = {}
            for i, code in enumerate(run_names):
                dev[i] = pd_.encode_sentences(tok, pd_.read_flores_split(a.flores_root, code, "dev", 997)[:a.pool_sentences])
            pool_batches = p1.make_batches(dev, bos, "mask", pad, 256, 32, a.max_tokens)
        for prec in [_precision_key(x) for x in a.precisions]:
            restore()
            if prec not in ("FP", 0):
                env["quantize_rtn_per_output_channel_tl"](model, bits=prec)
            configs = []
            if families:
                pool_fn = None
                if pool_batches is not None:
                    pool, pool_lang = collect_pool(lower, encode, pool_batches, a.pool_rows, 0, device=DEVICE)
                    pool_fn = lambda code, idx: pool[pool_lang != run_names.index(code)][:, idx].float().to(DEVICE)
                configs = build_configs(man, run_names, run_names, [], families, "NONE", mean_vals_fn=lambda code, idx: other_language_means(sel, names.index(code), idx).to(DEVICE), pool_fn=pool_fn)
            ts = time.time()
            out = evaluate_precision(lower, upper, encode, W_dec, batches, n_rows, configs, device=DEVICE, readout=letter_readout if fmt == "letter" else None)
            key = f"INT{prec}" if prec != "FP" else "FP"
            torch.save({"precision": key, "split": a.split, "format": fmt, "languages": run_names, "preset": a.preset, "keys": kept, "dropped": dropped,
                        "answers": {run_names[l]: [e["answer"] for e in data[l]] for l in data},
                        "items": {f"{run_names[l]}|{n}": v for (l, n), v in out["items"].items()}, "stats": {f"{run_names[l]}|{n}": v for (l, n), v in out["stats"].items()}},
                       Path(a.out) / f"func_mc_{a.split}_{fmt}_{key}.pt")
            summary[f"{fmt}/{key}"] = {"seconds": time.time() - ts, "configs": len(configs)}
            log(fmt, key, "done:", summary[f"{fmt}/{key}"], "peak GPU GB", torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0)
            del out; gc.collect()
    restore()
    (Path(a.out) / f"meta_mc_{a.split}.json").write_text(json.dumps({"task": "mc", "split": a.split, "formats": a.formats, "preset": a.preset, "precisions": a.precisions, "languages": run_names,
        "n_items": a.n_items, "n_items_cloze": a.n_items_cloze, "item_seed": a.item_seed, "max_len": a.max_len, "commit": a.commit, "model": model_name, "gates": gates, "summary": summary,
        "argv": list(argv) if argv is not None else None, "versions": p1_package_versions(), "seconds": time.time() - t0}, indent=1, default=str) + "\n")
    log("saved", a.out)


def p1_package_versions():
    import phase1_run as pr
    return pr.package_versions(["torch", "transformer-lens", "sae-lens", "transformers", "datasets"])


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    if "--task" in argv:                                # `--task lm` (default) or `--task mc`
        i = argv.index("--task"); task = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    else:
        task = "lm"
    {"lm": main, "mc": main_mc}[task](argv)
