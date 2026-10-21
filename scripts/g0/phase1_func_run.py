"""Engine of the functional-validation study (D-039/D-040; design section 5, plan Task 4): per precision, run the lower blocks once per batch, cache the layer-12 residual and the SAE
activations, then re-run only the upper blocks for every edit configuration and read out per-sentence next-token NLL. The model, SAE and quantizer are injected callables
(`lower`, `upper`, `encode`, `decode`, `W_dec`) so that the engine is tested on a toy causal model; `main` (added below) wires the real Gemma-2-2B and Gemma Scope.

Items are paired by construction: every configuration of a language is evaluated on exactly the batches of that language's sentences, and results are stored by sentence position."""
import hashlib

import torch

import phase1_func_intervene as fi
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


def evaluate_precision(lower, upper, encode, W_dec, batches, n_sent, configs, decode=None, donors=None, nll_chunk=4, device=None):
    """Run every configuration over the batches of the languages it applies to (`c["langs"]`), plus the unintervened baseline `none` on every batch.
    batches: p1.make_batches output (single-language, right-padded, `valid` = content positions); n_sent: {language: number of sentences}; configs: dicts with name, kind, idx (feature group),
    langs (text languages evaluated), optional kw (alpha, pool and seed for resample, mean_vals); donors: {batch index: donor activations [N, F]} for kind `donor`.
    Returns {"items": {(lang, name): {"nll" float64 [n], "ntok" int64 [n]}}, "stats": {(lang, name): sums of the perturbation norm ratio, active group features, rows}, "fidelity": {lang: sums of
    reconstruction error, residual norm, L0, rows}} (fidelity only when `decode` is given)."""
    validate_configs(configs)
    items, stats, fid = {}, {}, {}

    def record(l, name, sent, s, n):
        slot = items.setdefault((l, name), {"nll": torch.zeros(n_sent[l], dtype=torch.float64), "ntok": torch.zeros(n_sent[l], dtype=torch.long)})
        slot["nll"][sent] = s.double().cpu(); slot["ntok"][sent] = n.long().cpu()

    with torch.no_grad():
        for bi, b in enumerate(batches):
            l = b["lang"]
            tokens, valid = b["tokens"], b["valid"]
            if device is not None:
                tokens, valid = tokens.to(device), valid.to(device)
            resid = lower(tokens)
            a = fi.encode_valid(resid, valid, encode)
            s, n = p1.chunked_nll(upper(resid), tokens, valid, chunk=nll_chunk)
            record(l, "none", b["sent"], s, n)
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
                if c["kind"] == "donor":
                    if donors is None:
                        raise ValueError("a donor configuration needs the donor activations")
                    kw["a_donor"] = donors[bi]
                if c["kind"] == "resample":
                    kw["generator"] = torch.Generator().manual_seed(_seed(kw.pop("seed", 0), l, bi))
                resid2, st = fi.apply_edit(resid, valid, a, W_dec, c["idx"], c["kind"], **kw)
                s2, n2 = p1.chunked_nll(upper(resid2), tokens, valid, chunk=nll_chunk)
                record(l, c["name"], b["sent"], s2, n2)
                acc = stats.setdefault((l, c["name"]), {"sum_norm_ratio": 0.0, "sum_active": 0.0, "rows": 0})
                for k in acc:
                    acc[k] += st[k]
    return {"items": items, "stats": stats, "fidelity": fid}


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
