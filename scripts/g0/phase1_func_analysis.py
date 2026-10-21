"""Analysis of the functional-validation item files (D-039/D-040; design section 6). CPU only, numpy.

Harm of an edit on a sentence = NLL(edit) - NLL(none) at the SAME precision; effects are pooled as a ratio of sums (nats per token). The functional effect E of a language is the harm of
zero-ablating its selective group beyond the mean harm of its activity-matched controls on the language's own text. Uncertainty: a paired cluster bootstrap over sentence positions (the
same FLORES sentence id is the same position in every language, so a draw resamples whole translations; the same draws serve every precision, so differences between precisions are paired).
Retention ratio = E at a precision over E in full precision, classified against fixed bands. The observed value is the estimate; intervals are descriptive percentile intervals."""
import csv
import json
import re
from pathlib import Path

import numpy as np

BANDS = (0.75, 1.33)


class Run:
    """The item-level results of one precision: items {"<code>|<config>": {"nll", "ntok"}}, stats, fidelity, languages, ids (the same ids in the same order for every language)."""
    def __init__(self, label, items, stats, fidelity, languages, ids):
        self.label, self.languages, self.stats, self.fidelity, self.ids = label, list(languages), stats, fidelity, ids
        self._items = {}
        for key, v in items.items():
            text, name = key.split("|", 1)
            self._items[(text, name)] = (np.asarray(v["nll"], dtype=np.float64), np.asarray(v["ntok"], dtype=np.int64))
        for (text, name), (nll, ntok) in self._items.items():
            if name != "none" and not np.array_equal(ntok, self._items[(text, "none")][1]):
                raise ValueError(f"{label}: configuration {name} on {text} is not paired with the baseline (token counts differ)")

    @classmethod
    def from_file(cls, path):
        import torch
        d = torch.load(path, weights_only=False)
        items = {k: {"nll": v["nll"].numpy(), "ntok": v["ntok"].numpy()} for k, v in d["items"].items()}
        return cls(d["precision"], items, d["stats"], d["fidelity"], d["languages"], d["ids"])

    def delta(self, text, name):
        """(per-sentence NLL change of `name` against the unintervened baseline on `text`, token counts)."""
        nll, ntok = self._items[(text, name)]
        return nll - self._items[(text, "none")][0], ntok

    def base(self, text):
        """(NLL per sentence of the unintervened run on `text`, token counts)."""
        return self._items[(text, "none")]

    def ntok(self, text):
        return self._items[(text, "none")][1]

    def configs_on(self, text):
        return sorted(n for (t, n) in self._items if t == text and n != "none")

    def has(self, text, name):
        return (text, name) in self._items


def pooled(delta, ntok, idx=None):
    """Sum of the sentence NLL changes over the sum of tokens (nats per token), optionally over resampled positions."""
    if idx is None:
        return float(np.sum(delta) / np.sum(ntok))
    return float(np.sum(delta[idx]) / np.sum(ntok[idx]))


def draws(n, B, seed):
    """B bootstrap draws of n sentence positions with replacement, seeded."""
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def _boot(delta, ntok, D):
    return delta[D].sum(1) / ntok[D].sum(1)


def functional_effect(run, code, K, D):
    """E for a language on its own text: harm of `zero_<K>@<code>` minus the mean harm of its controls `ctrl<r>_<K>@<code>`, with bootstrap samples over the draws D."""
    name = f"zero_{K}@{code}"
    ctrl = [n for n in run.configs_on(code) if re.fullmatch(rf"ctrl\d+_{re.escape(K)}@{re.escape(code)}", n)]
    if not run.has(code, name) or not ctrl:
        raise KeyError(f"{run.label}: {code} lacks {name} or its controls")
    dG, nt = run.delta(code, name)
    dC = np.mean([run.delta(code, c)[0] for c in ctrl], axis=0)
    diff = dG - dC
    st_g = run.stats.get(f"{code}|{name}")
    st_c = [run.stats[f"{code}|{c}"] for c in ctrl if f"{code}|{c}" in run.stats]
    return {"point": pooled(diff, nt), "samples": _boot(diff, nt, D), "harm_group": pooled(dG, nt), "harm_control": pooled(dC, nt), "n_controls": len(ctrl),
            "norm_ratio_group": st_g["sum_norm_ratio"] / st_g["rows"] if st_g else None,
            "norm_ratio_control": sum(s["sum_norm_ratio"] for s in st_c) / sum(s["rows"] for s in st_c) if st_c else None}


def macro(per_language, covered):
    """Unweighted mean over the covered languages of the effects (point and bootstrap samples)."""
    return {"point": float(np.mean([per_language[c]["point"] for c in covered])), "samples": np.mean([per_language[c]["samples"] for c in covered], axis=0), "languages": list(covered)}


def _ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def _classify(lo, hi, bands):
    return "preserved" if bands[0] <= lo and hi <= bands[1] else "attenuated" if hi < bands[0] else "amplified" if lo > bands[1] else "inconclusive"


def retention(eb, efp, bands=BANDS):
    """Retention ratio E(b) / E(FP) with its percentile interval and class: preserved (interval inside the bands), attenuated (entirely below), amplified (entirely above),
    inconclusive otherwise; `undefined` when the full-precision interval does not lie above zero."""
    if _ci(efp["samples"])[0] <= 0 or efp["point"] <= 0:
        return {"ratio": None, "lo": None, "hi": None, "class": "undefined"}
    lo, hi = _ci(eb["samples"] / efp["samples"])
    return {"ratio": float(eb["point"] / efp["point"]), "lo": lo, "hi": hi, "class": _classify(lo, hi, bands)}


def retention_macro(eb_by_lang, efp_by_lang, covered, bands=BANDS):
    """Macro retention = the MEAN OF THE PER-LANGUAGE RATIOS E_l(b) / E_l(FP) over the covered languages whose full-precision interval lies above zero (amendment D-041, written after the
    full-precision pilot showed effects from 0.005 to 7 nats per token: a ratio of mean effects would be the retention of the largest language). Each language counts once; the interval comes
    from the same bootstrap draws applied to every language."""
    langs = [c for c in covered if _ci(efp_by_lang[c]["samples"])[0] > 0 and efp_by_lang[c]["point"] > 0]
    if not langs:
        return {"ratio": None, "lo": None, "hi": None, "class": "undefined", "n_languages": 0, "languages": []}
    samples = np.mean([eb_by_lang[c]["samples"] / efp_by_lang[c]["samples"] for c in langs], axis=0)
    lo, hi = _ci(samples)
    return {"ratio": float(np.mean([eb_by_lang[c]["point"] / efp_by_lang[c]["point"] for c in langs])), "lo": lo, "hi": hi, "class": _classify(lo, hi, bands), "n_languages": len(langs), "languages": langs}


def family_table(run, D):
    """Harm (nats per token, with a bootstrap interval) of every configuration family on the language's own text: the zero ablations of each group size, the mean of the control draws of each size,
    sham, doses, mean and resample ablation; plus E_<size> = zero minus controls for each size. Rescue configurations are handled by rescue_effect."""
    rows = []

    def add(code, family, d, nt):
        pt, sm = pooled(d, nt), _boot(d, nt, D)
        lo, hi = _ci(sm)
        rows.append({"language": code, "family": family, "harm": pt, "lo": lo, "hi": hi})

    for code in run.languages:
        names = [n for n in run.configs_on(code) if n.endswith(f"@{code}") and not n.startswith("rescue")]
        ctrl = {}
        for n in names:
            fam = n[:-len(f"@{code}")]
            m = re.fullmatch(r"ctrl\d+_(.+)", fam)
            if m:
                ctrl.setdefault(m.group(1), []).append(n)
            else:
                d, nt = run.delta(code, n); add(code, fam, d, nt)
        for size, ns in sorted(ctrl.items()):
            d = np.mean([run.delta(code, n)[0] for n in ns], axis=0); nt = run.ntok(code)
            add(code, f"ctrl_{size}", d, nt)
            if run.has(code, f"zero_{size}@{code}"):
                dz, _ = run.delta(code, f"zero_{size}@{code}"); add(code, f"E_{size}", dz - d, nt)
    return rows


def sham_check(run, code, K, D):
    d, nt = run.delta(code, f"sham_{K}@{code}")
    return {"point": pooled(d, nt), "samples": _boot(d, nt, D)}


def off_target(run, code, K, D):
    """Raw harm (nats per token) of the group's zero-ablation on every other language text present: the manipulation check that the edit does little where the features are silent."""
    name, out = f"zero_{K}@{code}", {}
    for text in run.languages:
        if text != code and run.has(text, name):
            d, nt = run.delta(text, name)
            out[text] = {"point": pooled(d, nt), "samples": _boot(d, nt, D)}
    return out


RESCUE_PARTS = (("all", "rescue_all"), ("error", "rescue_error"), ("full", "rescue_full"))


def rescue_effect(run_b, run_fp, code, K, D, min_gap=0.02):
    """FP-donor patching on a language's own text: how much of the quantization-induced loss (NLL of the quantized run minus NLL of the full-precision run, per token) is recovered by patching
    the language's group (`group`), its activity-matched controls (`control`, mean over draws), the difference (`group_minus_control`), all SAE features (`all`), the SAE error term (`error`)
    and the whole residual (`full`). Recovered amounts are nats per token (raw quantities, always reported); a fraction (recovered over the gap) is given only when the 95 % lower bound of the gap
    exceeds min_gap nats per token (a quantized run that is not clearly worse has nothing to recover); negative recovery is kept."""
    nll_b, nt = run_b.base(code)
    gap_i = nll_b - run_fp.base(code)[0]
    gap = {"point": pooled(gap_i, nt), "samples": _boot(gap_i, nt, D)}
    arr = {}
    gname = f"rescue_{K}@{code}"
    if run_b.has(code, gname):
        arr["group"] = -run_b.delta(code, gname)[0]
    ctrl = [n for n in run_b.configs_on(code) if re.fullmatch(rf"rescuectrl\d+_{re.escape(K)}@{re.escape(code)}", n)]
    if ctrl:
        arr["control"] = np.mean([-run_b.delta(code, n)[0] for n in ctrl], axis=0)
    if "group" in arr and "control" in arr:
        arr["group_minus_control"] = arr["group"] - arr["control"]
    for key, name in RESCUE_PARTS:
        if run_b.has(code, name):
            arr[key] = -run_b.delta(code, name)[0]
    rec = {k: {"point": pooled(v, nt), "samples": _boot(v, nt, D)} for k, v in arr.items()}
    ok = _ci(gap["samples"])[0] > min_gap
    frac = {}
    for k, v in rec.items():
        if not ok:
            frac[k] = None
            continue
        lo, hi = _ci(v["samples"] / gap["samples"])
        frac[k] = {"point": v["point"] / gap["point"], "lo": lo, "hi": hi, "samples": v["samples"] / gap["samples"]}
    return {"gap": gap, "recovered": rec, "fraction": frac}


def fidelity_table(runs):
    """Per precision and language: fraction of residual energy not reconstructed by the SAE (sum of squared error over sum of squared norm), mean number of active features, rms norm."""
    rows = []
    for label, run in runs.items():
        for code in run.languages:
            f = run.fidelity.get(code)
            if f:
                rows.append({"precision": label, "language": code, "fvu": f["sum_err2"] / f["sum_x2"], "l0": f["sum_l0"] / f["rows"], "rms_norm": (f["sum_x2"] / f["rows"]) ** 0.5, "rows": f["rows"]})
    return rows


def _row(point, samples):
    lo, hi = _ci(samples)
    return {"point": float(point), "lo": lo, "hi": hi}


def analyse(runs, out_dir, K="K64", n_boot=2000, seed=0, covered=None, bands=BANDS):
    """Effects, macro effects, retention against full precision, sham, off-target and fidelity tables -> analysis.json, CSVs and report.md in out_dir."""
    if "FP" not in runs:
        raise ValueError("the full-precision run is needed as the reference")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fp = runs["FP"]
    n = len(next(iter(fp.ids.values())))
    D = draws(n, n_boot, seed)
    langs = [c for c in fp.languages if all(r.has(c, f"zero_{K}@{c}") for r in runs.values())]
    cov = [c for c in (covered or langs) if c in langs]
    eff = {lab: {c: functional_effect(r, c, K, D) for c in langs} for lab, r in runs.items()}
    mac = {lab: macro(eff[lab], cov) for lab in runs}
    res = {"K": K, "n_boot": n_boot, "seed": seed, "bands": list(bands), "covered": cov, "n_positions": n, "effects": {}, "macro": {}, "retention": {}, "sham": {}, "off_target": {}}
    for lab in runs:
        res["effects"][lab] = {c: {**_row(eff[lab][c]["point"], eff[lab][c]["samples"]), "harm_group": eff[lab][c]["harm_group"], "harm_control": eff[lab][c]["harm_control"],
                                   "norm_ratio_group": eff[lab][c]["norm_ratio_group"], "norm_ratio_control": eff[lab][c]["norm_ratio_control"]} for c in langs}
        res["macro"][lab] = {**_row(mac[lab]["point"], mac[lab]["samples"]), "median": float(np.median([eff[lab][c]["point"] for c in cov])), "n_positive": int(sum(eff[lab][c]["point"] > 0 for c in cov))}
        sh = {c: sham_check(runs[lab], c, K, D) for c in langs if runs[lab].has(c, f"sham_{K}@{c}")}
        res["sham"][lab] = {c: _row(s["point"], s["samples"]) for c, s in sh.items()}
        res["off_target"][lab] = {c: {t: _row(o["point"], o["samples"]) for t, o in off_target(runs[lab], c, K, D).items()} for c in langs}
        if lab != "FP":
            res["retention"][lab] = {"macro": retention_macro(eff[lab], eff["FP"], cov, bands), "pooled": retention(mac[lab], mac["FP"], bands),
                                     "languages": {c: retention(eff[lab][c], eff["FP"][c], bands) for c in langs}}
    res["rescue"] = {}
    for lab, r in runs.items():
        if lab == "FP":
            continue
        per = {c: rescue_effect(r, fp, c, K, D) for c in langs if r.has(c, f"rescue_{K}@{c}")}
        if per:
            def pack(x):
                return {"gap": _row(x["gap"]["point"], x["gap"]["samples"]), "recovered": {k: _row(v["point"], v["samples"]) for k, v in x["recovered"].items()},
                        "fraction": {k: (None if v is None else {"point": float(v["point"]), "lo": v["lo"], "hi": v["hi"]}) for k, v in x["fraction"].items()}}
            macro_frac = {}
            for k in ("group", "control", "group_minus_control", "all", "error", "full"):
                have = [c for c in per if per[c]["fraction"].get(k)]
                if have:
                    s = np.mean([per[c]["fraction"][k]["samples"] for c in have], axis=0)
                    macro_frac[k] = {**_row(np.mean([per[c]["fraction"][k]["point"] for c in have]), s), "n_languages": len(have)}
            res["rescue"][lab] = {"languages": {c: pack(x) for c, x in per.items()}, "macro_fraction": macro_frac}
    if res["rescue"]:
        with open(out / "rescue.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["precision", "language", "gap", "part", "recovered", "recovered_lo", "recovered_hi", "fraction", "fraction_lo", "fraction_hi"])
            for lab, d in res["rescue"].items():
                for c, x in d["languages"].items():
                    for part, v in x["recovered"].items():
                        fr_ = x["fraction"].get(part)
                        w.writerow([lab, c, x["gap"]["point"], part, v["point"], v["lo"], v["hi"], fr_["point"] if fr_ else "", fr_["lo"] if fr_ else "", fr_["hi"] if fr_ else ""])
    fam = {lab: family_table(r, D) for lab, r in runs.items()}
    res["families"] = {lab: {"median_by_family": {f: float(np.median([x["harm"] for x in rows if x["family"] == f])) for f in sorted({x["family"] for x in rows})},
                             "n_languages_positive_E": {f: int(sum(x["lo"] > 0 for x in rows if x["family"] == f)) for f in sorted({x["family"] for x in rows}) if f.startswith("E_")}} for lab, rows in fam.items()}
    with open(out / "families.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["precision", "language", "family", "harm", "lo", "hi"])
        for lab, rows in fam.items():
            for x in rows:
                w.writerow([lab, x["language"], x["family"], x["harm"], x["lo"], x["hi"]])
    fid = fidelity_table(runs)
    res["fidelity"] = fid
    (out / "analysis.json").write_text(json.dumps(res, indent=1) + "\n")
    with open(out / "effects_by_language.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["precision", "language", "effect", "lo", "hi", "harm_group", "harm_control", "norm_ratio_group", "norm_ratio_control"])
        for lab in runs:
            for c in langs:
                e = res["effects"][lab][c]; w.writerow([lab, c, e["point"], e["lo"], e["hi"], e["harm_group"], e["harm_control"], e["norm_ratio_group"], e["norm_ratio_control"]])
    with open(out / "retention.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["precision", "language", "ratio", "lo", "hi", "class"])
        for lab, r in res["retention"].items():
            for c, v in [("MACRO_MEAN_OF_RATIOS", r["macro"]), ("POOLED_RATIO", r["pooled"])] + list(r["languages"].items()):
                w.writerow([lab, c, v["ratio"], v["lo"], v["hi"], v["class"]])
    with open(out / "fidelity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["precision", "language", "fvu", "l0", "rms_norm", "rows"]); w.writeheader(); w.writerows(fid)
    (out / "report.md").write_text(report_text(res, runs))
    return res


def report_text(res, runs):
    K, B = res["K"], res["n_boot"]
    L = [f"# Functional effects of language feature groups (group size {K})", "",
         "**Exploratory on already examined data** (FLORES devtest sentences; the features were selected on dev). Next-token loss only; no claim about task accuracy or meaning. "
         "E is the harm (nats per token, NLL after the edit minus NLL before it, same precision, same sentences) of zero-ablating the language's selective group beyond the mean harm of its "
         f"activity-matched controls on the language's own text; positive = the selective group matters more than equally active non-selective features. Intervals: {B} paired cluster-bootstrap draws "
         "over sentence ids (percentile, descriptive; the observed value is the estimate). Macro retention is the mean over languages of the per-language ratio E at a precision over E in full precision (D-041; languages whose full-precision interval lies above zero), classified preserved / attenuated / amplified "
         f"against the bands [{res['bands'][0]}, {res['bands'][1]}], inconclusive otherwise, undefined when the full-precision interval does not lie above zero.", "",
         f"Languages in the macro average: {len(res['covered'])} ({', '.join(res['covered'])}). Positions (sentences per language): {res['n_positions']}.", "", "## Macro effect and retention", "",
         "| precision | mean E | 95 % interval | median E | languages with E > 0 | retention (mean of per-language ratios) | interval | class (languages) | pooled ratio (dominated by large effects) |", "|---|---|---|---|---|---|---|---|---|"]
    for lab in res["macro"]:
        m = res["macro"][lab]; r = res["retention"].get(lab, {}).get("macro"); pl = res["retention"].get(lab, {}).get("pooled")
        head = f"| {lab} | {m['point']:.4f} | [{m['lo']:.4f}, {m['hi']:.4f}] | {m['median']:.4f} | {m['n_positive']} of {len(res['covered'])} | "
        if r and r["ratio"] is not None:
            L.append(head + f"{r['ratio']:.2f} | [{r['lo']:.2f}, {r['hi']:.2f}] | {r['class']} ({r['n_languages']}) | " + (f"{pl['ratio']:.2f} |" if pl and pl["ratio"] is not None else "- |"))
        else:
            L.append(head + f"- | - | {r['class'] if r else 'reference'} | - |")
    if res.get("families"):
        L += ["", "## Dose, size and replacement (median over languages of the harm in nats per token on the language's own text)", ""]
        fams = sorted({f for d in res["families"].values() for f in d["median_by_family"]})
        L += ["| family | " + " | ".join(res["families"]) + " |", "|---|" + "---|" * len(res["families"])]
        for f in fams:
            L.append(f"| {f} | " + " | ".join(f"{d['median_by_family'].get(f, float('nan')):.4f}" for d in res["families"].values()) + " |")
        L += ["", "Languages whose E (zero ablation beyond controls) has a 95 % lower bound above 0: " + "; ".join(f"{lab}: " + ", ".join(f"{k} {v}" for k, v in d["n_languages_positive_E"].items()) for lab, d in res["families"].items()), ""]
    L += ["", "## Sham check (the edit path with no intended change; must be about 0 nats per token)", "", "| precision | language | harm | interval |", "|---|---|---|---|"]
    for lab, d in res["sham"].items():
        for c, s in d.items():
            L.append(f"| {lab} | {c} | {s['point']:.5f} | [{s['lo']:.5f}, {s['hi']:.5f}] |")
    L += ["", "## Effect by language", "", "| precision | language | E | interval | group harm | control harm | removed norm group / control |", "|---|---|---|---|---|---|---|"]
    for lab, d in res["effects"].items():
        for c, e in d.items():
            ng = f"{e['norm_ratio_group']:.3f} / {e['norm_ratio_control']:.3f}" if e["norm_ratio_group"] is not None and e["norm_ratio_control"] is not None else "-"
            L.append(f"| {lab} | {c} | {e['point']:.4f} | [{e['lo']:.4f}, {e['hi']:.4f}] | {e['harm_group']:.4f} | {e['harm_control']:.4f} | {ng} |")
    if res.get("rescue"):
        L += ["", "## Rescue: share of the quantization-induced loss recovered by patching full-precision activations (same tokens) into the quantized model", "",
              "Fractions are given only where the gap (quantized minus full-precision NLL per token) is clearly above 0.02 nats; raw recovered amounts are in `rescue.csv`. This is a different question from the effects above: "
              "it says where the damage sits at layer 12, not that the quantized model natively kept the function.", "",
              "| precision | part | macro fraction | 95 % interval | languages |", "|---|---|---|---|---|"]
        for lab, d in res["rescue"].items():
            for k, v in d["macro_fraction"].items():
                L.append(f"| {lab} | {k} | {v['point']:.3f} | [{v['lo']:.3f}, {v['hi']:.3f}] | {v['n_languages']} |")
    L += ["", "## Off-target harm of the group's zero-ablation on other languages' text (manipulation check)", "", "| precision | group | text | harm | interval |", "|---|---|---|---|---|"]
    for lab, d in res["off_target"].items():
        for c, tx in d.items():
            for t, o in tx.items():
                L.append(f"| {lab} | {c} | {t} | {o['point']:.5f} | [{o['lo']:.5f}, {o['hi']:.5f}] |")
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Analyse functional-validation item files (T-LM).")
    ap.add_argument("--runs", required=True, help="directory with func_lm_<split>_<precision>.pt files"); ap.add_argument("--split", default="pilot"); ap.add_argument("--out", required=True)
    ap.add_argument("--K", default="K64"); ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--covered", nargs="*")
    a = ap.parse_args(argv)
    runs = {}
    for p in sorted(Path(a.runs).glob(f"func_lm_{a.split}_*.pt")):
        r = Run.from_file(p); runs[r.label] = r
    analyse(runs, a.out, a.K, a.n_boot, a.seed, a.covered)
    print("wrote", a.out, sorted(runs))


if __name__ == "__main__":
    main()
