"""Analysis of the Belebele (T-MC) item files of the functional-validation study (D-039/D-040; design sections 3 and 6; plan Task 6b). CPU only, numpy.

Item score: four option scores (letter format: log-probabilities of the four option letters at the final position; cloze format: length-normalised log-likelihood of the four option texts);
log p_correct = log_softmax(scores)[answer], the four-way probability of the correct option. Harm of an edit on an item = -(log p_correct(edit) - log p_correct(none)) at the SAME precision on the
SAME item (negative = improvement, never clipped). E of a language = mean harm of zero-ablating its selective group beyond the mean harm of its activity-matched controls. Uncertainty: a paired
cluster bootstrap with the PASSAGE as the cluster (the link of the item key; passages are parallel across languages): the same multinomial passage counts W serve every language and precision, so the
sample of a language is (W @ S) / (W @ n) with S the per-passage sums of the item values and n the per-passage item counts. Retention, macro retention (D-041), classes and bands are the T-LM
functions of phase1_func_analysis.py, so the outputs have the same schema and the figures module reads them unchanged. Accuracy is derived from the same scores and reported as a secondary table."""
import csv
import json
import re
from pathlib import Path

import numpy as np

import phase1_func_analysis as fa

BANDS = fa.BANDS
EFFECT_FIELDS = ["precision", "language", "effect", "lo", "hi", "harm_group", "harm_control", "norm_ratio_group", "norm_ratio_control"]


def _arr(x):
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=np.float64)


def option_logp_correct(scores, answers):
    """log softmax over the four options, at the correct one; stable for very negative or very large scores."""
    s = np.asarray(scores, dtype=np.float64)
    m = s.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(s - m).sum(axis=1))
    return s[np.arange(len(s)), np.asarray(answers, dtype=int)] - lse


def scores_from_item(fmt, item):
    """(n_items, 4) option scores of one configuration from the stored item arrays."""
    if fmt == "letter":
        return _arr(item["cand_logp"])
    if fmt == "cloze":
        return -(_arr(item["nll"]) / _arr(item["ntok"])).reshape(-1, 4)
    raise ValueError(f"unknown format {fmt!r}; choose letter or cloze")


class McRun:
    """One precision of one format: keys {code: [(link, question_number)]}, answers {code: [index of the correct option]}, scores {(code, config): (n_items, 4)}; config `none` is the baseline."""
    def __init__(self, label, fmt, languages, keys, answers, scores):
        self.label, self.fmt, self.languages = label, fmt, list(languages)
        self.keys = {c: [tuple(k) for k in v] for c, v in keys.items()}
        self.answers = {c: np.asarray(v, dtype=int) for c, v in answers.items()}
        self.scores = {k: np.asarray(v, dtype=np.float64) for k, v in scores.items()}
        for (code, name), s in self.scores.items():
            if s.shape != (len(self.keys[code]), 4):
                raise ValueError(f"{label}: scores of {name} on {code} have shape {s.shape}, expected ({len(self.keys[code])}, 4)")
        self._lp = {}

    @classmethod
    def from_file(cls, path):
        import torch
        d = torch.load(path, weights_only=False)
        scores = {}
        for key, item in d["items"].items():
            code, name = key.split("|", 1)
            scores[(code, name)] = scores_from_item(d["format"], item)
        return cls(d["precision"], d["format"], d["languages"], d["keys"], d["answers"], scores)

    def has(self, code, config):
        return (code, config) in self.scores

    def configs_on(self, code):
        return sorted(n for (c, n) in self.scores if c == code and n != "none")

    def logp(self, code, config):
        if (code, config) not in self._lp:
            self._lp[(code, config)] = option_logp_correct(self.scores[(code, config)], self.answers[code])
        return self._lp[(code, config)]

    def harm(self, code, config):
        return -(self.logp(code, config) - self.logp(code, "none"))

    def accuracy(self, code, config):
        return float((self.scores[(code, config)].argmax(1) == self.answers[code]).mean())


def passage_index(runs):
    """The global, sorted list of passages (links) over all runs and languages: the clusters of the bootstrap."""
    return sorted({k[0] for r in runs for code in r.languages for k in r.keys[code]})


def passage_draws(passages, B, seed):
    """B draws of len(passages) passages with replacement as counts (B x P), seeded; one matrix serves every language and precision."""
    P = len(passages)
    return np.random.default_rng(seed).multinomial(P, np.full(P, 1.0 / P), size=B)


def cluster_samples(values, passage_idx, W):
    """Bootstrap samples of the mean of `values` over items when whole passages are resampled with the counts W: (W @ per-passage sums) / (W @ per-passage item counts)."""
    P = W.shape[1]
    S = np.bincount(passage_idx, weights=np.asarray(values, dtype=np.float64), minlength=P)
    n = np.bincount(passage_idx, minlength=P)
    den = W @ n
    if (den == 0).any():
        raise ValueError("a bootstrap draw left a language without any item")
    return (W @ S) / den


def _passage_positions(run, code, index):
    pos = {p: i for i, p in enumerate(index)}
    return np.array([pos[k[0]] for k in run.keys[code]], dtype=int)


def mc_effect(run, code, K, index, W):
    """E for a language: harm of `zero_<K>@<code>` minus the mean harm of its controls `ctrl<r>_<K>@<code>` (item level), with the point value and the bootstrap samples under the draws W."""
    name = f"zero_{K}@{code}"
    ctrl = [n for n in run.configs_on(code) if re.fullmatch(rf"ctrl\d+_{re.escape(K)}@{re.escape(code)}", n)]
    if not run.has(code, name) or not ctrl:
        raise KeyError(f"{run.label}: {code} lacks {name} or its controls")
    hg = run.harm(code, name); hc = np.mean([run.harm(code, c) for c in ctrl], axis=0); diff = hg - hc
    return {"point": float(diff.mean()), "samples": cluster_samples(diff, _passage_positions(run, code, index), W), "harm_group": float(hg.mean()), "harm_control": float(hc.mean()),
            "n_items": len(diff), "n_controls": len(ctrl), "norm_ratio_group": None, "norm_ratio_control": None}


def sham_effect(run, code, K, index, W):
    h = run.harm(code, f"sham_{K}@{code}")
    return {"point": float(h.mean()), "samples": cluster_samples(h, _passage_positions(run, code, index), W)}


def accuracy_rows(runs, K):
    rows = []
    for lab, r in runs.items():
        for code in r.languages:
            zero = f"zero_{K}@{code}"
            if not r.has(code, zero):
                continue
            ctrl = [n for n in r.configs_on(code) if re.fullmatch(rf"ctrl\d+_{re.escape(K)}@{re.escape(code)}", n)]
            sham = f"sham_{K}@{code}"
            rows.append({"precision": lab, "language": code, "n_items": len(r.keys[code]), "acc_none": r.accuracy(code, "none"), "acc_zero": r.accuracy(code, zero),
                         "acc_control_mean": float(np.mean([r.accuracy(code, c) for c in ctrl])) if ctrl else None, "acc_sham": r.accuracy(code, sham) if r.has(code, sham) else None})
    return rows


def analyse_mc(runs, out_dir, K="K64", n_boot=2000, seed=0, covered=None, bands=BANDS, note=None, retention_level=0.95):
    """Effects, macro effects, retention against full precision, sham and accuracy -> analysis.json, effects_by_language.csv, retention.csv, accuracy.csv and report.md in out_dir. The intervals of the
    effects (statement S0) are 95 % intervals; `retention_level` is the level of the intervals and classes of the retention ratios (the T-MC statements S1 to S5 use 0.99, Q19 default a)."""
    if not 0.0 < retention_level < 1.0:
        raise ValueError(f"retention_level must be strictly between 0 and 1, not {retention_level}")
    if "FP" not in runs:
        raise ValueError("the full-precision run is needed as the reference")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fp = runs["FP"]
    for lab, r in runs.items():
        for code in fp.languages:
            if code in r.languages and r.keys[code] != fp.keys[code]:
                raise ValueError(f"the items of {lab} differ from the full-precision items on {code}")
    index = passage_index(list(runs.values())); W = passage_draws(index, n_boot, seed)
    langs = [c for c in fp.languages if all(r.has(c, f"zero_{K}@{c}") for r in runs.values())]
    cov = [c for c in (covered or langs) if c in langs]
    eff = {lab: {c: mc_effect(r, c, K, index, W) for c in langs} for lab, r in runs.items()}
    mac = {lab: fa.macro(eff[lab], cov) for lab in runs}
    res = {"task": "T-MC", "format": fp.fmt, "K": K, "n_boot": n_boot, "seed": seed, "bands": list(bands), "covered": cov, "n_passages": len(index), "n_positions": len(index), "note": note, "retention_level": retention_level,
           "effects": {}, "macro": {}, "retention": {}, "sham": {}}
    for lab in runs:
        res["effects"][lab] = {c: {**fa._row(eff[lab][c]["point"], eff[lab][c]["samples"]), "harm_group": eff[lab][c]["harm_group"], "harm_control": eff[lab][c]["harm_control"],
                                   "norm_ratio_group": None, "norm_ratio_control": None, "n_items": eff[lab][c]["n_items"], "n_controls": eff[lab][c]["n_controls"]} for c in langs}
        res["macro"][lab] = {**fa._row(mac[lab]["point"], mac[lab]["samples"]), "median": float(np.median([eff[lab][c]["point"] for c in cov])), "n_positive": int(sum(eff[lab][c]["point"] > 0 for c in cov))}
        res["sham"][lab] = {c: fa._row(s["point"], s["samples"]) for c in langs if runs[lab].has(c, f"sham_{K}@{c}") for s in [sham_effect(runs[lab], c, K, index, W)]}
        if lab != "FP":
            res["retention"][lab] = {"macro": fa.retention_macro(eff[lab], eff["FP"], cov, bands, level=retention_level), "pooled": fa.retention(mac[lab], mac["FP"], bands, level=retention_level),
                                     "languages": {c: fa.retention(eff[lab][c], eff["FP"][c], bands, level=retention_level) for c in langs}}
    acc = accuracy_rows(runs, K)
    res["accuracy"] = acc
    (out / "analysis.json").write_text(json.dumps(res, indent=1) + "\n")
    with open(out / "effects_by_language.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(EFFECT_FIELDS)
        for lab in runs:
            for c in langs:
                e = res["effects"][lab][c]; w.writerow([lab, c, e["point"], e["lo"], e["hi"], e["harm_group"], e["harm_control"], "", ""])
    with open(out / "retention.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["precision", "language", "ratio", "lo", "hi", "class"])
        for lab, r in res["retention"].items():
            for c, v in [("MACRO_MEAN_OF_RATIOS", r["macro"]), ("POOLED_RATIO", r["pooled"])] + list(r["languages"].items()):
                w.writerow([lab, c, v["ratio"], v["lo"], v["hi"], v["class"]])
    with open(out / "accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["precision", "language", "n_items", "acc_none", "acc_zero", "acc_control_mean", "acc_sham"]); w.writeheader(); w.writerows(acc)
    (out / "report.md").write_text(report_text(res))
    return res


def report_text(res):
    K, B = res["K"], res["n_boot"]
    L = [f"# Functional effects of language feature groups on Belebele (T-MC, {res['format']} format, group size {K})", ""]
    if res.get("note"):
        L += [f"**{res['note']}**", ""]
    L += ["E is the harm (nats, the drop in the log-probability of the correct option among the four, after the edit minus before it, same precision, same items) of zero-ablating the language's selective "
          f"group beyond the mean harm of its activity-matched controls; positive = the selective group matters more than equally active non-selective features. Intervals: {B} paired cluster-bootstrap draws "
          "over passages (the same draws for every language and precision; percentile, descriptive; the observed value is the estimate). Macro retention is the mean over languages of the per-language ratio E at a "
          f"precision over E in full precision (D-041), classified against the bands [{res['bands'][0]}, {res['bands'][1]}] using {100 * res['retention_level']:g} % intervals (the effect intervals are 95 % intervals); "
          "undefined when the full-precision 95 % interval does not lie above zero.", "",
          f"Covered languages: {len(res['covered'])} ({', '.join(res['covered'])}). Passages: {res['n_passages']}.", "", "## Macro effect and retention", "",
          "| precision | mean E | 95 % interval | median E | languages with E > 0 | retention (mean of per-language ratios) | " + f"{100 * res['retention_level']:g} % interval | class (languages) |", "|---|---|---|---|---|---|---|---|"]
    for lab in res["macro"]:
        m = res["macro"][lab]; r = res["retention"].get(lab, {}).get("macro")
        head = f"| {lab} | {m['point']:.4f} | [{m['lo']:.4f}, {m['hi']:.4f}] | {m['median']:.4f} | {m['n_positive']} of {len(res['covered'])} | "
        L.append(head + (f"{r['ratio']:.2f} | [{r['lo']:.2f}, {r['hi']:.2f}] | {r['class']} ({r['n_languages']}) |" if r and r["ratio"] is not None else f"- | - | {r['class'] if r else 'reference'} |"))
    L += ["", "## Sham check (the edit path with no intended change; must be exactly 0)", "", "| precision | language | harm | interval |", "|---|---|---|---|"]
    for lab, d in res["sham"].items():
        for c, s in d.items():
            L.append(f"| {lab} | {c} | {s['point']:.5f} | [{s['lo']:.5f}, {s['hi']:.5f}] |")
    L += ["", "## Effect by language", "", "| precision | language | E | interval | group harm | control harm | items |", "|---|---|---|---|---|---|---|"]
    for lab, d in res["effects"].items():
        for c, e in d.items():
            L.append(f"| {lab} | {c} | {e['point']:.4f} | [{e['lo']:.4f}, {e['hi']:.4f}] | {e['harm_group']:.4f} | {e['harm_control']:.4f} | {e['n_items']} |")
    L += ["", "## Accuracy (derived from the same scores; secondary)", "", "| precision | language | items | none | zero-ablation | control mean | sham |", "|---|---|---|---|---|---|---|"]
    fmt = lambda v: "-" if v is None else f"{v:.3f}"
    for a in res["accuracy"]:
        L.append(f"| {a['precision']} | {a['language']} | {a['n_items']} | {fmt(a['acc_none'])} | {fmt(a['acc_zero'])} | {fmt(a['acc_control_mean'])} | {fmt(a['acc_sham'])} |")
    return "\n".join(L) + "\n"


_NOTES = {"eval": "Held-out T-MC evaluation (devtest-origin passages); the confirmatory family of the design, settings frozen before scoring.",
          "cal": "Calibration-split run (dev-origin passages, the text the features were selected on): a pipeline check, not a result."}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Analyse functional-validation item files (T-MC, Belebele).")
    ap.add_argument("--runs", required=True, help="directory with func_mc_<split>_<format>_<precision>.pt files"); ap.add_argument("--split", choices=["cal", "eval"], default="eval")
    ap.add_argument("--format", choices=["letter", "cloze"], required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--K", default="K64"); ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--covered", nargs="*")
    ap.add_argument("--note", help="status line of the report (default by split)")
    ap.add_argument("--retention-level", type=float, default=0.95, help="level of the retention intervals and classes (the T-MC statements S1 to S5: 0.99)")
    a = ap.parse_args(argv)
    runs = {}
    for p in sorted(Path(a.runs).glob(f"func_mc_{a.split}_{a.format}_*.pt")):
        r = McRun.from_file(p); runs[r.label] = r
    analyse_mc(runs, a.out, a.K, a.n_boot, a.seed, a.covered, note=a.note or _NOTES[a.split], retention_level=a.retention_level)
    print("wrote", a.out, sorted(runs))


if __name__ == "__main__":
    main()
