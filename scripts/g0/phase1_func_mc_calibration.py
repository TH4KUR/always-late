"""Full-precision calibration of the Belebele task (design section 3, rules 1-3): accuracy per language and format on the dev-origin items, the format choice and the language coverage.
No intervention is involved, so using dev-origin text (on which the features were selected) leaks nothing. Rules fixed in the design before any outcome:
the format with the higher macro-average accuracy on the items both formats scored wins (ties go to the letter format); a language is covered when the 95 % Wilson lower bound of its
accuracy in the chosen format exceeds chance (0.25) plus 0.05."""
import json
from pathlib import Path

import numpy as np


def wilson(k, n, z=1.96):
    """95 % Wilson score interval of a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def _arr(x):
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=np.float64)


def correct_letter(run, code):
    """Per item: is the correct option the most probable of the four option letters?"""
    lp = _arr(run["items"][f"{code}|none"]["cand_logp"])
    return lp.argmax(1) == np.asarray(run["answers"][code])


def correct_cloze(run, code, normalise=True):
    """Per item: is the correct option the most likely continuation (mean log-likelihood per token when normalise, total otherwise)? Rows are four per item, option j of item i at 4 * i + j."""
    it = run["items"][f"{code}|none"]
    nll, ntok = _arr(it["nll"]).reshape(-1, 4), _arr(it["ntok"]).reshape(-1, 4)
    score = -nll / ntok if normalise else -nll
    return score.argmax(1) == np.asarray(run["answers"][code])


def _correct(run, code):
    return correct_letter(run, code) if run["format"] == "letter" else correct_cloze(run, code)


def calibrate(runs, chance=0.25, margin=0.05, out_dir=None):
    """runs: {"letter": file dict, "cloze": file dict (optional)} of one FP calibration. Returns the decision record; writes belebele_calibration.json and .md when out_dir is given."""
    fmts = [f for f in ("letter", "cloze") if f in runs]
    langs = list(runs[fmts[0]]["languages"])
    cor = {f: {c: _correct(runs[f], c) for c in langs} for f in fmts}
    keys = {f: {c: [tuple(k) for k in runs[f]["keys"][c]] for c in langs} for f in fmts}
    shared, macro = {}, {}
    if len(fmts) == 2:
        accs = {f: [] for f in fmts}
        for c in langs:
            common = sorted(set(keys["letter"][c]) & set(keys["cloze"][c]))
            shared[c] = len(common)
            for f in fmts:
                pos = {k: i for i, k in enumerate(keys[f][c])}
                accs[f].append(float(np.mean([cor[f][c][pos[k]] for k in common])) if common else float("nan"))
        macro = {f: float(np.nanmean(accs[f])) for f in fmts}
        chosen = "cloze" if macro["cloze"] > macro["letter"] else "letter"
    else:
        chosen = fmts[0]
        macro = {chosen: float(np.mean([cor[chosen][c].mean() for c in langs]))}
    out = {"format": chosen, "chance": chance, "margin": margin, "macro_on_shared_items": macro, "shared_items": shared, "languages": {}, "dropped": {}}
    for c in langs:
        row = {}
        for f in fmts:
            k, n = int(cor[f][c].sum()), int(cor[f][c].size)
            lo, hi = wilson(k, n)
            row[f] = {"acc": k / n if n else float("nan"), "lo": lo, "hi": hi, "n": n}
        row["covered"] = bool(row[chosen]["lo"] > chance + margin)
        out["languages"][c] = row
        out["dropped"][c] = {f: runs[f].get("dropped", {}).get(c, 0) for f in fmts}
    out["covered"] = [c for c in langs if out["languages"][c]["covered"]]
    out["uncovered"] = [c for c in langs if not out["languages"][c]["covered"]]
    if out_dir:
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
        (d / "belebele_calibration.json").write_text(json.dumps(out, indent=1) + "\n")
        (d / "belebele_calibration.md").write_text(report_text(out, fmts))
    return out


def report_text(res, fmts):
    L = ["# Belebele calibration (full precision, dev-origin items, no intervention)", "",
         "Rules fixed in the design before any outcome: the format with the higher macro accuracy on the items both formats scored wins (ties: letter); a language is **covered** when the 95 % Wilson lower bound of its accuracy "
         f"in the chosen format exceeds chance ({res['chance']}) + {res['margin']}. Uncovered languages get no claim from the multiple-choice task.", "",
         f"**Chosen format: {res['format']}.** Macro accuracy on shared items: " + ", ".join(f"{f} {v:.3f}" for f, v in res["macro_on_shared_items"].items()) + ".", "",
         f"## Coverage: {len(res['covered'])} of {len(res['languages'])} languages covered", "", "Covered: " + (", ".join(res["covered"]) or "none"), "", "Not covered: " + (", ".join(res["uncovered"]) or "none"), "",
         "| language | " + " | ".join(f"{f} acc [95 % interval] (n)" for f in fmts) + " | covered | items dropped as too long |", "|---|" + "---|" * (len(fmts) + 2)]
    for c, r in res["languages"].items():
        L.append(f"| {c} | " + " | ".join(f"{r[f]['acc']:.3f} [{r[f]['lo']:.3f}, {r[f]['hi']:.3f}] ({r[f]['n']})" for f in fmts) + f" | {'yes' if r['covered'] else 'no'} | " + ", ".join(f"{f}: {res['dropped'][c][f]}" for f in fmts) + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    import torch
    ap = argparse.ArgumentParser(description="Belebele calibration from func_mc_cal_<format>_FP.pt files.")
    ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    runs = {}
    for f in ("letter", "cloze"):
        p = Path(a.runs) / f"func_mc_cal_{f}_FP.pt"
        if p.exists():
            runs[f] = torch.load(p, weights_only=False)
    res = calibrate(runs, out_dir=a.out)
    print("format", res["format"], "covered", len(res["covered"]), "of", len(res["languages"]))


if __name__ == "__main__":
    main()
