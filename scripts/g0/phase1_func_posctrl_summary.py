"""Summary of the positive control of the Belebele edit path (plan Task 17; specified before it was written, after the item-level numbers had been read: see the plan). Pure csv and json.

Input: the `accuracy.csv` of the frozen analysis chain (`phase1_func_mc_analysis.py`) of each positive-control run, one run per dose (the label K64 stands for a language-independent random set of that many
non-selective supported features; its controls are random sets of the same size from the selective features of the other languages; phase1_func_posctrl_manifest.py). Per dose: the macro mean over the languages of the
accuracy without edit, with the zero-edit, with the controls and with the sham, the macro differences with percentile intervals from a bootstrap over languages, the control effect for English as the target language against
the others, and the verdicts of the predictions P1 to P4 of the plan. Exploratory; nothing here changes a frozen statement.

    python3 scripts/g0/phase1_func_posctrl_summary.py --dose all <accuracy.csv of the whole-pool run> --dose 1024 <accuracy.csv of the 1,024 run> [--dose 256 ...] --out <dir>"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ORDER = ["256", "1024", "all"]
NEEDED = ("language", "acc_none", "acc_zero", "acc_control_mean", "acc_sham")


def _boot(v, n_boot, seed):
    v = np.asarray(v, dtype=float); draws = np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v))); means = v[draws].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise_dose(rows, n_boot=2000, seed=0, english="eng_Latn"):
    """One dose from the rows of its accuracy.csv (full-precision rows only)."""
    fp = [r for r in rows if r.get("precision", "FP") == "FP"]
    if not fp:
        raise ValueError("no full-precision row")
    for c in NEEDED:
        if any(c not in r for r in fp):
            raise ValueError(f"missing column {c}")
    langs = [r["language"] for r in fp]; A = {k: np.array([float(r[c]) for r in fp]) for k, c in (("none", "acc_none"), ("zero", "acc_zero"), ("controls", "acc_control_mean"), ("sham", "acc_sham"))}
    out = {"n_languages": len(langs), "languages": langs, "macro": {k: float(v.mean()) for k, v in A.items()}, "diff": {}, "per_language": {l: {k: float(A[k][i]) for k in A} for i, l in enumerate(langs)}}
    for name, (a, b) in {"zero_minus_none": ("zero", "none"), "controls_minus_none": ("controls", "none"), "zero_minus_controls": ("zero", "controls"), "sham_minus_none": ("sham", "none")}.items():
        d = A[a] - A[b]; mean, lo, hi = _boot(d, n_boot, seed); out["diff"][name] = {"mean": mean, "lo": lo, "hi": hi, "n_negative": int((d < 0).sum()), "per_language": {l: float(d[i]) for i, l in enumerate(langs)}}
    cn = A["controls"] - A["none"]; eng = [i for i, l in enumerate(langs) if l == english]; oth = [i for i, l in enumerate(langs) if l != english]
    out["controls_by_target"] = {"english": float(cn[eng[0]]) if eng else None, "others_mean": float(cn[oth].mean()) if oth else None, "n_others": len(oth)}
    return out


def verdicts(doses):
    """The verdicts of P1 to P4 and the largest fall of any edit set; `doses` maps the dose labels ('256', '1024', 'all') to `summarise_dose` results."""
    v = {}; big = doses.get("all"); fall = lambda s, k="zero_minus_none": s["diff"][k]["mean"]
    if big is None:
        v["P1"] = {"verdict": "not evaluable", "reading": "no whole-pool dose"}; v["P4"] = {"verdict": "not evaluable"}
    else:
        f = fall(big); v["P1"] = {"verdict": "met" if f <= -0.05 + 1e-9 else "not met", "fall": f, "reading": "met" if f <= -0.05 + 1e-9 else ("intermediate" if f <= -0.01 + 1e-9 else "no fall")}
        z = big["diff"]["zero_minus_controls"]["mean"]; v["P4"] = {"verdict": "met" if z < 0 else "not met", "zero_minus_controls": z}
    present = [k for k in ORDER if k in doses]
    if len(present) < 2:
        v["P2"] = {"verdict": "not evaluable", "falls": {k: fall(doses[k]) for k in present}}
    else:
        falls = [fall(doses[k]) for k in present]; v["P2"] = {"verdict": "met" if all(b <= a + 1e-12 for a, b in zip(falls, falls[1:])) else "not met", "falls": dict(zip(present, falls))}
    v["P3"] = {"verdict": "met" if all(all(x == 0.0 for x in s["diff"]["sham_minus_none"]["per_language"].values()) for s in doses.values()) else "not met"}
    best = None
    for k in present:
        for name, key in (("zero", "zero_minus_none"), ("controls", "controls_minus_none")):
            f = doses[k]["diff"][key]["mean"]
            if best is None or f < best["fall"]:
                best = {"set": name, "dose": k, "fall": f}
    v["largest_fall"] = best
    return v


def _fmt(x):
    return "n/a" if x is None else f"{x:+.4f}"


def report_text(doses, v, meta):
    L = ["# Positive control of the Belebele edit path (plan Task 17)", "",
         "**Exploratory; a positive control, not a result about the language groups.** In each run the label K64 stands for a language-independent random set of the non-selective supported features of the stated size (full pool 2,860; 1,024); the controls are random sets of the same size from the selective features of the other languages "
         "(random sets of the same size from the selective features of the other languages). Full precision, the 18 covered languages, all evaluation items, letter format, the frozen analysis chain. The item-level numbers were read before the summary was specified (see the plan), so the verdicts below are applied to them, not predictions.", "",
         "| dose | accuracy without edit | zero-edit | controls | sham | zero - none [95 % interval over languages] | controls - none | zero - controls | languages with a fall (zero) |", "|---|---|---|---|---|---|---|---|---|"]
    for k in [x for x in ORDER if x in doses]:
        s = doses[k]; d = s["diff"]; z = d["zero_minus_none"]
        L.append(f"| {k} | {s['macro']['none']:.4f} | {s['macro']['zero']:.4f} | {s['macro']['controls']:.4f} | {s['macro']['sham']:.4f} | {z['mean']:+.4f} [{z['lo']:+.4f}, {z['hi']:+.4f}] | {_fmt(d['controls_minus_none']['mean'])} [{_fmt(d['controls_minus_none']['lo'])}, {_fmt(d['controls_minus_none']['hi'])}] | {_fmt(d['zero_minus_controls']['mean'])} | {z['n_negative']} of {s['n_languages']} |")
    L += ["", "Control effect (controls - none) for English as the target language against the other languages (the prompt scaffolding of every item is English: 'Passage:', 'Question:', 'Answer:'; the controls of a non-English language contain the English-selective features, those of English do not):", "",
          "| dose | English | mean of the other languages | other languages |", "|---|---|---|---|"]
    for k in [x for x in ORDER if x in doses]:
        c = doses[k]["controls_by_target"]; L.append(f"| {k} | {_fmt(c['english'])} | {_fmt(c['others_mean'])} | {c['n_others']} |")
    L += ["", "## Verdicts of the predictions of the plan", "",
          f"- **P1** (the whole pool lowers the macro accuracy by at least 0.05): **{v['P1']['verdict']}**" + (f" (fall {v['P1']['fall']:+.4f}, reading: {v['P1']['reading']})" if "fall" in v["P1"] else "") + ".",
          f"- **P2** (the fall of the zero-edit is monotone in the dose): **{v['P2']['verdict']}**" + (f" ({', '.join(f'{k} {x:+.4f}' for k, x in v['P2']['falls'].items())})" if v["P2"].get("falls") else "") + ".",
          f"- **P3** (the sham equals no edit in every language): **{v['P3']['verdict']}**.",
          f"- **P4** (at the whole pool the zero-edit lowers the accuracy more than the controls): **{v['P4']['verdict']}**" + (f" (zero - controls {v['P4']['zero_minus_controls']:+.4f})" if "zero_minus_controls" in v["P4"] else "") + ".", ""]
    b = v["largest_fall"]
    L += [f"**Largest fall of any edit set in the series:** {b['set']} at dose {b['dose']}, {b['fall']:+.4f} in macro accuracy. This, and not P1 alone, answers whether the edit path can register an effect on Belebele at all.", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summary of the positive control of the Belebele edit path.")
    ap.add_argument("--dose", nargs=2, action="append", metavar=("LABEL", "ACCURACY_CSV"), required=True); ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--english", default="eng_Latn"); a = ap.parse_args(argv)
    doses = {}
    for label, path in a.dose:
        if label not in ORDER:
            raise ValueError(f"the dose label must be one of {ORDER}")
        with open(path, newline="") as fh:
            doses[label] = summarise_dose(list(csv.DictReader(fh)), a.n_boot, a.seed, a.english)
    if len({tuple(s["languages"]) for s in doses.values()}) != 1:
        raise ValueError("the doses do not cover the same languages")
    v = verdicts(doses); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"doses": doses, "verdicts": v, "n_boot": a.n_boot, "seed": a.seed}, indent=1) + "\n"); (out / "summary.md").write_text(report_text(doses, v, {}))
    print("wrote", out, "|", {k: x["verdict"] for k, x in v.items() if "verdict" in x}, "| largest fall", v["largest_fall"])


if __name__ == "__main__":
    main()
