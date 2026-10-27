"""Per-language appendix tables of reports/PROJECT_RESULTS.md (stdlib only). Every number is read from a committed result file, nothing is typed by hand.
Usage: python3 -B scripts/g0/make_project_results_tables.py [--doc reports/PROJECT_RESULTS.md]
It replaces the text between the two marker lines BEGIN/END GENERATED TABLES of the document; without --doc it prints the tables.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports"
BEGIN, END = "<!-- BEGIN GENERATED TABLES -->", "<!-- END GENERATED TABLES -->"
LADDER = ["INT8", "INT7", "INT6", "INT5", "INT4"]


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def f(x, d=3, plus=False):
    if x in ("", None):
        return "–"
    x = float(x)
    return f"{x:+.{d}f}" if plus else f"{x:.{d}f}"


def table(head, body):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(out)


def median(xs):
    xs = [float(x) for x in xs if x not in ("", None)]
    return statistics.median(xs) if xs else None


def index(rs, *keys):
    return {tuple(r[k] for k in keys): r for r in rs}


def roster():
    langs = json.loads((R / "g0/phase1_languages.json").read_text())["languages"]
    return [(l["code"], l["name"], l["tier"], l["script"]) for l in langs]


def lang_label(l):
    return f"{l[1]} ({l[0][:3]}, {l[2]})"


def t_roster(L):
    return table(["code", "language", "resource tier", "script"], [(c, n, t, s) for c, n, t, s in L])


def t_survival(L):
    d = index(rows(R / "phase1/mask/eq/per_language.csv"), "language", "bits")
    body = [[lang_label(l)] + [f(d[(l[0], b)]["survived_>0.9_pct"], 1) for b in ("8", "7", "6", "5", "4")] for l in L]
    body.append(["**median of the 27**"] + [f(median(d[(l[0], b)]["survived_>0.9_pct"] for l in L), 1) for b in ("8", "7", "6", "5", "4")])
    return table(["language", "INT8", "INT7", "INT6", "INT5", "INT4"], body)


def t_association(L):
    d = index(rows(R / "association_retention/results/eq/language_summary.csv"), "group", "condition", "language")
    states = ["INT16"] + LADDER
    body = []
    for l in L:
        est = d[("selective", "INT8", l[0])]["fp_established"]
        body.append([lang_label(l), est] + [f(100 * float(d[("selective", s, l[0])]["conditional_fraction"]), 1) for s in states])
    if ("selective", "INT8", "POOLED") in d:
        body.append(["**all 27 pooled**", d[("selective", "INT8", "POOLED")]["fp_established"]] + [f(100 * float(d[("selective", s, "POOLED")]["conditional_fraction"]), 1) for s in states])
    return table(["language", "features with the language tag in full precision"] + [s + " kept %" for s in states], body)


def t_function(L):
    eff = index(rows(R / "functional_validation/results/eval/effects_by_language.csv"), "precision", "language")
    ret = index(rows(R / "functional_validation/results/eval/retention.csv"), "precision", "language")
    mark = {"preserved": "", "amplified": " ▲", "inconclusive": " ?"}
    body = []
    for l in L:
        e = eff[("FP", l[0])]
        body.append([lang_label(l), f(e["harm_group"]), f(e["harm_control"]), f(e["effect"]) + f" [{f(e['lo'])}, {f(e['hi'])}]"]
                    + [f(ret[(s, l[0])]["ratio"], 2) + mark[ret[(s, l[0])]["class"]] for s in LADDER])
    body.append(["**median of the 27**", f(median(eff[("FP", l[0])]["harm_group"] for l in L)), f(median(eff[("FP", l[0])]["harm_control"] for l in L)),
                 f(median(eff[("FP", l[0])]["effect"] for l in L))] + [f(median(ret[(s, l[0])]["ratio"] for l in L), 2) for s in LADDER])
    return table(["language", "harm of zeroing the group (nats/token)", "harm of zeroing matched controls", "effect = group − control [95 % interval]"] + [s + " ratio" for s in LADDER], body)


def t_function_rounding(L):
    ret = index(rows(R / "functional_validation/results/eval/retention.csv"), "precision", "language")
    return table(["precision", "median ratio over the 27 languages", "meaning"],
                 [[s, f(median(ret[(s, l[0])]["ratio"] for l in L), 2), m] for s, m in
                  (("INT16", "rounding only (control)"), ("INT12", "rounding only (control)"), ("INT0", "no quantization at all (control)"))]
                 + [[s, f(median(ret[(s, l[0])]["ratio"] for l in L), 2), "quantized"] for s in LADDER])


def t_patching(L):
    d = index(rows(R / "functional_validation/results/rescue/rescue.csv"), "precision", "language", "part")
    body = []
    for l in L:
        k = ("INT4", l[0], "group")
        if k in d:
            body.append([lang_label(l), f(d[k]["gap"]), f(d[k]["recovered"], 4), f(100 * float(d[k]["fraction"]), 1) + " %"
                         + f" [{f(100 * float(d[k]['fraction_lo']), 1)}, {f(100 * float(d[k]['fraction_hi']), 1)}]"])
    return table(["language", "loss gap INT4 minus full precision (nats/token)", "recovered by putting the group's full-precision activations back", "share of the gap [95 % interval]"], body)


def t_belebele(L):
    acc = index(rows(R / "functional_validation/results/mc_eval/accuracy.csv"), "precision", "language")
    body = []
    for l in L:
        k = ("FP", l[0])
        if k not in acc:
            body.append([lang_label(l), "not covered", "", "", ""] + ["" for _ in LADDER])
            continue
        a = acc[k]
        body.append([lang_label(l), a["n_items"], f(a["acc_none"]), f(a["acc_zero"]), f(a["acc_control_mean"])] + [f(acc[(s, l[0])]["acc_none"]) for s in LADDER])
    cov = [l for l in L if ("FP", l[0]) in acc]
    body.append(["**mean of the covered languages**", "", f(sum(float(acc[("FP", l[0])]["acc_none"]) for l in cov) / len(cov)), f(sum(float(acc[("FP", l[0])]["acc_zero"]) for l in cov) / len(cov)),
                 f(sum(float(acc[("FP", l[0])]["acc_control_mean"]) for l in cov) / len(cov))] + [f(sum(float(acc[(s, l[0])]["acc_none"]) for l in cov) / len(cov)) for s in LADDER])
    return table(["language", "items", "FP: nothing removed", "FP: group removed", "FP: matched controls removed"] + ["no edit, " + s for s in LADDER], body)


def gain_cell(d, state, lang, ratio, digits=3):
    for (s, l, r), row in d.items():
        if s == state and l == lang and abs(float(r) - ratio) < 1e-6:
            return f(row["net"], digits, plus=True)
    return "–"


def t_gain(L):
    cur = {(r["state"], r["language"], r["ratio"]): r for r in rows(R / "functional_validation/gain_study/results/curves.csv")}
    states = ["FP", "INT8", "INT7", "INT6", "INT5", "INT4"]
    zero = [[lang_label(l)] + [gain_cell(cur, s, l[0], 0.0) for s in states] for l in L]
    zero.append(["**median of the 27**"] + [f(median(row["net"] for (s, l, r), row in cur.items() if s == st and abs(float(r)) < 1e-9), 3, plus=True) for st in states])
    up = [[lang_label(l)] + [gain_cell(cur, s, l[0], x, 4) for s in ("FP", "INT6", "INT4") for x in (1.1, 1.2)] + [gain_cell(cur, s, l[0], 11.31370849898476) for s in ("FP", "INT4")] for l in L]
    return (table(["language"] + [f"{s}, group removed" for s in states], zero) + "\n\n"
            + table(["language", "FP ×1.1", "FP ×1.2", "INT6 ×1.1", "INT6 ×1.2", "INT4 ×1.1", "INT4 ×1.2", "FP ×11.3", "INT4 ×11.3"], up))


def t_transfer(L, name, langs=None):
    d = index(rows(R / f"intervention_transfer/results/{name}/group_per_language.csv"), "state", "language")
    sel = [l for l in L if (("INT16", l[0]) in d)]
    body = []
    for l in sel:
        a, b, c = d[("INT16", l[0])], d[("INT6", l[0])], d[("INT4", l[0])]
        body.append([lang_label(l), f(a["net_FP_FP"]), f(b["C"], 4, True), f(b["W"], 4, True), f(b["I"], 4, True), f(c["net_FP_FP"]), f(c["net_Q_FP"]), f(c["net_FP_Q"]), f(c["net_Q_Q"]),
                     f(c["net_Q_Q_native"]), f(c["C"], 3, True), f(c["W"], 3, True), f(c["I"], 3, True)])
    return table(["language", "full precision: harm of removing the group", "INT6: C", "INT6: W", "INT6: I", "INT4: nothing quantized",
                  "INT4: only the layers below the edit quantized", "INT4: only the layers above the edit quantized", "INT4: both quantized, same edit as before",
                  "INT4: both quantized, edit scaled to the model's own activations", "INT4: C", "INT4: W", "INT4: I"], body)


def t_llama(L):
    llama = index(rows(R / "functional_validation/llama_robust/languages.csv"), "state", "language")
    gemma = index(rows(R / "functional_validation/results/eval/effects_by_language.csv"), "precision", "language")
    body = []
    for l in L:
        k = ("FP", l[0])
        body.append([lang_label(l), f(gemma[k]["effect"]), f(llama[k]["E_mean"]), f(llama[k]["E_median"]), llama[k]["exceeds_all"]])
    body.append(["**median of the 27**", f(median(gemma[("FP", l[0])]["effect"] for l in L)), f(median(llama[("FP", l[0])]["E_mean"] for l in L)),
                 f(median(llama[("FP", l[0])]["E_median"] for l in L)), ""])
    return table(["language", "Gemma-2-2B effect (group − control)", "Llama-3.2-1B effect, mean of 3 control draws", "Llama effect, median of the draws", "group beats all draws?"], body)


SECTIONS = [
    ("A1. The 27 languages", t_roster, "Nine languages per resource tier. 'Tier' is the project's own sorting (D-011); the sources are in `reports/g0/phase1_languages.json`."),
    ("A2. Phase 1: share of active features that survive (token correlation with full precision above 0.9), per language",
     t_survival, "Source: `reports/phase1/mask/eq/per_language.csv` (BOS masked, equal budget). One row per language, in percent. The median row (INT6 90.4 %) differs from the pooled figure of section 4.2 (INT6 94.9 %), which pools all active features of all languages."),
    ("A3. Association retention: of the features that carry the language tag in full precision, how many still carry it (%)",
     t_association, "Source: `reports/association_retention/results/eq/language_summary.csv`, group 'selective', column `conditional_fraction`. INT16 is the rounding-only floor."),
    ("A4. Function: what zeroing the 64 most language-specific features does to the language's loss, and how much of that is kept when the model is quantized",
     t_function, "Source: `reports/functional_validation/results/eval/effects_by_language.csv` and `retention.csv`. Ratio = effect in the quantized model divided by effect in full precision. ▲ = clearly amplified, ? = inconclusive, no mark = within the band of 'kept'."),
    ("A4b. The same ratio for the control rows (rounding only, and no quantization)", t_function_rounding,
     "These rows show what 'no change' looks like: INT0 is exactly 1.00; INT16 and INT12 are rounding only."),
    ("A5. Patching experiment (F3): how much of the INT4 loss gap is repaired when the group's full-precision activations are put back",
     t_patching, "Source: `reports/functional_validation/results/rescue/rescue.csv`, part 'group'. Small numbers mean the language groups hold little of the damage."),
    ("A6. Downstream task (Belebele reading comprehension), accuracy per language", t_belebele,
     "Source: `reports/functional_validation/results/mc_eval/accuracy.csv`. Chance is 0.25. Nine languages are not covered because the model cannot read them well enough for the task to measure anything."),
    ("A7. Study G (scaling the group up and down): net harm per language in nats per token (group minus matched controls)", t_gain,
     "Source: `reports/functional_validation/gain_study/results/curves.csv`. First table: group removed (ratio 0). Second table: group scaled UP; negative = the loss got slightly lower (a 'repair')."),
    ("A8. Intervention transfer, main matrix (27 languages, FLORES): harm of removing the group, split into a context part and a weights part",
     lambda L: t_transfer(L, "main"),
     "Source: `reports/intervention_transfer/results/main/group_per_language.csv`. C = change from the activations (context), W = change from the weights, I = the two together beyond the sum. Net harm in nats per token."),
    ("A9. Intervention transfer, external panel (six languages, Wikipedia text)", lambda L: t_transfer(L, "external"),
     "Source: `reports/intervention_transfer/results/external/group_per_language.csv`. Same columns as A8, on fresh text."),
    ("A10. Second model check (Llama-3.2-1B): the group effect in full precision, next to Gemma", t_llama,
     "Source: `reports/functional_validation/llama_robust/languages.csv` and the Gemma table above. The pre-set rule needed 24 of 27 languages above zero."),
]


def build(L):
    parts = []
    for title, fn, note in SECTIONS:
        parts.append(f"### {title}\n\n{note}\n\n{fn(L)}\n")
    return "\n".join(parts)


def main_for(doc_path, L):
    """Replace the text between the markers of `doc_path` by the generated tables."""
    p = Path(doc_path)
    doc = p.read_text()
    i, j = doc.index(BEGIN), doc.index(END)
    p.write_text(doc[:i + len(BEGIN)] + "\n\n" + build(L) + "\n" + doc[j:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc")
    a = ap.parse_args()
    if a.doc:
        main_for(a.doc, roster())
    else:
        print(build(roster()))


if __name__ == "__main__":
    main()
