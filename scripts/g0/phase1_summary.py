"""N10 (decision D-024): a one-page state-of-the-evidence summary for the team, `reports/phase1_summary.md`, whose NUMBERS ARE GENERATED from the committed result files (summary.csv,
analysis.json, thresholds.json, magnitude.json, mixed_variants.json, language_level.json, behavior.json under reports/phase1/<regime>/eq/) so that they cannot drift; the static prose
around the generated block states no results (a test checks both: the committed file equals what this script regenerates, and the prose contains no result-like numbers).
It is a results summary, not paper prose (the team writes the prose). Stdlib only.
  python3 scripts/g0/phase1_summary.py --init     create reports/phase1_summary.md from the static text (refuses to overwrite)
  python3 scripts/g0/phase1_summary.py            regenerate the block between the markers
  python3 scripts/g0/phase1_summary.py --check    exit 1 if the committed file differs from the regenerated one"""
import csv
import json
import sys
from pathlib import Path

BEGIN = "<!-- BEGIN GENERATED (scripts/g0/phase1_summary.py; do not edit by hand) -->"
END = "<!-- END GENERATED -->"

STATIC_TEXT = f"""# Phase 1: state of the evidence (a results summary, not paper prose)

This file is a results summary for the team. It is **not paper prose** (course policy: the team writes the prose). Every number below sits in the generated block and is produced from the committed result files by `scripts/g0/phase1_summary.py`; a test fails if the committed file and the files disagree. The text around the block states no results.

Setting: Gemma-2-2B, layer-12 residual stream, Gemma Scope 16k SAE, round-to-nearest weight quantization with Duan et al.'s own functions, 27 FLORES-200 languages in three resource tiers. The primary stream masks the BOS token and uses equal token budgets per language; the BOS-free stream is a sensitivity analysis whose instrument fails its own noise-floor check (A-029).

Where the details are: the full account is `reports/phase1_results.md`, the decisions and anomalies are in `reports/decisions_and_anomalies.md`, the questions that change how the evidence is headlined are in `DECISIONS_FOR_REVIEW.md` (Q1, Q2, Q13, Q14, Q15).

{BEGIN}
{END}

Reading notes (qualitative; read with the caveats in the register):

- The confirmatory family is the proposal's pre-registered family; its results are in the second table. Everything in the third table was added after the confirmatory results were known; each row says how it was specified, and none is part of the Holm family.
- The tier and activation-scale analyses are correlational (27 languages, three tiers); the activation scale and the survival are computed from the same activations and are partly mechanically related.
- Which analysis is the confirmatory test of the resource-tier question is an open decision (Q14).
- The BOS-free column is a sensitivity analysis only.
"""

LABELS = {"conf": "confirmatory (proposal section 6; D-014)", "N1": "planned in the team's table, definitions provisional (D-018)", "N2": "additional, post hoc thresholds (D-019)",
          "N6": "additional, pre-specified before running (D-020)", "N7": "exploratory (D-014); variants pre-specified (D-021)",
          "N8": "additional, motivated post hoc and pre-specified before running (D-022)", "N9": "additional, pre-specified before running (D-023)"}
LEGEND = [("confirmatory", "the proposal's pre-registered family (section 6): the INT7 and INT6 selective-versus-matched contrasts and the language-level tier test, Holm over three"),
          ("planned in the team's table", "named in the team's evaluation-plan table but with undefined terms; definitions are provisional choices"),
          ("additional, pre-specified before running", "not in the proposal's family; the analysis and its reading rule were written down before it was run"),
          ("additional, motivated post hoc", "as above, but motivated by an unplanned observation made on the results"),
          ("additional, post hoc thresholds", "the threshold values were chosen after seeing the data"),
          ("exploratory", "reported and labelled as such; p-values uncorrected")]


class SummaryError(Exception):
    pass


def _json(path):
    path = Path(path)
    if not path.exists():
        raise SummaryError(f"missing file {path}")
    return json.loads(path.read_text())


def _get(d, path, where):
    cur = d
    for part in path.split("/"):
        if not isinstance(cur, dict) or part not in cur:
            raise SummaryError(f"{where}: missing key {path}")
        cur = cur[part]
    return cur


class Stream:
    def __init__(self, root, regime):
        self.dir = Path(root) / regime / "eq"; self.regime = regime
        self.csv = self.dir / "summary.csv"; self.cache = {}

    def j(self, rel, optional=False):
        p = self.dir / rel
        if optional and not p.exists():
            return None
        if rel not in self.cache:
            self.cache[rel] = _json(p)
        return self.cache[rel]

    def g(self, rel, key, optional=False):
        d = self.j(rel, optional)
        return None if d is None else _get(d, key, str(self.dir / rel))


def _ladder(stream):
    if not stream.csv.exists():
        raise SummaryError(f"missing file {stream.csv}")
    with open(stream.csv, newline="") as f:
        rows = {int(r["bits"]): r for r in csv.DictReader(f)}
    out = {}
    for b in (8, 7, 6, 5, 4, 16, 12, 0):
        if b not in rows:
            raise SummaryError(f"{stream.csv}: no row for bits {b}")
        r = rows[b]; out[b] = (float(r["survived_>0.9_pct"]), float(r["median_corr"]), float(r["ppl_delta_pct"]))
    return out


def table_ladder(a, b):
    la, lb = _ladder(a), _ladder(b)
    L = ["## 1. Pooled survival (their metric, all languages pooled, equal budget)", "", "| width | primary: survived > 0.9 | median r | perplexity change | BOS-free: survived > 0.9 | median r | perplexity change |", "|---|---|---|---|---|---|---|"]
    names = {8: "INT8", 7: "INT7", 6: "INT6", 5: "INT5", 4: "INT4", 16: "INT16 (noise floor)", 12: "INT12 (noise floor)", 0: "INT0 (restore only)"}
    for bits in (8, 7, 6, 5, 4, 16, 12, 0):
        (sa, ma, pa), (sb, mb, pb) = la[bits], lb[bits]
        L.append(f"| {names[bits]} | {sa:.2f} % | {ma:.4f} | {pa:+.2f} % | {sb:.2f} % | {mb:.4f} | {pb:+.2f} % |")
    return L


def _test_cell(s, key, hk):
    t = "analysis.json"; d = s.g(t, f"tests/{key}/delta"); lo, hi = s.g(t, f"tests/{key}/ci"); p = s.g(t, f"tests/{key}/p"); h = s.g(t, f"holm/{hk}")
    return f"{d * 100:+.2f} pp [{lo * 100:+.2f}, {hi * 100:+.2f}], p {p:.3g}, Holm {h:.3g}"


def _yes(x):
    return "yes" if x else "no"


def _t3_cell(s):
    t = "analysis.json"; sl = s.g(t, "tests/T3_tier/slope"); p = s.g(t, "tests/T3_tier/p"); h = s.g(t, "holm/T3_tier"); hl = s.g(t, "tests/T3_tier/high_minus_low")
    b1, e1 = s.g(t, "tests/T3_tier/tost_slope/bound"), s.g(t, "tests/T3_tier/tost_slope/equivalent"); e2 = s.g(t, "tests/T3_tier/tost_high_minus_low/equivalent")
    return f"slope {sl:+.4f}, p {p:.3g}, Holm {h:.3g}; equivalent within +-{b1:g}: {_yes(e1)}; High minus Low {hl:+.4f}, equivalent within +-{b1:g}: {_yes(e2)}"


def table_confirmatory(a, b):
    L = ["", "## 2. Confirmatory family (Holm over three tests)", "", f"Status of every row: {LABELS['conf']}.", "", "| test | primary stream (BOS masked, equal budget) | BOS-free sensitivity |", "|---|---|---|",
         f"| T1 INT7 selective minus matched universal survival | {_test_cell(a, 'T1_int7', 'T1_int7')} | {_test_cell(b, 'T1_int7', 'T1_int7')} |",
         f"| T2 INT6 selective minus matched universal survival | {_test_cell(a, 'T2_int6', 'T2_int6')} | {_test_cell(b, 'T2_int6', 'T2_int6')} |",
         f"| T3 language-level tier slope of the selective-minus-control gap (one-sided) | {_t3_cell(a)} | {_t3_cell(b)} |"]
    return L


def _row_by_label(s, label):
    for r in s.g("thresholds/thresholds.json", "rows"):
        if r.get("label") == label:
            return r
    raise SummaryError(f"{s.dir / 'thresholds/thresholds.json'}: no row labelled {label}")


def _n1(s, primary=True):
    f = "behavior.json"; rho = s.g(f, "widths/6/survival/spearman/rho"); p = s.g(f, "widths/6/survival/spearman/p_one_sided")
    return f"INT6 Spearman(selective damage, loss increase) {rho:+.2f} (one-sided p {p:.3g})"


def _n2(s, primary=True):
    r = _row_by_label(s, "5.0"); return f"INT6 selective minus control at S > 5: {r['tests']['T2_int6']['delta'] * 100:+.2f} pp (Holm {r['holm']['T2_int6']:.3g})"


def _n6(s, primary=True):
    f = "magnitude/magnitude.json"; d = s.g(f, "readings/5.0/by_bins/10/matched_delta"); h = s.g(f, "readings/5.0/by_bins/10/matched_holm"); rd = s.g(f, "readings/5.0/final/reading")
    return f"INT6 at S > 5, controls matched on firing rate x 10 magnitude bins: {d * 100:+.2f} pp (Holm {h:.3g}); headline reading {rd}"


def _n7(s, primary=True):
    f = "mixed_variants/mixed_variants.json"
    if not primary and s.j(f, optional=True) is None:           # only the BOS-free sensitivity stream may lack the variants; the primary stream may not
        return "not run"
    nt = s.g(f, "variants/no_tier_all/S_rank/coef"); p = s.g(f, "variants/no_tier_all/S_rank/p"); fa = s.g(f, "variants/full_all/S_rank/coef"); lab = s.g(f, "reading/label")
    return f"S~ coefficient without the tier terms {nt:+.2f} (p {p:.2g}); in the full model {fa:+.2f}; reading: {lab}"


def _n8(s, primary=True):
    f = "language_level/language_level.json"; m = [s.g(f, f"widths/6/mean_by_tier/{k}") * 100 for k in "012"]
    sl, h, sf, pf = s.g(f, "widths/6/slope"), s.g(f, "widths/6/holm_p"), s.g(f, "widths/6/slope_fertility"), s.g(f, "widths/6/p_fertility")
    return f"INT6 mean survival low / mid / high tier {m[0]:.1f} / {m[1]:.1f} / {m[2]:.1f} %; slope {sl * 100:+.1f} pp per tier step (Holm {h:.2g}); with fertility {sf * 100:+.1f} pp (p {pf:.2g})"


def _n9(s, primary=True):
    f = "language_level/language_level.json"; x, p = s.g(f, "activation_scale/widths/6/slope_scale"), s.g(f, "activation_scale/widths/6/p_scale")
    rho, joint = s.g(f, "activation_scale/widths/6/spearman_scale_survival"), s.g(f, "activation_scale/reading/joint")
    return f"INT6 slope with activation scale {x * 100:+.1f} pp (p {p:.3g}); Spearman(scale, survival) {rho:+.2f}; {joint}"


ROWS = [("N1", "Behavioural link: per-language selective damage against loss increase", _n1, "D-018, A-032, V-11; reports/phase1/mask/eq/behavior.json"),
        ("N2", "Fixed S thresholds for the selective split (deviation from the proposal's null percentiles)", _n2, "D-019, A-034; reports/phase1/mask/eq/thresholds/thresholds.json"),
        ("N6", "Activation-magnitude-matched controls", _n6, "D-020, A-035; reports/phase1/mask/eq/magnitude/magnitude.json"),
        ("N7", "Exploratory mixed model refitted in variants", _n7, "D-021, A-034, A-036, A-037; reports/phase1/mask/eq/mixed_variants/mixed_variants.json"),
        ("N8", "Language-level survival by resource tier", _n8, "D-022, A-037, A-038; reports/phase1/mask/eq/language_level/language_level.json"),
        ("N9", "Activation scale as a covariate of the language-level tier test", _n9, "D-023, A-039; reports/phase1/mask/eq/language_level/language_level.json")]


def table_additional(a, b):
    L = ["", "## 3. Additional analyses (not part of the Holm family)", "", "| id | analysis | status | primary stream (BOS masked, equal budget) | BOS-free sensitivity | register and file |", "|---|---|---|---|---|---|"]
    for key, name, fn, ref in ROWS:
        L.append(f"| {key} | {name} | {LABELS[key]} | {fn(a, True)} | {fn(b, False)} | {ref} |")
    return L


def table_legend():
    return ["", "## 4. Status labels", "", "| label | meaning |", "|---|---|"] + [f"| {a} | {b} |" for a, b in LEGEND]


def render_generated(root):
    a, b = Stream(root, "mask"), Stream(root, "none")
    return "\n".join(table_ladder(a, b) + table_confirmatory(a, b) + table_additional(a, b) + table_legend()) + "\n"


def _replace(text, root):
    if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) > text.index(END):
        raise SummaryError("the file must contain exactly one begin marker followed by one end marker")
    i, j = text.index(BEGIN) + len(BEGIN), text.index(END)
    return text[:i] + "\n" + render_generated(root) + text[j:]


def write_summary(root, md_path):
    md_path = Path(md_path)
    if not md_path.exists():
        raise SummaryError(f"{md_path} does not exist (create it with --init)")
    new = _replace(md_path.read_text(), root); md_path.write_text(new); return new


def check(root, md_path):
    md_path = Path(md_path)
    if not md_path.exists():
        return False
    text = md_path.read_text(); return _replace(text, root) == text


def init_summary(md_path):
    md_path = Path(md_path)
    if md_path.exists():
        raise SummaryError(f"{md_path} already exists")
    md_path.write_text(STATIC_TEXT)


def main(argv=None):
    import argparse
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(repo / "reports" / "phase1")); ap.add_argument("--md", default=str(repo / "reports" / "phase1_summary.md"))
    ap.add_argument("--init", action="store_true"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.init:
            init_summary(a.md); print("created", a.md)
        elif a.check:
            ok = check(a.root, a.md); print("up to date" if ok else "DIFFERS from the regenerated block"); return 0 if ok else 1
        else:
            write_summary(a.root, a.md); print("wrote", a.md)
    except SummaryError as e:
        print("error:", e, file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
