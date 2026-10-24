"""Study G, task G5 (design: docs/superpowers/specs/2026-10-24-graded-gain-study-design.md sections 2, 5, 7 and amendment A1): analysis of the graded-gain runs. CPU only, numpy and scipy.

Quantities, per language l, state s and ratio alpha (alpha 0 = zero-ablation, alpha 1 = the unedited model): the harm of the scaled group (nats per token on the language's own text, NLL after the edit minus
NLL before it in the same state, pooled over the language's tokens), the harm of the same ratio applied to the matched controls (mean over the draws), and the net harm (group minus controls). Uncertainty:
the paired cluster bootstrap over sentence positions of `phase1_func_analysis` (one set of draws for every language and state, so differences are paired). The rules of the design are in `rules` (below)."""
import csv
import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

import phase1_func_analysis as fa
import phase1_gain_ratios as rt
import phase1_gain_run as gr

LADDER = ("FP", "INT8", "INT7", "INT6", "INT5", "INT4")


class GainRuns:
    """The item files of the states of one gain study. runs: {state label: fa.Run}; down, up: the three chosen ratios each (down in (0, 1), up above 1, shallow to deep); fine_up: the extra ratios above 1;
    n_controls: control draws per language; label: group size label. Refuses a set without the FP run, with an unknown state, with different languages or sentence ids, or with a missing configuration."""
    def __init__(self, runs, down, up, fine_up, n_controls, label="K64"):
        if "FP" not in runs:
            raise ValueError("the full-precision run is required")
        unknown = [s for s in runs if s not in LADDER]
        if unknown:
            raise ValueError(f"states {unknown} are not on the ladder {LADDER}")
        if len(down) != 3 or len(up) != 3 or any(not 0 < d < 1 for d in down) or any(u <= 1 for u in list(up) + list(fine_up)):
            raise ValueError(f"three down ratios in (0, 1), three up ratios above 1 and fine ratios above 1 are required, got {down}, {up}, {fine_up}")
        self.runs, self.down, self.up, self.fine_up = dict(runs), list(down), list(up), list(fine_up)
        self.n_controls, self.label = int(n_controls), label
        self.states = [s for s in LADDER if s in runs]
        ref = runs["FP"]
        self.languages = list(ref.languages)
        for s in self.states:
            r = runs[s]
            if list(r.languages) != self.languages:
                raise ValueError(f"{s}: languages differ from the FP run")
            if list(r.ids) != list(ref.ids):
                raise ValueError(f"{s}: sentence ids differ from the FP run")
        self.all_ratios = sorted(set(self.down + self.up + self.fine_up))
        self.check_complete()

    def config_names(self, code):
        """Names of every configuration the study needs on language `code`."""
        names = [rt.zero_name(self.label, code)] + [rt.zeroed_control_name(r, self.label, code) for r in range(self.n_controls)]
        for a in self.all_ratios:
            names.append(rt.group_name(a, self.label, code))
            names += [rt.control_name(a, r, self.label, code) for r in range(self.n_controls)]
        return names

    def check_complete(self):
        for s in self.states:
            for code in self.languages:
                for n in self.config_names(code):
                    if not self.runs[s].has(code, n):
                        raise KeyError(f"{s}: {code} lacks the configuration {n}")


class Table:
    """Point values and bootstrap samples of every (state, language, ratio) cell."""
    def __init__(self, cells, base, n_boot):
        self._cells, self._base, self.n_boot = cells, base, n_boot

    def get(self, state, code, alpha):
        return self._cells[(state, code, float(alpha))]

    def base_loss(self, state, code):
        return self._base[(state, code)]

    def total_change(self, state, code, alpha):
        """Loss of the edited model in `state` minus the unedited full-precision loss, on the language's own text."""
        return self._base[(state, code)] + self._cells[(state, code, float(alpha))]["g"] - self._base[("FP", code)]


def build_table(gains, n_boot=2000, seed=0):
    """Cells for every state and language at the ratios of the study, at 0 (zero-ablation) and at 1 (the unedited model, all zeros)."""
    n = len(gains.runs["FP"].ids)
    D = fa.draws(n, n_boot, seed)
    cells, base = {}, {}
    zero = np.zeros(n_boot)
    for s in gains.states:
        run = gains.runs[s]
        for code in gains.languages:
            b, nt0 = run.base(code)
            base[(s, code)] = float(b.sum() / nt0.sum())
            sets = [(0.0, rt.zero_name(gains.label, code), [rt.zeroed_control_name(r, gains.label, code) for r in range(gains.n_controls)])]
            sets += [(a, rt.group_name(a, gains.label, code), [rt.control_name(a, r, gains.label, code) for r in range(gains.n_controls)]) for a in gains.all_ratios]
            for a, gname, cnames in sets:
                dg, nt = run.delta(code, gname)
                dc = np.mean([run.delta(code, c)[0] for c in cnames], axis=0)
                g_s, c_s = fa._boot(dg, nt, D), fa._boot(dc, nt, D)
                cells[(s, code, a)] = {"g": fa.pooled(dg, nt), "c": fa.pooled(dc, nt), "net": fa.pooled(dg - dc, nt), "g_s": g_s, "c_s": c_s, "net_s": g_s - c_s}
            cells[(s, code, 1.0)] = {"g": 0.0, "c": 0.0, "net": 0.0, "g_s": zero, "c_s": zero, "net_s": zero}
    return Table(cells, base, n_boot)


# ------------------------------------------------------------------------------------------------------------------------------ decision rules (design section 5, amendment A1)
def required(n, k27):
    """The design's count k27 out of 27 languages, proportionally for another number of languages (exact for 27)."""
    return math.ceil(k27 * n / 27 - 1e-9)


def _rho(x, y):
    """Spearman correlation; NaN (never an exception or a warning) when an input is constant."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.spearmanr(x, y)[0])


def _upper(samples):
    return fa._ci(samples)[1]


def _lower(samples):
    return fa._ci(samples)[0]


def h_g1(table, gains, state="FP"):
    """Dose, down: net harm strictly increases from the mildest chosen down ratio to the deepest and to zero-ablation. Met when at least 24 of 27 languages are strictly ordered and the median over
    languages is strictly ordered. `spearman`: per language, Spearman of (depth 0..3, net harm)."""
    ratios = list(gains.down) + [0.0]
    seqs = {c: [table.get(state, c, a)["net"] for a in ratios] for c in gains.languages}
    per = {c: bool(all(x < y for x, y in zip(s, s[1:]))) for c, s in seqs.items()}
    med = [float(np.median([seqs[c][i] for c in gains.languages])) for i in range(len(ratios))]
    ordered = all(x < y for x, y in zip(med, med[1:]))
    agree = sum(per.values())
    return {"state": state, "n": len(per), "agree": agree, "per_language": per, "median_net": med, "median_ordered": bool(ordered), "spearman": {c: _rho(range(len(ratios)), s) for c, s in seqs.items()},
            "met": bool(agree >= required(len(per), 24) and ordered)}


def h_g2(table, gains, state="FP"):
    """Down is specific: at the deepest chosen down ratio the lower bound of the 95 % interval of the net harm is above 0 in at least 24 of 27 languages."""
    d = gains.down[-1]
    per = {c: bool(_lower(table.get(state, c, d)["net_s"]) > 0) for c in gains.languages}
    count = sum(per.values())
    return {"state": state, "ratio": d, "n": len(per), "count": count, "per_language": per, "met": bool(count >= required(len(per), 24))}


def h_g3(table, gains, state="FP"):
    """Dose, up: group harm strictly increasing over the three up ratios in at least 24 of 27 languages, and the lower bound of the net harm interval above 0 at the largest up ratio in at least 20."""
    inc = {c: bool(all(table.get(state, c, a)["g"] < table.get(state, c, b)["g"] for a, b in zip(gains.up, gains.up[1:]))) for c in gains.languages}
    pos = {c: bool(_lower(table.get(state, c, gains.up[-1])["net_s"]) > 0) for c in gains.languages}
    n = len(inc)
    return {"state": state, "n": n, "count_increasing": sum(inc.values()), "count_net_positive": sum(pos.values()), "per_language_increasing": inc, "per_language_net_positive": pos,
            "met": bool(sum(inc.values()) >= required(n, 24) and sum(pos.values()) >= required(n, 20))}


def h_g4(table, gains):
    """Precision, down: along FP, INT8, ... the group harm at each chosen down ratio does not fall. Per language Kendall tau of (position on the ladder, harm); met when, at the deepest ratio, tau > 0 in at
    least 24 of 27 languages and the median over languages never falls by more than one bootstrap standard error (the larger of the two states' errors of the median)."""
    states = gains.states
    if len(states) < 2:
        return {"testable": False, "met": False}
    langs = gains.languages
    per_ratio = {}
    for d in gains.down:
        tau = {}
        for c in langs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tau[c] = float(stats.kendalltau(range(len(states)), [table.get(s, c, d)["g"] for s in states])[0])
        per_ratio[d] = {"tau": tau, "count_positive": sum(1 for v in tau.values() if v > 0)}
    deepest = gains.down[-1]
    med = {s: float(np.median([table.get(s, c, deepest)["g"] for c in langs])) for s in states}
    se = {s: float(np.median(np.stack([table.get(s, c, deepest)["g_s"] for c in langs]), axis=0).std(ddof=1)) for s in states}
    ok = all(med[b] >= med[a] - max(se[a], se[b]) for a, b in zip(states, states[1:]))
    count = per_ratio[deepest]["count_positive"]
    return {"testable": True, "per_ratio": per_ratio, "median_by_state": med, "se_by_state": se, "median_ok": bool(ok), "count_positive": count, "n": len(langs),
            "met": bool(count >= required(len(langs), 24) and ok)}


def alpha_star(table, gains):
    """Per state and language the ratio (1, the study ratios and the fine ratios; not zero-ablation) with the lowest loss, i.e. the lowest harm of the group; equal values go to the ratio nearest to 1 (log scale)."""
    grid = [1.0] + list(gains.all_ratios)
    return {s: {c: min(grid, key=lambda a: (table.get(s, c, a)["g"], abs(math.log(a)))) for c in gains.languages} for s in gains.states}


def h_g5(table, gains, baseline_g_ratio=None, seed=0, n_boot=2000):
    """Repair by turning up at INT4 (amendment A1). 5a generic: the median over languages of the matched-control harm at 1.1 or 1.2 is below 0 with the 95 % interval (bootstrap over languages) below 0.
    5b group-specific: per language the better (lower net harm) of the fine ratios; at least 14 of 27 languages with the upper bound of the net interval below 0. `alpha_star` per state and language; the
    rank correlation of alpha*(l, INT4) with 1 / g_ratio(l, INT4) of the baseline (`baseline_g_ratio` {language: ratio}) must exceed 0.3."""
    if "INT4" not in gains.states:
        return {"testable": False, "met": False}
    langs, fine = gains.languages, sorted(gains.fine_up)
    D = np.random.default_rng(seed).integers(0, len(langs), size=(n_boot, len(langs)))
    per_ratio = {}
    for a in fine:
        c = np.array([table.get("INT4", code, a)["c"] for code in langs])
        lo, hi = fa._ci(np.median(c[D], axis=1))
        per_ratio[a] = {"median": float(np.median(c)), "lo": lo, "hi": hi}
    chosen = {code: min(fine, key=lambda a: (table.get("INT4", code, a)["net"], a)) for code in langs}
    count = sum(1 for code in langs if _upper(table.get("INT4", code, chosen[code])["net_s"]) < 0)
    star = alpha_star(table, gains)
    corr = None
    if baseline_g_ratio is not None:
        rho = _rho([star["INT4"][c] for c in langs], [1.0 / baseline_g_ratio[c] for c in langs])
        corr = {"rho": rho, "n": len(langs), "met": bool(rho > 0.3)}
    return {"testable": True, "a": {"per_ratio": per_ratio, "met": bool(any(v["hi"] < 0 for v in per_ratio.values()))},
            "b": {"count": count, "n": len(langs), "chosen": chosen, "met": bool(count >= required(len(langs), 14))},
            "alpha_star": star, "alpha_up_count": {s: sum(1 for v in star[s].values() if v > 1) for s in gains.states}, "corr": corr}


def h_g6(table, gains):
    """Consistency across languages at the deepest chosen down ratio: in every state the net harm is positive in at least 24 of 27 languages; the ranking of languages by net harm agrees between FP and INT4
    (Spearman above 0.5); the spread (interquartile range over languages) of the net harm as a fraction of each language's own zero-ablation net harm is reported."""
    d, langs = gains.down[-1], gains.languages
    per_state, iqr = {}, {}
    for s in gains.states:
        nets = {c: table.get(s, c, d)["net"] for c in langs}
        count = sum(1 for v in nets.values() if v > 0)
        per_state[s] = {"count_positive": count, "met": bool(count >= required(len(langs), 24))}
        frac = [nets[c] / table.get(s, c, 0.0)["net"] for c in langs if table.get(s, c, 0.0)["net"] > 0]
        iqr[s] = float(np.subtract(*np.percentile(frac, [75, 25]))) if frac else float("nan")
    corr = None
    if "INT4" in gains.states:
        corr = _rho([table.get("FP", c, d)["net"] for c in langs], [table.get("INT4", c, d)["net"] for c in langs])
    return {"ratio": d, "per_state": per_state, "rank_corr": corr, "iqr_fraction": iqr, "met": bool(all(v["met"] for v in per_state.values()) and (corr is None or corr > 0.5))}


# ------------------------------------------------------------------------------------------------------------------------------ loading, evaluation, report
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_runs(dirs, ratios_json, split="eval"):
    """GainRuns from job directories (files func_lm_<split>_<state>.pt and meta_lm_<split>.json) and the decision file of the ratios. Refuses a job that is not complete, was cut by the time limit or failed a
    gate, presets that differ between states, and ratios that differ from the decision. Returns (GainRuns, {state: {"file", "sha256"}})."""
    dec = json.loads(Path(ratios_json).read_text())
    down, up, fine = dec["down"], dec["up"], dec["fine_up"]
    runs, info, presets = {}, {}, set()
    for d in map(Path, dirs):
        files = sorted(d.glob(f"func_lm_{split}_*.pt"))
        if not files:
            continue
        meta_path = d / f"meta_lm_{split}.json"
        meta = json.loads(meta_path.read_text())
        if not meta.get("complete") or meta.get("skipped_for_time"):
            raise ValueError(f"{d}: the job is not complete or skipped a state for time ({meta.get('skipped_for_time')})")
        failed = [k for k, v in (meta.get("gates") or {}).items() if not v]
        if failed:
            raise ValueError(f"{d}: gates failed: {failed}")
        import torch
        for f in files:
            run = fa.Run.from_file(f)
            presets.add(torch.load(f, weights_only=False)["preset"])
            runs[run.label] = run
            info[run.label] = {"file": f.name, "dir": str(d), "sha256": _sha256(f), "meta_sha256": _sha256(meta_path)}
    if not runs:
        raise FileNotFoundError(f"no func_lm_{split}_<state>.pt in {[str(d) for d in dirs]}")
    if len(presets) != 1:
        raise ValueError(f"the states were run with different presets: {sorted(presets)}")
    ratios, n_controls, label = gr.parse_preset(presets.pop())
    if set(ratios) != set(down + up + fine):
        raise ValueError(f"the ratios of the runs {sorted(ratios)} differ from the decision {sorted(down + up + fine)}")
    return GainRuns(runs, down, up, fine, n_controls, label), info


def load_baseline_g_ratio(path, state="INT4"):
    """{language: g_ratio} of one state from the G1 baseline.csv (the group's activation on its language relative to FP)."""
    with open(path) as fh:
        return {r["language"]: float(r["g_ratio"]) for r in csv.DictReader(fh) if r["state"] == state}


def evaluate(table, gains, baseline_g_ratio=None, n_boot=2000, seed=0):
    """Every rule of the design on a table: rules (per state where the rule is per state) and `verdicts` (the verdict of each hypothesis; the per-state rules are judged at FP, as pre-registered)."""
    st = gains.states
    rules = {"H-G1": {s: h_g1(table, gains, s) for s in st}, "H-G2": {s: h_g2(table, gains, s) for s in st}, "H-G3": {s: h_g3(table, gains, s) for s in st},
             "H-G4": h_g4(table, gains), "H-G5": h_g5(table, gains, baseline_g_ratio, seed, n_boot), "H-G6": h_g6(table, gains)}
    g4, g5 = rules["H-G4"], rules["H-G5"]
    verdicts = {"H-G1": rules["H-G1"]["FP"]["met"], "H-G2": rules["H-G2"]["FP"]["met"], "H-G3": rules["H-G3"]["FP"]["met"],
                "H-G4": g4["met"] if g4.get("testable") else None,
                "H-G5a": g5["a"]["met"] if g5.get("testable") else None, "H-G5b": g5["b"]["met"] if g5.get("testable") else None,
                "H-G5-alpha": g5["corr"]["met"] if g5.get("testable") and g5.get("corr") else None, "H-G6": rules["H-G6"]["met"]}
    return {"rules": rules, "verdicts": verdicts, "alpha_star": alpha_star(table, gains)}


def curve_rows(table, gains):
    """One row per state, language and ratio (zero-ablation is ratio 0): harms with 95 % intervals, net harm and the total change against the unedited FP model."""
    rows = []
    for s in gains.states:
        for c in gains.languages:
            for a in [0.0] + list(gains.all_ratios):
                cell = table.get(s, c, a)
                glo, ghi = fa._ci(cell["g_s"]); nlo, nhi = fa._ci(cell["net_s"])
                rows.append({"state": s, "language": c, "ratio": a, "group_harm": cell["g"], "group_lo": glo, "group_hi": ghi, "control_harm": cell["c"], "net": cell["net"], "net_lo": nlo, "net_hi": nhi,
                             "total_change": table.total_change(s, c, a)})
    return rows


def _verdict(v):
    return "not testable" if v is None else ("**met**" if v else "**not met**")


def _f(v, d=4):
    return "-" if v is None or v != v else f"{v:.{d}f}"


def report_text(res, table, gains, meta):
    r, v = res["rules"], res["verdicts"]
    n = len(gains.languages)
    g1, g2, g3, g4, g5, g6 = r["H-G1"]["FP"], r["H-G2"]["FP"], r["H-G3"]["FP"], r["H-G4"], r["H-G5"], r["H-G6"]
    out = ["# Study G: graded gain of language-selective features", "",
           "Exploratory (the FLORES devtest sentences were examined in earlier studies, D-039); the rules below were fixed in the design before any number "
           "(`docs/superpowers/specs/2026-10-24-graded-gain-study-design.md` sections 5 and 9). Harm = NLL after the edit minus NLL before it in the same state (nats per token on the language's own text); "
           "net harm = group minus the mean of the matched controls scaled by the same ratio; intervals are paired bootstraps over sentences.", "",
           f"Down ratios {', '.join(_f(a) for a in gains.down)} (plus zero-ablation); up ratios {', '.join(_f(a) for a in gains.up)}; fine up ratios {', '.join(_f(a, 2) for a in gains.fine_up)}; "
           f"{gains.n_controls} control draws; states {', '.join(gains.states)}; {n} languages.", "",
           "## Verdicts", "", "| hypothesis | rule (counts out of the languages) | result | verdict |", "|---|---|---|---|",
           f"| H-G1 dose, down (FP) | net harm strictly ordered over the three down ratios and zero-ablation, in at least {required(n, 24)}, and the median ordered | {g1['agree']} of {n}; median net {', '.join(_f(x) for x in g1['median_net'])} | {_verdict(v['H-G1'])} |",
           f"| H-G2 down is specific (FP) | lower bound of net harm above 0 at ratio {_f(g2['ratio'])}, in at least {required(n, 24)} | {g2['count']} of {n} | {_verdict(v['H-G2'])} |",
           f"| H-G3 dose, up (FP) | group harm increasing over the up ratios in at least {required(n, 24)} and net harm above 0 at the largest in at least {required(n, 20)} | {g3['count_increasing']} and {g3['count_net_positive']} of {n} | {_verdict(v['H-G3'])} |"]
    if g4.get("testable"):
        out.append(f"| H-G4 precision, down | tau > 0 at the deepest down ratio in at least {required(n, 24)}, median not falling by more than one standard error | {g4['count_positive']} of {n}; median ok {g4['median_ok']} | {_verdict(v['H-G4'])} |")
    if g5.get("testable"):
        a_txt = "; ".join(f"x{a:g}: median control harm {_f(x['median'])} [{_f(x['lo'])}, {_f(x['hi'])}]" for a, x in g5["a"]["per_ratio"].items())
        out.append(f"| H-G5a generic repair (INT4) | median control harm below 0 at 1.1 or 1.2, interval below 0 | {a_txt} | {_verdict(v['H-G5a'])} |")
        out.append(f"| H-G5b group-specific repair (INT4) | upper bound of net harm below 0 at the better fine ratio in at least {required(n, 14)} | {g5['b']['count']} of {n} | {_verdict(v['H-G5b'])} |")
        c = g5["corr"]
        out.append(f"| H-G5 alpha* against 1 / g_ratio (INT4) | Spearman above 0.3 | {_f(c['rho'], 3) if c else '-'} | {_verdict(v['H-G5-alpha'])} |")
    out.append(f"| H-G6 consistency | net harm positive at ratio {_f(g6['ratio'])} in at least {required(n, 24)} in every state and Spearman FP against INT4 above 0.5 | "
               f"{', '.join(f'{s} {x['count_positive']}' for s, x in g6['per_state'].items())}; rank correlation {_f(g6['rank_corr'], 3)} | {_verdict(v['H-G6'])} |")
    out += ["", "## Median over the languages by state and ratio (nats per token)", ""]
    ratios = [0.0] + list(gains.all_ratios)
    for s in gains.states:
        out += [f"### {s}", "", "| ratio | group harm | control harm | net harm | languages with net > 0 | total change vs unedited FP |", "|---|---|---|---|---|---|"]
        for a in ratios:
            cells = [table.get(s, c, a) for c in gains.languages]
            out.append(f"| {_f(a, 4)} | {_f(float(np.median([x['g'] for x in cells])))} | {_f(float(np.median([x['c'] for x in cells])))} | {_f(float(np.median([x['net'] for x in cells])))} | "
                       f"{sum(1 for x in cells if x['net'] > 0)} of {n} | {_f(float(np.median([table.total_change(s, c, a) for c in gains.languages])))} |")
        out.append("")
    d, u = gains.down[-1], gains.up[-1]
    out += [f"## By language: net harm at ratio {_f(d)} and {_f(u)}", "", "| language | " + " | ".join(f"{s} down" for s in gains.states) + " | " + " | ".join(f"{s} up" for s in gains.states) + " |",
            "|---|" + "---|" * (2 * len(gains.states))]
    for c in gains.languages:
        out.append(f"| {c} | " + " | ".join(_f(table.get(s, c, d)["net"]) for s in gains.states) + " | " + " | ".join(_f(table.get(s, c, u)["net"]) for s in gains.states) + " |")
    star = res["alpha_star"]
    out += ["", "## Optimal gain alpha* (the ratio with the lowest loss; 1 = leave the group alone)", "", "| state | languages with alpha* above 1 | languages with alpha* below 1 | median alpha* |", "|---|---|---|---|"]
    for s in gains.states:
        vals = list(star[s].values())
        out.append(f"| {s} | {sum(1 for x in vals if x > 1)} of {n} | {sum(1 for x in vals if x < 1)} of {n} | {_f(float(np.median(vals)), 3)} |")
    out += ["", "## Limits", "",
            "- Exploratory, already examined data; one model, one layer, one SAE, RTN weights only; the loss on the language's own text, not task accuracy.",
            "- The group is the frozen full-precision K64 group (small for most languages); the controls are activity-matched but not identical in function; alpha* is read on a coarse grid.",
            "- Counts are the decision rules; the cell tables are descriptive and uncorrected for multiple comparisons.", ""]
    return "\n".join(out)


def write_outputs(out_dir, res, table, gains, meta):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = curve_rows(table, gains)
    with open(out / "curves.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "alpha_star.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["state", "language", "alpha_star"])
        for s in gains.states:
            for c in gains.languages:
                w.writerow([s, c, res["alpha_star"][s][c]])
    (out / "analysis.json").write_text(json.dumps({"verdicts": res["verdicts"], "rules": _jsonable(res["rules"]), "ratios": {"down": gains.down, "up": gains.up, "fine_up": gains.fine_up},
                                                   "states": gains.states, "languages": gains.languages, "meta": meta}, indent=1) + "\n")
    (out / "report.md").write_text(report_text(res, table, gains, meta))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, float) and x != x:
        return None
    return x


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Study G, G5: analysis of the graded-gain runs (files func_lm_eval_<state>.pt of the main jobs).")
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--ratios-json", required=True); ap.add_argument("--baseline-csv"); ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="eval"); ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    gains, info = load_runs(a.runs, a.ratios_json, a.split)
    base = load_baseline_g_ratio(a.baseline_csv) if a.baseline_csv else None
    table = build_table(gains, a.n_boot, a.seed)
    res = evaluate(table, gains, base, a.n_boot, a.seed)
    meta = {"runs": info, "n_boot": a.n_boot, "seed": a.seed, "ratios_json_sha256": _sha256(a.ratios_json), "baseline_csv_sha256": _sha256(a.baseline_csv) if a.baseline_csv else None}
    write_outputs(a.out, res, table, gains, meta)
    print(f"wrote {a.out}: " + ", ".join(f"{k} {_verdict(v)}" for k, v in res["verdicts"].items()))


if __name__ == "__main__":
    main()
