"""Study G, extension C (design: docs/superpowers/specs/2026-10-24-graded-gain-study-design.md section 11): the repair near ratio 1 with a larger sample. CPU only.

Input: the item files of `gainc-<state>` jobs (preset `gain|<ratios>|c5|K64|nz`: scaled groups, scaled matched controls, sham; no zero-ablation) on the 562 reserve devtest sentences with all five control draws.
Units of the primary rules are the LANGUAGES: per language the net harm (group minus the mean of the five control draws, nats per token, pooled over the sentences) at a ratio; the median over languages gets a
percentile interval from a bootstrap over languages (97.5 %: Holm over the two primary ratios 1.1 and 1.2). Rules (fixed before any run):
  H-Gc1  at INT4, for at least one primary ratio the interval of the median net harm lies below 0 AND at least 19 of 27 languages have a negative net harm (the sign test is reported); the interval condition
         holds whenever at least 20 of 27 values are negative and fails with 18 or fewer (the median of a resample is negative iff at least 14 resampled values are negative: P = 0.0035 for 20,
         0.0127 for 19, against 0.0125); 19 is borderline and decided by the seeded bootstrap (seed 0, 5000 draws);
  H-Gc2  at each primary ratio the paired difference (INT4 minus FP, over languages) of the net harm has its interval below 0;
  H-Gc3  the repair is group-specific only if H-Gc1 is met and the interval of the median matched-control harm at that ratio does not lie entirely below 0;
  H-Gc4  descriptive: medians and counts by state and ratio, the ratio with the lowest median group harm, alpha* counts.
The reliability table of `phase1_gain_analysis` (sentence error, draw error, languages clear of twice the error) is reported next to the verdicts."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

import phase1_func_analysis as fa
import phase1_gain_analysis as ga
import phase1_gain_ratios as rt
import phase1_gain_run as gr

PRIMARY = (1.1, 1.2)
LO_PCT, HI_PCT = 1.25, 98.75                      # the 97.5 % interval
MIN_CONTROLS = 5
EXTC_RATIOS = (1.1, 1.2, 1.4, 1.7)


class ExtRuns:
    """The item files of the states of one extension C study. runs: {state: fa.Run}; ratios: the ratios above 1 that were run; n_controls: control draws (at least 5); expect_ids: the sentence ids every language
    must have (the reserve sentences); forbid_ids: ids that must not occur (the evaluation and pilot sentences of the main study)."""
    def __init__(self, runs, ratios, n_controls, label="K64", expect_ids=None, forbid_ids=None):
        if not runs:
            raise ValueError("no run")
        unknown = [s for s in runs if s not in ga.LADDER]
        if unknown:
            raise ValueError(f"states {unknown} are not on the ladder {ga.LADDER}")
        ratios = [float(r) for r in ratios]
        if not ratios or any(r <= 1 for r in ratios) or len(set(ratios)) != len(ratios):
            raise ValueError(f"the ratios must be distinct and above 1, got {ratios}")
        if n_controls < MIN_CONTROLS:
            raise ValueError(f"extension C uses all {MIN_CONTROLS} control draws, got {n_controls}")
        self.runs, self.ratios, self.n_controls, self.label = dict(runs), ratios, int(n_controls), label
        self.states = [s for s in ga.LADDER if s in runs]
        ref = runs[self.states[0]]
        self.languages = list(ref.languages)
        if len({tuple(v) for v in ref.ids.values()}) != 1:
            raise ValueError("the languages were evaluated on different sentence ids")
        self.ids = list(next(iter(ref.ids.values())))
        self.n_sentences = len(self.ids)
        for s in self.states:
            r = runs[s]
            if r.ids != ref.ids:                                   # {language: ids}: also the languages
                raise ValueError(f"{s}: languages or sentence ids differ")
        if expect_ids is not None and self.ids != list(expect_ids):
            raise ValueError("the sentence ids are not the expected ones")
        if forbid_ids is not None and set(self.ids) & set(forbid_ids):
            raise ValueError("the sentences overlap ids that earlier runs used")
        self.check_complete()

    def config_names(self, code):
        names = [f"sham_{self.label}@{code}"]
        for a in self.ratios:
            names.append(rt.group_name(a, self.label, code))
            names += [rt.control_name(a, r, self.label, code) for r in range(self.n_controls)]
        return names

    def check_complete(self):
        for s in self.states:
            for code in self.languages:
                for n in self.config_names(code):
                    if not self.runs[s].has(code, n):
                        raise KeyError(f"{s}: {code} lacks the configuration {n}")


def build_cells(e, n_boot=2000, seed=0):
    """{(state, language, ratio): {"g", "c", "net", "net_s" (bootstrap over sentences), "net_draws" (the net harm against each control draw)}}."""
    D = fa.draws(e.n_sentences, n_boot, seed)
    cells = {}
    for s in e.states:
        run = e.runs[s]
        for code in e.languages:
            for a in e.ratios:
                dg, nt = run.delta(code, rt.group_name(a, e.label, code))
                dcs = [run.delta(code, rt.control_name(a, r, e.label, code))[0] for r in range(e.n_controls)]
                dc = np.mean(dcs, axis=0)
                cells[(s, code, a)] = {"g": fa.pooled(dg, nt), "c": fa.pooled(dc, nt), "net": fa.pooled(dg - dc, nt), "net_s": fa._boot(dg - dc, nt, D),
                                       "net_draws": [fa.pooled(dg - d, nt) for d in dcs]}
    return cells


def median_interval(values, n_boot=5000, seed=0):
    """97.5 % percentile interval of the median over the values (languages), bootstrap over the values with a seeded generator."""
    v = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))
    boot = np.median(v[idx], axis=1)
    return float(np.percentile(boot, LO_PCT)), float(np.percentile(boot, HI_PCT))


def sign_p(k, n):
    """One-sided sign test: probability of at least k negative values out of n under a fair coin."""
    return float(stats.binom.sf(k - 1, n, 0.5))


def h_gc1(cells, langs, state="INT4", ratios=PRIMARY, n_boot=5000, seed=0):
    """Repair at `state`: per primary ratio the median net harm over languages, its 97.5 % interval, the number of negative languages and the sign test; met for a ratio when the interval is below 0 and at
    least 19 of 27 (proportionally) languages are negative; the hypothesis is met when one ratio is."""
    n = len(langs)
    per, met = {}, {}
    for r in ratios:
        nets = [cells[(state, l, r)]["net"] for l in langs]
        lo, hi = median_interval(nets, n_boot, seed)
        k = sum(1 for x in nets if x < 0)
        per[r] = {"median": float(np.median(nets)), "lo": lo, "hi": hi, "n_negative": k, "n": n, "sign_p": sign_p(k, n), "interval_ok": bool(hi < 0), "count_ok": bool(k >= ga.required(n, 19))}
        met[r] = bool(per[r]["interval_ok"] and per[r]["count_ok"])
    return {"state": state, "per_ratio": per, "ratio_met": met, "met": any(met.values())}


def h_gc2(cells, langs, states=("INT4", "FP"), ratios=PRIMARY, n_boot=5000, seed=0):
    """The sign depends on precision: per primary ratio the paired difference (first state minus second, over languages) of the net harm; met when every interval lies below 0."""
    per = {}
    for r in ratios:
        d = [cells[(states[0], l, r)]["net"] - cells[(states[1], l, r)]["net"] for l in langs]
        lo, hi = median_interval(d, n_boot, seed)
        per[r] = {"median_diff": float(np.median(d)), "lo": lo, "hi": hi, "ok": bool(hi < 0)}
    return {"states": list(states), "per_ratio": per, "met": all(v["ok"] for v in per.values())}


def h_gc3(cells, langs, state, h1, ratios=PRIMARY, n_boot=5000, seed=0):
    """Generic or specific: the matched controls' own harm per primary ratio with its interval; `specific` is None unless H-Gc1 is met, then True when at every ratio that met H-Gc1 the control interval does
    not lie entirely below 0."""
    per = {}
    for r in ratios:
        c = [cells[(state, l, r)]["c"] for l in langs]
        lo, hi = median_interval(c, n_boot, seed)
        per[r] = {"median": float(np.median(c)), "lo": lo, "hi": hi, "entirely_below_zero": bool(hi < 0)}
    specific = None
    if h1["met"]:
        specific = all(not per[r]["entirely_below_zero"] for r, ok in h1["ratio_met"].items() if ok)
    return {"state": state, "per_ratio": per, "specific": specific}


def h_gc4_rows(cells, langs, states, ratios):
    """Descriptive table: per state and ratio the median net, group and control harm, the number of negative net harms, the number of languages whose alpha* (the ratio with the lowest group harm among 1 and the
    ratios; ties to the ratio nearest 1) is this ratio, and whether this ratio has the lowest median group harm of the state."""
    rows = []
    for s in states:
        med_g = {r: float(np.median([cells[(s, l, r)]["g"] for l in langs])) for r in ratios}
        best = min(med_g, key=lambda r: (med_g[r], abs(math.log(r))))
        best_is_ratio = med_g[best] < 0
        star = {r: 0 for r in ratios}
        for l in langs:
            grid = [1.0] + list(ratios)
            gval = lambda a: 0.0 if a == 1.0 else cells[(s, l, a)]["g"]
            a_star = min(grid, key=lambda a: (gval(a), abs(math.log(a))))
            if a_star != 1.0:
                star[a_star] += 1
        for r in ratios:
            rows.append({"state": s, "ratio": r, "median_net": float(np.median([cells[(s, l, r)]["net"] for l in langs])), "median_group_harm": med_g[r],
                         "median_control_harm": float(np.median([cells[(s, l, r)]["c"] for l in langs])), "n_negative_net": sum(1 for l in langs if cells[(s, l, r)]["net"] < 0),
                         "n_alpha_star": star[r], "best_ratio_by_median_group_harm": bool(best_is_ratio and r == best), "n_languages": len(langs)})
    return rows


def reliability_rows(cells, e):
    """Sentence error, control-draw error and the number of languages clear of twice the error, per state and ratio (as `phase1_gain_analysis.reliability_rows`)."""
    rows = []
    for s in e.states:
        for a in e.ratios:
            cs = [cells[(s, l, a)] for l in e.languages]
            net = np.array([c["net"] for c in cs])
            se_s = np.array([float(np.std(c["net_s"], ddof=1)) for c in cs])
            se_d = np.array([float(np.std(c["net_draws"], ddof=1)) / math.sqrt(len(c["net_draws"])) for c in cs])
            n_sent, n_both = ga.clear_counts(net, se_s, se_d)
            rows.append({"state": s, "ratio": a, "median_abs_net": float(np.median(np.abs(net))), "median_se_sentences": float(np.median(se_s)), "median_se_draws": float(np.median(se_d)),
                         "n_clear_sentences": n_sent, "n_clear_both": n_both, "n_languages": len(cs)})
    return rows


def band_rows(cells, langs, states, ratios, n_boot=5000, seed=0):
    """Per state and ratio the median net harm over the languages, its 97.5 % interval (the function of the rules, the languages in the order given: the percentile of a bootstrap of 27 values moves with that
    order, so the figure reads these numbers and never recomputes them), the number of negative languages and the number of languages."""
    rows = []
    for s in states:
        for a in ratios:
            nets = [cells[(s, l, a)]["net"] for l in langs]
            lo, hi = median_interval(nets, n_boot, seed)
            rows.append({"state": s, "ratio": a, "median": float(np.median(nets)), "lo": lo, "hi": hi, "n_negative": sum(1 for x in nets if x < 0), "n_languages": len(nets)})
    return rows


def evaluate(cells, e, n_boot=5000, seed=0, ratios=PRIMARY):
    """All rules: H-Gc1 and H-Gc3 need INT4, H-Gc2 needs INT4 and FP; a rule whose state is missing has verdict None."""
    langs = e.languages
    r1 = h_gc1(cells, langs, "INT4", ratios, n_boot, seed) if "INT4" in e.states else None
    r2 = h_gc2(cells, langs, ("INT4", "FP"), ratios, n_boot, seed) if "INT4" in e.states and "FP" in e.states else None
    r3 = h_gc3(cells, langs, "INT4", r1, ratios, n_boot, seed) if r1 is not None else None
    rules = {"H-Gc1": r1, "H-Gc2": r2, "H-Gc3": r3, "H-Gc4": h_gc4_rows(cells, langs, e.states, e.ratios)}
    verdicts = {"H-Gc1": r1["met"] if r1 else None, "H-Gc2": r2["met"] if r2 else None, "H-Gc3": r3["specific"] if r3 else None}
    return {"rules": rules, "verdicts": verdicts, "reliability": reliability_rows(cells, e), "bands": band_rows(cells, langs, e.states, e.ratios, n_boot, seed)}


def _verdict(v):
    return "not testable" if v is None else ("**met**" if v else "**not met**")


def report_text(res, cells, e, meta):
    r, v = res["rules"], res["verdicts"]
    out = ["# Study G, extension C: the repair near ratio 1 with a larger sample", "",
           "Exploratory in status (devtest sentences, one model, one layer, RTN only); the rules below were fixed before the runs (spec section 11). "
           f"Sample: {e.n_sentences} sentences per language (the reserve sentences no earlier run used), {e.n_controls} matched-control draws, {len(e.languages)} languages, states {', '.join(e.states)}; "
           f"ratios {', '.join(f'{a:g}' for a in e.ratios)}; the primary ratios are {', '.join(f'{a:g}' for a in PRIMARY)} (intervals at 97.5 %, Holm). Units of the rules: the languages; net harm = group minus the mean of the "
           "control draws (nats per token); a negative net harm means that turning the group up lowers the loss.", "",
           "## Verdicts", "", "| hypothesis | rule | result | verdict |", "|---|---|---|---|"]
    if r["H-Gc1"]:
        for a, x in r["H-Gc1"]["per_ratio"].items():
            out.append(f"| H-Gc1 repair at INT4, ratio {a:g} | interval of the median net harm below 0 and at least 19 of 27 negative | median {x['median']:.4f} [{x['lo']:.4f}, {x['hi']:.4f}]; {x['n_negative']} of {x['n']} negative (sign test p = {x['sign_p']:.4f}) | {_verdict(r['H-Gc1']['ratio_met'][a])} |")
        out.append(f"| H-Gc1 (either primary ratio) | | | {_verdict(v['H-Gc1'])} |")
    if r["H-Gc2"]:
        for a, x in r["H-Gc2"]["per_ratio"].items():
            out.append(f"| H-Gc2 INT4 minus FP, ratio {a:g} | interval of the paired difference below 0 | median difference {x['median_diff']:.4f} [{x['lo']:.4f}, {x['hi']:.4f}] | {_verdict(x['ok'])} |")
        out.append(f"| H-Gc2 (every primary ratio) | | | {_verdict(v['H-Gc2'])} |")
    if r["H-Gc3"]:
        for a, x in r["H-Gc3"]["per_ratio"].items():
            out.append(f"| H-Gc3 matched controls at INT4, ratio {a:g} | group-specific only if the control interval is not entirely below 0 | median control harm {x['median']:.4f} [{x['lo']:.4f}, {x['hi']:.4f}] | |")
        out.append(f"| H-Gc3 (group-specific) | H-Gc1 met and controls not falling too | | {_verdict(v['H-Gc3'])} |")
    out += ["", "## H-Gc4: medians by state and ratio (nats per token)", "", "| state | ratio | median net | median group harm | median control harm | languages with net < 0 | languages with alpha* here | lowest median group harm |", "|---|---|---|---|---|---|---|---|"]
    for x in r["H-Gc4"]:
        out.append(f"| {x['state']} | {x['ratio']:g} | {x['median_net']:.4f} | {x['median_group_harm']:.4f} | {x['median_control_harm']:.4f} | {x['n_negative_net']} of {x['n_languages']} | {x['n_alpha_star']} | {'yes' if x['best_ratio_by_median_group_harm'] else ''} |")
    out += ["", "## Reliability of the sample", "",
            f"{e.n_sentences} sentences per language, {e.n_controls} control draws: standard errors of the net harm from the sentences and from the control draws (medians over languages) and the number of languages whose |net harm| exceeds twice the sentence error alone / both errors combined.", "",
            "| state | ratio | median abs net harm | SE sentences | SE control draws | languages clear (sentences) | languages clear (both) |", "|---|---|---|---|---|---|---|"]
    for x in res["reliability"]:
        out.append(f"| {x['state']} | {x['ratio']:g} | {x['median_abs_net']:.4f} | {x['median_se_sentences']:.4f} | {x['median_se_draws']:.4f} | {x['n_clear_sentences']} of {x['n_languages']} | {x['n_clear_both']} of {x['n_languages']} |")
    out += ["", "## Limits", "",
            "- Exploratory (devtest sentences); one model, layer, SAE and quantizer; the effect, if present, is small (a thousandth of a nat per token at ratio 1.1): state it with its size.",
            "- The interval condition of H-Gc1 holds whenever at least 20 of 27 language values are negative and fails with 18 or fewer; with exactly 19 it is borderline and decided by the seeded bootstrap (seed 0, 5000 draws).",
            "- The motivating data are the main-study data (D-074); the rules and ratios were fixed before the data of this extension existed.", ""]
    return "\n".join(out)


def write_outputs(out_dir, res, cells, e, meta):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    keys = sorted(cells)
    with open(out / "cells.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["state", "language", "ratio", "group_harm", "control_harm", "net", "net_lo", "net_hi", "net_draw_sd"])
        for s, l, a in keys:
            c = cells[(s, l, a)]
            w.writerow([s, l, a, c["g"], c["c"], c["net"], float(np.percentile(c["net_s"], 2.5)), float(np.percentile(c["net_s"], 97.5)), float(np.std(c["net_draws"], ddof=1))])
    for name, rows in (("reliability.csv", res["reliability"]), ("optimum.csv", res["rules"]["H-Gc4"]), ("bands.csv", res["bands"])):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "analysis.json").write_text(json.dumps({"verdicts": res["verdicts"], "rules": ga._jsonable({k: v for k, v in res["rules"].items() if k != "H-Gc4"}), "states": e.states, "languages": e.languages,
                                                   "ratios": e.ratios, "n_sentences": e.n_sentences, "n_controls": e.n_controls, "meta": meta}, indent=1) + "\n")
    (out / "report.md").write_text(report_text(res, cells, e, meta))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_runs(dirs, ratios, expect_ids=None, forbid_ids=None, split="eval"):
    """ExtRuns from job directories (func_lm_<split>_<state>.pt, meta_lm_<split>.json). Refuses a job that is not complete, was cut by the time limit or failed a gate, presets with zero-ablation, with other
    ratios or fewer than five control draws, and presets that differ between states. Returns (ExtRuns, {state: {"file", "sha256"}})."""
    import torch
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
        for f in files:
            run = fa.Run.from_file(f)
            presets.add(torch.load(f, weights_only=False)["preset"])
            runs[run.label] = run
            info[run.label] = {"file": f.name, "dir": str(d), "sha256": _sha256(f), "meta_sha256": _sha256(meta_path)}
    if not runs:
        raise FileNotFoundError(f"no func_lm_{split}_<state>.pt in {[str(d) for d in dirs]}")
    if len(presets) != 1:
        raise ValueError(f"the states were run with different presets: {sorted(presets)}")
    got, n_controls, label, zero = gr.parse_preset_full(presets.pop())
    if zero:
        raise ValueError("the preset has zero-ablation: it is not an extension C preset (`--no-zero`)")
    if set(got) != set(float(r) for r in ratios):
        raise ValueError(f"the ratios of the runs {sorted(got)} differ from the extension's {sorted(ratios)}")
    return ExtRuns(runs, ratios, n_controls, label, expect_ids, forbid_ids), info


def main(argv=None):
    import argparse
    import phase1_func_tasks as ft
    ap = argparse.ArgumentParser(description="Study G extension C: analysis of the gainc jobs (rules H-Gc1 to H-Gc4).")
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap draws over sentences (the reliability table)")
    ap.add_argument("--boot-languages", type=int, default=5000, help="bootstrap draws over languages (the rules)"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect-ids", help="comma-separated sentence ids the runs must have (default: the reserve ids of lm_ids(1012, 450, 562, 0))")
    ap.add_argument("--forbid-ids", help="comma-separated ids that must not occur (default: the pilot and evaluation ids of the main study)")
    a = ap.parse_args(argv)
    main_ids = ft.lm_ids(1012, 150, 300, 0)
    expect = [int(x) for x in a.expect_ids.split(",")] if a.expect_ids else ft.lm_ids(1012, 450, 562, 0)["eval"]
    forbid = [int(x) for x in a.forbid_ids.split(",")] if a.forbid_ids else main_ids["eval"] + main_ids["pilot"]
    e, info = load_runs(a.runs, EXTC_RATIOS, expect, forbid)
    cells = build_cells(e, a.n_boot, a.seed)
    res = evaluate(cells, e, a.boot_languages, a.seed)
    write_outputs(a.out, res, cells, e, {"runs": info, "n_boot": a.n_boot, "boot_languages": a.boot_languages, "seed": a.seed})
    print(f"wrote {a.out}: " + ", ".join(f"{k} {_verdict(v)}" for k, v in res["verdicts"].items()))


if __name__ == "__main__":
    main()
