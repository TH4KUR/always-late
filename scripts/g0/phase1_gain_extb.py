"""Study G, extension B (design: docs/superpowers/specs/2026-10-24-graded-gain-study-design.md section 10 and its note; D-073, D-081): analysis of the runs on groups selected in each precision.

Two sets of runs, each with the main design (eight ratios, 300 evaluation sentences, three control draws, zero-ablation) over FP, INT6 and INT4:
  state-defined  : `gain-sd-FP` (FP manifest at FP), `gain-sd-6` (INT6 manifest at INT6), `gain-sd-4` (INT4 manifest at INT4);
  FP reference   : `gain-sd-FP`, `gain-sd-FPref-6` (FP manifest at INT6), `gain-sd-FPref-4` (FP manifest at INT4): the same selection text at every precision.
Rules (fixed before any result):
  H-Gb1  the overlap (Jaccard) of a language's 64 features between the FP reference and INT4 is smaller than between the FP reference and INT6 in at least 24 of 27 languages;
  H-Gb2  at INT4 the net harm of the state-defined group at the deepest down ratio minus that of the FP reference group at INT4 is positive in at least 18 of 27 languages and its median is positive
         (reported at INT6 as well; per-language intervals from paired sentence draws);
  H-Gb3  H-G1 and H-G3 of the main study hold for the state-defined groups at their own state (the same counts);
  H-Gb4  H-G5a and H-G5b of the main study (amended A1) for the INT4-defined groups at INT4.
Reported, not rules: the table of median net harms of the state-defined group, the FP reference group at every ratio and state."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import phase1_gain_analysis as ga


def h_gb1(rows, need=24):
    """Per language Jaccard(FP reference, INT4) < Jaccard(FP reference, INT6); met when at least `need` of 27 languages (proportionally for another number)."""
    j = {s: {r["language"]: float(r["jaccard"]) for r in rows if r["state"] == s} for s in ("INT6", "INT4")}
    if not j["INT6"] or not j["INT4"]:
        raise ValueError("the overlap table needs rows for INT6 and INT4")
    langs = [l for l in j["INT4"] if l in j["INT6"]]
    count = sum(1 for l in langs if j["INT4"][l] < j["INT6"][l])
    return {"count": count, "n": len(langs), "median_int6": float(np.median([j["INT6"][l] for l in langs])), "median_int4": float(np.median([j["INT4"][l] for l in langs])),
            "met": bool(count >= ga.required(len(langs), need))}


def _same_design(g_sd, g_ref):
    if g_sd.languages != g_ref.languages or g_sd.n_sentences != g_ref.n_sentences or g_sd.down != g_ref.down or g_sd.up != g_ref.up or g_sd.fine_up != g_ref.fine_up:
        raise ValueError("the two sets of runs differ in languages, number of sentences or ratios")


def h_gb2(t_sd, t_ref, g_sd, g_ref, state="INT4"):
    """The state-defined group against the FP reference group at `state` at the deepest down ratio, per language and paired by sentence draws (both tables use the same draws); met when at least 18 of 27
    differences are positive and the median is positive (at any n the median condition follows from the count, since the count threshold is two thirds of n; it is kept because the pre-registered rule states it)."""
    _same_design(g_sd, g_ref)
    d = g_sd.down[-1]
    per = {}
    for l in g_sd.languages:
        a, b = t_sd.get(state, l, d), t_ref.get(state, l, d)
        diff_s = a["net_s"] - b["net_s"]
        lo, hi = float(np.percentile(diff_s, 2.5)), float(np.percentile(diff_s, 97.5))
        per[l] = {"diff": a["net"] - b["net"], "lo": lo, "hi": hi}
    diffs = [v["diff"] for v in per.values()]
    count = sum(1 for x in diffs if x > 0)
    med = float(np.median(diffs))
    return {"state": state, "ratio": d, "n": len(diffs), "count_positive": count, "count_lower_above_zero": sum(1 for v in per.values() if v["lo"] > 0), "median_diff": med,
            "per_language": per, "met": bool(count >= ga.required(len(diffs), 18) and med > 0)}


def compare_rows(t_sd, t_ref, g_sd):
    """Per state and ratio (zero-ablation included): median net harm of the state-defined group and of the FP reference group, the median per-language difference and the number of languages where the
    state-defined group has the larger net harm."""
    rows = []
    for s in g_sd.states:
        for a in sorted([0.0] + list(g_sd.all_ratios)):
            nsd = [t_sd.get(s, l, a)["net"] for l in g_sd.languages]; nref = [t_ref.get(s, l, a)["net"] for l in g_sd.languages]
            diff = [x - y for x, y in zip(nsd, nref)]
            rows.append({"state": s, "ratio": a, "median_net_state_defined": float(np.median(nsd)), "median_net_reference": float(np.median(nref)), "median_difference": float(np.median(diff)),
                         "n_state_defined_larger": sum(1 for x in diff if x > 0), "n_languages": len(diff)})
    return rows


def evaluate(t_sd, t_ref, g_sd, g_ref, overlap, n_boot=2000, seed=0):
    """All rules and the comparison table. H-Gb4 is the repair rule of the main study (`h_g5`) on the state-defined groups; its two parts are separate verdicts."""
    g1 = h_gb1(overlap)
    g2 = {s: h_gb2(t_sd, t_ref, g_sd, g_ref, s) for s in ("INT4", "INT6") if s in g_sd.states and s in g_ref.states}
    g3 = {s: {"h_g1": ga.h_g1(t_sd, g_sd, s)["met"], "h_g3": ga.h_g3(t_sd, g_sd, s)["met"]} for s in g_sd.states}
    g4 = ga.h_g5(t_sd, g_sd, None, seed, n_boot)
    verdicts = {"H-Gb1": g1["met"], "H-Gb2": g2["INT4"]["met"] if "INT4" in g2 else None, "H-Gb3": all(v["h_g1"] and v["h_g3"] for v in g3.values()),
                "H-Gb4a": g4["a"]["met"] if g4.get("testable") else None, "H-Gb4b": g4["b"]["met"] if g4.get("testable") else None}
    return {"rules": {"H-Gb1": g1, "H-Gb2": g2, "H-Gb3": g3, "H-Gb4": g4}, "verdicts": verdicts, "compare": compare_rows(t_sd, t_ref, g_sd)}


def _v(v):
    return "not testable" if v is None else ("**met**" if v else "**not met**")


def report_text(res, t_sd, t_ref, g_sd, g_ref, meta):
    r, v = res["rules"], res["verdicts"]
    n = len(g_sd.languages)
    out = ["# Study G, extension B: groups selected in each precision", "",
           "Exploratory in status (devtest sentences; selection on the 712 sentences outside the 300 evaluation sentences, `reserve` text of the D-035 sidecar; evaluation on the 300 held-out sentences); the rules were fixed "
           "before the runs (spec section 10 and its note). 'State-defined' = the 64 strongest own features of the language in that precision; 'FP reference' = the groups selected the same way at FP, run at every precision "
           f"(the same selection text); {g_sd.n_controls} matched-control draws; {n} languages; {g_sd.n_sentences} sentences per language.", "",
           "## Verdicts", "", "| hypothesis | rule | result | verdict |", "|---|---|---|---|",
           f"| H-Gb1 the contributing features move with precision | Jaccard(FP reference, INT4) < Jaccard(FP reference, INT6) in at least {ga.required(n, 24)} of {n} | {r['H-Gb1']['count']} of {r['H-Gb1']['n']}; median {r['H-Gb1']['median_int4']:.3f} (INT4) against {r['H-Gb1']['median_int6']:.3f} (INT6) | {_v(v['H-Gb1'])} |"]
    for s, x in r["H-Gb2"].items():
        out.append(f"| H-Gb2 state-defined minus FP reference, net harm at ratio {x['ratio']:.3g}, {s} | positive in at least {ga.required(n, 18)} and the median positive | {x['count_positive']} of {x['n']}; median difference {x['median_diff']:.4f} ({x['count_lower_above_zero']} with the lower bound above 0) | {_v(x['met'])} |")
    for s, x in r["H-Gb3"].items():
        out.append(f"| H-Gb3 dose-response of the state-defined group, {s} | H-G1 and H-G3 as in the main study | H-G1 {x['h_g1']}, H-G3 {x['h_g3']} | {_v(x['h_g1'] and x['h_g3'])} |")
    out.append(f"| H-Gb3 (all states) | | | {_v(v['H-Gb3'])} |")
    g4 = r["H-Gb4"]
    if g4.get("testable"):
        a_txt = "; ".join(f"x{a:g}: median control harm {x['median']:.4f} [{x['lo']:.4f}, {x['hi']:.4f}]" for a, x in g4["a"]["per_ratio"].items())
        out.append(f"| H-Gb4a generic repair at INT4, state-defined groups | rule H-G5a | {a_txt} | {_v(v['H-Gb4a'])} |")
        out.append(f"| H-Gb4b group-specific repair at INT4, state-defined groups | rule H-G5b | {g4['b']['count']} of {g4['b']['n']} | {_v(v['H-Gb4b'])} |")
    out += ["", "## Median net harm by state and ratio: state-defined group against the FP reference group (nats per token)", "",
            "| state | ratio | state-defined | FP reference | median difference | languages where the state-defined group is larger |", "|---|---|---|---|---|---|"]
    for x in res["compare"]:
        out.append(f"| {x['state']} | {x['ratio']:.3g} | {x['median_net_state_defined']:.4f} | {x['median_net_reference']:.4f} | {x['median_difference']:+.4f} | {x['n_state_defined_larger']} of {x['n_languages']} |")
    out += ["", "## Limits", "",
            "- Exploratory (devtest sentences); one model, layer, SAE and quantizer; the universe of selectable features is the 10,946 supported on FP dev; the state view has no bootstrap stability; selection and evaluation texts are disjoint devtest sentences, not independent samples of a population.",
            "- The comparison with the main study (groups selected on dev at FP) mixes selection state and selection text, which is why the FP reference run exists.", ""]
    return "\n".join(out)


def write_outputs(out_dir, res, t_sd, t_ref, g_sd, g_ref, meta):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "compare.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res["compare"][0])); w.writeheader(); w.writerows(res["compare"])
    (out / "analysis.json").write_text(json.dumps({"verdicts": res["verdicts"], "rules": ga._jsonable(res["rules"]), "states": g_sd.states, "languages": g_sd.languages, "meta": meta}, indent=1) + "\n")
    (out / "report.md").write_text(report_text(res, t_sd, t_ref, g_sd, g_ref, meta))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Study G extension B: analysis of the state-defined and FP-reference runs (rules H-Gb1 to H-Gb4).")
    ap.add_argument("--state-defined", nargs="+", required=True, help="job directories gain-sd-FP gain-sd-6 gain-sd-4"); ap.add_argument("--reference", nargs="+", required=True, help="job directories gain-sd-FP gain-sd-FPref-6 gain-sd-FPref-4")
    ap.add_argument("--ratios-json", required=True); ap.add_argument("--overlap", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    g_sd, info_sd = ga.load_runs(a.state_defined, a.ratios_json); g_ref, info_ref = ga.load_runs(a.reference, a.ratios_json)
    t_sd, t_ref = ga.build_table(g_sd, a.n_boot, a.seed), ga.build_table(g_ref, a.n_boot, a.seed)
    with open(a.overlap) as fh:
        overlap = list(csv.DictReader(fh))
    res = evaluate(t_sd, t_ref, g_sd, g_ref, overlap, a.n_boot, a.seed)
    write_outputs(a.out, res, t_sd, t_ref, g_sd, g_ref, {"state_defined_runs": info_sd, "reference_runs": info_ref, "overlap_sha256": _sha256(a.overlap), "n_boot": a.n_boot, "seed": a.seed})
    print(f"wrote {a.out}: " + ", ".join(f"{k} {_v(x)}" for k, x in res["verdicts"].items()))


if __name__ == "__main__":
    main()
