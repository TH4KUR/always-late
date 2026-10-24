"""Study G, task G1 (design: docs/superpowers/specs/2026-10-24-graded-gain-study-design.md section 6): the baseline of language selectivity in every precision state, before any intervention.

Input: the D-035 per-language response profiles of every state ([language, feature] positive-token counts and activation sums on FLORES devtest, tokens per language), the Phase 1 dev
selectivity (cohort flags, control pool, null cutoff) and the frozen groups manifest. Two views, both descriptive; neither selects a group that is evaluated on devtest (the profiles ARE devtest):
  state view  (per state, devtest): a feature is `own` for language l when it is supported (>= 50 positives at the peak, panel-mean rate >= 1e-5; association_status), has a unique firing-rate peak at l
              and S = max_l r / mean_l r is strictly above the dev null cutoff of the cohort; `other_selective` for another language; `general` when supported but not selective for anybody; `unsupported`
              otherwise. (The dev bootstrap stability of the peak cannot be recomputed from counts and is not applied: this is a looser rule than the cohort's.)
  fixed view  (the FP-dev classes carried through every state): the activation mass of language l (sum over features of the mean activation, zeros included, on l's text) split into the frozen group G_l
              (first K members of the manifest), the rest of l's cohort, the cohort of other languages, the matched-control pool (supported non-selective) and all other features; `g_ratio` = mass of G_l on l in the
              state over the same in FP (what quantization did to the group's activation on its own language); `group_peak_retained` = share of G_l still supported with a unique peak at l; `total_ratio` and `controls_ratio` = the same ratio to FP for the language's whole activation mass and for the matched-control pool; `g_rel` = g_ratio / total_ratio (does the group's activation change more than the language's as a whole).
Pure NumPy on arrays; the command line reads the profile files."""
import numpy as np

import phase1_association_metrics as am

ORDER = ("FP", "INT8", "INT7", "INT6", "INT5", "INT4")
HG7_QUANTITIES = ("share_group", "group_peak_retained")
HG7_REPORTED = ("g_ratio",)
HG7_MIN_LANGUAGES = 24


def state_classes(counts, n_tokens, cutoff, min_positives=am.MIN_POSITIVES, min_panel_rate=am.MIN_PANEL_RATE):
    """Classes of every feature in ONE state from positive counts [L, F] and tokens per language [L]: supported, peak (argmax rate), unique (unique supported peak), S, own [L, F] (selective for
    that language), selective [F], general [F]. A feature is selective for l when supported, uniquely peaked at l and S strictly above `cutoff`."""
    c = np.asarray(counts); n = np.asarray(n_tokens)
    n_lang, n_feat = c.shape
    status = am.association_status(c, n, np.zeros(n_feat, dtype=np.int64), min_positives, min_panel_rate)
    supported = ~(status["inactive"] | status["low_support"])
    rates = c.astype(np.float64) / n.astype(np.float64)[:, None]
    mean = rates.mean(0)
    S = np.where(mean > 0, rates.max(0) / np.maximum(mean, 1e-300), 0.0)
    peak = rates.argmax(0)
    unique = supported & (status["peak_count"] == 1)
    own = (unique & (S > cutoff))[None, :] & (peak[None, :] == np.arange(n_lang)[:, None])
    selective = own.any(0)
    return {"supported": supported, "peak": peak, "unique": unique, "S": S, "own": own, "selective": selective, "general": supported & ~selective}


def labels_for_language(cls, l):
    """Label of every feature for language l: own | other_selective | general | unsupported."""
    lab = np.full(cls["supported"].shape[0], "unsupported", dtype="<U15")
    lab[cls["general"]] = "general"
    lab[cls["selective"] & ~cls["own"][l]] = "other_selective"
    lab[cls["own"][l]] = "own"
    return lab


def baseline_rows(profiles, selectivity, groups, languages, K=64):
    """One row per (language, state). profiles: {state: {"n_tokens", "positive_counts", "activation_sums"}} (FP required); selectivity: {"selective" [F] bool, "controls" feature ids, "null_cutoff"};
    groups: {language: {"all": ranked feature ids of its cohort}} in the order of `languages`; K: size of the frozen group (first K members)."""
    if "FP" not in profiles:
        raise ValueError("the full-precision profile is required")
    if list(groups) != list(languages):
        raise ValueError("the groups are not in the language order of the profiles")
    selective = np.asarray(selectivity["selective"], dtype=bool)
    n_feat = selective.shape[0]
    controls = np.zeros(n_feat, dtype=bool); controls[np.asarray(selectivity["controls"], dtype=np.int64)] = True
    cutoff = float(selectivity["null_cutoff"])
    cohorts = {code: np.asarray(groups[code]["all"], dtype=np.int64) for code in languages}

    def masks(code):
        cohort = np.zeros(n_feat, dtype=bool); cohort[cohorts[code]] = True
        group = np.zeros(n_feat, dtype=bool); group[cohorts[code][:K]] = True
        other = selective & ~cohort
        rest = ~(cohort | other | controls)
        return {"group": group, "cohort_rest": cohort & ~group, "other_selective": other, "controls": controls & ~cohort & ~other, "rest": rest}

    def mean_activation(p, l):
        return np.asarray(p["activation_sums"], dtype=np.float64)[l] / float(np.asarray(p["n_tokens"])[l])

    fp_act = {code: mean_activation(profiles["FP"], l) for l, code in enumerate(languages)}
    fp_mass = {code: float(fp_act[code][masks(code)["group"]].sum()) for code in languages}
    fp_total = {code: float(fp_act[code].sum()) for code in languages}
    fp_controls = {code: float(fp_act[code][masks(code)["controls"]].sum()) for code in languages}
    ratio = lambda x, ref: x / ref if ref > 0 else float("nan")
    rows = []
    for state, p in profiles.items():
        cls = state_classes(p["positive_counts"], p["n_tokens"], cutoff)
        for l, code in enumerate(languages):
            m = masks(code); act = mean_activation(p, l); total = float(act.sum())
            own = cls["own"][l]; cohort = cohorts[code]
            n_overlap = int(own[cohort].sum())
            mass = float(act[m["group"]].sum())
            peaked = cls["unique"] & (cls["peak"] == l)
            with np.errstate(invalid="ignore", divide="ignore"):
                share = lambda key: float(act[m[key]].sum() / total) if total > 0 else float("nan")
                rows.append({"language": code, "state": state, "n_cohort_fp": int(cohort.shape[0]), "n_selective_state": int(own.sum()), "n_overlap": n_overlap,
                             "n_new": int(own.sum()) - n_overlap, "cohort_kept": n_overlap / cohort.shape[0], "group_size": int(m["group"].sum()),
                             "group_peak_retained": float(peaked[np.flatnonzero(m["group"])].mean()), "group_mass": mass,
                             "g_ratio": ratio(mass, fp_mass[code]), "total_ratio": ratio(total, fp_total[code]), "controls_ratio": ratio(float(act[m["controls"]].sum()), fp_controls[code]),
                             "g_rel": ratio(ratio(mass, fp_mass[code]), ratio(total, fp_total[code])),
                             "share_group": share("group"), "share_cohort_rest": share("cohort_rest"), "share_other_selective": share("other_selective"),
                             "share_controls": share("controls"), "share_rest": share("rest"), "n_general_state": int(cls["general"].sum())})
    return rows


def hg7(rows, order=ORDER, need=HG7_MIN_LANGUAGES):
    """Decision rule H-G7 of the design: along the states of `order` present in the rows, the median over languages of `share_group` and of `group_peak_retained` is non-increasing, and in at least
    `need` languages the value in the last state is strictly below the full-precision value. `g_ratio` is reported with the same counts but does not enter `met`."""
    states = [s for s in order if any(r["state"] == s for r in rows)]
    if "FP" not in states or len(states) < 2:
        raise ValueError("H-G7 needs the FP rows and at least one quantized state")
    def column(q, s):
        return {r["language"]: r[q] for r in rows if r["state"] == s}
    medians, ordered, declining = {}, {}, {}
    for q in HG7_QUANTITIES + HG7_REPORTED:
        medians[q] = {s: float(np.nanmedian(list(column(q, s).values()))) for s in states}
        seq = [medians[q][s] for s in states]
        ordered[q] = all(a >= b for a, b in zip(seq, seq[1:]))
        fp, last = column(q, "FP"), column(q, states[-1])
        declining[q] = sum(1 for l in fp if l in last and last[l] < fp[l])
    met = all(ordered[q] and declining[q] >= need for q in HG7_QUANTITIES)
    return {"met": bool(met), "states": states, "medians": medians, "ordered": ordered, "languages_declining_at_last_state": declining, "need": need}


def _clean(x):
    if isinstance(x, float) and x != x:
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _median(rows, state, key):
    v = [r[key] for r in rows if r["state"] == state]
    return float(np.nanmedian(v)) if v else float("nan")


def report_text(rows, res, meta):
    """Markdown report: medians over languages per state, the per-language table of the first and last state, the verdict of H-G7 and the limits."""
    states = res["states"]; first, last = states[0], states[-1]
    langs = list(dict.fromkeys(r["language"] for r in rows))
    by = {(r["language"], r["state"]): r for r in rows}
    f = lambda v, d=3: "-" if v != v else f"{v:.{d}f}"
    out = ["# Study G, G1: baseline of language selectivity in every precision state", "",
           f"Descriptive. Input: the D-035 profiles (FLORES devtest, equal-budget stream; sha256 `{meta['profiles_sha256']}`), the Phase 1 dev selectivity (sha256 `{meta['selectivity_sha256']}`), the frozen groups manifest; group size K = {meta['K']}. "
           "The profiles are devtest: they describe each state and do not select any group that is evaluated on devtest.", "",
           "## Median over the languages, by state", "",
           "| state | features selective for the language (state view) | of its FP-dev cohort still selective | features new to the language | group peak retained | group activation on the language, ratio to FP | the language's whole activation mass, ratio to FP | matched-control pool, ratio to FP | group ratio over whole-language ratio | share of the language's mass carried by the group | by the matched-control pool |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in states:
        out.append(f"| {s} | {f(_median(rows, s, 'n_selective_state'), 1)} | {f(_median(rows, s, 'cohort_kept'))} | {f(_median(rows, s, 'n_new'), 1)} | {f(_median(rows, s, 'group_peak_retained'))} | "
                   f"{f(_median(rows, s, 'g_ratio'))} | {f(_median(rows, s, 'total_ratio'))} | {f(_median(rows, s, 'controls_ratio'))} | {f(_median(rows, s, 'g_rel'))} | {f(_median(rows, s, 'share_group'))} | {f(_median(rows, s, 'share_controls'))} |")
    out += ["", f"## By language: {first} and {last}", "",
            f"| language | cohort size (FP dev) | cohort still selective {first} | {last} | group peak retained {first} | {last} | group activation ratio {last} | group mass share {first} | {last} |", "|---|---|---|---|---|---|---|---|---|"]
    for l in langs:
        a, b = by[(l, first)], by[(l, last)]
        out.append(f"| {l} | {a['n_cohort_fp']} | {f(a['cohort_kept'])} | {f(b['cohort_kept'])} | {f(a['group_peak_retained'])} | {f(b['group_peak_retained'])} | {f(b['g_ratio'])} | {f(a['share_group'])} | {f(b['share_group'])} |")
    d = res["languages_declining_at_last_state"]
    out += ["", "## H-G7 (decision rule of the design)", "",
            f"Verdict: **{'met' if res['met'] else 'not met'}**. Medians over languages non-increasing along {' > '.join(states)} and, in at least {res['need']} languages, the value in {last} strictly below FP, for the share of the language's activation mass carried by the group "
            f"(median ordered: {res['ordered']['share_group']}, languages declining: {d['share_group']}) and for the retained fraction of the group's peak (median ordered: {res['ordered']['group_peak_retained']}, languages declining: {d['group_peak_retained']}). "
            f"Reported, not part of the rule: the ratio of the group's activation to FP (median ordered: {res['ordered']['g_ratio']}, languages declining: {d['g_ratio']}).", "",
            "## Limits", "",
            "- Devtest profiles of the D-035 cohort (equal-budget stream, 27,036 content tokens per language); no confidence intervals: the rows are descriptive.",
            "- The state view applies the support rule and the dev null cutoff of the cohort to each state's own rates; the dev bootstrap stability of the peak is not recomputed, so the state view is looser than the cohort definition.",
            "- `group peak retained` and `cohort still selective` include the baseline loss of the cohort on devtest (only 82 % of the dev cohort reproduces its association on devtest in FP, D-035): compare states with FP, not with 1."]
    return "\n".join(out) + "\n"


def write_outputs(out_dir, rows, res, meta):
    """baseline.csv (one row per language and state), baseline.json (rows, H-G7 result, meta; NaN as null) and report.md."""
    import csv, json
    from pathlib import Path
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "baseline.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "baseline.json").write_text(json.dumps({"meta": meta, "hg7": _clean(res), "rows": _clean(rows)}, indent=1) + "\n")
    (out / "report.md").write_text(report_text(rows, res, meta))


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    import argparse, json
    import torch
    ap = argparse.ArgumentParser(description="Study G, G1: baseline of language selectivity per precision state from the D-035 profiles.")
    ap.add_argument("--profiles", required=True); ap.add_argument("--selectivity", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--states", nargs="*", default=list(ORDER))
    a = ap.parse_args(argv)
    prof = torch.load(a.profiles, weights_only=False); sel = torch.load(a.selectivity, weights_only=False)
    man = json.loads(open(a.manifest).read())
    languages = list(prof["languages"])
    if list(man["languages"]) != languages or list(sel["languages"]) != languages:
        raise ValueError("the profiles, the selectivity file and the manifest list the languages in different orders")
    missing = [s for s in a.states if s not in prof["conditions"]]
    if missing:
        raise ValueError(f"states missing from the profiles: {missing}")
    profiles = {s: {k: prof["conditions"][s][k].numpy() for k in ("n_tokens", "positive_counts", "activation_sums")} for s in a.states}
    selectivity = {"selective": sel["selective"].numpy(), "controls": sel["controls"].numpy(), "null_cutoff": float(sel["null_cutoff"])}
    rows = baseline_rows(profiles, selectivity, {c: man["groups"][c] for c in languages}, languages, K=a.K)
    res = hg7(rows, order=tuple(a.states))
    meta = {"profiles_sha256": _sha256(a.profiles), "selectivity_sha256": _sha256(a.selectivity), "manifest_sha256": _sha256(a.manifest), "K": a.K, "states": a.states, "null_cutoff": selectivity["null_cutoff"]}
    write_outputs(a.out, rows, res, meta)
    print(f"wrote {a.out}: {len(rows)} rows, H-G7 {'met' if res['met'] else 'not met'}")


if __name__ == "__main__":
    main()
