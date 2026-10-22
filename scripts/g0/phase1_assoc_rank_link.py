"""A1 (plan Task 8, specified 2026-10-22 before any number was computed): do the features that carry the function keep their language association better than the rest of the language's selective
features? Joins the per-feature D-035 status file (`feature_status.csv`: established in full precision = fp_status `retained`, retained at a precision = status `retained`) with the dev-mass ranking of
each language's selective features (`groups_manifest.json`: groups[lang]['all'] in rank order; the functional groups K16 and K64 are its prefixes).

Per precision and rank bin (1-16, 17-64, 65-256, 257+, fixed in the plan): established and retained features pooled over languages, the fraction with a Wilson 95 % interval; and per precision the
within-language contrast (fraction retained in the top K minus the rest; languages with at least `min_n` established features in both parts) with a percentile interval from a bootstrap over LANGUAGES.
Hypothesis written before looking (H-A1): association is retained at least as well in the top 64 as in the rest, monotone in rank; a reversed or null result is reported as it is. Exploratory and
descriptive: the statuses were measured on devtest, the ranking on dev; nothing here is causal. The pooled counts are checked against the D-035 headline (POOLED rows of language_summary.csv)."""
import csv
import json
from pathlib import Path

import numpy as np

from phase1_func_mc_calibration import wilson

BINS = ((1, 16), (17, 64), (65, 256), (257, None))
BIN_LABELS = [f"{lo}-{hi}" if hi else f"{lo}+" for lo, hi in BINS]
LADDER = ["INT16", "INT12", "INT8", "INT7", "INT6", "INT5", "INT4"]


def bin_of(rank):
    for (lo, hi), label in zip(BINS, BIN_LABELS):
        if rank >= lo and (hi is None or rank <= hi):
            return label
    raise ValueError(f"rank {rank} is below 1")


def rank_lookup(groups):
    """{(language, feature id): rank} from groups[lang]['all'] (feature ids in dev-mass order; rank 1 = the largest dev mass)."""
    return {(code, fid): r for code, g in groups.items() for r, fid in enumerate(g["all"], start=1)}


def read_status(path):
    """The rows of the selective group of feature_status.csv as dicts (language, feature_id as int, condition, fp_status, status)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["group"] == "selective":
                rows.append({"language": r["original_language"], "feature_id": int(r["feature_id"]), "condition": r["condition"], "fp_status": r["fp_status"], "status": r["status"]})
    return rows


def _conditions(rows, conditions):
    present = {r["condition"] for r in rows}
    return list(conditions) if conditions else [c for c in LADDER if c in present]


def _rank_of(rank, r):
    try:
        return rank[(r["language"], r["feature_id"])]
    except KeyError:
        raise KeyError(f"feature ({r['language']}, {r['feature_id']}) is not in the group ranking") from None


def rank_table(rows, rank, conditions=None):
    """Per condition and rank bin: established (fp_status retained), retained (status retained), fraction with a Wilson interval, languages with at least one established feature in the bin."""
    conds = _conditions(rows, conditions)
    acc = {(c, b): {"est": 0, "ret": 0, "langs": set()} for c in conds for b in BIN_LABELS}
    for r in rows:
        if r["condition"] not in conds:
            continue
        b = bin_of(_rank_of(rank, r))
        if r["fp_status"] == "retained":
            a = acc[(r["condition"], b)]
            a["est"] += 1; a["ret"] += r["status"] == "retained"; a["langs"].add(r["language"])
    out = []
    for c in conds:
        for b in BIN_LABELS:
            a = acc[(c, b)]; n = a["est"]
            lo, hi = wilson(a["ret"], n) if n else (None, None)
            out.append({"condition": c, "bin": b, "n_established": n, "n_retained": a["ret"], "fraction": a["ret"] / n if n else None, "lo": lo, "hi": hi, "n_languages": len(a["langs"])})
    return out


def language_contrast(rows, rank, K=64, min_n=3, n_boot=2000, seed=0, conditions=None):
    """Per condition: for every language with at least `min_n` established features in the top K and in the rest, the fraction retained in each part and their difference (top minus rest); the mean
    difference over languages with a percentile interval from a bootstrap over languages (the same draws for every condition), and how many languages retain better in the top / the rest."""
    conds = _conditions(rows, conditions)
    cnt = {}
    for r in rows:
        if r["condition"] not in conds:
            continue
        k = rank_k = _rank_of(rank, r)
        if r["fp_status"] != "retained":
            continue
        part = "top" if rank_k <= K else "rest"
        a = cnt.setdefault((r["condition"], r["language"]), {"top": [0, 0], "rest": [0, 0]})[part]
        a[0] += 1; a[1] += r["status"] == "retained"
    languages = sorted({code for _, code in cnt})
    draws = {}
    out = {}
    for c in conds:
        per = {}
        for code in languages:
            a = cnt.get((c, code))
            if a and a["top"][0] >= min_n and a["rest"][0] >= min_n:
                top, rest = a["top"][1] / a["top"][0], a["rest"][1] / a["rest"][0]
                per[code] = {"top": top, "rest": rest, "diff": top - rest, "n_top": a["top"][0], "n_rest": a["rest"][0]}
        L = len(per)
        if L == 0:
            out[c] = {"per_language": per, "n_languages": 0, "mean_diff": None, "lo": None, "hi": None, "n_top_better": 0, "n_top_worse": 0}
            continue
        d = np.array([v["diff"] for v in per.values()])
        if L not in draws:
            draws[L] = np.random.default_rng(seed).integers(0, L, size=(n_boot, L))
        lo, hi = np.percentile(d[draws[L]].mean(axis=1), [2.5, 97.5])
        out[c] = {"per_language": per, "n_languages": L, "mean_diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "n_top_better": int((d > 0).sum()), "n_top_worse": int((d < 0).sum())}
    return out


def check_against_d035(table, summary_path):
    """The pooled established and retained counts of the rank table must equal the POOLED rows of the D-035 language summary (the headline numbers); raises ValueError otherwise."""
    pooled = {}
    with open(summary_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["group"] == "selective" and r["language"] == "POOLED":
                pooled[r["condition"]] = (int(r["fp_established"]), int(r["retained"]))
    checked = 0
    for c in sorted({r["condition"] for r in table}):
        if c not in pooled:
            continue
        est = sum(r["n_established"] for r in table if r["condition"] == c); ret = sum(r["n_retained"] for r in table if r["condition"] == c)
        if (est, ret) != pooled[c]:
            raise ValueError(f"{c}: the rank table pools {ret} of {est} but the D-035 summary has {pooled[c][1]} of {pooled[c][0]}")
        checked += 1
    if not checked:
        raise ValueError("no condition of the rank table is in the D-035 summary: nothing could be checked")
    return checked


def _fmt(v, f="{:.3f}"):
    return "-" if v is None else f.format(v)


def report_text(table, contrast, K, min_n, n_boot):
    L = [f"# Association retention by dev-mass rank within each language (A1)", "",
         "**Exploratory and descriptive; nothing here is causal.** The statuses are the D-035 statuses (measured on devtest, features selected and ranked on dev). Established = the feature was classed "
         "`retained` in full precision (6,663 of 8,086 selective features); retained at a precision = still `retained` there. Pooled counts reproduce the D-035 headline (checked against its summary file).", "",
         "**Hypothesis written before any number was computed (H-A1):** association is retained at least as well in the top 64 features of a language (the features that carry the function, ranked by dev "
         "mass) as in the rest, monotone in rank, i.e. marginal features lose their association first. A reversed or null result is reported as it is.", "",
         "**Limitation that travels with the numbers:** the rank by dev mass is confounded with the activity level and the selectivity margin of a feature (high-mass features are less likely to fall below a "
         "support threshold in a quantized model), so this describes where association loss concentrates; it does not show that function protects association. The comparison at equal native activity is F4 "
         "(subgroup interventions, a later GPU job).", "",
         "## Fraction retained among the features established in full precision, by rank bin (pooled over languages, Wilson 95 % intervals)", "",
         "| precision | " + " | ".join(BIN_LABELS) + " |", "|---|" + "---|" * len(BIN_LABELS)]
    for c in dict.fromkeys(r["condition"] for r in table):
        cells = []
        for b in BIN_LABELS:
            r = next(x for x in table if x["condition"] == c and x["bin"] == b)
            cells.append("-" if r["fraction"] is None else f"{r['fraction']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] ({r['n_retained']}/{r['n_established']})")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", f"## Within-language contrast: fraction retained in the top {K} minus the rest", "",
          f"Languages with at least {min_n} established features in both parts; mean over languages with a percentile interval from {n_boot} bootstrap draws over languages (seed 0).", "",
          "| precision | languages | mean difference | 95 % interval | top retains better | rest retains better |", "|---|---|---|---|---|---|"]
    for c, d in contrast.items():
        L.append(f"| {c} | {d['n_languages']} | {_fmt(d['mean_diff'], '{:+.3f}')} | " + ("-" if d["lo"] is None else f"[{d['lo']:+.3f}, {d['hi']:+.3f}]") + f" | {d['n_top_better']} | {d['n_top_worse']} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="A1: association retention by dev-mass rank (joins feature_status.csv and the groups manifest).")
    ap.add_argument("--status", required=True, help="feature_status.csv of the D-035 analysis (stream eq)")
    ap.add_argument("--manifest", default=str(root / "reports/functional_validation/groups_manifest.json"))
    ap.add_argument("--d035-summary", default=str(root / "reports/association_retention/results/eq/language_summary.csv"))
    ap.add_argument("--out", required=True); ap.add_argument("--K", type=int, default=64); ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--conditions", nargs="*")
    a = ap.parse_args(argv)
    rows = read_status(a.status); rank = rank_lookup(json.loads(Path(a.manifest).read_text())["groups"])
    table = rank_table(rows, rank, a.conditions)
    checked = check_against_d035(table, a.d035_summary)
    contrast = language_contrast(rows, rank, a.K, a.min_n, a.n_boot, a.seed, a.conditions)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "rank_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "bin", "n_established", "n_retained", "fraction", "lo", "hi", "n_languages"]); w.writeheader(); w.writerows(table)
    with open(out / "language_contrast.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["condition", "n_languages", "mean_diff", "lo", "hi", "n_top_better", "n_top_worse"])
        for c, d in contrast.items():
            w.writerow([c, d["n_languages"], d["mean_diff"], d["lo"], d["hi"], d["n_top_better"], d["n_top_worse"]])
    with open(out / "language_contrast_by_language.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["condition", "language", "top", "rest", "diff", "n_top", "n_rest"])
        for c, d in contrast.items():
            for code, v in d["per_language"].items():
                w.writerow([c, code, v["top"], v["rest"], v["diff"], v["n_top"], v["n_rest"]])
    (out / "report.md").write_text(report_text(table, contrast, a.K, a.min_n, a.n_boot))
    print(f"wrote {out}; pooled counts equal the D-035 summary for {checked} conditions")


if __name__ == "__main__":
    main()
