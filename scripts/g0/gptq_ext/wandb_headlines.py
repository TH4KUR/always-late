"""Headline values of the wandb extension (design reports/wandb_schema_v2_gptq_extension.md, D-115): one flat dictionary per run, every value derived from a COMMITTED result file (CSV or JSON), per arm with the
suffixes _legacy, _matched, _gptq. The v2 builder reads headline values only from committed JSON; this module writes `reports/gptq_extension/wandb_headlines.json` so that the extension runs can point into it.
A test regenerates the file byte for byte and pins values to the decision entries D-104 to D-111.

  python3 scripts/g0/gptq_ext/wandb_headlines.py [--root DIR] [--out FILE]      (writes the file; without --out only prints a summary)"""
import argparse
import csv
import json
import statistics
from pathlib import Path

ARMS = ("legacy", "matched", "gptq")
SLUGS = ["gptq-ext-stage1-survival-loss", "gptq-ext-transfer-main", "gptq-ext-transfer-external", "gptq-ext-loss-task-rescue", "gptq-ext-belebele", "gptq-ext-f5", "gptq-ext-study-g"]
GX = "reports/gptq_extension/results/"
FV = "reports/functional_validation/"
CLAIM_COL = {"legacy": "legacy RTN", "matched": "matched RTN", "gptq": "GPTQ"}
DOWN = 0.08838834764831845


def rows(root, rel):
    with open(Path(root) / rel, newline="") as f:
        return list(csv.DictReader(f))


def jload(root, rel):
    return json.loads((Path(root) / rel).read_text())


def arms(d):
    """{arm: value} -> {name_arm: value} helper is applied by the caller; this keeps the suffix rule in one place."""
    return {f"{a}": v for a, v in d.items()}


def put(out, base, per_arm):
    for a in ARMS:
        out[f"{base}_{a}"] = per_arm[a]


def mean(xs):
    return float(sum(xs) / len(xs))


def stage1(root):
    paired = rows(root, GX + "stage1/stage1_paired_per_language.csv"); by = {}
    for r in paired:                                                                    # legacy and matched from 'matched minus legacy' (b, a), GPTQ from 'GPTQ minus matched' (a)
        d = by.setdefault((r["language"], r["bits"]), {})
        if r["contrast"] == "matched RTN minus legacy RTN":
            d["legacy"], d["matched"] = r, r
        elif r["contrast"] == "GPTQ minus matched RTN":
            d["gptq"] = r
    langs = sorted({k[0] for k in by}); out = {}
    def col(arm, lang, bits, key):
        r = by[(lang, bits)][arm]
        if arm == "legacy":
            return float(r[key + "_b"])
        return float(r[key + "_a"])
    for key, name in (("survived_pct", "survival_mean"), ("nll_delta", "loss_increase")):
        for bits in ("6", "4"):
            put(out, f"{name}_INT{bits}", {a: mean([col(a, l, bits, key) for l in langs]) for a in ARMS})
    holm = {"legacy": jload(root, "reports/phase1/mask/eq/analysis.json")["holm"], "matched": jload(root, GX + "stage1/analysis_rtnm/analysis.json")["holm"], "gptq": jload(root, GX + "stage1/analysis_gptq/analysis.json")["holm"]}
    for t, k in (("T1", "T1_int7"), ("T2", "T2_int6"), ("T3", "T3_tier")):
        put(out, f"holm_{t}", {a: float(holm[a][k]) for a in ARMS})
    return out


def transfer(root, sub, external=False):
    claims = rows(root, GX + f"{sub}/transfer_claims.csv"); feat = rows(root, GX + f"{sub}/transfer_feature_level.csv"); out = {}
    for scope, name in (("overall", "utility_overall"), ("INT4", "utility_INT4"), ("INT6", "utility_INT6")):
        r = [x for x in claims if x["claim"] == "UTILITY" and x["scope"] == scope][0]
        put(out, name, {a: r[CLAIM_COL[a]] for a in ARMS})
    for state, metric, name in (("INT4", "error_over_floor", "error_over_floor_INT4"), ("INT6", "error_over_floor", "error_over_floor_INT6"), ("INT4", "cosine", "cosine_INT4")):
        r = [x for x in feat if x["state"] == state and x["metric"] == metric][0]
        put(out, name, {a: float(r[CLAIM_COL[a]]) for a in ARMS})
    if external:
        paths = {"legacy": "reports/intervention_transfer/results/external/predictive_validation.csv", "matched": GX + "external/analysis_rtnm/predictive_validation.csv", "gptq": GX + "external/analysis_gptq/predictive_validation.csv"}
        for state in ("INT4", "INT6"):
            vals = {}
            for a in ARMS:
                r = [x for x in rows(root, paths[a]) if x["state"] == state and x["variant"] == "eligible" and x["model"] == "A+B"][0]
                vals[a] = float(r["improvement_diagnostic"])
            put(out, f"heldout_improvement_{state}", vals)
    return out


def loss_task_rescue(root):
    ret = {"legacy": FV + "results/eval/retention.csv", "matched": GX + "stage3/analysis_rtnm_eval/retention.csv", "gptq": GX + "stage3/analysis_gptq_eval/retention.csv"}
    resc = {"legacy": FV + "results/rescue/analysis.json", "matched": GX + "stage3/analysis_rtnm_rescue/analysis.json", "gptq": GX + "stage3/analysis_gptq_rescue/analysis.json"}
    out = {}; R = {a: rows(root, ret[a]) for a in ARMS}; langs = {l["code"] for l in jload(root, "reports/g0/phase1_languages.json")["languages"]}      # the files also hold aggregate rows (MACRO_MEAN_OF_RATIOS, POOLED_RATIO)
    put(out, "retention_median_INT4", {a: statistics.median(float(r["ratio"]) for r in R[a] if r["precision"] == "INT4" and r["language"] in langs) for a in ARMS})
    for s in ("INT4", "INT6"):
        put(out, f"retention_mean_{s}", {a: float([r for r in R[a] if r["precision"] == s and r["language"] == "MACRO_MEAN_OF_RATIOS"][0]["ratio"]) for a in ARMS})
    J = {a: jload(root, resc[a]) for a in ARMS}
    put(out, "rescue_group_share_INT4", {a: float(J[a]["rescue"]["INT4"]["macro_fraction"]["group"]["point"]) for a in ARMS})
    put(out, "rescue_residual_share_INT4", {a: float(J[a]["rescue"]["INT4"]["macro_fraction"]["full"]["point"]) for a in ARMS})
    return out


def belebele(root):
    acc = {"legacy": FV + "results/mc_eval/accuracy.csv", "matched": GX + "stage3/analysis_rtnm_mc/accuracy.csv", "gptq": GX + "stage3/analysis_gptq_mc/accuracy.csv"}
    ana = {"legacy": FV + "results/mc_eval/analysis.json", "matched": GX + "stage3/analysis_rtnm_mc/analysis.json", "gptq": GX + "stage3/analysis_gptq_mc/analysis.json"}
    out = {}
    for s in ("INT4", "INT6"):
        put(out, f"accuracy_noedit_{s}", {a: mean([float(r["acc_none"]) for r in rows(root, acc[a]) if r["precision"] == s]) for a in ARMS})
    put(out, "group_effect_INT4", {a: float(jload(root, ana[a])["macro"]["INT4"]["point"]) for a in ARMS})
    return out


def f5(root):
    paths = {"legacy": FV + "results/f5/f5_results.json", "matched": GX + "f5/rtnm/f5_results.json", "gptq": GX + "f5/gptq/f5_results.json"}
    J = {a: jload(root, paths[a]) for a in ARMS}; out = {}
    for s in ("INT6", "INT4"):
        put(out, f"w1_median_{s}", {a: float(J[a]["tables"][s]["all"]["W1"]["median"]) for a in ARMS})
        put(out, f"spearman_r1_w1_{s}", {a: float(J[a]["validity"]["spearman"][s]["rho"]) for a in ARMS})
    put(out, "share_at_floor_INT4", {a: float(J[a]["tables"]["INT4"]["all"]["W1_above_floor"]["share"]) for a in ARMS})
    put(out, "s2_auroc", {a: float(J[a]["validity"]["statements"]["S2"]["auroc"]) for a in ARMS})
    return out


def study_g(root):
    cur = {"legacy": FV + "gain_study/results/curves.csv", "matched": GX + "gain/rtnm/curves.csv", "gptq": GX + "gain/gptq/curves.csv"}
    alpha = {"legacy": FV + "gain_study/results/alpha_star.csv", "matched": GX + "gain/rtnm/alpha_star.csv", "gptq": GX + "gain/gptq/alpha_star.csv"}
    ver = {"legacy": FV + "gain_study/results/analysis.json", "matched": GX + "gain/rtnm/analysis.json", "gptq": GX + "gain/gptq/analysis.json"}; out = {}
    for s in ("INT4", "INT6"):
        put(out, f"net_harm_x0088_{s}", {a: statistics.median(float(r["net"]) for r in rows(root, cur[a]) if r["state"] == s and abs(float(r["ratio"]) - DOWN) < 1e-6) for a in ARMS})
    put(out, "alpha_above_1_INT4", {a: sum(1 for r in rows(root, alpha[a]) if r["state"] == "INT4" and float(r["alpha_star"]) > 1.0) for a in ARMS})
    put(out, "hg4", {a: bool(jload(root, ver[a])["verdicts"]["H-G4"]) for a in ARMS})
    return out


def build(root):
    return {SLUGS[0]: stage1(root), SLUGS[1]: transfer(root, "stage2"), SLUGS[2]: transfer(root, "external", external=True), SLUGS[3]: loss_task_rescue(root), SLUGS[4]: belebele(root), SLUGS[5]: f5(root), SLUGS[6]: study_g(root)}


def render(h):
    return json.dumps(h, indent=1, sort_keys=True) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--root", default=str(Path(__file__).resolve().parents[3])); ap.add_argument("--out"); a = ap.parse_args(argv)
    h = build(a.root)
    if a.out:
        Path(a.out).write_text(render(h)); print("wrote", a.out)
    print({s: len(d) for s, d in h.items()})


if __name__ == "__main__":
    main()
