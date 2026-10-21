"""D-035 descriptive check, run on Ada after the analysis: change of the target share and of the margin between full precision and each condition among the selective features that are retained
in both (feature_status.csv of analysis-eq). A plain script (no project modules); output quoted in reports/association_retention/interpretation.md."""
import csv, statistics as st
path = "/home2/karma/duan-verify/out/phase1/association-full/analysis-eq/feature_status.csv"
fp = {}; q = {}
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        if r["group"] != "selective": continue
        if r["condition"] == "FP": fp[r["feature_id"]] = r
        else: q.setdefault(r["condition"], {})[r["feature_id"]] = r
def pct(v, p): v = sorted(v); return v[min(len(v) - 1, int(p / 100 * len(v)))]
print("condition n_retained | d_share: p10 median p90 frac(|d|>0.05) | d_margin: p10 median p90 | share FP median -> Q median")
for c in ["INT8", "INT6", "INT5", "INT4", "INT16"]:
    ds, dm, s0, s1 = [], [], [], []
    for fid, r in q[c].items():
        if r["status"] == "retained" and fp[fid]["status"] == "retained":
            a, b = float(fp[fid]["target_share"]), float(r["target_share"]); ds.append(b - a); s0.append(a); s1.append(b)
            dm.append(float(r["margin"]) - float(fp[fid]["margin"]))
    print(c, len(ds), "| %.4f %.4f %.4f %.3f | %.4f %.4f %.4f | %.3f -> %.3f" % (pct(ds, 10), st.median(ds), pct(ds, 90), sum(abs(x) > 0.05 for x in ds) / len(ds), pct(dm, 10), st.median(dm), pct(dm, 90), st.median(s0), st.median(s1)))
