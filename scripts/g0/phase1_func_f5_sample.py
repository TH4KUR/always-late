"""The feature sample of the F5 study (plan Task 12; design docs/superpowers/specs/2026-10-23-feature-function-equivalence-design.md section 4): which SAE features are tested for sameness of function across
quantization, and in which role. Roles: `k64` (the K64 language groups of F1 to F3: the features that matter most for their language), `lost@<precision>` and `retained@<precision>` (the F4 subsets: the features
whose language association was lost at that precision and the activity-matched features that kept it). Every feature appears once with all its roles, its language, its D-035 status and its native activity at every
precision of the status file, so that the validity analysis can compare the two survival statuses at equal native activity. Stdlib plus the status reader of phase1_func_f4_groups; no model.

    python3 scripts/g0/phase1_func_f5_sample.py --status <feature_status.csv of D-035, stream eq>      (writes reports/functional_validation/f5/f5_sample.json and f5_sample.md)
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from phase1_func_f4_groups import read_status

ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_sample(groups_manifest, f4_manifests, status_rows, hashes):
    """groups_manifest: the frozen groups manifest; f4_manifests: {condition: F4 manifest}; status_rows: rows of read_status; hashes: recorded as the sources. Raises ValueError on inconsistent inputs."""
    g = groups_manifest; roster = list(g["languages"]); roles = defaultdict(set); lang_of = {}; counts = defaultdict(lambda: defaultdict(int)); f4 = {}

    def add(fid, lang, role):
        if lang_of.setdefault(fid, lang) != lang:
            raise ValueError(f"feature {fid} is listed for the languages {lang_of[fid]} and {lang} (the manifests disagree on its language)")
        if role not in roles[fid]:
            roles[fid].add(role); counts[role][lang] += 1

    for lang in roster:
        for fid in g["groups"][lang]["all"][:g["sizes"][lang]["K64"]]:
            add(int(fid), lang, "k64")
    for cond, m in f4_manifests.items():
        if m["f4"]["condition"] != cond:
            raise ValueError(f"the manifest given for {cond} is the manifest of {m['f4']['condition']}")
        f4[cond] = {}
        for lang in m["f4"]["included"]:
            lost = [int(x) for x in m["groups"][lang]["all"][:m["sizes"][lang]["K64"]]]; draws = [[int(x) for x in d] for d in m["controls"][lang]["K64"]]
            both = set(lost) & {x for d in draws for x in d}
            if both:
                raise ValueError(f"{cond} {lang}: the lost and the retained subsets overlap in {sorted(both)[:5]}")
            for fid in lost:
                add(fid, lang, f"lost@{cond}")
            for fid in sorted({x for d in draws for x in d}):
                add(fid, lang, f"retained@{cond}")
            f4[cond][lang] = {"lost": lost, "retained": draws}
    by_feature = defaultdict(list)
    for r in status_rows:
        by_feature[r["feature_id"]].append(r)
    features, missing = {}, []
    for fid in sorted(roles):
        rows = by_feature.get(fid)
        if not rows:
            missing.append(fid); continue
        langs = {r["language"] for r in rows}
        if langs != {lang_of[fid]}:
            raise ValueError(f"feature {fid}: the status file gives the language {sorted(langs)}, the manifests {lang_of[fid]}")
        fps = {r["fp_status"] for r in rows}
        if len(fps) != 1:
            raise ValueError(f"feature {fid}: the status file has more than one full-precision status {sorted(fps)}")
        features[str(fid)] = {"language": lang_of[fid], "roles": sorted(roles[fid]), "fp_established": fps == {"retained"}, "status": {r["condition"]: r["status"] for r in rows},
                              "native": {r["condition"]: [r["rate"], r["mean"]] for r in rows}}
    if missing:
        raise ValueError(f"{len(missing)} sample features have no row in the status file, for example {missing[:5]}")
    return {"schema": 1, "kind": "f5-sample", "source": hashes, "languages": roster, "n_features": len(features), "n_multi_role": sum(1 for f in features.values() if len(f["roles"]) > 1),
            "counts": {role: dict(sorted(c.items())) for role, c in sorted(counts.items())}, "f4": f4, "features": features}


def report_text(s):
    L = ["# F5 feature sample", "",
         f"{s['n_features']} features, of which {s['n_multi_role']} in more than one role. Roles: `k64` (the K64 language groups), `lost@<precision>` and `retained@<precision>` (the F4 subsets: association lost, and activity-matched retained). "
         "Every feature has its language, its D-035 status and its native activity at every precision of the status file (`f5_sample.json`).", "",
         "| role | languages | features |", "|---|---|---|"]
    for role, c in s["counts"].items():
        L.append(f"| {role} | {len(c)} | {sum(c.values())} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the F5 feature sample from the frozen groups manifest, the F4 manifests and the D-035 status file.")
    ap.add_argument("--status", required=True, help="feature_status.csv of the D-035 analysis (stream eq)")
    ap.add_argument("--groups-manifest", default=str(ROOT / "reports/functional_validation/groups_manifest.json")); ap.add_argument("--f4-dir", default=str(ROOT / "reports/functional_validation/f4"))
    ap.add_argument("--conditions", nargs="*", default=["INT6", "INT5", "INT4"]); ap.add_argument("--out-dir", default=str(ROOT / "reports/functional_validation/f5"))
    a = ap.parse_args(argv)
    g = json.loads(Path(a.groups_manifest).read_text()); f4 = {c: json.loads((Path(a.f4_dir) / f"f4_manifest_{c}.json").read_text()) for c in a.conditions}
    hashes = {"groups": _sha(a.groups_manifest), "f4": {c: _sha(Path(a.f4_dir) / f"f4_manifest_{c}.json") for c in a.conditions}, "status": _sha(a.status)}
    s = build_sample(g, f4, read_status(a.status), hashes)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "f5_sample.json").write_text(json.dumps(s, sort_keys=True, separators=(",", ":")) + "\n"); (out / "f5_sample.md").write_text(report_text(s))
    print("wrote", out, s["n_features"], "features", {r: sum(c.values()) for r, c in s["counts"].items()})


if __name__ == "__main__":
    main()
