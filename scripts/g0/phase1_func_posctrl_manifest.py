"""Positive-control manifest for the Belebele task (plan Task 17, written 2026-10-24 before any code or number; the user's question about the reliability of the null). Pure json, no torch.

Can the T-MC pipeline register an effect of a feature edit on Belebele accuracy at all? The language groups give a null (D-050, D-062); to read that null one needs an edit that SHOULD hurt. The frozen runner reads a
group as groups[l]["all"][:sizes[l][label]] and its controls as controls[l][label][:cap]; as for the K = all manifest (phase1_func_ksens_manifest.py) a manifest that redefines these lets the unchanged runner compute the
contrast. Here, in every language, the edited set is the same LANGUAGE-INDEPENDENT random order of the pool of supported non-selective features (the control pool of the frozen matching: the `controls` group of the D-035
status file), and the three labels are nested prefixes of it: K16 := the first doses[0] (default 256), K64 := the first doses[1] (default 1,024), all := the whole pool. The controls of a label are `n_draws` random sets of
the same size drawn from the selective features of the OTHER languages (a random comparator of the same size). Every table made from a run with this manifest must say what the labels stand for.

Because the frozen runner (phase1_func_run.py: its file is hash-checked against the freeze) and the frozen preset `mc_eval` read the label K64 only, one manifest per dose is written (K64 standing for 256, 1,024 or all
the pool features) and one run is made per dose.

    python3 scripts/g0/phase1_func_posctrl_manifest.py --status <feature_status.csv> --out-dir reports/functional_validation/posctrl --manifest-dir <dir>      (writes the spec, the report and the three manifests)
    python3 scripts/g0/phase1_func_posctrl_manifest.py --from-spec reports/functional_validation/posctrl/posctrl_spec.json --manifest-dir <dir>               (rebuilds them; must equal the recorded hashes)"""
import argparse
import copy
import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ("K16", "K64", "all")
DERIVATION = ("in every language the edited set is the same random order of the pool of supported non-selective features: the label K16 stands for its first doses[0] features, K64 for its first doses[1], all for the whole pool; the controls of "
              "a label are random sets of the same size from the selective features of the other languages; everything else is the source manifest (the matching diagnostic `gap` is emptied)")


def _sha(obj):
    return hashlib.sha256(json.dumps(obj).encode()).hexdigest()


def _rng(*parts):
    return random.Random(int(hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:12], 16))


def read_pool(status_path):
    """The supported non-selective pool: the distinct feature ids of the `controls` group of the D-035 status file (rows of the full-precision condition), sorted; and the sha256 of the file."""
    ids = set()
    with open(status_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["group"] == "controls" and r["condition"] == "FP":
                ids.add(int(r["feature_id"]))
    if not ids:
        raise ValueError("the status file has no control feature (group 'controls', condition FP)")
    return {"ids": sorted(ids), "sha256": hashlib.sha256(Path(status_path).read_bytes()).hexdigest()}


def derive(src, pool, seed=0, doses=(256, 1024), n_draws=3, source_sha256=""):
    """The positive-control manifest derived from a groups manifest (schema of groups_manifest.json) and a pool of feature ids; the input is not modified."""
    pool = list(pool)
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1")
    if len(set(pool)) != len(pool):
        raise ValueError("the pool has duplicate features")
    if len(doses) != 2 or not (0 < doses[0] < doses[1] < len(pool)):
        raise ValueError(f"the doses {tuple(doses)} must be two ascending sizes below the pool size {len(pool)}")
    union = {f for l in src["languages"] for f in src["groups"][l]["all"]}
    if union & set(pool):
        raise ValueError("the pool contains selective features")
    for l in src["languages"]:
        if len(union - set(src["groups"][l]["all"])) < len(pool):
            raise ValueError(f"{l}: fewer selective features of the other languages than the pool size {len(pool)}")
    pool_sha = _sha(sorted(pool)); order = sorted(pool); _rng(seed, "order", pool_sha).shuffle(order)
    out = copy.deepcopy(src); sizes = {"K16": doses[0], "K64": doses[1], "all": len(pool)}
    for l in out["languages"]:
        cand = sorted(union - set(src["groups"][l]["all"]))
        out["groups"][l] = {"all": list(order)}; out["sizes"][l] = dict(sizes); out["controls"][l] = {"gap": {}}
        for label in LABELS:
            out["controls"][l][label] = [sorted(_rng(seed, l, label, r).sample(cand, sizes[label])) for r in range(n_draws)]
    out["kind"] = "posctrl"
    out["posctrl"] = {"source_manifest_sha256": source_sha256, "source": src["source"], "derivation": DERIVATION, "seed": seed, "doses": [doses[0], doses[1], len(pool)], "n_pool": len(pool), "n_selective": len(union),
                      "n_draws": n_draws, "pool_sha256": pool_sha, "order_sha256": _sha(order)}
    return out


def select_dose(m, label):
    """The manifest in which the label K64 stands for the dose `label` (K16, K64 or all): sizes[l]['K64'] := sizes[l][label] and controls[l]['K64'] := controls[l][label], the other labels kept. The frozen held-out preset
    `mc_eval` (and the frozen runner, whose file is hash-checked against the freeze and must not change) reads the label K64 only, so one run per dose is made with this manifest."""
    if label not in LABELS:
        raise ValueError(f"the dose label must be one of {LABELS}")
    out = copy.deepcopy(m)
    for l in out["languages"]:
        out["sizes"][l]["K64"] = m["sizes"][l][label]; out["controls"][l]["K64"] = copy.deepcopy(m["controls"][l][label])
    out["posctrl"]["k64_stands_for"] = label; out["posctrl"]["k64_size"] = m["sizes"][out["languages"][0]][label]
    return out


def report_text(m):
    b = m["posctrl"]
    L = ["# Positive-control manifest for the Belebele task (plan Task 17)", "",
         "**Exploratory; a positive control, not a result about the language groups.** Derived from the frozen `groups_manifest.json` (file sha256 `" + b["source_manifest_sha256"] + "`). In every language the edited set is the same random order of the pool of "
         f"{b['n_pool']} supported non-selective features (sha256 of the sorted ids `{b['pool_sha256']}`); the label K16 stands for its first {b['doses'][0]} features, K64 for its first {b['doses'][1]}, all for the whole pool ({b['doses'][2]}); "
         f"the controls of a label are {b['n_draws']} random sets of the same size from the selective features of the other languages ({b['n_selective']} selective features in all). The unchanged frozen runner computes the contrasts. " +
         ("In THIS manifest the label K64 stands for the dose `" + b["k64_stands_for"] + f"` ({b['k64_size']} features), so that the frozen preset `mc_eval` computes the contrast for it. " if "k64_stands_for" in b else "") +
         "**Every table made from a run with this manifest must say what the labels stand for.** Reason: the null of the language groups on Belebele (D-050, D-062) can only be read if an edit that should hurt the task is shown to hurt it.", "",
         "| language | K16 | K64 | all | control draws per label |", "|---|---|---|---|---|"]
    L += [f"| {l} | {m['sizes'][l]['K16']} | {m['sizes'][l]['K64']} | {m['sizes'][l]['all']} | {b['n_draws']} |" for l in m["languages"]]
    return "\n".join(L) + "\n"


def _write_manifests(m, directory):
    """One manifest per dose (K64 standing for K16, K64 or all) in `directory`; returns {label: sha256 of the file}."""
    d = Path(directory); d.mkdir(parents=True, exist_ok=True); out = {}
    for label in LABELS:
        text = json.dumps(select_dose(m, label), separators=(",", ":")) + "\n"; (d / f"posctrl_{label}_manifest.json").write_text(text); out[label] = hashlib.sha256(text.encode()).hexdigest()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Derive the positive-control manifest for the Belebele task from the frozen groups manifest and the control pool.")
    ap.add_argument("--src", default=str(ROOT / "reports/functional_validation/groups_manifest.json")); ap.add_argument("--status"); ap.add_argument("--from-spec"); ap.add_argument("--out-dir"); ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--doses", type=int, nargs=2, default=[256, 1024]); ap.add_argument("--n-draws", type=int, default=3); a = ap.parse_args(argv)
    raw = Path(a.src).read_bytes(); src = json.loads(raw); src_sha = hashlib.sha256(raw).hexdigest()
    if a.from_spec:
        spec = json.loads(Path(a.from_spec).read_text())
        if spec["source_manifest_sha256"] != src_sha:
            raise ValueError("the source manifest differs from the one the spec was made from")
        m = derive(src, spec["pool"], spec["seed"], tuple(spec["doses"]), spec["n_draws"], src_sha); shas = _write_manifests(m, a.manifest_dir)
        if shas != spec["manifest_sha256"]:
            raise ValueError("a rebuilt manifest differs from the recorded one")
        print("rebuilt the manifests in", a.manifest_dir, "sha256", shas, "equal to the recorded ones")
        return
    if not (a.status and a.out_dir):
        raise ValueError("--status and --out-dir are needed (or --from-spec)")
    pool = read_pool(a.status); m = derive(src, pool["ids"], a.seed, tuple(a.doses), a.n_draws, src_sha); shas = _write_manifests(m, a.manifest_dir)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    spec = {"kind": "posctrl-spec", "source_manifest_sha256": src_sha, "status_sha256": pool["sha256"], "pool": pool["ids"], "pool_sha256": m["posctrl"]["pool_sha256"], "seed": a.seed, "doses": list(a.doses), "n_draws": a.n_draws,
            "manifest_sha256": shas, "derivation": DERIVATION}
    (out / "posctrl_spec.json").write_text(json.dumps(spec, separators=(",", ":")) + "\n"); (out / "posctrl_report.md").write_text(report_text(m))
    print("wrote", out, "| manifests in", a.manifest_dir, shas, "| pool", len(pool["ids"]), "| doses", m["posctrl"]["doses"])


if __name__ == "__main__":
    main()
