"""K = all sensitivity manifest for the Belebele task (plan Task 7d, D-047, Q20; written 2026-10-22 before the T-MC freeze). Pure json, no torch.

The design (section 6) asks for the K sensitivity to be reported with a null, and the held-out Belebele preset `mc_eval` holds the group size K64 only. The frozen runner reads a group as
groups[l]["all"][:sizes[l][label]] and its controls as controls[l][label][:cap], so a manifest in which the LABEL K64 stands for the whole selective group (sizes[l]["K64"] := sizes[l]["all"], controls[l]["K64"] :=
controls[l]["all"], the three matched draws of the full size) makes the unchanged runner compute the K = all contrast under the preset's own configuration set (zero, three controls, sham, resample). Same device as the F4
manifests (phase1_func_f4_groups.py): no frozen file is changed. Every table made from such a run must say that K64 stands for the whole group.

    python3 scripts/g0/phase1_func_ksens_manifest.py      (writes reports/functional_validation/ksens/ksens_manifest.json and ksens_report.md from the frozen groups_manifest.json)
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DERIVATION = "the label K64 stands for the whole selective group: sizes[l]['K64'] := sizes[l]['all'] and controls[l]['K64'] := controls[l]['all'] (the matched draws of the full size); everything else is the source manifest"


def derive(src, source_file_sha256):
    """The K = all manifest derived from a groups manifest (schema of groups_manifest.json); the input is not modified."""
    out = copy.deepcopy(src); equal = []
    for l in out["languages"]:
        size = out["sizes"][l].get("all"); draws = out["controls"][l].get("all")
        if size is None:
            raise ValueError(f"{l}: the manifest has no size for the whole group (label 'all')")
        if len(out["groups"][l]["all"]) != size:
            raise ValueError(f"{l}: the size of the whole group ({size}) does not match the list of its features ({len(out['groups'][l]['all'])})")
        if not draws:
            raise ValueError(f"{l}: no matched control draw of the size of the whole group")
        if any(len(d) != size for d in draws):
            raise ValueError(f"{l}: a control draw of the whole group is not of the size of the group ({size})")
        if out["sizes"][l]["K64"] == size:
            equal.append(l)
        out["sizes"][l]["K64"] = size
        out["controls"][l]["K64"] = copy.deepcopy(draws)
    out["kind"] = "ksens-all"
    out["ksens"] = {"source_file_sha256": source_file_sha256, "source": src["source"], "derivation": DERIVATION, "k64_equals_all": equal,
                    "sizes": {l: out["sizes"][l]["all"] for l in out["languages"]}, "n_draws": {l: len(out["controls"][l]["all"]) for l in out["languages"]}}
    return out


def report_text(m):
    k = m["ksens"]
    L = ["# K = all sensitivity manifest for the Belebele task", "",
         "**Exploratory.** Derived from the frozen `groups_manifest.json` (file sha256 `" + k["source_file_sha256"] + "`): in this manifest the label K64 stands for the whole selective group of each language, with its matched control draws "
         "of the full size; the unchanged frozen runner then computes the K = all contrast under the preset's own configuration set. **Every table made from a run with this manifest must say that K64 stands for the whole group.** "
         "Reason: the design (section 6) asks for the K sensitivity to be reported with a null, and `mc_eval` holds K64 only (D-047, Q20).", "",
         "| language | features in the group | matched control draws |", "|---|---|---|"]
    L += [f"| {l} | {k['sizes'][l]} | {k['n_draws'][l]} |" for l in m["languages"]]
    if k["k64_equals_all"]:
        L += ["", "Languages whose K64 group already was the whole group (the two coincide): " + ", ".join(k["k64_equals_all"]) + "."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Derive the K = all sensitivity manifest from the frozen groups manifest.")
    ap.add_argument("--src", default=str(ROOT / "reports/functional_validation/groups_manifest.json")); ap.add_argument("--out-dir", default=str(ROOT / "reports/functional_validation/ksens"))
    a = ap.parse_args(argv)
    raw = Path(a.src).read_bytes(); m = derive(json.loads(raw), hashlib.sha256(raw).hexdigest())
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "ksens_manifest.json").write_text(json.dumps(m, separators=(",", ":")) + "\n"); (out / "ksens_report.md").write_text(report_text(m))
    print("wrote", out, "languages", len(m["languages"]), "k64 equals all:", m["ksens"]["k64_equals_all"])


if __name__ == "__main__":
    main()
