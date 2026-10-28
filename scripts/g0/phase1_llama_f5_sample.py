"""The feature sample of the Llama-3.2-1B F5 check (Stage 5 of docs/superpowers/plans/2026-10-28-survival-and-intervention-publication-rescue.md; addendum reports/survival_effect_rescue/llama/llama_f5_addendum_2026-10-28.md).
Built ONLY from the dev tagging pass of the Llama consistency check (D-087: `llama-tag`, equal-budget stream, the frozen group builder): no outcome of any F5 run enters it. The roles replace the Gemma roles that need
the D-035 association statuses (which Llama does not have): `k64` = the K64 group of the language (the first 64 of its ranked selective list, the same family as F1 to F3) and `selective_other` = the rest of the
language's ranked selective list. Every feature appears once, with its language and its native activity in that language. The frozen runner then keeps at most `--per-cell` features per (language, role) cell in a
seeded hash order (`phase1_func_f5_run.sample_arrays`), exactly as for Gemma (`--per-cell 12 --subset-seed 0`).

    python3 scripts/g0/phase1_llama_f5_sample.py --selectivity <eq/selectivity.pt> --manifest <groups_manifest.json> --out reports/survival_effect_rescue/llama/llama_f5_sample.json"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build_llama_sample(sel, manifest, roster, hashes, k=64):
    """sel: dict with `rates` [L, F], `mean_activation` [L, F], `peak` [F], `selective` [F] (arrays or tensors); manifest: the group manifest (`languages`, `groups[lang]["all"]` = the ranked selective list of the language);
    roster: the language codes in the order of the manifest. Raises ValueError when a feature is listed twice, is not selective or its peak language is not the listed language."""
    langs = list(manifest["languages"])
    if list(roster) != langs:
        raise ValueError("the roster and the manifest have different language orders")
    feats, counts = {}, {}
    for li, lang in enumerate(langs):
        members = [int(x) for x in manifest["groups"][lang]["all"]]; counts[lang] = {"k64": min(k, len(members)), "selective_other": max(0, len(members) - k)}
        for rank, f in enumerate(members):
            if str(f) in feats:
                raise ValueError(f"feature {f} is listed for more than one language")
            if not bool(sel["selective"][f]) or int(sel["peak"][f]) != li:
                raise ValueError(f"feature {f} is listed for {lang} but the tagging pass says selective={bool(sel['selective'][f])}, peak={int(sel['peak'][f])}")
            feats[str(f)] = {"language": lang, "roles": ["k64" if rank < k else "selective_other"], "native": {"FP": [float(sel["rates"][li, f]), float(sel["mean_activation"][li, f])]}, "fp_established": True, "rank": rank}
    return {"schema": 1, "kind": "llama-f5-sample", "languages": langs, "n_features": len(feats), "k": k, "counts": counts, "source": hashes, "features": feats}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    import torch
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--selectivity", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--roster", default=str(ROOT / "reports/g0/phase1_languages.json")); ap.add_argument("--k", type=int, default=64)
    a = ap.parse_args(argv)
    sel = torch.load(a.selectivity, map_location="cpu", weights_only=False); man = json.loads(Path(a.manifest).read_text())
    roster = [l["code"] for l in json.loads(Path(a.roster).read_text())["languages"]]
    s = build_llama_sample(sel, man, roster, {"selectivity_sha256": _sha(a.selectivity), "manifest_sha256": _sha(a.manifest), "roster_sha256": _sha(a.roster)}, k=a.k)
    Path(a.out).write_text(json.dumps(s, separators=(",", ":")) + "\n")
    tot = {r: sum(c[r] for c in s["counts"].values()) for r in ("k64", "selective_other")}
    print("wrote", a.out, "| features", s["n_features"], "| roles", tot, "| per language min/max", min(sum(c.values()) for c in s["counts"].values()), max(sum(c.values()) for c in s["counts"].values()))


if __name__ == "__main__":
    main()
