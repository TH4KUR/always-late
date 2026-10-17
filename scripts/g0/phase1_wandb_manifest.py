"""Manifest entries for wandb_rebuild.py that put the Phase 1 runs under the fixed schema (reports/wandb_schema.md): one run per BOS regime and stream
(`eq` = equal token budget, primary; `all` = all tokens, sensitivity; decision D-015). The files named in the entries are the small result files fetched into
reports/phase1/<regime>/<stream>/ (summary.csv, per_language.csv, meta.json, analysis.json, analysis.md); the per-feature outputs go to Hugging Face (D-012).

  python3 scripts/g0/phase1_wandb_manifest.py add --manifest reports/g0/wandb_manifest.json --name p1-mask --regime mask --slurm 25745 --commit <sha> --tokens-eq N --tokens-all M
  python3 scripts/g0/wandb_rebuild.py rebuild --manifest reports/g0/wandb_manifest.json --only phase1/mask/eq phase1/mask/all"""
import argparse, json
from pathlib import Path

RESULT_FILES = ("per_language.csv", "meta.json", "analysis.json", "analysis.md")


def entries_for_run(name, regime, slurm_job, commit, tokens_eq, tokens_all, root="reports/phase1"):
    out = []
    for stream, tokens in (("eq", tokens_eq), ("all", tokens_all)):
        base = f"{root}/{regime}/{stream}"
        out.append({"name": f"phase1/{regime}/{stream}",
                    "config": {"experiment": "phase1", "cell": f"{name}-{stream}", "stack": "TL", "bos_regime": regime, "n_tokens": tokens, "slurm_job": str(slurm_job),
                               "code_ref": f"phase1_run.py@{commit[:7]}", "stream": stream,
                               "budget_rule": "tokens per language: minimum over languages (D-015)" if stream == "eq" else "all content tokens of every language"},
                    "summary_csv": f"{base}/summary.csv", "files": [f"{base}/{f}" for f in RESULT_FILES], "tags": ["phase1", regime, stream]})
    return out


def add_to_manifest(path, entries):
    """Append entries whose name is not yet present; everything else in the manifest is kept as is."""
    path = Path(path); m = json.loads(path.read_text()); have = {r.get("name") for r in m["runs"]}
    m["runs"] += [e for e in entries if e["name"] not in have]
    path.write_text(json.dumps(m, indent=1) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("cmd", choices=["add"]); ap.add_argument("--manifest", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--regime", required=True); ap.add_argument("--slurm", required=True); ap.add_argument("--commit", required=True)
    ap.add_argument("--tokens-eq", type=int, required=True); ap.add_argument("--tokens-all", type=int, required=True)
    a = ap.parse_args(argv)
    add_to_manifest(a.manifest, entries_for_run(a.name, a.regime, a.slurm, a.commit, a.tokens_eq, a.tokens_all)); print("manifest updated:", a.manifest)


if __name__ == "__main__":
    main()
