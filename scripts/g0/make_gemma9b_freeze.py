"""Writes reports/functional_validation/gemma9b/freeze.json (handoff 040, S3): the setting, the preset, the 2B ratios and the predictions P1 to P3b with the verdict function, copied VERBATIM from sections 3 and 7
of the Llama plan, with the sha256 of that text. Run once, before any full 9B run; a later change is a dated amendment, never a rewrite.

    python make_gemma9b_freeze.py --flores-sha256 <sha of flores200_dataset.tar.gz> --base-commit <sha of main the code was built on>"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/superpowers/plans/2026-10-26-llama-consistency-check.md"
RATIOS = ROOT / "reports/functional_validation/gain_study/calibration/ratios.json"
OUT = ROOT / "reports/functional_validation/gemma9b/freeze.json"
CODE = ("scripts/g0/phase1_gemma9b_run.py", "scripts/g0/run_gemma9b.sbatch", "scripts/g0/tests_torch/test_phase1_gemma9b_run.py", "reports/g0/phase1_languages.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def section(text, number):
    """The text of `## <number>. ...` up to the next `## ` heading, verbatim."""
    m = re.search(rf"^## {number}\. .*?(?=^## \d+\. |\Z)", text, flags=re.S | re.M)
    if not m:
        raise ValueError(f"section {number} not found")
    return m.group(0).rstrip() + "\n"


def build(flores_sha, base_commit, date):
    plan = PLAN.read_text(encoding="utf-8")
    predictions = section(plan, 3) + "\n" + section(plan, 7)
    ratios = json.loads(RATIOS.read_text())
    return {
        "schema": 1, "kind": "gemma9b_freeze", "decision": "D-089", "created": date, "base_commit_of_main": base_commit,
        "model": "google/gemma-2-9b", "sae_release": "gemma-scope-9b-pt-res-canonical", "sae_id": "layer_20/width_16k/canonical", "layer": 20, "hook": "blocks.20.hook_resid_post", "d_model": 3584,
        "flores": {"archive": "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz", "sha256": flores_sha, "dev_lines": 997, "devtest_lines": 1012},
        "preset": {"name": "Study G MAIN", "configurations_per_language": 37, "configurations_per_state": 999, "ratios_down": ratios["down"], "ratios_up": ratios["up"], "ratios_fine": ratios["fine_up"],
                   "ratios_source": str(RATIOS.relative_to(ROOT)), "ratios_sha256": sha(RATIOS), "recalibrated_on_9b": False, "n_controls": 3, "label": "K64", "n_eval_sentences": 300,
                   "states": ["FP", "INT8", "INT7", "INT6", "INT5", "INT4"], "languages": 27, "group_families": ["K16", "K64", "all"]},
        "predictions": {"source": str(PLAN.relative_to(ROOT)) + " sections 3 and 7", "text": predictions, "text_sha256": hashlib.sha256(predictions.encode("utf-8")).hexdigest()},
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "rules": ["no frozen file edited", "the 2B ratios are not recalibrated", "any deviation is a dated amendment that says why", "no post hoc language selection, no new thresholds after seeing numbers"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flores-sha256", required=True); ap.add_argument("--base-commit", required=True); ap.add_argument("--out", default=str(OUT)); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists: a freeze is never silently rewritten (use a dated amendment, or --force only before any full run)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(a.flores_sha256, a.base_commit, time.strftime("%Y-%m-%d %H:%M")), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
