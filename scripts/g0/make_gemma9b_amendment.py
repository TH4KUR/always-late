"""Writes `reports/functional_validation/gemma9b/amendment_A1_three_arm.json`, the dated amendment A1 of the Gemma-2-9B freeze (plan docs/superpowers/plans/2026-10-28-gemma9b-three-arm-confirmation.md, Task 5; spec
docs/superpowers/specs/2026-10-28-gemma9b-three-arm-confirmation-design.md section 5). The freeze of 2026-10-26 (`freeze.json`, predictions P1 to P3b, legacy RTN only) stays as it is; this amendment adds, BEFORE any 9B
number exists, the verbatim text of the new predictions (Q0 to Q3 for Study G across the three quantizer arms, S1 to S4 for Phase 1, and the two verdict functions), the arms and their states, the sha256 of the new code
files and of the unchanged `freeze.json`. A later change is a new dated amendment, never a rewrite (the command refuses to overwrite).

    python make_gemma9b_amendment.py --base-commit <sha of the commit the code was built on>          write the amendment
    python make_gemma9b_amendment.py --verify                                                          check this checkout against it (exit 0 = identical)"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import make_gemma9b_freeze as mf

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "docs/superpowers/specs/2026-10-28-gemma9b-three-arm-confirmation-design.md"
FREEZE = ROOT / "reports/functional_validation/gemma9b/freeze.json"
OUT = ROOT / "reports/functional_validation/gemma9b/amendment_A1_three_arm.json"
CODE = ("scripts/g0/gptq_ext/tokenizer_check.py", "scripts/g0/gptq_ext/ram_guard.py", "scripts/g0/gptq_ext/record_hashes.py", "scripts/g0/phase1_gemma9b_ext.py", "scripts/g0/phase1_gemma9b_compare.py")
START, END = "P1 to P3b (legacy arm)", "(The Q0 to Q3 list"
ARMS = {"tier1": {"legacy": ["FP", "INT8", "INT7", "INT6", "INT5", "INT4"], "rtn-matched": ["INT6", "INT4"], "gptq": ["INT6", "INT4"], "note": "the FP file of the two new arms is the legacy arm's"},
        "tier2": {"legacy": [8, 7, 6, 5, 4, 16, 12, 0], "rtn-matched": [7, 6, 4, 16, 12, 0], "gptq": [7, 6, 4, 16, 12, 0], "note": "bits; 16, 12 and 0 are the legacy rounding floors and the restore control in every arm"}}
RULES = ["no frozen file edited; the freeze of 2026-10-26 is unchanged (its sha256 is recorded here)", "the 2B ratios are not recalibrated", "predictions and thresholds are fixed before any 9B number exists (this file is the proof of order)",
         "a deviation is a new dated amendment that says why", "no post hoc language selection, no new thresholds after seeing numbers"]

sha = mf.sha


def extract_text(spec_path):
    """The block of the spec from the line that starts `P1 to P3b (legacy arm)` (the definitions: tested languages, req(x), the code the numbers come from) to the line that starts `(The Q0 to Q3 list`, verbatim (with a final newline)."""
    lines = Path(spec_path).read_text(encoding="utf-8").splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith(START)]
    ends = [i for i, l in enumerate(lines) if l.startswith(END)]
    if not starts or not ends or ends[0] < starts[0]:
        raise ValueError(f"the spec has no block from a line starting {START!r} to a line starting {END!r}")
    return "\n".join(lines[starts[0]:ends[0] + 1]) + "\n"


def build(spec_path, date, base_commit, root=ROOT, code=CODE, freeze_path=FREEZE):
    text = extract_text(spec_path)
    return {"kind": "gemma9b_amendment", "id": "A1", "decision": "D-089, amendment A1 (the number of the decision entry is assigned when this branch is merged into main)", "created": date, "base_commit": base_commit,
            "spec": str(Path(spec_path).relative_to(root)) if Path(spec_path).is_relative_to(root) else str(spec_path), "text": text, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "arms": ARMS,
            "code_sha256": {rel: sha(Path(root) / rel) for rel in code}, "freeze_sha256": sha(freeze_path), "rules": RULES}


def verify(path, root=ROOT, code=None, freeze_path=FREEZE, spec_path=SPEC):
    """-> the mismatching items of an existing amendment ([] = identical): `text_sha256` (the stored text no longer hashes to the stored hash), `code:<file>`, `freeze.json`, `spec_text` (the spec block differs from the stored text)."""
    a = json.loads(Path(path).read_text(encoding="utf-8"))
    bad = []
    if hashlib.sha256(a["text"].encode("utf-8")).hexdigest() != a["text_sha256"]:
        bad.append("text_sha256")
    bad += [f"code:{rel}" for rel, h in a["code_sha256"].items() if sha(Path(root) / rel) != h]
    if sha(freeze_path) != a["freeze_sha256"]:
        bad.append("freeze.json")
    if spec_path is not None and Path(spec_path).exists() and extract_text(spec_path) != a["text"]:
        bad.append("spec_text")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=str(SPEC)); ap.add_argument("--base-commit", default=""); ap.add_argument("--out", default=str(OUT)); ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--code", nargs="+", default=list(CODE)); ap.add_argument("--freeze", default=str(FREEZE)); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.verify:
        bad = verify(out, root=a.root, freeze_path=a.freeze, spec_path=a.spec)
        if bad:
            raise SystemExit(f"this checkout differs from the amendment: {bad}")
        print("amendment verified: text, code files and the 9B freeze are the frozen ones")
        return 0
    if not a.base_commit:
        ap.error("--base-commit is required unless --verify")
    if out.exists():
        raise SystemExit(f"{out} exists: an amendment is never rewritten (write a new dated amendment)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(a.spec, time.strftime("%Y-%m-%d %H:%M"), a.base_commit, a.root, tuple(a.code), a.freeze), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
