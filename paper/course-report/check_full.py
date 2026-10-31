"""Run the unmodified official checker, including its optional reference checks."""
from pathlib import Path
import json

from aclpubcheck.formatchecker import Formatter, Error, Warn

HERE = Path(__file__).resolve().parent
out = HERE / "validation/pubcheck-full"
out.mkdir(parents=True, exist_ok=True)
checker = Formatter()
checker.format_check(str(HERE / "main.pdf"), "long",
                     output_dir=str(out), check_references=True)
errors = sum(len(v) for k, v in checker.logs.items() if isinstance(k, Error))
warnings = sum(len(v) for k, v in checker.logs.items() if isinstance(k, Warn))
summary = {"errors_including_parsing": errors, "warnings": warnings,
           "checks_disabled": [], "references_checked": True}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
raise SystemExit(1 if errors else 0)
