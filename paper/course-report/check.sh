#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p validation/pubcheck
uvx --from git+https://github.com/acl-org/aclpubcheck@f626497051df8689e990fac36ba9f42cd630e31c \
  aclpubcheck --paper_type long --output-dir validation/pubcheck AlwaysLate-Final-Report.pdf \
  2>&1 | tee validation/pubcheck-console.txt
# The upstream process status is insufficient: require its report to be empty.
python3 - <<'PY'
from pathlib import Path
import json
report = Path('validation/pubcheck/errors-AlwaysLate-Final-Report.json')
findings = json.loads(report.read_text())
if findings:
    raise SystemExit('Official checker recorded findings; inspect ' + str(report))
print('Verified: official checker JSON contains no findings.')
PY
