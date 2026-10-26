#!/bin/bash
# Builds the course submission archive <TeamNameWithNoSpaces>-Final.zip (course guidelines, section 4.1: the write-up, the code as a ZIP archive, and the presentation as a PDF).
# Contents: the report PDF and the presentation PDF at the top level, and the code (every file tracked by git at the committed state HEAD; no untracked or ignored files, no model weights or data) under code/.
# Usage: bash scripts/make_final_archive.sh <output dir> [presentation.pdf]        (the archive holds the committed state; uncommitted changes of tracked files are reported and left out)
set -euo pipefail
OUT=${1:?output directory}; PRES=${2:-}
cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "WARNING: tracked files have uncommitted changes; the archive holds the COMMITTED state HEAD only (these changes are not in it):"; git status --short --untracked-files=no | head; fi
TEAM=AlwaysLate; REPORT=paper/course-report/AlwaysLate-Final-Report.pdf
[ -f "$REPORT" ] || { echo "missing $REPORT"; exit 2; }
if [ -n "$PRES" ] && [ ! -f "$PRES" ]; then echo "missing presentation $PRES"; exit 2; fi
mkdir -p "$OUT"; STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$TEAM-Final/code"
git archive HEAD | tar -x -C "$STAGE/$TEAM-Final/code"
cp "$REPORT" "$STAGE/$TEAM-Final/$TEAM-Final-Report.pdf"
[ -n "$PRES" ] && cp "$PRES" "$STAGE/$TEAM-Final/$TEAM-Final-Presentation.pdf"
cat > "$STAGE/$TEAM-Final/README.txt" <<TXT
$TEAM final submission (ANLP 2026). Commit $(git rev-parse HEAD) of https://github.com/TH4KUR/always-late
  $TEAM-Final-Report.pdf        the write-up (ACL format; source and build files in code/paper/course-report/)
  $TEAM-Final-Presentation.pdf  the presentation for the viva (if included)
  code/                         the repository at the commit above: README.md first; reports/ holds every result with its status, scripts/ the code and its tests, docs/ the designs and plans
No model weights, datasets or large artifacts are included; the links to the private Hugging Face dataset and to the wandb project are in code/README.md.
TXT
rm -f "$OUT/$TEAM-Final.zip"; (cd "$STAGE" && zip -q -r -X "$OUT/$TEAM-Final.zip" "$TEAM-Final")
SIZE=$(du -k "$OUT/$TEAM-Final.zip" | cut -f1); echo "wrote $OUT/$TEAM-Final.zip ($SIZE KB)"
unzip -l "$OUT/$TEAM-Final.zip" | tail -1
