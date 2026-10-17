#!/bin/bash
# Fetch the files of one Phase 1 run from Ada that the analysis needs (about 25 MB per stream; the per-feature outputs stay on Ada and go to Hugging Face).
# Usage: bash scripts/g0/fetch_phase1.sh p1-mask [DEST_ROOT]      -> DEST_ROOT/p1-mask/{eq,all}/{summary.csv,per_language.csv,meta.json,survival.pt,selectivity.pt}
set -euo pipefail
NAME=${1:?run name, e.g. p1-mask}; DEST=${2:-/tmp/claude-1000/phase1_runs}
for stream in eq all; do
  mkdir -p "$DEST/$NAME/$stream"
  for f in summary.csv per_language.csv meta.json survival.pt selectivity.pt; do
    scp -q "karma@ada-gw1:/home2/karma/duan-verify/out/phase1/$NAME/$stream/$f" "$DEST/$NAME/$stream/$f"
  done
done
ls -la "$DEST/$NAME/eq" "$DEST/$NAME/all"
