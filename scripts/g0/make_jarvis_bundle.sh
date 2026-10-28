#!/bin/bash
# Pack what the Jarvis machine needs, without pushing anything: (1) the committed repository (git archive of HEAD under always-late/, plus a COMMIT file with the full sha), (2) the calibration pool token files of the GPTQ
# extension (gptq/calibration/...), (3) SHA256SUMS of both. Prints the commands that copy and unpack them on the Jarvis machine.
#
#   bash scripts/g0/make_jarvis_bundle.sh <outdir> --pool-from <dir with the pool's token files>
#   bash scripts/g0/make_jarvis_bundle.sh <outdir> --pool-from-ada           copies /home2/karma/duan-verify/gptq/calibration from Ada over ssh (about 1 MB; run it only when the user agrees)
#
# Options: --verify-python <python with torch>  also checks the pool against reports/gptq_extension/calibration_freeze.json (gptq_ext.calibration.load_pool) before packing.
# Refuses: uncommitted or untracked files (the code and amendment A1 must be committed: the bundle IS the frozen code), a missing pool source or option, an existing output (never overwrites).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../.." && pwd)}
die() { echo "ERROR: $*" >&2; exit 1; }

OUT=${1:-}; [ -n "$OUT" ] || die "usage: make_jarvis_bundle.sh <outdir> --pool-from <dir> | --pool-from-ada [--verify-python <python>]"
shift
POOL_FROM=""; FROM_ADA=0; VERIFY_PY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --pool-from) POOL_FROM=${2:?--pool-from needs a directory}; shift 2 ;;
        --pool-from-ada) FROM_ADA=1; shift ;;
        --verify-python) VERIFY_PY=${2:?--verify-python needs a python}; shift 2 ;;
        *) die "unknown option $1" ;;
    esac
done
[ -n "$POOL_FROM" ] || [ "$FROM_ADA" = 1 ] || die "give the pool: --pool-from <dir> or --pool-from-ada"
[ -z "$POOL_FROM" ] || [ -d "$POOL_FROM" ] || die "the pool source $POOL_FROM is not a directory"

[ -z "$(git -C "$REPO" status --porcelain)" ] || die "the repository has uncommitted or untracked files (git status): commit them first, the bundle must be the frozen code"
HEAD_SHA=$(git -C "$REPO" rev-parse HEAD)
REPO_TAR=$OUT/always-late-gemma9b.tar.gz; POOL_TAR=$OUT/gptq_calibration_pool.tar.gz; SUMS=$OUT/SHA256SUMS
for f in "$REPO_TAR" "$POOL_TAR" "$SUMS"; do [ ! -e "$f" ] || die "$f exists: never overwritten (choose a new output directory)"; done
mkdir -p "$OUT"

STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
git -C "$REPO" archive --format=tar --prefix=always-late/ HEAD | tar -x -C "$STAGE"
echo "$HEAD_SHA" > "$STAGE/always-late/COMMIT"
tar -czf "$REPO_TAR" -C "$STAGE" always-late

if [ "$FROM_ADA" = 1 ]; then
    ssh -o HostKeyAlias=ada-gw1 -o BatchMode=yes karma@10.4.24.30 "cd /home2/karma/duan-verify/gptq && tar czf - calibration" > "$STAGE/pool.tar.gz" || die "could not copy the pool from Ada (ssh failed)"
    mkdir -p "$STAGE/pool_unpacked"; tar -xzf "$STAGE/pool.tar.gz" -C "$STAGE/pool_unpacked" || die "the pool copied from Ada is not a valid tar.gz"
    POOL_FROM=$STAGE/pool_unpacked/calibration
fi
if [ -n "$VERIFY_PY" ]; then
    ( cd "$REPO/scripts/g0" && "$VERIFY_PY" - "$REPO/reports/gptq_extension/calibration_freeze.json" "$POOL_FROM" <<'PYEOF' ) || die "the pool does not match calibration_freeze.json"
import json, sys
from gptq_ext import calibration as cal
freeze = json.load(open(sys.argv[1])); pool = cal.load_pool(freeze, sys.argv[2]); sha = cal.pool_sha256(pool)
assert sha == freeze["pool_sha256"], (sha, freeze["pool_sha256"]); print("pool verified:", sha[:16])
PYEOF
fi
[ -n "$(ls -A "$POOL_FROM")" ] || die "the pool source $POOL_FROM is empty (pool)"
mkdir -p "$STAGE/pool_layout/gptq"; rm -rf "$STAGE/pool_layout/gptq/calibration"; cp -R "$POOL_FROM" "$STAGE/pool_layout/gptq/calibration"
tar -czf "$POOL_TAR" -C "$STAGE/pool_layout" gptq

( cd "$OUT" && sha256sum always-late-gemma9b.tar.gz gptq_calibration_pool.tar.gz > SHA256SUMS )

cat <<EOF
Bundle written to $OUT (commit $HEAD_SHA):
$(cd "$OUT" && ls -l always-late-gemma9b.tar.gz gptq_calibration_pool.tar.gz SHA256SUMS)

On this machine (copy; the Jarvis ssh line is the one the user was given):
  scp -o StrictHostKeyChecking=no $OUT/always-late-gemma9b.tar.gz $OUT/gptq_calibration_pool.tar.gz $OUT/SHA256SUMS ubuntu@<jarvis ip>:~/

On the Jarvis machine (in its home directory: BASE is the parent of the repository):
  sha256sum -c SHA256SUMS
  tar xzf always-late-gemma9b.tar.gz
  tar xzf gptq_calibration_pool.tar.gz          # creates gptq/calibration next to always-late/ (POOL_DIR default)
  cd always-late && HF_TOKEN=hf_xxx bash scripts/g0/jarvis_run_all.sh        # DRY_RUN=1 first prints the plan; follow it with: tail -f ../run_all.log
EOF
