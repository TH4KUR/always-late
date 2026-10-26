#!/bin/bash
# Submits the Gemma-2-9B confirmation jobs (handoff 040) with the right dependencies. Run on the Ada login node from the repo root, after `python scripts/g0/prefetch_gemma9b.py`.
#   export BASE=$HOME/gemma9b REPO=$BASE/repo PY=<python of the torch env>
#   SB="--constraint=<gpu type with >= 40 GB>" bash scripts/g0/submit_gemma9b_all.sh smoke    # S2: 3 languages, FP and INT4 (instrument gate, not a look at results)
#   SB="--constraint=<gpu type with >= 40 GB>" bash scripts/g0/submit_gemma9b_all.sh full     # S4: tagging + manifest, then one job per state (FP 8 7 6 5 4)
# SB: extra sbatch options for every job (partition, GPU count and type, --exclude ...). TIME_STATE: wall time of one state job (default 30:00:00).
# GEMMA9B_N_DEVICES: GPUs the model is split over (default 1). On Ada's 11 GB cards use 4 together with SB="-p u22 --constraint=2080ti --gres=gpu:4" (the QOS allows 4 GPUs per user, so the state jobs then run one after another).
set -euo pipefail
: "${REPO:?export REPO}"; : "${BASE:?export BASE}"
cd "$REPO"; mkdir -p logs
S=scripts/g0/run_gemma9b.sbatch
SB=${SB:-}
TIME_STATE=${TIME_STATE:-30:00:00}
export GEMMA9B_N_DEVICES=${GEMMA9B_N_DEVICES:-1}
case "${1:?smoke or full}" in
  smoke)
    T=$(sbatch --parsable -t 02:00:00 $SB $S g9-smoke-tag tag --languages eng_Latn hin_Deva khm_Khmr --max-sentences 40)
    L=$(sbatch --parsable --dependency=afterok:$T -t 04:00:00 $SB $S g9-smoke-lm lm g9-smoke-tag --precisions FP 4 --languages eng_Latn hin_Deva khm_Khmr --max-sentences 40 --split pilot --panel)
    echo "smoke: tag $T, lm $L (afterok)" ;;
  full)
    [ -f reports/functional_validation/gemma9b/freeze.json ] || { echo "freeze.json missing: the freeze must exist before any full run"; exit 1; }
    T=$(sbatch --parsable -t 06:00:00 $SB $S g9-tag tag)
    echo "tag: $T"
    for P in FP 8 7 6 5 4; do
      J=$(sbatch --parsable --dependency=afterok:$T -t "$TIME_STATE" $SB $S g9-lm-$P lm g9-tag --precisions $P --split eval --n-eval 300 --panel)
      echo "state $P: $J"
    done ;;
  *) echo "usage: smoke | full"; exit 2 ;;
esac
