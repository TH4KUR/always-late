#!/bin/bash
# GPTQ extension (D-101): ONE job on ONE GPU of Ada (u22, RTX 2080 Ti 11 GB), Gemma-2-2B, jobs offline. Two modes:
#   gate : build the artifacts and run the gates (scripts/g0/gptq_ext/gate_run.py): sbatch -t 02:00:00 --exclude=gnode068,gnode069,gnode076 --export=ALL,REPO=<snapshot>,RECORD=1 scripts/g0/run_gptq.sbatch gptq-gate gate --methods gptq rtn-matched --bits 4
#   wrap : run an OLD entry point with another weight method (scripts/g0/gptq_ext/run_with_weights.py); the entry point's own arguments follow `--`, the token {OUT} stands for the output directory:
#          sbatch ... --export=ALL,REPO=<snapshot>,FROZEN=1 scripts/g0/run_gptq.sbatch gptq-p1 wrap --entry phase1_run:main --weights gptq -- --corpus flores --regime mask --out {OUT} --flores-root /home2/karma/duan-verify/phase1/data --commit <c>
# The calibration pool is read from $BASE/gptq/calibration (token files, not in git) and verified against reports/gptq_extension/calibration_freeze.json. RECORD=1 adds --record-weights-hashes (gate jobs only);
# FROZEN=1 adds --gptq-freeze reports/gptq_extension/artifact_hashes.json (the recorded artifact hashes: a stage job refuses an artifact whose hash differs).
# Read the `GPU 0:` line AND the first lines of the log for `CUDA initialization` (a node whose CUDA fails would run on the CPU: refused below). Artifacts are built in RAM at first use (plan decision 4); /home2 has little free disk.
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH -J gptq
#SBATCH -o /home2/karma/duan-verify/logs/%x-%j.out
set -euo pipefail
NAME=${1:?name}; MODE=${2:?gate|wrap}; shift 2
BASE=/home2/karma/duan-verify
export HF_TOKEN_PATH=$BASE/hf/token
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$BASE/logs"
ls $HF_HOME/hub/models--google--gemma-2-2b/snapshots/*/model-00003-of-00003.safetensors > /dev/null || { echo "model cache missing: run fetch_model.sh on the login node"; exit 3; }
trap 'kill ${MONPID:-0} 2>/dev/null || true' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=$BASE/out/phase1/$NAME; mkdir -p "$OUT"
REPO=${REPO:-$BASE/phase1/repo}; COMMIT=$(cat $REPO/COMMIT)
nvidia-smi -L; date; hostname
nvidia-smi -L | grep -q 'GPU 0' || { echo 'no GPU visible on this node: refusing to run on the CPU'; exit 4; }
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 10 > "$OUT/gpu_mem.log" &
MONPID=$!
CAL="--calibration-freeze $REPO/reports/gptq_extension/calibration_freeze.json --calibration-text $BASE/gptq/calibration"
[ "${RECORD:-0}" = 1 ] && CAL="$CAL --record-weights-hashes"
[ "${FROZEN:-0}" = 1 ] && CAL="$CAL --gptq-freeze $REPO/reports/gptq_extension/artifact_hashes.json"
ARGS=(); for a in "$@"; do ARGS+=("${a//\{OUT\}/$OUT}"); done
cd $REPO/scripts/g0
if [ "$MODE" = gate ]; then
    $BASE/env/bin/python -m gptq_ext.gate_run --out "$OUT" --flores-root $BASE/phase1/data --roster $REPO/reports/g0/phase1_languages.json --commit "$COMMIT" $CAL "${ARGS[@]}"
elif [ "$MODE" = wrap ]; then
    SEP=0; for a in "${ARGS[@]}"; do [ "$a" = "--" ] && SEP=1; done
    [ "$SEP" = 1 ] || { echo "wrap mode needs the separator --"; exit 5; }
    WRAP=(); REST=(); S=0; for a in "${ARGS[@]}"; do if [ "$S" = 0 ] && [ "$a" = "--" ]; then S=1; elif [ "$S" = 0 ]; then WRAP+=("$a"); else REST+=("$a"); fi; done
    $BASE/env/bin/python gptq_ext/run_with_weights.py "${WRAP[@]}" $CAL -- "${REST[@]}"
else
    echo "unknown mode $MODE"; exit 6
fi
date
