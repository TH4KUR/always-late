#!/bin/bash
# Paired arithmetic-precision audit (task 2 of the publication recovery plan; protocol reports/intervention_transfer/revision_R5_precision_audit.json) on ONE GPU of Ada (u22, RTX 2080 Ti 11 GB):
# scripts/g0/phase1_transfer_precision_main.py. bf16 suffix against fp32 suffix (the same stored bf16-valued weights promoted exactly), 512 selected pairs, states INT16 and INT4, INT16 executed twice.
# Usage: sbatch -t 01:30:00 --exclude=gnode068,gnode069,gnode076 --export=ALL,REPO=<snapshot> scripts/g0/run_precision.sbatch <name> --instrument        (the instrument job: ONE chunk of 16 features: memory, throughput, gates, stop rules)
#        sbatch -t 04:00:00 --exclude=gnode068,gnode069,gnode076 --export=ALL,REPO=<snapshot> scripts/g0/run_precision.sbatch <name>                       (the full job; chunk checkpoints make it resumable)
# Prereqs: code deployed by scripts/g0/deploy_phase1.sh (it ships the audit freeze and the aux file); env, model cache and FLORES as for run_transfer.sbatch. Read the `GPU 0:` line AND the head of the log.
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH -J precision
#SBATCH -o /home2/karma/duan-verify/logs/%x-%j.out
set -euo pipefail
NAME=${1:?name}; shift
BASE=/home2/karma/duan-verify
export HF_TOKEN_PATH=$BASE/hf/token
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$BASE/logs"
ls $HF_HOME/hub/models--google--gemma-2-2b/snapshots/*/model-00003-of-00003.safetensors > /dev/null || { echo "model cache missing: run fetch_model.sh on the login node"; exit 3; }
trap 'kill ${MONPID:-0} 2>/dev/null || true' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=$BASE/out/phase1/$NAME; mkdir -p "$OUT"
REPO=${REPO:-$BASE/phase1/repo}; COMMIT=$(cat $REPO/COMMIT)      # REPO may point to a snapshot deployed under another name (deploy_phase1.sh DEST)
nvidia-smi -L; date
nvidia-smi -L | grep -q 'GPU 0' || { echo 'no GPU visible on this node: refusing to run on the CPU (job 40402 did, 2026-10-26)'; exit 4; }     # a node whose CUDA fails would run on the CPU and still pass every gate
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 10 > "$OUT/gpu_mem.log" &
MONPID=$!
cd $REPO/scripts/g0
$BASE/env/bin/python phase1_transfer_precision_main.py --out "$OUT" --flores-root $BASE/phase1/data --freeze $REPO/reports/intervention_transfer/freeze.json \
    --core $REPO/reports/intervention_transfer/core_sample.json --audit-freeze $REPO/reports/intervention_transfer/precision_audit_freeze.json --aux $REPO/reports/intervention_transfer/precision_audit_aux.json --commit "$COMMIT" "$@"
date
