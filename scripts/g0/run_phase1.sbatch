#!/bin/bash
# One Phase 1 run (one BOS regime) on ONE GPU of Ada (u22, RTX 2080 Ti 11 GB), on top of Duan et al.'s functions (see scripts/g0/phase1_run.py).
# Usage: sbatch [-A nlp] scripts/g0/run_phase1.sbatch <name> <flores|wikitext> <mask|none> [extra args for phase1_run.py ...]
# S5 anchor : sbatch run_phase1.sbatch anchor wikitext none --budget 20000 --bits 8 6 --batch-size 1 --loss tl
# S5 smoke  : sbatch run_phase1.sbatch smoke flores mask --languages eng_Latn hin_Deva khm_Khmr --max-sentences 120 --bits 8 6 0 --n-boot 20 --n-perm 5
# S6 full   : sbatch run_phase1.sbatch p1-mask flores mask      (and  p1-none flores none)
# Prereqs: env at /home2/karma/duan-verify/env; HF token at /home2/karma/duan-verify/hf/token; code deployed by scripts/g0/deploy_phase1.sh; FLORES extracted in phase1/data.
# NOTE: /share1 is NOT mounted on compute nodes. Code, data, outputs and the Hugging Face cache live on /home2 (NFS).
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=40G
#SBATCH -t 10:00:00
#SBATCH -J phase1
#SBATCH -o /home2/karma/duan-verify/logs/%x-%j.out
set -euo pipefail
NAME=${1:?name}; CORPUS=${2:?corpus}; REGIME=${3:?regime}; shift 3
BASE=/home2/karma/duan-verify
export HF_TOKEN_PATH=$BASE/hf/token
# The model (10.5 GB), SAE, WikiText-2 and tokenizer come from the PERSISTENT cache on /home2 (filled once on the login node by fetch_model.sh and prime_hf_cache.py):
# three jobs fetching the model in parallel stalled for 20+ minutes on 2026-10-17, while one login-node download ran at ~20 MB/s. Jobs run offline.
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$BASE/logs"
ls $HF_HOME/hub/models--google--gemma-2-2b/snapshots/*/model-00003-of-00003.safetensors > /dev/null || { echo "model cache missing: run fetch_model.sh on the login node"; exit 3; }
trap 'kill ${MONPID:-0} 2>/dev/null || true' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=$BASE/out/phase1/$NAME; mkdir -p "$OUT"
REPO=$BASE/phase1/repo; COMMIT=$(cat $REPO/COMMIT)
nvidia-smi -L; date
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 5 > "$OUT/gpu_mem.log" &
MONPID=$!
cd $REPO/scripts/g0
$BASE/env/bin/python phase1_run.py --corpus "$CORPUS" --regime "$REGIME" --out "$OUT" --flores-root $BASE/phase1/data --commit "$COMMIT" "$@"
date
