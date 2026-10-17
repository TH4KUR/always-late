#!/bin/bash
# One Phase 1 run (one BOS regime) on ONE GPU of Ada (u22, RTX 2080 Ti 11 GB), on top of Duan et al.'s functions (see scripts/g0/phase1_run.py).
# Usage: sbatch [-A nlp] scripts/g0/run_phase1.sbatch <name> <flores|wikitext> <mask|none> [extra args for phase1_run.py ...]
# S5 anchor : sbatch run_phase1.sbatch anchor wikitext none --budget 20000 --bits 8 6 --batch-size 1
# S5 smoke  : sbatch run_phase1.sbatch smoke flores mask --languages eng_Latn hin_Deva khm_Khmr --max-sentences 120 --bits 8 6 0 --n-boot 20 --n-perm 5
# S6 full   : sbatch run_phase1.sbatch p1-mask flores mask      (and  p1-none flores none)
# Prereqs: env at /home2/karma/duan-verify/env; HF token at /home2/karma/duan-verify/hf/token; code deployed by scripts/g0/deploy_phase1.sh; FLORES extracted in phase1/data.
# NOTE: /share1 is NOT mounted on compute nodes. Code, data and outputs live on /home2 (NFS); the gated model is fetched to node-local /scratch and deleted at the end.
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
export HF_HOME=/scratch/karma-p1-hf-${SLURM_JOB_ID:-manual}
export HF_XET_CACHE=$HF_HOME/xet HF_XET_HIGH_PERFORMANCE=1
# NOTE: do NOT set HF_HUB_DISABLE_XET=1 (plain HTTP was ~15x slower and kept dropping connections).
mkdir -p "$HF_HOME" "$BASE/logs"
trap 'kill ${MONPID:-0} 2>/dev/null || true; rm -rf "$HF_HOME"' EXIT
for attempt in 1 2 3 4 5 6; do
  $BASE/env/bin/hf download google/gemma-2-2b model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors config.json generation_config.json model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json && break
  echo "model download attempt $attempt failed; retrying"; sleep 10
done
du -sh "$HF_HOME"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6
OUT=$BASE/out/phase1/$NAME; mkdir -p "$OUT"
REPO=$BASE/phase1/repo; COMMIT=$(cat $REPO/COMMIT)
nvidia-smi -L; date
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 5 > "$OUT/gpu_mem.log" &
MONPID=$!
cd $REPO/scripts/g0
$BASE/env/bin/python phase1_run.py --corpus "$CORPUS" --regime "$REGIME" --out "$OUT" --flores-root $BASE/phase1/data --commit "$COMMIT" "$@"
date
