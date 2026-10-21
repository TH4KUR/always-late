#!/bin/bash
# D-035: association-retention analysis of the sidecar of a finished association run, on a CPU node of Ada (the sidecar is too large to copy home: about 0.26 MB/s).
# Usage: sbatch scripts/g0/run_association_analysis.sbatch <run name under out/phase1> [extra args for phase1_association_analysis.py, e.g. --n-boot 1000 --seed 0 --stream eq]
# Prereqs: the run finished and validated (association/manifest.json), code deployed by scripts/g0/deploy_phase1.sh, the original mask/eq run in out/phase1/p1-mask/eq (cohort and historical correlations).
#SBATCH -p u22-cpu
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J assoc-analysis
#SBATCH -o /home2/karma/duan-verify/logs/%x-%j.out
set -euo pipefail
NAME=${1:?run name}; shift 1
BASE=/home2/karma/duan-verify
RUN=$BASE/out/phase1/$NAME; OUT=$RUN/analysis
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
mkdir -p "$BASE/logs"; date; cat $BASE/phase1/repo/COMMIT
cd $BASE/phase1/repo/scripts/g0
$BASE/env/bin/python phase1_association_analysis.py --run "$RUN" --baseline $BASE/out/phase1/p1-mask/eq --out "$OUT" "$@"
date; ls -la "$OUT"
