#!/bin/bash
# Deploy the COMMITTED Phase 1 code to Ada (git archive of HEAD, so what runs is exactly what the tests ran on) and record the commit.
# Usage: bash scripts/g0/deploy_phase1.sh      (refuses to deploy with uncommitted changes in the deployed paths)
set -euo pipefail
cd "$(dirname "$0")/../.."
PATHS="scripts/g0 third_party/duan2026/pristine/scripts/03_gemma_phase3_streaming_sweep.py reports/g0/their_functions.json reports/g0/phase1_languages.json reports/g0/baseline_smoke_vendored/phase3_summary_final.csv reports/functional_validation/groups_manifest.json"
if [ -n "$(git status --porcelain -- $PATHS)" ]; then echo "uncommitted changes in deployed paths: commit first"; git status --short -- $PATHS; exit 1; fi
COMMIT=$(git rev-parse HEAD)
git archive HEAD $PATHS | ssh karma@ada-gw1 "set -e; R=/home2/karma/duan-verify/phase1/repo; rm -rf \$R; mkdir -p \$R; tar x -C \$R; echo $COMMIT > \$R/COMMIT; mkdir -p /home2/karma/duan-verify/out/phase1; cp \$R/scripts/g0/run_phase1.sbatch \$R/scripts/g0/run_diag.sbatch /home2/karma/duan-verify/; (cd \$R && sha256sum scripts/g0/phase1_run.py scripts/g0/phase1_lib.py scripts/g0/phase1_data.py scripts/g0/their_code.py)"
echo "deployed $COMMIT"
