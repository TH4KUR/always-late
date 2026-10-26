#!/bin/bash
# Gemma-2-9B confirmation (D-089, handoff 040) on ONE rented GPU machine (Jarvis Labs or any Linux box with NVIDIA GPU(s) of >= 40 GB: A100 40/80, A6000, L40S, H100). No Slurm.
#
#   HF_TOKEN=hf_xxx bash scripts/g0/jarvis_run_all.sh
#
# That is the whole job. It detaches itself (survives a closed browser tab or ssh), logs to <BASE>/run_all.log, and is RESUMABLE: run the same command again and finished steps are skipped.
#   1 env     python venv with the pinned versions (torch 2.14.1 cu126, transformer-lens 3.9.0, transformers 5.19.0, sae-lens 6.54.5), CUDA check
#   2 data    FLORES-200 (bundled; downloaded and hash-checked otherwise)
#   3 access  the token must reach the gated google/gemma-2-9b; then download of model + Gemma Scope SAE (about 19 GB)
#   4 checks  unit tests of the adapter (CPU) and the freeze file's code hashes
#   5 smoke   3 languages, FP and INT4: the instrument gate (gates must pass, no CUDA warning); a failed gate STOPS the run
#   6 full    tagging + manifest on all 27 languages, then the six states FP INT8 INT7 INT6 INT5 INT4 (300 sentences, 3 control draws), one process per GPU when several GPUs and enough RAM
#   7 package <BASE>/gemma9b_results.tar.gz (outputs, logs, checksums, pip freeze): send this file back
# Needs: HF_TOKEN of an account that has accepted the Gemma licence at https://huggingface.co/google/gemma-2-9b ; about 80 GB of free disk; >= 64 GB RAM per concurrent process.
# Options (environment): BASE (default: parent of the repo), STATES ("FP 8 7 6 5 4"), SKIP_SMOKE=1, FOREGROUND=1 (do not detach), DRY_RUN=1 (print the plan, run nothing),
#   MIN_GPU_MIB (38000), RAM_PER_JOB_GB (64), TORCH_INDEX, MAX_PARALLEL (cap on concurrent states), FAKE_GPUS / FAKE_RAM_GB (tests of this script),
#   LM_EXTRA (extra arguments for the language-model runs, e.g. "--batch-size 16" if a 40 GB card runs out of memory: a deviation, record it in amendments.md).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../.." && pwd)}
BASE=${BASE:-$(dirname "$REPO")}
DRY_RUN=${DRY_RUN:-0}; SKIP_SMOKE=${SKIP_SMOKE:-0}; FOREGROUND=${FOREGROUND:-0}
STATES=${STATES:-"FP 8 7 6 5 4"}
MIN_GPU_MIB=${MIN_GPU_MIB:-38000}; RAM_PER_JOB_GB=${RAM_PER_JOB_GB:-64}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
FLORES_SHA256=b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6
export HF_HOME=${HF_HOME:-$BASE/hf_cache}
export GEMMA9B_N_DEVICES=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VENV=$BASE/venv; PY=$VENV/bin/python
STATE_DIR=$BASE/state; LOGS=$BASE/logs; OUTROOT=$BASE/out/phase1
say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

gpu_mib_list() { if [ -n "${FAKE_GPUS:-}" ]; then tr ' ' '\n' <<<"$FAKE_GPUS"; else nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits; fi; }
ram_gb() { if [ -n "${FAKE_RAM_GB:-}" ]; then echo "$FAKE_RAM_GB"; else awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo; fi; }

preflight() {                                                    # fast checks before anything is installed or detached
    [ "$DRY_RUN" = 1 ] || [ -n "${HF_TOKEN:-}" ] || die "set HF_TOKEN (a Hugging Face token of an account that accepted the Gemma licence): HF_TOKEN=hf_xxx bash scripts/g0/jarvis_run_all.sh"
    GPUS=(); local i=0 m
    while read -r m; do
        [ -z "$m" ] && continue
        if [ "$m" -ge "$MIN_GPU_MIB" ]; then GPUS+=("$i"); else say "GPU $i has only $m MiB (< $MIN_GPU_MIB): not used"; fi
        i=$((i + 1))
    done < <(gpu_mib_list)
    [ "${#GPUS[@]}" -ge 1 ] || die "no GPU with >= $MIN_GPU_MIB MiB: Gemma-2-9B in bf16 needs a card of at least 40 GB (A100, A6000, L40S, H100)"
    local ram; ram=$(ram_gb)
    NPROC=$(( ram / RAM_PER_JOB_GB )); [ "$NPROC" -ge 1 ] || NPROC=1
    [ "$NPROC" -le "${#GPUS[@]}" ] || NPROC=${#GPUS[@]}
    [ -z "${MAX_PARALLEL:-}" ] || [ "$NPROC" -le "$MAX_PARALLEL" ] || NPROC=$MAX_PARALLEL
    say "usable GPUs: ${GPUS[*]} (of $i); RAM ${ram} GB; concurrent state processes: $NPROC"
    if [ "$ram" -lt "$RAM_PER_JOB_GB" ]; then say "WARNING: only ${ram} GB RAM; the RTN quantizer keeps a full-precision CPU copy of the model (about 19 GB) besides the loaded model"; fi
}

if [ "${JARVIS_DETACHED:-0}" != 1 ] && [ "$DRY_RUN" != 1 ] && [ "$FOREGROUND" != 1 ]; then
    preflight
    mkdir -p "$BASE"
    JARVIS_DETACHED=1 nohup bash "${BASH_SOURCE[0]}" "$@" > "$BASE/run_all.log" 2>&1 &
    echo "Started in the background (pid $!). Follow it with:  tail -f $BASE/run_all.log"
    echo "It is safe to close this terminal. To resume after a stop, run the same command again."
    exit 0
fi

preflight
if [ "$DRY_RUN" != 1 ]; then mkdir -p "$STATE_DIR" "$LOGS" "$OUTROOT" "$HF_HOME"; fi
COMMIT=$(cat "$REPO/COMMIT" 2>/dev/null || git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
say "repo $REPO | base $BASE | commit $COMMIT"

run() { echo "+ $*"; [ "$DRY_RUN" = 1 ] || "$@"; }
step() { local name=$1; shift
    if [ -f "$STATE_DIR/$name.done" ]; then say "[skip] $name (already done)"; return 0; fi
    say "== $name"; "$@" || return $?
    [ "$DRY_RUN" = 1 ] || touch "$STATE_DIR/$name.done"; }
g9() { ( cd "$REPO/scripts/g0" && "$PY" phase1_gemma9b_run.py "$@" ); }

do_env() {
    if [ ! -x "$PY" ]; then
        run python3 -m venv "$VENV" || die "python3 -m venv failed (on Debian/Ubuntu: apt-get install -y python3-venv), then run the command again"
    fi
    run "$PY" -m pip install --quiet --upgrade pip
    run "$PY" -m pip install --quiet "torch==2.14.1" --index-url "$TORCH_INDEX"
    run "$PY" -m pip install --quiet "transformer-lens==3.9.0" "transformers==5.19.0" "sae-lens==6.54.5" numpy scipy safetensors huggingface_hub tqdm
    if [ "$DRY_RUN" != 1 ]; then
        "$PY" -m pip freeze > "$STATE_DIR/pip_freeze.txt"
        "$PY" - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "CUDA is not available to torch (driver too old for CUDA 12.6? set TORCH_INDEX to another wheel index)"
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PYEOF
    fi
}

do_flores() {
    if [ -d "$BASE/data/flores200_dataset/dev" ] && [ -d "$BASE/data/flores200_dataset/devtest" ]; then say "FLORES-200 present"; return 0; fi
    run mkdir -p "$BASE/data"
    run curl -fSL -o "$BASE/data/flores200_dataset.tar.gz" https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
    if [ "$DRY_RUN" != 1 ]; then
        echo "$FLORES_SHA256  $BASE/data/flores200_dataset.tar.gz" | sha256sum -c - || die "FLORES archive hash differs from the frozen one"
        tar xzf "$BASE/data/flores200_dataset.tar.gz" -C "$BASE/data"
    fi
}

do_access() {
    [ "$DRY_RUN" = 1 ] && { echo "+ check the token reaches google/gemma-2-9b; python scripts/g0/prefetch_gemma9b.py"; return 0; }
    "$PY" - <<'PYEOF' || die "the token cannot read google/gemma-2-9b: log in at huggingface.co with the account that owns the token, open https://huggingface.co/google/gemma-2-9b and accept the licence (fine-grained tokens also need 'read access to public gated repos')"
import os, sys
from huggingface_hub import hf_hub_download
hf_hub_download("google/gemma-2-9b", "config.json", token=os.environ["HF_TOKEN"])
print("token reaches google/gemma-2-9b")
PYEOF
    "$PY" "$REPO/scripts/g0/prefetch_gemma9b.py" || die "prefetch failed (disk space? about 19 GB are needed under $HF_HOME)"
}

do_checks() {
    ( cd "$REPO/scripts/g0" && run "$PY" -B -m unittest tests_torch.test_phase1_gemma9b_run tests_torch.test_phase1_llama_run tests.test_make_gemma9b_freeze ) || die "unit tests failed: the code is not the tested code"
    run "$PY" -B "$REPO/scripts/g0/make_gemma9b_freeze.py" --verify
}

# a finished run of the language-model runner: exit 0 is not enough, the gates and a result file must be there
check_lm_log() {                                                 # <log> <out dir>
    local log=$1 out=$2
    grep -q "gates passed" "$log" || { say "no 'gates passed' in $log"; return 1; }
    ! grep -q "CUDA initialization" "$log" || { say "CUDA initialization warning in $log (ran on CPU?)"; return 1; }
    ls "$out"/func_lm_*_*.pt > /dev/null 2>&1 || { say "no result file in $out"; return 1; }
    return 0
}

tag_and_manifest() {                                             # <name> <extra tag args...>
    local name=$1; shift; local out=$OUTROOT/$name
    if [ ! -f "$out/eq/selectivity.pt" ]; then
        CUDA_VISIBLE_DEVICES=${GPUS[0]} g9 tag --out "$out" --flores-root "$BASE/data" --commit "$COMMIT" "$@" > "$LOGS/$name.log" 2>&1 || { tail -n 30 "$LOGS/$name.log"; die "tagging failed, see $LOGS/$name.log"; }
    fi
    if ! g9 manifest --selectivity "$out/eq/selectivity.pt" --manifest "$out/groups_manifest.json" >> "$LOGS/$name.log" 2>&1; then
        say "the frozen group builder refused the full manifest; building the K16/K64 families only (reported in MANIFEST_FALLBACK.txt, never silent)"
        echo "families full (K16, K64, all) refused by the frozen builder; used --families k. See $LOGS/$name.log" > "$out/MANIFEST_FALLBACK.txt"
        g9 manifest --selectivity "$out/eq/selectivity.pt" --manifest "$out/groups_manifest.json" --families k >> "$LOGS/$name.log" 2>&1 || die "manifest failed even with K16/K64 only, see $LOGS/$name.log"
    fi
}

do_smoke() {
    [ "$DRY_RUN" = 1 ] && { echo "+ smoke: tag (eng_Latn hin_Deva khm_Khmr, 40 sentences) -> manifest -> lm FP 4 (pilot split), then check gates/CUDA/result files"; return 0; }
    tag_and_manifest g9-smoke-tag --languages eng_Latn hin_Deva khm_Khmr --max-sentences 40
    local out=$OUTROOT/g9-smoke-lm log=$LOGS/g9-smoke-lm.log
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} g9 lm --out "$out" --flores-root "$BASE/data" --selectivity "$OUTROOT/g9-smoke-tag/eq/selectivity.pt" --manifest "$OUTROOT/g9-smoke-tag/groups_manifest.json" \
        --commit "$COMMIT" --precisions FP 4 --languages eng_Latn hin_Deva khm_Khmr --max-sentences 40 --split pilot --panel ${LM_EXTRA:-} > "$log" 2>&1 || { tail -n 40 "$log"; die "smoke job failed, see $log"; }
    check_lm_log "$log" "$out" || { tail -n 40 "$log"; die "smoke gate FAILED: stop here and send $log to whoever is running this study; do not start the full run"; }
    say "smoke gate passed ($(grep -o 'peak GPU GB [0-9.]*' "$log" | tail -n 1))"
}

state_job() {                                                    # <state> <gpu index>
    local P=$1 gpu=$2 out=$OUTROOT/g9-lm-$1 log=$LOGS/g9-lm-$1.log
    mkdir -p "$out"
    say "state $P on GPU $gpu"
    { nvidia-smi -L || true; date; } >> "$log" 2>&1
    if CUDA_VISIBLE_DEVICES=$gpu g9 lm --out "$out" --flores-root "$BASE/data" --selectivity "$OUTROOT/g9-tag/eq/selectivity.pt" --manifest "$OUTROOT/g9-tag/groups_manifest.json" \
        --commit "$COMMIT" --precisions "$P" --split eval --n-eval 300 --panel ${LM_EXTRA:-} >> "$log" 2>&1 && check_lm_log "$log" "$out"; then
        touch "$STATE_DIR/state_$P.done"; say "state $P finished and validated"
    else
        touch "$STATE_DIR/state_$P.failed"; say "state $P FAILED, see $log"
    fi
}

do_full() {
    [ "$DRY_RUN" = 1 ] && { echo "+ full: tag all 27 languages -> manifest -> states [$STATES] (300 sentences, 3 control draws), $NPROC at a time on GPUs ${GPUS[*]}"; return 0; }
    tag_and_manifest g9-tag
    local -a slot_pid=(); local P s
    for P in $STATES; do
        if [ -f "$STATE_DIR/state_$P.done" ]; then say "[skip] state $P (done)"; continue; fi
        rm -f "$STATE_DIR/state_$P.failed"
        while true; do                                           # wait for a free slot (one slot = one GPU)
            s=-1
            for ((i = 0; i < NPROC; i++)); do
                if [ -z "${slot_pid[$i]:-}" ] || ! kill -0 "${slot_pid[$i]}" 2>/dev/null; then s=$i; break; fi
            done
            [ "$s" -ge 0 ] && break
            sleep 30
        done
        ( state_job "$P" "${GPUS[$s]}" ) &
        slot_pid[$s]=$!
    done
    wait
    local failed=""
    for P in $STATES; do [ -f "$STATE_DIR/state_$P.done" ] || failed="$failed $P"; done
    [ -z "$failed" ] || { say "STATES NOT FINISHED:$failed (run the same command again to retry them; logs in $LOGS)"; return 1; }
}

do_package() {
    ( cd "$BASE" && { echo "# sha256 of every output file"; find out -type f | sort | xargs sha256sum; } > artifact_checksums.md
      local f files=(out logs artifact_checksums.md)
      for f in state/pip_freeze.txt run_all.log repo/COMMIT repo/reports/functional_validation/gemma9b/freeze.json; do [ -e "$f" ] && files+=("$f"); done
      tar czf gemma9b_results.tar.gz "${files[@]}" )
    say "results: $BASE/gemma9b_results.tar.gz ($(du -h "$BASE/gemma9b_results.tar.gz" | cut -f1))"
}

step env do_env
step flores do_flores
step access_and_download do_access
step checks do_checks
if [ "$SKIP_SMOKE" != 1 ]; then step smoke do_smoke; fi
rc=0
step full do_full || rc=$?
if [ "$rc" -ne 0 ]; then rm -f "$STATE_DIR/full.done"; fi
if [ "$DRY_RUN" != 1 ]; then do_package; fi
if [ "$rc" -eq 0 ]; then say "ALL DONE. Send $BASE/gemma9b_results.tar.gz back."; else say "PARTIAL: some states failed (see above). Send $BASE/gemma9b_results.tar.gz and $BASE/run_all.log back."; exit "$rc"; fi
