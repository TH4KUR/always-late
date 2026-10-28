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
# THREE-ARM EXTENSION (docs/superpowers/specs/2026-10-28-gemma9b-three-arm-confirmation-design.md; the legacy path above is unchanged and is what ARMS=legacy STAGES=g runs): the same study with three weight-state arms (legacy RTN =
# the frozen run, matched RTN, GPTQ) and two stages (g = tier 1, Study G; p1 = tier 2, Phase 1 survival and loss). Extra steps for the two new arms: tokenizer (the 2B calibration pool may be reused only if the 9B tokenizer is the
# same), pool (the calibration token files, verified against reports/gptq_extension/calibration_freeze.json), smoke_p1, gate9b (builds and gates the INT6/INT4 weight sets of both methods and records their hashes in
# reports/functional_validation/gemma9b/artifact_hashes.json; every later unit refuses a different hash), smoke_gptq. Units run in the order of spec section 9; a unit is (stage, arm, state) for g and (p1, arm) for tier 2.
#   STOP_AFTER=<step> stops cleanly after that step (use STOP_AFTER=smoke_gptq for a first run, read the timings, then continue); LM_EXTRA / P1_EXTRA: extra arguments of the tier-1 / tier-2 runs (amendment first).
#   ARMS ("legacy rtn-matched gptq"), STAGES ("g p1"), STATES (legacy tier-1 states), STATES_NEW ("6 4"), P1_BITS_LEGACY ("8 7 6 5 4 16 12 0"), P1_BITS_NEW ("7 6 4 16 12 0"), PROCS_PER_GPU (1; 2 only if two units fit the RAM),
#   POOL_DIR (default <BASE>/gptq/calibration), FAKE_CC / FAKE_DRIVER (tests). torch 2.14.1 comes from the cu130 index on a Blackwell GPU (compute capability >= 12, driver >= 580), from cu126 otherwise; TORCH_INDEX overrides.
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
TORCH_INDEX=${TORCH_INDEX:-}                                    # chosen in preflight from the compute capability unless set
ARMS=${ARMS:-"legacy rtn-matched gptq"}; STAGES=${STAGES:-"g p1"}; STATES_NEW=${STATES_NEW:-"6 4"}
P1_BITS_LEGACY=${P1_BITS_LEGACY:-"8 7 6 5 4 16 12 0"}; P1_BITS_NEW=${P1_BITS_NEW:-"7 6 4 16 12 0"}; PROCS_PER_GPU=${PROCS_PER_GPU:-1}
MODEL_GB=${MODEL_GB:-18.5}; TARGET_GB=${TARGET_GB:-17.6}        # host RAM terms of a job (gptq_ext/ram_guard.py)
POOL_DIR=${POOL_DIR:-$BASE/gptq/calibration}
FREEZE_DIR=$REPO/reports/functional_validation/gemma9b
HASHES=$FREEZE_DIR/artifact_hashes.json; AMEND=$FREEZE_DIR/amendment_A1_three_arm.json; CAL_FREEZE=$REPO/reports/gptq_extension/calibration_freeze.json
FLORES_SHA256=b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6
export HF_HOME=${HF_HOME:-$BASE/hf_cache}
export GEMMA9B_N_DEVICES=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VENV=$BASE/venv; PY=$VENV/bin/python
STATE_DIR=$BASE/state; LOGS=$BASE/logs; OUTROOT=$BASE/out/phase1
say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

gpu_mib_list() { if [ -n "${FAKE_GPUS:-}" ]; then tr ' ' '\n' <<<"$FAKE_GPUS"; else nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits; fi; }
gpu_cc() { if [ -n "${FAKE_CC:-}" ]; then echo "$FAKE_CC"; elif command -v nvidia-smi > /dev/null 2>&1; then { nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | head -n 1 | tr -d ' '; } || true; fi; }
driver_ver() { if [ -n "${FAKE_DRIVER:-}" ]; then echo "$FAKE_DRIVER"; elif command -v nvidia-smi > /dev/null 2>&1; then { nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null | head -n 1 | tr -d ' '; } || true; fi; }
has_word() { case " $1 " in *" $2 "*) return 0 ;; esac; return 1; }
need_weights() { has_word "$ARMS" rtn-matched || has_word "$ARMS" gptq; }
need_gb() { awk -v m="$MODEL_GB" -v t="$TARGET_GB" -v n="$1" 'BEGIN { e = (n > 0) ? t * (1 + n) : 0; printf "%.1f", (m + e) * 1.1 }'; }     # same formula as gptq_ext/ram_guard.estimate_gb
in_g0() { ( cd "$REPO/scripts/g0" && "$@" ); }

choose_torch_index() {                                           # an explicit TORCH_INDEX wins; Blackwell needs the CUDA 13.0 build (torch 2.14.1 has none for 12.8) and a driver >= 580
    [ -z "$TORCH_INDEX" ] || return 0
    local cc drv major dmajor; cc=$(gpu_cc); drv=$(driver_ver); major=${cc%%.*}; dmajor=${drv%%.*}
    if [ -n "$cc" ] && [ "${major:-0}" -ge 12 ] 2> /dev/null; then
        { [ -n "$drv" ] && [ "${dmajor:-0}" -ge 580 ]; } 2> /dev/null || die "this GPU has compute capability $cc (Blackwell): torch 2.14.1 needs its CUDA 13.0 build, which needs an NVIDIA driver of 580 or newer (found '${drv:-unknown}'). Update the driver or choose the image with a newer driver, or set TORCH_INDEX to a wheel index whose torch supports sm_${cc/./}"
        TORCH_INDEX=https://download.pytorch.org/whl/cu130
    else
        TORCH_INDEX=https://download.pytorch.org/whl/cu126
    fi
    say "compute capability ${cc:-unknown}, driver ${drv:-unknown} -> torch index $TORCH_INDEX"
}

ram_gb() { if [ -n "${FAKE_RAM_GB:-}" ]; then echo "$FAKE_RAM_GB"; else awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo; fi; }

preflight() {                                                    # fast checks before anything is installed or detached
    [ "$DRY_RUN" = 1 ] || [ -n "${HF_TOKEN:-}" ] || die "set HF_TOKEN (a Hugging Face token of an account that accepted the Gemma licence): HF_TOKEN=hf_xxx bash scripts/g0/jarvis_run_all.sh"
    GPUS=(); local i=0 m
    while read -r m; do
        [ -z "$m" ] && continue
        if [ "$m" -ge "$MIN_GPU_MIB" ]; then GPUS+=("$i"); else say "GPU $i has only $m MiB (< $MIN_GPU_MIB): not used"; fi
        i=$((i + 1))
    done < <(gpu_mib_list)
    [ "${#GPUS[@]}" -ge 1 ] || die "no GPU with >= $MIN_GPU_MIB MiB: Gemma-2-9B in bf16 needs a card of at least 40 GB (A100, A6000, L40S, H100, RTX PRO 6000). If the card exists, nvidia-smi must list it: check that the NVIDIA driver is loaded (a Blackwell card needs driver 580 or newer)"
    local ram; ram=$(ram_gb)
    NPROC=$(( ram / RAM_PER_JOB_GB )); [ "$NPROC" -ge 1 ] || NPROC=1
    local slots=$(( ${#GPUS[@]} * PROCS_PER_GPU )); [ "$NPROC" -le "$slots" ] || NPROC=$slots
    [ -z "${MAX_PARALLEL:-}" ] || [ "$NPROC" -le "$MAX_PARALLEL" ] || NPROC=$MAX_PARALLEL
    say "usable GPUs: ${GPUS[*]} (of $i); RAM ${ram} GB; concurrent state processes: $NPROC"
    choose_torch_index
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
CUR_STEP=start
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "[$(date +%H:%M:%S)] FAILED (exit $rc) during: $CUR_STEP" >&2; fi' EXIT
stop_after() { if [ "${STOP_AFTER:-}" = "$1" ]; then say "STOP_AFTER=$1 reached: stopping here (run the same command without STOP_AFTER to continue)"; exit 0; fi; }
step() { local name=$1; shift; CUR_STEP=$name                 # the function is called plainly (not on the left of ||): bash ignores errexit inside a function called that way, which let failed commands pass
    if [ -f "$STATE_DIR/$name.done" ]; then say "[skip] $name (already done)"
    else
        say "== $name"; "$@"
        if [ "$DRY_RUN" != 1 ]; then touch "$STATE_DIR/$name.done"; fi
    fi
    stop_after "$name"
}
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
assert torch.cuda.is_available(), "CUDA is not available to torch (driver too old for this CUDA build? set TORCH_INDEX to another wheel index)"
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
major, minor = torch.cuda.get_device_capability(0)
arch = f"sm_{major}{minor}"
assert any(a == arch or a.startswith(f"sm_{major}") or a == f"compute_{major}{minor}" for a in torch.cuda.get_arch_list()), f"this torch build ({torch.cuda.get_arch_list()}) has no kernels for {arch}: choose another TORCH_INDEX"
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite((x @ x).float()).all(), "a bf16 matrix product on the GPU gave non-finite values"
print("bf16 matmul on", arch, "ok")
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

verify_frozen() {                                                # EVERY invocation, before anything is installed or downloaded: the code must be the frozen code (a resume after an edit must not pass)
    local py=$PY; [ -x "$py" ] || py=python3                     # both tools are stdlib only
    run "$py" -B "$REPO/scripts/g0/make_gemma9b_freeze.py" --verify
    if need_weights || has_word "$STAGES" p1; then
        [ "$DRY_RUN" = 1 ] || [ -f "$AMEND" ] || die "amendment A1 ($AMEND) is missing: the predictions of the new arms are not frozen"
        run "$py" -B "$REPO/scripts/g0/make_gemma9b_amendment.py" --verify
    fi
}

do_checks() {
    ( cd "$REPO/scripts/g0" && run "$PY" -B -m unittest tests_torch.test_phase1_gemma9b_run tests_torch.test_phase1_llama_run tests.test_make_gemma9b_freeze ) || die "unit tests failed: the code is not the tested code"
    if need_weights || has_word "$STAGES" p1; then
        ( cd "$REPO/scripts/g0" && run "$PY" -B -m unittest tests.test_gptq_ext_tokenizer_check tests.test_gptq_ext_ram_guard tests.test_gptq_ext_record_hashes tests.test_make_gemma9b_amendment tests_torch.test_phase1_gemma9b_ext tests_torch.test_phase1_gemma9b_compare ) || die "unit tests of the three-arm extension failed: the code is not the tested code"
    fi
    if need_weights; then
        ( cd "$REPO/scripts/g0" && run "$PY" -B -m unittest discover -s tests_torch -p "test_gptq_ext_*.py" ) || die "the GPTQ extension tests failed on this machine"
    fi
}

do_tokenizer() {                                                 # the 2B calibration pool may be reused only if the 9B tokenizer is the 2B one
    [ "$DRY_RUN" = 1 ] && { echo "+ tokenizer: fetch the google/gemma-2-2b tokenizer files; python -m gptq_ext.tokenizer_check --a google/gemma-2-2b --b google/gemma-2-9b"; return 0; }
    "$PY" - <<'PYEOF' || die "the token cannot read the tokenizer of google/gemma-2-2b: accept its licence too (https://huggingface.co/google/gemma-2-2b)"
import os
from huggingface_hub import snapshot_download
snapshot_download("google/gemma-2-2b", allow_patterns=["tokenizer*", "*.model", "special_tokens_map.json", "config.json"], token=os.environ["HF_TOKEN"])
print("2B tokenizer files present")
PYEOF
    in_g0 "$PY" -m gptq_ext.tokenizer_check --a google/gemma-2-2b --b google/gemma-2-9b --out "$STATE_DIR/tokenizer_check.json" || die "the Gemma-2-9B tokenizer differs from the 2B one: the frozen calibration pool cannot be reused. STOP and tell the author."
}

do_pool() {                                                      # the calibration token files, verified against the frozen pool (file hashes, passage hashes, pool sha256)
    [ "$DRY_RUN" = 1 ] && { echo "+ pool: verify $POOL_DIR against $CAL_FREEZE (gptq_ext.calibration.load_pool and pool_sha256)"; return 0; }
    [ -d "$POOL_DIR" ] || die "the calibration pool is missing: $POOL_DIR (unpack gptq_calibration_pool.tar.gz of the bundle there, or set POOL_DIR)"
    in_g0 "$PY" - "$CAL_FREEZE" "$POOL_DIR" <<'PYEOF' || die "the calibration pool does not match calibration_freeze.json"
import json, sys
from gptq_ext import calibration as cal
freeze = json.load(open(sys.argv[1]))
pool = cal.load_pool(freeze, sys.argv[2])
sha = cal.pool_sha256(pool)
assert sha == freeze["pool_sha256"], (sha, freeze["pool_sha256"])
print("calibration pool verified:", sha[:16], "languages", len(pool))
PYEOF
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
    [ "$DRY_RUN" = 1 ] && { echo "+ smoke: tag (all 27 languages, 100 sentences each) -> manifest -> lm FP 4 on eng_Latn hin_Deva khm_Khmr (pilot split), then check gates/CUDA/result files"; return 0; }
    tag_and_manifest g9-smoke-tag --max-sentences 100            # the frozen lm runner asserts that the manifest and the selectivity cover the whole roster: only the lm run below is restricted to 3 languages
    local out=$OUTROOT/g9-smoke-lm log=$LOGS/g9-smoke-lm.log
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} g9 lm --out "$out" --flores-root "$BASE/data" --selectivity "$OUTROOT/g9-smoke-tag/eq/selectivity.pt" --manifest "$OUTROOT/g9-smoke-tag/groups_manifest.json" \
        --commit "$COMMIT" --precisions FP 4 --languages eng_Latn hin_Deva khm_Khmr --max-sentences 40 --split pilot --panel ${LM_EXTRA:-} > "$log" 2>&1 || { tail -n 40 "$log"; die "smoke job failed, see $log"; }
    check_lm_log "$log" "$out" || { tail -n 40 "$log"; die "smoke gate FAILED: stop here and send $log to whoever is running this study; do not start the full run"; }
    say "smoke gate passed ($(grep -o 'peak GPU GB [0-9.]*' "$log" | tail -n 1))"
}

do_smoke_p1() {                                                  # the tier-2 path on 3 languages, legacy rounding, 4 bits: proves the p1 task before hours are spent
    [ "$DRY_RUN" = 1 ] && { echo "+ smoke_p1: phase1_gemma9b_ext.py p1 --corpus flores --regime mask --languages eng_Latn hin_Deva khm_Khmr --max-sentences 40 --bits 4 ${P1_EXTRA:-}"; return 0; }
    local out=$OUTROOT/g9-smoke-p1 log=$LOGS/g9-smoke-p1.log; mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} in_g0 "$PY" phase1_gemma9b_ext.py p1 --corpus flores --regime mask --languages eng_Latn hin_Deva khm_Khmr --max-sentences 40 --bits 4 ${P1_EXTRA:-} --out "$out" --flores-root "$BASE/data" --commit "$COMMIT" > "$log" 2>&1 || { tail -n 40 "$log"; die "tier-2 smoke job failed, see $log"; }
    ! grep -q "CUDA initialization" "$log" || die "CUDA initialization warning in $log (ran on CPU?)"
    find "$out" -name survival.pt | grep -q . || { tail -n 40 "$log"; die "tier-2 smoke produced no survival.pt, see $log"; }
    say "tier-2 smoke passed"
}

gate_bits_wanted() {                                             # artifact bit widths (4 to 8) the weight-set units build, in order of first use: STATES_NEW (tier 1), then P1_BITS_NEW (tier 2)
    local b out=""
    if has_word "$STAGES" g; then for b in $STATES_NEW; do case $b in 4|5|6|7|8) has_word "$out" "$b" || out="$out $b" ;; esac; done; fi
    if has_word "$STAGES" p1; then for b in $P1_BITS_NEW; do case $b in 4|5|6|7|8) has_word "$out" "$b" || out="$out $b" ;; esac; done; fi
    echo $out
}

gate_chunks() {                                                  # one line per gate job: at most two bit widths (a weight set is about 17.6 GB of host RAM and a gate job keeps two copies of the model besides)
    local -a w; local i
    read -r -a w <<< "$(gate_bits_wanted)"
    for ((i = 0; i < ${#w[@]}; i += 2)); do echo "${w[*]:i:2}"; done
}

do_gate9b() {                                                    # build the weight sets of both methods on the real model, run the gates, record the hashes every later unit must reproduce
    local -a chunks=(); local -a dirs=(); local chunk out log i=0 n
    mapfile -t chunks < <(gate_chunks)
    [ "${#chunks[@]}" -gt 0 ] || die "no artifact bit width among STATES_NEW ($STATES_NEW) and P1_BITS_NEW ($P1_BITS_NEW): nothing to gate"
    for chunk in "${chunks[@]}"; do
        i=$((i + 1)); out=$OUTROOT/g9-gate; log=$LOGS/g9-gate.log
        if [ "$i" -gt 1 ]; then out=$OUTROOT/g9-gate$i; log=$LOGS/g9-gate$i.log; fi
        dirs+=(--gate-dir "$out")
        n=$(( $(wc -w <<< "$chunk") + 2 ))                       # the weight sets of the chunk, one rebuilt for the determinism check, one transient
        if [ "$DRY_RUN" = 1 ]; then
            echo "+ gate9b: python -m gptq_ext.ram_guard --model-gb $MODEL_GB --target-gb $TARGET_GB --n-artifacts $n"
            echo "+ gate9b: phase1_gemma9b_ext.py gate --out $out --methods gptq rtn-matched --bits $chunk --record-weights-hashes --calibration-freeze $CAL_FREEZE --calibration-text $POOL_DIR"
            continue
        fi
        mkdir -p "$out"
        in_g0 "$PY" -m gptq_ext.ram_guard --model-gb "$MODEL_GB" --target-gb "$TARGET_GB" --n-artifacts "$n" --ram-gb "$(ram_gb)" || die "not enough host RAM for the gate job (it keeps two full-precision copies and the weight sets of bits $chunk)"
        if [ ! -f "$out/gate_report.json" ]; then
            CUDA_VISIBLE_DEVICES=${GPUS[0]} in_g0 "$PY" phase1_gemma9b_ext.py gate --out "$out" --flores-root "$BASE/data" --roster "$REPO/reports/g0/phase1_languages.json" --methods gptq rtn-matched --bits $chunk \
                --commit "$COMMIT" --calibration-freeze "$CAL_FREEZE" --calibration-text "$POOL_DIR" --record-weights-hashes > "$log" 2>&1 || { tail -n 40 "$log"; die "gate job (bits $chunk) failed, see $log"; }
        fi
        ! grep -q "CUDA initialization" "$log" || die "CUDA initialization warning in $log (ran on CPU?)"
        grep -q '"all_gates_ok": true' "$out/gate_report.json" || die "a gate FAILED (see $out/gate_report.json): stop and send it to the author"
    done
    if [ "$DRY_RUN" = 1 ]; then echo "+ gate9b: python -m gptq_ext.record_hashes ${dirs[*]} --out $HASHES"; return 0; fi
    if [ -f "$HASHES" ]; then
        in_g0 "$PY" -m gptq_ext.record_hashes "${dirs[@]}" --out "$HASHES" --check || die "$HASHES exists and differs from these gate jobs' hashes (a second build gave other weights?): stop and tell the author"
    else
        in_g0 "$PY" -m gptq_ext.record_hashes "${dirs[@]}" --out "$HASHES" || die "recording the artifact hashes failed"
    fi
    say "gate jobs passed; artifact hashes recorded in $HASHES"
}

do_smoke_gptq() {                                                # the whole GPTQ path (wrapper, prebuild, frozen hash check, lm run) on the 3 smoke languages, INT4
    [ "$DRY_RUN" = 1 ] && { echo "+ smoke_gptq: run_with_weights.py --entry phase1_gemma9b_ext:main --weights gptq --gptq-freeze $HASHES -- lm ... --precisions 4 --languages eng_Latn hin_Deva khm_Khmr --max-sentences 40 --split pilot"; return 0; }
    local out=$OUTROOT/g9-smoke-gptq log=$LOGS/g9-smoke-gptq.log; mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} in_g0 "$PY" gptq_ext/run_with_weights.py --entry phase1_gemma9b_ext:main --weights gptq --gptq-freeze "$HASHES" --calibration-freeze "$CAL_FREEZE" --calibration-text "$POOL_DIR" -- \
        lm --out "$out" --flores-root "$BASE/data" --selectivity "$OUTROOT/g9-smoke-tag/eq/selectivity.pt" --manifest "$OUTROOT/g9-smoke-tag/groups_manifest.json" --commit "$COMMIT" --precisions 4 \
        --languages eng_Latn hin_Deva khm_Khmr --max-sentences 40 --split pilot --panel ${LM_EXTRA:-} > "$log" 2>&1 || { tail -n 40 "$log"; die "GPTQ smoke job failed, see $log"; }
    check_lm_log "$log" "$out" && [ -f "$out/weights_manifest.json" ] || { tail -n 40 "$log"; die "GPTQ smoke gate FAILED: stop here and send $log to the author"; }
    say "GPTQ smoke passed ($(grep -o 'peak GPU GB [0-9.]*' "$log" | tail -n 1))"
}

plan_units() {                                                   # "<stage> <arm> <token>" lines in the order of the spec (section 9); states outside the fixed lists are appended
    local spec arm s
    if has_word "$STAGES" g; then
        for spec in legacy:FP gptq:4 gptq:6 rtn-matched:4 rtn-matched:6 legacy:4 legacy:6 legacy:8 legacy:7 legacy:5; do
            arm=${spec%%:*}; s=${spec##*:}
            has_word "$ARMS" "$arm" || continue
            if [ "$arm" = legacy ]; then has_word "$STATES" "$s" || continue; else has_word "$STATES_NEW" "$s" || continue; fi
            echo "g $arm $s"
        done
        if has_word "$ARMS" legacy; then
            for s in $STATES; do case " FP 4 6 8 7 5 " in *" $s "*) ;; *) echo "g legacy $s" ;; esac; done
        fi
        for arm in gptq rtn-matched; do
            has_word "$ARMS" "$arm" || continue
            for s in $STATES_NEW; do case " 4 6 " in *" $s "*) ;; *) echo "g $arm $s" ;; esac; done
        done
    fi
    if has_word "$STAGES" p1; then
        for arm in gptq legacy rtn-matched; do
            if has_word "$ARMS" "$arm"; then echo "p1 $arm all"; fi
        done
    fi
    return 0
}

unit_names() {                                                   # <stage> <arm> <token>: sets U_MARK (marker name), U_OUT, U_LOG, U_NART (weight sets the unit builds), U_EXCL (1: runs alone)
    local stage=$1 arm=$2 tok=$3 b n=0
    case "$stage:$arm" in
        g:legacy)  U_MARK=state_$tok; U_OUT=$OUTROOT/g9-lm-$tok; U_LOG=$LOGS/g9-lm-$tok.log; U_NART=0 ;;     # the names of the original driver: a resumed old run keeps its markers
        g:*)       U_MARK=unit_g_${arm}_$tok; U_OUT=$OUTROOT/g9-lm-$arm-$tok; U_LOG=$LOGS/g9-lm-$arm-$tok.log; U_NART=1 ;;
        p1:legacy) U_MARK=unit_p1_legacy; U_OUT=$OUTROOT/g9-p1-legacy; U_LOG=$LOGS/g9-p1-legacy.log; U_NART=0 ;;
        p1:*)      for b in $P1_BITS_NEW; do case $b in 4|5|6|7|8) n=$((n + 1)) ;; esac; done
                   U_MARK=unit_p1_$arm; U_OUT=$OUTROOT/g9-p1-$arm; U_LOG=$LOGS/g9-p1-$arm.log; U_NART=$n ;;
        *) die "unknown unit $stage $arm $tok" ;;
    esac
    U_EXCL=0; if [ "$U_NART" -ge 3 ]; then U_EXCL=1; fi
    return 0
}

unit_command() {                                                 # <stage> <arm> <token> <out dir> -> CMD (an array, run from scripts/g0)
    local stage=$1 arm=$2 tok=$3 out=$4 sel=$OUTROOT/g9-tag/eq/selectivity.pt man=$OUTROOT/g9-tag/groups_manifest.json bits
    local -a wrap=()
    if [ "$arm" != legacy ]; then wrap=("$PY" gptq_ext/run_with_weights.py --entry phase1_gemma9b_ext:main --weights "$arm" --gptq-freeze "$HASHES" --calibration-freeze "$CAL_FREEZE" --calibration-text "$POOL_DIR" --); fi
    if [ "$stage" = g ]; then
        if [ "$arm" = legacy ]; then CMD=("$PY" phase1_gemma9b_run.py lm); else CMD=("${wrap[@]}" lm); fi
        CMD+=(--out "$out" --flores-root "$BASE/data" --selectivity "$sel" --manifest "$man" --commit "$COMMIT" --precisions "$tok" --split eval --n-eval 300 --panel ${LM_EXTRA:-})
    else
        bits=$P1_BITS_NEW; if [ "$arm" = legacy ]; then bits=$P1_BITS_LEGACY; fi
        if [ "$arm" = legacy ]; then CMD=("$PY" phase1_gemma9b_ext.py p1); else CMD=("${wrap[@]}" p1); fi
        CMD+=(--corpus flores --regime mask --bits $bits --out "$out" --flores-root "$BASE/data" --commit "$COMMIT" ${P1_EXTRA:-})
    fi
}

validate_unit() {                                                # <stage> <arm> <out dir> <log>: exit 0 is not trusted
    local stage=$1 arm=$2 out=$3 log=$4
    if [ "$stage" = g ]; then
        check_lm_log "$log" "$out" || return 1
    else
        ! grep -q "CUDA initialization" "$log" || { say "CUDA initialization warning in $log (ran on CPU?)"; return 1; }
        { [ -f "$out/eq/survival.pt" ] && [ -f "$out/all/survival.pt" ]; } || { say "no survival.pt in $out/eq and $out/all"; return 1; }
    fi
    if [ "$arm" != legacy ] && [ ! -f "$out/weights_manifest.json" ]; then say "no weights_manifest.json in $out"; return 1; fi
    return 0
}

unit_job() {                                                     # <stage> <arm> <token> <gpu index>
    local stage=$1 arm=$2 tok=$3 gpu=$4
    unit_names "$stage" "$arm" "$tok"
    local mark=$U_MARK out=$U_OUT log=$U_LOG
    mkdir -p "$out"
    say "unit $stage $arm $tok on GPU $gpu (host RAM estimate $(need_gb "$U_NART") GB)"
    if [ "$arm" != legacy ] && [ ! -f "$HASHES" ]; then
        say "unit $stage $arm $tok REFUSED: no recorded artifact hashes ($HASHES): the gate step has not run"; touch "$STATE_DIR/$mark.failed"; return 1
    fi
    if ! in_g0 "$PY" -m gptq_ext.ram_guard --model-gb "$MODEL_GB" --target-gb "$TARGET_GB" --n-artifacts "$U_NART" --ram-gb "$(ram_gb)" >> "$log" 2>&1; then
        say "unit $stage $arm $tok REFUSED: not enough host RAM (see $log)"; touch "$STATE_DIR/$mark.failed"; return 1
    fi
    unit_command "$stage" "$arm" "$tok" "$out"
    { nvidia-smi -L || true; date; echo "+ ${CMD[*]}"; } >> "$log" 2>&1
    if ( cd "$REPO/scripts/g0" && CUDA_VISIBLE_DEVICES=$gpu "${CMD[@]}" ) >> "$log" 2>&1 && validate_unit "$stage" "$arm" "$out" "$log"; then
        touch "$STATE_DIR/$mark.done"; say "unit $stage $arm $tok finished and validated"
    else
        touch "$STATE_DIR/$mark.failed"; say "unit $stage $arm $tok FAILED, see $log"; return 1
    fi
}

do_full() {
    local -a PLAN=(); local u stage arm tok s i failed=""
    mapfile -t PLAN < <(plan_units)
    if [ "$DRY_RUN" = 1 ]; then
        echo "+ full: tag all 27 languages -> manifest -> ${#PLAN[@]} units (stages: $STAGES; arms: $ARMS; 300 sentences, 3 control draws), $NPROC at a time on GPUs ${GPUS[*]}"
        for u in "${PLAN[@]}"; do
            read -r stage arm tok <<< "$u"; unit_names "$stage" "$arm" "$tok"
            if [ -f "$STATE_DIR/$U_MARK.done" ]; then echo "[skip] unit $stage $arm $tok (done)"; continue; fi
            unit_command "$stage" "$arm" "$tok" "$U_OUT"
            echo "+ UNIT $stage $arm $tok :: ${CMD[*]}"
            echo "  host RAM estimate for unit $stage $arm $tok: $(need_gb "$U_NART") GB$([ "$U_EXCL" = 1 ] && echo ', runs alone')"
        done
        return 0
    fi
    tag_and_manifest g9-tag
    local -a slot_pid=()
    for u in "${PLAN[@]}"; do
        read -r stage arm tok <<< "$u"; unit_names "$stage" "$arm" "$tok"
        if [ -f "$STATE_DIR/$U_MARK.done" ]; then say "[skip] unit $stage $arm $tok (done)"; continue; fi
        rm -f "$STATE_DIR/$U_MARK.failed"
        if [ "$U_EXCL" = 1 ]; then                               # several weight sets in host RAM: run alone, after everything running has finished
            wait
            unit_job "$stage" "$arm" "$tok" "${GPUS[0]}" < /dev/null || true
            slot_pid=()
            continue
        fi
        while true; do                                           # wait for a free slot (a slot is one process; PROCS_PER_GPU slots share a GPU)
            s=-1
            for ((i = 0; i < NPROC; i++)); do
                if [ -z "${slot_pid[$i]:-}" ] || ! kill -0 "${slot_pid[$i]}" 2>/dev/null; then s=$i; break; fi
            done
            if [ "$s" -ge 0 ]; then break; fi
            sleep "${POLL_SECONDS:-30}"
        done
        ( unit_job "$stage" "$arm" "$tok" "${GPUS[$((s / PROCS_PER_GPU))]}" ) < /dev/null &
        slot_pid[$s]=$!
    done
    wait
    for u in "${PLAN[@]}"; do
        read -r stage arm tok <<< "$u"; unit_names "$stage" "$arm" "$tok"
        [ -f "$STATE_DIR/$U_MARK.done" ] || failed="$failed $stage:$arm:$tok"
    done
    [ -z "$failed" ] || { say "UNITS NOT FINISHED:$failed (run the same command again to retry them; logs in $LOGS)"; return 1; }
}

do_package() {
    mkdir -p "$BASE/out/frozen"
    for f in "$HASHES" "$AMEND" "$FREEZE_DIR/freeze.json" "$STATE_DIR/tokenizer_check.json"; do if [ -f "$f" ]; then cp -f "$f" "$BASE/out/frozen/"; fi; done
    ( cd "$BASE" && { echo "# sha256 of every output file"; find out -type f | sort | xargs sha256sum; } > artifact_checksums.md
      local f files=(out logs artifact_checksums.md)
      for f in state/pip_freeze.txt run_all.log repo/COMMIT repo/reports/functional_validation/gemma9b/freeze.json; do [ -e "$f" ] && files+=("$f"); done
      tar czf gemma9b_results.tar.gz "${files[@]}" )
    say "results: $BASE/gemma9b_results.tar.gz ($(du -h "$BASE/gemma9b_results.tar.gz" | cut -f1))"
}

CUR_STEP=verify_frozen; say "== verify_frozen"; verify_frozen
step env do_env
step flores do_flores
step access_and_download do_access
step checks do_checks
if need_weights; then step tokenizer do_tokenizer; step pool do_pool; fi
if [ "$SKIP_SMOKE" != 1 ]; then
    step smoke do_smoke
    if has_word "$STAGES" p1; then step smoke_p1 do_smoke_p1; fi
fi
if need_weights; then
    step gate9b do_gate9b
    if [ "$SKIP_SMOKE" != 1 ]; then step smoke_gptq do_smoke_gptq; fi
fi
rc=0; CUR_STEP=full
if [ -f "$STATE_DIR/full.done" ]; then
    say "[skip] full (already done)"
else
    say "== full"
    do_full || rc=$?                                             # failed units come back as a status (do_full reports its own errors); the packaging below must still run
    if [ "$rc" -eq 0 ] && [ "$DRY_RUN" != 1 ]; then touch "$STATE_DIR/full.done"; fi
fi
stop_after full
if [ "$DRY_RUN" != 1 ]; then do_package; fi
if [ "$rc" -eq 0 ]; then say "ALL DONE. Send $BASE/gemma9b_results.tar.gz back."; else say "PARTIAL: some states failed (see above). Send $BASE/gemma9b_results.tar.gz and $BASE/run_all.log back."; exit "$rc"; fi
