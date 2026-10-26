#!/bin/bash
# Re-run the analyses of the functional study from their input files and compare every output with the committed one (docs: reports/functional_validation/REPRODUCE.md).
#   bash scripts/g0/reproduce_functional.sh <inputs root> <scratch output dir>
# Inputs root layout (files fetched from Ada one by one, sha256 checked against reports/functional_validation/artifact_checksums.md; job names are Ada's under out/phase1/):
#   eval/       func_lm_eval_{FP,INT0,INT12,INT16,INT8,INT7,INT6,INT5,INT4}.pt + meta_lm_eval.json          (func-eval-a, -b, -c, -d)
#   rescue/     func-rescue-a files            f4-INT4/ f4-INT5/ f4-INT6/   func-f4-INT<b> files
#   mc-eval/    func-mc-eval-a and -b files (their two meta_mc_eval.json renamed meta_mc_eval_a.json and meta_mc_eval_b.json)      mc-ksens/   func-mc-ksens files
#   f5-a/ f5-b/ f5-smoke2/ f5-m8/ f5-m32/ f5-dir/ f5-dir2/ f5-fa/ f5-fa-dir/    the per-state files of the F5 jobs
#   d035/ profiles.pt selectivity.pt (Study G baseline); gain-cal-FP/ gain-cal-INT4/ gain-FP/ gain-8/ gain-7/ gain-6/ gain-5/ gain-4/   the item files and metadata of the Study G jobs of the same names (D-070 to D-080); gainc-FP/ gainc-6/ gainc-5/ gainc-4b/   the extension C jobs (D-082); gain-sd-FP/ gain-sd-6/ gain-sd-4/ gain-sd-FPref-6/ gain-sd-FPref-4/   the extension B jobs (D-085); promptcheck-frozen/ promptcheck-harness/   the two variant directories of the Task 17 (b) job (D-086); llama-lm-a/ llama-lm-b/ llama-tag/   the Llama-3.2-1B check (D-090: item files and meta of the two lm jobs, and llama-tag/groups_manifest.json)
#   mc-pc-all/ mc-pc-1024/ mc-pc-256/   func-mc-pc-<dose> files func_mc_eval_letter_FP.pt and meta_mc_eval.json (the positive control, D-069)
#   feature_status_eq.csv   Ada association-full/analysis-eq/feature_status.csv
# Prints identical / DIFFERS / skipped (input missing) for every output file; exit 1 if anything DIFFERS. Needs the torch venv (numpy, scipy, statsmodels).
# Known, benign differences are compared with the field ignored and say so: the run paths and the flags direction_control / fixed_amplitude (added to the F5 metadata after the main run), the figure links of f5/report.md, the free-text note of mc_ksens.
set -u
IN=${1:?inputs root}; OUT=${2:?scratch output dir}; ROOT=$(cd "$(dirname "$0")/../.." && pwd); RES=$ROOT/reports/functional_validation/results; PY=${PY:-$HOME/.venvs/anlp-torch/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}; mkdir -p "$OUT"; FAIL=0
cmpf() { # new committed label
    if [ ! -e "$1" ]; then echo "skipped   $3 (not produced)"; return; fi
    if cmp -s "$1" "$2"; then echo "identical $3"; else echo "DIFFERS   $3"; FAIL=1; fi
}
cmpdir() { # new_dir committed_dir label files...
    n=$1; c=$2; l=$3; shift 3; for f in "$@"; do cmpf "$n/$f" "$c/$f" "$l/$f"; done
}
jsoneq() { # new committed label ignored-key-paths (dot-separated, comma-separated)
    if [ ! -e "$1" ]; then echo "skipped   $3 (not produced)"; return; fi
    if $PY -I - "$1" "$2" "$4" <<'PYEOF'
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
for path in filter(None, sys.argv[3].split(",")):
    for d in (a, b):
        x = d; ks = path.split(".")
        for k in ks[:-1]:
            x = x.get(k, {}) if isinstance(x, dict) else {}
        if isinstance(x, dict): x.pop(ks[-1], None)
sys.exit(0 if a == b else 1)
PYEOF
    then echo "identical $3 (apart from: ${4:-nothing})"; else echo "DIFFERS   $3"; FAIL=1; fi
}
have() { [ -d "$IN/$1" ] || [ -e "$IN/$1" ] || { echo "skipped   $2 (missing input $1)"; return 1; }; }
cd "$ROOT" || exit 2
COV=$($PY -I -c "import json; print(' '.join(json.load(open('reports/functional_validation/evaluation_freeze_mc.json'))['calibration']['covered']))")
if have eval "results/eval"; then $PY -B scripts/g0/phase1_func_analysis.py --runs "$IN/eval" --split eval --out "$OUT/eval" --K K64 --n-boot 2000 > /dev/null; cmpdir "$OUT/eval" "$RES/eval" eval analysis.json effects_by_language.csv families.csv fidelity.csv retention.csv report.md; fi
if have eval "results/eval_subsets"; then       # D-068: the same analysis restricted to the 18 languages with Belebele headroom and to the 9 without
    UNC=$($PY -I -c "import json; c=set('$COV'.split()); print(' '.join(l for l in json.load(open('reports/functional_validation/results/eval/analysis.json'))['covered'] if l not in c))")
    $PY -B scripts/g0/phase1_func_analysis.py --runs "$IN/eval" --split eval --out "$OUT/eval_subsets/covered18" --K K64 --n-boot 2000 --covered $COV > /dev/null; cmpdir "$OUT/eval_subsets/covered18" "$RES/eval_subsets/covered18" eval_subsets/covered18 analysis.json effects_by_language.csv families.csv fidelity.csv retention.csv report.md
    $PY -B scripts/g0/phase1_func_analysis.py --runs "$IN/eval" --split eval --out "$OUT/eval_subsets/uncovered9" --K K64 --n-boot 2000 --covered $UNC > /dev/null; cmpdir "$OUT/eval_subsets/uncovered9" "$RES/eval_subsets/uncovered9" eval_subsets/uncovered9 analysis.json effects_by_language.csv families.csv fidelity.csv retention.csv report.md
fi
if have rescue "results/rescue"; then $PY -B scripts/g0/phase1_func_analysis.py --runs "$IN/rescue" --split eval --out "$OUT/rescue" --K K64 --n-boot 2000 > /dev/null; cmpdir "$OUT/rescue" "$RES/rescue" rescue analysis.json effects_by_language.csv families.csv fidelity.csv rescue.csv retention.csv report.md; fi
for b in INT4 INT5 INT6; do if have f4-$b "results/f4/$b"; then $PY -B scripts/g0/phase1_func_f4_analysis.py --runs "$IN/f4-$b" --split eval --condition $b --out "$OUT/f4/$b" > /dev/null; cmpdir "$OUT/f4/$b" "$RES/f4/$b" f4/$b f4_per_language.csv f4_results.json report.md; fi; done
if have mc-eval "results/mc_eval"; then
    $PY -B scripts/g0/phase1_func_mc_analysis.py --runs "$IN/mc-eval" --split eval --format letter --out "$OUT/mc_eval" --K K64 --n-boot 2000 --seed 0 --retention-level 0.99 --covered $COV --note "Held-out T-MC evaluation (devtest-origin passages); the confirmatory family of the design, settings frozen before scoring." > /dev/null 2>&1
    cmpdir "$OUT/mc_eval" "$RES/mc_eval" mc_eval analysis.json accuracy.csv effects_by_language.csv retention.csv report.md
    $PY -B scripts/g0/phase1_func_mc_statements.py --freeze reports/functional_validation/evaluation_freeze_mc.json --analysis "$OUT/mc_eval/analysis.json" --out "$OUT/mc_statements" > /dev/null; cmpdir "$OUT/mc_statements" "$RES/mc_eval" mc_eval statements.csv statements.json statements.md
fi
if have mc-ksens "results/mc_ksens"; then
    $PY -B scripts/g0/phase1_func_mc_analysis.py --runs "$IN/mc-ksens" --split eval --format letter --out "$OUT/mc_ksens" --K K64 --n-boot 2000 --seed 0 --retention-level 0.99 --covered $COV --note "x" > /dev/null 2>&1
    cmpdir "$OUT/mc_ksens" "$RES/mc_ksens" mc_ksens accuracy.csv effects_by_language.csv retention.csv; jsoneq "$OUT/mc_ksens/analysis.json" "$RES/mc_ksens/analysis.json" mc_ksens/analysis.json note
fi
if have f5-a "results/f5" && have f5-b "results/f5"; then
    $PY -B scripts/g0/phase1_func_f5_analysis.py --run "$IN/f5-a" --run "$IN/f5-b" --out "$OUT/f5" > /dev/null; cmpdir "$OUT/f5" "$RES/f5" f5 f5_tables.csv statements.md; jsoneq "$OUT/f5/f5_results.json" "$RES/f5/f5_results.json" f5/f5_results.json instrument.runs,instrument.direction_control,instrument.fixed_amplitude
    A="$IN/f5-a"; B="$IN/f5-b"; S=reports/functional_validation/results/f5_sensitivity
    sens() { name=$1; shift; $PY -B scripts/g0/phase1_func_f5_sensitivity.py "$@" --label x --out "$OUT/sens/$name" > /dev/null 2>&1; cmpf "$OUT/sens/$name/sensitivity.json" "$RES/f5_sensitivity/$name/sensitivity.json" "f5_sensitivity/$name/sensitivity.json"; }
    have f5-m8 "f5_sensitivity/m8" && sens m8 --main-run "$A" --main-run "$B" --sens-run "$IN/f5-m8" --states INT16 INT4
    have f5-m32 "f5_sensitivity/m32" && sens m32 --main-run "$A" --main-run "$B" --sens-run "$IN/f5-m32" --states INT16 INT4
    have f5-dir "f5_sensitivity/direction_swap" && sens direction_swap --main-run "$A" --main-run "$B" --sens-run "$IN/f5-dir" --states INT16 INT4
    have f5-fa "f5_sensitivity/fixed_amplitude" && sens fixed_amplitude --main-run "$A" --main-run "$B" --sens-run "$IN/f5-fa" --states INT16 INT4
    have f5-fa "f5_sensitivity/fixed_amplitude_direction_swap" && have f5-fa-dir "f5_sensitivity/fixed_amplitude_direction_swap" && sens fixed_amplitude_direction_swap --main-run "$IN/f5-fa" --sens-run "$IN/f5-fa-dir" --states INT16 INT4
    have f5-dir2 "f5_sensitivity/direction_swap_int8_6_5" && sens direction_swap_int8_6_5 --main-run "$A" --main-run "$B" --sens-run "$IN/f5-dir2" --states INT16 INT8 INT6 INT5
    if have f5-dir2 "figures/f5/ladder_control_table.csv" && have f5-dir "figures/f5/ladder_control_table.csv"; then
        $PY -B scripts/g0/phase1_func_f5_figures.py ladder --main-run "$A" --main-run "$B" --control-run "$IN/f5-dir2" --control-run "$IN/f5-dir" --out "$OUT/ladder" --note x --formats svg > /dev/null 2>&1; cmpf "$OUT/ladder/ladder_control_table.csv" reports/functional_validation/figures/f5/ladder_control_table.csv figures/f5/ladder_control_table.csv
    fi
fi
if have f5-smoke2 "results/f5_smoke"; then $PY -B scripts/g0/phase1_func_f5_analysis.py --run "$IN/f5-smoke2" --out "$OUT/f5_smoke" > /dev/null; cmpdir "$OUT/f5_smoke" "$RES/f5_smoke" f5_smoke f5_tables.csv statements.md; jsoneq "$OUT/f5_smoke/f5_results.json" "$RES/f5_smoke/f5_results.json" f5_smoke/f5_results.json instrument.runs,instrument.direction_control,instrument.fixed_amplitude; fi
if have feature_status_eq.csv "the association analyses"; then
    $PY -B scripts/g0/phase1_assoc_rank_link.py --status "$IN/feature_status_eq.csv" --out "$OUT/assoc_rank" > /dev/null; cmpdir "$OUT/assoc_rank" "$RES/assoc_rank" assoc_rank $(ls "$RES/assoc_rank")
    $PY -B scripts/g0/phase1_assoc_margin.py --status "$IN/feature_status_eq.csv" --out "$OUT/within-language" > /dev/null; cmpdir "$OUT/within-language" reports/association_retention/results/within-language within-language analysis.json report.md loss_by_margin_quintile.csv
    if have f5-a "margin-function" && have f5-b "margin-function"; then $PY -B scripts/g0/phase1_assoc_function_link.py --status "$IN/feature_status_eq.csv" --run "$IN/f5-a" --run "$IN/f5-b" --out "$OUT/margin-function" > /dev/null; cmpdir "$OUT/margin-function" reports/association_retention/results/margin-function margin-function analysis.json report.md margin_vs_function.csv; fi
fi
for d in all 1024 256; do       # D-069, D-072: the positive control of the Belebele edit path, one run per dose (the committed analysis.json supplies the note of the run)
    if have mc-pc-$d "results/mc_posctrl/dose_$d"; then
        NOTE=$($PY -I -c "import json, sys; print(json.load(open(sys.argv[1]))['note'])" "$RES/mc_posctrl/dose_$d/analysis.json")
        $PY -B scripts/g0/phase1_func_mc_analysis.py --runs "$IN/mc-pc-$d" --split eval --format letter --out "$OUT/mc_posctrl/dose_$d" --K K64 --n-boot 2000 --seed 0 --retention-level 0.99 --covered $COV --note "$NOTE" > /dev/null 2>&1
        cmpdir "$OUT/mc_posctrl/dose_$d" "$RES/mc_posctrl/dose_$d" mc_posctrl/dose_$d analysis.json accuracy.csv effects_by_language.csv retention.csv report.md
    fi
done
if [ -e "$OUT/mc_posctrl/dose_all/accuracy.csv" ] && [ -e "$OUT/mc_posctrl/dose_1024/accuracy.csv" ] && [ -e "$OUT/mc_posctrl/dose_256/accuracy.csv" ]; then
    $PY -B scripts/g0/phase1_func_posctrl_summary.py --dose all "$OUT/mc_posctrl/dose_all/accuracy.csv" --dose 1024 "$OUT/mc_posctrl/dose_1024/accuracy.csv" --dose 256 "$OUT/mc_posctrl/dose_256/accuracy.csv" --out "$OUT/mc_posctrl" > /dev/null
    cmpdir "$OUT/mc_posctrl" "$RES/mc_posctrl" mc_posctrl summary.md summary.json
fi
# Study G (graded gain, D-070 to D-080): baseline, calibration, the main analysis and the two previews; inputs d035/{profiles.pt,selectivity.pt}, gain-cal-FP, gain-cal-INT4, gain-FP, gain-8, gain-7, gain-6, gain-5, gain-4
GS=reports/functional_validation/gain_study; G="$OUT/gain"
if have d035 "$GS/baseline" && [ -e "$IN/d035/profiles.pt" ] && [ -e "$IN/d035/selectivity.pt" ]; then
    $PY -B scripts/g0/phase1_gain_baseline.py --profiles "$IN/d035/profiles.pt" --selectivity "$IN/d035/selectivity.pt" --manifest reports/functional_validation/groups_manifest.json --out "$G/baseline" > /dev/null
    cmpdir "$G/baseline" "$ROOT/$GS/baseline" "$GS/baseline" baseline.csv baseline.json report.md
fi
if have gain-cal-FP "$GS/calibration"; then
    $PY -B scripts/g0/phase1_gain_ratios.py --runs "$IN/gain-cal-FP" --out "$G/calibration" --n-controls 2 > /dev/null
    cmpdir "$G/calibration" "$ROOT/$GS/calibration" "$GS/calibration" calibration.csv calibration.md ratios.json
    if have gain-cal-INT4 "$GS/calibration_int4_check"; then
        $PY -B scripts/g0/phase1_gain_ratios.py --runs "$IN/gain-cal-FP" "$IN/gain-cal-INT4" --out "$G/calibration_int4_check" --n-controls 2 > /dev/null
        cmpdir "$G/calibration_int4_check" "$ROOT/$GS/calibration_int4_check" "$GS/calibration_int4_check" calibration.csv calibration.md ratios.json
    fi
fi
gain_an() { # name out-name dirs...   (the analysis lists the job directories in its metadata: that field is ignored)
    name=$1; shift; runs=(); for d in "$@"; do have "$d" "$GS/$name" || return 0; runs+=("$IN/$d"); done
    $PY -B scripts/g0/phase1_gain_analysis.py --runs "${runs[@]}" --ratios-json $GS/calibration/ratios.json --baseline-csv $GS/baseline/baseline.csv --out "$G/$name" > /dev/null
    cmpdir "$G/$name" "$ROOT/$GS/$name" "$GS/$name" curves.csv alpha_star.csv reliability.csv report.md; jsoneq "$G/$name/analysis.json" "$ROOT/$GS/$name/analysis.json" "$GS/$name/analysis.json" meta.runs
}
gain_an results gain-FP gain-8 gain-7 gain-6 gain-5 gain-4
gain_an preview_FP_INT4 gain-FP gain-4
gain_an preview_FP_INT6_INT5_INT4 gain-FP gain-6 gain-5 gain-4
# Study G extension C (D-082): the repair near ratio 1 with 562 reserve sentences and five control draws; inputs gainc-FP, gainc-6, gainc-5, gainc-4b (func_lm_eval_<state>.pt and meta_lm_eval.json of the jobs of the same names); the figure is recomputed from bands.csv
if [ -d "$ROOT/$GS/extension_c" ]; then
    ok=1; runs=(); for d in gainc-FP gainc-6 gainc-5 gainc-4b; do have "$d" "$GS/extension_c" || { ok=0; break; }; runs+=("$IN/$d"); done
    if [ $ok -eq 1 ]; then
        $PY -B scripts/g0/phase1_gain_extc.py --runs "${runs[@]}" --out "$G/extension_c" > /dev/null
        cmpdir "$G/extension_c" "$ROOT/$GS/extension_c" "$GS/extension_c" cells.csv bands.csv optimum.csv reliability.csv report.md; jsoneq "$G/extension_c/analysis.json" "$ROOT/$GS/extension_c/analysis.json" "$GS/extension_c/analysis.json" meta.runs
        $PY -B scripts/g0/phase1_gain_extc_figures.py --bands "$G/extension_c/bands.csv" --out "$G/figures_extension_c" --formats svg --note "exploratory (devtest, one model, layer and quantizer); 562 held-out sentences per language, 5 matched-control draws" > /dev/null
        cmpdir "$G/figures_extension_c" "$ROOT/$GS/figures_extension_c" "$GS/figures_extension_c" fig_gc1_repair_light.svg fig_gc1_repair_dark.svg
    fi
fi
# Study G extension D (D-084): the optimal gain against the shrinkage; input gainc-4b and the committed baseline.csv
if [ -d "$ROOT/$GS/extension_d" ] && have gainc-4b "$GS/extension_d"; then
    $PY -B scripts/g0/phase1_gain_extd.py --runs "$IN/gainc-4b" --baseline-csv $GS/baseline/baseline.csv --out "$G/extension_d" > /dev/null
    cmpdir "$G/extension_d" "$ROOT/$GS/extension_d" "$GS/extension_d" alpha.csv report.md; jsoneq "$G/extension_d/analysis.json" "$ROOT/$GS/extension_d/analysis.json" "$GS/extension_d/analysis.json" meta.runs
    $PY -B scripts/g0/phase1_gain_extd_figures.py --alpha "$G/extension_d/alpha.csv" --analysis "$G/extension_d/analysis.json" --out "$G/figures_extension_d" --formats svg --note "exploratory (post hoc in origin, D-083, D-084)" > /dev/null
    cmpdir "$G/figures_extension_d" "$ROOT/$GS/figures_extension_d" "$GS/figures_extension_d" fig_gd1_alpha_light.svg fig_gd1_alpha_dark.svg
fi
# Study G extension B (D-085): state-defined groups; inputs gain-sd-FP, gain-sd-6, gain-sd-4 (state-defined) and gain-sd-FPref-6, gain-sd-FPref-4 (the FP reference groups at INT6 and INT4; the FP reference at FP is gain-sd-FP), with the committed ratios.json and overlap.csv
if [ -d "$ROOT/$GS/state_defined/results" ]; then
    ok=1; for d in gain-sd-FP gain-sd-6 gain-sd-4 gain-sd-FPref-6 gain-sd-FPref-4; do have "$d" "$GS/state_defined/results" || { ok=0; break; }; done
    if [ $ok -eq 1 ]; then
        $PY -B scripts/g0/phase1_gain_extb.py --state-defined "$IN/gain-sd-FP" "$IN/gain-sd-6" "$IN/gain-sd-4" --reference "$IN/gain-sd-FP" "$IN/gain-sd-FPref-6" "$IN/gain-sd-FPref-4" --ratios-json $GS/calibration/ratios.json --overlap $GS/state_defined/overlap.csv --out "$G/state_defined_results" > /dev/null
        cmpdir "$G/state_defined_results" "$ROOT/$GS/state_defined/results" "$GS/state_defined/results" compare.csv report.md; jsoneq "$G/state_defined_results/analysis.json" "$ROOT/$GS/state_defined/results/analysis.json" "$GS/state_defined/results/analysis.json" meta.state_defined_runs,meta.reference_runs
        $PY -B scripts/g0/phase1_gain_extb_figures.py --compare "$G/state_defined_results/compare.csv" --out "$G/figures_state_defined" --formats svg --note "exploratory (devtest, one model, layer and quantizer); 300 held-out sentences per language, 3 matched-control draws" > /dev/null
        cmpdir "$G/figures_state_defined" "$ROOT/$GS/figures_state_defined" "$GS/figures_state_defined" fig_gb1_state_defined_light.svg fig_gb1_state_defined_dark.svg
    fi
fi
# Task 17 part (b) (D-086): the prompt cross-check; inputs promptcheck-frozen/ and promptcheck-harness/ (the frozen/ and harness/ directories of the Ada job `promptcheck`: func_mc_cal_letter_FP.pt and meta_mc_cal.json each) and the committed belebele_calibration.json
if [ -d "$ROOT/reports/functional_validation/promptcheck" ] && have promptcheck-frozen "promptcheck" && have promptcheck-harness "promptcheck"; then
    $PY -B scripts/g0/phase1_func_promptcheck_analysis.py --frozen-dir "$IN/promptcheck-frozen" --harness-dir "$IN/promptcheck-harness" --calibration-json reports/functional_validation/belebele_calibration.json --out "$OUT/promptcheck" > /dev/null
    cmpdir "$OUT/promptcheck" "$RES/../promptcheck" promptcheck languages.csv report.md; jsoneq "$OUT/promptcheck/analysis.json" "$RES/../promptcheck/analysis.json" promptcheck/analysis.json meta.frozen,meta.harness
fi
# Llama-3.2-1B consistency check (D-090): the pre-registered comparison and the post hoc robustness tables; inputs llama-lm-a/ (FP, INT8, INT6), llama-lm-b/ (INT5, INT4: func_lm_eval_<state>.pt and meta_lm_eval.json each) and llama-tag/groups_manifest.json; the Gemma reference is read from the committed outputs, the ratios from the committed ratios.json
if [ -d "$ROOT/reports/functional_validation/llama" ] && have llama-lm-a "llama" && have llama-lm-b "llama" && have llama-tag/groups_manifest.json "llama"; then
    $PY -B scripts/g0/phase1_llama_compare.py --runs "$IN/llama-lm-a" "$IN/llama-lm-b" --manifest "$IN/llama-tag/groups_manifest.json" --ratios-json reports/functional_validation/gain_study/calibration/ratios.json --gemma-root reports/functional_validation --out "$OUT/llama" --sae fineweb42 > /dev/null
    cmpf "$OUT/llama/report.md" "$ROOT/reports/functional_validation/llama/report.md" llama/report.md; jsoneq "$OUT/llama/analysis.json" "$ROOT/reports/functional_validation/llama/analysis.json" llama/analysis.json meta.runs
    $PY -B scripts/g0/phase1_llama_robust.py --runs "$IN/llama-lm-a" "$IN/llama-lm-b" --manifest "$IN/llama-tag/groups_manifest.json" --out "$OUT/llama_robust" > /dev/null
    cmpdir "$OUT/llama_robust" "$ROOT/reports/functional_validation/llama_robust" llama_robust languages.csv report.md; jsoneq "$OUT/llama_robust/robust.json" "$ROOT/reports/functional_validation/llama_robust/robust.json" llama_robust/robust.json meta.runs
fi
[ $FAIL -eq 0 ] && echo "all compared outputs are identical" || { echo "SOME OUTPUTS DIFFER"; exit 1; }
