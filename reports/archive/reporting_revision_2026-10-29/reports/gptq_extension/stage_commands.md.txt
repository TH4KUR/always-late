# Stage commands of the GPTQ extension (ready to queue; written 2026-10-27 while gate jobs A and B ran)

Every stage is the OLD study's entry point and arguments, unchanged, through `run_gptq.sbatch ... wrap --entry <module:function> --weights <gptq|rtn-matched> -- <old arguments>` (plan: "how a stage is run"). `SNAP` is the deployed snapshot of the then-current commit (`ADA_SSH='ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30' bash scripts/g0/deploy_phase1.sh repo_<short commit>`; deploy ONCE, check `COMMIT` before submitting), `FROZEN=1,RECORD=1` (D-103): a job refuses an artifact whose hash differs from the recorded one (`reports/gptq_extension/artifact_hashes.json`, bit width 4 from gate jobs A and B) and records the hashes of the other bit widths in its `weights_manifest.json`; the manifests of all jobs are compared after the fact. All jobs: `-A nlp --exclude=gnode068,gnode069,gnode076`, `--export=ALL,REPO=$SNAP,FROZEN=1,RECORD=1`, read the `GPU 0:` line and the first lines of every log. At most two jobs RUN at once (the scheduler enforces it for this account's habit; queue all).

Template (run on Ada, `cd /home2/karma/duan-verify`, `SNAP=/home2/karma/duan-verify/phase1/repo_<short commit>`, `S="sbatch -A nlp --exclude=gnode068,gnode069,gnode076 --export=ALL,REPO=$SNAP,FROZEN=1,RECORD=1"`, `R=$SNAP/scripts/g0/run_gptq.sbatch`):

**Stage 1 (E1 and E2; amendment A1: the whole ladder, defaults of the old `p1-mask` run; old limit 10 h):** one job per arm, bits `8 7 6 5 4 16 12 0` (the old default, passed explicitly so that the wrapper builds all five artifacts BEFORE the job starts: a hash mismatch or a memory error then stops it at once):
- `$S -t 10:00:00 $R gptq-p1-mask wrap --entry phase1_run:main --weights gptq -- --corpus flores --regime mask --bits 8 7 6 5 4 16 12 0 --out {OUT} --flores-root {BASE}/phase1/data --commit {COMMIT}`
- `$S -t 10:00:00 $R rtnm-p1-mask wrap --entry phase1_run:main --weights rtn-matched -- --corpus flores --regime mask --bits 8 7 6 5 4 16 12 0 --out {OUT} --flores-root {BASE}/phase1/data --commit {COMMIT}`
Outputs `out/phase1/<name>/{eq,all}/` (`survival.pt`, `per_language.csv`, ...). Analysis: the unchanged `phase1_analysis.py` on each arm; the old `p1-mask` files are the legacy arm. FP numbers must equal the old run's (bridge gate).

**Stage 2 (E6 transfer matrix, INT4 and INT6, FLORES; old limits 6 h feature, 5 h group; old arguments of the jobs 40399 to 40514):** per arm four jobs (the INT16 floor jobs are the unchanged legacy rounding and identical in both arms: reuse the old `transfer-feature-int16` and `transfer-group-int16` outputs in the analysis, to be confirmed when the analysis is prepared):
- feature level: `$S -t 06:00:00 $R <arm>-transfer-feature-int<N> wrap --entry phase1_transfer_run:main --weights <arm> -- --out {OUT} --flores-root {BASE}/phase1/data --freeze {REPO}/reports/intervention_transfer/freeze.json --core {REPO}/reports/intervention_transfer/core_sample.json --commit {COMMIT} --states <N> --chunk-features 25`
- group level: `$S -t 05:00:00 $R <arm>-transfer-group-int<N> wrap --entry phase1_transfer_run:main --weights <arm> -- --out {OUT} --flores-root {BASE}/phase1/data --freeze {REPO}/reports/intervention_transfer/freeze.json --core {REPO}/reports/intervention_transfer/core_sample.json --commit {COMMIT} --task group --states <N> --chunk-sentences 60`
with `<arm>` in `gptq`, `rtnm` (job name prefix) and `<N>` in 4, 6 (the `--weights` value is `gptq` or `rtn-matched`). Validation per job: `phase1_transfer_validate.py` as in `reports/intervention_transfer/STATUS.md` (sacct, `GPU 0:` line, meta complete, gates, item counts, sha256), plus the artifact hash of `weights_manifest.json` against the recorded one. Cost: about 13 GPU hours per arm.

**Later stages (only if time, in the order of the freeze):** functional panels E4/E5 (`phase1_func_run:main`, the T-LM evaluation commands of `reports/functional_validation/STATUS.md`), E7 (`phase1_func_run:main_mc`), F4/F5 (`phase1_func_f5_run:main`), Study G, the external panel (`phase1_transfer_run:main --external-panel ...`), association retention E3 (its 2.7 GB sidecar does not fit the disk: last).

## Analysis commands (written 2026-10-27 21:30; run after the jobs of an arm are validated)

Raw job outputs stay outside git (`/home/karma/Documents/ANLP/project/transfer_runs/`, as for the old study). Fetch one directory per job (about 25 MB for a feature-level job, 21 MB for a group-level job; scp from Ada runs at about 0.26 MB/s, so fetch only what the analysis reads, one `scp -r -o HostKeyAlias=ada-gw1 karma@10.4.24.30:/home2/karma/duan-verify/out/phase1/<job dir> <dest>` per job, and compare sha256 with Ada's after the copy has finished). Stage 1: fetch `out/phase1/<arm>-p1-mask/{eq,all}` (about 20 MB each: `survival.pt`, `per_language.csv`, `summary.csv`, `meta.json`, `selectivity.pt`).

**Transfer report, ONCE per arm, the unchanged code and the main study's arguments** (INT16 floors = the legacy rounding, identical in all arms: the OLD floor directories are reused; `--survival` is the arm's own stage-1 `all/survival.pt`):
```
OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B scripts/g0/phase1_transfer_report.py --freeze reports/intervention_transfer/freeze.json --core reports/intervention_transfer/core_sample.json \
  --out <transfer_runs/analysis_ARM> --feature-dir INT16=<transfer_runs/transfer-feature-int16> --feature-dir INT6=<transfer_runs/ARM-transfer-feature-int6> --feature-dir INT4=<transfer_runs/ARM-transfer-feature-int4> \
  --group-dir INT16=<transfer_runs/transfer-group-int16> --group-dir INT6=<transfer_runs/ARM-transfer-group-int6> --group-dir INT4=<transfer_runs/ARM-transfer-group-int4> --survival <transfer_runs/ARM-p1-mask-all/survival.pt>
```
with `ARM` = `gptq` and `rtnm`. Then the comparison (reads `claim_status.json`, `component_summary.csv`, `group_per_language.csv`; the legacy arm is the committed `reports/intervention_transfer/results/main`):
```
python3 scripts/g0/gptq_ext/compare_transfer.py --legacy reports/intervention_transfer/results/main --matched <analysis_rtnm> --gptq <analysis_gptq> --roster reports/g0/phase1_languages.json --out reports/gptq_extension/results/stage2
```
**Stage 1:** `OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B scripts/g0/phase1_analysis.py ...` per arm exactly as the old `p1-mask` analysis (arguments in the Phase 1 handoffs, 001 to 004), then
```
OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B scripts/g0/gptq_ext/compare_stage1.py --legacy ~/.cache/anlp-always-late/phase1_runs/p1-mask/eq --matched <rtnm-p1-mask/eq> --gptq <gptq-p1-mask/eq> --roster reports/g0/phase1_languages.json --out reports/gptq_extension/results/stage1
```

## Stage 3 analysis (E4 loss task, E5 rescue; written 2026-10-28 06:35)

Inputs live in `/home/karma/Documents/ANLP/project/transfer_runs/func_inputs/` (raw files stay outside git). **Reference files of the legacy arm, fetched once from Ada with sha256 equal to Ada's (8 files):** `legacy_eval/` = `func-eval-a` FP and INT6, `func-eval-b` INT4, `func-eval-d` INT16, INT12, INT0 (the floors are the legacy rounding and identical in every arm), `legacy_rescue/` = `func-rescue-a` INT4 and INT6 plus the FP file. For an arm `ARM` (`gptq`, `rtnm`) build `ARM_eval/` = the legacy FP and floor files + the arm's `func_lm_eval_INT6.pt` and `func_lm_eval_INT4.pt` (jobs `ARM-func-eval-int6|4`) and `ARM_rescue/` = the legacy FP file + the arm's two rescue files (jobs `ARM-func-rescue-int6|4`); validate each job first (sacct, `GPU 0:`, `meta_lm_eval.json` complete with the four gates true, sha256 after the copy, `weights_manifest.json`). Then, per arm, the unchanged analysis exactly as `REPRODUCE.md` lists it:
```
OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B scripts/g0/phase1_func_analysis.py --runs <func_inputs/ARM_eval>   --split eval --out <func_inputs/analysis_ARM_eval>   --K K64 --n-boot 2000
OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B scripts/g0/phase1_func_analysis.py --runs <func_inputs/ARM_rescue> --split eval --out <func_inputs/analysis_ARM_rescue> --K K64 --n-boot 2000
```
**Control done 2026-10-28 06:33:** the same command on `legacy_eval` (FP, INT16, INT12, INT0, INT6, INT4) reproduces EVERY INT6 and INT4 row of the committed `reports/functional_validation/results/eval/effects_by_language.csv` (162 rows) and `retention.csv` (145 rows) exactly (0 differ). The legacy arm of the comparison is the committed `results/eval` and `results/rescue`.

## F5 and Study G analysis (written 2026-10-28 11:20)

Inputs in `/home/karma/Documents/ANLP/project/transfer_runs/func_inputs/` (raw files stay outside git). **Reference files of the legacy arm, fetched once from Ada with sha256 equal to Ada's (8 files):** `legacy_f5_a/` = `f5-a` (`f5_stage_a.pt`, `f5_state_FP.pt`, `f5_state_INT16.pt`, `meta_f5.json`), `legacy_f5_b/` = `f5-b` (`f5_stage_a.pt`, `f5_state_FP.pt`, `f5_state_INT6.pt`, `f5_state_INT4.pt`, `meta_f5.json`), `legacy_gain/` = `gain-FP`, `gain-6`, `gain-4` (`func_lm_eval_<state>.pt`). `phase1_gain_analysis.py` reads a `meta_lm_eval.json` in EVERY run directory, so the control directories `lg_FP`, `lg_6`, `lg_4` each hold one (copied from Ada `out/phase1/gain-FP|6|4/meta_lm_eval.json`) next to the symlinked state file.
```
OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B scripts/g0/phase1_func_f5_analysis.py --run <func_inputs/ARM_f5_a> --run <func_inputs/ARM_f5_b> --out <func_inputs/analysis_ARM_f5>
OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B scripts/g0/phase1_gain_analysis.py --runs <FP dir> <INT6 dir> <INT4 dir> --ratios-json reports/functional_validation/gain_study/calibration/ratios.json --baseline-csv reports/functional_validation/gain_study/baseline/baseline.csv --out <func_inputs/analysis_ARM_gain>
```
**Controls done 2026-10-28 11:15 (unchanged code on the legacy files, outputs outside git in `func_inputs/analysis_legacy_f5` and `analysis_legacy_gain`):**
- F5: states INT16, INT6, INT4, floor 0.9489; all 10 rows of the control's `f5_tables.csv` equal a row of the committed `reports/functional_validation/results/f5/f5_tables.csv` (16 rows; the committed file also covers the states and groups the subset does not).
- Study G: states FP, INT6, INT4; all 810 rows of the control's `curves.csv` equal the committed `gain_study/results/curves.csv` rows for the same (state, language, ratio) (maximum absolute difference 0; the committed file has 1620 rows for six states). The control's own hypothesis table (H-G1 to H-G6) is on three states only and is not a replacement for the committed one.

**Study G, new arms (run 2026-10-28 12:3x):** per arm the command above with `<FP dir>` = `func_inputs/lg_FP` (old FP file + its `meta_lm_eval.json`), `<INT6 dir>` = `func_inputs/jobs/ARM-gain-6`, `<INT4 dir>` = `func_inputs/jobs/ARM-gain-4`, `--out func_inputs/analysis_ARM_gain` (ARM = `gptq`, `rtnm`); validation `validate_func_job.py --bits 6|4 --precisions 6|4 --preset 'gain|...|c3|K64' --configs INT6|INT4=999`. Outputs copied to `reports/gptq_extension/results/gain/{gptq,rtnm}/` (README there). **F5, new arms:** `phase1_func_f5_analysis.py --run func_inputs/legacy_f5_a --run func_inputs/jobs/ARM-f5-int64 --out func_inputs/analysis_ARM_f5`, validation `validate_func_job.py --task f5 --states FP 6 4 --reference-meta func_inputs/legacy_f5_b/meta_f5.json`; outputs in `reports/gptq_extension/results/f5/`.

## External panel analysis (written 2026-10-28 13:5x; run after ALL FOUR feature jobs of the panel are validated)

Group-level external jobs 44209 to 44212 are validated; the feature-level jobs 44213 to 44216 are running. **Report, ONCE per arm, the unchanged code and the old external options** (INT16 floors = the legacy rounding, identical in all arms: the OLD `transfer-ext-*-int16` directories are reused; there is no `--survival` input, as in the old external run):
```
OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B scripts/g0/phase1_transfer_report.py --freeze reports/intervention_transfer/freeze.json --core reports/intervention_transfer/core_external.json \
  --cluster-mod 1000000 --group-languages eng_Latn spa_Latn rus_Cyrl zho_Hans hin_Deva yor_Latn --group-sentences 300 --out <transfer_runs/analysis_external_ARM> \
  --feature-dir INT16=<transfer-ext-feature-int16> --feature-dir INT6=<transfer_runs/ARM-transfer-ext-feature-int6> --feature-dir INT4=<transfer_runs/ARM-transfer-ext-feature-int4> \
  --group-dir INT16=<transfer-ext-group-int16> --group-dir INT6=<transfer_runs/ARM-transfer-ext-group-int6> --group-dir INT4=<transfer_runs/ARM-transfer-ext-group-int4>
```
with `ARM` = `gptq` and `rtnm`. **Control done 2026-10-28 13:5x:** the same command with the OLD directories for all six inputs (`--out analysis_external_control`) writes 12 files byte-identical to the committed `reports/intervention_transfer/results/external/` (55 s, no per-arm change of code). Validation of the feature jobs: `validate_job.py --task feature --states INT4|INT6 --core reports/intervention_transfer/core_external.json --gate-max-abs 3.0` (jobnames `gptq-ef4`, `rtnm-ef4`, `gptq-ef6`, `rtnm-ef6`); FP bridge: `bridge_transfer.py --old <transfer-ext-feature-int4|6> --new <job copy> --prefix transfer_INT4|6` (group level: `--prefix group_INT4|6`, done for 44209 to 44212: exact).

**Comparison of the three arms on the external panel (tool ready 2026-10-28):** after the per-arm reports above,
```
~/.venvs/anlp-torch/bin/python -B scripts/g0/gptq_ext/compare_transfer.py --legacy reports/intervention_transfer/results/external --matched <transfer_runs/analysis_external_rtnm> --gptq <transfer_runs/analysis_external_gptq> --roster reports/g0/phase1_languages.json --label "external panel, six languages" --out reports/gptq_extension/results/external
```
(`--label` goes into the heading and the text states the number of languages of the data; the six languages are paired by language, so the bootstrap of the group-level contrasts is over six languages: read those intervals as very coarse.) The tool was dry-run on the control outputs (all three arms = the old external analysis: loads, writes the five files, zero differences); the main-study comparison it wrote before is reproduced byte for byte after this change.
