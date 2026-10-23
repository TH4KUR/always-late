# Reproducing the results of the functional study

Written 2026-10-23 22:36 IST (Claude). **One command re-runs every analysis of the study from its input files and compares each output with the committed one:**

```
bash scripts/g0/reproduce_functional.sh <inputs root> <scratch output dir>      # CPU only, about 30 seconds; exit 0 if everything is identical
```

It needs the torch venv (`~/.venvs/anlp-torch`: numpy, scipy, statsmodels, matplotlib are in it; the script sets `OMP_NUM_THREADS=4`) and the repository at the commit that produced the results. What it does **not** do: it does not re-run the GPU jobs (those are the item-level and per-state files, listed with their sha256 in [`artifact_checksums.md`](artifact_checksums.md); the commands that produced them are in the `Commands` block of [`STATUS.md`](STATUS.md) and in the Log lines of each job), and it does not compare figures (PNG and SVG depend on the plotting library).

## Verification record

**2026-10-23, repository commit `58a403e` plus the script itself, inputs = the files fetched from Ada on the days of the jobs (sizes and sha256 equal to Ada's at fetch time, as recorded in `artifact_checksums.md`):** 57 output files compared with `cmp`, **57 identical** (the script now also covers the 12 files of `eval_subsets`, D-068: **69 of 69 identical** on 2026-10-24), three of them with a named field ignored: `f5/f5_results.json` and `f5_smoke/f5_results.json` (the run paths, and the flags `direction_control` and `fixed_amplitude`, which the metadata gained after the main run) and `mc_ksens/analysis.json` (the free-text note). Two outputs are not byte-compared on purpose: `f5/report.md` (it embeds the links of the figures given on the command line of the original run) and the figures. The same run found one provenance slip, fixed in the same commit: the A3 `analysis.json` listed the scratch names of the F5 jobs instead of `f5-a` and `f5-b`.

## Inputs: where they come from and the layout the script expects

Files are fetched from Ada (`/home2/karma/duan-verify/out/phase1/<job>/<file>`) one `scp` per file (`-o HostKeyAlias=ada-gw1 karma@10.4.24.30`), then checked with `sha256sum` against `artifact_checksums.md`. Layout of the inputs root (`<IN>`):

| directory in `<IN>` | Ada job(s) | files |
|---|---|---|
| `eval/` | `func-eval-a`, `-b`, `-c`, `-d` | `func_lm_eval_<p>.pt` for FP, INT0, INT12, INT16, INT8, INT7, INT6, INT5, INT4 and `meta_lm_eval.json` |
| `rescue/` | `func-rescue-a` | `func_lm_eval_{FP,INT4,INT5,INT6}.pt`, `meta_lm_eval.json` |
| `f4-INT4/`, `f4-INT5/`, `f4-INT6/` | `func-f4-INT<b>` | `func_lm_eval_FP.pt`, `func_lm_eval_INT<b>.pt`, `meta_lm_eval.json` |
| `mc-eval/` | `func-mc-eval-a`, `-b` | the six `func_mc_eval_letter_<p>.pt`; the two `meta_mc_eval.json` renamed `meta_mc_eval_a.json` and `meta_mc_eval_b.json` |
| `mc-ksens/` | `func-mc-ksens` | `func_mc_eval_letter_{FP,INT6}.pt`, `meta_mc_eval.json` |
| `f5-a/`, `f5-b/` | `f5-a`, `f5-b` | `f5_stage_a.pt`, `f5_state_<p>.pt`, `meta_f5.json` (A: FP, INT16, INT8; B: FP, INT6, INT5, INT4) |
| `f5-smoke2/`, `f5-m8/`, `f5-m32/`, `f5-dir/`, `f5-dir2/`, `f5-fa/`, `f5-fa-dir/` | the F5 jobs of the same names | the same file types |
| `feature_status_eq.csv` | `association-full/analysis-eq/feature_status.csv` | the D-035 per-feature statuses (16 MB; equal to `analysis-eq-rerun`) |

## Result directories, modules and commands

All commands from the repository root; `PY=~/.venvs/anlp-torch/bin/python`, `$FS` the status file, `$COV` the 18 covered Belebele languages (`python3 -c "import json;print(' '.join(json.load(open('reports/functional_validation/evaluation_freeze_mc.json'))['calibration']['covered']))"`).

| committed output | command (the script uses exactly these) |
|---|---|
| `results/eval/` (D-041, D-050) | `$PY -B scripts/g0/phase1_func_analysis.py --runs <IN>/eval --split eval --out results/eval --K K64 --n-boot 2000` |
| `results/eval_subsets/covered18`, `uncovered9` (D-068) | the same command as `results/eval/` with `--covered $COV` (the 18 languages of `calibration.covered`) and with `--covered` the other nine |
| `results/rescue/` (D-058) | `$PY -B scripts/g0/phase1_func_analysis.py --runs <IN>/rescue --split eval --out results/rescue --K K64 --n-boot 2000` |
| `results/f4/INT4`, `INT5`, `INT6` (D-048, D-049, D-051) | `$PY -B scripts/g0/phase1_func_f4_analysis.py --runs <IN>/f4-<b> --split eval --condition <b> --out results/f4/<b>` |
| `results/mc_eval/` (D-047, D-050) | `$PY -B scripts/g0/phase1_func_mc_analysis.py --runs <IN>/mc-eval --split eval --format letter --out results/mc_eval --K K64 --n-boot 2000 --seed 0 --retention-level 0.99 --covered $COV --note "Held-out T-MC evaluation (devtest-origin passages); the confirmatory family of the design, settings frozen before scoring."`, then `python3 scripts/g0/phase1_func_mc_statements.py --freeze reports/functional_validation/evaluation_freeze_mc.json --analysis results/mc_eval/analysis.json --out results/mc_eval` |
| `results/mc_ksens/` (D-062) | the same analysis command with `--runs <IN>/mc-ksens` and the exploratory note (K64 stands for the whole group) |
| `results/f5/` (D-059) | `$PY -B scripts/g0/phase1_func_f5_analysis.py --run <IN>/f5-a --run <IN>/f5-b --out results/f5` (the original run added `--figures ...` links) |
| `results/f5_smoke/` | `$PY -B scripts/g0/phase1_func_f5_analysis.py --run <IN>/f5-smoke2 --out results/f5_smoke` |
| `results/f5_sensitivity/{m8,m32,direction_swap,fixed_amplitude}/` (D-060, D-061, D-063) | `$PY -B scripts/g0/phase1_func_f5_sensitivity.py --main-run <IN>/f5-a --main-run <IN>/f5-b --sens-run <IN>/f5-{m8,m32,dir,fa} --states INT16 INT4 --label "..." --out results/f5_sensitivity/<name>` |
| `results/f5_sensitivity/fixed_amplitude_direction_swap/` (D-063) | `... phase1_func_f5_sensitivity.py --main-run <IN>/f5-fa --sens-run <IN>/f5-fa-dir --states INT16 INT4 --label "..." --out ...` |
| `results/f5_sensitivity/direction_swap_int8_6_5/` (D-064) | `... --main-run <IN>/f5-a --main-run <IN>/f5-b --sens-run <IN>/f5-dir2 --states INT16 INT8 INT6 INT5 --label "..." --out ...` |
| `figures/f5/ladder_control_table.csv` (and the figure) | `$PY -B scripts/g0/phase1_func_f5_figures.py ladder --main-run <IN>/f5-a --main-run <IN>/f5-b --control-run <IN>/f5-dir2 --control-run <IN>/f5-dir --out figures/f5 --note "..."` |
| `results/assoc_rank/` (A1) | `$PY -B scripts/g0/phase1_assoc_rank_link.py --status $FS --out results/assoc_rank` |
| `../association_retention/results/within-language/` (A2, D-066) | `$PY -B scripts/g0/phase1_assoc_margin.py --status $FS --out reports/association_retention/results/within-language` (figure: `phase1_assoc_margin_figures.py --analysis .../analysis.json --out ... --note ...`) |
| `../association_retention/results/margin-function/` (A3, D-067) | `$PY -B scripts/g0/phase1_assoc_function_link.py --status $FS --run <IN>/f5-a --run <IN>/f5-b --out reports/association_retention/results/margin-function` (figure: `phase1_assoc_function_figures.py ...`) |

The pairs figures of the sensitivity comparisons add `--figures --label-a ... --label-b ... --figure-title ...` to the comparison command; the figures of `eval`, `rescue`, `mc_eval` and `f5` come from `phase1_func_figures.py` and `phase1_func_f5_figures.py` (usage in their docstrings).

## Not covered by this check

- `results/pilot/` and `results/mc_pilot/`: preliminary, superseded by the evaluation runs; their item files are on Ada (`func-pilot-a`, `func-pilot-b`, `func-mc-pilot`) with checksums in the register.
- The D-035 study itself (`reports/association_retention/results/eq`, `all`, `crosscheck-eq`, `robustness-eq`): its reproduction is documented in [`../association_retention/README.md`](../association_retention/README.md) and `reports/phase1_reproduction.md` (it re-runs from the Phase 1 run files, not from the files above).
- The GPU jobs: deterministic up to GPU arithmetic; the stored per-state and item files are the inputs here, with sha256 in the register. A rerun of one job is expected to match to numerical tolerance, not bit for bit across hardware (the F5 logit gate allows 0.28 native, 0.68 with the float32 residual; the INT16 and FP states of two runs of the same code differ by up to 9e-5 nats, see D-055).

## Tests

`python3 -B -m unittest discover -s scripts/g0/tests` (stdlib, 265) and `OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B -m unittest discover -s scripts/g0/tests_torch` (torch, 1220); key behaviours of each module are mutation-checked with `scripts/g0/mutcheck.py` (the counts are in the Log of `STATUS.md`).
