# wandb schema (v2): a post-hoc upload of the finished results, one run per study or result family

Written 2026-10-26 08:55 IST, BEFORE any code (decision D-091). The user chose Option A on 2026-10-26: "take option A for the wandb choice" (the choice offered after the question "are the runs still being logged to wandb?"). Schema v1 ([`wandb_schema.md`](wandb_schema.md)) covers the 19 existing runs (the 15 replication runs of the closure protocol and the 4 `phase1/*` runs) and fits only survival-style runs; the functional study, Study G, the Belebele family, F5, the prompt cross-check and the Llama check never logged to wandb (the jobs run on compute nodes without a network, and no job script calls wandb). v2 uploads their finished, committed results after the fact. It does not change v1, does not touch the 19 old runs and deletes nothing.

## 1. What is uploaded, what is not

**Uploaded (from files committed in this repository; every number is read from a committed json or csv, never retyped):** per run a fixed config, a short list of headline summary values, the committed per-condition and per-language csv files as `wandb.Table`s, the committed light-theme PNG figures as `wandb.Image`s, and the small committed `report.md` / `analysis.json` / csv files (plus a `MANIFEST.json` with the repository path and sha256 of each file) as one artifact per run. For the ladder families a history (one step per precision) is logged so that the wandb UI draws the ladder as a line chart.

**Not uploaded:** model weights or SAE weights (the Llama licence applies; nothing derived from the weights that reproduces them), the item-level `.pt` result files of the cluster (the user did not ask for them; their sizes and sha256 are in [`functional_validation/artifact_checksums.md`](functional_validation/artifact_checksums.md)), the groups manifests (feature index lists), the pilot and smoke results (never results), dark figures and svg files (the light PNG is the figure), tokens or keys, anything outside `reports/`. A guard in the builder refuses any file with the extensions `.pt .pth .bin .safetensors .npy .npz .ckpt .gguf .zip .tar .gz`, any file above 2 MB, a run above 20 MB, and any path outside `reports/`.

## 2. Run naming and identity

- Project `anlp-project`, entity `karma-skz-` (the existing project). Every v2 run has the tags `schema-v2` and `post-hoc-upload` plus its family tag and its status tag (`exploratory` or `confirmatory`), `job_type = analysis-upload`, a `group` per study family (`functional`, `belebele`, `f5`, `study-g`, `llama`) and a name that says what it is (`<family>/<what>`).
- **Deterministic id** `v2-<slug>` (for example `v2-study-g-main`), `resume="allow"`: a re-run overwrites config and summary of the same run instead of creating a second one; the artifact is re-logged under the same name (identical content gives no new version); the history, the tables and the images are logged only when the run is new (`run.resumed` is false; the step counter cannot be used, it is 0 after a resume), so a re-run adds no row; `--force-media` repeats the tables and images of an interrupted run on purpose. Figures whose file stem occurs twice in a run (`pair_light.png` in four folders) get their folder in the key (`fig/<folder>-<stem>`).
- The 19 old runs keep their names and ids. v2 names never collide with them (old names are `baseline/full`, `R1/...`, `R5a/...`, `phase1/...`, `smoke/...`).

## 3. Config (the same keys in every v2 run; unknown values are `n/a`)

| key | meaning |
|---|---|
| `schema_version` | 2 |
| `study` | the slug of the run (`functional-lm-ladder`, ...) |
| `title` | one line saying what the run holds |
| `status` | `exploratory`, `confirmatory (frozen)` or a mixed statement copied from `functional_validation/RESULTS_SUMMARY.md` |
| `decisions` | the D-entries of `reports/decisions_and_anomalies.md` that fixed or read the result |
| `model`, `sae`, `layer`, `quantizer` | the setting (Gemma-2-2B, Gemma Scope 16k at layer 12, RTN per output channel; the Llama run states its own model and SAE) |
| `states` | the precisions covered |
| `slurm_jobs` | `name=id` pairs of the Ada jobs whose item files the analysis read (from `sacct`, 2026-10-26) |
| `code_ref` | the commit or snapshot of the job code where the repository records it, else `n/a` |
| `freeze_file`, `freeze_sha256` | the committed freeze file of the setting and its sha256, where one exists |
| `source` | `post-hoc-upload-of-committed-results` |
| `upload_commit` | the commit of this repository whose files were read |
| `files_sha256` | JSON string `{repository path: sha256}` of every uploaded file |
| `limits` | the limit that must travel with the result (copied in words from `RESULTS_SUMMARY.md`) |

## 4. Summary (the runs-table columns)

Every summary key is `headline/<name>` (a number, a boolean or a short string) plus `n_tables`, `n_images`, `n_files`. The same quantity has the same name in every run in which it occurs: `headline/macro_effect`, `..._lo`, `..._hi`, `headline/n_positive`, `headline/retention_INT<b>`, `headline/verdict`. The exact list of each run is printed by `--dry-run` and stored in the payload; it is declared in `scripts/g0/wandb_upload_v2.py` as (name, file, path in the json) triples, so a name that does not resolve in the committed file stops the build.

## 5. History (ladder runs only)

`functional-lm-ladder`, `belebele-confirmatory-tmc` and `llama-consistency-check` log one history row per precision with the x-axis `bits` (INT bit-width; full precision is 32, INT0 the restore-only control is 0, as in v1): `effect`, `effect_lo`, `effect_hi`, `retention`, `retention_lo`, `retention_hi` (Belebele: `effect*` and the accuracy columns; Llama: the Llama and the Gemma retention by state). Other runs have no history; their `state` / `precision` columns are in the tables.

## 6. The 15 runs

| id | name | group | status | decisions | what the run holds |
|---|---|---|---|---|---|
| `v2-functional-lm-ladder` | `functional/lm-ladder-F1-F2` | functional | exploratory (frozen setting) | D-041, D-050, D-068 | loss-task evaluation on the 300 frozen devtest sentences: group effect E per language and precision (FP, INT8 to INT4, rounding-only controls INT16, INT12, INT0), macro retention, sham, SAE fidelity, the covered-18 and uncovered-9 subsets, seven figures |
| `v2-functional-rescue` | `functional/rescue-F3` | functional | exploratory | D-058 | share of the quantization loss recovered by patching full-precision activations (group, controls, all SAE features, error, whole residual) at INT6, INT5, INT4, figure 4 |
| `v2-functional-f4-association` | `functional/f4-association-vs-function` | functional | exploratory | D-048, D-049, D-051 | F4 at INT4, INT5, INT6 (lost against activity-matched retained features), the association retention by rank (A1), figures 5, 8, 9 |
| `v2-association-margin-a2-a3` | `functional/association-margin-A2-A3` | functional | exploratory (hypotheses written before the numbers) | D-066, D-067 | loss of the D-035 association by margin quintile (A2) and the margin against function (A3), two figures |
| `v2-belebele-confirmatory-tmc` | `belebele/confirmatory-T-MC` | belebele | confirmatory (frozen) | D-047, D-050 | the only confirmatory family: statements S0 to S5, macro effect per precision, accuracies, the calibration of the 18 covered languages, the freeze, three figures |
| `v2-belebele-ksens-k-all` | `belebele/sensitivity-K-all` | belebele | exploratory | D-062 | the whole selective group zeroed (K = all) at FP and INT6 |
| `v2-belebele-positive-control` | `belebele/positive-control-edit-sensitivity` | belebele | exploratory | D-069, D-072 | random sets of 256, 1,024 and all-control size zeroed: the edit path can move the accuracy |
| `v2-belebele-prompt-check` | `belebele/prompt-cross-check` | belebele | exploratory | D-086 | frozen letter prompt against the lm-evaluation-harness prompt, 18 languages |
| `v2-f5-main` | `f5/feature-function-main` | f5 | exploratory (rule fixed before the first number) | D-059 | 1,487 features: floor, read-side and write-side statistics per precision, statements S1 to S3, six figures |
| `v2-f5-sensitivity-controls` | `f5/sensitivity-and-swap-controls` | f5 | exploratory | D-060, D-061, D-063, D-064 | M = 8 and 32 contexts, the direction-swap control, fixed amplitude, the swap control along the ladder, the ladder figure |
| `v2-study-g-main` | `study-g/main-graded-gain` | study-g | exploratory (pre-registered rules H-G1 to H-G6) | D-070 to D-078 | dose-response of the language-selective group at FP, INT8 to INT4: curves, optimal gain, reliability, calibration, baseline, four figures |
| `v2-study-g-ext-c` | `study-g/extension-C-repair` | study-g | exploratory | D-079, D-082 | the small INT4 repair on 562 fresh sentences with five control draws |
| `v2-study-g-ext-d` | `study-g/extension-D-optimal-gain` | study-g | exploratory (post hoc in origin) | D-083, D-084 | optimal gain against the shrinkage of the group |
| `v2-study-g-ext-b` | `study-g/extension-B-state-defined` | study-g | exploratory | D-081, D-085 | groups selected in the quantized model against the FP selection |
| `v2-llama-consistency-check` | `llama/consistency-check-3.2-1B` | llama | exploratory (pre-registered verdict) | D-087, D-089, D-090 | predictions P1 to P3b, verdict, Gemma reference, the post hoc robustness tables (R1 to R6), per-language rows |

## 7. How it runs (and how to re-run)

1. `python3 -B scripts/g0/wandb_upload_v2.py build --out <payload dir> [--only <slug>]` reads the committed files and writes `payload.json` plus the `files.tar` of exactly the files to ship (no network, no wandb). `--dry-run` prints the run list, the metric, table, image and file counts and the sizes and writes nothing else.
2. The local virtual environments have no wandb: the files go to Ada (`scp`), the upload runs on the login node with the Ada environment (`/home2/karma/duan-verify/env`, wandb 0.30.0, credentials from `~/.netrc`, never read or printed): `python wandb_upload_v2.py upload --payload payload.json --root <extracted dir> [--only <slug>]`. A short run, no job. If the login node could not reach wandb the fallback is `WANDB_MODE=offline` and `wandb sync` from the same place.
3. First one small run as a probe (`--only v2-belebele-prompt-check`), checked through `wandb.Api()` (config, summary, table, image, artifact files present); then the rest.
4. Tests (offline, with fakes): `OMP_NUM_THREADS=4 ~/.venvs/anlp-torch/bin/python -B -m unittest scripts/g0/tests_torch/test_wandb_upload_v2.py`; mutation check `scripts/g0/mutcheck.py` on the builder.
5. A re-run is idempotent by the deterministic ids; delete nothing by hand. The key is never printed; the upload log is searched for key patterns afterwards.

## 8. State after the upload (2026-10-26 09:01 to 09:11 IST, from the login node of Ada)

Project <https://wandb.ai/karma-skz-/anlp-project>: **34 runs = the 19 old ones (all `finished`, untouched: the uploader refuses any id that does not start with `v2-`) + the 15 new ones.** Built from commit `acc6ebf` (146 files, 8.8 MB, every shipped file tracked and unchanged: `build` refuses a modified or untracked file). The run URLs:

| run | name | URL |
|---|---|---|
| `v2-functional-lm-ladder` | functional/lm-ladder-F1-F2 | <https://wandb.ai/karma-skz-/anlp-project/runs/v2-functional-lm-ladder> |
| `v2-functional-rescue` | functional/rescue-F3 | <https://wandb.ai/karma-skz-/anlp-project/runs/v2-functional-rescue> |
| `v2-functional-f4-association` | functional/f4-association-vs-function | <https://wandb.ai/karma-skz-/anlp-project/runs/v2-functional-f4-association> |
| `v2-association-margin-a2-a3` | functional/association-margin-A2-A3 | <https://wandb.ai/karma-skz-/anlp-project/runs/v2-association-margin-a2-a3> |
| `v2-belebele-confirmatory-tmc` | belebele/confirmatory-T-MC | <https://wandb.ai/karma-skz-/anlp-project/runs/v2-belebele-confirmatory-tmc> |
| `v2-belebele-ksens-k-all` | belebele/sensitivity-K-all | <https://wandb.ai/karma-skz-/anlp-project/runs/v2-belebele-ksens-k-all> |
| `v2-belebele-positive-control` | belebele/positive-control-edit-sensitivity | <https://wandb.ai/karma-skz-/anlp-project/runs/v2-belebele-positive-control> |
| `v2-belebele-prompt-check` | belebele/prompt-cross-check | <https://wandb.ai/karma-skz-/anlp-project/runs/v2-belebele-prompt-check> |
| `v2-f5-main` | f5/feature-function-main | <https://wandb.ai/karma-skz-/anlp-project/runs/v2-f5-main> |
| `v2-f5-sensitivity-controls` | f5/sensitivity-and-swap-controls | <https://wandb.ai/karma-skz-/anlp-project/runs/v2-f5-sensitivity-controls> |
| `v2-study-g-main` | study-g/main-graded-gain | <https://wandb.ai/karma-skz-/anlp-project/runs/v2-study-g-main> |
| `v2-study-g-ext-c` | study-g/extension-C-repair | <https://wandb.ai/karma-skz-/anlp-project/runs/v2-study-g-ext-c> |
| `v2-study-g-ext-d` | study-g/extension-D-optimal-gain | <https://wandb.ai/karma-skz-/anlp-project/runs/v2-study-g-ext-d> |
| `v2-study-g-ext-b` | study-g/extension-B-state-defined | <https://wandb.ai/karma-skz-/anlp-project/runs/v2-study-g-ext-b> |
| `v2-llama-consistency-check` | llama/consistency-check-3.2-1B | <https://wandb.ai/karma-skz-/anlp-project/runs/v2-llama-consistency-check> |

**Verification through `wandb.Api()` (after the last pass):** all 15 runs are `finished`; every summary key of the payload is present with the payload value (0 missing, 0 different); the artifact `v2-<slug>-files` of every run holds exactly the payload files plus `MANIFEST.json`; every table and figure of the payload is present. The upload logs were searched for key patterns (none) and the scratch directory on Ada (payload, tar, logs, the local wandb folder) was deleted afterwards.

**What went wrong on the way, and what it left in the runs (stated because nothing on wandb was deleted):**
1. The first full pass stopped at a table column that holds numbers and text (`auroc_function`); fixed by building the tables before the run starts and allowing mixed types. Three runs (`lm-ladder`, `rescue`, `f4-association`) had been finished by then.
2. The second pass logged the tables, figures and history of those three runs and of the probe run (`belebele-prompt-check`) a second time, because the step counter of a resumed run is 0 (the uploader now uses `run.resumed`; a third full pass changed nothing). **Left behind: `v2-functional-lm-ladder` has its 9 history rows twice (20 rows: 2 x 9 + 2 media rows), its 6 tables and 5 figures twice; `v2-functional-rescue` and `v2-functional-f4-association` have their tables and figures twice; `v2-belebele-prompt-check` has its table twice.** The duplicates are identical in content (the line charts overlap exactly); summary, config and artifacts are correct and unique.
3. `v2-f5-sensitivity-controls` had four figures `pair_light.png` under one key (only the last survived); it was logged again with distinct keys (`fig/direction_swap-pair_light`, ...), which left its first two figures and its table once more.
If clean runs are wanted, the five runs can be deleted by the user (or with the user's go-ahead) and uploaded again with the same command; the ids are deterministic.

**Re-run:** `python3 -B scripts/g0/wandb_upload_v2.py build --out <dir>` (refuses files that differ from the commit; `--only <slug>`, `--dry-run`), `scp` the `payload.json`, `files.tar` and the script to a scratch directory on Ada, extract the tar there, then `WANDB_SILENT=true WANDB_DIR=$PWD /home2/karma/duan-verify/env/bin/python wandb_upload_v2.py upload --payload payload.json --root <extracted> [--only <slug>] [--force-media]`. When a result changes (a new decision, a new run), add its family to `FAMILIES` in the script (tests first) and upload only that slug.
