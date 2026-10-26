# Handoff 040: Gemma-2-9B confirmation test, to be run on a SEPARATE GPU cluster (2026-10-26)

**For:** the agent or teammate running the Gemma-2-9B runs outside Ada. Self-contained: you do not need Ada, the cron, or any other chat.
**Decision:** D-089 in [`reports/decisions_and_anomalies.md`](../decisions_and_anomalies.md). Plan: [`docs/superpowers/plans/2026-10-26-gemma9b-confirmation.md`](../../docs/superpowers/plans/2026-10-26-gemma9b-confirmation.md) (written for Ada; this handoff overrides its compute section: 2/3-GPU splitting is not needed on a big-memory GPU).
**Deliver to:** a NEW branch `gemma9b-confirmation` of `git@github.com:TH4KUR/always-late`, created from `main` at the commit that adds this file. Push there only. **Do not push to `main` and do not merge.** The user merges it later.

## 1. What and why, in five lines

The main study (Gemma-2-2B + Gemma Scope, layer 12, 16k, JumpReLU) finds that zero-ablating a language's selective SAE feature group harms that language's next-token loss beyond matched controls (27/27 languages), flat through INT6 and stronger at INT4 (Study G). A Llama-3.2-1B check confounded model, SAE type (TopK), SAE recipe and depth, so the user replaced it. **You re-run the same experiment on Gemma-2-9B with Gemma Scope `layer_20/width_16k/canonical`**: same architecture, same SAE family, same relative depth ((20+1)/42 = (12+1)/26 = 0.5). Then compare the 2B and 9B results.

## 2. Read first (in this order, about 30 minutes)

1. [`reports/functional_validation/RESULTS_SUMMARY.md`](../functional_validation/RESULTS_SUMMARY.md): what the 2B results are.
2. [`reports/functional_validation/gain_study/README.md`](../functional_validation/gain_study/README.md): Study G, the experiment you replicate (preset, ratios, analysis, results).
3. [`docs/superpowers/plans/2026-10-26-llama-consistency-check.md`](../../docs/superpowers/plans/2026-10-26-llama-consistency-check.md), sections 3, 6, 7 and 8: **the predictions P1 to P3b, the verdict function and the robustness plan R1 to R5. You use them unchanged.** Section 5 has the job layout you copy.
4. `scripts/g0/phase1_llama_run.py`, `scripts/g0/run_llama.sbatch`: the adapter you generalise (it sets the frozen runners' constants and calls them; no frozen file is edited).
5. [`third_party/duan2026/UPSTREAM.md`](../../third_party/duan2026/UPSTREAM.md): the frozen upstream functions (quantizer `quantize_rtn_per_output_channel_tl`, `load_gemma_tl_model`, `load_sae_compat`). Their hashes are checked at runtime.

## 3. Environment

- **GPU:** one GPU with at least 40 GB (A100 40/80 GB, A6000 48 GB, L40S, H100). Gemma-2-9B in bf16 is 18.5 GB plus the 256k-vocabulary logits. A single device keeps the frozen loader unchanged. If you only have 24 GB cards, multi-device loading (`n_devices`) needs an adapter change with its own test and a note in the report. 80+ GB of CPU RAM (the RTN quantizer restores from a full-precision copy).
- **Packages (the versions of the 2B runs):** torch 2.14.1 (cu126), transformer-lens 3.9.0, transformers 5.19.0, sae-lens 6.54.5, plus numpy, scipy, safetensors, huggingface_hub, tqdm. Record yours in every `meta.json` (the runners do it).
- **Hugging Face:** accept the Gemma terms for `google/gemma-2-9b` and use a token with read access. The SAE repo is `google/gemma-scope-9b-pt-res` (sae_lens release `gemma-scope-9b-pt-res-canonical`). Never commit the token or any weights.
- **Data:** FLORES-200, the ORIGINAL archive `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` (not FLORES+). Unpack it so that `<FLORES_ROOT>/flores200_dataset/{dev,devtest}/<code>.<split>` exists. The reader checks 997 dev and 1,012 devtest lines per language. The 27 languages are in `reports/g0/phase1_languages.json`.
- **Tests before anything:** `python3 -B -m unittest discover -s scripts/g0/tests` and `python -B -m unittest discover -s scripts/g0/tests_torch` (torch venv) must be green on your machine.

## 4. Steps

**S1 Adapter (tests first).** Generalise `phase1_llama_run.py` into a model switch (for example `--model gemma2-9b`) or add `phase1_gemma9b_run.py`, with:
- `MODEL_CANDIDATES = ["google/gemma-2-9b"]`
- `SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"`, `SAE_ID = "layer_20/width_16k/canonical"`
- `LAYER = 20` in both namespaces the Llama adapter patches (`pr.LAYER`, `fr.LAYER`)
- **no** TopK loader and **no** hook alias: Gemma Scope loads through the frozen `load_sae_compat`, and the hook is natively `blocks.20.hook_resid_post`.

Keep the Llama path working (its tests stay green). Write tests with fakes that check the constants, the restore of the patched globals and the hook assertion. The runner asserts `hook == blocks.{LAYER}.hook_resid_post` and `d_in == d_model` (3584).

**S2 Smoke (instrument gate, not a look at results).** Tagging and the language model on 3 languages:
```
<tag>  --languages eng_Latn hin_Deva khm_Khmr --max-sentences 40
<lm>   --precisions FP 4 --languages eng_Latn hin_Deva khm_Khmr --max-sentences 40 --split pilot --panel
```
Check that:
- the log names the GPU and shows no `CUDA initialization` warning;
- the gates pass (the stop-at-layer forward equals the full forward, the sham ablation is exact, the frozen-function hashes are verified);
- the quantizer covers all Gemma-2 weight tensors (as for 2B).

Record peak memory and seconds per configuration.

**S3 Freeze, BEFORE any full-run number exists.** Commit `reports/functional_validation/gemma9b/freeze.json` with:
- model, SAE, layer, commit and the sha256 of the FLORES archive;
- the preset (the Study G MAIN preset: 37 configurations per language, 999 per state), the ratios from `reports/functional_validation/gain_study/calibration/ratios.json` (the 2B ratios, NOT recalibrated, as for Llama), 300 eval sentences, 3 matched-control draws;
- the predictions P1 to P3b and the verdict function copied verbatim from the Llama plan sections 3 and 7, with the sha256 of that text.

Any later deviation is a dated amendment that says why, never a silent change.

**S4 Full runs.** The user's sample floor (D-075) is not negotiable:
- all 27 languages, 6 states (**FP, INT8, INT7, INT6, INT5, INT4**; the Llama check dropped INT7, you do not), 300 sentences, 3 control draws;
- tagging on dev with the equal token budget, then the frozen group builder.

Group families: the Gemma-2B manifest has K64 (and `all`), and the predictions use K64. If the 9B control pool cannot fill K64 for a language, the frozen builder refuses; report it (Llama needed K16 as a fallback) and never shrink silently. Run one job per state if your wall-time limit needs it. Example (Llama form): `lm <tag> --precisions FP 8 7 --split eval --n-eval 300 --panel eng_Latn hin_Deva`.

Validate every job (exit 0, GPU line, gates, 999 configurations per state, nothing skipped) and record sha256 of every output in `reports/functional_validation/gemma9b/artifact_checksums.md`.

**S5 Analysis.** Run `phase1_llama_compare.py` (generalise the `--sae` flag and the labels so they are model-neutral, tests first) with:
- `--runs <your run dirs>`
- `--manifest <tag>/groups_manifest.json`
- `--ratios-json reports/functional_validation/gain_study/calibration/ratios.json`
- `--gemma-root reports/functional_validation` (the 2B reference values)
- `--out reports/functional_validation/gemma9b`

Then run `phase1_llama_robust.py` (R1 to R5). Report every prediction as met or not met, and the verdict.

**S6 The 2B against 9B comparison (the user asked for it explicitly)** goes in `reports/functional_validation/gemma9b/comparison_2b_vs_9b.md`, plus CSV and figures:
1. per-language E(K64 zero-ablation) at FP, 2B against 9B (scatter, Spearman across languages);
2. the retention ladder (E at the state over E at FP, macro and per language) at INT8/7/6/5/4, both models on one plot;
3. the dose-response curves (net harm against the down ratios 0.354, 0.177, 0.088) at FP and INT4, both models;
4. the number of selective features per language and the group sizes, both models;
5. a table of P1 to P3b and the verdict, both models.

The 2B numbers are already in the repo (`reports/functional_validation/gain_study/results/`, `RESULTS_SUMMARY.md`); read them, do not recompute them. Use the `dataviz` skill if you are a Claude agent. Use light and dark variants as in `FIGURE_INDEX.md`. State the remaining differences under every claim: scale (2.6B against 9.2B) and the dictionary-to-width ratio (16k/2304 = 7.1 against 16k/3584 = 4.6).

## 5. Rules for the branch (they keep the later merge clean)

- **Write only inside `reports/functional_validation/gemma9b/`** plus your new or generalised scripts and tests under `scripts/g0/`. Do NOT edit the shared ledgers on the branch (`STATUS.md`, `decisions_and_anomalies.md`, `DECISIONS_FOR_REVIEW.md`, `README.md`, `reports/INDEX.md`, `progress_log.md`): they change on `main` every 40 minutes and would conflict.
- Inside your folder, keep: `STATUS.md` (your run ledger: job ids, runtimes, commits, what is next), `decisions.md` (provisional ids `G9-01`, `G9-02`, ...; they are renumbered into the D-series at merge), `handoffs/` (your own handoffs), `report.md` (verdict and limits), `comparison_2b_vs_9b.md`.
- **Never edit frozen files** (`third_party/`, the frozen runners `phase1_run.py`, `phase1_func_run.py`, `phase1_gain_run.py`, etc.; `scripts/g0/tests/test_vendor.py` enforces part of this). New behaviour goes in an adapter.
- No large artifacts in git (`.pt` stays on your cluster; commit checksums, CSV, JSON summaries, figures). No weights, no tokens. Do a secret scan before each commit. Do not change the visibility of, or delete anything on, Hugging Face or wandb.
- Report everything, including failed and missing cells and unmet predictions. No post hoc language selection, no new thresholds after seeing numbers. A "not consistent" verdict is a valid result.
- Commit messages: describe the step. If you are an AI agent, end with your own co-author line.

## 6. Cost and done

On Ada's 2080 Tis the 2B runs took about 3 GPU hours per state. Expect about 3.5x the compute for 9B, so on an A100 very roughly 3 to 6 h per state, 20 to 35 GPU hours for the six states plus about 1 h for tagging. Measure it in S2 and write the estimate in your STATUS.md.

**Done** = all of the following on branch `gemma9b-confirmation`:
- the freeze committed before the runs;
- 6 validated states, with checksums;
- `report.md` with the verdict and P1 to P3b;
- `comparison_2b_vs_9b.md` with its figures;
- a final handoff in `gemma9b/handoffs/` that says what was run, where the raw files live, and anything left open.

Tell the user that the branch is ready for review.
