# Does Compression Forget Language? Language-Specific Feature Survival Under Quantization in Multilingual LLMs

**Course**: Advanced Natural Language Processing (ANLP, Monsoon 2026)  
**Institution**: International Institute of Information Technology, Hyderabad (IIIT-H)  
**Team**: Always Late (AlwaysLate)  
**Authors**: Hemang Joshi, Eashaan Thakur, Shardul Joshi, Nikhita Ravi  
**GitHub Repository**: [https://github.com/TH4KUR/always-late](https://github.com/TH4KUR/always-late)

> **Status (updated 2026-10-18 morning). Read before using any number from this repository.**
> - **Our previous pipeline (`src/` and the Phase-1 scripts) was discarded.** It was an unverified re-implementation of Duan et al. (2026) and its headline interpretations contradicted its own data. The repository now builds on **Duan's own code** (`third_party/duan2026/`) and the Ada setup that reproduces it. The post-mortem (what was wrong, why results conflicted, whether the direction was worth it) is the entry "POST-MORTEM of our previous codebase" in [`reports/progress_log.md`](reports/progress_log.md). The old code is recoverable from the git tag `legacy-pipeline-2026-10-16`.
> - **What is established:** their published Gemma Phase-3 table reproduces with their own code on Ada (within 0.15 pp), and the disagreement between their numbers and ours is a **regime difference, not a bug**: their windows have no BOS token (baseline perplexity 457), ours had one. With one BOS token their own pipeline gives INT7 / INT6 survival of 99.97% / 97.5% instead of 81% / 51%, confirmed in two software stacks (their TransformerLens code and an HF reimplementation agree to 0.1 pp). Evidence: [`reports/audit_our_vs_duan.md`](reports/audit_our_vs_duan.md).
> - **What is open:** the team's confirmation of the provisional regime choice (primary `mask`, `none` as sensitivity; decisions D-001..D-014 in `reports/decisions_and_anomalies.md` are provisional and can be overridden), the meaning of the protocol's closure (their no-BOS regime fails the pre-committed noise-floor gate by a small margin, 0.989 vs 0.99, while the BOS-masked regime passes; disposition D-NOISE, `reports/audit_our_vs_duan.md` section 6.7), and whether the OSF registration was frozen (V-10).
> - **Phase 1 (language-selective features), rebuilt as a labelled extension of their code, has run** (`PLAN.md` S1-S8; pipeline `scripts/g0/phase1_*.py`, anchor reproduces their smoke run in every digit; two full runs on Ada). **Result** ([`reports/phase1_results.md`](reports/phase1_results.md)): **in the primary stream (BOS masked, equal token budgets) none of the three pre-registered confirmatory tests rejects after Holm correction** (INT7 selective minus matched universal controls -0.28 pp, Holm 0.51; INT6 -1.44 pp, Holm 0.13; tier slope -0.0048, p 0.99; equivalence within +-0.02 median r holds) and the effect is small next to the width effect (pooled survival 94.9% at INT6, 59% at INT5, 13.7% at INT4); in the BOS-free sensitivity regime the sign reverses (+9 pp) in an instrument that fails its own noise-floor check (A-029, A-030). Numbers cross-checked independently and byte-reproducible (D-017). **Threshold sensitivity (N2, D-019, a labelled deviation; A-034): the primary-stream null is threshold-dependent: with stricter fixed S thresholds selective features survive significantly LESS than firing-rate-matched controls (INT6 -2.1 pp at S > 3, Holm 0.017; -7.2 pp at S > 5, Holm 0.0003), post hoc, not yet controlled for activation magnitude, and not reconciled with the exploratory continuous model; the tier test stays null at every threshold.** **Magnitude check (N6, D-020, A-035): activation magnitude accounts for the N2 gap: selective features have lower maximum activations and magnitude predicts survival; with controls matched on magnitude as well the INT6 contrast is no longer negative (modestly positive, +3 to +5 pp at 5 and 10 bins; INT7 about 0; not significant at terciles), so the selectivity-specific reading is not supported by the rule fixed beforehand; whether magnitude is a confounder or a mediator is a question for the team (Q13).** **Mixed-model reconciliation (N7, D-021, A-036, A-037): the positive S~ coefficient of the exploratory model exists only with the tier terms in the model (-0.81 without them); survival depends strongly on the tier of the peak language for selective and universal features alike (INT6: 80 % low-tier against 96 to 97 % mid- and high-tier), which the selective-minus-control tier test cannot see; the S~ x tier interaction, the proposal's primary-endpoint term (+0.94), vanishes once magnitude is in the model (-0.007); which analysis is the confirmatory RQ2 test is Q14.** **Language-level tier analysis (N8, D-022, A-038): mean survival of a language's active features at INT6 is 69.4 / 90.6 / 95.1 % for low / mid / high tier (slope +12.3 pp per step, script-adjusted, Holm 0.0002; about half carried by character fertility, +6.3 pp with fertility, p 0.003; same in every stream): feature survival depends strongly on resource tier, selective or not; additional to the proposal's Holm family, correlational, Q14.** **Activation scale (N9, D-023, A-039): per-language activation scale (log10 median maximum activation of the active features) rank-orders survival almost perfectly (Spearman +0.96 to +0.98) and carries the whole tier effect at INT7 and INT6 (tier slope +12.3 pp -> -0.8 pp, p 0.67, once scale is in the model; INT5 residual +5.8 pp, p 0.002, none with the mean-activation scale); low-resource languages have lower scale (Spearman +0.83). In the proposal's terms the tier gap localizes upstream, in activation scale, not in selective-feature erasure; correlational and partly mechanical; why scale differs by tier is untested (Q15).** **Threshold margin (N11, D-025, A-040): the SAE's JumpReLU thresholds are almost constant (80 % within 3.00 to 3.19), so the margin above the threshold adds nothing beyond raw magnitude; the A-035 residual (+5.3 pp at INT6, S > 5, margin-matched) and the INT5 tier residual remain unexplained; the CPU analyses available without new data are exhausted.** One-page summary with generated tables and status labels: [`reports/phase1_summary.md`](reports/phase1_summary.md) (D-024). Provisional: the team's decisions and V-10 are still open.
> - Entry points: **[`HANDOFF.md`](HANDOFF.md)** (for a new agent or teammate), [`PLAN.md`](PLAN.md), [`PROGRESS.md`](PROGRESS.md), [`issues.md`](issues.md) (review of the discarded work), [`reports/duan_codebase_audit.md`](reports/duan_codebase_audit.md).
> - `paper/main.tex` still contains superseded text (see `issues.md` A1-A6). It must be rewritten by the team.

---

## 1. Project Overview & Core Mechanistic Question

Post-training quantization (PTQ) induces well-documented, uneven performance degradations across typologically diverse languages, disproportionately penalizing non-Latin scripts and low-resource languages. Concurrently, recent mechanistic audits demonstrate that aggregate metrics like perplexity fail to capture internal Sparse Autoencoder (SAE) feature damage. However, existing compression audits treat all internal features as a monolithic population.

In this project, we ask:
> **Core Question:** *Does compression-induced damage to language-selective internal representations provide a mechanistic explanation for the uneven degradation of multilingual language models?*

The setting is Gemma-2-2B residual-stream features at layer 12 (Gemma Scope JumpReLU SAE, 16k latents) under round-to-nearest (RTN) weight quantization (INT8 to INT4), following Duan et al. (2026).

---

## 2. Where the evidence stands

**Their pipeline, reproduced.** Their own Phase-3 script (vendored, 500k WikiText-2 tokens, 256-token windows, no BOS) reproduces their published table on Ada: INT7 survival 81.19% (published 81.26%), INT6 51.15% (51.30%), baseline perplexity 458.4 (458.5), INT7 perplexity -5.61% (-5.65%).

**Why our old pipeline disagreed** (survival = % of active features with r > 0.9; `reports/audit_our_vs_duan.md` section 6):

| factor | INT7 / INT6 survival |
|---|---|
| their pipeline, no BOS (their regime) | 81.4 / 51.3 |
| the same with one BOS token prepended (masked from the statistics) | 99.96 / 97.5 (their TransformerLens code, R2: 99.97 / 97.52) |
| per-row instead of their shared q/k/v scales, no BOS | 89.2 / 54.9 |
| per-row quantizer, but with BOS | within 0.83 pp of theirs with BOS |
| HF `sdpa` attention (no attention soft-capping); bf16 SAE | no measurable effect |

Perplexity is 457 without BOS and 16 with it; without BOS INT7 even *improves* perplexity by 5.7%, a sign of a degraded regime. Their 81% / 51% are therefore a property of feeding windows with no BOS. This says nothing yet about which regime the study should use.

**Old Phase-1 results** (FLORES, 27 languages, selective vs matched-universal features, resource tiers) are in `reports/legacy_pipeline/` and are **hypotheses, not findings** (see its README and `issues.md`).

## 3. The baseline replication of Duan (2026)

We run **Duan et al.'s own Phase-3 script**. `third_party/duan2026/` is their repository downloaded fresh and pruned to what this study needs; the working copy of the one script we run differs from their file by a short explicit whitelist (delete their unrunnable tail cells, switch off their pruning flag, and four device references so the second bf16 model sits on a second 11 GB card) and nothing else. The tests rebuild the working copy from their pristine file plus that whitelist and require byte equality over the whole file, and carry a committed mutation suite. Details and the exact edit table: `third_party/duan2026/UPSTREAM.md`; runbook: **[`HANDOFF.md`](HANDOFF.md)**. Quick start (needs the Ada environment described there):

```bash
python3 -m unittest discover -s scripts/g0/tests -v     # must say OK (vendoring + mutation tests, report and wandb tooling)
python3 scripts/g0/verify_vendor.py --verify            # pristine files match upstream; working copy == pristine + whitelist
# deploy and run on Ada (see HANDOFF.md):  sbatch -J base-full run_baseline.sbatch full full     (~4.1 h compute)
```

Diagnostics that modify THEIR code (BOS arms, INT0/INT16/INT12 controls, snapshots; always labelled "diagnostic", never "replication") are generated by `scripts/g0/make_duan_runners_v3.py` from their pristine scripts and launched with `scripts/g0/run_g0.sbatch`. `scripts/g0/r5_compare.py` compares per-cell summary CSVs against the baseline using the rules fixed in the audit.

Results are uploaded to the wandb project `anlp-project` (entity `karma-skz-`).

---

## 4. External Models and Dataset Links

* **Base Model Checkpoint:** [`google/gemma-2-2b`](https://huggingface.co/google/gemma-2-2b) (gated, fp32, 10.5 GB; used by Duan's code; each Ada job fetches it to node-local scratch).
* **Gemma Scope JumpReLU SAE:** [`google/gemma-scope-2b-pt-res`](https://huggingface.co/google/gemma-scope-2b-pt-res) (Layer 12, `average_l0_82`, 16,384 latents; the `layer_12/width_16k/canonical` SAE in `sae_lens`).
* **WikiText-2:** `Salesforce/wikitext`, `wikitext-2-raw-v1` (their token source).
* **FLORES-200 Parallel Evaluation Corpus** (not used by the current work; the old Phase-1 stage used it): [Meta NLLB FLORES-200 Official Public Archive](https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz) (original FLORES-200, not FLORES+).
* **Reference implementation (not ours):** Duan (2026), arXiv:2606.03002, artifact `https://anonymous.4open.science/r/sae-feature-survival-quantization-0141/`.
* **Extension Models & SAEs (planned, never run):** [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) and [`seonglae/gemma-2-2b-l12-topk-16k`](https://huggingface.co/seonglae/gemma-2-2b-l12-topk-16k).
* **Experiment tracking:** wandb project `anlp-project` (entity `karma-skz-`).
* **Large artifacts:** private Hugging Face dataset [`karma-skz/always-late-artifacts`](https://huggingface.co/datasets/karma-skz/always-late-artifacts) (per-feature tables, bootstrap snapshots, token windows); index and provenance in [`reports/artifacts.md`](reports/artifacts.md).

---

## 5. Repository Layout

```
always-late/
├── HANDOFF.md               # START HERE for a new agent: mission, rules, runbook, pending jobs
├── PLAN.md  PROGRESS.md     # current plan and honest status
├── issues.md                # review of the discarded pipeline and paper (kept as evidence)
├── paper/                   # ACL-format draft (contains superseded text; team must rewrite)
├── reports/
│   ├── progress_log.md      # dated evidence log, incl. the post-mortem of the discarded codebase
│   ├── audit_our_vs_duan.md # why our old pipeline disagreed with theirs; results of R2 and R5
│   ├── wandb_schema.md      # the one fixed wandb schema every run follows
│   ├── decisions_and_anomalies.md  # decision log, anomaly register, source-verification queue
│   ├── duan_codebase_audit.md
│   ├── g0/                  # baseline, diagnostics R2/R5 (+ R1 pending), benchmark, env freeze
│   └── legacy_pipeline/     # results of the discarded pipeline: hypotheses, not findings
├── third_party/duan2026/    # THEIR code (pruned, pristine copies + the one working copy we run); see UPSTREAM.md
├── scripts/g0/              # vendoring verifier, baseline launcher and report, diagnostic generator for their code, wandb, tests
└── anlp_revised.txt  project_guidelines.txt   # proposal and course guidelines
```

The earlier pipeline (`src/`, Phase-1 scripts, `requirements.txt`, root `figures/`, the HF audit harness) is not in the tree; check out the tag `legacy-pipeline-2026-10-16` to see it.

---

## 6. Compiling the Paper
The `paper/` directory is self-contained: `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`, or upload it to Overleaf. Do not submit it before its results and discussion are rewritten to match `reports/` (see `issues.md` A1-A6). The paper text must be written by the team (course policy).
