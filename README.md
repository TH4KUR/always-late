# Does Compression Forget Language? Language-Specific Feature Survival Under Quantization in Multilingual LLMs

**Current state (this block is kept short and points only at files that are always up to date):** the 27-language association-retention study (D-035) is finished, its result is summarised just below; the cross-precision functional-validation study (D-039 to D-065: F1 to F5 and the confirmatory Belebele family) is finished and summarised on one page for the team: [`RESULTS_SUMMARY.md`](reports/functional_validation/RESULTS_SUMMARY.md). Live state, next steps and log: [`reports/functional_validation/STATUS.md`](reports/functional_validation/STATUS.md); newest handoff: [`HANDOFF.md`](HANDOFF.md); results and figures: [`reports/functional_validation/`](reports/functional_validation/README.md). Earlier banners of this file, superseded by the above, are preserved verbatim in [`reports/archive/superseded_banners_2026-10-22.md`](reports/archive/superseded_banners_2026-10-22.md).

**Result of the D-035 study, 2026-10-21:** the run (Slurm 31640) and its analyses (31807, 31808; reproduced byte for byte by 31919, 31920) are finished. Of the 8,086 original selective features, 6,663 (82.4 %) reproduce their dev language peak on devtest in full precision; among those, 98.1 / 97.5 / 94.7 / 89.1 / 70.8 % keep their original language as the unique, supported peak at INT8 / INT7 / INT6 / INT5 / INT4 (numerical floor INT16/INT12: 99.2 %). Features are lost by the peak moving to another language, almost never by going silent; features that keep their peak show no visible change of target share or margin; the correlation criterion of the earlier survival analysis disagrees with association retention in both directions. Exploratory on already examined data; nothing is shown about function or meaning. [Interpretation and limits](reports/association_retention/interpretation.md), [generated report](reports/association_retention/results/eq/association_retention.md), [status](reports/association_retention/STATUS.md).

**Functional validation (D-039 to D-065), exploratory except one frozen family, 2026-10-23 (one-page summary with every number traced to its file: [`RESULTS_SUMMARY.md`](reports/functional_validation/RESULTS_SUMMARY.md)):** zero-ablating a language's selective SAE feature group harms that language beyond activity-matched controls in 27 of 27 languages on held-out next-token loss (300 frozen FLORES devtest sentences, median 0.26 nats per token; Gemma-2-2B, layer 12, round-to-nearest weights). The dependence is preserved through INT5 (macro retention of the effect 1.01 at INT8, 1.03 at INT7, 1.04 at INT6, 1.13 at INT5) and increases at INT4 (2.14, 16 of 27 languages amplified), where the D-035 association also falls fastest (70.8 % retained) and the SAE reconstruction degrades most. Association is lost mostly among the lower-ranked, less active features (at INT4 the top 64 of a language retain better in 27 of 27 languages), but the matched comparison at equal native activity (F4, INT4, INT5 and INT6) shows no systematic difference between features that lost and features that kept their association (medians within 0.001 nats per token). **The one confirmatory family, the Belebele multiple-choice task (frozen before any item was scored; 18 covered languages, 385 held-out items each, letter format), did not show the dependence: the macro effect at full precision is -0.0001 nats per item [-0.0035, +0.0031] (statement S0 not met) and no covered language has a clearly positive full-precision effect, so the retention statements S1 to S5 are undefined; the ablation leaves the accuracy unchanged (0.426) and the INT4 model is below chance (0.236).** In this design (a 64-feature group, zero ablation, one prompt format) the dependence is therefore a next-token-loss phenomenon; the exploratory K = all sensitivity (the whole selective group, up to 1,719 features) gives the same null (macro effect +0.0007 [-0.0030, +0.0042]; accuracy 0.4257 without edit, 0.4209 group zeroed, 0.4226 matched controls zeroed; D-062). **Where the quantization damage sits (F3, D-058):** the language groups account for at most about 3 % of the loss gap to full precision (2.6 % at INT4, nothing measurable at INT5 and INT6) and at most 30 to 40 % of it sits in the layer-12 residual. **Does a single feature still do the same thing (F5, D-059 to D-064; 1,487 features):** the effect of switching one feature off on the output logits stays within weight-rounding noise through INT8 (median cosine with full precision 0.997 at INT8, 0.988 at INT6, 0.961 at INT5, 0.839 at INT4; numerical floor 0.949), but a control that ablates a different feature's direction at the same positions loses about as much similarity at every precision, so the loss is generic drift of the quantized model, not feature-specific, while each feature stays individually identifiable (retrieval 0.997 to 1.000). The activation-based survival criterion misclassifies about 31 % of the functionally intact features at INT5 and the D-035 association status carries no functional information, so feature-level functional continuity cannot be told apart from the model's overall continuity; the association loss itself is mostly a threshold effect (the feature's own full-precision margin predicts it within a language with an AUROC of 0.90 at INT6, 0.81 at INT5 and 0.75 at INT4; A2, D-066) while the same margin hardly predicts whether the feature's function changed (AUROC 0.58 at INT6 on the same features; A3, D-067). All of this is exploratory, for one model, one layer, one SAE and round-to-nearest weights. [Status](reports/functional_validation/STATUS.md), [F5 report](reports/functional_validation/results/f5/report.md), [F5 control figure](reports/functional_validation/figures/f5/ladder_control_light.png), [loss-task results](reports/functional_validation/results/eval/report.md), [Belebele statements](reports/functional_validation/results/mc_eval/statements.md), [F4](reports/functional_validation/results/f4/INT4/report.md), [rank analysis](reports/functional_validation/results/assoc_rank/report.md), [figures (loss task)](reports/functional_validation/figures/eval/), [figures (Belebele)](reports/functional_validation/figures/mc_eval/), [decisions D-043 to D-065](reports/decisions_and_anomalies.md).

**Course**: Advanced Natural Language Processing (ANLP, Monsoon 2026)  
**Institution**: International Institute of Information Technology, Hyderabad (IIIT-H)  
**Team**: Always Late (AlwaysLate)  
**Authors**: Hemang Joshi, Eashaan Thakur, Shardul Joshi, Nikhita Ravi  
**GitHub Repository**: [https://github.com/TH4KUR/always-late](https://github.com/TH4KUR/always-late)

> **Status (updated 2026-10-21). Read before using any number from this repository.**
> - **Q13-Q15 applied:** report both matching comparisons as associations and investigate magnitude's role without assuming a mediator. Keep overall tier disparity and an extra selective-feature penalty distinct; their historical tests already exist. Investigate confidence/diffuse activity against representation-scale and SAE-fidelity alternatives. [Research source register](reports/research_sources.md) keeps the papers, official sources, publication limits, saved results and search record together. No particular new GPU job is selected.
> - **Q3/Q4 update:** survival detection is under reassessment; INT5 and INT4 are included. Existing survival percentages measure activation correlation in a fixed SAE for a cohort selected before quantization. They do not establish retained language specificity or causal function. [Accepted sources, method limitations and proposed checks](reports/survival_method_reassessment.md). No replacement metric or new confirmatory pair has been selected.
> - **Main objective (Q2, D-029): publishability through a novel, well-supported contribution.** Questions, proposal and direction may change for documented reasons grounded in current results and analysis. A new direction needs a dated check against the closest prior work; none has been certified novel merely by this policy decision. Result-informed changes are disclosed, and historical pre-registration claims remain unverified without an actual OSF record.
> - **Current scope (Q1, D-028): `mask` only.** `none` is retired from ongoing work, including sensitivities and future diagnostics. Our evaluation methodology has diverged from Duan's; no further numerical comparisons or baseline acceptance gates against their work are planned. The completed comparisons remain [historical evidence with direct result links](reports/audit_our_vs_duan.md#current-disposition----closed-2026-10-21-q1-d-028). Reused code retains its attribution.
> - **Our previous pipeline (`src/` and the Phase-1 scripts) was discarded.** It was an unverified re-implementation of Duan et al. (2026) and its headline interpretations contradicted its own data. The repository now builds on **Duan's own code** (`third_party/duan2026/`) and the Ada setup that reproduces it. The post-mortem (what was wrong, why results conflicted, whether the direction was worth it) is the entry "POST-MORTEM of our previous codebase" in [`reports/progress_log.md`](reports/progress_log.md). The old code is recoverable from the git tag `legacy-pipeline-2026-10-16`.
> - **What is established:** their published Gemma Phase-3 table reproduces with their own code on Ada (within 0.15 pp), and the disagreement between their numbers and ours is a **regime difference, not a bug**: their windows have no BOS token (baseline perplexity 457), ours had one. With one BOS token their own pipeline gives INT7 / INT6 survival of 99.97% / 97.5% instead of 81% / 51%, confirmed in two software stacks (their TransformerLens code and an HF reimplementation agree to 0.1 pp). Evidence: [`reports/audit_our_vs_duan.md`](reports/audit_our_vs_duan.md).
> - **Q1 and Q2's operating policy are resolved:** `mask` only, no further Duan comparison, and justified revisions allowed (D-028 / D-029). Historical OSF freeze status (V-10) remains unverified for reporting purposes and is not a blocker to revising the research plan. Other choices are listed in `DECISIONS_FOR_REVIEW.md`.
> - Phase 1 (language-selective features; `PLAN.md` S1-S8 and the analyses N1 to N13) has run on top of their code and is **reproduced end to end** from the cached run files by the committed scripts ([`reports/phase1_reproduction.md`](reports/phase1_reproduction.md), D-026). **The state of the evidence is in one place: [`reports/phase1_summary.md`](reports/phase1_summary.md)** (every number generated from the committed result files, a status label per analysis); the full account is `reports/phase1_results.md`. In words:
>   - the pre-registered selective-versus-control tests are null after correction in the primary regime (BOS masked, equal token budgets);
>   - a selectivity penalty appears only at strict S thresholds chosen after the fact, and activation magnitude accounts for it;
>   - the large effect in the data is the resource-tier effect on survival, for selective and universal features alike; it is carried statistically by per-language activation scale (the proposal's "localizes upstream" branch), and character fertility explains only part of it;
>   - the JumpReLU threshold margin adds nothing beyond raw magnitude; two residuals stay unexplained;
>   - everything beyond the pre-registered family is additional or exploratory, correlational (27 languages) and labelled so; the completed BOS-free results are historical evidence from a regime that failed the project's noise-floor check.
>   - Q3/Q4 and Q13-Q15's reporting/investigation directions are recorded; the replacement measurement design remains to be selected. Other unanswered review items remain in `DECISIONS_FOR_REVIEW.md`; historical OSF status remains unverified.
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
* **FLORES-200 Parallel Evaluation Corpus:** [Meta NLLB FLORES-200 Official Public Archive](https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz) (original FLORES-200, not FLORES+). The completed current Phase 1 uses dev for tagging and devtest for evaluation; these examined outcomes are not fresh confirmation data for a revised hypothesis.
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
