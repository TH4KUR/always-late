# Does Compression Forget Language? Language-selective SAE features under weight quantization

ANLP course project (IIIT Hyderabad, 2026). Gemma-2-2B, the Gemma Scope 16k JumpReLU SAE at layer 12, 27 FLORES-200 languages, round-to-nearest weight quantization from INT16 to INT4, built on Duan (2026)'s own code (vendored in `third_party/duan2026/`). Full original README (before the 2026-10-26 cleanup): [`reports/archive/root_pages_2026-10-26/README.md`](reports/archive/root_pages_2026-10-26/README.md).

## New here? Start in this order

1. **[`HANDOFF.md`](HANDOFF.md)**: how to take over (the work loop, the rules, the environment, the do-not list).
2. **[`reports/functional_validation/STATUS.md`](reports/functional_validation/STATUS.md), block `NEXT STEP`**: exactly what to do next. The `Log` below it says what was done last.
3. **[`reports/INDEX.md`](reports/INDEX.md)**: the full map of every document, and the trail of directions explored.

## Where to look for what

| you want | look in |
|---|---|
| the results on one page, with what each claim may and may not say | [`reports/functional_validation/RESULTS_SUMMARY.md`](reports/functional_validation/RESULTS_SUMMARY.md) |
| Study G (graded gain, the user's major study, finished 2026-10-25) | [`reports/functional_validation/gain_study/README.md`](reports/functional_validation/gain_study/README.md) |
| what is running, what is next, the dated log of the current work | [`reports/functional_validation/STATUS.md`](reports/functional_validation/STATUS.md) |
| the newest handoffs | [`reports/handoffs/`](reports/handoffs/) (highest number; 040 is the Gemma-2-9B handoff for the other cluster) |
| questions waiting for the user, with defaults | [`DECISIONS_FOR_REVIEW.md`](DECISIONS_FOR_REVIEW.md) |
| every decision and anomaly (append-only, D-/A-/V- ids) | [`reports/decisions_and_anomalies.md`](reports/decisions_and_anomalies.md) |
| designs and plans (written before the data) | [`docs/superpowers/specs/`](docs/superpowers/specs/), [`docs/superpowers/plans/`](docs/superpowers/plans/) |
| figures for the paper | [`reports/functional_validation/FIGURE_INDEX.md`](reports/functional_validation/FIGURE_INDEX.md) |
| re-running every analysis | [`reports/functional_validation/REPRODUCE.md`](reports/functional_validation/REPRODUCE.md) |
| current research direction and publication limits | [RESEARCH_DIRECTION.md](RESEARCH_DIRECTION.md), [novelty assessment](reports/novelty_assessment_2026-10-26.md) |
| historical literature and search records | [`reports/research_sources.md`](reports/research_sources.md), [`reports/functional_validation/prior_art.md`](reports/functional_validation/prior_art.md) |
| the 40-minute cron job's instructions | [`reports/functional_validation/cron_prompt.md`](reports/functional_validation/cron_prompt.md) |
| old text moved out of living pages (verbatim) | [`reports/archive/`](reports/archive/) |

## Studies and their state (2026-10-26)

| study | state | entry point |
|---|---|---|
| Replication of Duan (2026) on Ada | done; reproduced within 0.15 pp | [`reports/duan_codebase_audit.md`](reports/duan_codebase_audit.md) |
| Phase 1: selective vs control survival | done; pre-registered tests null after Holm | [`reports/phase1_summary.md`](reports/phase1_summary.md) |
| D-035: language association retention, 27 languages | done | [`reports/association_retention/README.md`](reports/association_retention/README.md) |
| Functional validation F1 to F5, Belebele (confirmatory) | done | [`RESULTS_SUMMARY.md`](reports/functional_validation/RESULTS_SUMMARY.md) |
| Study G: graded gain, extensions B, C, D | done | [`gain_study/README.md`](reports/functional_validation/gain_study/README.md) |
| Llama-3.2-1B consistency check (D-087/D-090) | complete; exploratory verdict **not consistent** | [results](reports/functional_validation/llama/report.md) |
| **Gemma-2-9B confirmation** (replaces Llama, D-089) | handed to another team on a separate cluster, branch `gemma9b-confirmation` | [handoff 040](reports/handoffs/040-2026-10-26-gemma9b-confirmation-external-cluster.md) |
| **Intervention-transfer study** (D-088/D-094) | main matrix complete; overall utility not met; external validation pending in latest recorded ledger | [verification](reports/intervention_transfer/verification_2026-10-26.md), [`RESEARCH_DIRECTION.md`](RESEARCH_DIRECTION.md) |
| Course report | revision 3; see its recorded official-check status | [report](paper/course-report/README.md) |

## Headline results (exploratory unless marked; one model, one layer, one SAE)

- Zeroing a language's 64 most selective features harms that language's next-token loss beyond activity-matched controls in 27 of 27 languages (macro 0.704 nats per token). The dependence is flat through INT5 and larger at INT4 (macro retention 2.14).
- **Confirmatory:** the dependence does not show on Belebele reading comprehension.
- The FP-donor language-group patch recovers about 2.6 % of the INT4 loss gap (matched controls: 1.2 %); this is not a universal bound on their causal contribution.
- F5 fingerprints change little through INT8 relative to the INT16 RTN reference. F5 swapped-direction comparisons showed similar INT4 drift; the new matrix's specificity remains qualified by unequal eligibility and FP effect scales.
- Study G: harm is graded in the down-scaling ratio in 27 of 27 languages; turning features up gives only a very small repair at INT4, and only near ratio 1.1 (extension C, D-082). Inverse shrinkage does not predict the best gain.

Details and limits: [`RESULTS_SUMMARY.md`](reports/functional_validation/RESULTS_SUMMARY.md).

## External models, data and tracking

- **Base model:** [`google/gemma-2-2b`](https://huggingface.co/google/gemma-2-2b) (gated).
- **SAE:** [`google/gemma-scope-2b-pt-res`](https://huggingface.co/google/gemma-scope-2b-pt-res) `layer_12/width_16k/canonical`. Second model (D-089): `google/gemma-2-9b` with `gemma-scope-9b-pt-res-canonical` `layer_20/width_16k/canonical`.
- **Data:** FLORES-200, the original archive `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` (dev for tagging, devtest for evaluation). Belebele for the task test.
- **Reference implementation:** Duan (2026), arXiv:2606.03002.
- **Tracking and artifacts:** wandb project `anlp-project` (entity `karma-skz-`). Large artifacts are in the private Hugging Face dataset [`karma-skz/always-late-artifacts`](https://huggingface.co/datasets/karma-skz/always-late-artifacts); index in [`reports/artifacts.md`](reports/artifacts.md).

## Repository layout

`scripts/g0/` all code (runners, analyses, `tests/` stdlib, `tests_torch/` torch) · `third_party/duan2026/` their code, never edited by hand · `reports/` everything written (see [`reports/INDEX.md`](reports/INDEX.md)) · `docs/superpowers/` designs and plans · `paper/` the team's paper (agents do not edit it) · `anlp_revised.txt` the original proposal · `project_guidelines.txt` the course rules.
