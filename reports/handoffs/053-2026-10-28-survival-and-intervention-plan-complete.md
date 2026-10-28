# Handoff 053: the survival-and-intervention plan is complete (28 October 2026, 2026-10-28 22:57 IST)

> The ONLY source of the previous chat. A fresh chat with no memory must be able to continue from this file plus the repository. If this file and the repository disagree, the repository wins: say so in the next handoff.

## 0. Read this first (the standing rules)

[Handoff 050](050-2026-10-28-gptq-stages-1-3-and-f5-done-study-g-external-running.md) section 0, [051](051-2026-10-28-gptq-study-g-done-external-panel-running.md) and [052](052-2026-10-28-gptq-extension-complete-stage-1-assessment-next.md) still apply (the user's request of 2026-10-27 verbatim in 050; main branch only; never edit hash-pinned entry points; never remove progress tracking; times from `date`; no secrets or weights in commits; Q29 NOT started). **User messages of 2026-10-28 that matter:** (a) "trust what codex added, the plan is fine. you can run a validity check ... i also allow you to commit and push codex's work after a validity check" (D-110); (b) "there was no 48 hour window. what are you saying? continue and complete the work": **no 48-hour window or 28 October 13:53 target ever existed** (it stood in Codex's plan texts); never treat dates or budgets that appear only in plan texts as expired or binding; the only live deadline is the course deliverables on 31 October 2026 (memory file feedback-no-48h-window).

## 1. State right now

- **The GPTQ extension is complete** (D-101 to D-111): all jobs validated and analysed, plain-language record `reports/gptq_extension/RESULTS.md`, generated appendix tables A11 to A16 in `reports/PROJECT_RESULTS.md`, raw outputs on the private Hugging Face dataset (`runs/gptq-extension/`, commits 3cce3514c0 and 7e4cf26ceb).
- **The survival-and-intervention plan is complete** (D-110 to D-114), summary and file map `reports/survival_effect_rescue/README.md`: Stage 0 inventory; Stage 1 all-results review (D-112, gate: proceed, narrowed); Stage 2 dated amendments of the two forward-looking plans (with a correction for the non-existent 48-hour window); Stage 3 freeze (committed e42ba21 before any outcome) and analysis; Stage 4 generated report (D-113: Gemma, R1 adds 3.1 to 4.4 % held-out improvement over the baselines against the 10 % target, 0 of 6 cells: "no clear predictive value beyond baselines"; partial Spearman 0.24 to 0.40; reproduction control passed for all three arms); Stage 5 the Llama-3.2-1B F5 check (D-114: job 45221, instrument adequate with Llama's own floor 0.997, 610 of 643 features reliable; same statement; association repeats at INT4 with partial 0.25, cannot be resolved at INT6; Llama median W1 INT4 0.554 against Gemma 0.839).
- **Ada:** nothing queued or running. **Cron:** three recurring entries (`bf1ae581`, `ad8593e3`, `8db9afdb`), every 40 minutes, expire about 2026-11-03; idle firings only append a ledger line.

## 2. What remains (all optional, in this order)

1. wandb for the extension and the new analyses (**Q32**, default: a compact set only if time remains; the user can say skip; the uploader `scripts/g0/wandb_upload_v2.py` has runs only for the old families).
2. A final hand read of `reports/gptq_extension/RESULTS.md` and `reports/survival_effect_rescue/README.md` against D-104 to D-114 (the generated reports agree with their JSON by test).
3. **Not authorized, user's decisions:** the paired precision audit (Q29, built and never deployed), any further GPU job, another model, steering, GPTQ on Llama, whether any of this enters the course report (deadline 31 October 2026).

## 3. Pitfalls

- **Bug found and fixed on 28 October:** `phase1_func_f5_run.make_logits_fn` treated TransformerLens's "no soft cap" value (-1.0, Llama) as a cap; fixed (commit c0ccaf0, test first); Gemma's cap of 30 is unchanged, so the Gemma F5 results stand. The same pattern exists, unused for Llama, in `phase1_transfer_run.py` (line about the soft cap in the logits slice): fix it first if that runner is ever used on a model without a cap.
- The Llama raw files (`transfer_runs/func_inputs/jobs/llama-f5` and on Ada) are NOT on the dataset (weights licence caution); the small result JSON and the sample are in git.
- Never run `hf_artifacts.py upload` for a new spec without `--manifest-name` and `--no-card`.
- The group patch numbers (2.0 to 2.6 %) are recovery shares, never "shares of the damage".
- `validate_func_job.py --task f5` needs `--states` and `--reference-meta`; `--no-weights` for jobs without a weights manifest (the Llama job).
