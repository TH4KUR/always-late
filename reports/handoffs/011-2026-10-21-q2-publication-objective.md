# Handoff 011: Q2 policy resolved -- publishability, justified revisions and checked novelty (2026-10-21)

## Current user instruction and review state

The user is reviewing `DECISIONS_FOR_REVIEW.md` one question at a time, with explanations before each decision. Q1 is closed: `mask` only, no further `none` work and no further numerical comparison with Duan (D-028). For Q2 the user says questions, proposal and direction can change whenever current results and analysis justify it; the intended direction must make a novel contribution, and **publishability is the main objective** (D-029).

Q2's operating policy is applied. This does not establish whether a frozen OSF registration historically existed. V-10 remains unverified for reporting purposes, not a blocker to justified revisions. Do not infer "not frozen" from permission to revise, do not retroactively call decisions pre-registered, and do not repeatedly ask the user to resolve that history before continuing the review.

## Project and evidence

ANLP project on language-selective SAE feature survival and resource-tier differences under RTN quantization of Gemma-2-2B. Existing experiments use a fixed layer-12 Gemma Scope 16k SAE and 27 FLORES-200 languages. Phase 1 and the existing additional analyses are complete; `reports/phase1_summary.md` and `reports/phase1_results.md` describe the evidence and limitations. `reports/phase1_reproduction.md` records regeneration from cached outputs. Many additional findings are exploratory or motivated by earlier observations, not established causal mechanisms.

The original Gemma table was reproduced using Duan's code before the multilingual extension. The closed audit, `reports/audit_our_vs_duan.md`, links completed BOS-masked and no-BOS WikiText/FLORES results. Preserve them as historical evidence and retain reused-code attribution. Duan's numerical results are no longer study targets or acceptance gates. BOS remains present in model input but excluded from feature statistics; appending EOS has not been tested or adopted.

## Research policy to apply

- The old proposal is revisable. Do not pursue a weak question just because it was proposed, and do not treat historical fixed-proposal instructions as current constraints.
- A change needs a recorded reason grounded in existing evidence: old/new question, what results were already seen, closest prior work, specific contribution, alternative explanations and the proposed evaluation.
- Before selecting a new direction, perform a dated search of the closest primary work. A new parameter combination alone does not establish a useful contribution; a bounded search cannot prove universal absence of prior work. No candidate has been certified novel in this documentation step.
- Publishability requires credible methods and evidence as well as the user's novelty requirement. Do not promise publication or choose thresholds to obtain a desirable result.
- Preserve earlier null and contradictory results. A result-inspired hypothesis is exploratory on those data. Independent confirmation needs a plan fixed before genuinely unseen outcomes are evaluated. COS guidance cited in the progress log supports transparent changes and the distinction between planned/unplanned analyses.
- The current change records policy only: no new research direction, analysis, GPU run, proposal rewrite or paper text was selected or carried out in this step.

## Next action

Continue with **Q3: what should the behavioural-link analysis measure, and what role does it have in a publishable contribution?** Explain that the existing analysis asks whether languages with more changes in selective features also have worse next-token loss after compression. The definitions under discussion are failure to survive (`r <= 0.9`) versus the narrower damaged class (`r < 0.5`), or dropping that correlation in favour of the separately proposed causal-ablation extension. They are different questions, not interchangeable thresholds to select by significance. Read Q3 and the existing behavioural report before recommending its role; no causal-ablation run is selected by Q2's general permission to revise.

Keep the one-question-at-a-time format. Do not silently launch a broad new-direction search or experiments in place of answering the user's next question. Other old review items may be revised or become irrelevant under D-029, but do not silently answer them.

## Operational scope and paths

- Work in this chat; no cron or second chat. No new jobs were launched, no code or saved results changed, and no commit/push was made in this documentation step.
- Q5's two-regime window run and Q9's optional Duan diagnostics are closed. Any future Q6 or Q15 run is mask-only and needs its own decision. Historical reproduction tools may still read `none` outputs for provenance; their presence does not reopen that work.
- Do not edit `paper/` prose or `third_party/`; the team writes the paper. Preserve attribution and historical outputs. No credentials or large artifacts in commits.
- Repository: `/home/karma/Documents/ANLP/project/always-late`. Sources of current policy: `DECISIONS_FOR_REVIEW.md`, `reports/decisions_and_anomalies.md` (D-028/D-029), `PLAN.md`, `PROGRESS.md`, and dated `reports/progress_log.md` entries.
- If a later authorized task needs the runtime: Ada project `/home2/karma/duan-verify`, environment `env/`; artifact index `reports/artifacts.md`; deployment/job guidance `reports/scheduled_routine.md`, subject to current research policy and mask-only scope. Never expose stored credentials.
