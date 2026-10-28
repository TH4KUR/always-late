# Stage 2 design: the survival sidecar (D-119; written 2026-10-29 05:28 IST BEFORE any code)

Plan: [`2026-10-29-survival-measurement-validity-study.md`](../../docs/superpowers/plans/2026-10-29-survival-measurement-validity-study.md) (Stage 2); ledger [`STATUS.md`](STATUS.md); Stage 1: [`stage1/README.md`](stage1/README.md) (D-118). Written by the 40-minute cron session (project-6d) as task 3 of the ledger. **This note only designs; no code, no Ada action, no number exists for it.** Its facts were read from the repository on 2026-10-29: `phase1_run.run_survival` (and its `profile_sink` hook), `phase1_lib.LangPearson`, `phase1_association_collect.ProfileSink`, their `encode_sae`, the Stage 1 run metadata.

## 1. What Stage 2 must deliver and why a sidecar

Stage 1 could only use the finished per-language correlations (`survival.pt`: `corr`, `firing_rate`, `mean_activation`, `max_activation`). The raw sums behind the correlation were not saved, so the decomposition b = g a + c + e (gain, noise, flips), the per-feature spread sigma_a (so that the scale covariate is no longer a per-language proxy), co-active statistics, flip counts and the encoder pre-activation perturbation in threshold units (metric 2, nu) cannot be recovered from the committed files. Stage 2 streams them once.

**No frozen or hash-pinned file is edited.** `phase1_run.run_survival(..., profile_sink=sink)` already hands every batch of every condition (FP, then each bit width in the order given, including INT0 = restore-only) to a sink and states that its returned results are identical with and without a sink (D-035). The sidecar is a NEW entry point `scripts/g0/survival_sidecar.py` (module `survival_sidecar:main`, run through `scripts/g0/gptq_ext/run_with_weights.py --entry survival_sidecar:main --weights rtn|rtn-matched|gptq`, so the three arms use the same weight machinery as stage 1) with a new `SurvivalSink` class; its setup copies the survival-mode setup of `phase1_run.main` (their functions verified against the frozen manifest, model, SAE, FLORES devtest, equal-budget masks) and calls `run_survival` unchanged.

## 2. Streamed quantities (per condition, per language, per feature j; eq stream only)

Notation: a = full-precision activation (post-JumpReLU), b = quantized activation, on the same tokens; tokens are the valid, non-BOS, non-pad positions that belong to the equal-budget stream of the language (`b["eq"]`, D-015: the pre-registered primary stream); pre = encoder pre-activation, theta_j = the JumpReLU threshold, d = pre_FP - theta_j (the margin; d >= 0 on FP-active tokens). All sums float64 (as `LangPearson`), counts int64. Per (condition, language, feature), 21 values:

| group | values |
|---|---|
| all tokens (5) | Sa, Sb, Saa, Sbb, Sab (the sums of `LangPearson`, so that the committed correlation is rebuilt with the same formula) |
| co-active tokens, a > 0 and b > 0 (6) | n_both, Sa, Sb, Saa, Sbb, Sab over those tokens |
| activity and flips (4) | n_a (a > 0), n_b (b > 0), n_on_off (a > 0, b = 0), n_off_on (a = 0, b > 0) |
| FP-active tokens, encoder side (6) | S_dpre = sum(pre_Q - pre_FP), S_dpre2 = sum((pre_Q - pre_FP)^2), S_d = sum d, S_d2 = sum d^2, n_exceed2 = count(\|dpre\| > d), n_exceed1 = count(dpre < -d) |

Per language also the token count n (scalar, equal to the committed `n`). Per condition 21 x 27 x 16,384 x 8 bytes = **74 MB**. Derived later (Stage 3, not here): r, g = Cov(a,b)/Var(a), sigma_a, sigma_e, lambda = log(1/r^2 - 1), nu_j = sqrt(S_dpre2 / n_a) / theta_j, predicted-flip share n_exceed1 / n_a, and the identity check that r rebuilt from g, sigma_a, sigma_e equals the measured r.

**Pre-activation source:** the SAE's own pre-activation hook (`sae.hook_sae_acts_pre` in sae_lens 6.x, captured through a forward hook inside a wrapper of their `encode`; fallback: `x @ W_enc + b_enc` of the SAE with the same input processing); gate G3 verifies `acts == pre * (pre > theta)` on the first batch before anything is accumulated. theta_j comes from the loaded SAE and is cross-checked against `reports/phase1/sae_thresholds.csv`.

**Not streamed:** the all-token stream (the sink would double its size; the equal-budget stream is the primary one, and the all-token committed correlations remain available for Stage 1); per-token data; the loss (already in the committed runs).

## 3. Conditions and jobs

| job | arm / weights | conditions (in order) | build |
|---|---|---|---|
| `sc-legacy` | legacy RTN (`--weights rtn`, nothing patched) | FP, INT16, INT6, INT4, INT0 | none |
| `sc-matched` | matched RTN (`--weights rtn-matched`) | FP, INT6, INT4, INT0 | 2 artifacts, about 15 min each |
| `sc-gptq` | GPTQ (`--weights gptq`) | FP, INT6, INT4, INT0 | 2 artifacts, about 19 min each; every artifact sha256 checked against `reports/gptq_extension/artifact_hashes.json` (INT6 recorded in earlier jobs, INT4 enforced) |

INT16 (the legacy rounding, identical in all arms by construction) is streamed once, in the legacy job. FP is not saved (its sums follow from b = a); INT0 is used as a gate only. Saved: `sums_<condition>.pt` for INT16 (legacy job), INT6 and INT4 (each job), `meta.json` (theta, token counts per language, the gate results, versions, commit), `weights_manifest.json` as in the GPTQ extension. Outputs per job about 150 MB (legacy 225 MB), total about 525 MB; Ada has about 2.3 GB free on `/home2/karma`: after validation each job's directory is copied locally and its sha256 recorded; **nothing is deleted on Ada**.

## 4. Gates (a failed gate ends the job; nothing is claimed from a job that failed one)

- **G1, INT0 equals FP:** in the INT0 condition every streamed statistic equals the FP value of the identity (n_on_off = n_off_on = 0, S_dpre = S_dpre2 = 0, Sab = Saa = Sbb up to float tolerance 1e-12 relative), checked in the job.
- **G2, the correlation is rebuilt:** r from (Sa, Sb, Saa, Sbb, Sab, n) with the `LangPearson` formula equals (i) the correlation returned by `run_survival` in the same job (to 1e-9; same inputs, same formula) and (ii) the committed `survival.pt` corr of the arm's stage-1 run, `mask/eq` (legacy `~/.cache/anlp-always-late/phase1_runs/p1-mask/eq`, `transfer_runs/{rtnm,gptq}-p1-mask/eq`), for every language, feature and condition present in both (to 1e-4; the committed files may come from other GPU nodes), reported as the maximum difference. A larger difference is investigated, not tolerated.
- **G3, the SAE arithmetic:** on the first batch `acts == pre * (pre > theta)` (max abs difference 1e-5) and theta equals the committed threshold table.
- **G4, flips agree with the pre-activation:** n_on_off equals n_exceed1 exactly in every (language, feature) (a flip off is exactly pre_Q < theta); n_a equals the committed firing rate times n.
- **G5, resources:** a smoke job (three languages, 40 sentences each, FP, INT6, INT0; G1 to G4 on it), then a timing and memory reading scaled to the full run before the full jobs are queued; GPU memory below 11 GB; `df` on Ada checked before the jobs; at most two jobs RUNNING; the full jobs are queued at once after the gates pass (user rule of 2025-10-25), each with its own time limit of 5 h.

## 5. Runtime and memory estimate (to be replaced by the smoke job's measurement)

The stage-1 GPTQ job (FP + eight conditions + their builds + the dev tagging pass) took 19,280 s (5 h 21 min), the matched-RTN job 21,097 s; five GPTQ builds of about 19 min are about 1.6 h of that. A sidecar job has no tagging pass, three or four conditions and two builds: **about 1.5 to 2.5 h per job, about 6 GPU hours in total** for three jobs. The sink adds 74 MB of float64 accumulators per condition on the GPU and a host-side sparse cache of the FP activations and pre-activations (about 60 M active entries x 12 bytes, under 1 GB, against `--mem=40G`). The Phase 1 job fits an 11 GB card; the sink's extra memory is small.

## 6. Tests (test first, mutation-checked; before any Ada job)

Hand-computable toy sink tests on a tiny dictionary (3 features, 2 languages, known a, b, pre, theta): every one of the 21 sums for a known token set; the eq mask; a language with no selected token; flips and exceed counts; the correlation rebuilt from the sums equals `LangPearson`; the INT0 identity; accumulation across batches equals one batch; the sink integrated with `run_survival` on the existing toy harness (forward, encode, quantize, restore) returns the same `results` with and without the sink; saved files load and validate (a validator in the style of `gptq_ext/validate_job.py`: complete meta, gates true, weights hashes). Mutation battery with `scripts/g0/mutcheck.py`. Then the CPU run on the toy and a read-only review by session project-fe before the first Ada submission.

## 7. Authorization and the hold point

The Ada GPU jobs of this stage rest on the user's Q33 answer (a), "you are free to use ada", as recorded in `DECISIONS_FOR_REVIEW.md` and relayed in the cron-prompt block of 2026-10-29 by session project-fe, which quotes the user's instruction to it. The executing session has not received those words from the user directly; before the FIRST Ada submission (the smoke job) it asks the user to confirm in its own report, and submits after the confirmation (or after the user's own message). Everything before that point (this design, the code, the tests, the review) is CPU-side and local.

## 8. What this design does not decide

The Stage 3 thresholds and the H4 equivalence bound (task 8; H4 against the INT16 tier slope, not zero) and the analysis itself. No new model, layer, SAE, quantizer, no edit of a frozen file, no rerun that replaces a legacy result.
