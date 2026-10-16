# Does Compression Forget Language? Language-Specific Feature Survival Under Quantization in Multilingual LLMs

**Course**: Advanced Natural Language Processing (ANLP, Monsoon 2026)  
**Institution**: International Institute of Information Technology, Hyderabad (IIIT-H)  
**Team**: Always Late (AlwaysLate)  
**Authors**: Hemang Joshi, Eashaan Thakur, Shardul Joshi, Nikhita Ravi  
**GitHub Repository**: [https://github.com/TH4KUR/always-late](https://github.com/TH4KUR/always-late)

> **Status (2026-10-16). Read before using any number from this repository.**
> A code-and-results review (`issues.md`) found that the earlier headline claims in this README and in `paper/main.tex` are **not supported by the team's own data**, and that our replication of Duan et al. (2026) was an independent re-implementation that was never validated against their code. We are now running **their code** (vendored in `third_party/duan2026/`, edits limited to a short explicit whitelist) on Ada to establish a baseline before any new claim is made.
> - What is going on and what to do next: **[`HANDOFF.md`](HANDOFF.md)** (for a new agent or teammate), [`PLAN.md`](PLAN.md), [`PROGRESS.md`](PROGRESS.md).
> - Evidence: [`issues.md`](issues.md), [`reports/duan_codebase_audit.md`](reports/duan_codebase_audit.md), [`reports/audit_our_vs_duan.md`](reports/audit_our_vs_duan.md) (why our pipeline disagrees with theirs; experiments running), [`reports/progress_log.md`](reports/progress_log.md).
> - `paper/main.tex` still contains superseded text (see `issues.md` A1-A6). It must be rewritten by the team.

---

## 1. Project Overview & Core Mechanistic Question

Post-training quantization (PTQ) induces well-documented, uneven performance degradations across typologically diverse languages, disproportionately penalizing non-Latin scripts and low-resource languages. Concurrently, recent mechanistic audits demonstrate that aggregate metrics like perplexity fail to capture internal Sparse Autoencoder (SAE) feature damage. However, existing compression audits treat all internal features as a monolithic population.

In this project, we ask:
> **Core Question:** *Does compression-induced damage to language-selective internal representations provide a mechanistic explanation for the uneven degradation of multilingual language models?*

We track Gemma-2-2B residual-stream features at Layer 12 using Gemma Scope JumpReLU SAEs across a 27-language panel from FLORES-200 under round-to-nearest (RTN) quantization (INT8 to INT4).

---

## 2. What the results currently say (20,000 tokens/language run; details in `PROGRESS.md` and `issues.md`)

These are measured numbers from `reports/`, stated without the earlier over-interpretation. Treat them as *provisional* until the baseline (section 3) closes the instrument-validity question.

- **Selectivity:** 2,342 of 11,670 supported layer-12 features pass the selectivity cut S(f) >= 3 (High tier 839, Mid 444, Low 1,059). S >= 3 is a loose criterion (a feature firing equally in 9 of 27 languages passes), the null cutoff is not a calibrated null, panel content is not matched across languages, and only 19 of 27 languages have >= 50 selective features (`issues.md` C1-C3).
- **RQ1 (selective vs firing-rate-matched universal features):** at the pre-registered widths both arms are at ceiling and indistinguishable (INT7 99.87% vs 99.96%, p = 0.63; INT6 97.01% vs 97.35%, p = 0.53). The sign flips at INT5 (selective more robust, p = 4e-4) and INT4 (less, p = 3e-4). There is no consistent feature-type effect. Matching on firing rate alone does not control the main driver of survival (activation magnitude; Duan's own data: Spearman 0.78 vs -0.085 for rate).
- **RQ2 (resource tier):** a small, consistent low-resource penalty exists (INT6 survival High 99.05% / Mid 99.3% / Low 94.4%; language-level exact permutation tests p ~ 0.0003-0.003 at INT6-INT4). This **contradicts** the earlier "null, so the gap is upstream in tokenization" conclusion. It is confounded with script, tokenization and content coverage.
- **Behavioral link:** exploratory Spearman rho ~ 0.49 (n = 27, uncorrected); about 0.35 after controlling for tier; INT7 loss changes are at noise level.
- **Instrument validity (the open problem):** our pipeline gives INT7/INT6 survival of 98%/94% where Duan reports 81%/51%. Their released data show a degraded regime (no BOS token; baseline WikiText-2 perplexity 458; non-monotone perplexity vs bits). **Baseline done (2026-10-16): their own code, vendored with only device/flag edits, reproduces their published Gemma Phase-3 table on Ada to within 0.15 pp at every bit-width** (INT7 survival 81.19% vs 81.26%, INT6 51.15% vs 51.30%, baseline perplexity 458.4 vs 458.5, INT7 perplexity -5.61% vs -5.65%). So their degraded regime is real in their pipeline. Why our own pipeline differs is being audited (`reports/audit_our_vs_duan.md`: 12 differences, including that our HF pipeline silently skips Gemma-2's attention soft-capping and runs with a BOS token; R5 result: their regime reproduces inside our HF stack; the BOS token is the dominant factor (with BOS, perplexity 16 instead of 457 and INT6 survival 97-98% instead of 51%); R2, their TransformerLens code with BOS, agrees with our HF result to 0.1 pp; see the audit).

## 3. The baseline replication of Duan (2026)

We run **Duan et al.'s own Phase-3 script**. `third_party/duan2026/` is their repository downloaded fresh and pruned to what this study needs; the working copy of the one script we run differs from their file by a short explicit whitelist (delete their unrunnable tail cells, switch off their pruning flag, and four device references so the second bf16 model sits on a second 11 GB card) and nothing else. The tests rebuild the working copy from their pristine file plus that whitelist and require byte equality over the whole file, and carry a committed mutation suite. Details and the exact edit table: `third_party/duan2026/UPSTREAM.md`; instructions, runbook and pass criteria: **[`HANDOFF.md`](HANDOFF.md)**. Quick start (needs the Ada environment described there):

```bash
python3 -m unittest discover -s scripts/g0/tests -v     # must say OK (includes the vendoring + mutation tests)
python3 scripts/g0/verify_vendor.py --verify            # pristine files match upstream; working copy == pristine + whitelist
# deploy and run on Ada (see HANDOFF.md section 3):  sbatch -J base-full run_baseline.sbatch full full     (~4.1 h compute)
```

Results are compared with their published table (`scripts/g0/baseline_report.py`, pre-committed bands) and uploaded to the wandb project `anlp-project` after the run, from their own output files, never from inside their code.

---

## 4. External Models and Dataset Links

* **Base Model Checkpoint:** [`unsloth/gemma-2-2b`](https://huggingface.co/unsloth/gemma-2-2b) (bf16 ungated mirror, 5.2 GB; used by our own pipeline; not byte-compared with the official checkpoint). Duan's code uses the gated [`google/gemma-2-2b`](https://huggingface.co/google/gemma-2-2b) (fp32, 10.5 GB).
* **Gemma Scope JumpReLU SAE:** [`google/gemma-scope-2b-pt-res`](https://huggingface.co/google/gemma-scope-2b-pt-res) (Layer 12, `average_l0_82`, 16,384 latents; identical to Duan's `layer_12/width_16k/canonical`).
* **FLORES-200 Parallel Evaluation Corpus:** [Meta NLLB FLORES-200 Official Public Archive](https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz) (original FLORES-200, not FLORES+).
* **Reference implementation (not ours):** Duan (2026), arXiv:2606.03002, artifact `https://anonymous.4open.science/r/sae-feature-survival-quantization-0141/`.
* **Extension Models & SAEs (planned, never run):** [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) and [`seonglae/gemma-2-2b-l12-topk-16k`](https://huggingface.co/seonglae/gemma-2-2b-l12-topk-16k).
* **Experiment tracking:** wandb project `anlp-project` (entity `karma-skz-`).

---

## 5. Repository Layout

```
always-late/
├── HANDOFF.md               # START HERE for a new agent: mission, rules, runbook
├── PLAN.md  PROGRESS.md     # current plan and honest status
├── issues.md                # review findings (what is wrong or unsupported, with evidence)
├── paper/                   # ACL-format draft (contains superseded text; team must rewrite)
├── figures/  paper/figures/ # NOTE: some figures here are synthetic/stale (issues.md D5)
├── reports/
│   ├── progress_log.md      # dated evidence log
│   ├── duan_codebase_audit.md
│   ├── g0/                  # Duan baseline/diagnostic artifacts, benchmark, env freeze, runner diffs
│   └── *.json *.csv         # Phase-1 reports (20k-token run; see PROGRESS.md for which are current)
├── third_party/duan2026/    # THEIR code (pruned, pristine copies + the one working copy we run); see UPSTREAM.md
├── src/                     # our own pipeline (HF transformers; independent re-implementation)
├── scripts/
│   ├── g0/                  # baseline launcher, edit whitelist + verifier, comparison report, diagnostics, wandb, tests (see HANDOFF.md)
│   └── run_phase1_*.py ...  # our Phase-1 scripts
├── requirements.txt         # our own pinned stack (not the Ada env used for Duan's code)
└── writeoff.md              # earlier handoff brief (superseded by HANDOFF.md)
```

---

## 6. Reproducing our own Phase-1 numbers (provisional)

The exact commands that produced the reported 20k-token results are recorded in `reports/progress_log.md` (2026-10-02 entry). Caveats found in the review (`issues.md` D4): the default language list in `run_phase1_selectivity_pipeline.py` contains the `cmn_Hans` alias (28 entries, Mandarin twice), so **pass an explicit 27-code `--languages` list**; the survival numbers come from `run_phase1_survival_gpu.py` (all selective features; needs ~18 GB of GPU memory at 20k tokens), not the CPU script; the `.features.npz` it needs is git-ignored and must be regenerated by the selectivity step.

```bash
pip install -r requirements.txt
```

### Compiling the Paper
The `paper/` directory is self-contained: `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`, or upload it to Overleaf. Do not submit it before its results and discussion are rewritten to match `reports/` (see `issues.md` A1-A6).
