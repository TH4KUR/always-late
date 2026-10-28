# Gemma-2-9B confirmation: start here (for the person running it)

Written 2026-10-28 by the author of this branch (`karma-skz`). Read this page from top to bottom before you type anything. It is the whole job in plain steps. The technical version, written for a Claude Code session on the machine, is [`CLAUDE.md`](../../../CLAUDE.md) in the repository root; the design is in [`docs/superpowers/specs/2026-10-28-gemma9b-three-arm-confirmation-design.md`](../../../docs/superpowers/specs/2026-10-28-gemma9b-three-arm-confirmation-design.md).

## What you are doing

The project ("Always Late") asks whether compressing a language model (quantization) damages the SAE features that are specific to one language. The main study is on Gemma-2-2B. **This job repeats it on Gemma-2-9B** (decision D-089) to see whether the result holds on a second, larger model.

- **Three ways of quantizing the weights ("arms"):** `legacy` (the project's original rounding, exactly as frozen), `rtn-matched` (the standard way of rounding) and `gptq` (GPTQ, calibrated on Wikipedia text in the 27 languages).
- **Two stages:** tier 1 = Study G (switch a language's selective features off or scale them, measure the harm to that language's next-token loss); tier 2 = Phase 1 (how many of those features survive quantization, and the loss).
- **Same data and seeds as the 2B runs:** the same FLORES-200 sentences, languages, ratios, seeds and calibration pool. Only the feature groups differ, because the 9B SAE has different features.
- **The predictions were written down before any 9B number existed** (`freeze.json`, `amendment_A1_three_arm.json`). Whatever comes out, including "not consistent with the prediction", is a valid result. **Do not tune anything to get a nicer result.**

Everything is built and tested on a CPU machine. **Nothing has ever run on a real GPU with the real 9B model.** The first real execution is the first run below, so expect that a step may fail for a reason nobody could see yet. That is why you stop and report instead of working around it.

## What you need before you start

1. **A Linux machine with an NVIDIA GPU.** The job was planned for one RTX PRO 6000 Blackwell (96 GB GPU memory), 160 GB RAM and 250 GB disk (Jarvis Labs, about 210 INR per hour). On a Blackwell card the NVIDIA driver must be **580 or newer** (`nvidia-smi` shows it; the script stops with a message if it is older). The disk must have **about 80 GB free** (the model alone is 37 GB). RAM: a tier-2 unit needs about 98 GB of host RAM and runs alone; these are estimates, nobody has measured them yet. The script refuses a unit, and prints the numbers, if the RAM is too small.
2. **A Hugging Face token of your own** (`hf_...`) from an account that has accepted the licences of **both** `google/gemma-2-9b` and `google/gemma-2-2b` (open each model page while logged in and accept). The 2B tokenizer is compared with the 9B one. A fine-grained token also needs "read access to public gated repos". **Never paste the token into a file, the chat, a commit or the log.**
3. **The calibration pool file `gptq_calibration_pool.tar.gz`** (0.6 MB). It is deliberately **not in git** (the project keeps its pool files out of the repository); get it from the author. The author's copy has this sha256: `5f30e9ba98dee29de7b1af730a8e320df834373738bac3d9b90cdced7624ca64`.
4. `git`, and `python3` with `venv` (on Ubuntu: `sudo apt-get install -y python3-venv`).
5. **The author's OK for the cost** and for the long run (step 7). The machine is billed by the hour while it is on: pause it when you are only waiting for an answer.

## Step by step

All commands are typed on the GPU machine, in its home directory (`cd ~`).

**1. Look at the machine.**
```bash
nvidia-smi        # GPU name, memory, driver version
free -g           # RAM
df -h ~           # free disk
```

**2. Get the code** (the branch is `gemma9b-confirmation-v2`; the folder must be called `always-late`).
```bash
git clone --branch gemma9b-confirmation-v2 https://github.com/TH4KUR/always-late.git always-late
```
If the repository asks for a login, it is private: ask the author for access instead of using someone else's credentials.

**3. Put the calibration pool next to the code.** Copy `gptq_calibration_pool.tar.gz` into your home directory (the folder that contains `always-late/`), then:
```bash
sha256sum gptq_calibration_pool.tar.gz     # must equal the value in "What you need", item 3
tar xzf gptq_calibration_pool.tar.gz       # creates ~/gptq/calibration
```
(The run itself checks the files against the frozen pool and stops if they differ.)

**4. Give the token to the shell, without showing it.**
```bash
read -rs HF_TOKEN; export HF_TOKEN         # paste the token, press Enter: nothing is displayed
```
Do this again in every new shell before you run the script.

**5. Dry run: prints the plan, installs and downloads nothing.**
```bash
cd ~/always-late
DRY_RUN=1 bash scripts/g0/jarvis_run_all.sh
```
Read it. It lists the steps and the units (tier 1 and tier 2, three arms). The first thing it does is check that the code equals the frozen code; if that fails, **stop and tell the author** (the code on your machine is not the tested code).

**6. First real run: the safe one.** It stops by itself after the smoke tests and the build-and-gate step.
```bash
STOP_AFTER=smoke_gptq bash scripts/g0/jarvis_run_all.sh
tail -f ~/run_all.log                      # follow it; Ctrl-C only stops the tail, not the job
```
It installs the Python environment (the right PyTorch build for the GPU), downloads FLORES and the **37 GB** model, runs the unit tests, compares the 2B and 9B tokenizers, checks the pool, runs the smoke tests on 3 languages, builds the quantized weight sets once to check them (the "gate"), and runs a small GPTQ test. It is **resumable**: if it stops, run the same command again and finished steps are skipped. The job detaches itself, so a closed terminal does not kill it. Do not start a second copy (`ps aux | grep jarvis_run_all`).

**7. Report to the author, then wait for the OK.** Send:
- whether it finished (`STOP_AFTER=smoke_gptq reached` at the end of `~/run_all.log`) or the last 40 lines of the log and the step that failed (the log says `FAILED ... during: <step>`);
- the seconds per quantized weight-set build (in `~/logs/g9-gate*.log` and `~/logs/g9-smoke-gptq.log`);
- the peak GPU memory (`peak GPU GB` in the logs), the host RAM seen (`free -g` while it ran) and the wall time.

**Do not start the long run before the author says OK.** It builds about 16 weight sets and takes many hours at the hourly price.

**8. The long run (only after the OK).**
```bash
cd ~/always-late
bash scripts/g0/jarvis_run_all.sh          # the same command, without STOP_AFTER
```
Order of the units, the markers it leaves and how to retry a failed unit are in `CLAUDE.md` ("Monitoring and troubleshooting"). A unit counts only if it is validated (exit code 0 alone is not trusted). A failed unit is reported as failed, never hidden.

**9. Hand back.** When it finishes, send the author `~/gemma9b_results.tar.gz` (outputs, logs, checksums, the frozen files used, the recorded hashes) and `~/run_all.log`. The author does the analysis.

## Stop and ask the author (do not work around)

A tokenizer mismatch, a pool mismatch, a failed gate, an artifact hash mismatch (in the gate or in a unit), a code-hash failure at the start, a driver or GPU "compute capability" error, a RAM refusal, an out-of-memory error. These mean the instrument is not what was tested. If something has to change to get through (for example a smaller batch size), it is a **dated amendment** written into `reports/functional_validation/gemma9b/amendments.md` before the re-run, and the author decides.

Likely first failures, in order: a token that cannot read a gated model; a driver older than 580; a missing GPU kernel in the PyTorch build; too little host RAM; the frozen group builder refusing the `all` family on the 9B features (the script then falls back to K16/K64 and writes `MANIFEST_FALLBACK.txt`); the SAE id missing in the installed `sae-lens`.

## Never do

- Edit any code, the freeze, the amendment, the ratios, languages, sentence counts, bit widths, seeds, thresholds or the pool. No post hoc language selection.
- Skip the smoke tests (`SKIP_SMOKE=1` exists only for resuming after they passed).
- Push to `main`, merge, or change anything on Hugging Face or wandb. Commit no weights, tokens or pool files. If you commit, commit only new files under `reports/functional_validation/gemma9b/` and say so.
- Print, log or paste the Hugging Face token.

## If money or time runs out

Drop in this order: legacy INT7, INT5, INT8; then the tier-2 matched-RTN arm. **Never** drop GPTQ, legacy FP, INT6 and INT4, the tagging, or the tier-2 legacy and GPTQ arms. Say plainly in the report what was not run.

## Where things are

- **Technical hand-over (for a Claude session, but readable):** [`CLAUDE.md`](../../../CLAUDE.md), with every environment variable (`ARMS`, `STAGES`, `STOP_AFTER`, ...), the order of the units and the analysis command.
- **Design and plan:** `docs/superpowers/specs/2026-10-28-gemma9b-three-arm-confirmation-design.md` and `docs/superpowers/plans/2026-10-28-gemma9b-three-arm-confirmation.md`.
- **Frozen predictions:** `reports/functional_validation/gemma9b/freeze.json` (legacy arm) and `amendment_A1_three_arm.json` (the new predictions).
- **Handoff note with the verified and unverified list:** `reports/handoffs/9b-001-2026-10-28-gemma9b-three-arm-ready.md`.
- **Who to ask:** the author of this branch (`karma-skz`).
