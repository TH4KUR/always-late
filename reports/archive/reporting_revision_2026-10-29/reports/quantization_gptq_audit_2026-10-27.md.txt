# GPTQ feasibility and fair-comparison audit

Date: 27 October 2026 (IST). Repository inspected at `3d5c6d7160907bfc6bf75459faf1cb836b64f7a8`.

**Verdict: replacing RTN with GPTQ as an additional experimental condition is feasible in the design and supported by source inspection and small mathematical checks. Real Gemma/GPU integration has not been verified. Replacing the INT16 numerical reference with low-bit GPTQ would confound the comparison.** No scientific experiments, cluster jobs, installations or deployments were started for this audit.

The user clarified that the aim is to explore another quantization method. The recommendation below is the auditor's proposal, not an approved implementation or frozen experiment protocol. The existing research HOLD remains in force for execution.

## 1. What changes, and what stays fixed

RTN and GPTQ are algorithms for choosing quantized weights. INT4 and INT6 describe bit widths. The existing INT16 label means very fine **RTN weight rounding evaluated in the existing floating-point execution path**; it is neither FP16 inference nor a distinct advanced quantization algorithm. Stored reference weights are bf16, while our intervention calculation and custom readout use FP32.

The useful comparison is reference / RTN-4 / GPTQ-4, optionally repeated at six bits. Keep the original INT16 rounding reference and add exact-weight/rebatched controls for the new execution path. INT16 includes a real weight perturbation; it is not a pure measurement-noise estimate.

Keep Gemma-2-2B, the exact reference checkpoint, TransformerLens version and loader, GemmaScope layer-12 canonical SAE, token IDs, BOS policy, feature groups, feature/context sample, decoder directions, FP-active edit positions, batching, masks, gradient target, arithmetic precision and readout unchanged. Keep both fixed-FP-amplitude and native-amplitude edits. Quantization calibration is a new, separate input; do not confuse it with the existing diagnostic model's calibration-feature split.

## 2. What the repository actually permits

| Component | Finding | Required action |
|---|---|---|
| `phase1_transfer_run.py:246`, `:287` | Feature/group cores accept a `setup(state)` callable; the crossed five-cell experiment does not assume how Q weights were obtained. | Reuse the numerical core. |
| `phase1_transfer_run.py:473` | Current setup restores FP and invokes RTN on every switch; a chunk switches between FP and Q repeatedly. | Load a previously calibrated immutable GPTQ artifact; **never recalibrate inside setup**. |
| `phase1_func_f5_run.py:272` | State parsing accepts FP, FP_rebatched and INT-prefixed bit widths. | New driver/state manifest with algorithm, bits and artifact identity. |
| `phase1_run.py:312`; `phase1_func_run.py:484,619`; `phase1_func_f5_run.py:386`; `phase1_transfer_precision_main.py:175` | Separate model-facing entry points each invoke RTN. | An all-experiment rerun needs adapters at these boundaries, not one global substitution. |
| `phase1_transfer_analysis.py:419`; `phase1_transfer_report.py:32` | Survival lookup and judged conditions encode RTN-era state names. | An additive comparison report must use method-aware states and GPTQ's own survival data. |
| `phase1_transfer_run.py:65,345,426` | Resume identity and frozen protocol checks depend on the old freeze and state string. | Bind the extension manifest, calibration identity and weight-artifact hashes into every cache/chunk identity. |

Do not edit the pristine Duan source, its extraction manifest, the original freeze or old result files. A new extension driver can validate the original core/inputs and a separate extension manifest, then call the existing core. Reusing its math does not mean reusing the old claim-status rules unchanged.

## 3. Two fairness mismatches verified during this audit

### Attention scale sharing

The legacy quantizer reduces **all dimensions except the last** when choosing scales. In TransformerLens, Q/K/V weights have shape `[heads, d_model, d_head]`. Consequently, the old quantizer uses one scale per head coordinate **shared across heads**. A standard linear Q/K/V projection treats every `(head, coordinate)` as a separate output channel.

For Gemma grouped-query attention, the actual stored K/V parameters are `_W_K` and `_W_V`; the legacy name filter includes them. The adapter must operate on these stored parameters, preserving the shared KV heads, rather than quantizing expanded property views.

The standard linear matrices are:

| TransformerLens parameter | Linear `[out, in]` layout |
|---|---|
| Q/K/V `[h,m,d]` | permute to `[h,d,m]`, reshape to `[h*d,m]` |
| O `[h,d,m]` | permute to `[m,h,d]`, reshape to `[m,h*d]` |
| MLP input/gate/output `[in,out]` | transpose |

O and the MLP matrices already have output-channel scale semantics compatible with those layouts. These mappings follow the TransformerLens 3.9.0 Gemma conversion code, obtained from its published wheel; the cluster's installed source still needs a hash check against the inspected source before implementation. [Published release](https://pypi.org/project/transformer-lens/3.9.0/), [official conversion documentation](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.pretrained.weight_conversions.gemma.html).

This scale sharing is a quantizer design choice, not evidence that the old within-RTN experiments were invalid. It does mean that an old-RTN versus standard-GPTQ difference combines algorithm and granularity changes.

### “Symmetric” does not guarantee the same grid

Our legacy scale is `max_abs / (2^(b-1)-1)`, with signed clamping. The original GPTQ repository's symmetric quantizer uses a different scale/zero-point convention. A CPU check at four bits gave scale `1/7` for legacy RTN versus `2/15` for the authors' quantizer on a row spanning [-1,1]. Identical `sym=True` labels therefore do not establish matching grids. [Authors' quantizer, pinned source](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/quant.py).

The old quantizer also performs its scale/rounding operations on bf16 tensors; the inspected GPTQ optimization uses FP32 before returning weights to the original dtype. Any claim specifically about the benefit of error compensation needs a matched RTN implementation with the same grid, granularity and quantization arithmetic.

## 4. Recommended comparison conditions

| Condition | Purpose |
|---|---|
| Original FP and legacy RTN-4 | Preserve the original comparison and connect to existing findings. |
| **Matched RTN-4** | Use the same canonical matrices, scale/zero-point routine, grouping, FP32 quantization arithmetic and final bf16 storage as GPTQ, without Hessian compensation. |
| **GPTQ-4, no input grouping** | Primary algorithm comparison against matched RTN. |
| GPTQ-4 group size 128, with corresponding matched RTN | Optional practical configuration, separately labelled. Do not compare only this against the old coarse-scale RTN and attribute all gains to GPTQ. |

Start with four bits because that is where our strongest existing distortion occurs. A six-bit replication can retain the earlier ladder; the low-level authors' algorithm can compute six-bit dense weight values even when a particular packed inference backend cannot represent them. A claim requiring both bit widths must await both.

For the primary comparison, propose fixed `blocksize=128`, damping 1% of mean Hessian diagonal, no activation ordering, no MSE clipping search, symmetric grid and no input grouping. These are transparent audit defaults, not a claim to the strongest possible accuracy setting. Record the exact sequential calibration policy. A more optimized configuration may be added after a separate predeclared calibration-only selection rule; intervention outcomes must not select the configuration.

One can alternatively give GPTQ the legacy shared-head signed grid to match that exact design. That is a valid controlled variant if labelled explicitly, but it adds custom quantizer behavior and should not be presented as an off-the-shelf GPTQ deployment result. The standard-grid matched pair above is preferable as the main extension.

## 5. Integration options and recommendation

**Recommended: calibrate against the existing TransformerLens model, using a thin matrix adapter around a pinned GPTQ core, then save dense dequantized bf16 target weights.** This keeps the inference engine and residual coordinates identical to the old study. The authors' implementation already returns quantized values to the layer's original dtype; packing is a separate concern. [Pinned GPTQ core](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py).

Temporary linear proxies during calibration can expose the canonical `[out,in]` matrices. They must receive the actual inputs to each projection and be removed before evaluation. The model's forward modules are not replaced by quantized kernels. Freeze target names/shapes and copy the saved values into the original model parameters for each state switch. Preserve non-target tensors byte-for-byte; verify dtype, shape, stride and parameter identity.

**Alternative: calibrate a Hugging Face Gemma model with GPTQModel, export dense dequantized projection weights, and map only those weights into our original TransformerLens instance.** Its source explicitly handles Gemma2. This benefits from existing calibration infrastructure but introduces another model implementation into artifact creation. It needs exact projection round trips plus numerical equivalence checks; replacing the whole checkpoint risks changing Gemma embedding scaling or RMSNorm conventions. [Pinned Gemma2 support](https://github.com/ModelCloud/GPTQModel/blob/ebfcbb2bdf1297b26754bc6b53ef14c73e815753/gptqmodel/models/definitions/gemma2.py).

Loading a packed GPTQ model as the main experiment backend is the least controlled initial route. It changes execution kernels and instrumentation as well as weight selection. It is appropriate for a later deployed-kernel validation, which is a stronger but separate scope.

## 6. Calibration requirements

GPTQ chooses weights to reduce layerwise output reconstruction error on calibration inputs. Its objective gives a reason to expect ordinary computation to improve, but does not guarantee preserved SAE intervention effects. [Original paper, objective and algorithm](https://arxiv.org/html/2210.17323v2).

Use calibration text independent of FLORES dev/devtest and the external evaluation articles. Exact and normalized deduplication should cover the existing evaluation pools, and the external panel requires article-ID exclusion. Freeze the dataset revision, article/sample IDs, token IDs, seed, language token counts and sequence boundaries. Use one fixed multilingual calibration pool across configurations; do not pick a pool using intervention results.

Proposed starting budget: 8,192 content tokens per language across the existing 27-language roster (221,184 total), with the current 128-token limit and a separate real BOS per sequence. This is a proposal pending corpus-coverage and memory checks, not an available dataset or an asserted optimum. If a language cannot meet the budget, report and settle that before freezing; do not silently fill its allocation with English. Exclude padding from Hessian statistics. Include real BOS consistently in calibration forward passes; keep BOS excluded from evaluation statistics and edits as before.

Calibration is sequential through the **whole** model: Q/K/V together, then O, input/gate together, then MLP output, followed by the next transformer block. Recompute inputs after preceding quantized stages; collect actual normalized projection inputs and actual post-attention/post-gating activations. Do not substitute the raw residual or SAE reconstruction for these inputs. This follows the stage dependencies of the authors' true-sequential workflow while using Gemma's own block computation. [Pinned sequential workflow](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py).

Construct one immutable fully quantized model artifact per configuration. The crossed experiment uses prefix and suffix weights from that **same artifact**. Recalibrating its halves separately changes the treatment being studied.

Dead calibration dimensions, damping failures and nonfinite values must be reported. The upstream algorithm can zero columns unobserved in calibration; a matched diagnostic that applies the same dead-column handling, or a report of its contribution, is needed before attributing changes narrowly to error compensation.

## 7. Numerical controls and paired analysis

Retain the old RTN16 reference for continuity, labelled as a small weight perturbation. Add exact-weight save/load and repeated-batching controls under the extension's actual batching and dtype path. Run split/full, sham and gradient checks in **every new quantized state**, not only FP. Existing custom-readout gates admit bf16 disagreement; they are not sufficiently strict to validate a mere parameter-layout conversion. Conversion must first pass exact tensor round trips and algebraic projection checks.

Use a common reference-derived eligibility rule and the exact same feature/context/direction keys across methods. Do not allow each quantizer to select an easier eligible population. If FP values or numerical controls differ materially from the historical run, show that bridge difference and report both historical-floor and new-control sensitivities; do not merge unmatched runs silently.

Compare paired absolute and normalized transfer errors, cosine, fixed/native amplitude effects, signed NLL changes, behavioral endpoints, and C/W/I contrasts. Keep their existing sentence/feature dependence in resampling. Quantization calibration samples must remain separate from diagnostic training/held-out feature splits. Refit diagnostic models only on their declared training partitions; distinguish within-method refits from a model locked on RTN and transferred to GPTQ.

Report ordinary model degradation next to intervention distortion. Use absolute rescue/NLL effects before rescue percentages: if GPTQ shrinks the original degradation denominator toward zero, a rescue fraction can become unstable. Avoid ratios that make an almost-preserved intervention look badly distorted merely because its reference effect is tiny. Keep all-item results and coverage alongside eligible-item summaries. Existing null findings do not become confirmatory positives merely because a new method changes their sign.

## 8. Gates before the full rerun

1. **Source/environment:** pin the checkpoint revision, exact stored reference values, SAE, package/source hashes, GPTQ core, manifest and calibration corpus. Match the actual cluster package files, not only version strings. Preserve the current environment rather than upgrading it for GPTQModel.
2. **Adapter:** exact matrix round trips; projection agreement; correct stored GQA tensors and unexpanded KV heads; all intended target weights present and all other weights unchanged.
3. **Algorithm:** reproduce the pinned authors' core on tiny inputs; equal-grid RTN/GPTQ agreement when the Hessian is diagonal; finite deterministic outputs and dtype restoration. Freeze dead-column policy.
4. **Real-model calibration:** bounded correctness instrument on the actual GPU, including the largest MLP output projection; validate masks, staging, damping and peak memory. This is an integration gate, not a search for a publishable effect.
5. **State identity:** save/load is exact, setup never recalibrates, repeated switches restore exact FP and Q target values, gradients remain available through the unchanged floating-point suffix.
6. **Measurement:** split/full agreement for FP and each Q state; exact zero-amplitude sham in the actual batches; fixed tokens/positions; FP bridge outputs and effect measurements agree with the historical setup within predeclared numerical limits.
7. **Identity and completeness:** method-aware output names; artifact/manifest/calibration hashes in resume keys; no RTN survival file joined to GPTQ by matching bit width; complete identical key sets across conditions.

The largest Hessian scales quadratically with projection input width, not with total model parameters. For an input width of 9,216, one FP32 Hessian is 324 MiB; factorization needs additional storage. Calibration must process stages with bounded buffers and can offload inactive weights/caches. This calculation is **not proof that the full implementation fits an 11 GB GPU**. No measured runtime or complete two-day rerun estimate is available yet.

## 9. Scientific decision the extension would resolve

The primary comparison is a quantizer-by-intervention preservation study at matched bit width and measurement arithmetic. Three outcomes are meaningful:

- Ordinary quality and intervention effects both improve: the old RTN finding is substantially method-dependent. Report it directly.
- Ordinary quality improves but a practically meaningful intervention discrepancy remains above the new numerical controls: stronger evidence of a gap between ordinary accuracy preservation and intervention preservation in this system.
- Activation survival and intervention transfer respond differently to quantizer choice: evidence to test whether activation survival is an adequate proxy, using held-out comparisons against existing sensitivity baselines.

GPTQ is designed around unedited calibration inputs; our interventions deliberately evaluate changed hidden states. That motivates the hypothesis. It does not establish off-distribution failure, novelty, a selective language effect, safety relevance or publication readiness in advance. A limited fixed/native intervention contrast remains a valid causal contrast in the tested model; broader claims require additional evidence.

## 10. What was actually verified

The audit inspected the repository, the published TransformerLens 3.9.0 wheel and pinned upstream source. Local environment: PyTorch 2.14.1+cpu; no TransformerLens, Transformers, SAE Lens or GPTQModel installed; no CUDA device. Model access was not attempted with credentials, and the public Gemma checkpoint config URL returned HTTP 401.

[Evidence JSON](quantization_gptq_audit_2026-10-27/evidence.json) records source hashes and the CPU checks. [Reproduction script](quantization_gptq_audit_2026-10-27/check_compatibility.py) downloads only the pinned small source files into a temporary directory and executes extracted class/function bodies; it installs nothing and loads no model.

- Q/K/V and O layout round trips were exact on synthetic tensors; projection discrepancies were 0 and 1.78e-15 in float64.
- A four-head example confirmed 3 legacy Q/K/V scales versus 12 canonical output-channel scales.
- The symmetric-grid example confirmed unequal scales and unequal quantized values.
- The authors' unmodified GPTQ class body matched same-grid nearest rounding exactly for an identity Hessian at four and six bits in FP32 and bf16. On CPU, only the unused Transformers type check and CUDA timing synchronization were stubbed; optimization arithmetic was unchanged.

These checks establish adapter algebra and low-level arithmetic compatibility. They do **not** validate Gemma calibration, actual hook captures, real GPU fit, packed-kernel equivalence, stronger accuracy, or preserved intervention effects. Implementation and a real-model correctness gate remain necessary before scheduling the rerun.

## 11. Recommended implementation order

Prepare a new method/artifact manifest and immutable weight-state loader; implement the canonical matrix/calibration adapter and matched RTN; add an extension driver for the existing intervention-transfer core; add paired method-aware analysis; then adapt the remaining survival, association, functional and behavioral entry points. Reuse original FP selections and data freezes. Leave historical reproduction/debugging runs as historical evidence.

Only after real-model gates should the full experiment family run. Prioritize ordinary NLL plus survival and the central fixed/native transfer matrix first, then the existing functional/behavioral panels and external corpus. This ordering provides a validated shared artifact to every experiment; it does not authorize selecting or dropping endpoints based on their outcome.

**Execution recommendation:** add GPTQ and a matched RTN comparator; retain legacy RTN as the bridge and INT16 as a labelled reference. Do not replace INT16 with GPTQ, rewrite old results, or claim the integration has already passed on Gemma.
