"""Generate runnable, auditable derivatives of Duan (2026) scripts 03 and 06 (v3).

Every edit is a literal substitution ASSERTED to match exactly the stated number of times, so the originals are
never silently altered. Edit kinds:  [FIX] make the notebook export runnable   [PLACE] numerics-preserving
two-GPU placement   [ARM] opt-in experimental arms (env-controlled)   [META] logging / safety / resume.
usage: python make_duan_runners_v3.py <orig_03.py> <orig_06.py> <out_03.py> <out_06.py>
"""
import sys

def make(src, edits):
    for old, new, count, tag in edits:
        n = src.count(old)
        assert n == count, f"anchor {tag!r} matched {n}x (expected {count}): {old[:80]!r}"
        src = src.replace(old, new)
    return src

# ---------------------------------------------------------------- script 03
s3 = open(sys.argv[1], encoding="utf-8").read()
assert s3.count("# %% Cell 22") == 1
s3 = s3.split("# %% Cell 22")[0]            # [FIX] cells 22+ call 3 functions that are never defined in the file
E3 = []
E3.append(("display(", "print(", 4, "display"))                                        # [FIX]
E3.append(('DEVICE = "cuda" if torch.cuda.is_available() else "cpu"\nprint("Device:", DEVICE)',
 'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"\n'
 'WORK_DEVICE = os.environ.get("G0_WORK_DEVICE", DEVICE)  # [PLACE] keep as a STRING such as "cuda:1" (a torch.device would drop the index inside TransformerLens)\n'
 'print("Device:", DEVICE, "| work-model device:", WORK_DEVICE)\n'
 'def _g0_assert_model(m, dev, tag):  # [META] nothing in the shipped script asserts dtype/device\n'
 '    if not str(dev).startswith("cuda"):\n        return\n'
 '    p = next(m.parameters()); d = torch.device(dev)\n'
 '    assert p.dtype == torch.bfloat16, (tag, p.dtype)\n'
 '    assert p.device.type == d.type and (d.index is None or p.device.index == d.index), (tag, p.device, dev)\n', 1, "WORK_DEVICE"))
E3.append(('dtype=torch.bfloat16 if device == "cuda" else torch.float32,',
 'dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,  # [PLACE] TRAP: shipped test `device == "cuda"` gives float32 for "cuda:1"', 1, "dtype trap"))
E3.append(("model_work, _ = load_gemma_tl_model([MODEL_NAME_USED])",
 "model_work, _ = load_gemma_tl_model([MODEL_NAME_USED], device=WORK_DEVICE)  # [PLACE]", 1, "load work"))
E3.append(('print("Reference and work models loaded.")',
 '_g0_assert_model(model_ref, DEVICE, "ref"); _g0_assert_model(model_work, WORK_DEVICE, "work")  # [META]\n'
 'print("Reference and work models loaded.")', 1, "assert after load"))
E3.append(("    model_work.to(DEVICE)\n    model_work.eval()\n",
 "    model_work.to(WORK_DEVICE)  # [PLACE]\n    model_work.eval()\n    _g0_assert_model(model_work, WORK_DEVICE, 'restore')  # [META]\n", 1, "restore"))
E3.append(('RUN_MODE = "3b_full"',
 'RUN_MODE = os.environ.get("G0_RUN_MODE", "3a_test")  # [ARM] shipped values 3a_test | 3b_full ; added g0_core. NOTE: default differs from shipped (3b_full)\n'
 'G0_BOS = os.environ.get("G0_BOS", "none")  # [ARM] none (shipped) | prepend_keep | prepend_mask\n'
 'SNAP_EVERY = int(os.environ.get("G0_SNAP_EVERY", "100"))  # [META] cumulative-sum snapshot every N windows (block bootstrap, no extra GPU cost)', 1, "RUN_MODE"))
E3.append(('else:\n    raise ValueError("RUN_MODE must be \'3a_test\' or \'3b_full\'.")',
 'elif RUN_MODE == "g0_core":  # [ARM] our G0 configuration: no pruning; INT0 = restore-only identity control\n'
 '    TOKEN_BUDGET = int(os.environ.get("G0_TOKENS", "500000"))\n'
 '    BITS_TO_TEST = [int(b) for b in os.environ.get("G0_BITS", "0,16,12,8,7,6").split(",")]\n'
 '    BATCH_SIZE = 1\n    INCLUDE_INT4 = False\n    RUN_PRUNING = False\n'
 'else:\n    raise ValueError("RUN_MODE must be \'3a_test\', \'3b_full\' or \'g0_core\'.")', 1, "modes"))
E3.append(('OUTPUT_DIR = Path(f"phase3_gemma_outputs_{RUN_MODE}_{TOKEN_BUDGET//1000}k_L{LAYER}")\nOUTPUT_DIR.mkdir(parents=True, exist_ok=True)',
 'assert G0_BOS in ("none", "prepend_keep", "prepend_mask"), G0_BOS\n'
 'OUTPUT_DIR = Path(os.environ.get("G0_OUT", f"phase3_gemma_outputs_{RUN_MODE}_{TOKEN_BUDGET//1000}k_L{LAYER}_bos-{G0_BOS}"))\n'
 'OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n'
 'import json as _json, platform as _pl, importlib.metadata as _ilm  # [META]\n'
 '_meta = dict(run_mode=RUN_MODE, bos=G0_BOS, token_budget=TOKEN_BUDGET, bits=BITS_TO_TEST, pruning=RUN_PRUNING,\n'
 '             torch=torch.__version__, transformer_lens=_ilm.version("transformer-lens"), sae_lens=_ilm.version("sae-lens"),\n'
 '             transformers=_ilm.version("transformers"), python=_pl.python_version(),\n'
 '             gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],\n'
 '             work_device=WORK_DEVICE, model_candidates=MODEL_CANDIDATES, sae=[SAE_RELEASE, SAE_ID], snap_every=SNAP_EVERY)\n'
 '(OUTPUT_DIR / "run_meta.json").write_text(_json.dumps(_meta, indent=2))\nprint(_meta)\n'
 'import sys, time  # [META] wandb logging via the helper next to this runner (never fatal; G0_WANDB=0 disables)\n'
 'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n'
 'from g0_wandb import G0Logger, gpu_mem_gib  # gpu_mem_gib never raises (it runs outside any logging wrapper)\n'
 '_WB = G0Logger.start(project=os.environ.get("WANDB_PROJECT", "anlp-project"), entity=os.environ.get("WANDB_ENTITY") or None,\n'
 '                     name=os.environ.get("G0_WANDB_NAME", f"duan03-{RUN_MODE}-bos-{G0_BOS}"), config=_meta, out_dir=OUTPUT_DIR,\n'
 '                     tags=["g0", "duan03", f"bos-{G0_BOS}", RUN_MODE])', 1, "output/meta"))
# [ARM] BOS arm: prepend BOS to the IDENTICAL 256-token windows used by the shipped arm (rows are 257 long), so arms differ only by BOS
E3.append(("    tokens_2d = tokens[:usable_tokens].reshape(n_seqs, seq_len).to(DEVICE)\n",
 "    base_windows = tokens[:usable_tokens].reshape(n_seqs, seq_len)  # identical text windows in every arm\n"
 "    if G0_BOS == 'none':\n"
 "        tokens_2d = base_windows.to(DEVICE)  # shipped behaviour\n"
 "    else:  # [ARM]\n"
 "        bos_id = tokenizer.bos_token_id if getattr(tokenizer, 'bos_token_id', None) is not None else 2\n"
 "        tokens_2d = torch.cat([torch.full((n_seqs, 1), bos_id, dtype=torch.long), base_windows], dim=1).to(DEVICE)\n"
 "        print('BOS arm:', G0_BOS, '| row length', tokens_2d.shape[1], '| first row head:', tokens_2d[0, :4].tolist())\n", 1, "BOS windows"))
E3.append(("    acts = cache[hook_name].detach()\n    acts = acts.reshape(-1, acts.shape[-1])\n",
 "    acts = cache[hook_name].detach()\n"
 "    if G0_BOS == 'prepend_mask':  # [ARM] drop exactly the BOS position (index 0) from the feature statistics\n"
 "        acts = acts[:, 1:, :]\n"
 "    acts = acts.reshape(-1, acts.shape[-1])\n", 1, "mask BOS"))
E3.append(("tokens_2d = build_tokens_2d(model_ref.tokenizer, TOKEN_BUDGET, SEQ_LEN)\n",
 "tokens_2d = build_tokens_2d(model_ref.tokenizer, TOKEN_BUDGET, SEQ_LEN)\n"
 "torch.save(tokens_2d.cpu(), OUTPUT_DIR / 'tokens_2d.pt')  # [META] exact tokens used, for the cross-check with our own pipeline\n", 1, "save tokens"))
# [META] block snapshots inside the streaming loop
E3.append(('    total_positions = 0\n\n    for i in tqdm(range(0, tokens_2d.shape[0], batch_size), desc=f"Streaming {condition_name}"):',
 '    total_positions = 0\n    _snaps, _snap_pos = [], []  # [META]\n    _t0 = time.time()\n\n    for i in tqdm(range(0, tokens_2d.shape[0], batch_size), desc=f"Streaming {condition_name}"):', 1, "snap init"))
E3.append(("        total_positions += x.shape[0]\n\n        del acts_a",
 "        total_positions += x.shape[0]\n"
 "        if SNAP_EVERY and ((i // batch_size) + 1) % SNAP_EVERY == 0:  # [META]\n"
 "            _snaps.append(torch.stack([sum_x, sum_y, sum_x2, sum_y2, sum_xy, fire_count]).clone()); _snap_pos.append(total_positions)\n"
 "            _WB.heartbeat(condition_name, (i // batch_size) + 1, -(-tokens_2d.shape[0] // batch_size), time.time() - _t0, gpu_mem_gib(torch))\n\n"
 "        del acts_a", 1, "snap step"))
E3.append(('    all_summaries.append(summary)\n    print("\\nCurrent summary:")',
 '    all_summaries.append(summary)\n    _WB.log_condition(summary)  # [META] wandb\n    print("\\nCurrent summary:")', 1, "add_summary"))
E3.append(("        free_memory()\n\n    n = total_positions\n",
 "        free_memory()\n\n"
 "    if SNAP_EVERY:  # [META] final cumulative snapshot + save; block k = snap[k] - snap[k-1]\n"
 "        _snaps.append(torch.stack([sum_x, sum_y, sum_x2, sum_y2, sum_xy, fire_count]).clone()); _snap_pos.append(total_positions)\n"
 "        torch.save({'snaps': torch.stack(_snaps), 'n_positions': _snap_pos, 'snap_every_batches': SNAP_EVERY, 'batch_size': batch_size},\n"
 "                   OUTPUT_DIR / ('snap_' + condition_name.replace(' ', '_') + '.pt'))\n"
 "    n = total_positions\n", 1, "snap final"))
# [ARM]/[META] INT0 = restore-only control; weight perturbation log; resume
E3.append(('    slug = f"RTN_INT{bits}"\n',
 '    slug = f"RTN_INT{bits}"\n'
 '    if os.environ.get("G0_RESUME", "1") == "1" and (OUTPUT_DIR / f"{slug}_summary.csv").exists():  # [META] resume after a crash\n'
 '        all_summaries.append(pd.read_csv(OUTPUT_DIR / f"{slug}_summary.csv").iloc[0].to_dict())\n'
 '        per_feature_files[slug] = str(OUTPUT_DIR / f"{slug}_per_feature.csv")\n'
 '        print("RESUME: skipping", slug)\n        continue\n', 1, "resume"))
E3.append(("    n_tensors, n_params = quantize_rtn_per_output_channel_tl(model_work, bits=bits)\n    print(f\"Quantized tensors={n_tensors}, params={n_params:,}\")\n",
 "    n_tensors, n_params = quantize_rtn_per_output_channel_tl(model_work, bits=bits) if bits > 0 else (0, 0)  # [ARM] INT0 = restore only (must give r == 1 exactly)\n"
 "    print(f\"Quantized tensors={n_tensors}, params={n_params:,}\")\n"
 "    _num = 0.0; _den = 0.0  # [META] relative L2 weight change vs original (x-axis for the scaling / noise-floor check)\n"
 "    for _n, _p in model_work.named_parameters():\n"
 "        if target_tl_weight(_n) and _n in original_state_cpu:\n"
 "            _o = original_state_cpu[_n].to(_p.device).float()\n"
 "            _num += float(((_p.data.float() - _o) ** 2).sum()); _den += float((_o ** 2).sum())\n"
 "    _rel = (_num / max(_den, 1e-30)) ** 0.5\n"
 "    print(f'INT{bits}: relative L2 weight change = {_rel:.3e}')\n"
 "    (OUTPUT_DIR / f'weight_perturbation_INT{bits}.json').write_text(_json.dumps({'bits': bits, 'rel_l2': _rel}))\n", 1, "quantize + perturbation"))
s3 = make(s3, E3)
s3 += ("\n# [META] wandb: final table, gate metrics (closure protocol), artifact\n"
       "_df = pd.DataFrame(all_summaries)\n_WB.log_table(_df)\n_WB.log_gates(_df, OUTPUT_DIR)\n_WB.finish(OUTPUT_DIR)\n")
hdr = ('# DERIVED FILE (v3): generated by make_duan_runners_v3.py from Duan (2026) scripts/03_gemma_phase3_streaming_sweep.py.\n'
       '# [FIX] runnable export  [PLACE] two-GPU placement  [ARM] opt-in arms (BOS, INT0, g0_core)  [META] asserts/logging/resume/snapshots.\n'
       '# Defaults differ from shipped in ONE way only: G0_RUN_MODE defaults to the shipped smoke mode 3a_test instead of 3b_full.\n')
open(sys.argv[3], "w", encoding="utf-8").write(hdr + s3)

# ---------------------------------------------------------------- script 06
s6 = open(sys.argv[2], encoding="utf-8").read()
assert s6.count("# %% Cell 12") == 1
s6 = s6.split("# %% Cell 12")[0]   # [FIX] drop the Colab zip/download cell (contains the anonymization artefact that makes the file unparsable)
E6 = [
 ("display(", "print(", 2, "display06"),   # [FIX] notebook-only; the shipped order also runs display() BEFORE to_csv, so a NameError would lose the whole run
 ('RUN_MODE = "full"  # "smoke_test" or "full"', 'RUN_MODE = os.environ.get("G0_RUN_MODE06", "smoke_test")  # [ARM] shipped value: full', 1, "mode06"),
 ("BITS_TO_TEST = [None, 8, 7, 6]", 'BITS_TO_TEST = [None if b == "none" else int(b) for b in os.environ.get("G0_BITS06", "none,8,7,6").split(",")]  # [ARM]', 1, "bits06"),
 ('OUT_DIR = Path("/content/gemma_sliding_ppl_outputs")', 'OUT_DIR = Path(os.environ.get("G0_OUT06", "gemma_sliding_ppl_outputs"))  # [FIX] was /content (Colab)', 1, "out06"),
 ("        torch_dtype=DTYPE,\n        device_map=\"auto\",\n        low_cpu_mem_usage=True,\n    )",
  "        dtype=DTYPE,  # [FIX] torch_dtype is deprecated in the installed transformers\n        device_map=\"auto\",\n        low_cpu_mem_usage=True,\n"
  "        **({\"attn_implementation\": os.environ[\"G0_ATTN\"]} if os.environ.get(\"G0_ATTN\") else {}),  # [ARM] eager vs default attention\n    )\n"
  "    print('attention implementation in use:', getattr(model.config, '_attn_implementation', '?'))  # [META]", 1, "attn06"),
]
E6.append(("rows = []\n\nfor bits in BITS_TO_TEST:\n    row = run_one_condition(bits, input_ids_all)\n    rows.append(row)\n",
 "import sys, time\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # [META] wandb helper next to this runner\nfrom g0_wandb import G0Logger\n"
 "_WB = G0Logger.start(project=os.environ.get('WANDB_PROJECT', 'anlp-project'), entity=os.environ.get('WANDB_ENTITY') or None,\n"
 "                     name=os.environ.get('G0_WANDB_NAME', f'duan06-{RUN_MODE}-attn-' + os.environ.get('G0_ATTN', 'default')),\n"
 "                     config=dict(run_mode=RUN_MODE, window=WINDOW_SIZE, stride=STRIDE, bits=[str(b) for b in BITS_TO_TEST],\n"
 "                                 attn=os.environ.get('G0_ATTN', 'default')), out_dir=OUT_DIR, tags=['g0', 'duan06'])\n"
 "rows = []\n\nfor bits in BITS_TO_TEST:\n    row = run_one_condition(bits, input_ids_all)\n    rows.append(row)\n    _WB.log_condition(row)  # [META] wandb\n", 1, "wandb06"))
s6 = make(s6, E6)
s6 += "\n# [META] wandb: final table and artifact\n_WB.log_table(sliding_df)\n_WB.finish(OUT_DIR)\n"
s6 = "# DERIVED FILE (v3) from Duan (2026) scripts/06_gemma_sliding_window_ppl_check.py: [FIX] parse error/Colab paths, [ARM] env-controlled mode/bits/attention.\n" + s6
open(sys.argv[4], "w", encoding="utf-8").write(s6)
print("ok")
