"""Live self-test of g0_wandb's FIXED SCHEMA against the real W&B service: logs a tiny synthetic run, reads it back through the API,
checks that summary and config keys are inside the schema, that history and artifact arrived and snapshots were excluded, then deletes
the run. Usage: python selftest_wandb.py [project]   (credentials come from the wandb library; never print them)"""
import json, os, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wandb
import g0_wandb as g
from g0_wandb import G0Logger
from wandb_rebuild import check_run

project = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WANDB_PROJECT", "anlp-project")
name = f"selftest/schema-{int(time.time())}"
REL = {16: 1.7e-3, 12: 2.3e-3, 8: 2.7e-2, 7: 5.4e-2, 6: 1.1e-1}
with tempfile.TemporaryDirectory() as out:
    out = Path(out)
    lg = G0Logger.start(project=project, name=name, config={"experiment": "selftest", "transformer_lens": "x", "bos": "prepend_mask", "token_budget": 123},
                        out_dir=out, tags=["selftest"])
    assert lg.active, "logger failed to start (auth/network?)"
    rows = [{"condition": "FP16 baseline", "bits": 16, "median_corr": 1.0, "survived_>0.9_pct": 100.0, "perplexity": 458.5, "n_active_features": 7134},
            {"condition": "RTN INT0", "bits": 0, "median_corr": 1.0, "survived_>0.9_pct": 100.0, "perplexity": 458.5}]
    rows += [{"condition": f"RTN INT{b}", "bits": b, "median_corr": 1 - 0.7 * REL[b], "survived_>0.9_pct": 99.0, "perplexity": 460.0, "ppl_delta_pct": 0.3} for b in REL]
    for b, rel in REL.items():
        (out / f"weight_perturbation_INT{b}.json").write_text(json.dumps({"bits": b, "rel_l2": rel}))
    (out / "run_meta.json").write_text("{}"); (out / "snap_RTN_INT8.pt").write_text("x")
    for r in rows:
        lg.log_condition(r)
    lg.heartbeat("RTN INT8", 100, 1953, 12.3, {"cuda:0": 7.5, "cuda:1": 6.9})
    lg.log_table(rows); lg.log_gates(rows, out)
    run_path = f"{lg.run.entity}/{lg.run.project}/{lg.run.id}"
    lg.finish(out)

api = wandb.Api(timeout=60)
run = api.run(run_path)
hist = run.history(samples=500, pandas=False)
hkeys = set().union(*[h.keys() for h in hist])
bad_s, bad_c = check_run(run)
need = {"bits", "median_corr", "survival_pct", "perplexity", "weight_rel_l2", "sys/windows_done", "sys/mem_cuda0_gib"}
print("run:", run_path, "| state:", run.state)
print("summary keys:", sorted(k for k in run.summary.keys() if not k.startswith("_")))
print("bad summary keys:", bad_s, "| bad config keys:", bad_c)
print("history metrics present:", sorted(need & hkeys), "| missing:", sorted(need - hkeys))
print("gate1_verdict:", run.summary.get("gate1_verdict"), "| survival_pct_int7:", run.summary.get("survival_pct_int7"), "| config bos_regime/n_tokens/stack:",
      run.config.get("bos_regime"), run.config.get("n_tokens"), run.config.get("stack"))
arts = list(run.logged_artifacts())
files = [f.name for a in arts for f in a.files()]
print("artifact files:", sorted(files), "| snapshot excluded:", not any(f.startswith("snap_") for f in files))
ok = (not bad_s and not bad_c and not (need - hkeys) and run.summary.get("gate1_verdict") == "signal_limited" and run.summary.get("survival_pct_int7") == 99.0
      and run.config.get("bos_regime") == "mask" and files and not any(f.startswith("snap_") for f in files))
for a in arts:
    try: a.delete(delete_aliases=True)
    except Exception as e: print("artifact cleanup note:", type(e).__name__)
run.delete(delete_artifacts=True)
print("cleanup: run deleted")
print("LIVE SELFTEST", "PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)
