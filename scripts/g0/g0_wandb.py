"""wandb logging and closure-protocol gate metrics for the G0 runs (see reports/duan_codebase_audit.md section 7)."""
import json, math, os, re, sys, uuid
from pathlib import Path

SLOPE_RUNGS = (12, 10, 8, 7, 6)   # widths used for the scaling test of (1 - median r) vs relative weight perturbation


def compute_gate_metrics(rows, pert):
    """rows: per-condition summary dicts (keys: condition, median_corr, survived_>0.9_pct, perplexity).
    pert: {bits: relative L2 weight change}. Returns the Gate 0 / Gate 1 quantities of the closure protocol."""
    by = {r["condition"]: r for r in rows}
    base, int0 = by.get("FP16 baseline"), by.get("RTN INT0")
    identity = None
    if base is not None and int0 is not None:
        identity = (int0["median_corr"] >= 1.0 - 1e-12 and int0["survived_>0.9_pct"] == 100.0
                    and abs(int0["perplexity"] - base["perplexity"]) <= 1e-6 * abs(base["perplexity"]))

    def median_r(b):
        r = by.get(f"RTN INT{b}")
        return None if r is None else r["median_corr"]

    pts = []
    for b in SLOPE_RUNGS:
        r, rel = by.get(f"RTN INT{b}"), pert.get(b)
        if r is not None and rel and rel > 0 and 1.0 - r["median_corr"] > 0:
            pts.append((math.log10(rel), math.log10(1.0 - r["median_corr"])))
    slope = None
    if len(pts) >= 3:
        mx, my = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
        slope = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)

    r16, r12 = median_r(16), median_r(12)
    if r16 is None or r12 is None or slope is None:
        verdict = "incomplete"
    elif r16 >= 0.99 and r12 >= 0.99 and 0.5 <= slope <= 1.5:
        verdict = "signal_limited"
    else:
        verdict = "noise_dominated"
    return {"gate0_int0_identity": identity, "gate1_median_r_int16": r16, "gate1_median_r_int12": r12,
            "gate1_slope": slope, "gate1_verdict": verdict}


# ------------------------------------------------------------------------------------------ logger
def _clean(key):
    return str(key).replace(">", "gt").replace("<", "lt")


def _label(condition):
    c = str(condition)
    return "baseline" if c.startswith("FP16") else c.replace("RTN ", "").replace(" ", "_")


def _numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _safe(fn):
    """Logging must never break an experiment: swallow and report any error."""
    def wrapper(self, *args, **kwargs):
        if not self.active:
            return None
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"[g0_wandb] {fn.__name__} failed ({type(e).__name__}: {e}); continuing", file=sys.stderr)
            return None
    return wrapper


class G0Logger:
    def __init__(self, run=None, wb=None, name="g0"):
        self.run, self.wb, self.name = run, wb, name

    @property
    def active(self):
        return self.run is not None

    @classmethod
    def start(cls, *, project, name, config, out_dir, entity=None, tags=(), enabled=None, wandb_module=None):
        if enabled is None:
            enabled = os.environ.get("G0_WANDB", "1") != "0"
        if not enabled:
            return cls()
        try:
            wb = wandb_module if wandb_module is not None else __import__("wandb")
            out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            id_file = out_dir / "wandb_run_id.txt"
            run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:8]
            id_file.write_text(run_id)
            run = wb.init(project=project, entity=entity, name=name, config=config, id=run_id, resume="allow",
                          tags=list(tags), job_type="g0", dir=str(out_dir))
            return cls(run, wb, name)
        except Exception as e:  # noqa: BLE001
            print(f"[g0_wandb] logging disabled: init failed ({type(e).__name__}: {e})", file=sys.stderr)
            return cls()

    @_safe
    def log(self, d, step=None):
        self.run.log(d)

    @_safe
    def log_condition(self, summary):
        label = _label(summary.get("condition", "unknown"))
        self.run.log({f"{label}/{_clean(k)}": float(v) for k, v in summary.items() if _numeric(v)})

    @_safe
    def heartbeat(self, condition, windows_done, windows_total, elapsed_s, mem_gib=None):
        d = {"progress/windows_done": windows_done, "progress/windows_total": windows_total,
             "progress/elapsed_s": elapsed_s}
        d.update({f"mem/{dev}_gib": v for dev, v in (mem_gib or {}).items()})
        self.run.log(d)

    @_safe
    def log_table(self, rows, key="summary"):
        if hasattr(rows, "to_dict"):
            rows = rows.to_dict("records")
        cols = list(rows[0].keys())
        self.run.log({key: self.wb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])})

    @_safe
    def log_gates(self, rows, out_dir):
        if hasattr(rows, "to_dict"):
            rows = rows.to_dict("records")
        pert = {}
        for f in Path(out_dir).glob("weight_perturbation_INT*.json"):
            j = json.loads(f.read_text()); pert[int(j["bits"])] = float(j["rel_l2"])
        g = compute_gate_metrics(rows, pert)
        d = {f"pert/INT{b}_rel_l2": v for b, v in pert.items()}
        for k in ("slope", "median_r_int16", "median_r_int12"):
            if g[f"gate1_{k}"] is not None:
                d[f"gate1/{k}"] = g[f"gate1_{k}"]
        if g["gate0_int0_identity"] is not None:
            d["gate0/int0_identity"] = float(g["gate0_int0_identity"])
        self.run.log(d)
        for k, v in g.items():
            self.run.summary[k] = v

    @_safe
    def finish(self, out_dir):
        keep_snaps = os.environ.get("G0_WANDB_SNAPS", "0") == "1"
        art = self.wb.Artifact(re.sub(r"[^A-Za-z0-9_.-]", "-", self.name), type="results")
        for f in sorted(Path(out_dir).iterdir()):
            if f.is_file() and (keep_snaps or not f.name.startswith("snap_")):
                art.add_file(str(f))
        self.run.log_artifact(art)
        self.run.finish()


def gpu_mem_gib(torch_module=None):
    """Peak allocated GPU memory per device in GiB. Never raises: it is called from inside the experiment loop,
    before any logging wrapper, so a CUDA hiccup must not kill a multi-hour run."""
    try:
        t = torch_module if torch_module is not None else __import__("torch")
        if not t.cuda.is_available():
            return {}
        return {f"cuda:{i}": t.cuda.max_memory_allocated(i) / 2**30 for i in range(t.cuda.device_count())}
    except Exception:  # noqa: BLE001
        return {}
