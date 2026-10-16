"""wandb logging and closure-protocol gate metrics for the G0 runs (see reports/duan_codebase_audit.md section 7)."""
import json, math, numbers, os, re, sys, uuid
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


# ------------------------------------------------------------------------------------------ fixed schema (v1)
# ONE vocabulary for every run (live or rebuilt): the condition lives in the `bits` column of a row / in the key SUFFIX of a small
# fixed set of summary rungs, never in a free-form metric name, so the runs table has the same headers for every run.
SCHEMA_VERSION = 1
CONFIG_KEYS = ("experiment", "cell", "stack", "bos_regime", "attn_impl", "sae_dtype", "quantizer", "n_tokens", "layer", "model",
               "slurm_job", "source", "code_ref", "schema_version", "config_extra")
METRIC_KEYS = ("bits", "n_tokens", "n_active_features", "mean_corr", "median_corr", "survival_pct", "degraded_pct", "damaged_pct",
               "perplexity", "loss", "ppl_delta_pct", "weight_rel_l2")
SUMMARY_RUNGS = (16, 12, 8, 7, 6, 5, 4)
_PER_RUNG = ("survival_pct", "median_corr", "ppl_delta_pct")
GATE_KEYS = ("gate0_int0_identity", "gate1_median_r_int16", "gate1_median_r_int12", "gate1_slope", "gate1_verdict")
SUMMARY_KEYS = (("fp_perplexity", "fp_n_active_features") + tuple(f"{m}_int{b}" for b in SUMMARY_RUNGS for m in _PER_RUNG)
                + GATE_KEYS + ("q1_pass",))
TABLE_KEYS = ("conditions", "comparison_vs_published", "per_feature_int6")
FP_BITS = 32    # the unquantized (bf16) baseline; 16 would collide with the INT16 rung
_METRIC_ALIASES = {"survived_>0.9_pct": "survival_pct", "degraded_0.5_0.9_pct": "degraded_pct", "damaged_<0.5_pct": "damaged_pct"}
_CONFIG_ALIASES = {"token_budget": "n_tokens", "tokens": "n_tokens", "bos": "bos_regime", "regime": "bos_regime", "attn": "attn_impl", "quant": "quantizer"}
_BOS = {"none": "none", "prepend_keep": "keep", "prepend_mask": "mask", "keep": "keep", "mask": "mask"}


def _is_number(v):
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)


def condition_bits(condition):
    c = str(condition)
    if c.upper().startswith("FP"):
        return FP_BITS
    m = re.search(r"INT(\d+)", c)
    return int(m.group(1)) if m else None


def condition_row(raw, rel_l2=None):
    """One condition -> the fixed metric names (no condition inside any name); non-metrics and non-finite values are dropped."""
    row = {"condition": str(raw.get("condition", ""))}
    for k, v in raw.items():
        name = _METRIC_ALIASES.get(k, k)
        if name in METRIC_KEYS and name != "bits" and _is_number(v):
            row[name] = v
    bits = condition_bits(row["condition"])
    if bits is None and _is_number(raw.get("bits")):
        bits = int(raw["bits"])
    if bits is not None:
        row["bits"] = bits
    if _is_number(rel_l2):
        row["weight_rel_l2"] = float(rel_l2)
    return row


def run_summary(rows):
    """The fixed runs-table columns: FP anchors plus survival / median r / perplexity change at the fixed rungs (absent if not measured)."""
    s = {}
    for r in rows:
        b = r.get("bits")
        if b == FP_BITS:
            for src, dst in (("perplexity", "fp_perplexity"), ("n_active_features", "fp_n_active_features")):
                if src in r:
                    s[dst] = r[src]
        elif b in SUMMARY_RUNGS:
            for m in _PER_RUNG:
                if m in r:
                    s[f"{m}_int{b}"] = r[m]
    return s


def normalize_config(raw, env=None):
    """Exactly CONFIG_KEYS, 'n/a' when unknown; legacy runner keys are mapped; everything else goes to one JSON string `config_extra`."""
    env = os.environ if env is None else env
    cfg, extra = {k: "n/a" for k in CONFIG_KEYS}, {}
    cfg["schema_version"] = SCHEMA_VERSION
    for k, v in (raw or {}).items():
        name = _CONFIG_ALIASES.get(k, k)
        if name in CONFIG_KEYS and name not in ("schema_version", "config_extra"):
            cfg[name] = _BOS.get(v, v) if name == "bos_regime" else v
        else:
            extra[k] = v
    if cfg["stack"] == "n/a" and "transformer_lens" in extra:
        cfg["stack"] = "TL"
    if cfg["stack"] == "TL":      # their TransformerLens script is pinned to this model, layer, fp32 sae_lens SAE and quantizer
        for k, v in (("attn_impl", "tl"), ("sae_dtype", "fp32"), ("quantizer", "duan"), ("model", "google/gemma-2-2b"), ("layer", 12)):
            if cfg[k] == "n/a":
                cfg[k] = v
    if cfg["source"] == "n/a":
        cfg["source"] = "live"
    if cfg["code_ref"] == "n/a" and env.get("G0_CODE_REF"):
        cfg["code_ref"] = env["G0_CODE_REF"]
    if cfg["slurm_job"] == "n/a" and env.get("SLURM_JOB_ID"):
        cfg["slurm_job"] = env["SLURM_JOB_ID"]
    if cfg["experiment"] == "n/a" and env.get("G0_EXPERIMENT"):
        cfg["experiment"] = env["G0_EXPERIMENT"]
    cfg["config_extra"] = json.dumps(extra, sort_keys=True, default=str) if extra else "n/a"
    return cfg


# ------------------------------------------------------------------------------------------ logger
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
        self._rows, self._out_dir, self._conditions_table_logged = [], None, False

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
            run = wb.init(project=project, entity=entity, name=name, config=normalize_config(config), id=run_id, resume="allow",
                          tags=list(tags), job_type="g0", dir=str(out_dir))
            lg = cls(run, wb, name)
            lg._out_dir = out_dir
            lg._define_metrics()
            return lg
        except Exception as e:  # noqa: BLE001
            print(f"[g0_wandb] logging disabled: init failed ({type(e).__name__}: {e})", file=sys.stderr)
            return cls()

    @_safe
    def _define_metrics(self):
        self.run.define_metric("bits", summary="none")
        for k in METRIC_KEYS:
            if k != "bits":
                self.run.define_metric(k, step_metric="bits", summary="none")     # history only: the runs table shows SUMMARY_KEYS
        self.run.define_metric("sys/*", summary="none")

    def _rel_l2(self, bits):
        if self._out_dir is None or bits is None:
            return None
        f = self._out_dir / f"weight_perturbation_INT{bits}.json"
        return json.loads(f.read_text()).get("rel_l2") if f.exists() else None

    @_safe
    def log(self, d, step=None):
        self.run.log(d)

    @_safe
    def log_condition(self, summary):
        row = condition_row(summary)
        row = condition_row(summary, rel_l2=self._rel_l2(row.get("bits")))
        self._rows.append(row)
        self.run.log({k: v for k, v in row.items() if k != "condition"})
        self.run.summary.update(run_summary(self._rows))

    @_safe
    def heartbeat(self, condition, windows_done, windows_total, elapsed_s, mem_gib=None):
        d = {"sys/windows_done": windows_done, "sys/windows_total": windows_total, "sys/elapsed_s": elapsed_s}
        d.update({f"sys/mem_{re.sub(r'[^a-z0-9]', '', str(dev).lower())}_gib": v for dev, v in (mem_gib or {}).items()})
        self.run.log(d)

    def _conditions_rows(self):
        last = {}
        for r in self._rows:                      # a resumed run may log a condition twice: keep the last
            last[r["condition"]] = r
        return list(last.values())

    @_safe
    def log_table(self, rows, key="conditions"):
        if hasattr(rows, "to_dict"):
            rows = rows.to_dict("records")
        if key == "conditions":
            cols = ["condition"] + list(METRIC_KEYS)
            rows = [condition_row(r, rel_l2=self._rel_l2(condition_bits(r.get("condition", "")))) for r in rows]
            self._conditions_table_logged = True
        else:
            cols = list(rows[0].keys())
        self.run.log({key: self.wb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])})

    @_safe
    def set_summary(self, d):
        """Set fixed summary keys that are not derived from condition rows (e.g. q1_pass); anything outside SUMMARY_KEYS is ignored."""
        self.run.summary.update({k: v for k, v in d.items() if k in SUMMARY_KEYS})

    @_safe
    def log_gates(self, rows, out_dir):
        if hasattr(rows, "to_dict"):
            rows = rows.to_dict("records")
        pert = {}
        for f in Path(out_dir).glob("weight_perturbation_INT*.json"):
            j = json.loads(f.read_text()); pert[int(j["bits"])] = float(j["rel_l2"])
        g = compute_gate_metrics(rows, pert)
        self.run.summary.update({k: g[k] for k in GATE_KEYS if g.get(k) is not None})

    @_safe
    def finish(self, out_dir):
        if not self._conditions_table_logged and self._rows:
            cols = ["condition"] + list(METRIC_KEYS)
            data = [[r.get(c) for c in cols] for r in self._conditions_rows()]
            self.run.log({"conditions": self.wb.Table(columns=cols, data=data)})
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
