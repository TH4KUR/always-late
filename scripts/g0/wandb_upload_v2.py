"""Post-hoc upload of the FINISHED results of the project to wandb (schema v2, reports/wandb_schema_v2.md, decision D-091; the user chose Option A on 2026-10-26).
One run per study or result family. Every number is read from a file committed under reports/ (json or csv); nothing is retyped. No weights, no item-level .pt files, no key.

  python3 scripts/g0/wandb_upload_v2.py build  --out DIR [--only SLUG ...] [--dry-run] [--commit HASH]     reads the committed files, writes DIR/payload.json and DIR/files.tar (offline, no wandb)
  python wandb_upload_v2.py upload --payload DIR/payload.json --root EXTRACTED_DIR [--only SLUG ...]        uploads (needs wandb and credentials in the environment of the machine it runs on: Ada, login node)

The builder is pure (`build_run`, `build_all`); the uploader takes the wandb module as an argument so that it is tested with a fake one."""
import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ENTITY, PROJECT = "karma-skz-", "anlp-project"
SCHEMA_VERSION = 2
SOURCE = "post-hoc-upload-of-committed-results"
BANNED_EXT = {".pt", ".pth", ".bin", ".safetensors", ".npy", ".npz", ".ckpt", ".gguf", ".zip", ".tar", ".gz"}
MAX_FILE_BYTES = 2_000_000
MAX_RUN_BYTES = 20_000_000
MAX_TABLE_ROWS = 10_000
CONFIG_KEYS = ["schema_version", "study", "title", "status", "decisions", "model", "sae", "layer", "quantizer", "states", "slurm_jobs", "code_ref", "freeze_file", "freeze_sha256", "source", "upload_commit", "files_sha256", "limits"]
BITS = {"FP": 32, "INT16": 16, "INT12": 12, "INT8": 8, "INT7": 7, "INT6": 6, "INT5": 5, "INT4": 4, "INT0": 0}
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,60}$")
NUMBER = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$")
INTEGER = re.compile(r"^-?\d+$")


class SpecError(ValueError):
    pass


# ---------------------------------------------------------------- reading the committed files

def check_file(root, rel):
    """The guards: below reports/, no traversal, no banned extension, small, present. Returns the size in bytes."""
    if not isinstance(rel, str) or not rel or rel.startswith("/") or ".." in rel.split("/"):
        raise SpecError(f"unsafe path {rel!r}")
    parts = rel.split("/")
    if parts[0] != "reports" or len(parts) < 2:
        raise SpecError(f"{rel}: only files below reports/ are uploaded")
    if Path(rel).suffix.lower() in BANNED_EXT:
        raise SpecError(f"{rel}: this file type is never uploaded")
    p = Path(root) / rel
    if not p.is_file():
        raise SpecError(f"{rel}: not a file")
    size = p.stat().st_size
    if size > MAX_FILE_BYTES:
        raise SpecError(f"{rel}: {size} bytes is above the cap of {MAX_FILE_BYTES}")
    return size


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(root, rel):
    check_file(root, rel)
    return json.loads((Path(root) / rel).read_text())


def resolve(obj, path, label):
    """Follow `path` through `obj`: a str is a key, an int an index, a dict selects the ONE list element whose fields equal it, the last element '#len' gives the length."""
    cur = obj
    for i, step in enumerate(path):
        where = f"{label}: {list(path)!r} at {step!r}"
        if step == "#len" and i == len(path) - 1:
            if not isinstance(cur, (list, dict)):
                raise SpecError(where)
            return len(cur)
        if isinstance(step, dict):
            if not isinstance(cur, list):
                raise SpecError(where)
            hits = [e for e in cur if isinstance(e, dict) and all(e.get(k) == v for k, v in step.items())]
            if len(hits) != 1:
                raise SpecError(f"{where}: {len(hits)} matches")
            cur = hits[0]
        elif isinstance(step, int) and not isinstance(step, bool):
            if not isinstance(cur, list) or not -len(cur) <= step < len(cur):
                raise SpecError(where)
            cur = cur[step]
        elif isinstance(step, str):
            if not isinstance(cur, dict) or step not in cur:
                raise SpecError(where)
            cur = cur[step]
        else:
            raise SpecError(where)
    return cur


def _cell(s):
    if s == "":
        return None
    if INTEGER.match(s):
        return int(s)
    if NUMBER.match(s):
        return float(s)
    return s


def read_csv(root, rel):
    check_file(root, rel)
    with open(Path(root) / rel, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SpecError(f"{rel}: empty csv")
    cols, body = rows[0], rows[1:]
    if len(body) > MAX_TABLE_ROWS:
        raise SpecError(f"{rel}: {len(body)} rows is above the cap of {MAX_TABLE_ROWS}")
    for k, r in enumerate(body):
        if len(r) != len(cols):
            raise SpecError(f"{rel}: row {k + 2} has {len(r)} cells, the header {len(cols)}")
    return cols, [[_cell(c) for c in r] for r in body]


# ---------------------------------------------------------------- history of the ladder runs

def ladder_rows(macro, retention):
    rows = []
    for p, m in macro.items():
        if p not in BITS:
            raise SpecError(f"unknown precision {p!r}")
        row = {"bits": BITS[p], "effect": m["point"], "effect_lo": m["lo"], "effect_hi": m["hi"]}
        if m.get("median") is not None:
            row["effect_median"] = m["median"]
        r = (retention.get(p) or {}).get("macro") or {}
        if r.get("ratio") is not None:
            row["retention"], row["retention_lo"], row["retention_hi"] = r["ratio"], r["lo"], r["hi"]
            if r.get("median_ratio") is not None:
                row["retention_median"] = r["median_ratio"]
        rows.append(row)
    rows.sort(key=lambda r: -r["bits"])
    return rows


def history_lm_ladder(root):
    a = read_json(root, "reports/functional_validation/results/eval/analysis.json")
    return ladder_rows(a["macro"], a["retention"])


def history_belebele(root):
    base = "reports/functional_validation/results/mc_eval/"
    a = read_json(root, base + "analysis.json")
    rows = {r["bits"]: r for r in ladder_rows(a["macro"], a["retention"])}
    cols, body = read_csv(root, base + "accuracy.csv")
    ix = {c: cols.index(c) for c in ("precision", "acc_none", "acc_zero", "acc_control_mean", "acc_sham")}
    for p, bits in BITS.items():
        sel = [r for r in body if r[ix["precision"]] == p]
        if sel and bits in rows:
            for name, col in (("accuracy_none", "acc_none"), ("accuracy_zero", "acc_zero"), ("accuracy_controls", "acc_control_mean"), ("accuracy_sham", "acc_sham")):
                rows[bits][name] = statistics.fmean(r[ix[col]] for r in sel)
    return sorted(rows.values(), key=lambda r: -r["bits"])


def history_llama(root):
    base = "reports/functional_validation/"
    a, rb = read_json(root, base + "llama/analysis.json"), read_json(root, base + "llama_robust/robust.json")
    rows = []
    for s, v in a["P2"]["retention"].items():
        rr = rb["retention"][s]
        rows.append({"bits": BITS[s], "retention_mean": v, "retention_median": rr["median_ratio"], "retention_median_lo": rr["median_lo"], "retention_median_hi": rr["median_hi"], "gemma_retention": a["gemma_reference"]["retention"][s]})
    return sorted(rows, key=lambda r: -r["bits"])


HISTORIES = {"lm-ladder": history_lm_ladder, "belebele": history_belebele, "llama": history_llama}


# ---------------------------------------------------------------- the builder

def _jobs(d):
    return "; ".join(f"{k}={v}" for k, v in d.items()) if d else "n/a"


def _na(v):
    return "n/a" if v is None or v == "" else v


def build_run(root, f, upload_commit):
    root = Path(root)
    slug = f["slug"]
    if not isinstance(slug, str) or not SLUG.match(slug):
        raise SpecError(f"bad slug {slug!r}")
    if f["status"] not in ("exploratory", "confirmatory"):
        raise SpecError(f"{slug}: status must be exploratory or confirmatory")
    cache = {}

    def doc(rel):
        if rel not in cache:
            cache[rel] = read_json(root, rel)
        return cache[rel]

    summary, seen = {}, set()
    for name, rel, path in f["metrics"]:
        if not name.startswith("headline/"):
            raise SpecError(f"{slug}: metric name {name!r} must start with headline/")
        if name in seen:
            raise SpecError(f"{slug}: duplicate metric {name!r}")
        seen.add(name)
        v = resolve(doc(rel), path, rel)
        if not isinstance(v, (int, float, bool, str)):
            raise SpecError(f"{slug}: {name} resolves to a non-scalar value in {rel}")
        summary[name] = v
    tables = {}
    for key, rel in f["tables"]:
        cols, rows = read_csv(root, rel)
        tables[key] = {"columns": cols, "rows": rows, "file": rel}
    images = []
    for rel in f["images"]:
        check_file(root, rel)
        if not rel.lower().endswith(".png"):
            raise SpecError(f"{rel}: figures are uploaded as png")
        stem = Path(rel).stem
        images.append({"key": f"fig/{stem}", "path": rel, "caption": stem})
    paths = set(f["files"]) | {t["file"] for t in tables.values()} | {i["path"] for i in images} | ({f["freeze"]} if f.get("freeze") else set())
    files = []
    for rel in sorted(paths):
        size = check_file(root, rel)
        files.append({"path": rel, "sha256": sha256_file(root / rel), "bytes": size})
    total = sum(x["bytes"] for x in files)
    if total > MAX_RUN_BYTES:
        raise SpecError(f"{slug}: {total} bytes is above the run cap of {MAX_RUN_BYTES}")
    freeze = f.get("freeze")
    code_ref = f.get("code_ref")
    if isinstance(code_ref, dict):
        code_ref = resolve(doc(code_ref["file"]), code_ref["path"], code_ref["file"])
    history = []
    if f.get("history"):
        if f["history"] not in HISTORIES:
            raise SpecError(f"{slug}: unknown history {f['history']!r}")
        history = HISTORIES[f["history"]](root)
        if not all(isinstance(r, dict) and "bits" in r for r in history):
            raise SpecError(f"{slug}: every history row needs bits")
    config = {"schema_version": SCHEMA_VERSION, "study": slug, "title": f["title"], "status": f["status_text"], "decisions": ", ".join(f["decisions"]) or "n/a",
              "model": _na(f.get("model")), "sae": _na(f.get("sae")), "layer": _na(f.get("layer")), "quantizer": _na(f.get("quantizer")), "states": ", ".join(f["states"]) or "n/a",
              "slurm_jobs": _jobs(f.get("jobs")), "code_ref": _na(code_ref), "freeze_file": freeze or "n/a", "freeze_sha256": sha256_file(root / freeze) if freeze else "n/a",
              "source": SOURCE, "upload_commit": upload_commit, "files_sha256": json.dumps({x["path"]: x["sha256"] for x in files}, sort_keys=True), "limits": f["limits"]}
    summary = {**summary, "n_tables": len(tables), "n_images": len(images), "n_files": len(files)}
    tags = ["schema-v2", "post-hoc-upload", f["status"], slug, f["group"]]
    return {"id": f"v2-{slug}", "name": f["name"], "group": f["group"], "job_type": "analysis-upload", "tags": tags, "config": config, "summary": summary, "history": history,
            "tables": tables, "images": images, "files": files, "bytes_total": total}


def build_all(root, families=None, only=None, upload_commit="n/a"):
    families = FAMILIES if families is None else families
    slugs = [f["slug"] for f in families]
    if len(set(slugs)) != len(slugs):
        raise SpecError("duplicate slugs")
    if only:
        unknown = [s for s in only if s not in slugs]
        if unknown:
            raise SpecError(f"unknown slugs {unknown}")
    return {"schema_version": SCHEMA_VERSION, "entity": ENTITY, "project": PROJECT, "upload_commit": upload_commit,
            "runs": [build_run(root, f, upload_commit) for f in families if not only or f["slug"] in only]}


def shipped_files(payload):
    return sorted({x["path"] for r in payload["runs"] for x in r["files"]})


def write_payload(root, payload, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / "payload.json").write_text(json.dumps(payload, indent=1) + "\n")
    with tarfile.open(out / "files.tar", "w") as t:
        for p in shipped_files(payload):
            check_file(root, p)
            t.add(str(Path(root) / p), arcname=p, recursive=False)


def dirty_files(root, paths):
    """The paths among `paths` that differ from the committed version (modified, staged, untracked). A directory that is not a git work tree has none: the check then says nothing."""
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--"] + list(paths), cwd=root, capture_output=True, text=True, check=True).stdout
    return sorted(line[3:] for line in out.splitlines() if line.strip())


def dry_run_text(payload):
    runs = payload["runs"]
    lines = [f"{len(runs)} runs, {len(shipped_files(payload))} files to ship, {sum(r['bytes_total'] for r in runs)} bytes (upload commit {payload['upload_commit']})"]
    for r in runs:
        lines.append(f"{r['id']:<34} {r['name']:<46} group {r['group']:<10} metrics {len(r['summary']) - 3}  tables {len(r['tables'])}  images {len(r['images'])}  files {len(r['files'])}  history {len(r['history'])}  {r['bytes_total']} bytes")
    return "\n".join(lines)


# ---------------------------------------------------------------- the uploader (the wandb module is an argument)

def upload(payload, root, wandb, only=None, entity=ENTITY, project=PROJECT, out=print, force_media=False):
    root = Path(root)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SpecError("the payload is not of schema version 2")
    runs = payload["runs"]
    if only:
        unknown = [s for s in only if f"v2-{s}" not in {r["id"] for r in runs}]
        if unknown:
            raise SpecError(f"unknown slugs {unknown}")
        runs = [r for r in runs if r["id"][3:] in only]
    for r in runs:                                               # everything is checked before the first run starts
        if not r["id"].startswith("v2-"):
            raise SpecError(f"{r['id']}: only v2 run ids may be written, the older runs stay untouched")
        for x in r["files"]:
            check_file(root, x["path"])
            if sha256_file(root / x["path"]) != x["sha256"]:
                raise SpecError(f"{x['path']}: the file differs from the payload")
    results = []
    for r in runs:
        run = wandb.init(project=project, entity=entity, id=r["id"], name=r["name"], group=r["group"], job_type=r["job_type"], tags=r["tags"], config=r["config"], resume="allow")
        first = getattr(run, "step", 0) == 0
        if r["history"] and first:
            run.define_metric("bits")
            for k in sorted({k for row in r["history"] for k in row if k != "bits"}):
                run.define_metric(k, step_metric="bits")
            for row in r["history"]:
                run.log(dict(row))
        media = {f"table/{k}": wandb.Table(columns=t["columns"], data=t["rows"]) for k, t in r["tables"].items()}
        media.update({i["key"]: wandb.Image(str(root / i["path"]), caption=i["caption"]) for i in r["images"]})
        if media and (first or force_media):                       # a re-run of a finished run adds no history row; --force-media repeats the tables and images of an interrupted run
            run.log(media)
        run.summary.update(r["summary"])
        art = wandb.Artifact(f"{r['id']}-files", type="analysis-results", metadata={"upload_commit": payload["upload_commit"], "study": r["config"]["study"], "n_files": len(r["files"])})
        for x in r["files"]:
            art.add_file(str(root / x["path"]), name=x["path"])
        with tempfile.TemporaryDirectory() as td:
            mf = Path(td) / "MANIFEST.json"
            mf.write_text(json.dumps({"id": r["id"], "upload_commit": payload["upload_commit"], "files": {x["path"]: x["sha256"] for x in r["files"]}}, indent=1, sort_keys=True) + "\n")
            art.add_file(str(mf), name="MANIFEST.json")
            run.log_artifact(art)
            run.finish()
        results.append({"id": r["id"], "name": r["name"], "url": getattr(run, "url", None), "history_logged": bool(r["history"] and first)})
        out(f"uploaded {r['id']}  {getattr(run, 'url', '')}")
    return results


# ---------------------------------------------------------------- the families

FV = "reports/functional_validation/"
GEMMA = {"model": "google/gemma-2-2b", "sae": "Gemma Scope gemma-scope-2b-pt-res-canonical, layer_12/width_16k/canonical (JumpReLU, 16,384 features)", "layer": 12, "quantizer": "round-to-nearest per output channel (RTN), weights only"}
FREEZE_LM = FV + "evaluation_freeze_lm.json"
FREEZE_MC = FV + "evaluation_freeze_mc.json"
LM_LIMITS = "exploratory: FLORES devtest sentences examined in earlier work, features selected on dev; one model, one layer, one SAE, RTN weights only; next-token loss on the language's own text, not a task result"
LADDER = ["FP", "INT8", "INT7", "INT6", "INT5", "INT4"]


def M(name, rel, *path):
    return (name, rel, list(path))


def _ladder_metrics():
    E = FV + "results/eval/analysis.json"
    m = [M("headline/macro_effect", E, "macro", "FP", "point"), M("headline/macro_effect_lo", E, "macro", "FP", "lo"), M("headline/macro_effect_hi", E, "macro", "FP", "hi"), M("headline/median_effect", E, "macro", "FP", "median"),
         M("headline/n_positive", E, "macro", "FP", "n_positive"), M("headline/n_sentences", E, "n_positions"), M("headline/n_languages", E, "covered", "#len")]
    for b in (8, 7, 6, 5, 4):
        m.append(M(f"headline/retention_INT{b}", E, "retention", f"INT{b}", "macro", "ratio"))
    for b in (5, 4):
        m += [M(f"headline/retention_INT{b}_lo", E, "retention", f"INT{b}", "macro", "lo"), M(f"headline/retention_INT{b}_hi", E, "retention", f"INT{b}", "macro", "hi")]
    m += [M("headline/retention_median_INT4", E, "retention", "INT4", "macro", "median_ratio"), M("headline/retention_class_INT4", E, "retention", "INT4", "macro", "class")]
    for b in (16, 12, 0):                                                                               # the rounding-only controls
        m.append(M(f"headline/retention_INT{b}", E, "retention", f"INT{b}", "macro", "ratio"))
    return m


def _rescue_metrics():
    R = FV + "results/rescue/analysis.json"
    comp = {"group": "group", "control": "control", "group_minus_control": "group_minus_control", "all": "all_features", "error": "error_term", "full": "full_residual"}
    m = [M(f"headline/rescue_{name}_INT{b}", R, "rescue", f"INT{b}", "macro_fraction", key, "point") for b in (4, 5, 6) for key, name in comp.items()]
    return m + [M(f"headline/n_languages_INT{b}", R, "rescue", f"INT{b}", "macro_fraction", "group", "n_languages") for b in (4, 5, 6)]


def _f4_metrics():
    m = []
    for b in (4, 5, 6):
        F = FV + f"results/f4/INT{b}/f4_results.json"
        m += [M(f"headline/f4_E_b_point_INT{b}", F, "macro", "E_b", "point"), M(f"headline/f4_E_b_median_INT{b}", F, "macro", "E_b", "median"), M(f"headline/f4_E_b_median_lo_INT{b}", F, "macro", "E_b", "median_lo"),
              M(f"headline/f4_E_b_median_hi_INT{b}", F, "macro", "E_b", "median_hi"), M(f"headline/f4_E_b_without_top_point_INT{b}", F, "macro", "E_b", "without_top", "point"),
              M(f"headline/f4_E_fp_median_INT{b}", F, "macro", "E_fp", "median"), M(f"headline/f4_delta_median_INT{b}", F, "macro", "delta", "median"),
              M(f"headline/f4_n_languages_INT{b}", F, "languages", "#len"), M(f"headline/f4_E_b_positive_INT{b}", F, "counts", "E_b", "positive"), M(f"headline/f4_E_b_negative_INT{b}", F, "counts", "E_b", "negative"),
              M(f"headline/f4_E_b_inconclusive_INT{b}", F, "counts", "E_b", "inconclusive")]
    return m


def _mc_metrics():
    A, S = FV + "results/mc_eval/analysis.json", FV + "results/mc_eval/statements.json"
    m = [M("headline/macro_effect", A, "macro", "FP", "point"), M("headline/macro_effect_lo", A, "macro", "FP", "lo"), M("headline/macro_effect_hi", A, "macro", "FP", "hi"), M("headline/median_effect", A, "macro", "FP", "median"),
         M("headline/n_positive", A, "macro", "FP", "n_positive"), M("headline/n_covered", S, "n_covered"), M("headline/n_items_per_language", A, "accuracy", 0, "n_items"), M("headline/statements_complete", S, "complete"),
         M("headline/S0_point", S, "statements", {"id": "S0"}, "point"), M("headline/S0_lo", S, "statements", {"id": "S0"}, "lo"), M("headline/S0_hi", S, "statements", {"id": "S0"}, "hi")]
    return m + [M(f"headline/S{i}_outcome", S, "statements", {"id": f"S{i}"}, "outcome") for i in range(6)]


def _posctrl_metrics():
    P = FV + "results/mc_posctrl/summary.json"
    m = [M("headline/n_boot", P, "n_boot")]
    for d in ("256", "1024", "all"):
        m += [M(f"headline/fall_zero_{d}", P, "doses", d, "diff", "zero_minus_none", "mean"), M(f"headline/fall_zero_{d}_lo", P, "doses", d, "diff", "zero_minus_none", "lo"),
              M(f"headline/fall_zero_{d}_hi", P, "doses", d, "diff", "zero_minus_none", "hi"), M(f"headline/n_languages_{d}", P, "doses", d, "n_languages")]
    m += [M(f"headline/{p}_verdict", P, "verdicts", p, "verdict") for p in ("P1", "P2", "P3", "P4")]
    return m + [M("headline/largest_fall", P, "verdicts", "largest_fall", "fall"), M("headline/largest_fall_set", P, "verdicts", "largest_fall", "set")]


def _f5_metrics():
    F = FV + "results/f5/f5_results.json"
    m = [M("headline/n_tested", F, "counts", "n_tested"), M("headline/floor_tau", F, "floor", "tau"), M("headline/floor_source", F, "floor", "source"), M("headline/floor_adequate", F, "floor", "adequate")]
    for b in (8, 6, 5, 4):
        m += [M(f"headline/median_read_R1_INT{b}", F, "hypotheses", "H5a", "median_R1", f"INT{b}"), M(f"headline/median_write_W1_INT{b}", F, "hypotheses", "H5b", "median_W1", f"INT{b}")]
    m += [M(f"headline/share_below_floor_INT{b}", F, "hypotheses", "H5b", "share_below_floor", f"INT{b}") for b in (6, 5, 4)]
    m += [M("headline/H5a_holds", F, "hypotheses", "H5a", "holds"), M("headline/H5b_holds", F, "hypotheses", "H5b", "holds")]
    m += [M("headline/S1_share", F, "validity", "statements", "S1", "share"), M("headline/S3_share", F, "validity", "statements", "S3", "share"), M("headline/overall", F, "validity", "overall")]
    return m + [M(f"headline/S{i}_verdict", F, "validity", "statements", f"S{i}", "verdict") for i in (1, 2, 3)]


def _f5_sens_metrics():
    base = FV + "results/f5_sensitivity/"
    m = []
    for var, states in (("m8", ("INT4",)), ("m32", ("INT4",)), ("direction_swap", ("INT4",)), ("fixed_amplitude", ("INT4",)), ("fixed_amplitude_direction_swap", ("INT4",)), ("direction_swap_int8_6_5", ("INT8", "INT6", "INT5"))):
        rel = f"{base}{var}/sensitivity.json"
        m.append(M(f"headline/{var}_n_common", rel, "n_common"))
        for s in states:
            m += [M(f"headline/{var}_{s}_median_a", rel, "states", s, "median_a"), M(f"headline/{var}_{s}_median_b", rel, "states", s, "median_b"), M(f"headline/{var}_{s}_median_diff", rel, "states", s, "median_diff"),
                  M(f"headline/{var}_{s}_flip_share", rel, "states", s, "flip_share")]
    return m


def _gmain_metrics():
    G = FV + "gain_study/results/analysis.json"
    m = [M(f"headline/{k.replace('-', '_')}_met", G, "verdicts", k) for k in ("H-G1", "H-G2", "H-G3", "H-G4", "H-G5a", "H-G5b", "H-G6")]
    m += [M("headline/H_G1_agree_FP", G, "rules", "H-G1", "FP", "agree"), M("headline/H_G1_agree_INT4", G, "rules", "H-G1", "INT4", "agree"), M("headline/H_G2_count_FP", G, "rules", "H-G2", "FP", "count"),
          M("headline/H_G2_count_INT4", G, "rules", "H-G2", "INT4", "count"), M("headline/H_G3_count_increasing_FP", G, "rules", "H-G3", "FP", "count_increasing"),
          M("headline/H_G3_count_increasing_INT4", G, "rules", "H-G3", "INT4", "count_increasing"), M("headline/H_G6_rank_corr", G, "rules", "H-G6", "rank_corr"), M("headline/n_languages", G, "languages", "#len"), M("headline/n_states", G, "states", "#len")]
    return m


def _gc_metrics():
    C = FV + "gain_study/extension_c/analysis.json"
    m = [M(f"headline/{k.replace('-', '_')}_met", C, "verdicts", k) for k in ("H-Gc1", "H-Gc2", "H-Gc3")]
    for r in ("1.1", "1.2"):
        m += [M(f"headline/INT4_median_net_{r}", C, "rules", "H-Gc1", "per_ratio", r, "median"), M(f"headline/INT4_median_net_{r}_lo", C, "rules", "H-Gc1", "per_ratio", r, "lo"),
              M(f"headline/INT4_median_net_{r}_hi", C, "rules", "H-Gc1", "per_ratio", r, "hi"), M(f"headline/INT4_n_negative_{r}", C, "rules", "H-Gc1", "per_ratio", r, "n_negative")]
    return m + [M("headline/n_sentences", C, "n_sentences"), M("headline/n_controls", C, "n_controls"), M("headline/n_languages", C, "languages", "#len")]


def _gd_metrics():
    D = FV + "gain_study/extension_d/analysis.json"
    return [M("headline/verdict", D, "verdict"), M("headline/rho_shrinkage", D, "rd1", "rho"), M("headline/p_shrinkage", D, "rd1", "p"), M("headline/rho_placebo", D, "rd2", "rho"), M("headline/p_placebo", D, "rd2", "p"),
            M("headline/split_half_rho", D, "rd4", "rho_half"), M("headline/split_half_reliability", D, "rd4", "sb"), M("headline/n_languages", D, "n_languages")]


def _gb_metrics():
    B = FV + "gain_study/state_defined/results/analysis.json"
    m = [M(f"headline/{k.replace('-', '_')}_met", B, "verdicts", k) for k in ("H-Gb1", "H-Gb2", "H-Gb3", "H-Gb4a", "H-Gb4b")]
    m += [M("headline/jaccard_median_INT6", B, "rules", "H-Gb1", "median_int6"), M("headline/jaccard_median_INT4", B, "rules", "H-Gb1", "median_int4"), M("headline/H_Gb1_count", B, "rules", "H-Gb1", "count")]
    for s in ("INT6", "INT4"):
        m += [M(f"headline/H_Gb2_count_positive_{s}", B, "rules", "H-Gb2", s, "count_positive"), M(f"headline/H_Gb2_median_diff_{s}", B, "rules", "H-Gb2", s, "median_diff")]
    return m


def _promptcheck_metrics():
    P = FV + "promptcheck/analysis.json"
    return [M("headline/macro_diff", P, "macro", "diff"), M("headline/macro_diff_lo", P, "macro", "lo"), M("headline/macro_diff_hi", P, "macro", "hi"), M("headline/macro_label", P, "macro", "label"),
            M("headline/n_languages", P, "macro", "n_languages"), M("headline/english_acc_frozen", P, "primary", "acc_frozen"), M("headline/english_acc_harness", P, "primary", "acc_harness"),
            M("headline/english_diff", P, "primary", "diff"), M("headline/english_label", P, "primary", "label"), M("headline/reproduction_status", P, "reproduction", "status"), M("headline/margin", P, "margin")]


def _assoc_metrics():
    W, A = "reports/association_retention/results/within-language/analysis.json", "reports/association_retention/results/margin-function/analysis.json"
    return [M("headline/A2a_verdict", W, "hypotheses", "H-A2a", "verdict"), M("headline/A2a_auroc_margin_INT6", W, "hypotheses", "H-A2a", "auroc_m0_int6"), M("headline/A2b_verdict", W, "hypotheses", "H-A2b", "verdict"),
            M("headline/A2c_verdict", W, "hypotheses", "H-A2c", "verdict"), M("headline/n_established", W, "all", "n_established"), M("headline/A3a_verdict", A, "hypotheses", "H-A3a", "verdict"),
            M("headline/A3a_spearman_INT6", A, "hypotheses", "H-A3a", "spearman_mean", "INT6"), M("headline/A3a_spearman_INT5", A, "hypotheses", "H-A3a", "spearman_mean", "INT5"),
            M("headline/A3a_spearman_INT4", A, "hypotheses", "H-A3a", "spearman_mean", "INT4"), M("headline/A3b_verdict", A, "hypotheses", "H-A3b", "verdict"),
            M("headline/auroc_association_INT6", A, "hypotheses", "H-A3b", "auroc_association", "INT6"), M("headline/auroc_association_INT5", A, "hypotheses", "H-A3b", "auroc_association", "INT5"),
            M("headline/auroc_function_INT6", A, "hypotheses", "H-A3b", "auroc_function", "INT6"), M("headline/auroc_function_INT5", A, "hypotheses", "H-A3b", "auroc_function", "INT5"), M("headline/n_tested", A, "all", "n_tested")]


def _llama_metrics():
    L, R = FV + "llama/analysis.json", FV + "llama_robust/robust.json"
    m = [M("headline/verdict", L, "verdict"), M("headline/n_tested", L, "n_tested"), M("headline/P1_count_positive", L, "P1", "count_positive"), M("headline/P1_required", L, "P1", "required"), M("headline/P1_median_E", L, "P1", "median"),
         M("headline/P1_median_E_lo", L, "P1", "lo"), M("headline/P1_median_E_hi", L, "P1", "hi"), M("headline/P1_met", L, "P1", "met"), M("headline/P2a_met", L, "P2a", "met"), M("headline/P2b_met", L, "P2b", "met"),
         M("headline/P3a_met", L, "P3a", "met"), M("headline/P3b_met", L, "P3b", "met"), M("headline/P3b_count_larger", L, "P3b", "count_larger"), M("headline/P3b_median_ratio", L, "P3b", "median_ratio"),
         M("headline/gemma_median_E_FP", L, "gemma_reference", "median_E_FP"), M("headline/gemma_ratio_int4_over_fp", L, "gemma_reference", "ratio_int4_over_fp_median"),
         M("headline/group_harm_FP", L, "diagnostics", "FP", "harm_group_median"), M("headline/control_harm_FP", L, "diagnostics", "FP", "harm_control_median"),
         M("headline/group_harm_INT4", L, "diagnostics", "INT4", "harm_group_median"), M("headline/control_harm_INT4", L, "diagnostics", "INT4", "harm_control_median")]
    for b in (8, 6, 5, 4):
        m += [M(f"headline/retention_mean_INT{b}", L, "P2", "retention", f"INT{b}"), M(f"headline/retention_median_INT{b}", R, "retention", f"INT{b}", "median_ratio")]
    for s in ("FP", "INT4"):
        m += [M(f"headline/robust_E_median_positive_{s}", R, "summary", s, "E_median_positive"), M(f"headline/robust_exceeds_all_{s}", R, "summary", s, "exceeds_all_count"),
              M(f"headline/robust_outlier_languages_{s}", R, "summary", s, "languages_with_an_outlier_draw")]
    return m


def _fig(d, *names):
    return [f"{d}/{n}_light.png" for n in names]


FAMILIES = [
    {"slug": "functional-lm-ladder", "name": "functional/lm-ladder-F1-F2", "group": "functional", "status": "exploratory", "status_text": "exploratory (frozen setting)", "decisions": ["D-041", "D-050", "D-068"],
     "title": "Loss-task evaluation of the language feature groups across the precision ladder (F1, F2) with the rounding-only floor controls", **GEMMA, "states": ["FP", "INT8", "INT7", "INT6", "INT5", "INT4", "INT16", "INT12", "INT0"],
     "jobs": {"func-eval-a": 34251, "func-eval-b": 34252, "func-eval-c": 34550, "func-eval-d (floor controls)": 35314}, "code_ref": {"file": FREEZE_LM, "path": ["commit"]}, "freeze": FREEZE_LM, "limits": LM_LIMITS,
     "metrics": _ladder_metrics(),
     "tables": [("effects_by_language", FV + "results/eval/effects_by_language.csv"), ("retention", FV + "results/eval/retention.csv"), ("fidelity", FV + "results/eval/fidelity.csv"), ("families", FV + "results/eval/families.csv"),
                ("subset_covered18_retention", FV + "results/eval_subsets/covered18/retention.csv"), ("subset_uncovered9_retention", FV + "results/eval_subsets/uncovered9/retention.csv")],
     "images": _fig(FV + "figures/eval", "fig1_effect_per_language", "fig2_retention", "fig3_group_size", "fig6_sae_fidelity", "fig7_belebele_coverage"),
     "files": [FV + "results/eval/report.md", FV + "results/eval/analysis.json", FV + "results/eval_subsets/covered18/report.md", FV + "results/eval_subsets/uncovered9/report.md"], "history": "lm-ladder"},
    {"slug": "functional-rescue", "name": "functional/rescue-F3", "group": "functional", "status": "exploratory", "status_text": "exploratory", "decisions": ["D-058"],
     "title": "Share of the quantization loss recovered by patching full-precision activations into the quantized model at layer 12 (F3)", **GEMMA, "states": ["INT6", "INT5", "INT4"],
     "jobs": {"func-rescue-a": 35319}, "code_ref": {"file": FREEZE_LM, "path": ["commit"]}, "freeze": FREEZE_LM,
     "limits": "exploratory; layer 12 only; INT6 shares are over the 12 of 27 languages with a clearly positive gap; at most about 3 % of the damage sits in the language groups, 30 to 40 % in the layer-12 residual",
     "metrics": _rescue_metrics(), "tables": [("rescue", FV + "results/rescue/rescue.csv"), ("effects_by_language", FV + "results/rescue/effects_by_language.csv")], "images": _fig(FV + "figures/rescue", "fig4_rescue"),
     "files": [FV + "results/rescue/report.md", FV + "results/rescue/analysis.json"], "history": None},
    {"slug": "functional-f4-association", "name": "functional/f4-association-vs-function", "group": "functional", "status": "exploratory", "status_text": "exploratory", "decisions": ["D-048", "D-049", "D-051"],
     "title": "Does the language-association status of the earlier study predict function at equal native activity (F4, lost against activity-matched retained features) and association retention by rank (A1)", **GEMMA,
     "states": ["INT6", "INT5", "INT4"], "jobs": {"func-f4-INT4": 34795, "func-f4-INT5": 34804, "func-f4-INT6": 35201}, "code_ref": {"file": FREEZE_LM, "path": ["commit"]}, "freeze": FREEZE_LM,
     "limits": "exploratory; no systematic difference between lost and retained features; the INT4 macro mean is carried by one language (Lao); 26 of the 27 languages enter",
     "metrics": _f4_metrics(),
     "tables": [(f"f4_per_language_INT{b}", FV + f"results/f4/INT{b}/f4_per_language.csv") for b in (4, 5, 6)] + [("association_retention_by_precision", FV + "figures/eval/table_retention.csv"), ("assoc_rank_table", FV + "results/assoc_rank/rank_table.csv"),
               ("assoc_language_contrast", FV + "results/assoc_rank/language_contrast.csv"), ("assoc_language_contrast_by_language", FV + "results/assoc_rank/language_contrast_by_language.csv")],
     "images": _fig(FV + "figures/eval", "fig5_association_vs_function", "fig8_association_by_rank", "fig9_f4_INT4", "fig9_f4_INT5", "fig9_f4_INT6"),
     "files": [FV + f"results/f4/INT{b}/{x}" for b in (4, 5, 6) for x in ("report.md", "f4_results.json")] + [FV + "results/assoc_rank/report.md", FV + "f4/f4_matching.md"], "history": None},
    {"slug": "association-margin-a2-a3", "name": "functional/association-margin-A2-A3", "group": "functional", "status": "exploratory", "status_text": "exploratory (hypotheses written before the numbers)", "decisions": ["D-066", "D-067"],
     "title": "Loss of the D-035 language association by margin quintile (A2) and the margin against a change of function (A3)", **GEMMA, "states": ["INT6", "INT5", "INT4"], "jobs": {"f5-a": 35424, "f5-b": 35435},
     "code_ref": None, "freeze": None, "limits": "the margin and the statuses come from the same devtest text (regression to the mean; the rounding-only floor is the reference); stratified sample, 38 of 1,487 features dropped, no function-event contrast at INT4",
     "metrics": _assoc_metrics(), "tables": [("loss_by_margin_quintile", "reports/association_retention/results/within-language/loss_by_margin_quintile.csv"), ("margin_vs_function", "reports/association_retention/results/margin-function/margin_vs_function.csv")],
     "images": ["reports/association_retention/results/within-language/fig_loss_by_margin_light.png", "reports/association_retention/results/margin-function/fig_margin_auroc_light.png"],
     "files": ["reports/association_retention/results/within-language/report.md", "reports/association_retention/results/within-language/analysis.json", "reports/association_retention/results/margin-function/report.md",
               "reports/association_retention/results/margin-function/analysis.json"], "history": None},
    {"slug": "belebele-confirmatory-tmc", "name": "belebele/confirmatory-T-MC", "group": "belebele", "status": "confirmatory", "status_text": "confirmatory (frozen; the only confirmatory family): statement S0 not met, S1 to S5 undefined", "decisions": ["D-047", "D-050"],
     "title": "Belebele reading comprehension with the language groups zeroed against matched controls (statements S0 to S5 fixed in a freeze before scoring)", **GEMMA, "states": ["FP", "INT8", "INT7", "INT6", "INT5", "INT4"],
     "jobs": {"func-mc-cal": 34551, "func-mc-eval-a": 34823, "func-mc-eval-b": 34830}, "code_ref": {"file": FREEZE_MC, "path": ["commit"]}, "freeze": FREEZE_MC,
     "limits": "18 of 27 languages covered (the 9 others show no measurable reading ability of the model), letter format, zero ablation; the INT4 model is at chance; S1 to S5 undefined; accuracies are specific to the frozen prompt",
     "metrics": _mc_metrics(), "tables": [("accuracy", FV + "results/mc_eval/accuracy.csv"), ("effects_by_language", FV + "results/mc_eval/effects_by_language.csv"), ("retention", FV + "results/mc_eval/retention.csv"),
                                         ("statements", FV + "results/mc_eval/statements.csv")],
     "images": _fig(FV + "figures/mc_eval", "fig10_macro_effect", "fig1_effect_per_language") + _fig(FV + "figures/eval", "fig7_belebele_coverage"),
     "files": [FV + "results/mc_eval/report.md", FV + "results/mc_eval/statements.md", FV + "results/mc_eval/statements.json", FV + "results/mc_eval/analysis.json", FV + "belebele_calibration.json", FV + "belebele_calibration.md"], "history": "belebele"},
    {"slug": "belebele-ksens-k-all", "name": "belebele/sensitivity-K-all", "group": "belebele", "status": "exploratory", "status_text": "exploratory sensitivity of the confirmatory family", "decisions": ["D-062"],
     "title": "Belebele with the WHOLE selective group zeroed (K = all, up to 1,719 features) at FP and INT6", **GEMMA, "states": ["FP", "INT6"], "jobs": {"func-mc-ksens": 35505}, "code_ref": {"file": FREEZE_MC, "path": ["commit"]}, "freeze": FREEZE_MC,
     "limits": "exploratory; 18 covered languages; the label K64 in the files stands for the whole selective group; the same null as the frozen K = 64 edit",
     "metrics": [M("headline/macro_effect", FV + "results/mc_ksens/analysis.json", "macro", "FP", "point"), M("headline/macro_effect_lo", FV + "results/mc_ksens/analysis.json", "macro", "FP", "lo"),
                 M("headline/macro_effect_hi", FV + "results/mc_ksens/analysis.json", "macro", "FP", "hi"), M("headline/median_effect", FV + "results/mc_ksens/analysis.json", "macro", "FP", "median"),
                 M("headline/n_positive", FV + "results/mc_ksens/analysis.json", "macro", "FP", "n_positive"), M("headline/macro_effect_INT6", FV + "results/mc_ksens/analysis.json", "macro", "INT6", "point"),
                 M("headline/n_languages", FV + "results/mc_ksens/analysis.json", "covered", "#len")],
     "tables": [("accuracy", FV + "results/mc_ksens/accuracy.csv"), ("effects_by_language", FV + "results/mc_ksens/effects_by_language.csv"), ("retention", FV + "results/mc_ksens/retention.csv")], "images": [],
     "files": [FV + "results/mc_ksens/report.md", FV + "results/mc_ksens/analysis.json", FV + "ksens/ksens_report.md"], "history": None},
    {"slug": "belebele-positive-control", "name": "belebele/positive-control-edit-sensitivity", "group": "belebele", "status": "exploratory", "status_text": "exploratory (instrument check)", "decisions": ["D-069", "D-072"],
     "title": "Positive control of the Belebele edit path: zeroing random feature sets of 256, 1,024 and control-set size lowers the accuracy", **GEMMA, "states": ["FP"],
     "jobs": {"func-mc-pc-all": 36504, "func-mc-pc-1024": 36505, "func-mc-pc-256": 36760}, "code_ref": None, "freeze": None,
     "limits": "the language groups are small (median 192 features): the null excludes large group-specific effects, not tenths of a point; a random set of 256 features lowers the accuracy by 0.7 to 0.8 points",
     "metrics": _posctrl_metrics(), "tables": [(f"accuracy_dose_{d}", FV + f"results/mc_posctrl/dose_{d}/accuracy.csv") for d in ("256", "1024", "all")], "images": [],
     "files": [FV + "results/mc_posctrl/summary.json", FV + "results/mc_posctrl/summary.md", FV + "posctrl/posctrl_report.md", FV + "posctrl/posctrl_spec.json"], "history": None},
    {"slug": "belebele-prompt-check", "name": "belebele/prompt-cross-check", "group": "belebele", "status": "exploratory", "status_text": "exploratory", "decisions": ["D-086"],
     "title": "Frozen Belebele letter prompt against the lm-evaluation-harness prompt (Task 17 part b), 18 covered languages, full precision", **GEMMA, "states": ["FP"], "jobs": {"promptcheck": 39406, "promptcheck-smoke2": 39405},
     "code_ref": "5ddf16e (snapshot repo_5ddf16e)", "freeze": None, "limits": "one alternative prompt, zero-shot, one base model at full precision, 250 items per language; do not compare the absolute Belebele accuracies with published harness numbers",
     "metrics": _promptcheck_metrics(), "tables": [("languages", FV + "promptcheck/languages.csv")], "images": [], "files": [FV + "promptcheck/report.md", FV + "promptcheck/analysis.json"], "history": None},
    {"slug": "f5-main", "name": "f5/feature-function-main", "group": "f5", "status": "exploratory", "status_text": "exploratory (the rule S1 to S3 was fixed before the first number)", "decisions": ["D-053", "D-054", "D-059"],
     "title": "Does a single feature do the same thing after quantization (effect of switching it off on the output logits, full precision against quantized), 1,487 features of all 27 languages", **GEMMA,
     "states": ["INT16", "INT8", "INT6", "INT5", "INT4"], "jobs": {"f5-a": 35424, "f5-b": 35435}, "code_ref": None, "freeze": None,
     "limits": "1,487 features, a sample stratified by role; the floor is the 5th percentile of INT16, so a share 'within noise' cannot exceed about 0.95; the INT4 loss of similarity is generic (a swapped direction changes as much)",
     "metrics": _f5_metrics(), "tables": [("f5_tables", FV + "results/f5/f5_tables.csv")],
     "images": _fig(FV + "figures/f5", "f5_fig1_w1_distribution", "f5_fig2_read_vs_write", "f5_fig3_statements", "f5_fig4_retrieval", "f5_fig5_median_ladder", "f5_fig6_paired"),
     "files": [FV + "results/f5/report.md", FV + "results/f5/statements.md", FV + "results/f5/f5_results.json"], "history": None},
    {"slug": "f5-sensitivity-controls", "name": "f5/sensitivity-and-swap-controls", "group": "f5", "status": "exploratory", "status_text": "exploratory", "decisions": ["D-060", "D-061", "D-063", "D-064"],
     "title": "F5 sensitivity to the number of contexts (M = 8, 32), the direction-swap control, fixed amplitudes and the swap control along the ladder", **GEMMA, "states": ["INT8", "INT6", "INT5", "INT4"],
     "jobs": {"f5-m8": 35473, "f5-m32": 35487, "f5-dir": 35529, "f5-fa": 35587, "f5-fa-dir": 35658, "f5-dir2": 35885}, "code_ref": None, "freeze": None,
     "limits": "404 features, one matched control per feature, no intervals for the differences; the control is new to our knowledge, which the team must confirm (D-065)",
     "metrics": _f5_sens_metrics(), "tables": [("ladder_control_table", FV + "figures/f5/ladder_control_table.csv")],
     "images": _fig(FV + "results/f5_sensitivity/direction_swap", "pair") + _fig(FV + "results/f5_sensitivity/fixed_amplitude", "pair") + _fig(FV + "results/f5_sensitivity/fixed_amplitude_direction_swap", "pair")
     + _fig(FV + "results/f5_sensitivity/direction_swap_int8_6_5", "pair") + _fig(FV + "figures/f5", "ladder_control"),
     "files": [FV + f"results/f5_sensitivity/{v}/sensitivity.{e}" for v in ("m8", "m32", "direction_swap", "fixed_amplitude", "fixed_amplitude_direction_swap", "direction_swap_int8_6_5") for e in ("md", "json")] + [FV + "figures/f5/ladder_control_table.md"],
     "history": None},
    {"slug": "study-g-main", "name": "study-g/main-graded-gain", "group": "study-g", "status": "exploratory", "status_text": "exploratory (pre-registered rules H-G1 to H-G6)", "decisions": ["D-070", "D-071", "D-072", "D-073", "D-074", "D-075", "D-076", "D-077", "D-078"],
     "title": "Study G: dose-response of a language's selective features scaled down and up, FP and INT8 to INT4, 27 languages, 300 sentences, 3 control draws", **GEMMA, "states": LADDER,
     "jobs": {"gain-cal-FP": 36834, "gain-cal-INT4": 38128, "gain-FP": 36925, "gain-8": 37948, "gain-7": 37947, "gain-6": 37693, "gain-5": 37694, "gain-4": 36926}, "code_ref": "7fba511 (snapshot repo_7fba511)", "freeze": None,
     "limits": "devtest sentences examined before; one model, layer, SAE, RTN only; loss on the language's own text; the INT4 repair at fine ratios is a suggestion of size 0.001 nats per token (see extension C)",
     "metrics": _gmain_metrics(),
     "tables": [("curves", FV + "gain_study/results/curves.csv"), ("alpha_star", FV + "gain_study/results/alpha_star.csv"), ("reliability", FV + "gain_study/results/reliability.csv"), ("calibration", FV + "gain_study/calibration/calibration.csv"),
                ("baseline", FV + "gain_study/baseline/baseline.csv")],
     "images": _fig(FV + "gain_study/figures", "fig_g1_language_curves", "fig_g2_macro", "fig_g3_heatmap", "fig_g4_repair"),
     "files": [FV + "gain_study/README.md", FV + "gain_study/results/report.md", FV + "gain_study/results/analysis.json", FV + "gain_study/calibration/ratios.json", FV + "gain_study/calibration/calibration.md",
               FV + "gain_study/baseline/report.md", FV + "gain_study/baseline/baseline.json"], "history": None},
    {"slug": "study-g-ext-c", "name": "study-g/extension-C-repair", "group": "study-g", "status": "exploratory", "status_text": "exploratory", "decisions": ["D-079", "D-082"],
     "title": "Study G extension C: the small INT4 repair (turning the group up by 1.1 or 1.2) on 562 fresh sentences per language with five control draws", **GEMMA, "states": ["FP", "INT6", "INT5", "INT4"],
     "jobs": {"gainc-FP": 38125, "gainc-6": 38126, "gainc-5": 38127, "gainc-4b": 38720}, "code_ref": "ab7790e (snapshot repo_ab7790e)", "freeze": None,
     "limits": "effect size about 0.05 % of the loss (a thousandth to two thousandths of a nat per token); state it as a small, precision-specific effect, not as a method for repairing quantization; gainc-4 (38124) ran on a different GPU and is a hardware sensitivity run only",
     "metrics": _gc_metrics(), "tables": [(k, FV + f"gain_study/extension_c/{k}.csv") for k in ("cells", "bands", "optimum", "reliability")], "images": _fig(FV + "gain_study/figures_extension_c", "fig_gc1_repair"),
     "files": [FV + "gain_study/extension_c/report.md", FV + "gain_study/extension_c/analysis.json"], "history": None},
    {"slug": "study-g-ext-d", "name": "study-g/extension-D-optimal-gain", "group": "study-g", "status": "exploratory", "status_text": "exploratory (post hoc in origin)", "decisions": ["D-083", "D-084"],
     "title": "Study G extension D: is a language's optimal gain at INT4 set by how far its group shrank (rank correlation, split-half reliability)", **GEMMA, "states": ["INT4"], "jobs": {"gainc-4b": 38720}, "code_ref": "ab7790e (snapshot repo_ab7790e)", "freeze": None,
     "limits": "27 points; four-ratio grid; 'not supported' is a failure to find the relation, not proof of its absence; do not state a compensation mechanism",
     "metrics": _gd_metrics(), "tables": [("alpha", FV + "gain_study/extension_d/alpha.csv")], "images": _fig(FV + "gain_study/figures_extension_d", "fig_gd1_alpha"),
     "files": [FV + "gain_study/extension_d/report.md", FV + "gain_study/extension_d/analysis.json"], "history": None},
    {"slug": "study-g-ext-b", "name": "study-g/extension-B-state-defined", "group": "study-g", "status": "exploratory", "status_text": "exploratory", "decisions": ["D-081", "D-085"],
     "title": "Study G extension B: groups selected in the quantized model (INT6, INT4) against the full-precision selection", **GEMMA, "states": ["FP", "INT6", "INT4"],
     "jobs": {"gain-sd-manifests": 38815, "gain-sd-FP": 38817, "gain-sd-6": 38818, "gain-sd-4": 38816, "gain-sd-FPref-4": 38869, "gain-sd-FPref-6": 38870}, "code_ref": "dad2760 (snapshot repo_dad2760)", "freeze": None,
     "limits": "devtest; selection on sentences outside the evaluation sentences; state view without bootstrap stability; a null for the median over languages; state it as: the choice of the full-precision group does not drive the INT4 results",
     "metrics": _gb_metrics(), "tables": [("compare", FV + "gain_study/state_defined/results/compare.csv"), ("overlap", FV + "gain_study/state_defined/overlap.csv"), ("diagnostics", FV + "gain_study/state_defined/diagnostics.csv")],
     "images": _fig(FV + "gain_study/figures_state_defined", "fig_gb1_state_defined"),
     "files": [FV + "gain_study/state_defined/results/report.md", FV + "gain_study/state_defined/results/analysis.json", FV + "gain_study/state_defined/meta.json"], "history": None},
    {"slug": "llama-consistency-check", "name": "llama/consistency-check-3.2-1B", "group": "llama", "status": "exploratory", "status_text": "exploratory (pre-registered predictions and verdict function; verdict: not consistent)", "decisions": ["D-087", "D-089", "D-090"],
     "title": "Llama-3.2-1B consistency check of three Gemma findings: predictions P1 to P3b, verdict, Gemma reference, post hoc robustness tables R1 to R6", "model": "meta-llama/Llama-3.2-1B",
     "sae": "seonglae/Llama-3.2-1B-sae FineWeb TopK (k = 48, 14,336 features) at blocks.12.hook_resid_pre (= blocks.11.hook_resid_post)", "layer": 11, "quantizer": GEMMA["quantizer"], "states": ["FP", "INT8", "INT6", "INT5", "INT4"],
     "jobs": {"llama-tag": 39717, "llama-manifest": 39720, "llama-lm-a": 39721, "llama-lm-b": 39722}, "code_ref": "0cf138e7ebaa0d4bb16047b87a8e488508da150c", "freeze": None,
     "limits": "model family, SAE architecture (TopK), SAE recipe and training data, relative depth and tokenizer all differ from the Gemma setting; this SAE labels 91 % of its supported features language-selective (control pool 898); not a replication; the robustness tables are post hoc and cannot change the verdict",
     "metrics": _llama_metrics(), "tables": [("robust_languages", FV + "llama_robust/languages.csv")], "images": [],
     "files": [FV + "llama/report.md", FV + "llama/analysis.json", FV + "llama_robust/report.md", FV + "llama_robust/robust.json"], "history": "llama"},
]


# ---------------------------------------------------------------- command line

def _git_head(root):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "n/a"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Post-hoc upload of the committed results to wandb (schema v2).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--root", default=str(Path(__file__).resolve().parents[2])); b.add_argument("--out", required=True); b.add_argument("--only", nargs="*"); b.add_argument("--dry-run", action="store_true"); b.add_argument("--commit"); b.add_argument("--allow-dirty", action="store_true")
    u = sub.add_parser("upload"); u.add_argument("--payload", required=True); u.add_argument("--root", required=True); u.add_argument("--only", nargs="*"); u.add_argument("--force-media", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        payload = build_all(a.root, FAMILIES, a.only, a.commit or _git_head(a.root))
        print(dry_run_text(payload))
        dirty = dirty_files(a.root, shipped_files(payload))
        if dirty and not a.allow_dirty:
            raise SpecError(f"files that differ from the committed version (commit or pass --allow-dirty): {dirty}")
        if not a.dry_run:
            write_payload(a.root, payload, a.out)
            print(f"wrote {a.out}/payload.json and files.tar")
        return
    import os
    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb
    payload = json.loads(Path(a.payload).read_text())
    for r in upload(payload, a.root, wandb, a.only, force_media=a.force_media):
        pass


if __name__ == "__main__":
    main()
