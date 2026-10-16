"""Bring a wandb project onto the fixed schema of g0_wandb.py: back up the existing runs, rebuild them from the result files
kept in this repo, verify that every run conforms, and delete the old runs ONLY after that (and only the ones listed).

  python3 scripts/g0/wandb_rebuild.py backup  --out reports/g0/wandb_backup_pre_schema.json
  python3 scripts/g0/wandb_rebuild.py rebuild --manifest reports/g0/wandb_manifest.json [--only NAME ...]
  python3 scripts/g0/wandb_rebuild.py verify
  python3 scripts/g0/wandb_rebuild.py delete  --manifest reports/g0/wandb_manifest.json            (dry run)
  python3 scripts/g0/wandb_rebuild.py delete  --manifest reports/g0/wandb_manifest.json --execute  (irreversible)

Uses the wandb library's own credentials (never read or print a key). Network access is only needed by the CLI; the functions take
an `api` object so the tests run offline with fakes."""
import argparse, csv, glob, json, os, shutil, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g0_wandb as g  # noqa: E402

DEFAULT_PATH = "karma-skz-/anlp-project"


def _is_system(key):
    return key.startswith("_") or key.startswith("sys/")


def _summary_keys(run):
    return list(run.summary.keys())


def check_run(run):
    """(bad summary keys, bad config keys) of one run against the fixed schema."""
    bad_s = sorted(k for k in _summary_keys(run) if not _is_system(k) and k not in g.SUMMARY_KEYS and k not in g.TABLE_KEYS)
    bad_c = sorted(k for k in run.config.keys() if not _is_system(k) and k not in g.CONFIG_KEYS)
    return bad_s, bad_c


def verify_schema(api, path=DEFAULT_PATH):
    """Every run's summary keys must be inside SUMMARY_KEYS (or a fixed table name) and its config keys inside CONFIG_KEYS."""
    offenders, n = [], 0
    for run in api.runs(path):
        n += 1
        bad_s, bad_c = check_run(run)
        if bad_s or bad_c:
            offenders.append({"id": run.id, "name": run.name, "bad_summary_keys": bad_s, "bad_config_keys": bad_c})
    return {"ok": not offenders, "n_runs": n, "offending_runs": offenders}


def artifact_files(run):
    return {f.name for a in run.logged_artifacts() for f in a.files()}


def artifact_gaps(old, new, accepted=()):
    """Files in the old run's artifacts that the replacement lacks (the run-id file and wandb's own table dumps never count; `accepted`
    lists files that are deliberately kept elsewhere)."""
    skip = set(accepted) | {"wandb_run_id.txt"}
    return sorted(n for n in artifact_files(old) - artifact_files(new) if n not in skip and not n.endswith(".table.json"))


def backup_runs(api, path=DEFAULT_PATH):
    out = []
    for run in api.runs(path):
        summ = {k: run.summary[k] for k in _summary_keys(run)}
        out.append({"id": run.id, "name": run.name, "state": run.state, "created_at": run.created_at, "job_type": run.job_type,
                    "config": json.loads(json.dumps(dict(run.config), default=str)), "summary": json.loads(json.dumps(summ, default=str))})
    return out


def delete_old(api, old_ids, keep_ids=(), replacements=None, dry_run=False, path=DEFAULT_PATH, accepted_missing=None):
    """Delete the listed non-conforming runs. Refuses (deletes nothing) if any OTHER run does not conform, or, when `replacements`
    ({old_id: new_run_name}) is given, if an old run has no finished conforming replacement. Never deletes a conforming run."""
    runs = list(api.runs(path))
    by_id = {r.id: r for r in runs}
    bad = {o["id"] for o in verify_schema(api, path)["offending_runs"]}
    targets = [i for i in old_ids if i in by_id and i in bad and i not in set(keep_ids)]
    blocked = set(bad) - set(old_ids) - set(keep_ids)
    if replacements:
        good_names = {r.name for r in runs if r.id not in bad and r.state == "finished"}
        blocked |= {i for i in targets if replacements.get(i) not in good_names}
        by_name = {r.name: r for r in runs if r.id not in bad and r.state == "finished"}
        for i in targets:
            new = by_name.get(replacements.get(i))
            if new is not None and artifact_gaps(by_id[i], new, (accepted_missing or {}).get(i, ())):
                blocked.add(i)
    if blocked:
        return {"deleted": [], "would_delete": [], "blocked_by": sorted(blocked)}
    if dry_run:
        return {"deleted": [], "would_delete": targets, "blocked_by": []}
    deleted = []
    for i in targets:
        by_id[i].delete(delete_artifacts=True)
        deleted.append(i)
    return {"deleted": deleted, "would_delete": targets, "blocked_by": []}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def read_summary_csv(path):
    with open(path, newline="") as f:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]


def rebuild_run(entry, root, project, entity, logger_cls=None, **start_kw):
    """Recreate one run under the fixed schema from the result files named in the manifest entry; uploads small files only."""
    logger_cls = logger_cls or g.G0Logger
    root = Path(root)
    csv_path = root / entry["summary_csv"]
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows = read_summary_csv(csv_path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy2(csv_path, tmp / csv_path.name)
        for pattern in entry.get("files", []):
            for f in sorted(glob.glob(str(root / pattern))):
                f = Path(f)
                if f.is_file() and not f.name.startswith("snap_") and f.suffix != ".pt":
                    shutil.copy2(f, tmp / f.name)
        cfg = {**entry["config"], "source": "rebuilt-from-files"}
        lg = logger_cls.start(project=project, entity=entity, name=entry["name"], config=cfg, out_dir=tmp,
                              tags=list(entry.get("tags", [])) + ["rebuilt"], **start_kw)
        for r in rows:
            lg.log_condition(r)
        lg.log_gates(rows, tmp)
        lg.log_table(rows)
        if entry.get("comparison_json"):
            comp = json.loads((root / entry["comparison_json"]).read_text())["comparison"]["rows"]
            lg.log_table(comp, key="comparison_vs_published")
        if entry.get("summary_extra"):
            lg.set_summary(entry["summary_extra"])
        lg.finish(tmp)
    return {"name": entry["name"], "rows": len(rows)}


def _api():
    import wandb
    return wandb.Api(timeout=60)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backup", "rebuild", "verify", "delete"])
    ap.add_argument("--path", default=DEFAULT_PATH); ap.add_argument("--out"); ap.add_argument("--manifest")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2])); ap.add_argument("--only", nargs="*")
    ap.add_argument("--keep", nargs="*", default=[]); ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    entity, project = a.path.split("/")
    if a.cmd == "backup":
        b = backup_runs(_api(), a.path)
        Path(a.out).write_text(json.dumps(b, indent=1)); print(f"backed up {len(b)} runs to {a.out}")
    elif a.cmd == "rebuild":
        manifest = json.loads(Path(a.manifest).read_text())
        for e in manifest["runs"]:
            if a.only and e["name"] not in a.only:
                continue
            print("rebuilt:", rebuild_run(e, a.root, project, entity), flush=True)
    elif a.cmd == "verify":
        v = verify_schema(_api(), a.path)
        print(json.dumps({"ok": v["ok"], "n_runs": v["n_runs"], "offending": [(o["id"], o["name"], o["bad_summary_keys"][:4], o["bad_config_keys"][:4]) for o in v["offending_runs"]]}, indent=1))
        sys.exit(0 if v["ok"] else 1)
    elif a.cmd == "delete":
        manifest = json.loads(Path(a.manifest).read_text())
        old = [e["old_run_id"] for e in manifest["runs"] if e.get("old_run_id")]
        repl = {e["old_run_id"]: e["name"] for e in manifest["runs"] if e.get("old_run_id")}
        acc = {e["old_run_id"]: e.get("accepted_missing", []) for e in manifest["runs"] if e.get("old_run_id")}
        out = delete_old(_api(), old, keep_ids=a.keep, replacements=repl, dry_run=not a.execute, path=a.path, accepted_missing=acc)
        print(json.dumps(out, indent=1))
        sys.exit(0 if not out["blocked_by"] else 1)


if __name__ == "__main__":
    main()
