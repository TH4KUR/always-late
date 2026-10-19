"""N13 (decision D-026): a clean-room reproduction of every committed Phase 1 result file. The committed scripts are run from the cached run files (the Ada outputs fetched with
fetch_phase1.sh) at 4 BLAS threads into a fresh directory outside the repository, and each regenerated file is compared with the committed one.
Comparison rule (fixed before the run): a regenerated file is *reproduced* if it equals the committed one byte for byte; JSON files are compared after dropping keys that hold
filesystem paths (`run`, at any depth), numbers EXACTLY (no tolerance); the verdict is REPRODUCED only if every compared file is reproduced. Files that are Ada outputs (per_language.csv,
meta.json, summary.csv) are inputs here: the committed copies are compared with the cached fetched copies instead of being regenerated. Stdlib only.
  python3 scripts/g0/reproduce_phase1.py [--dry-run] [--streams mask/eq none/eq ...] [--work DIR] [--report reports/phase1_reproduction.md]"""
import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STREAMS = (("mask", "eq"), ("mask", "all"), ("none", "eq"), ("none", "all"))
DEFAULT_RUN_ROOT = Path.home() / ".cache" / "anlp-always-late" / "phase1_runs"
DEFAULT_WORK = Path.home() / ".cache" / "anlp-always-late" / "reproduction"
DEFAULT_PYTHON = str(Path.home() / ".venvs" / "anlp-torch" / "bin" / "python")


def normalise_json(obj):
    if isinstance(obj, dict):
        return {k: normalise_json(v) for k, v in obj.items() if k != "run"}
    if isinstance(obj, list):
        return [normalise_json(v) for v in obj]
    return obj


def _same_scalar(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def compare_json(committed, regenerated, path=""):
    """Differences between two JSON documents as [{path, committed, regenerated}], exact; `run` keys (filesystem paths) are ignored."""
    return _diff(normalise_json(committed), normalise_json(regenerated), path)


def _diff(a, b, path):
    here = path or "(root)"
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in a:
            out += _diff(a[k], b[k], f"{path}/{k}".lstrip("/")) if k in b else [{"path": f"{path}/{k}".lstrip("/"), "committed": a[k], "regenerated": "(missing)"}]
        for k in b:
            if k not in a:
                out.append({"path": f"{path}/{k}".lstrip("/"), "committed": "(missing)", "regenerated": b[k]})
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [{"path": here, "committed": f"list of {len(a)}", "regenerated": f"list of {len(b)}"}]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff(x, y, f"{path}/{i}".lstrip("/"))
        return out
    return [] if _same_scalar(a, b) else [{"path": here, "committed": a, "regenerated": b}]


def compare_files(regenerated, committed):
    regenerated, committed = Path(regenerated), Path(committed)
    if not regenerated.exists():
        return {"status": "missing regenerated", "detail": str(regenerated)}
    if not committed.exists():
        return {"status": "missing committed", "detail": str(committed)}
    if committed.suffix == ".json":
        d = compare_json(json.loads(committed.read_text()), json.loads(regenerated.read_text()))
        if not d:
            return {"status": "identical", "detail": ""}
        nums = [abs(x["committed"] - x["regenerated"]) for x in d if isinstance(x["committed"], (int, float)) and isinstance(x["regenerated"], (int, float))
                and not isinstance(x["committed"], bool) and not isinstance(x["regenerated"], bool)]
        tail = f"; largest absolute numeric difference {max(nums):.6g}" if nums else ""
        return {"status": "differs", "detail": f"{len(d)} differing values; first at {d[0]['path']}{tail}"}
    a, b = committed.read_bytes(), regenerated.read_bytes()
    if a == b:
        return {"status": "identical", "detail": ""}
    la, lb = a.decode("utf-8", "replace").split("\n"), b.decode("utf-8", "replace").split("\n")
    for i, (x, y) in enumerate(zip(la, lb), 1):
        if x != y:
            return {"status": "differs", "detail": f"first difference at line {i}"}
    return {"status": "differs", "detail": f"same leading lines, {len(la)} against {len(lb)} lines (final newline or trailing text)"}


@dataclass
class Step:
    name: str
    argv: list
    outputs: list = field(default_factory=list)       # (path relative to the stream's work directory, path relative to the committed stream directory)


def steps_for(regime, stream, run_dir, work, repo, python):
    g = Path(repo) / "scripts" / "g0"; w = Path(work) / regime / stream; c = Path(repo) / "reports" / "phase1" / regime / stream; stats = Path(repo) / "reports" / "phase1" / "flores_text_stats.json"
    run_dir = Path(run_dir); s = str; P = lambda name: s(g / name)
    steps = [Step("analysis", [python, "-B", P("phase1_analysis.py"), "--run", s(run_dir), "--out", s(w / "analysis"), "--seed", "0"], [("analysis/analysis.json", "analysis.json"), ("analysis/analysis.md", "analysis.md")]),
             Step("thresholds", [python, "-B", P("phase1_threshold.py"), "--run", s(run_dir), "--out", s(w / "thresholds"), "--taus", "2", "3", "5", "--seed", "0"],
                  [("thresholds/thresholds.json", "thresholds/thresholds.json"), ("thresholds/thresholds.md", "thresholds/thresholds.md")] + [(f"thresholds/tau_{t}/analysis.md", f"thresholds/tau_{t}/analysis.md") for t in ("2.0", "3.0", "5.0")]),
             Step("magnitude", [python, "-B", P("phase1_magnitude.py"), "--run", s(run_dir), "--out", s(w / "magnitude"), "--taus", "3", "5", "--bins", "3", "5", "10", "--seed", "0", "--rate-only-json", s(w / "thresholds" / "thresholds.json")],
                  [("magnitude/magnitude.json", "magnitude/magnitude.json"), ("magnitude/magnitude.md", "magnitude/magnitude.md")])]
    if regime == "mask":
        steps.append(Step("mixed_variants", [python, "-B", P("phase1_mixed_variants.py"), "--run", s(run_dir), "--out", s(w / "mixed_variants"), "--tau", "5", "--seed", "0", "--analysis-json", s(w / "analysis" / "analysis.json")],
                          [("mixed_variants/mixed_variants.json", "mixed_variants/mixed_variants.json"), ("mixed_variants/mixed_variants.md", "mixed_variants/mixed_variants.md")]))
    steps += [Step("language_level", [python, "-B", P("phase1_language_level.py"), "--per-language", s(c / "per_language.csv"), "--meta", s(c / "meta.json"), "--stats", s(stats), "--survival-pt", s(run_dir / "survival.pt"),
                                      "--out", s(w / "language_level"), "--seed", "0"], [("language_level/language_level.json", "language_level/language_level.json"), ("language_level/language_level.md", "language_level/language_level.md")]),
              Step("behavior", [python, "-B", P("phase1_behavior.py"), "--run", s(run_dir), "--stats", s(stats), "--out", s(w / "behavior")], [("behavior/behavior.json", "behavior.json"), ("behavior/behavior.md", "behavior.md")]),
              Step("margin", [python, "-B", P("phase1_margin.py"), "--run", s(run_dir), "--stream-dir", s(c), "--out", s(w / "margin"), "--seed", "0"],
                   [(f"margin/{f}", f"margin/{f}") for f in ("margin_summary.json", "margin_summary.md", "magnitude/magnitude.md", "mixed_variants/mixed_variants.md", "language_level/language_level.md")])]
    return steps


def input_pairs(regime, stream, run_dir, repo):
    c = Path(repo) / "reports" / "phase1" / regime / stream
    return [(Path(run_dir) / f, c / f) for f in ("per_language.csv", "meta.json", "summary.csv")]


def thread_env(base=None):
    e = dict(os.environ if base is None else base); e["OMP_NUM_THREADS"] = "4"; e["MKL_NUM_THREADS"] = "4"; return e


def run_step(step, env, runner=subprocess.run):
    r = runner(step.argv, env=env, capture_output=True, text=True)
    return {"ok": r.returncode == 0, "returncode": r.returncode, "stderr_tail": "\n".join((r.stderr or "").strip().splitlines()[-3:])}


def reproduce_stream(regime, stream, run_root, work_root, repo, python, runner=subprocess.run):
    """Rows {stream, step, file, status, detail} for the inputs and every output of every step of one stream."""
    name = f"{regime}/{stream}"; run_dir = Path(run_root) / f"p1-{regime}" / stream; work = Path(work_root); w = work / regime / stream; c = Path(repo) / "reports" / "phase1" / regime / stream; rows = []
    for cached, committed in input_pairs(regime, stream, run_dir, repo):
        rows.append({"stream": name, "step": "inputs", "file": committed.name, **compare_files(cached, committed)})
    env = thread_env()
    for step in steps_for(regime, stream, run_dir, work, repo, python):
        res = run_step(step, env, runner)
        if not res["ok"]:
            rows.append({"stream": name, "step": step.name, "file": "(step)", "status": "step failed", "detail": f"exit {res['returncode']}: {res['stderr_tail']}"})
        for reg, com in step.outputs:
            rows.append({"stream": name, "step": step.name, "file": com, **compare_files(w / reg, c / com)})
    return rows


def verdict(rows):
    return "REPRODUCED" if rows and all(r["status"] == "identical" for r in rows) else "NOT REPRODUCED"


def render_report(rows, meta):
    ok = sum(r["status"] == "identical" for r in rows); L = [f"# Clean-room reproduction of the Phase 1 result files (decision D-026)", "",
         f"Verdict: **{verdict(rows)}** ({ok} of {len(rows)} compared files identical).", "",
         f"Every committed result file under `reports/phase1/` was regenerated from the cached run files (the Ada outputs fetched with `fetch_phase1.sh`) by the committed scripts at commit `{meta['commit']}`, in a fresh directory outside the repository, "
         f"with `OMP_NUM_THREADS={meta['threads']}` and `MKL_NUM_THREADS={meta['threads']}` (A-036: other thread counts move the mixed-model coefficients at about 1e-10), on {meta['date']}, Python/numpy/scipy/torch {meta['python']}. "
         "**Comparison rule (fixed before the run):** a regenerated file is reproduced if it equals the committed one byte for byte; JSON files are compared after dropping path-valued `run` keys, numbers exactly; "
         "the committed `per_language.csv`, `meta.json` and `summary.csv` are Ada outputs and are compared with the cached fetched copies instead of being regenerated. The generated summary is checked with `phase1_summary.py --check`.", "",
         "| stream | compared | identical |", "|---|---|---|"]
    for s in sorted({r["stream"] for r in rows}):
        sub = [r for r in rows if r["stream"] == s]; L.append(f"| {s} | {len(sub)} | {sum(r['status'] == 'identical' for r in sub)} |")
    bad = [r for r in rows if r["status"] != "identical"]
    if bad:
        L += ["", "## Differences", "", "| stream | step | file | status | detail |", "|---|---|---|---|---|"] + [f"| {r['stream']} | {r['step']} | {r['file']} | {r['status']} | {r['detail']} |" for r in bad]
    return "\n".join(L) + "\n"


def _fmt(argv, env_note="OMP_NUM_THREADS=4 MKL_NUM_THREADS=4"):
    return env_note + " " + " ".join(str(x) for x in argv)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT)); ap.add_argument("--work", default=str(DEFAULT_WORK)); ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--report", default=str(REPO / "reports" / "phase1_reproduction.md")); ap.add_argument("--streams", nargs="*", default=[f"{r}/{s}" for r, s in STREAMS]); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repo", default=str(REPO)); a = ap.parse_args(argv); streams = [tuple(x.split("/")) for x in a.streams]
    if a.dry_run:
        for regime, stream in streams:
            for step in steps_for(regime, stream, Path(a.run_root) / f"p1-{regime}" / stream, a.work, a.repo, a.python):
                print(_fmt(step.argv))
        print(f"{a.python} -B {Path(a.repo) / 'scripts/g0/phase1_summary.py'} --check")
        return 0
    with ThreadPoolExecutor(max_workers=len(streams)) as ex:
        results = list(ex.map(lambda rs: reproduce_stream(rs[0], rs[1], a.run_root, a.work, a.repo, a.python), streams))
    rows = [r for rr in results for r in rr]
    chk = subprocess.run([a.python, "-B", str(Path(a.repo) / "scripts" / "g0" / "phase1_summary.py"), "--check"], capture_output=True, text=True)
    rows.append({"stream": "all", "step": "summary", "file": "reports/phase1_summary.md", "status": "identical" if chk.returncode == 0 else "differs", "detail": (chk.stdout + chk.stderr).strip()})
    try:
        commit = subprocess.run(["git", "-C", a.repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
        py = subprocess.run([a.python, "-c", "import sys, numpy, scipy, torch; print(sys.version.split()[0], numpy.__version__, scipy.__version__, torch.__version__)"], capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        commit, py = "unknown", "unknown"
    Path(a.report).write_text(render_report(rows, {"commit": commit, "threads": 4, "date": date.today().isoformat(), "python": py}))
    print(verdict(rows), f"({sum(r['status'] == 'identical' for r in rows)} of {len(rows)} identical); report:", a.report); return 0 if verdict(rows) == "REPRODUCED" else 1


if __name__ == "__main__":
    sys.exit(main())
