"""Mutation-check harness: apply one textual mutation at a time to a source file, run a test command, report KILLED (command fails), SURVIVED (command passes) or
NOT_APPLICABLE (the pattern does not occur exactly once), and ALWAYS restore the file (try/finally, byte comparison at the end). Built after an aborted ad-hoc battery
left a mutant in phase1_analysis.py (2026-10-18); the suite's own failure at the end caught it, this makes it impossible.

  python3 scripts/g0/mutcheck.py FILE MUTANTS.json -- <test command ...>      MUTANTS.json: [[old, new, label], ...]; exit 1 if any mutant SURVIVED"""
import glob, hashlib, json, shutil, subprocess, sys
from pathlib import Path


def _clear_caches(root):
    for d in glob.glob(str(Path(root) / "**" / "__pycache__"), recursive=True):
        shutil.rmtree(d, ignore_errors=True)


def run_mutants(path, mutants, test_cmd, cwd=None, cache_root=None):
    """-> [{"label", "status"}]; raises RuntimeError if the unmutated file already fails the command; the file is restored whatever happens."""
    path = Path(path); original = path.read_bytes(); text = original.decode("utf-8"); results = []
    try:
        if cache_root:
            _clear_caches(cache_root)
        if subprocess.run(test_cmd, cwd=cwd, capture_output=True).returncode != 0:
            raise RuntimeError("the unmutated source already fails the test command")
        for old, new, label in mutants:
            if text.count(old) != 1:
                results.append({"label": label, "status": "NOT_APPLICABLE"}); continue
            path.write_text(text.replace(old, new))
            if cache_root:
                _clear_caches(cache_root)
            results.append({"label": label, "status": "KILLED" if subprocess.run(test_cmd, cwd=cwd, capture_output=True).returncode != 0 else "SURVIVED"})
    finally:
        path.write_bytes(original)
        if cache_root:
            _clear_caches(cache_root)
    assert path.read_bytes() == original and hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(original).hexdigest()
    return results


def main(argv):
    i = argv.index("--"); path, mut = argv[1], argv[2]; cmd = argv[i + 1:]
    res = run_mutants(path, [tuple(m) for m in json.loads(Path(mut).read_text())], cmd, cache_root=Path(path).parent)
    for r in res:
        print(f"{r['status']:15s} {r['label']}")
    sys.exit(1 if any(r["status"] == "SURVIVED" for r in res) else 0)


if __name__ == "__main__":
    main(sys.argv)
