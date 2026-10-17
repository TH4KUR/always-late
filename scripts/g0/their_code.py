"""Their functions, taken VERBATIM from the vendored pristine script by AST (module-level definitions only, first definition of a name), never retyped.
Nothing else of their script is executed. A frozen manifest (reports/g0/their_functions.json) records file and per-function sha256 so drift is visible.

  python3 scripts/g0/their_code.py freeze   # (re)write the manifest from the vendored pristine file
  python3 scripts/g0/their_code.py verify   # check it

Decision D-004 (Phase 1 is a diff on their code); the functions they define in notebook cells that need globals (DEVICE, model_work, ...) run against a
namespace we inject, so they behave as in their script."""
import ast, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRISTINE = ROOT / "third_party/duan2026/pristine/scripts/03_gemma_phase3_streaming_sweep.py"
MANIFEST = ROOT / "reports/g0/their_functions.json"
THEIR_FUNCTIONS = ["load_gemma_tl_model", "load_sae_compat", "build_tokens_2d", "clone_state_to_cpu", "restore_model_work",
                   "target_tl_weight", "quantize_rtn_per_output_channel_tl", "encode_sae"]


def sha256_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_functions(path, names):
    """{name: {"source", "lineno", "sha256"}} for the first MODULE-LEVEL `def name` in the file; KeyError naming the first one not found."""
    text = Path(path).read_text()
    first = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name not in first:
            first[node.name] = node
    out = {}
    for n in names:
        if n not in first:
            raise KeyError(f"no module-level function {n!r} in {path}")
        src = ast.get_source_segment(text, first[n])
        out[n] = {"source": src, "lineno": first[n].lineno, "sha256": sha256_text(src)}
    return out


def build_namespace(extracted, env):
    """Define the extracted functions in `env` (their globals); no other statement of their script runs. Returns env."""
    for name, e in extracted.items():
        exec(compile(e["source"], f"<their:{name}@{e['lineno']}>", "exec"), env)
    return env


def freeze(extracted):
    return {n: {"lineno": e["lineno"], "sha256": e["sha256"]} for n, e in extracted.items()}


def verify_hashes(extracted, frozen):
    """ValueError naming every function that is missing from the manifest or whose source hash differs."""
    bad = sorted(n for n, e in extracted.items() if n not in frozen or frozen[n]["sha256"] != e["sha256"])
    if bad:
        raise ValueError(f"functions differ from the frozen manifest or are missing in it: {bad}")


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "verify"
    ex = extract_functions(PRISTINE, THEIR_FUNCTIONS)
    if cmd == "freeze":
        MANIFEST.write_text(json.dumps({"file": str(PRISTINE.relative_to(ROOT)), "file_sha256": sha256_file(PRISTINE), "functions": freeze(ex)}, indent=1) + "\n")
        print("wrote", MANIFEST)
    else:
        m = json.loads(MANIFEST.read_text()); verify_hashes(ex, m["functions"])
        assert m["file_sha256"] == sha256_file(PRISTINE), "pristine file differs from the frozen manifest"
        print("ok:", len(ex), "functions match the frozen manifest")


if __name__ == "__main__":
    main(sys.argv)
