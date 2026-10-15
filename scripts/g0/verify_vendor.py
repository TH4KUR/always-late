"""The whitelist of edits that turns THEIR scripts/03_gemma_phase3_streaming_sweep.py into the file we run, plus verification helpers.

third_party/duan2026/pristine/  byte-identical retained originals; MANIFEST.sha256 lists ALL 36 upstream files (retained or removed).
third_party/duan2026/scripts/   the working copy = pristine + EDITS (nothing else).
CLI:  python scripts/g0/verify_vendor.py --verify                      # check manifest + working copy
      python scripts/g0/verify_vendor.py --build <fresh upstream dir>  # rebuild pristine/, MANIFEST.sha256, working copy, patch, working.sha256
"""
import difflib, hashlib, shutil, sys
from pathlib import Path

TRUNCATE_AT = "# %% Cell 22"   # their cells 22+ call corrected_ppl_sweep / compute_ppl_shifted_manual / restore_work_from_ref, never defined in the file
EDITS = [
    ("    RUN_PRUNING = True\n",
     "    RUN_PRUNING = False  # [G0-EDIT 1] their own flag: their pruning stage is not needed for this study\n"),
    ('DEVICE = "cuda" if torch.cuda.is_available() else "cpu"\n',
     'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"\n'
     'WORK_DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE  # [G0-EDIT 2] second bf16 model on a second card when one exists (11 GB cards); identical to the shipped behaviour on one large GPU\n'),
    ('dtype=torch.bfloat16 if device == "cuda" else torch.float32,',
     'dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,  # [G0-EDIT 3] was: device == "cuda" (would silently give float32 for "cuda:1")'),
    ("model_work, _ = load_gemma_tl_model([MODEL_NAME_USED])",
     "model_work, _ = load_gemma_tl_model([MODEL_NAME_USED], device=WORK_DEVICE)  # [G0-EDIT 4]"),
    ("    model_work.to(DEVICE)\n",
     "    model_work.to(WORK_DEVICE)  # [G0-EDIT 5]\n"),
]
RETAINED = ["LICENSE", "README.md", "scripts/03_gemma_phase3_streaming_sweep.py", "scripts/06_gemma_sliding_window_ppl_check.py",
            "results/tables/phase3_summary_final.csv", "results/tables/gemma_chunked_vs_sliding_ppl_w2048_s512_FULL.csv",
            "results/per_feature/phase5b_gemma_rtn_int6_vs_pruning_per_feature_overlap.csv"]
SCRIPT03 = "scripts/03_gemma_phase3_streaming_sweep.py"


def reconstruct(pristine_text):
    """Apply the whitelist to their pristine script. Raises AssertionError if an anchor does not match exactly once."""
    text = pristine_text.split(TRUNCATE_AT)[0]
    assert pristine_text.count(TRUNCATE_AT) == 1
    for old, new in EDITS:
        assert text.count(old) == 1, f"anchor matched {text.count(old)}x: {old[:60]!r}"
        text = text.replace(old, new)
    return text


def working_matches(working_text, pristine_text):
    try:
        return working_text == reconstruct(pristine_text)
    except AssertionError:
        return False


def sha(b):
    return hashlib.sha256(b).hexdigest()


def tree_hash(entries):
    lines = "".join(f"{e['sha256']}  {e['path']}\n" for e in sorted(entries, key=lambda e: e["path"]))
    return hashlib.sha256(lines.encode()).hexdigest()


def parse_manifest(path):
    entries, tree = [], None
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith("TREE "):
            tree = line.split()[1]; continue
        h, size, status, p = line.split(None, 3)
        entries.append({"sha256": h, "size": int(size), "status": status, "path": p})
    return entries, tree


def check_pristine(vendor):
    """Return a list of problems (empty = every retained file is byte-identical to upstream and every removed file is absent)."""
    vendor = Path(vendor); problems = []
    entries, tree = parse_manifest(vendor / "MANIFEST.sha256")
    if tree_hash(entries) != tree:
        problems.append("manifest tree hash mismatch")
    for e in entries:
        p = vendor / "pristine" / e["path"]
        if e["status"] == "retained":
            if not p.exists() or sha(p.read_bytes()) != e["sha256"]:
                problems.append(f"retained file differs from upstream: {e['path']}")
        elif p.exists():
            problems.append(f"removed file still present: {e['path']}")
    return problems


def build(upstream, vendor):
    upstream, vendor = Path(upstream), Path(vendor)
    files = sorted(str(p.relative_to(upstream)) for p in upstream.rglob("*") if p.is_file())
    entries = [{"sha256": sha((upstream / f).read_bytes()), "size": (upstream / f).stat().st_size,
                "status": "retained" if f in RETAINED else "removed", "path": f} for f in files]
    for sub in ("pristine", "scripts", "patches"):
        shutil.rmtree(vendor / sub, ignore_errors=True)
    for f in RETAINED:
        (vendor / "pristine" / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upstream / f, vendor / "pristine" / f)
    (vendor / "MANIFEST.sha256").write_text(
        "# sha256  size  status  upstream path   (status: retained = byte-identical copy in pristine/; removed = deleted, not needed for the Gemma Phase-3 replication)\n"
        + "".join(f"{e['sha256']}  {e['size']}  {e['status']}  {e['path']}\n" for e in entries) + f"TREE {tree_hash(entries)}\n")
    pristine = (upstream / SCRIPT03).read_text(); working = reconstruct(pristine)
    (vendor / "scripts").mkdir(parents=True, exist_ok=True); (vendor / "patches").mkdir(parents=True, exist_ok=True)
    (vendor / SCRIPT03).write_text(working)
    (vendor / "patches" / "03_working_vs_pristine.diff").write_text("".join(difflib.unified_diff(
        pristine.splitlines(True), working.splitlines(True), "pristine/" + SCRIPT03, SCRIPT03)))
    (vendor / "working.sha256").write_text(f"{sha(working.encode())}  {SCRIPT03}\n")
    return entries


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2] / "third_party" / "duan2026"
    if len(sys.argv) >= 3 and sys.argv[1] == "--build":
        es = build(sys.argv[2], root); print(f"built: {len(es)} upstream files, tree {tree_hash(es)}")
    else:
        probs = check_pristine(root)
        ok = working_matches((root / SCRIPT03).read_text(), (root / "pristine" / SCRIPT03).read_text())
        print("pristine problems:", probs or "none", "| working copy == pristine + whitelist:", ok)
        sys.exit(0 if not probs and ok else 1)
