"""Upload large run artifacts to a Hugging Face dataset repo (one commit, sha256 manifest, remote size check).
Large artifacts (per-feature CSVs, bootstrap snapshots, token windows) live on HF; the git repo keeps only small summaries and the links.
Usage (token comes from HF_TOKEN_PATH / HF_TOKEN via huggingface_hub; never print it):
  python3 hf_artifacts.py upload --repo karma-skz/always-late-artifacts --spec reports/g0/hf_artifact_spec.json --base /home2/karma/duan-verify
  python3 hf_artifacts.py verify --repo karma-skz/always-late-artifacts --spec ... --base ...
Spec: {"card": "path to README.md for the dataset card", "sources": [{"local": "<rel to base>", "repo": "<prefix>", "include": [globs], "exclude": [globs]}]}"""
import argparse, fnmatch, hashlib, json, sys
from pathlib import Path


def _match(rel, patterns):
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(Path(rel).name, p) for p in patterns)


def collect_files(sources):
    """-> sorted [(local_path, path_in_repo, size)]; a missing source is an error."""
    out = []
    for s in sources:
        local = Path(s["local"])
        if not local.exists():
            raise FileNotFoundError(local)
        if local.is_file():
            out.append((str(local), s["repo"], local.stat().st_size))
            continue
        include, exclude = s.get("include") or ["*"], s.get("exclude") or []
        for f in sorted(local.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(local).as_posix()
            if _match(rel, include) and not _match(rel, exclude):
                out.append((str(f), f"{s['repo']}/{rel}", f.stat().st_size))
    return sorted(out, key=lambda x: x[1])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_text(files):
    return "".join(f"{sha256_file(p)}  {repo}\n" for p, repo, _ in files)


def upload(api, repo_id, files, message, extra_text=None, op_cls=None):
    """One commit containing every file plus the small text files ({path_in_repo: text})."""
    if not files:
        raise ValueError("nothing to upload")
    if op_cls is None:
        from huggingface_hub import CommitOperationAdd as op_cls
    ops = [op_cls(path_in_repo=repo, path_or_fileobj=p) for p, repo, _ in files]
    ops += [op_cls(path_in_repo=name, path_or_fileobj=text.encode("utf-8")) for name, text in (extra_text or {}).items()]
    api.create_commit(repo_id=repo_id, operations=ops, commit_message=message, repo_type="dataset")


def verify_remote(api, repo_id, files):
    remote = {e.path: getattr(e, "size", None) for e in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)}
    missing = [r for _, r, _ in files if r not in remote]
    bad = [r for _, r, s in files if r in remote and remote[r] != s]
    return {"ok": not missing and not bad, "missing": missing, "size_mismatch": bad, "n_files": len(files)}


def _load(spec_path, base):
    spec = json.loads(Path(spec_path).read_text())
    srcs = [{**s, "local": str(Path(base) / s["local"])} for s in spec["sources"]]
    return spec, collect_files(srcs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["upload", "verify", "list"]); ap.add_argument("--repo", required=True); ap.add_argument("--spec", required=True)
    ap.add_argument("--base", default="."); ap.add_argument("--message", default="Upload run artifacts"); ap.add_argument("--card")
    a = ap.parse_args(argv)
    spec, files = _load(a.spec, a.base)
    total = sum(s for _, _, s in files)
    print(f"{len(files)} files, {total / 2**20:.1f} MB")
    if a.cmd == "list":
        for _, r, s in files:
            print(f"{s / 2**20:8.2f} MB  {r}")
        return
    from huggingface_hub import HfApi
    api = HfApi()
    if a.cmd == "upload":
        extra = {"MANIFEST.sha256": manifest_text(files)}
        card = a.card or spec.get("card")
        if card:
            extra["README.md"] = Path(card).read_text()
        upload(api, a.repo, files, a.message, extra_text=extra)
        print("uploaded in one commit")
    v = verify_remote(api, a.repo, files)
    print(json.dumps(v, indent=1))
    sys.exit(0 if v["ok"] else 1)


if __name__ == "__main__":
    main()
