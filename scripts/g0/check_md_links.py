"""Link and anchor checker for the markdown of the repository (plan Task 9, step 1; stdlib only).

Scope: every tracked markdown file (`git ls-files`; a directory walk when the root is not a git checkout) except `paper/` and `third_party/`. Inline links and images (`[t](target "title")`,
`<target with spaces>`) and reference definitions (`[id]: target`) are checked; links inside fenced or inline code are not links; external schemes (`https:`, `mailto:`, ...) are skipped. A relative
target must exist (file or directory, case-sensitive, percent-decoded; a leading `/` means the repository root). A `#fragment` on a markdown target, or on the same file, must match a heading slug by
the GitHub rule (lower case, punctuation removed except hyphens and underscores, every space becomes a hyphen, duplicates get `-1`, `-2`) or an explicit `<a name=...>` / `id=...` anchor, compared
case-insensitively; a fragment on any other file type (`code.py#L10`) is not checked. Setext headings are not recognised (the repository uses ATX headings only).

    python3 scripts/g0/check_md_links.py [ROOT] [--list-files]        exit code 1 when anything is broken"""
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

EXCLUDED = ("paper", "third_party")
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv"}
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_INLINE = re.compile(r"!?\[[^\]]*\]\((<[^>]*>|[^)\s]*)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\)")
_REFDEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(<[^>]*>|\S+)")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")
_ATX = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_HTML_ANCHOR = re.compile(r"""<[A-Za-z][^>]*\b(?:name|id)\s*=\s*["']([^"']+)["']""")


def strip_fences(text):
    """The text with the lines of fenced code blocks (fences included) blanked, so that line numbers survive."""
    out, fence = [], None
    for line in text.split("\n"):
        m = _FENCE.match(line)
        if fence is None and m:
            fence = m.group(1)[0] * 3; out.append(""); continue
        if fence is not None:
            if line.strip().startswith(fence):
                fence = None
            out.append(""); continue
        out.append(line)
    return "\n".join(out)


def strip_code(text):
    """The text without fenced blocks and without inline code spans (neither contains links)."""
    return "\n".join(re.sub(r"`+[^`]*`+", "", line) for line in strip_fences(text).split("\n"))


def slugify(heading):
    s = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)                  # a link in a heading contributes its text
    s = re.sub(r"\*+", "", s)                                                 # asterisk emphasis (underscore emphasis is not interpreted: underscores stay, as in identifiers)
    s = re.sub(r"\s+#+\s*$", "", s.strip()).lower()                         # a closing sequence of hashes
    s = re.sub(r"[^\w\- ]", "", s)                                           # punctuation goes; the content of inline code stays, without its backticks
    return s.replace(" ", "-")


def heading_slugs(text):
    """Heading slugs in document order, duplicates numbered `-1`, `-2` (headings inside code fences do not count)."""
    seen, out = {}, []
    for line in strip_fences(text).split("\n"):
        m = _ATX.match(line)
        if not m:
            continue
        s = slugify(m.group(2))
        n = seen.get(s, 0); seen[s] = n + 1
        out.append(s if n == 0 else f"{s}-{n}")
    return out


def extract_links(text):
    """[(line number, target)] of the inline links, images and reference definitions outside code; empty targets are dropped."""
    out = []
    for n, line in enumerate(strip_code(text).split("\n"), start=1):
        for m in _INLINE.finditer(line):
            t = m.group(1)
            t = t[1:-1] if t.startswith("<") and t.endswith(">") else t
            if t:
                out.append((n, t))
        m = _REFDEF.match(line)
        if m:
            t = m.group(1)
            out.append((n, t[1:-1] if t.startswith("<") and t.endswith(">") else t))
    return out


def md_files(root):
    """Tracked markdown files below `root` (relative paths), without `paper/` and `third_party/`."""
    root = Path(root)
    try:
        r = subprocess.run(["git", "ls-files", "*.md"], cwd=root, capture_output=True, text=True, check=True)
        files = [l for l in r.stdout.split("\n") if l]
        if not files:
            raise subprocess.CalledProcessError(1, "git")
    except (subprocess.CalledProcessError, FileNotFoundError):
        files = [str(p.relative_to(root)) for p in sorted(root.rglob("*.md")) if not (_SKIP_DIRS & set(p.relative_to(root).parts))]
    return [f for f in files if f.split("/")[0] not in EXCLUDED]


def _anchors(path, cache):
    if path not in cache:
        text = path.read_text(encoding="utf-8", errors="replace")
        cache[path] = {s.lower() for s in heading_slugs(text)} | {a.lower() for a in _HTML_ANCHOR.findall(strip_fences(text))}
    return cache[path]


def check_tree(root, files=None):
    """A list of {file, line, target, reason} for every broken link and anchor."""
    root = Path(root); cache = {}; out = []
    for rel in (files if files is not None else md_files(root)):
        path = root / rel
        for line, target in extract_links(path.read_text(encoding="utf-8", errors="replace")):
            if _SCHEME.match(target):
                continue
            base, _, frag = target.partition("#")
            base = base.partition("?")[0]
            if base:
                dest = (root / base.lstrip("/")) if base.startswith("/") else (path.parent / unquote(base))
                if base.startswith("/"):
                    dest = root / unquote(base).lstrip("/")
                if not dest.exists():
                    out.append({"file": rel, "line": line, "target": target, "reason": "target does not exist"}); continue
            else:
                dest = path
            if frag and dest.is_file() and dest.suffix.lower() == ".md":
                if unquote(frag).lower() not in _anchors(dest, cache):
                    out.append({"file": rel, "line": line, "target": target, "reason": "no heading with this anchor"})
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    list_only = "--list-files" in argv
    args = [a for a in argv if not a.startswith("--")]
    root = Path(args[0] if args else Path(__file__).resolve().parents[2])
    if list_only:
        print("\n".join(md_files(root)))
        return 0
    files = md_files(root); problems = check_tree(root, files)
    for p in problems:
        print(f"{p['file']}:{p['line']}: {p['target']} -- {p['reason']}")
    print(f"{len(files)} markdown files checked, {len(problems)} broken")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
