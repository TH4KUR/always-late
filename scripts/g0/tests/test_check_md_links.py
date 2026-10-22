import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_md_links as cl


def tree(files):
    root = Path(tempfile.mkdtemp())
    for rel, text in files.items():
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    return root


def problems(files):
    root = tree(files)
    return [(p["file"], p["line"], p["target"], p["reason"]) for p in cl.check_tree(root)]


class Slugs(unittest.TestCase):
    def test_github_rule(self):
        self.assertEqual(cl.slugify("Hello World"), "hello-world")
        self.assertEqual(cl.slugify("Q16. Which follow-up?  [HIGH, now]"), "q16-which-follow-up--high-now")      # punctuation removed, every space becomes a hyphen
        self.assertEqual(cl.slugify("The `phase1_run.py` file"), "the-phase1_runpy-file")                        # backticks and the dot are punctuation, the underscore stays
        self.assertEqual(cl.slugify("**Bold** and *emphasis*"), "bold-and-emphasis")
        self.assertEqual(cl.slugify("A [link](http://x.org/y) here"), "a-link-here")
        self.assertEqual(cl.slugify("Café ÜBER"), "café-über")
        self.assertEqual(cl.slugify("Trailing hashes ##"), "trailing-hashes")

    def test_duplicates_get_a_numeric_suffix_in_document_order(self):
        self.assertEqual(cl.heading_slugs("# Notes\n\n## Notes\n\n### Notes\n\n## Other\n"), ["notes", "notes-1", "notes-2", "other"])

    def test_headings_in_code_fences_do_not_count(self):
        self.assertEqual(cl.heading_slugs("# Real\n\n```\n# Fake\n```\n"), ["real"])


class Extraction(unittest.TestCase):
    def test_inline_links_images_titles_and_angle_brackets(self):
        text = 'a [x](a.md) b ![img](pics/p.png) c [t](b.md "a title") d [s](<dir with space/c.md>)\n[ref]: refs/r.md\n'
        got = [(ln, t) for ln, t in cl.extract_links(text)]
        self.assertEqual(got, [(1, "a.md"), (1, "pics/p.png"), (1, "b.md"), (1, "dir with space/c.md"), (2, "refs/r.md")])

    def test_code_is_not_links_and_line_numbers_survive(self):
        text = "line1\n```\n[no](nothing.md)\n```\nline5 `[no](inline.md)` and [yes](yes.md)\n"
        self.assertEqual(cl.extract_links(text), [(5, "yes.md")])

    def test_nothing_but_the_target_is_returned_for_empty_and_anchor_only_links(self):
        self.assertEqual(cl.extract_links("[a](#frag) [b]()\n"), [(1, "#frag")])


class Files(unittest.TestCase):
    def test_existing_targets_directories_and_parent_links_are_fine(self):
        self.assertEqual(problems({"a.md": "[b](sub/b.md) [d](sub)\n", "sub/b.md": "[up](../a.md) [self](b.md)\n"}), [])

    def test_a_missing_file_is_reported_with_its_line(self):
        self.assertEqual(problems({"a.md": "ok\n\n[gone](nope.md)\n"}), [("a.md", 3, "nope.md", "target does not exist")])

    def test_percent_encoding_is_decoded_and_paths_are_case_sensitive(self):
        self.assertEqual(problems({"a.md": "[s](my%20file.md) [c](Other.md)\n", "my file.md": "x\n", "other.md": "y\n"}), [("a.md", 1, "Other.md", "target does not exist")])

    def test_a_link_is_resolved_against_the_directory_of_its_file(self):
        self.assertEqual(problems({"docs/a.md": "[x](b.md)\n", "b.md": "root\n"}), [("docs/a.md", 1, "b.md", "target does not exist")])
        self.assertEqual(problems({"docs/a.md": "[x](../b.md)\n", "b.md": "root\n"}), [])

    def test_a_root_relative_target_is_resolved_against_the_repository_root(self):
        self.assertEqual(problems({"docs/a.md": "[x](/b.md)\n", "b.md": "root\n"}), [])

    def test_external_and_mail_links_are_not_checked(self):
        self.assertEqual(problems({"a.md": "[x](https://example.org/none) [m](mailto:a@b.c) [f](ftp://h/x)\n"}), [])

    def test_reference_definitions_and_images_are_checked(self):
        got = problems({"a.md": "![i](pics/missing.png)\n\n[r]: refs/missing.md\n"})
        self.assertEqual({(f, ln, t) for f, ln, t, _ in got}, {("a.md", 1, "pics/missing.png"), ("a.md", 3, "refs/missing.md")})

    def test_links_in_code_are_ignored_by_the_tree_check(self):
        self.assertEqual(problems({"a.md": "```\n[x](nope.md)\n```\n`[y](nope2.md)`\n"}), [])

    def test_paper_and_third_party_are_not_scanned(self):
        self.assertEqual(problems({"paper/p.md": "[x](nope.md)\n", "third_party/t/readme.md": "[x](nope.md)\n", "a.md": "fine\n"}), [])


class Anchors(unittest.TestCase):
    def test_same_file_and_cross_file_anchors(self):
        files = {"a.md": "# Top\n\n[ok](#top) [bad](#nothing) [x](b.md#second-part) [y](b.md#missing)\n", "b.md": "# B\n\n## Second part\n"}
        got = problems(files)
        self.assertEqual({(t, r) for _, _, t, r in got}, {("#nothing", "no heading with this anchor"), ("b.md#missing", "no heading with this anchor")})

    def test_anchors_use_the_github_slug_including_duplicates_and_punctuation(self):
        files = {"a.md": "[1](#q16-which-follow-up) [2](#notes-1) [3](#the-phase1_runpy-file)\n\n## Q16. Which follow-up?\n\n## Notes\n\n## Notes\n\n## The `phase1_run.py` file\n"}
        self.assertEqual(problems(files), [])

    def test_the_anchor_match_ignores_case_and_decodes_percent_signs(self):
        self.assertEqual(problems({"a.md": "[x](#Some%20Thing) [y](#SOME-HEADING)\n\n## Some heading\n"})[0][2:], ("#Some%20Thing", "no heading with this anchor"))
        self.assertEqual(len(problems({"a.md": "[y](#SOME-HEADING)\n\n## Some heading\n"})), 0)

    def test_explicit_html_anchors_count(self):
        self.assertEqual(problems({"a.md": '[x](#custom)\n\n<a name="custom"></a>\n[y](#byid)\n<div id="byid"></div>\n'}), [])

    def test_fragments_on_other_file_types_are_not_checked(self):
        self.assertEqual(problems({"a.md": "[x](code.py#L10) [y](data.csv#whatever)\n", "code.py": "x\n", "data.csv": "a\n"}), [])

    def test_a_fragment_on_a_missing_markdown_file_reports_the_file_not_the_anchor(self):
        self.assertEqual(problems({"a.md": "[x](gone.md#frag)\n"}), [("a.md", 1, "gone.md#frag", "target does not exist")])


class Cli(unittest.TestCase):
    def test_exit_code_and_output_format(self):
        import io, contextlib
        root = tree({"a.md": "[gone](nope.md)\n"}); out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cl.main([str(root)])
        self.assertEqual(code, 1); self.assertIn("a.md:1: nope.md", out.getvalue()); self.assertIn("1 broken", out.getvalue())
        clean = tree({"a.md": "fine\n"}); out2 = io.StringIO()
        with contextlib.redirect_stdout(out2):
            self.assertEqual(cl.main([str(clean)]), 0)
        self.assertIn("0 broken", out2.getvalue())

    def test_list_files_prints_the_scope(self):
        import io, contextlib
        root = tree({"a.md": "x\n", "sub/b.md": "x\n", "paper/p.md": "x\n"}); out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cl.main([str(root), "--list-files"])
        self.assertEqual(sorted(out.getvalue().split()), ["a.md", "sub/b.md"])


if __name__ == "__main__":
    unittest.main()
