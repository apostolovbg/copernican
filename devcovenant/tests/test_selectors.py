"""Tests for the shared selector helpers."""



from devcovenant.base import PolicyCheck
from devcovenant.selectors import SelectorSet, build_watchlists


class DummyPolicy(PolicyCheck):
    """Test double exposing deterministic options."""

    policy_id = "dummy"

    def __init__(self, options):
        super().__init__()
        self._options = options

    def get_option(self, key, default=None):
        return self._options.get(key, default)

    def check(self, context):  # pragma: no cover - not used
        return []


def make_policy(**options):
    """Helper returning a dummy policy configured with options."""
    return DummyPolicy(options)


def test_selector_matches_with_suffix_includes(tmp_path):
    policy = make_policy(include_suffixes=".py,.md")
    selector = SelectorSet.from_policy(policy)
    python_file = tmp_path / "project_lib" / "module.py"
    python_file.parent.mkdir(parents=True)
    python_file.write_text("pass")
    assert selector.matches(python_file, tmp_path)
    txt_file = tmp_path / "README.txt"
    txt_file.write_text("hi")
    assert not selector.matches(txt_file, tmp_path)


def test_selector_excludes_take_precedence(tmp_path):
    policy = make_policy(
        include_globs="project_lib/**/*.py",
        exclude_prefixes="project_lib/vendor",
    )
    selector = SelectorSet.from_policy(policy)
    path = tmp_path / "project_lib" / "vendor" / "pkg.py"
    path.parent.mkdir(parents=True)
    path.write_text("pass")
    assert not selector.matches(path, tmp_path)


def test_force_include_globs_override_excludes(tmp_path):
    policy = make_policy(
        exclude_globs="data/**",
        force_include_globs="data/**/cosmo_parser_*.py",
    )
    selector = SelectorSet.from_policy(policy)
    parser = tmp_path / "data" / "sne" / "cosmo_parser_custom.py"
    parser.parent.mkdir(parents=True)
    parser.write_text("# parser")
    assert selector.matches(parser, tmp_path)
    metadata = tmp_path / "data" / "sne" / "metadata.yml"
    metadata.write_text("info")
    assert not selector.matches(metadata, tmp_path)


def test_selector_without_includes_matches_everything(tmp_path):
    selector = SelectorSet()
    random_file = tmp_path / "docs" / "note.txt"
    random_file.parent.mkdir(parents=True)
    random_file.write_text("ok")
    assert selector.matches(random_file, tmp_path)


def test_build_watchlists_normalizes_paths():
    policy = make_policy(
        watch_files="README.md, docs/guide.md ",
        watch_dirs=["project_lib", "engines\\custom"],
    )
    files, dirs = build_watchlists(policy)
    assert files == ["README.md", "docs/guide.md"]
    assert dirs == ["project_lib", "engines/custom"]


def test_selector_prefixes_handle_relative_and_absolute(tmp_path):
    policy = make_policy(include_prefixes="/docs,notes")
    selector = SelectorSet.from_policy(policy)
    doc = tmp_path / "docs" / "overview.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("overview")
    assert selector.matches(doc, tmp_path)
    nested = tmp_path / "notes" / "todo.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("todo")
    assert selector.matches(nested, tmp_path)
    other = tmp_path / "src" / "main.py"
    other.parent.mkdir(parents=True)
    other.write_text("code")
    assert not selector.matches(other, tmp_path)
