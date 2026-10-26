"""The checkpoints folder may have subfolders.

A library gets tidied sooner or later -- 4b/ and 9b/, by author, by family. The listing
used to be a single os.listdir, so tidying made every model vanish from the app, and the
presets that named them broke. Both halves are pinned here: the walk, and the lookup by
bare file name for everything written before the tidy-up.

Neither GPU nor model: the files are empty, only the paths are checked.

Run:  .venv/Scripts/python tests/test_checkpoints_subfolders.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P  # noqa: E402


class _Lib:
    def __init__(self):
        self.main = tempfile.mkdtemp()
        self.extra = tempfile.mkdtemp()
        self.loras = tempfile.mkdtemp()

    def __enter__(self):
        self.old = (P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR,
                    P.LORAS_DIR, list(P.LORAS_EXTRA_DIRS))
        P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR = self.main, self.extra
        P.LORAS_DIR, P.LORAS_EXTRA_DIRS = self.loras, []
        return self

    def __exit__(self, *exc):
        (P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR,
         P.LORAS_DIR, P.LORAS_EXTRA_DIRS) = self.old
        for d in (self.main, self.extra, self.loras):
            shutil.rmtree(d, ignore_errors=True)
        return False

    def put(self, where, rel):
        p = os.path.join(where, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"x")
        return p


def test_the_walk_reaches_the_subfolders():
    with _Lib() as lib:
        lib.put(lib.main, "flat.safetensors")
        lib.put(lib.main, "9b/deep.safetensors")
        lib.put(lib.main, "4b/other/deeper.gguf")
        lib.put(lib.main, "9b/notes.txt")                  # not a model
        assert P._walk_checkpoints(lib.main) == [
            "4b/other/deeper.gguf", "9b/deep.safetensors", "flat.safetensors"]


def test_the_index_and_cache_folders_are_left_out():
    """The Asset Browser writes its thumbnails next to the models."""
    with _Lib() as lib:
        lib.put(lib.main, "_index/thumbs/x.safetensors")
        lib.put(lib.main, ".cache/y.safetensors")
        lib.put(lib.main, ".hidden/z.safetensors")            # a hidden folder
        lib.put(lib.main, "9b/real.safetensors")
        assert P._walk_checkpoints(lib.main) == ["9b/real.safetensors"]


def test_a_relative_path_resolves():
    with _Lib() as lib:
        real = lib.put(lib.main, "9b/model.safetensors")
        got = P.resolve_checkpoint("9b/model.safetensors")
        assert os.path.normcase(got) == os.path.normcase(real), got


def test_a_bare_name_written_before_the_tidy_up_still_resolves():
    """The point of the whole thing: a preset, a queued job or an image's metadata holds
    the name the model had, not the folder it ended up in."""
    with _Lib() as lib:
        real = lib.put(lib.main, "9b/model.safetensors")
        got = P.resolve_checkpoint("model.safetensors")
        assert os.path.normcase(got) == os.path.normcase(real), got
        assert os.path.isfile(got)


def test_the_extra_folder_is_searched_too():
    with _Lib() as lib:
        real = lib.put(lib.extra, "4b/elsewhere.safetensors")
        got = P.resolve_checkpoint("elsewhere.safetensors")
        assert os.path.normcase(got) == os.path.normcase(real), got


def test_a_name_found_nowhere_falls_back_to_the_main_folder():
    with _Lib() as lib:
        got = P.resolve_checkpoint("ghost.safetensors")
        assert os.path.normcase(got) == os.path.normcase(
            os.path.join(lib.main, "ghost.safetensors")), got


def test_an_absolute_path_is_left_alone():
    with _Lib() as lib:
        p = lib.put(lib.main, "9b/model.safetensors")
        assert P.resolve_checkpoint(p) == p


def _same(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def test_the_windows_form_of_an_entry_resolves_too():
    with _Lib() as lib:
        real = lib.put(lib.extra, "9b/model.safetensors")
        assert _same(P.resolve_checkpoint("9b" + chr(92) + "model.safetensors"), real)
        assert _same(P.resolve_checkpoint("MODEL.safetensors"), real)      # case-insensitive


def test_a_stale_relative_entry_is_found_by_its_file_name():
    """The library was re-filed: 'old/base.safetensors' now lives under '9b/'."""
    with _Lib() as d:
        real = d.put(d.main, "9b/base.safetensors")
        assert _same(P.resolve_checkpoint("old/base.safetensors"), real)


def test_the_catalogue_name_of_an_absolute_path():
    """The Asset Browser focus: the models catalogue is indexed by the relative path."""
    with _Lib() as lib:
        real = lib.put(lib.extra, "4b/turbo.safetensors")
        assert P.checkpoint_relname(real) == "4b/turbo.safetensors"
        assert P.checkpoint_relname("4b/turbo.safetensors") == "4b/turbo.safetensors"
        assert P.checkpoint_relname(os.path.join(tempfile.gettempdir(), "x", "out.safetensors")) \
            == "out.safetensors"


def test_a_lora_in_a_subfolder_is_listed_and_resolved():
    with _Lib() as lib:
        real = lib.put(lib.loras, "Style/noir.safetensors")
        lib.put(lib.loras, "root.safetensors")
        lib.put(lib.loras, ".cache/junk.safetensors")
        assert P.list_loras() == ["Style/noir.safetensors", "root.safetensors"], P.list_loras()
        assert _same(P.resolve_lora_path("Style/noir.safetensors"), real)
        assert _same(P.resolve_lora_path("Style" + chr(92) + "noir.safetensors"), real)


def test_the_ui_focus_and_presets_accept_subfolder_entries():
    import cz_ui
    with _Lib() as lib:
        lib.put(lib.main, "9b/base.safetensors")
        # No catalogue rebuild against these temp folders (it writes under the output dir).
        keep = cz_ui.ab_build_catalog
        cz_ui.ab_build_catalog = lambda *a, **k: None
        try:
            _msg, url = cz_ui._asset_focus_url("models", "9b/base.safetensors")
        finally:
            cz_ui.ab_build_catalog = keep
        assert url.endswith("&focus=9b/base.safetensors"), url
        # The auto-preset is named after the file alone, not the subfolder.
        assert cz_ui._preset_sanitize(os.path.splitext(os.path.basename("9b/base.safetensors"))[0]) == "base"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} checkpoint subfolder tests passed.")
