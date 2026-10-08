"""The CivitAI sheet of a model kept in an EXTRA folder.

The Asset Browser catalogue lists the main folder AND the extra ones (loras_extra_dirs /
checkpoints_extra_dir), but the "Fetch from CivitAI" button joined the relative path to the
main folder ALONE: any LoRA of a library kept outside the app folder -- the normal case --
answered "model file not found".

Neither network nor model: only the path resolution is checked.

Run:  .venv/Scripts/python tests/test_civitai_extra_dirs.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P  # noqa: E402
import cz_ui  # noqa: E402


class _Dirs:
    """An empty main folder + an extra folder that holds everything."""

    def __init__(self):
        self.main = tempfile.mkdtemp()
        self.extra = tempfile.mkdtemp()

    def __enter__(self):
        self.old = (P.LORAS_DIR, list(P.LORAS_EXTRA_DIRS),
                    P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR)
        P.LORAS_DIR = P.CHECKPOINTS_DIR = self.main
        P.LORAS_EXTRA_DIRS = [self.extra]
        P.CHECKPOINTS_EXTRA_DIR = self.extra
        return self

    def __exit__(self, *exc):
        (P.LORAS_DIR, P.LORAS_EXTRA_DIRS,
         P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR) = self.old
        for d in (self.main, self.extra):
            shutil.rmtree(d, ignore_errors=True)
        return False

    def put(self, rel):
        p = os.path.join(self.extra, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"x")
        return p


def test_a_lora_of_an_extra_folder_is_found():
    with _Dirs() as d:
        real = d.put("Style/50sNoir.safetensors")
        got = cz_ui._civitai_model_path("Style/50sNoir.safetensors", "loras")
        assert os.path.isfile(got), got
        assert os.path.normcase(got) == os.path.normcase(real), (got, real)


def test_a_checkpoint_of_the_extra_folder_is_found():
    with _Dirs() as d:
        real = d.put("flux2-klein-4b.safetensors")
        got = cz_ui._civitai_model_path("flux2-klein-4b.safetensors", "models")
        assert os.path.normcase(got) == os.path.normcase(real), (got, real)


def test_a_model_missing_everywhere_falls_back_to_the_main_folder():
    """No file -> the main folder's path, so the message stays clear ("model file not
    found") instead of an exception."""
    with _Dirs() as d:
        got = cz_ui._civitai_model_path("ghost.safetensors", "loras")
        assert os.path.normcase(got) == os.path.normcase(
            os.path.join(d.main, "ghost.safetensors")), got
        assert not os.path.isfile(got)


def test_an_empty_name_resolves_to_nothing():
    with _Dirs():
        assert cz_ui._civitai_model_path("", "loras") == ""
        assert cz_ui._civitai_model_path(None, "models") == ""


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} CivitAI extra-folder tests passed.")
