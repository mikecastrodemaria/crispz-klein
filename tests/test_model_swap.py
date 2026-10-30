"""Unit tests for the transformer hot-swap (_swap_transformer) — no model is loaded.

Regression guard for: switching checkpoint A -> B (same base repo) used to
free_vram() and reload the WHOLE pipeline (transformer + VAE + Qwen3-4B text encoder).
Only the transformer must be reloaded; the base components stay in VRAM.

Run:  .venv/Scripts/python tests/test_model_swap.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P  # noqa: E402


class FakeT:
    def __init__(self, tag):
        self.tag = tag
        self.moved = None

    def to(self, dev):
        self.moved = dev
        return self


class FakePipe:
    def __init__(self, transformer=None):
        self.transformer = transformer
        self.calls = []

    def register_modules(self, **kw):
        self.calls.append(("register", tuple(kw)))
        for k, v in kw.items():
            setattr(self, k, v)

    def remove_all_hooks(self):
        self.calls.append(("remove_hooks",))

    def enable_model_cpu_offload(self):
        self.calls.append(("offload_model",))

    def unload_lora_weights(self):
        self.calls.append(("unload_lora",))

    def load_lora_weights(self, folder, weight_name=None, adapter_name=None, **kw):
        self.calls.append(("load_lora", weight_name))

    def set_adapters(self, names, weights):
        self.calls.append(("set_adapters", tuple(names)))


def _swap_env(monkey_new, offload="none", loras=None):
    P.OFFLOAD_MODE = offload
    P.LORAS = list(loras or [])
    P._APPLIED_LORAS = []
    P._DERIVED = {"img2img": object()}          # must be empty after the swap
    P._load_transformer = lambda: monkey_new


def test_swap_keeps_pipe_and_replaces_only_transformer():
    old_t, new_t = FakeT("old"), FakeT("new")
    pipe = FakePipe(old_t)
    _swap_env(new_t)
    assert P._swap_transformer(pipe) is True
    assert pipe.transformer is new_t, "the transformer must be replaced"
    assert ("register", ("transformer",)) in pipe.calls
    assert P._DERIVED == {}, "the derived pipes still pointed at the old transformer"


def test_swap_reapplies_loras_on_the_new_transformer():
    d = tempfile.mkdtemp()
    lp = os.path.join(d, "a.safetensors")
    with open(lp, "wb") as f:
        f.write(b"x")
    new_t = FakeT("new")
    pipe = FakePipe(FakeT("old"))
    _swap_env(new_t, loras=[(lp, 0.6)])
    assert P._swap_transformer(pipe) is True
    # the adapters were on the old transformer -> set again on the new one
    assert ("load_lora", "a.safetensors") in pipe.calls
    assert P._APPLIED_LORAS == [(lp, 0.6)]


def test_swap_offload_removes_and_reapplies_hooks():
    if P.DEVICE != "cuda":
        print("   (skip: offload path is CUDA-only on this machine)")
        return
    new_t = FakeT("new")
    pipe = FakePipe(FakeT("old"))
    _swap_env(new_t, offload="model")
    assert P._swap_transformer(pipe) is True
    assert ("remove_hooks",) in pipe.calls and ("offload_model",) in pipe.calls


def test_swap_failure_falls_back():
    pipe = FakePipe(FakeT("old"))
    P.OFFLOAD_MODE = "none"
    P.LORAS = []
    P._load_transformer = lambda: (_ for _ in ()).throw(RuntimeError("corrupt file"))
    assert P._swap_transformer(pipe) is False    # -> the caller will do free_vram + a reload


def test_set_zimage_transformer_does_not_free_the_pipe():
    """The heart of the fix: changing single-file checkpoint must no longer throw the pipeline away."""
    sentinel = object()
    P._BASE_PIPE = sentinel
    P.ZIMAGE_TRANSFORMER = "D:/models/A.safetensors"
    P.set_zimage_transformer("D:/models/B.safetensors")
    assert P.ZIMAGE_TRANSFORMER == "D:/models/B.safetensors"
    assert P._BASE_PIPE is sentinel, "the pipe must stay loaded (the transformer alone is swapped)"
    P._BASE_PIPE = None


def test_set_zimage_model_single_file_does_not_free():
    # _is_single_file requires a REAL file (as resolve_checkpoint supplies)
    d = tempfile.mkdtemp()
    ck = os.path.join(d, "Juggernaut.safetensors")
    with open(ck, "wb") as f:
        f.write(b"x")
    sentinel = object()
    P._BASE_PIPE = sentinel
    P.ZIMAGE_TRANSFORMER = None
    P.set_zimage_model(ck)
    assert P.ZIMAGE_TRANSFORMER == ck
    assert P._BASE_PIPE is sentinel, "a single-file checkpoint must not throw the pipe away"
    P._BASE_PIPE = None


def test_set_zimage_model_single_file_of_the_other_variant_switches_the_base(monkeypatch):
    """The CLI / protocol path: a 9B file while the 4B base is loaded used to be set
    blindly and die at load time. Now the base follows the file (full reload) and the
    log says so; a file of the SAME variant still swaps the transformer alone."""
    d = tempfile.mkdtemp()
    ck = os.path.join(d, "darkBeast9B.safetensors")
    with open(ck, "wb") as f:
        f.write(b"x")
    monkeypatch.setattr(P, "_flux2_hidden_dim", lambda p: 4096)
    monkeypatch.setattr(P, "_base_hidden_dim", lambda base=None: 3072)
    said = []
    monkeypatch.setattr(P, "_log", lambda m: said.append(m))
    P._BASE_PIPE = object()
    P._LOADED_KEY = ("black-forest-labs/FLUX.2-klein-4B", None, "none")
    P.BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"
    P.ZIMAGE_TRANSFORMER = None
    P.set_zimage_model(ck)
    assert P.BASE_REPO == "black-forest-labs/FLUX.2-klein-9B"
    assert P.ZIMAGE_TRANSFORMER == ck
    assert P._BASE_PIPE is None, "another base: VAE and text encoder change, full reload"
    assert any("FLUX.2-klein-9B" in m and "NON-COMMERCIAL" in m for m in said)
    # the same variant: nothing moves but the transformer
    monkeypatch.setattr(P, "_flux2_hidden_dim", lambda p: 3072)
    P.BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"
    sentinel = object()
    P._BASE_PIPE = sentinel
    P.set_zimage_model(os.path.join(d, "darkBeast9B.safetensors"))
    assert P.BASE_REPO == "black-forest-labs/FLUX.2-klein-4B" and P._BASE_PIPE is sentinel
    P._BASE_PIPE = None


def test_set_zimage_model_base_repo_removes_the_single_file_override(monkeypatch):
    """Asking for the base repo by name while a file of the same variant is the
    transformer must drop that file (the official transformer comes back on the next
    run), without throwing the pipe away: VAE and text encoder are the same."""
    said = []
    monkeypatch.setattr(P, "_log", lambda m: said.append(m))
    sentinel = object()
    P._BASE_PIPE = sentinel
    P.BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"
    P.ZIMAGE_TRANSFORMER = "D:/models/fluxKlein4BPro_v10.safetensors"
    P.set_zimage_model("black-forest-labs/FLUX.2-klein-4B")
    assert P.ZIMAGE_TRANSFORMER is None
    assert P.BASE_REPO == "black-forest-labs/FLUX.2-klein-4B" and P._BASE_PIPE is sentinel
    assert any("removed" in m and "fluxKlein4BPro_v10" in m for m in said)
    P._BASE_PIPE = None


def test_set_zimage_model_new_base_repo_still_reloads():
    """The base repo changes -> the VAE/encoder change too -> a full reload is mandatory."""
    P._BASE_PIPE = object()
    P._LOADED_KEY = ("old/repo", None, "none")
    P.BASE_REPO = "old/repo"
    P.set_zimage_model("black-forest-labs/FLUX.2-klein-4B")
    assert P.BASE_REPO == "black-forest-labs/FLUX.2-klein-4B"
    assert P._BASE_PIPE is None, "changing the base repo must release the pipe"
    assert P._LOADED_KEY is None


if __name__ == "__main__":
    _real = P._load_transformer
    try:
        for fn in (test_swap_keeps_pipe_and_replaces_only_transformer,
                   test_swap_reapplies_loras_on_the_new_transformer,
                   test_swap_offload_removes_and_reapplies_hooks,
                   test_swap_failure_falls_back,
                   test_set_zimage_transformer_does_not_free_the_pipe,
                   test_set_zimage_model_single_file_does_not_free,
                   test_set_zimage_model_new_base_repo_still_reloads):
            fn()
            print(f"OK {fn.__name__}")
    finally:
        P._load_transformer = _real
    print("All model swap tests passed.")
