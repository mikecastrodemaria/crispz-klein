"""A real CFG for a single-file checkpoint: the pipeline's `is_distilled` flag.

The FLUX.2 klein pipeline decides on its own whether it does the prompt-free pass:

    do_classifier_free_guidance = guidance_scale > 1 and not config.is_distilled

and `config.is_distilled` comes from the BASE REPO (True for klein 4B and 9B). A
single-file only replaces the transformer: the config stays "distilled". So the app
passed on the guidance of an 'undistilled' checkpoint while writing "guidance 3.5
passed on", and diffusers answered on the next line "Guidance scale 3.5 is ignored
for step-wise distilled models". Caught on the 2026-09-10 bench: kleinForeskin at 28
steps cost 0.6 s/step, exactly like a distilled one, instead of double.

These tests lock it down: the flag is raised DURING the call, restored AFTER (even on
an error -- the pipeline is shared, a forgotten flag would put all the following calls
into CFG), never touched on the base repo, and the empty negative the
pipeline imposes is cached like the positive one.

Run:  .venv/Scripts/python tests/test_real_cfg.py

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import cz_pipeline as P


class _Cfg(dict):
    """Attribute access, like diffusers' FrozenDict."""
    __getattr__ = dict.get


class FakePipe:
    """Just what _qwen_call touches: config, register_to_config, __call__ -- and
    the encoding API when the embeddings cache is to be exercised."""

    def __init__(self, with_encoder=False, fail=False):
        self.config = _Cfg(is_distilled=True)
        self.fail = fail
        self.calls = []
        self.encoded = []
        if with_encoder:
            self.text_encoder = object()
            self._execution_device = "cpu"

    def register_to_config(self, **kw):
        self.config = _Cfg({**self.config, **kw})

    def encode_prompt(self, prompt, device=None):
        self.encoded.append(prompt)
        return (torch.zeros(1, 4, 8),)

    def __call__(self, **kw):
        self.calls.append({"is_distilled": bool(self.config.is_distilled),
                           "guidance": kw.get("guidance_scale"),
                           "has_negative": kw.get("negative_prompt_embeds") is not None})
        if self.fail:
            raise RuntimeError("boom")
        return "ok"


def _state(**kw):
    old = {k: getattr(P, k) for k in ("GUIDANCE", "ZIMAGE_TRANSFORMER", "_APPLIED_LORAS")}
    P.ZIMAGE_TRANSFORMER, P._APPLIED_LORAS = None, []
    P._CFG_REAL_SAID.clear()
    P._embed_cache_clear()
    for k, v in kw.items():
        setattr(P, k, v)
    return old


def _restore(old):
    for k, v in old.items():
        setattr(P, k, v)


def test_the_base_repo_keeps_guidance_inert():
    """On the base repo we KNOW the CFG is inert (measured bit for bit): 1.0, and
    the flag is not touched."""
    old = _state(GUIDANCE=3.5)
    try:
        pipe = FakePipe()
        P._qwen_call(pipe, prompt="p")
    finally:
        _restore(old)
    assert pipe.calls == [{"is_distilled": True, "guidance": 1.0,
                           "has_negative": False}], pipe.calls
    print("OK test_the_base_repo_keeps_guidance_inert")


def test_a_single_file_gets_real_cfg_during_the_call_only():
    old = _state(GUIDANCE=3.5, ZIMAGE_TRANSFORMER="kleinForeskin.safetensors")
    try:
        pipe = FakePipe()
        P._qwen_call(pipe, prompt="p")
    finally:
        _restore(old)
    call = pipe.calls[0]
    assert call["is_distilled"] is False, "le pipeline doit voir un modele NON distille"
    assert call["guidance"] == 3.5, call
    assert pipe.config.is_distilled is True, "drapeau non retabli apres l'appel"
    print("OK test_a_single_file_gets_real_cfg_during_the_call_only")


def test_the_flag_is_restored_even_when_the_call_fails():
    """The pipeline is shared: a flag left raised would put ALL the following calls
    into CFG, the base repo included."""
    old = _state(GUIDANCE=3.5, ZIMAGE_TRANSFORMER="kleinForeskin.safetensors")
    try:
        pipe = FakePipe(fail=True)
        try:
            P._qwen_call(pipe, prompt="p")
            raise AssertionError("l'erreur du pipeline doit remonter")
        except RuntimeError:
            pass
    finally:
        _restore(old)
    assert pipe.config.is_distilled is True, "drapeau non retabli apres une erreur"
    print("OK test_the_flag_is_restored_even_when_the_call_fails")


def test_guidance_one_changes_nothing():
    old = _state(GUIDANCE=1.0, ZIMAGE_TRANSFORMER="rayKlein.safetensors")
    try:
        pipe = FakePipe()
        P._qwen_call(pipe, prompt="p")
    finally:
        _restore(old)
    assert pipe.calls[0]["is_distilled"] is True, pipe.calls
    assert pipe.calls[0]["guidance"] == 1.0, pipe.calls
    print("OK test_guidance_one_changes_nothing")


def test_the_empty_negative_is_encoded_once():
    """With no negative supplied, the pipeline would encode a "" on EVERY call -- under
    offload, the encoder would come back up onto the GPU for every image."""
    old = _state(GUIDANCE=3.5, ZIMAGE_TRANSFORMER="kleinForeskin.safetensors")
    try:
        pipe = FakePipe(with_encoder=True)
        P._qwen_call(pipe, prompt="p")
        P._qwen_call(pipe, prompt="p")
    finally:
        _restore(old)
    assert all(c["has_negative"] for c in pipe.calls), pipe.calls
    assert pipe.encoded.count("") == 1, pipe.encoded
    print("OK test_the_empty_negative_is_encoded_once")


def test_the_announcement_is_made_once_not_per_image():
    """The old message came out on EVERY call (three times per model in the bench)."""
    old = _state(GUIDANCE=3.5, ZIMAGE_TRANSFORMER="kleinForeskin.safetensors")
    logged = []
    real_log = P._log
    P._log = lambda m: logged.append(m)
    try:
        pipe = FakePipe()
        for _ in range(3):
            P._qwen_call(pipe, prompt="p")
    finally:
        P._log = real_log
        _restore(old)
    said = [m for m in logged if "REAL CFG" in m]
    assert len(said) == 1, logged
    assert not any("transmise" in m for m in logged), logged
    print("OK test_the_announcement_is_made_once_not_per_image")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All real-CFG tests passed.")
