"""Vraie CFG pour un checkpoint single-file: le drapeau `is_distilled` du pipeline.

Le pipeline FLUX.2 klein decide seul s'il fait la passe sans prompt:

    do_classifier_free_guidance = guidance_scale > 1 and not config.is_distilled

et `config.is_distilled` vient du DEPOT DE BASE (True pour klein 4B et 9B). Un
single-file ne remplace que le transformer: la config reste "distillee". L'app
transmettait donc la guidance d'un checkpoint 'undistilled' en ecrivant "guidance 3.5
transmise", et diffusers repondait a la ligne suivante "Guidance scale 3.5 is ignored
for step-wise distilled models". Releve sur le banc du 2026-09-10: kleinForeskin a 28
steps coutait 0.6 s/step, exactement comme un distille, au lieu du double.

Ces tests verrouillent: le drapeau est leve PENDANT l'appel, retabli APRES (meme sur
erreur -- le pipeline est partage, un drapeau oublie ferait passer tous les appels
suivants en CFG), jamais touche sur le repo de base, et le negatif vide que le
pipeline impose est mis en cache comme le positif.

Run:  .venv/Scripts/python tests/test_real_cfg.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import cz_pipeline as P


class _Cfg(dict):
    """Acces par attribut, comme la FrozenDict de diffusers."""
    __getattr__ = dict.get


class FakePipe:
    """Juste ce que _qwen_call touche: config, register_to_config, __call__ -- et
    l'API d'encodage quand on veut exercer le cache d'embeddings."""

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
    """Sur le repo de base on SAIT que la CFG est inerte (mesure bit a bit): 1.0, et
    le drapeau n'est pas touche."""
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
    """Le pipeline est partage: un drapeau laisse leve ferait passer TOUS les appels
    suivants en CFG, repo de base compris."""
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
    """Sans negatif fourni, le pipeline encoderait un "" a CHAQUE appel -- sous
    offload, l'encodeur remonterait sur le GPU a chaque image."""
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
    """L'ancien message sortait a CHAQUE appel (trois fois par modele dans le banc)."""
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
    said = [m for m in logged if "VRAIE CFG" in m]
    assert len(said) == 1, logged
    assert not any("transmise" in m for m in logged), logged
    print("OK test_the_announcement_is_made_once_not_per_image")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All real-CFG tests passed.")
