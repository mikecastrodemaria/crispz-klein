"""Un checkpoint "undistilled" rendait une bouillie floue, sans un mot.

klein-4B/9B sont distilles: la guidance y est INERTE (mesuree bit-a-bit identique
de 1.0 a 8.0, cf. test_klein_guidance.py), donc _qwen_call la force a 1.0. Mais ce
drapeau `is_distilled` decrit le REPO DE BASE. Avec un override single-file il ne
dit plus rien du modele qui calcule -- et il existe des checkpoints communautaires
explicitement NON distilles ("undistilled - use with Turbo Lora") qui exigent une
vraie CFG. Les forcer a 1.0 donnait une image lissee, illisible.

Run:  .venv/Scripts/python tests/test_undistilled_guidance.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P


class Pipe:
    config = type("c", (), {"is_distilled": True})()
    _execution_device = "cpu"
    text_encoder = None

    def __call__(self, **kw):
        self.kw = kw
        return type("o", (), {"images": ["I"]})()


def _guidance(cfg, transformer):
    old = (P.GUIDANCE, P.ZIMAGE_TRANSFORMER)
    P.GUIDANCE, P.ZIMAGE_TRANSFORMER = cfg, transformer
    try:
        p = Pipe()
        P._qwen_call(p, prompt="x")
        return p.kw["guidance_scale"]
    finally:
        P.GUIDANCE, P.ZIMAGE_TRANSFORMER = old


CKPT = os.path.join("F:", "x", "kleinUndistilled_v1.safetensors")


def test_the_base_repo_stays_at_one():
    """Sur le repo de base, la guidance est inerte: prouve, on ne la transmet pas."""
    assert _guidance(1.0, None) == 1.0
    assert _guidance(4.0, None) == 1.0, "le repo de base ignore la CFG (mesure)"
    print("OK test_the_base_repo_stays_at_one")


def test_an_override_gets_the_slider():
    """Sur un checkpoint tiers, on ne sait PAS s'il est distille: l'utilisateur decide."""
    assert _guidance(4.0, CKPT) == 4.0
    assert _guidance(1.0, CKPT) == 1.0, "curseur a 1.0: rien ne change"
    print("OK test_an_override_gets_the_slider")


def test_an_explicit_guidance_is_never_overridden():
    p = Pipe()
    P._qwen_call(p, prompt="x", guidance_scale=7.5)
    assert p.kw["guidance_scale"] == 7.5
    print("OK test_an_explicit_guidance_is_never_overridden")


if __name__ == "__main__":
    test_the_base_repo_stays_at_one()
    test_an_override_gets_the_slider()
    test_an_explicit_guidance_is_never_overridden()
    print("ALL OK")
