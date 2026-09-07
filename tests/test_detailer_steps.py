"""Une passe de detailer tournait a 12 steps sur un modele distille pour 4.

Le curseur "Refine steps" (defaut 12) vient de crispz-studio / Z-Image. Sur klein
il triplait le cout de CHAQUE main et de CHAQUE visage sans rien changer a l'image.
Mesure (RTX 5090, 2 mains): 4B 2.0 -> 0.9 s/main, 9B GGUF 12.2 -> 6.4 s/main, pour
un ecart d'image de MAE 0.7.

Run:  .venv/Scripts/python tests/test_detailer_steps.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_detailer as D
import cz_pipeline as P


def _with_model(base, transformer, fn):
    old = (P.BASE_REPO, P.ZIMAGE_TRANSFORMER, D.CONFIG.get("detailer_steps"))
    P.BASE_REPO, P.ZIMAGE_TRANSFORMER = base, transformer
    try:
        return fn()
    finally:
        P.BASE_REPO, P.ZIMAGE_TRANSFORMER = old[0], old[1]
        if old[2] is None:
            D.CONFIG.pop("detailer_steps", None)
        else:
            D.CONFIG["detailer_steps"] = old[2]


def test_follows_the_model_profile_by_default():
    def check():
        D.CONFIG.pop("detailer_steps", None)
        assert D._detailer_steps(12) == 4, "klein est distille a 4 steps"
    _with_model("black-forest-labs/FLUX.2-klein-4B", None, check)
    # un transformer single-file decide aussi du profil
    _with_model("black-forest-labs/FLUX.2-klein-9B",
                r"F:\x\pGGUFAishaNSFWFlux2Klein_v1.gguf", check)
    print("OK test_follows_the_model_profile_by_default")


def test_never_raises_the_slider():
    """Descendre le curseur sous le profil reste un choix de l'utilisateur."""
    def check():
        D.CONFIG.pop("detailer_steps", None)
        assert D._detailer_steps(3) == 3
        assert D._detailer_steps(1) == 1
    _with_model("black-forest-labs/FLUX.2-klein-4B", None, check)
    print("OK test_never_raises_the_slider")


def test_config_overrides():
    def check():
        D.CONFIG["detailer_steps"] = 8
        assert D._detailer_steps(12) == 8, "un entier positif force la valeur"
        D.CONFIG["detailer_steps"] = -1
        assert D._detailer_steps(12) == 12, "-1 rend la main au curseur"
        D.CONFIG["detailer_steps"] = "n'importe quoi"
        assert D._detailer_steps(12) == 4, "une valeur illisible retombe sur le profil"
    _with_model("black-forest-labs/FLUX.2-klein-4B", None, check)
    print("OK test_config_overrides")


if __name__ == "__main__":
    test_follows_the_model_profile_by_default()
    test_never_raises_the_slider()
    test_config_overrides()
    print("ALL OK")
