"""Grille X/Y/Z: l'axe "Edit LoRA weight", et le parseur qui plantait sur un saut de ligne.

DEUX JEUX DE LoRA. Le jeu de BASE (cz_pipeline.LORAS) est pose sur toute generation;
le jeu d'EDITION (EDIT_LORAS) n'est pose que par la branche omni de _ui_generate --
use_input + mode "Reference (Omni)" + au moins une image de reference. Un axe qui
ferait varier le poids d'edition sur une grille txt2img rendrait donc N images
IDENTIQUES, sans erreur ni explication: le pire resultat possible. L'axe refuse ce
cas, en disant quoi cocher.

LE PARSEUR. `csv.reader([s])` refusait un retour a la ligne dans un champ non quote
(_csv.Error: new-line character seen in unquoted field), et l'erreur partait en trace
Gradio: bruyante en console, muette dans l'interface. Coller une liste sur plusieurs
lignes est pourtant un geste normal.

Run:  .venv/Scripts/python tests/test_xyz_edit_lora.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_ui as U        # noqa: E402


def _vals(edit=False):
    """Stand-in de _gen_inputs. edit=True -> un vrai job d'edition."""
    v = [None] * 36
    v[U._Q_IDX["prompt"]] = "a cat"
    v[U._Q_IDX["use_input"]] = bool(edit)
    v[U._Q_IDX["input_mode"]] = "Reference (Omni)" if edit else "Image to image"
    if edit:
        v[U._Q_REF_IDX[0]] = object()          # une reference suffit
    return v


def _ms(edit_loras=(("/x/consistence-edit.safetensors", 0.6),), enabled=True):
    return {"base_repo": "r", "transformer": None, "loras": [],
            "edit_loras": [tuple(t) for t in edit_loras],
            "edit_loras_enabled": enabled,
            "sampler": "euler", "schedule": "sgm_uniform"}


# ------------------------------------------------------------------ parseur ---

def test_newlines_are_separators_like_commas():
    for raw in ("-0.4, -0.2, 0", "-0.4\n-0.2\n0", "-0.4,\r\n-0.2,\r\n0",
                "  -0.4 ,\n -0.2 \n\n 0  "):
        assert U._xyz_parse_values(raw) == ["-0.4", "-0.2", "0"], repr(raw)
    print("OK test_newlines_are_separators_like_commas")


def test_quotes_still_protect_commas_and_newlines():
    assert U._xyz_parse_values('a, "b, with comma", c') == ["a", "b, with comma", "c"]
    # une valeur qui contient VRAIMENT un saut de ligne se quote, comme une virgule
    assert U._xyz_parse_values('"two\nlines", other') == ["two\nlines", "other"]
    print("OK test_quotes_still_protect_commas_and_newlines")


def test_an_empty_field_is_empty_not_an_error():
    for raw in ("", "   ", "\n", None):
        assert U._xyz_parse_values(raw) == []
    print("OK test_an_empty_field_is_empty_not_an_error")


def test_a_csv_error_becomes_a_message_never_a_traceback():
    """Le parseur ne laisse fuir aucune csv.Error: elle repart en ValueError, que
    _ui_xyz_build affiche. C'est le point qui manquait -- l'echec etait invisible
    cote interface."""
    import csv
    real = csv.reader

    def boom(*_a, **_k):
        raise csv.Error("simulated")

    csv.reader = boom
    try:
        U._xyz_parse_values("a, b")
    except ValueError as e:
        assert "unreadable value list" in str(e), e
    except csv.Error:                      # noqa: B902 - c'est exactement le bug
        raise AssertionError("csv.Error a fuit: elle repartirait en trace Gradio")
    else:
        raise AssertionError("aucune erreur levee")
    finally:
        csv.reader = real
    print("OK test_a_csv_error_becomes_a_message_never_a_traceback")


# --------------------------------------------------------------------- axe ---

def test_the_axis_exists_and_is_calibrated():
    assert "Edit LoRA weight" in U._XYZ_AXES
    assert U._XYZ_AXES["Edit LoRA weight"]["kind"] == "edit_lora_weight"
    # le calibrage propose un balayage centre sur 0: les poids negatifs inversent
    # l'effet de la LoRA, et 0 sert de reference sans effet.
    assert "0" in U._XYZ_CALIB["Edit LoRA weight"]
    assert "-" in U._XYZ_CALIB["Edit LoRA weight"]
    print("OK test_the_axis_exists_and_is_calibrated")


def test_on_an_edit_run_the_weights_are_accepted():
    vals, err = U._xyz_validate_axis("Edit LoRA weight", ["-0.4", "0", "0.4"],
                                     _vals(edit=True), _ms())
    assert err is None, err
    assert vals == [-0.4, 0.0, 0.4], vals
    print("OK test_on_an_edit_run_the_weights_are_accepted")


def test_a_txt2img_grid_is_refused_not_silently_useless():
    """Le coeur de l'affaire: sur une grille sans edition, le jeu d'edition n'est
    jamais pose. Rendre N images identiques serait pire qu'une erreur."""
    _v, err = U._xyz_validate_axis("Edit LoRA weight", ["0.4"], _vals(edit=False), _ms())
    assert err and "EDIT run" in err, err
    assert "Reference (Omni)" in err and "identical" in err, err
    print("OK test_a_txt2img_grid_is_refused_not_silently_useless")


def test_no_edit_lora_selected_is_refused():
    _v, err = U._xyz_validate_axis("Edit LoRA weight", ["0.4"], _vals(edit=True),
                                   _ms(edit_loras=()))
    assert err and "no active edit LoRA" in err, err
    print("OK test_no_edit_lora_selected_is_refused")


def test_the_edit_loras_checkbox_being_off_is_refused():
    """Case decochee = jeu memorise mais jamais applique: meme piege silencieux."""
    _v, err = U._xyz_validate_axis("Edit LoRA weight", ["0.4"], _vals(edit=True),
                                   _ms(enabled=False))
    assert err and "checkbox" in err, err
    print("OK test_the_edit_loras_checkbox_being_off_is_refused")


def test_applying_a_value_rewrites_only_the_edit_set():
    """Le poids remplace celui de CHAQUE slot d'edition, et le jeu de BASE ne bouge
    pas: les deux jeux sont distincts, l'axe ne doit pas deborder sur l'autre."""
    ms = _ms(edit_loras=(("/a.safetensors", 0.6), ("/b.safetensors", 1.0)))
    ms["loras"] = [("/base.safetensors", 0.9)]
    vals = _vals(edit=True)
    U._xyz_apply("Edit LoRA weight", -0.25, vals, ms)
    assert ms["edit_loras"] == [("/a.safetensors", -0.25), ("/b.safetensors", -0.25)]
    assert ms["loras"] == [("/base.safetensors", 0.9)], ms["loras"]
    print("OK test_applying_a_value_rewrites_only_the_edit_set")


def test_the_base_lora_axis_still_ignores_the_edit_set():
    """Controle symetrique: l'axe historique ne doit pas toucher au jeu d'edition."""
    ms = _ms()
    ms["loras"] = [("/base.safetensors", 0.9)]
    U._xyz_apply("LoRA weight", 0.3, _vals(), ms)
    assert ms["loras"] == [("/base.safetensors", 0.3)]
    assert ms["edit_loras"] == [("/x/consistence-edit.safetensors", 0.6)]
    print("OK test_the_base_lora_axis_still_ignores_the_edit_set")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All XYZ edit-LoRA tests passed.")
