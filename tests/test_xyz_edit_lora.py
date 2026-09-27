"""An X/Y/Z grid: the "Edit LoRA weight" axis, and the parser that crashed on a newline.

TWO LoRA SETS. The BASE set (cz_pipeline.LORAS) is applied on every generation;
the EDIT set (EDIT_LORAS) is only applied by _ui_generate's omni branch --
use_input + the "Reference (Omni)" mode + at least one reference image. So an axis that
varied the edit weight on a txt2img grid would render N IDENTICAL images, with no error
and no explanation: the worst possible outcome. The axis refuses that
case, saying what to tick.

THE PARSER. `csv.reader([s])` refused a newline in an unquoted field
(_csv.Error: new-line character seen in unquoted field), and the error left as a Gradio
trace: noisy in the console, mute in the interface. Pasting a list over several
lines is a normal gesture, though.

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
        v[U._Q_REF_IDX[0]] = object()          # one reference is enough
    return v


def _ms(edit_loras=(("/x/consistence-edit.safetensors", 0.6),), enabled=True):
    return {"base_repo": "r", "transformer": None, "loras": [],
            "edit_loras": [tuple(t) for t in edit_loras],
            "edit_loras_enabled": enabled,
            "sampler": "euler", "schedule": "sgm_uniform"}


# ------------------------------------------------------------------- parser ---

def test_newlines_are_separators_like_commas():
    for raw in ("-0.4, -0.2, 0", "-0.4\n-0.2\n0", "-0.4,\r\n-0.2,\r\n0",
                "  -0.4 ,\n -0.2 \n\n 0  "):
        assert U._xyz_parse_values(raw) == ["-0.4", "-0.2", "0"], repr(raw)
    print("OK test_newlines_are_separators_like_commas")


def test_quotes_still_protect_commas_and_newlines():
    assert U._xyz_parse_values('a, "b, with comma", c') == ["a", "b, with comma", "c"]
    # a value that REALLY holds a newline gets quoted, like a comma
    assert U._xyz_parse_values('"two\nlines", other') == ["two\nlines", "other"]
    print("OK test_quotes_still_protect_commas_and_newlines")


def test_an_empty_field_is_empty_not_an_error():
    for raw in ("", "   ", "\n", None):
        assert U._xyz_parse_values(raw) == []
    print("OK test_an_empty_field_is_empty_not_an_error")


def test_a_csv_error_becomes_a_message_never_a_traceback():
    """The parser lets no csv.Error escape: it leaves as a ValueError, which
    _ui_xyz_build displays. That is the point that was missing -- the failure was invisible
    on the interface side."""
    import csv
    real = csv.reader

    def boom(*_a, **_k):
        raise csv.Error("simulated")

    csv.reader = boom
    try:
        U._xyz_parse_values("a, b")
    except ValueError as e:
        assert "unreadable value list" in str(e), e
    except csv.Error:                      # noqa: B902 - that is exactly the bug
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
    # the calibration offers a sweep centred on 0: negative weights invert
    # the LoRA's effect, and 0 serves as the reference with no effect.
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
    """The heart of the matter: on a grid with no editing, the edit set is
    never applied. Rendering N identical images would be worse than an error."""
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
    """The box unticked = a set remembered but never applied: the same silent trap."""
    _v, err = U._xyz_validate_axis("Edit LoRA weight", ["0.4"], _vals(edit=True),
                                   _ms(enabled=False))
    assert err and "checkbox" in err, err
    print("OK test_the_edit_loras_checkbox_being_off_is_refused")


def test_applying_a_value_rewrites_only_the_edit_set():
    """The weight replaces the one of EVERY edit slot, and the BASE set does not
    move: the two sets are distinct, the axis must not spill onto the other."""
    ms = _ms(edit_loras=(("/a.safetensors", 0.6), ("/b.safetensors", 1.0)))
    ms["loras"] = [("/base.safetensors", 0.9)]
    vals = _vals(edit=True)
    U._xyz_apply("Edit LoRA weight", -0.25, vals, ms)
    assert ms["edit_loras"] == [("/a.safetensors", -0.25), ("/b.safetensors", -0.25)]
    assert ms["loras"] == [("/base.safetensors", 0.9)], ms["loras"]
    print("OK test_applying_a_value_rewrites_only_the_edit_set")


def test_the_base_lora_axis_still_ignores_the_edit_set():
    """The symmetric control: the historical axis must not touch the edit set."""
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
