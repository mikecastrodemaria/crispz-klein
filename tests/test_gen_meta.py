"""The generation metadata: it must describe the image, not the intention.

Three holes, all of the same kind -- an image that cannot be reproduced from its own
file, or worse, that asserts something false:

1. THE EDIT LoRAs were not in there at all. The edit set is SEPARATE from the base set
   and it is the one that shapes an edit's result; the omni path only passed
   `extra={"refs": n}`.
2. THE LIST WAS THE ONE OF THE LoRAs ASKED FOR, not of the LoRAs applied. Now that a LoRA
   can be discarded along the way (the wrong 4B/9B variant, an unsupported LyCORIS, a
   quantised build, a missing file), signing an image with a LoRA it does not carry
   becomes easy. And a LoKr, merged into the weights, appears in NO PEFT
   adapter: without _APPLIED_LOKRS it disappeared from the metadata.
3. THE BASE REPO was missing as soon as a single-file was chosen. A single-file only
   replaces the transformer -- the VAE, the text encoder and the architecture config come
   from the repo -- and the 4B/9B are not interchangeable.

Run:  .venv/Scripts/python tests/test_gen_meta.py

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_imageio as IO
import cz_pipeline as P

BASE = "black-forest-labs/FLUX.2-klein-9B"
CKPT = r"F:\models\rayKlein9bBFS_fp8V2.safetensors"


def _state(**kw):
    """Sets the model state _gen_meta reads and returns what is needed to restore it."""
    old = {k: getattr(P, k) for k in
           ("BASE_REPO", "ZIMAGE_TRANSFORMER", "LORAS", "_APPLIED_LORAS",
            "_APPLIED_LOKRS", "_APPLIED_EDIT_LORAS", "EDIT_SPEED")}
    P.BASE_REPO, P.ZIMAGE_TRANSFORMER = BASE, None
    P.LORAS = P._APPLIED_LORAS = P._APPLIED_LOKRS = P._APPLIED_EDIT_LORAS = []
    P.EDIT_SPEED = None
    for k, v in kw.items():
        setattr(P, k, v)
    return old


def _restore(old):
    for k, v in old.items():
        setattr(P, k, v)


def test_a_lokr_appears_although_it_is_no_adapter():
    old = _state(_APPLIED_LOKRS=[("/l/klein_snofs_v1_4.safetensors", 1.0)])
    try:
        m = P._gen_meta("txt2img", "p")
    finally:
        _restore(old)
    assert m["loras"] == ["klein_snofs_v1_4.safetensors@1.0"], m.get("loras")
    print("OK test_a_lokr_appears_although_it_is_no_adapter")


def test_a_refused_lora_is_not_claimed_as_applied():
    """The quiet lie: asked for, discarded, and listed all the same."""
    old = _state(LORAS=[("/l/ok.safetensors", 0.8), ("/l/refused.safetensors", 0.5)],
                 _APPLIED_LORAS=[("/l/ok.safetensors", 0.8)])
    try:
        m = P._gen_meta("txt2img", "p")
    finally:
        _restore(old)
    assert m["loras"] == ["ok.safetensors@0.8"], m.get("loras")
    assert m["loras_not_applied"] == ["refused.safetensors@0.5"], m.get("loras_not_applied")
    print("OK test_a_refused_lora_is_not_claimed_as_applied")


def test_edit_loras_are_recorded_on_an_edit():
    old = _state(_APPLIED_EDIT_LORAS=[("/e/consistence-edit.safetensors", 0.6)],
                 EDIT_SPEED={"name": "Rapid 8-step", "steps": 8})
    try:
        m = P._gen_meta("omni", "p")
    finally:
        _restore(old)
    assert m["edit_loras"] == ["consistence-edit.safetensors@0.6"], m.get("edit_loras")
    assert m["edit_speed"] == "Rapid 8-step", m.get("edit_speed")
    print("OK test_edit_loras_are_recorded_on_an_edit")


def test_edit_loras_do_not_leak_into_a_txt2img():
    """_APPLIED_EDIT_LORAS survives the edit that set it: with no mode guard, the next
    txt2img would claim a set it never carried."""
    old = _state(_APPLIED_EDIT_LORAS=[("/e/consistence-edit.safetensors", 0.6)],
                 EDIT_SPEED={"name": "Rapid 8-step", "steps": 8})
    try:
        m = P._gen_meta("txt2img", "p")
    finally:
        _restore(old)
    assert "edit_loras" not in m, m
    assert "edit_speed" not in m, m
    print("OK test_edit_loras_do_not_leak_into_a_txt2img")


def test_a_single_file_records_the_base_repo_it_needs():
    old = _state(ZIMAGE_TRANSFORMER=CKPT)
    try:
        m = P._gen_meta("txt2img", "p")
    finally:
        _restore(old)
    assert m["model"] == CKPT, m["model"]
    assert m["base_repo"] == BASE, m.get("base_repo")
    print("OK test_a_single_file_records_the_base_repo_it_needs")


def test_the_base_repo_alone_needs_no_second_line():
    old = _state()
    try:
        m = P._gen_meta("txt2img", "p")
    finally:
        _restore(old)
    assert m["model"] == BASE
    assert "base_repo" not in m, "redondant quand le modele EST le repo"
    print("OK test_the_base_repo_alone_needs_no_second_line")


def test_the_a1111_chunk_carries_them_too():
    """That is the line Civitai and the A1111 viewers read."""
    old = _state(ZIMAGE_TRANSFORMER=CKPT,
                 _APPLIED_LORAS=[("/l/style.safetensors", 0.8)],
                 _APPLIED_EDIT_LORAS=[("/e/edit.safetensors", 0.6)])
    try:
        line = IO._a1111_parameters(P._gen_meta("omni", "p", seed=1, steps=8))
    finally:
        _restore(old)
    assert "Loras: style.safetensors@0.8" in line, line
    assert "Edit loras: edit.safetensors@0.6" in line, line
    assert f"Base: {BASE}" in line, line
    print("OK test_the_a1111_chunk_carries_them_too")


# ---------------------------------------------------------------------------
# The INPUT image. An img2img, an inpaint or an edit are defined as much by their
# input as by their prompt. Only one of the four outputs named it: the batch (a hardcoded
# basename), not the plain img2img, not the inpaint, and the edit only wrote the NUMBER
# of references.
# The default name and not the path: the PNG travels while the sidecar stays local, and on
# the UI side Gradio drops the uploads into a temporary folder where only the base name
# carries the file's original name.
# ---------------------------------------------------------------------------

# Full input paths: os.path.join keeps the separator of the running OS, and
# os.path.basename does not split on a backslash under Linux, where the CI runs.
TMP_UPLOAD = os.path.join("C:" + os.sep, "Users", "x", "AppData", "Local", "Temp",
                          "gradio", "ab12", "ma_photo.png")
SRC = os.path.join("F:" + os.sep, "in", "shot.png")
OTHER = os.path.join("F:" + os.sep, "in", "other.png")


def _pil(name=None):
    from PIL import Image
    im = Image.new("RGB", (8, 8))
    if name:
        im.filename = name
    return im


def test_a_path_a_pil_and_an_editor_all_give_the_name():
    assert P.source_meta(SRC) == {"source": "shot.png"}
    assert P.source_meta(_pil(TMP_UPLOAD)) == {"source": "ma_photo.png"}
    # gr.ImageEditor: after a crop the composite is new and nameless, the background
    # keeps the one of the loaded file -- hence the order in which we try.
    assert P.source_meta({"background": _pil(TMP_UPLOAD),
                          "composite": _pil()}) == {"source": "ma_photo.png"}
    print("OK test_a_path_a_pil_and_an_editor_all_give_the_name")


def test_an_unknown_source_records_nothing():
    """An image pasted or generated has no file: nothing rather than an invented name."""
    assert P.source_meta(_pil()) == {}
    assert P.source_meta(None) == {}
    assert P.source_meta([None, None]) == {}
    print("OK test_an_unknown_source_records_nothing")


def test_several_references_come_back_as_a_list():
    got = P.source_meta([_pil(TMP_UPLOAD), None, OTHER, None], "ref_images")
    assert got == {"ref_images": ["ma_photo.png", "other.png"]}, got
    print("OK test_several_references_come_back_as_a_list")


def test_full_and_off_are_honoured():
    old = P.METADATA_SOURCE
    try:
        P.METADATA_SOURCE = "full"
        got = P.source_meta(SRC)["source"]
        assert got.endswith("shot.png") and "in" in got, got
        P.METADATA_SOURCE = "off"
        assert P.source_meta(SRC) == {}
    finally:
        P.METADATA_SOURCE = old
    print("OK test_full_and_off_are_honoured")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All generation-metadata tests passed.")
