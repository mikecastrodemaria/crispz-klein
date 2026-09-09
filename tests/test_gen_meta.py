"""Metadonnees de generation: elles doivent decrire l'image, pas l'intention.

Trois trous, tous du meme genre -- une image qu'on ne peut pas reproduire depuis son
propre fichier, ou pire, qui affirme quelque chose de faux:

1. LES LoRA D'EDITION n'y etaient pas du tout. Le jeu d'edition est SEPARE du jeu de
   base et c'est lui qui faconne le resultat d'une edition; le chemin omni ne passait
   que `extra={"refs": n}`.
2. LA LISTE ETAIT CELLE DES LoRA DEMANDEES, pas des LoRA posees. Depuis qu'une LoRA
   peut etre ecartee en route (mauvaise variante 4B/9B, LyCORIS non supporte, build
   quantifie, fichier absent), signer une image avec une LoRA qu'elle ne porte pas
   devient facile. Et une LoKr, fusionnee dans les poids, n'apparait dans AUCUN
   adaptateur PEFT: sans _APPLIED_LOKRS elle disparaissait des metadonnees.
3. LE REPO DE BASE manquait des qu'un single-file etait choisi. Un single-file ne
   remplace que le transformer -- VAE, encodeur texte et config d'archi viennent du
   repo -- et 4B/9B ne sont pas interchangeables.

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
    """Pose l'etat modele lu par _gen_meta et rend de quoi le restaurer."""
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
    """Le mensonge tranquille: demandee, ecartee, et pourtant listee."""
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
    """_APPLIED_EDIT_LORAS survit a l'edition qui l'a pose: sans garde de mode, le
    txt2img suivant revendiquerait un jeu qu'il n'a jamais porte."""
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
    """C'est la ligne que lisent Civitai et les visionneuses A1111."""
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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All generation-metadata tests passed.")
