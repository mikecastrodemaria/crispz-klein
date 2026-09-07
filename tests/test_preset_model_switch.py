"""Charger un preset changeait tout SAUF le modele, en silence.

Les presets sont auto-crees par modele local. Une bibliotheque klein-9B sur un
install 4B en laisse donc une pile qui nomment des checkpoints que list_checkpoints
ecarte desormais. Au Load, cz_ui poussait ce nom dans le dropdown: le frontend
Gradio refuse une valeur hors 'choices', la chaine .then qui applique le modele
relit alors l'ANCIENNE valeur, et le run suivant tournait sur le modele precedent.
Vu de l'utilisateur: "je change de modele, c'est toujours le modele par defaut".

Aucun modele n'est charge ici: tout se joue sur des en-tetes safetensors.

Run:  .venv/Scripts/python tests/test_preset_model_switch.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

import cz_core
import cz_pipeline as P
import cz_ui as U

TMP = os.path.join(os.environ.get("TEMP") or "/tmp", "cz_preset_switch")
os.makedirs(TMP, exist_ok=True)


def _ckpt(name, dim):
    """Faux transformer FLUX.2: seul l'en-tete est lu, les poids sont vides."""
    p = os.path.join(TMP, name)
    save_file({"double_stream_modulation_img.lin.weight": torch.zeros(dim * 6, dim),
               "x_embedder.weight": torch.zeros(2, 2)}, p)
    return p


def _with_lib(fn, base_dim=3072):
    """Execute fn avec TMP comme dossier de checkpoints ET de presets, sur un repo de
    base 4B. Le dossier presets/ de l'install ne doit rien voir de ces fixtures:
    _apply_checkpoint et _refresh_checkpoints creent des presets par modele local."""
    old = (P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR, P.BASE_REPO,
           P.ZIMAGE_TRANSFORMER, dict(P._BASE_DIM_CACHE), U._PRESETS_DIR)
    P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR = TMP, ""
    P.BASE_REPO = "base-4b"
    P._BASE_DIM_CACHE["base-4b"] = base_dim
    U._PRESETS_DIR = os.path.join(TMP, "presets")
    os.makedirs(U._PRESETS_DIR, exist_ok=True)
    try:
        return fn()
    finally:
        (P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR, P.BASE_REPO,
         P.ZIMAGE_TRANSFORMER, cache, U._PRESETS_DIR) = old
        P._BASE_DIM_CACHE.clear()
        P._BASE_DIM_CACHE.update(cache)


def test_refusal_names_the_reason_and_the_fix():
    _ckpt("good4b.safetensors", 3072)
    _ckpt("big9b.safetensors", 4096)

    def check():
        assert P.checkpoint_refusal("good4b.safetensors") is None
        why = P.checkpoint_refusal("big9b.safetensors")
        assert why and "9B" in why and "4B" in why, why
        # un refus nomme fichier par fichier doit porter le mode d'emploi COMPLET:
        # contrairement au listage, il n'y a pas de resume derriere pour le dire.
        assert P.CFG_MODEL_KEY in why, why
        assert "NON-COMMERCIAL" in why, why
        # le refus doit nommer le repo EXACT a choisir et l'endroit ou le choisir:
        # renvoyer vers "le repo correspondant" et un fichier de config, alors qu'un
        # dropdown fait le travail, c'est renvoyer au mauvais endroit.
        assert "black-forest-labs/FLUX.2-klein-9B" in why, why
        assert "Klein checkpoint" in why, why
        assert "GATED" in why and "licence" in why, why
        # un repo HF / dossier diffusers ne passe pas par ce filtre
        assert P.checkpoint_refusal("black-forest-labs/FLUX.2-klein-4B") is None
        # un nom qui n'existe plus sur le disque doit se dire, pas se taire
        gone = P.checkpoint_refusal("moved-away.safetensors")
        assert gone and "no such file" in gone, gone
    _with_lib(check)
    print("OK test_refusal_names_the_reason_and_the_fix")


def test_apply_checkpoint_refuses_the_wrong_variant():
    """Sans garde, un 9B devenait le transformer courant et mourait au run suivant,
    apres avoir lu des gigaoctets, sur un message diffusers illisible."""
    _ckpt("good4b.safetensors", 3072)
    _ckpt("big9b.safetensors", 4096)

    def check():
        P.ZIMAGE_TRANSFORMER = None
        msg = U._apply_checkpoint("good4b.safetensors")[0]
        assert P.ZIMAGE_TRANSFORMER and "good4b" in P.ZIMAGE_TRANSFORMER, msg
        assert "Klein transformer" in msg, msg
        assert "Qwen" not in msg, f"nom du fork amont: {msg}"

        kept = P.ZIMAGE_TRANSFORMER
        msg = U._apply_checkpoint("big9b.safetensors")[0]
        assert P.ZIMAGE_TRANSFORMER == kept, "un 9B a ete applique sur un install 4B"
        assert "NOT applied" in msg and "9B" in msg, msg
        assert "good4b" in msg, f"le message doit dire ce qui reste en place: {msg}"
    _with_lib(check)
    print("OK test_apply_checkpoint_refuses_the_wrong_variant")


def _preset(name, ckpt, steps=7):
    os.makedirs(U._PRESETS_DIR, exist_ok=True)
    p = os.path.join(U._PRESETS_DIR, name + ".json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"prompt": "a lighthouse", "steps": steps, "checkpoint": ckpt,
                   "transformer": "", "loras": []}, f)
    return p


def _ckpt_update(outs):
    """Le gr.update du dropdown checkpoint dans la sortie de _ui_preset_load."""
    return outs[U._PRESET_KEYS.index("checkpoint")]


def test_preset_never_pushes_a_checkpoint_the_dropdown_refuses():
    _ckpt("good4b.safetensors", 3072)
    _ckpt("big9b.safetensors", 4096)

    def check():
        _preset("zz-test-ok", "good4b.safetensors")
        _preset("zz-test-bad", "big9b.safetensors")
        P.ZIMAGE_TRANSFORMER = P.resolve_checkpoint("good4b.safetensors")

        outs = U._ui_preset_load("zz-test-ok")
        assert _ckpt_update(outs).get("value") == "good4b.safetensors"
        assert "NOT switched" not in outs[-1], outs[-1]
        # le reste du preset s'applique dans les deux cas
        assert outs[U._PRESET_KEYS.index("steps")].get("value") == 7

        outs = U._ui_preset_load("zz-test-bad")
        # rien ne doit etre pousse: une valeur hors 'choices' est rejetee par le
        # frontend, et c'est ce rejet silencieux qui laissait le modele en place.
        assert "value" not in _ckpt_update(outs), _ckpt_update(outs)
        status = outs[-1]
        assert "NOT switched" in status and "big9b.safetensors" in status, status
        assert "9B" in status, status
        assert "good4b" in status, f"doit dire quel modele reste actif: {status}"
        assert outs[U._PRESET_KEYS.index("steps")].get("value") == 7, \
            "le reste du preset doit s'appliquer quand meme"
    _with_lib(check)
    print("OK test_preset_never_pushes_a_checkpoint_the_dropdown_refuses")


def test_both_base_repos_are_selectable_and_the_9b_is_announced():
    """4B et 9B se choisissent au dropdown. Le 4B reste le defaut; le 9B doit dire
    ce qu'il coute AVANT le run: licence non commerciale, repo gated, VRAM."""
    assert U.KLEIN_BASE_4B in U.ZIMAGE_BASE_REPOS
    assert U.KLEIN_BASE_9B in U.ZIMAGE_BASE_REPOS
    assert P.DEFAULT_BASE_REPO == U.KLEIN_BASE_4B, "le defaut doit rester le 4B"
    note = U.BASE_REPO_NOTES[U.KLEIN_BASE_9B]
    assert "Non-Commercial" in note and "gated" in note and "VRAM" in note, note
    print("OK test_both_base_repos_are_selectable_and_the_9b_is_announced")


def test_base_swap_refreshes_the_checkpoint_list():
    """4B et 9B n'acceptent pas les memes fichiers: choisir un repo de base doit
    reconstruire la liste, sinon le dropdown propose des modeles impossibles."""
    _ckpt("good4b.safetensors", 3072)
    _ckpt("big9b.safetensors", 4096)
    saved = {}

    def check():
        # aucune ecriture dans le vrai preferences.json pendant un test
        real_save, U._save_prefs_keys = U._save_prefs_keys, saved.update
        # la dimension des deux repos est pre-calee: aucun acces reseau ici
        P._BASE_DIM_CACHE[U.KLEIN_BASE_4B] = 3072
        P._BASE_DIM_CACHE[U.KLEIN_BASE_9B] = 4096
        try:
            P.BASE_REPO = U.KLEIN_BASE_4B
            P.ZIMAGE_TRANSFORMER = P.resolve_checkpoint("good4b.safetensors")
            out = U._apply_checkpoint(U.KLEIN_BASE_9B)
            assert P.BASE_REPO == U.KLEIN_BASE_9B, P.BASE_REPO
            assert not P.ZIMAGE_TRANSFORMER, \
                "un repo de base complet doit effacer l'override single-file"
            status, choices = out[0], out[4].get("choices")
            assert "Non-Commercial" in status and "gated" in status, status
            assert choices and "big9b.safetensors" in choices, choices
            assert "good4b.safetensors" not in choices, \
                f"un checkpoint 4B ne charge pas dans un pipeline 9B: {choices}"
            assert saved.get(P.CFG_MODEL_KEY) == U.KLEIN_BASE_9B, saved

            # et retour au 4B: la liste doit s'inverser
            out = U._apply_checkpoint(U.KLEIN_BASE_4B)
            choices = out[4].get("choices")
            assert "good4b.safetensors" in choices and "big9b.safetensors" not in choices, choices
            assert "Non-Commercial" not in out[0], out[0]
        finally:
            U._save_prefs_keys = real_save
    _with_lib(check)
    print("OK test_base_swap_refreshes_the_checkpoint_list")


def test_choosing_a_base_repo_retries_its_dimension():
    """La dimension d'un repo est cachee, echecs compris. Un repo gated refuse avant
    l'acceptation de la licence laissait un None colle pour toute la session: filtre
    4B/9B eteint meme une fois la licence acceptee. Le choisir vaut "reessaie"."""
    def check():
        P._BASE_DIM_CACHE[U.KLEIN_BASE_9B] = None     # echec precedent (403 gated)
        P.BASE_REPO = U.KLEIN_BASE_4B
        P.set_zimage_model(U.KLEIN_BASE_9B)
        assert U.KLEIN_BASE_9B not in P._BASE_DIM_CACHE,             "l'echec cache doit etre purge quand on rechoisit le repo"
    _with_lib(check)
    print("OK test_choosing_a_base_repo_retries_its_dimension")


def test_gated_repo_error_says_what_to_do():
    """Un 401/403 du Hub sur le 9B doit devenir une consigne, pas une trace."""
    hint = P._hf_access_hint(U.KLEIN_BASE_9B,
                             RuntimeError("401 Client Error: Access to model ... is restricted"))
    assert hint and U.KLEIN_BASE_9B in hint, hint
    assert "accept its licence" in hint and "token" in hint, hint
    # un token pose par 'huggingface-cli login' compte: dire "aucun token" a
    # quelqu'un qui en a un l'envoie chercher la mauvaise cause.
    real, cz_core.hf_token_is_set = cz_core.hf_token_is_set, lambda: True
    try:
        with_tok = P._hf_access_hint(U.KLEIN_BASE_9B, RuntimeError("403 gated"))
    finally:
        cz_core.hf_token_is_set = real
    assert "licence itself" in with_tok, with_tok
    # une panne ordinaire ne doit PAS etre maquillee en probleme de licence
    assert P._hf_access_hint(U.KLEIN_BASE_9B, OSError("disk full")) is None
    print("OK test_gated_repo_error_says_what_to_do")


if __name__ == "__main__":
    test_refusal_names_the_reason_and_the_fix()
    test_apply_checkpoint_refuses_the_wrong_variant()
    test_preset_never_pushes_a_checkpoint_the_dropdown_refuses()
    test_both_base_repos_are_selectable_and_the_9b_is_announced()
    test_base_swap_refreshes_the_checkpoint_list()
    test_choosing_a_base_repo_retries_its_dimension()
    test_gated_repo_error_says_what_to_do()
    print("ALL OK")
