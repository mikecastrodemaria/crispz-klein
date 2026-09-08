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



def test_the_ignored_guidance_is_announced_once():
    """Regle maison: une valeur jetee se dit. Mais une fois, pas a chaque image."""
    P._CFG_IGNORED_SAID.clear()
    said = []
    real, P._log = P._log, said.append
    try:
        _guidance(4.0, None)
        _guidance(4.0, None)
        _guidance(4.0, None)
    finally:
        P._log = real
    assert len(said) == 1, f"{len(said)} lignes pour la meme valeur"
    assert "4" in said[0] and "distille" in said[0], said[0]
    P._CFG_IGNORED_SAID.clear()
    print("OK test_the_ignored_guidance_is_announced_once")


def test_a_preset_exists_for_undistilled_checkpoints():
    """Sans preset, le seul chemin etait d'editer config.txt a la main."""
    import cz_ui
    hits = [(n, v) for n, v in cz_ui.PERFORMANCE.items() if float(v[1]) > 1.0]
    assert hits, f"aucun preset avec une vraie CFG: {list(cz_ui.PERFORMANCE)}"
    name, (steps, cfg) = hits[0]
    assert steps >= 20 and cfg >= 2.0, (name, steps, cfg)
    # et le radio doit pouvoir se rallumer dessus depuis les curseurs
    assert cz_ui._performance_label_for(steps, cfg) == name
    # le repli du radio ne doit jamais nommer un preset inexistant
    assert cz_ui._valid_performance(None) in cz_ui.PERFORMANCE
    assert cz_ui._valid_performance("Turbo (8 steps)") in cz_ui.PERFORMANCE
    print("OK test_a_preset_exists_for_undistilled_checkpoints")



def test_an_undistilled_checkpoint_gets_the_right_profile():
    """Choisir le modele posait 4 steps / CFG 1.0 d'apres le NOM DE FICHIER, ce qui
    ecrasait le preset Performance que l'utilisateur venait de choisir -- en silence.
    Le nom de fichier ne dit pas qu'un build est undistilled; le sidecar CivitAI si."""
    import json
    import tempfile
    import cz_ui as U

    tmp = tempfile.mkdtemp(prefix="cz_undist_")
    ck = os.path.join(tmp, "kleinSomething_v1.safetensors")
    open(ck, "wb").write(bytes(16))
    side = os.path.join(tmp, "kleinSomething_v1.civitai.json")

    def profile(model_name):
        with open(side, "w", encoding="utf-8") as f:
            json.dump({"modelName": model_name}, f)
        return U._profile_for_checkpoint(ck)

    st, g, why = profile("Klein Something (undistilled - use with Turbo Lora)")
    assert g > 1.0 and st >= 20, (st, g)
    assert "undistilled" in why, why

    st, g, why = profile("Klein Something Turbo")
    assert (st, g) == (4, 1.0), (st, g)     # profil du nom de fichier, comme avant
    assert why == ""
    print("OK test_an_undistilled_checkpoint_gets_the_right_profile")


def test_the_civitai_consensus_wins_over_the_filename():
    """Le profil par substring ne sait rien de CE modele: il imposait 4 steps a tout
    ce qui s'appelle 'klein', alors que le consensus deja telecharge en demande 10
    pour l'un, 8 pour l'autre. La donnee etait sur le disque, inutilisee."""
    import json
    import tempfile
    import cz_ui as U

    tmp = tempfile.mkdtemp(prefix="cz_reco_")
    ck = os.path.join(tmp, "kleinSomething_v1.safetensors")
    open(ck, "wb").write(bytes(16))
    with open(os.path.join(tmp, "kleinSomething_v1.civitai.json"), "w",
              encoding="utf-8") as f:
        json.dump({"modelName": "Klein Something",
                   "recommended": {"n": 10, "steps": 10, "guidance": 1.0}}, f)
    st, g, why = U._profile_for_checkpoint(ck)
    assert (st, g) == (10, 1.0), (st, g)
    assert "consensus" in why and "10 community" in why, why

    # et il bat aussi le drapeau undistilled: plus specifique que la categorie
    with open(os.path.join(tmp, "kleinSomething_v1.civitai.json"), "w",
              encoding="utf-8") as f:
        json.dump({"modelName": "Klein Something (undistilled - use with Turbo Lora)",
                   "recommended": {"n": 6, "steps": 20, "guidance": 2.5}}, f)
    st, g, _why = U._profile_for_checkpoint(ck)
    assert (st, g) == (20, 2.5), (st, g)
    print("OK test_the_civitai_consensus_wins_over_the_filename")


def test_the_undistilled_profile_comes_from_the_preset():
    """Une seule source de verite: pas de 28/3.5 code en dur en plus de config.txt."""
    import cz_ui as U
    name, st, g = U._undistilled_profile()
    assert name in U.PERFORMANCE, name
    assert U.PERFORMANCE[name] == (st, g) or list(U.PERFORMANCE[name]) == [st, g]
    print("OK test_the_undistilled_profile_comes_from_the_preset")


if __name__ == "__main__":
    test_the_base_repo_stays_at_one()
    test_an_override_gets_the_slider()
    test_an_explicit_guidance_is_never_overridden()
    test_the_ignored_guidance_is_announced_once()
    test_a_preset_exists_for_undistilled_checkpoints()
    test_an_undistilled_checkpoint_gets_the_right_profile()
    test_the_civitai_consensus_wins_over_the_filename()
    test_the_undistilled_profile_comes_from_the_preset()
    print("ALL OK")
