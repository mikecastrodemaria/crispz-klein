"""Elagage de l'encodeur texte, et le budget VRAM qui en depend.

FLUX.2 ne lit pas la sortie du LLM: il empile les etats caches de trois couches
INTERMEDIAIRES (d'ou context_embedder large de 3 x hidden). Tout ce qui vient apres la
derniere couche lue -- huit blocs d'un Qwen3-8B et une projection sur 152 000 jetons --
est calcule a chaque image puis jete. Comme hidden_states[k] est la sortie APRES k
blocs, les retirer est EXACT, pas approche: verifie bit a bit sur le modele reel
(15.3 -> 11.2 Go, encodage 5.0 -> 3.2 s, torch.equal vrai sur les trois couches lues).

Ces tests verrouillent les deux pieges rencontres en l'ecrivant:

1. LE NOM DE LA METHODE. La premiere version cherchait `_get_qwen_prompt_embeds`;
   diffusers l'appelle ici `_get_qwen3_prompt_embeds`. L'elagage ne se declenchait donc
   jamais -- et il le DISAIT, mais personne ne lit un log quand un chiffre plus bas est
   deja faux. On cherche desormais la methode par son PARAMETRE.

2. LE BUDGET SUR UNE INTENTION. Pire consequence du point 1: le budget VRAM defalquait
   les 4 Go de l'elagage sans verifier qu'il avait eu lieu. Mesure: 29.7 Go annonces,
   32.3 Go reellement residents, 0.0 Go libre -- et sous Windows ca ne plante meme pas,
   ca deborde en memoire partagee et le rendu s'effondre en silence. Le budget ne suit
   donc plus que le drapeau _ENCODER_TRIMMED, jamais TRIM_TEXT_ENCODER.

Run:  .venv/Scripts/python tests/test_encoder_trim.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import cz_pipeline as P


class _FakeEncoder(torch.nn.Module):
    """Le strict necessaire: .model.layers, .lm_head, des parametres a compter."""

    def __init__(self, n=36, width=8):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [torch.nn.Linear(width, width) for _ in range(n)])
        self.lm_head = torch.nn.Linear(width, 512)


def _fake_pipe(layers=(9, 18, 27), n=36, method="_get_qwen3_prompt_embeds"):
    """Un pipe dont la CLASSE porte la methode, comme chez diffusers."""
    def _embeds(prompt, tokenizer, text_encoder, hidden_states_layers=layers):
        return None

    cls = type("FakePipe", (), {method: staticmethod(_embeds)})
    p = cls()
    p.text_encoder = _FakeEncoder(n=n)
    return p


def test_the_read_layers_are_found_by_parameter_not_by_name():
    """Le piege exact: la methode s'appelle _get_qwen3_..., pas _get_qwen_...."""
    for name in ("_get_qwen3_prompt_embeds", "_get_qwen_prompt_embeds",
                 "_get_some_future_name_embeds"):
        p = _fake_pipe(method=name)
        assert P._encoder_layers_used(p) == 27, name
    print("OK test_the_read_layers_are_found_by_parameter_not_by_name")


def test_an_unreadable_signature_trims_nothing():
    """Sans information fiable, on garde TOUT: des embeddings faux en silence seraient
    infiniment pires que quelques gigaoctets gaspilles."""
    cls = type("NoSuchMethod", (), {})
    p = cls()
    p.text_encoder = _FakeEncoder()
    assert P._encoder_layers_used(p) is None
    P._trim_text_encoder(p)
    assert len(p.text_encoder.model.layers) == 36
    assert P._ENCODER_TRIMMED is False, "le budget ne doit rien defalquer"
    print("OK test_an_unreadable_signature_trims_nothing")


def test_trimming_keeps_exactly_the_blocks_that_are_read():
    p = _fake_pipe()
    kept = list(p.text_encoder.model.layers)[:28]
    P._trim_text_encoder(p)
    assert len(p.text_encoder.model.layers) == 28, len(p.text_encoder.model.layers)
    # ce sont bien les MEMES objets, dans l'ordre: on coupe, on ne reconstruit pas
    assert all(a is b for a, b in zip(p.text_encoder.model.layers, kept))
    assert isinstance(p.text_encoder.lm_head, torch.nn.Identity)
    assert P._ENCODER_TRIMMED is True
    print("OK test_trimming_keeps_exactly_the_blocks_that_are_read")


def test_trimming_twice_changes_nothing():
    p = _fake_pipe()
    P._trim_text_encoder(p)
    P._trim_text_encoder(p)
    assert len(p.text_encoder.model.layers) == 28
    print("OK test_trimming_twice_changes_nothing")


def test_the_budget_follows_the_deed_not_the_intent():
    """Le bug qui a coute une carte pleine: 4 Go defalques d'un elagage jamais fait."""
    old_repo, old_flag = P.BASE_REPO, P._ENCODER_TRIMMED
    P.BASE_REPO = "test-only/FLUX.2-klein-9B"
    P._BASE_DIM_CACHE[P.BASE_REPO] = 4096
    P.ZIMAGE_TRANSFORMER = None
    try:
        P._ENCODER_TRIMMED = False
        whole = P._base_vram_need_gb()
        P._ENCODER_TRIMMED = True
        trimmed = P._base_vram_need_gb()
        assert abs((whole - trimmed) - P._ENCODER_TRIM_GB["9B"]) < 1e-6, (whole, trimmed)
        assert whole > trimmed, (whole, trimmed)
    finally:
        P.BASE_REPO, P._ENCODER_TRIMMED = old_repo, old_flag
    print("OK test_the_budget_follows_the_deed_not_the_intent")


def test_a_trimmed_9B_still_gets_the_offload_it_needs():
    """29.7 Go de poids sur une carte de 31.8 ne laissent pas de quoi diffuser. La
    marge est ABSOLUE: un pourcentage se resserre sur les petites cartes, alors que le
    contexte CUDA et les activations coutent la meme chose partout."""
    old = (P.BASE_REPO, P._ENCODER_TRIMMED, P.OFFLOAD_MODE, P.DEVICE, P._total_vram_gb)
    P.BASE_REPO = "test-only/FLUX.2-klein-9B"
    P._BASE_DIM_CACHE[P.BASE_REPO] = 4096
    P.ZIMAGE_TRANSFORMER = None
    P._ENCODER_TRIMMED = True
    P.OFFLOAD_MODE = "none"
    P.DEVICE = "cuda"
    try:
        P._total_vram_gb = lambda: 31.8
        assert P._effective_offload() == "model", "32 Go: il faut garder l'offload"
        P._total_vram_gb = lambda: 48.0
        assert P._effective_offload() == "none", "48 Go: la, ca tient vraiment"
    finally:
        (P.BASE_REPO, P._ENCODER_TRIMMED, P.OFFLOAD_MODE, P.DEVICE,
         P._total_vram_gb) = old
    print("OK test_a_trimmed_9B_still_gets_the_offload_it_needs")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All text-encoder trim tests passed.")
