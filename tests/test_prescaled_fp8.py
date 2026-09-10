"""Poids FP8/INT8 stockes DEJA a l'echelle: le weight_scale fourni ne s'applique pas.

Releve sur la bibliotheque le 2026-09-10: kleinFinalcutFP16FP8_comfyQuant rendait du
bruit colore pour tout prompt. Le fichier stocke les poids tels quels en FP8 et fournit
quand meme weight_scale = amax / 448. Le chargeur multipliait: poids 1 200 a 1 700 fois
trop petits. Critere retenu, mesure sur les 17 fichiers quantifies de la bibliotheque:
max|stocke| / (echelle x plage) vaut 1,03 pour lui, 71 a 1 691 pour les autres.

Run:  .venv/Scripts/python tests/test_prescaled_fp8.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

import cz_pipeline as P

torch.manual_seed(0)
E4 = torch.float8_e4m3fn


def _pair(n=64):
    w = torch.randn(n, n) * 0.02
    return w, (w.abs().max() / 448.0).reshape(())


def test_the_detector_separates_the_two_layouts():
    w, s = _pair()
    regular = (w / s).to(E4)                 # poids / echelle: le FP8 'scaled' normal
    prescaled = w.to(E4)                     # poids tels quels, echelle fournie en plus
    assert not P._stored_at_scale(regular.float(), s, E4)
    assert P._stored_at_scale(prescaled.float(), s, E4)
    # echelle arbitraire sur de petites valeurs (donnees de test synthetiques):
    # rapport tres inferieur a 1, ce n'est PAS le cas 'deja a l'echelle'
    assert not P._stored_at_scale((w * 50).to(E4).float(), torch.tensor(0.5), E4)
    # INT8 normal
    s8 = (w.abs().max() / 127.0).reshape(())
    q8 = torch.round(w / s8).clamp(-127, 127).to(torch.int8)
    assert not P._stored_at_scale(q8.float(), s8, torch.int8)
    # INT8 plein (+-127) avec une echelle proche de 1: rapport ~1 lui aussi, mais la
    # plage est REMPLIE -- c'est un fichier normal (cas de test_quant_formats)
    full = torch.randint(-127, 128, (4, 3), dtype=torch.int8)
    full[0, 0] = 127
    assert not P._stored_at_scale(full.float(), torch.full((4, 1), 0.9), torch.int8)
    # echelles MX (exposant E8M0 en uint8): jamais concernees
    assert not P._stored_at_scale(prescaled.float(), torch.tensor([120], dtype=torch.uint8), E4)
    assert not P._stored_at_scale(prescaled.float(), s, E4, {"format": "mxfp8"})
    print("OK test_the_detector_separates_the_two_layouts")


def _tiny(path, prescaled, both=False):
    """x_embedder deja a l'echelle si `prescaled`; context_embedder aussi si `both`
    (un vrai fichier est homogene: la cle de cache n'echantillonne qu'un tenseur)."""
    w1, s1 = _pair(32)
    w2, s2 = _pair(32)
    sd = {"x_embedder.weight": (w1 if prescaled else w1 / s1).to(E4),
          "x_embedder.weight_scale": s1.float(),
          "context_embedder.weight": (w2 if both else w2 / s2).to(E4),
          "context_embedder.weight_scale": s2.float()}
    save_file(sd, path)
    return w1, w2


def test_the_loader_reads_a_file_mixing_both_layouts():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "mixed.safetensors")
    w1, w2 = _tiny(p, prescaled=True)
    out = P._load_dequant_state_dict(p)
    for k, w in (("x_embedder.weight", w1), ("context_embedder.weight", w2)):
        rel = ((out[k].float() - w).norm() / w.norm()).item()
        assert rel < 0.1, (k, rel)      # sans le correctif: ~1.0 sur x_embedder
    print("OK test_the_loader_reads_a_file_mixing_both_layouts")


def test_the_cache_key_changes_only_for_prescaled_files():
    d = tempfile.mkdtemp()
    cache = tempfile.mkdtemp()
    pre, reg = os.path.join(d, "pre.safetensors"), os.path.join(d, "reg.safetensors")
    _tiny(pre, prescaled=True, both=True)
    _tiny(reg, prescaled=False)
    old = P._DQ_CACHE_CFG
    try:
        P._DQ_CACHE_CFG = cache
        assert P._dequant_cache_path(pre) != P._dequant_cache_path(pre, legacy=True)
        assert P._dequant_cache_path(reg) == P._dequant_cache_path(reg, legacy=True), \
            "un fichier normal perdrait son cache pour rien"
        # le nouveau cache ecrit, l'ancien (faux) du meme fichier disparait
        stale = P._dequant_cache_path(pre, legacy=True)
        with open(stale, "wb") as f:
            f.write(b"x")
        P._dequant_cache_store(pre, {"x": torch.zeros(1)})
        assert os.path.isfile(P._dequant_cache_path(pre))
        assert not os.path.exists(stale), "cache faux laisse sur le disque"
    finally:
        P._DQ_CACHE_CFG = old
    print("OK test_the_cache_key_changes_only_for_prescaled_files")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All prescaled-FP8 tests passed.")
