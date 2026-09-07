"""MXFP8: l'echelle est un EXPOSANT, pas un multiplicateur.

Le checkpoint FLUX.2 'snofs14Flux2Klein9b_14Distilled' est quantifie en mxfp8
(OCP microscaling, group_size 32). Son echelle est un uint8 codant un exposant
E8M0 -> facteur reel 2^(s-127). Deux bugs successifs y menaient:
  1. l'echelle [out, nb] faisait planter le broadcast au fond de torch, sur
     "The size of tensor a (4096) must match the size of tensor b (128)";
  2. lue comme un multiplicateur lineaire (~115 au lieu de 2^-12), elle donnait
     des poids 470000x trop grands -- std 17093 au lieu de 0.024 -- soit une
     image lissee, sans qu'aucune etape ne signale quoi que ce soit.

Run:  .venv/Scripts/python tests/test_mxfp8_scale.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import cz_pipeline as P


def test_mxfp8_scale_is_an_exponent():
    w = torch.full((4, 64), 8.0)
    s = torch.full((4, 2), 127 + 3, dtype=torch.uint8)      # 2^3 = 8
    out = P._apply_quant_scale(w.clone(), s, "k.weight", "f.safetensors",
                               {"format": "mxfp8", "group_size": 32})
    assert torch.allclose(out, torch.full((4, 64), 64.0)), out[0, :3]
    # sans metadonnees, un uint8 ne peut etre qu'un exposant: meme resultat
    out2 = P._apply_quant_scale(w.clone(), s, "k.weight", "f.safetensors", None)
    assert torch.equal(out, out2)
    print("OK test_mxfp8_scale_is_an_exponent")


def test_groups_run_along_the_input_dimension():
    """group_size 32: chaque echelle couvre 32 colonnes consecutives."""
    w = torch.ones(1, 64)
    s = torch.tensor([[127 + 1, 127 + 2]], dtype=torch.uint8)   # 2 et 4
    out = P._apply_quant_scale(w, s, "k.weight", "f", {"format": "mxfp8"})
    assert out[0, :32].eq(2).all() and out[0, 32:].eq(4).all(), out[0, ::16]
    print("OK test_groups_run_along_the_input_dimension")


def test_linear_float_scales_are_untouched():
    """Les FP8/INT8 'scaled' classiques restent multiplicatifs."""
    w = torch.full((3, 4), 2.0)
    assert torch.equal(P._apply_quant_scale(w.clone(), torch.tensor(3.0), "k", "f"),
                       torch.full((3, 4), 6.0))
    per_row = torch.tensor([1.0, 2.0, 3.0])
    got = P._apply_quant_scale(w.clone(), per_row, "k", "f")
    assert torch.equal(got, w * per_row.unsqueeze(1))
    print("OK test_linear_float_scales_are_untouched")


def test_an_unknown_layout_is_named_not_crashed():
    """Un format inconnu doit se dire avec ses dimensions, pas exploser dans torch."""
    try:
        P._apply_quant_scale(torch.ones(6, 8), torch.ones(3, 5), "bad.weight",
                             "lib/f.safetensors")
    except RuntimeError as e:
        assert "f.safetensors" in str(e) and "(6, 8)" in str(e) and "(3, 5)" in str(e), e
        print("OK test_an_unknown_layout_is_named_not_crashed")
        return
    raise AssertionError("aucune erreur levee")


if __name__ == "__main__":
    test_mxfp8_scale_is_an_exponent()
    test_groups_run_along_the_input_dimension()
    test_linear_float_scales_are_untouched()
    test_an_unknown_layout_is_named_not_crashed()
    print("ALL OK")
