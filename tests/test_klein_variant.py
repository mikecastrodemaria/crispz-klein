"""4B et 9B partagent l'architecture ET les noms de tenseurs.

La garde d'architecture les laisse donc passer toutes les deux, et un 9B charge
dans un pipeline 4B explosait apres avoir lu des gigaoctets, sur un message
diffusers illisible: "expected shape [18432, 3072], but got [24576, 4096]".
Seule la DIMENSION CACHEE les separe -- lisible a l'en-tete, sans charger un poids.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

import cz_pipeline as P

TMP = os.path.join(os.environ.get("TEMP") or "/tmp", "cz_variant")
os.makedirs(TMP, exist_ok=True)

# (dim cachee, nom) — 3072 = klein-4B, 4096 = klein-9B
DIMS = {"4B": 3072, "9B": 4096}


def _ckpt(name, dim, key="double_stream_modulation_img.lin.weight"):
    """Faux transformer: seul l'en-tete compte, les poids sont vides."""
    p = os.path.join(TMP, name)
    save_file({key: torch.zeros(dim * 6, dim),
               "single_transformer_blocks.0.attn.to_q.weight": torch.zeros(2, 2),
               "x_embedder.weight": torch.zeros(2, 2)}, p)
    return p


def test_hidden_dim_read_from_header():
    for label, dim in DIMS.items():
        p = _ckpt(f"{label}.safetensors", dim)
        assert P._flux2_hidden_dim(p) == dim, (label, P._flux2_hidden_dim(p))
    # le layout diffusers ecrit '.linear.' au lieu de '.lin.'
    p = _ckpt("diffusers_layout.safetensors", 3072,
              key="double_stream_modulation_img.linear.weight")
    assert P._flux2_hidden_dim(p) == 3072
    # prefixe ComfyUI
    p = _ckpt("comfy.safetensors", 4096,
              key="model.diffusion_model.double_stream_modulation_img.lin.weight")
    assert P._flux2_hidden_dim(p) == 4096
    print("OK test_hidden_dim_read_from_header")


def _with_base(repo, dim, fn):
    old_repo, old_cache = P.BASE_REPO, dict(P._BASE_DIM_CACHE)
    P.BASE_REPO = repo
    P._BASE_DIM_CACHE[repo] = dim
    try:
        return fn()
    finally:
        P.BASE_REPO = old_repo
        P._BASE_DIM_CACHE.clear()
        P._BASE_DIM_CACHE.update(old_cache)


def test_mismatch_is_refused_both_ways():
    p4 = _ckpt("m4.safetensors", 3072)
    p9 = _ckpt("m9.safetensors", 4096)

    # base 4B: le 9B est refuse, le 4B passe
    assert _with_base("base-4b", 3072, lambda: P._safetensors_unsupported(p4)) is None
    msg = _with_base("base-4b", 3072, lambda: P._safetensors_unsupported(p9))
    assert msg and "9B" in msg and "4B" in msg, msg
    assert "NON-COMMERCIAL" in msg, "la licence du 9B doit etre signalee"

    # base 9B: la symetrie doit tenir
    assert _with_base("base-9b", 4096, lambda: P._safetensors_unsupported(p9)) is None
    msg = _with_base("base-9b", 4096, lambda: P._safetensors_unsupported(p4))
    assert msg and "4B" in msg, msg
    print("OK test_mismatch_is_refused_both_ways")


def test_unknown_base_never_discards():
    """Regle maison: on n'ecarte JAMAIS un modele sur un doute."""
    p9 = _ckpt("u9.safetensors", 4096)
    assert _with_base("mystere", None, lambda: P._safetensors_unsupported(p9)) is None
    print("OK test_unknown_base_never_discards")


def test_bogus_shape_is_not_taken_for_a_hidden_dim():
    """Le nom seul ne suffit pas: le ratio structurel out/in doit valoir 6 (double
    stream) ou 3 (single stream), sinon ce n'est pas une dimension cachee. Sans ca,
    une fixture 2x2 portant le bon nom faisait refuser un fichier valide."""
    p = os.path.join(TMP, "bogus.safetensors")
    save_file({"double_stream_modulation_img.lin.weight": torch.zeros(2, 2)}, p)
    assert P._flux2_hidden_dim(p) is None
    # bon ratio pour le single stream (3) -> reconnu
    p = os.path.join(TMP, "single.safetensors")
    save_file({"single_stream_modulation.lin.weight": torch.zeros(9216, 3072)}, p)
    assert P._flux2_hidden_dim(p) == 3072
    print("OK test_bogus_shape_is_not_taken_for_a_hidden_dim")


def test_file_without_signature_is_not_filtered():
    """Un checkpoint sans les cles de modulation (autre layout) ne doit pas etre
    ecarte pour cause de variante: la dimension est simplement inconnue."""
    p = os.path.join(TMP, "nosig.safetensors")
    save_file({"transformer_blocks.0.attn.to_q.weight": torch.zeros(2, 2),
               "x_embedder.weight": torch.zeros(2, 2)}, p)
    assert P._flux2_hidden_dim(p) is None
    assert _with_base("base-4b", 3072, lambda: P._flux2_variant_mismatch(None)) is None
    print("OK test_file_without_signature_is_not_filtered")


if __name__ == "__main__":
    for fn in (test_hidden_dim_read_from_header, test_mismatch_is_refused_both_ways,
               test_unknown_base_never_discards, test_bogus_shape_is_not_taken_for_a_hidden_dim,
               test_file_without_signature_is_not_filtered):
        fn()
    print("All 4B/9B variant tests passed.")
