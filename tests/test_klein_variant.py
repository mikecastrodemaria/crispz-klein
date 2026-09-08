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
    # la raison par fichier reste COURTE: elle est repetee autant de fois qu'il y a
    # de fichiers ecartes. Le mode d'emploi va dans le resume, une seule fois.
    assert len(msg) < 90, f"raison trop longue ({len(msg)} car.): {msg}"
    assert "NON-COMMERCIAL" not in msg

    # base 9B: la symetrie doit tenir
    assert _with_base("base-9b", 4096, lambda: P._safetensors_unsupported(p9)) is None
    msg = _with_base("base-9b", 4096, lambda: P._safetensors_unsupported(p4))
    assert msg and "4B" in msg, msg
    print("OK test_mismatch_is_refused_both_ways")


def test_summary_carries_the_instructions_once():
    """Le resume porte la clef a changer ET la licence du 9B - une seule fois."""
    line = _with_base("base-4b", 3072, lambda: P._variant_skip_summary(11, 4096))
    assert "11 checkpoint" in line, line
    assert "FLUX.2-klein-9B" in line and "FLUX.2-klein-4B" in line, line
    assert P.CFG_MODEL_KEY in line, f"la clef de config doit etre nommee: {line}"
    assert "zimage_model" not in line, "l ancien nom ne doit plus apparaitre"
    assert "NON-COMMERCIAL" in line, line
    # sens inverse: pas de note de licence quand on ecarte du 4B
    line = _with_base("base-9b", 4096, lambda: P._variant_skip_summary(2, 3072))
    assert "NON-COMMERCIAL" not in line, line
    print("OK test_summary_carries_the_instructions_once")


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


# ---------------------------------------------------------------------------
# Cote LoRA. Une LoRA ne contient aucun poids du modele -- mais ses deux matrices en
# gardent la trace: lora_A a la forme [rang, entree], lora_B [sortie, rang]. Sur une
# projection dont l'entree EST la dimension cachee, la forme la donne donc.
# Sans cette garde, une LoRA 4B posee sur une base 9B faisait deverser a peft quarante
# lignes de "size mismatch ... torch.Size([27648, 128]) ... torch.Size([36864, 128])",
# ou rien ne dit que 27648 = 9 x 3072, donc 4B. Releve en vrai sur une LoRA d'EDITION,
# qui faisait echouer toute l'edition sans jamais nommer la cause.
# ---------------------------------------------------------------------------

def _lora(name, dim, rank=128):
    """Fausse LoRA FLUX.2: seules les formes comptent."""
    p = os.path.join(TMP, name)
    save_file({
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(rank, dim),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(dim, rank),
        "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight":
            torch.zeros(rank, dim),
        "transformer.single_transformer_blocks.0.attn.to_out.lora_B.weight":
            torch.zeros(dim, rank),
    }, p)
    return p


def test_a_lora_declares_its_variant_through_its_shapes():
    for label, dim in DIMS.items():
        p = _lora(f"lora_{label}.safetensors", dim)
        assert P._flux2_lora_hidden_dim(p) == dim, (label, P._flux2_lora_hidden_dim(p))
    print("OK test_a_lora_declares_its_variant_through_its_shapes")


def test_a_4B_lora_on_a_9B_base_is_refused_in_one_sentence():
    old = P.BASE_REPO
    P.BASE_REPO = "test-only/FLUX.2-klein-9B"
    P._BASE_DIM_CACHE[P.BASE_REPO] = 4096
    try:
        why = P._lora_unsupported(_lora("lora_4B_on_9B.safetensors", 3072))
        assert why, "une LoRA 4B doit etre refusee sur une base 9B"
        assert "4B" in why and "9B" in why, why
        assert "3072" in why and "4096" in why, why      # les deux nombres, nommes
        assert "switch the base model" in why, why       # et quoi faire
        # controle: la bonne variante passe
        assert P._lora_unsupported(_lora("lora_9B_on_9B.safetensors", 4096)) is None
    finally:
        P.BASE_REPO = old
    print("OK test_a_4B_lora_on_a_9B_base_is_refused_in_one_sentence")


def test_a_lora_without_a_signature_is_never_filtered():
    """On ne connait pas toutes les LoRA du monde: sans signature reconnue, on ne
    filtre PAS. Ecarter une LoRA valide serait pire que le message qu'on remplace."""
    p = os.path.join(TMP, "lora_exotic.safetensors")
    save_file({"some.other.arch.lora_A.weight": torch.zeros(8, 999),
               "some.other.arch.lora_B.weight": torch.zeros(999, 8)}, p)
    assert P._flux2_lora_hidden_dim(p) is None
    assert P._lora_unsupported(p) is None
    print("OK test_a_lora_without_a_signature_is_never_filtered")


def test_a_lokr_has_no_lora_signature():
    """Un LoKr n'a ni lora_A ni lora_B: la garde de variante doit le laisser passer,
    c'est la fusion (_merge_lokr) qui le prend en charge."""
    p = os.path.join(TMP, "lokr_no_sig.safetensors")
    save_file({f"diffusion_model.double_blocks.{i}.img_attn.proj.lokr_w{j}":
               torch.zeros(4, 4) for i in range(3) for j in (1, 2)}, p)
    assert P._flux2_lora_hidden_dim(p) is None
    print("OK test_a_lokr_has_no_lora_signature")


if __name__ == "__main__":
    for fn in (test_hidden_dim_read_from_header, test_mismatch_is_refused_both_ways,
               test_summary_carries_the_instructions_once,
               test_unknown_base_never_discards, test_bogus_shape_is_not_taken_for_a_hidden_dim,
               test_file_without_signature_is_not_filtered,
               test_a_lora_declares_its_variant_through_its_shapes,
               test_a_4B_lora_on_a_9B_base_is_refused_in_one_sentence,
               test_a_lora_without_a_signature_is_never_filtered,
               test_a_lokr_has_no_lora_signature):
        fn()
    print("All 4B/9B variant tests passed.")
