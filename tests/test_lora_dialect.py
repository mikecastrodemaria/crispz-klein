"""Deux pieges de LoRA propres au fork klein (pipe d'edition == pipe de base).

1. DIALECTE DE CLES. diffusers/peft attend `.lora_A.weight` / `.lora_B.weight`.
   Des LoRA publiees ecrivent `.lora.down.weight` / `.lora.up.weight` -- et
   certaines MELANGENT les deux (lrzjason/Consistance_Edit_Lora: 160 cles PEFT +
   40 cles down/up). peft charge alors ce qu'il reconnait et cree un adaptateur
   NEUF pour le reste: le LoRA s'applique partiellement, SANS erreur.

2. ESPACE DE NOMS DES ADAPTATEURS. Chez l'amont, edition et base sont deux modeles
   distincts. Ici c'est le meme objet: les deux jeux se disputaient `cz_lora_i`
   ("Adapter name cz_lora_0 already in use") et `set_adapters` desactivait
   silencieusement les LoRA de base. _apply_edit_loras synchronise donc l'UNION.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

import cz_pipeline as P

TMP = os.path.join(os.environ.get("TEMP") or "/tmp", "cz_lora_dialect")
os.makedirs(TMP, exist_ok=True)


def _write(name, keys):
    sd = {k: torch.zeros(2, 2) for k in keys}
    p = os.path.join(TMP, name)
    save_file(sd, p)
    return p


def test_detects_the_alternate_dialect():
    peft = _write("peft.safetensors", [
        "transformer.blocks.0.attn.to_q.lora_A.weight",
        "transformer.blocks.0.attn.to_q.lora_B.weight"])
    assert P._lora_needs_normalizing(peft) is False

    for alt in (("lora.down", "lora.up"), ("lora_down", "lora_up")):
        p = _write(f"alt_{alt[0]}.safetensors".replace(".", "_", 1), [
            f"transformer.blocks.0.attn.to_q.{alt[0]}.weight",
            f"transformer.blocks.0.attn.to_q.{alt[1]}.weight"])
        assert P._lora_needs_normalizing(p) is True, alt
    print("OK test_detects_the_alternate_dialect")


def test_normalizes_a_mixed_file():
    """Le cas reel: un fichier qui melange les deux dialectes."""
    p = _write("mixed.safetensors", [
        "transformer.single_transformer_blocks.0.attn.to_out.lora_A.weight",
        "transformer.single_transformer_blocks.0.attn.to_out.lora_B.weight",
        "transformer.transformer_blocks.0.attn.to_k.lora.down.weight",
        "transformer.transformer_blocks.0.attn.to_k.lora.up.weight"])
    assert P._lora_needs_normalizing(p) is True
    sd, n = P._load_lora_normalized(p)
    assert n == 2, n                       # seules les 2 cles down/up sont renommees
    assert set(sd) == {
        "transformer.single_transformer_blocks.0.attn.to_out.lora_A.weight",
        "transformer.single_transformer_blocks.0.attn.to_out.lora_B.weight",
        "transformer.transformer_blocks.0.attn.to_k.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_k.lora_B.weight"}, sorted(sd)
    assert len(sd) == 4, "aucune cle ne doit etre perdue ni ecrasee"
    print("OK test_normalizes_a_mixed_file")


def test_edit_loras_sync_the_union_with_the_base_set():
    """Regression: un LoRA de base + un preset d'edition -> 'Adapter name cz_lora_0
    already in use', edition impossible. _apply_edit_loras doit poser l'UNION."""
    calls = {}

    def fake_sync(pipe, wanted, applied, force=False, tag="LoRA"):
        calls["wanted"] = list(wanted)
        return True, list(wanted)

    old_sync, old_loras, old_edit, old_en = (
        P._sync_adapters, list(P.LORAS), list(P.EDIT_LORAS), P.EDIT_LORAS_ENABLED)
    P._sync_adapters = fake_sync
    P.LORAS = [("/base/style.safetensors", 0.8)]
    P.EDIT_LORAS = [("/edit/consist.safetensors", 0.6)]
    P.EDIT_LORAS_ENABLED = True
    try:
        P._apply_edit_loras(object())
    finally:
        (P._sync_adapters, P.LORAS, P.EDIT_LORAS,
         P.EDIT_LORAS_ENABLED) = old_sync, old_loras, old_edit, old_en

    assert calls["wanted"] == [("/base/style.safetensors", 0.8),
                               ("/edit/consist.safetensors", 0.6)], calls["wanted"]
    print("OK test_edit_loras_sync_the_union_with_the_base_set")


def test_union_dedupes_on_path_first_weight_wins():
    calls = {}

    def fake_sync(pipe, wanted, applied, force=False, tag="LoRA"):
        calls["wanted"] = list(wanted)
        return True, list(wanted)

    old = (P._sync_adapters, list(P.LORAS), list(P.EDIT_LORAS), P.EDIT_LORAS_ENABLED)
    P._sync_adapters = fake_sync
    P.LORAS = [("/same.safetensors", 0.2)]
    P.EDIT_LORAS = [("/same.safetensors", 0.9)]
    P.EDIT_LORAS_ENABLED = True
    try:
        P._apply_edit_loras(object())
    finally:
        (P._sync_adapters, P.LORAS, P.EDIT_LORAS, P.EDIT_LORAS_ENABLED) = old
    assert calls["wanted"] == [("/same.safetensors", 0.2)], calls["wanted"]
    print("OK test_union_dedupes_on_path_first_weight_wins")


# ---------------------------------------------------------------------------
# 3. LyCORIS (LoKr / LoHa). La mise a jour y est factorisee en produit de Kronecker
#    (LoKr) ou de Hadamard (LoHa), et diffusers n'en convertit ni l'un ni l'autre.
#    Le piege etait qu'il n'y avait AUCUNE erreur: les cles d'ai-toolkit s'appellent
#    'diffusion_model.<module>.lokr_w1' -- ni '.lora_A/B' ni le prefixe 'lora_unet_'
#    -- donc la garde checkpoint les prenait pour un modele, et la garde LoRA les
#    passait telles quelles a peft, qui n'appliquait rien en silence.
#    Le LoKr est desormais supporte par FUSION dans les poids (cf. test_lokr_merge);
#    ces tests-ci ne verifient plus que l'AIGUILLAGE. Le LoHa, lui, reste refuse.
#    Releve sur Ashen3/SNOFS (Klein9b, 112 couches x w1/w2/alpha).
# ---------------------------------------------------------------------------

def _lycoris(name, suffixes):
    sd = {}
    for i in range(6):
        b = f"diffusion_model.double_blocks.{i}.img_attn.proj"
        sd[b + ".alpha"] = torch.tensor(1.0)
        for s in suffixes:
            sd[f"{b}.{s}"] = torch.zeros(4, 4)
    p = os.path.join(TMP, name)
    save_file(sd, p)
    return p


def test_a_lokr_is_routed_to_the_lora_folder():
    """Depuis 1.27.0 le LoKr est SUPPORTE, par fusion dans les poids: il n'est donc
    refuse que la ou il ne va pas -- le dossier des checkpoints -- et le refus dit ou
    le mettre. La fusion elle-meme est couverte par tests/test_lokr_merge.py."""
    p = _lycoris("snofs_like_lokr.safetensors", ("lokr_w1", "lokr_w2"))
    why = P._safetensors_unsupported(p)
    assert why and "LoKr" in why, why
    assert "LoRA folder" in why, why
    assert P._lora_unsupported(p) is None, P._lora_unsupported(p)
    print("OK test_a_lokr_is_routed_to_the_lora_folder")


def test_a_loha_is_named_as_such():
    p = _lycoris("loha.safetensors", ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"))
    why = P._safetensors_unsupported(p)
    assert why and "LoHa" in why, why
    print("OK test_a_loha_is_named_as_such")


def test_a_real_peft_lora_still_passes():
    """Le controle: la garde ne doit toucher a rien de ce qui marchait."""
    p = _write("real_peft.safetensors",
               [f"transformer.blocks.{i}.attn.to_q.lora_{ab}.weight"
                for i in range(6) for ab in ("A", "B")])
    assert P._lora_unsupported(p) is None
    assert P._safetensors_unsupported(p) is not None   # une LoRA reste refusee en checkpoint
    print("OK test_a_real_peft_lora_still_passes")


if __name__ == "__main__":
    for fn in (test_detects_the_alternate_dialect, test_normalizes_a_mixed_file,
               test_edit_loras_sync_the_union_with_the_base_set,
               test_union_dedupes_on_path_first_weight_wins,
               test_a_lokr_is_routed_to_the_lora_folder,
               test_a_loha_is_named_as_such,
               test_a_real_peft_lora_still_passes):
        fn()
    print("All LoRA dialect / namespace tests passed.")
