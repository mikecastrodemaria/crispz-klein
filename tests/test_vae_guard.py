"""Un VAE range parmi les checkpoints est refuse par son nom.

Releve sur la bibliotheque le 2026-09-10: `diffusion_pytorch_model.safetensors`, 160 Mo,
251 tenseurs encoder/decoder/bn/post_quant_conv -- l'autoencodeur de FLUX.2, pas un
transformer. Il passait toutes les gardes, apparaissait dans le menu, et plantait sur
"Cannot copy out of meta tensor": charge comme transformer, aucun poids ne trouve sa
place et tout reste sur 'meta'. "diffusion_pytorch_model" est le nom que diffusers donne
aux poids de N'IMPORTE QUEL composant, d'ou la confusion (qui a d'abord ete la mienne:
je l'avais pris pour un shard de transformer).

Deux pieges que ces tests verrouillent, tous deux releves sur les vrais fichiers:
  - les BUNDLES tout-en-un (transformer + encodeur + VAE) portent les memes 251 cles de
    VAE. Le compteur de transformer historique ne voyait pas leurs cles prefixees
    'model.diffusion_model.double_blocks.*': reutilise, il les aurait fait refuser.
  - un encodeur texte T5 a lui aussi des cles 'encoder.*'. Seul un marqueur propre au
    VAE (post_quant_conv, decoder.conv_in) l'en distingue.

Run:  .venv/Scripts/python tests/test_vae_guard.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P


def _t(shape=(4, 4)):
    return {"dtype": "BF16", "shape": list(shape)}


def _refusal(hdr):
    """_safetensors_unsupported sur un en-tete synthetique, sans fichier."""
    real = P._safetensors_header
    P._safetensors_header = lambda _p: hdr
    try:
        return P._safetensors_unsupported("fake.safetensors")
    finally:
        P._safetensors_header = real


VAE = {
    **{f"decoder.up_blocks.{i}.resnets.0.conv1.weight": _t() for i in range(6)},
    **{f"encoder.down_blocks.{i}.resnets.0.conv1.weight": _t() for i in range(6)},
    "decoder.conv_in.weight": _t(),
    "post_quant_conv.weight": _t(),
    "quant_conv.weight": _t(),
    "bn.running_mean": _t((4,)),
}


def test_a_bare_vae_is_refused_by_name():
    why = _refusal(VAE)
    assert why and "VAE" in why, why
    assert "base repo" in why, why
    print("OK test_a_bare_vae_is_refused_by_name")


def test_an_all_in_one_bundle_is_not_taken_for_a_vae():
    """gonzalomoKlein et flux2KleinAIO: transformer + encodeur + VAE dans un fichier.
    Ils chargent tres bien -- les refuser serait une regression."""
    bundle = {f"vae.{k}": v for k, v in VAE.items()}
    bundle.update({f"model.diffusion_model.double_blocks.{i}.img_attn.qkv.weight": _t()
                   for i in range(4)})
    bundle.update({f"model.diffusion_model.single_blocks.{i}.linear1.weight": _t()
                   for i in range(4)})
    bundle.update({f"text_encoders.qwen3_8b.model.layers.{i}.mlp.up_proj.weight": _t()
                   for i in range(4)})
    why = _refusal(bundle)
    assert not (why and "VAE" in why), why
    print("OK test_an_all_in_one_bundle_is_not_taken_for_a_vae")


def test_a_t5_text_encoder_is_not_called_a_vae():
    """Des cles 'encoder.*' sans aucun marqueur de VAE: ce n'est pas un VAE."""
    t5 = {f"encoder.block.{i}.layer.0.SelfAttention.q.weight": _t() for i in range(12)}
    t5["shared.weight"] = _t()
    why = _refusal(t5)
    assert not (why and "VAE" in why), why
    print("OK test_a_t5_text_encoder_is_not_called_a_vae")


def test_a_real_transformer_still_passes():
    dit = {f"transformer_blocks.{i}.attn.to_q.weight": _t() for i in range(8)}
    dit["x_embedder.weight"] = _t()
    assert _refusal(dit) is None, _refusal(dit)
    print("OK test_a_real_transformer_still_passes")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All VAE-guard tests passed.")
