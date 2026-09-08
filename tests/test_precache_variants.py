"""Pre-remplir le cache de dequant ne depend pas du repo de base courant.

La clef du cache est le FICHIER (chemin+taille+mtime): un checkpoint 4B se
dequantifie exactement pareil que la base soit 4B ou 9B. Le refus de variante est un
refus de CHARGEMENT, pas de conversion -- l'ecarter du pre-remplissage faisait
repayer les minutes de conversion a chaque bascule 4B <-> 9B, ce que ce cache existe
precisement pour eviter.

tools/rebuild_dequant_cache.py neutralise donc ce refus-la, et lui SEUL, en comparant
la raison rendue par _safetensors_unsupported a celle de _flux2_variant_mismatch.
Ces tests verrouillent cette egalite: si la raison de variante etait un jour composee
avec autre chose, le pre-remplissage se remettrait a sauter des fichiers valides --
ou, pire, cesserait de reconnaitre un vrai refus et convertirait des LoRA.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

import cz_pipeline as P

TMP = os.path.join(os.environ.get("TEMP") or "/tmp", "cz_precache")
os.makedirs(TMP, exist_ok=True)

# Base 9B fixee a la main: _BASE_DIM_CACHE court-circuite la lecture de
# transformer/config.json -> le test ne depend ni du reseau ni de la config locale.
BASE_9B = "test-only/FLUX.2-klein-9B"
P._BASE_DIM_CACHE[BASE_9B] = 4096
P.BASE_REPO = BASE_9B

FP8 = torch.float8_e4m3fn


def _fp8_ckpt(name, dim):
    """Faux transformer FP8 'scaled' facon ComfyUI: seul l'en-tete compte."""
    p = os.path.join(TMP, name)
    save_file({
        # signature de dimension: out == dim * 6 sur la modulation double-flux
        "double_stream_modulation_img.lin.weight": torch.zeros(dim * 6, dim, dtype=FP8),
        "single_transformer_blocks.0.attn.to_q.weight": torch.zeros(4, 4, dtype=FP8),
        "single_transformer_blocks.0.attn.to_q.weight_scale": torch.ones(4, 1),
        "x_embedder.weight": torch.zeros(4, 4, dtype=FP8),
    }, p)
    return p


def _lora(name):
    """Une LoRA egaree dans le dossier des checkpoints: vrai refus, toutes bases."""
    p = os.path.join(TMP, name)
    save_file({f"lora_unet_blocks_{i}.lora_down.weight": torch.zeros(2, 2)
               for i in range(6)}, p)
    return p


def test_a_4B_checkpoint_is_refused_only_for_its_variant():
    """La base est en 9B: un 4B est refuse au chargement, et c'est TOUT ce qu'on
    lui reproche -- donc le pre-remplissage peut le convertir quand meme."""
    p = _fp8_ckpt("precache_4B.safetensors", 3072)
    dim = P._flux2_hidden_dim(p)
    assert dim == 3072, dim
    why = P._safetensors_unsupported(p)
    assert why, "un 4B doit etre refuse tant que la base tourne en 9B"
    assert why == P._flux2_variant_mismatch(dim), why
    # ... et il reste parfaitement dequantifiable.
    assert P._safetensors_dequant(p) == "FP8 scaled", P._safetensors_dequant(p)


def test_a_9B_checkpoint_is_not_refused_at_all():
    p = _fp8_ckpt("precache_9B.safetensors", 4096)
    assert P._flux2_hidden_dim(p) == 4096
    assert P._safetensors_unsupported(p) is None, P._safetensors_unsupported(p)
    assert P._safetensors_dequant(p) == "FP8 scaled"


def test_a_real_refusal_is_never_mistaken_for_a_variant_mismatch():
    """Le filet du pre-remplissage: une LoRA n'a pas de signature de dimension, donc
    _flux2_variant_mismatch rend None, donc l'egalite ne peut pas la blanchir."""
    p = _lora("precache_lora.safetensors")
    why = P._safetensors_unsupported(p)
    assert why and "LoRA" in why, why
    assert why != P._flux2_variant_mismatch(P._flux2_hidden_dim(p))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK", name)
    print("tout vert")
