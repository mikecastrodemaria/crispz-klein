"""Pre-filling the dequant cache does not depend on the current base repo.

The cache key is the FILE (path+size+mtime): a 4B checkpoint dequantises
exactly the same whether the base is a 4B or a 9B. The variant refusal is a
LOADING refusal, not a conversion one -- discarding it from the pre-filling made one
pay the minutes of conversion again on every 4B <-> 9B switch, which is precisely what
this cache exists to avoid.

So tools/rebuild_dequant_cache.py neutralises that refusal, and it ALONE, by comparing
the reason _safetensors_unsupported returns with _flux2_variant_mismatch's.
These tests lock that equality down: should the variant reason one day be composed
with something else, the pre-filling would start skipping valid files again --
or, worse, would stop recognising a real refusal and would convert LoRAs.

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

import cz_pipeline as P

TMP = os.path.join(os.environ.get("TEMP") or "/tmp", "cz_precache")
os.makedirs(TMP, exist_ok=True)

# A 9B base fixed by hand: _BASE_DIM_CACHE short-circuits the reading of
# transformer/config.json -> the test depends neither on the network nor on the local config.
BASE_9B = "test-only/FLUX.2-klein-9B"
P._BASE_DIM_CACHE[BASE_9B] = 4096
P.BASE_REPO = BASE_9B

FP8 = torch.float8_e4m3fn


def _fp8_ckpt(name, dim):
    """A fake 'scaled' FP8 transformer, ComfyUI-style: only the header counts."""
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
    """A LoRA gone astray in the checkpoints folder: a real refusal, on every base."""
    p = os.path.join(TMP, name)
    save_file({f"lora_unet_blocks_{i}.lora_down.weight": torch.zeros(2, 2)
               for i in range(6)}, p)
    return p


def test_a_4B_checkpoint_is_refused_only_for_its_variant():
    """The base is a 9B: a 4B is refused at load time, and that is ALL we hold
    against it -- so the pre-filling can convert it anyway."""
    p = _fp8_ckpt("precache_4B.safetensors", 3072)
    dim = P._flux2_hidden_dim(p)
    assert dim == 3072, dim
    why = P._safetensors_unsupported(p)
    assert why, "un 4B doit etre refuse tant que la base tourne en 9B"
    assert why == P._flux2_variant_mismatch(dim), why
    # ... and it stays perfectly dequantisable.
    assert P._safetensors_dequant(p) == "FP8 scaled", P._safetensors_dequant(p)


def test_a_9B_checkpoint_is_not_refused_at_all():
    p = _fp8_ckpt("precache_9B.safetensors", 4096)
    assert P._flux2_hidden_dim(p) == 4096
    assert P._safetensors_unsupported(p) is None, P._safetensors_unsupported(p)
    assert P._safetensors_dequant(p) == "FP8 scaled"


def test_a_real_refusal_is_never_mistaken_for_a_variant_mismatch():
    """The pre-filling's safety net: a LoRA has no dimension signature, so
    _flux2_variant_mismatch returns None, so the equality cannot clear it."""
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
