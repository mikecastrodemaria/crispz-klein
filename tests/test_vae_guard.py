"""A VAE kept among the checkpoints is refused by its name.

Caught in the library on 2026-09-10: `diffusion_pytorch_model.safetensors`, 160 MB,
251 encoder/decoder/bn/post_quant_conv tensors -- FLUX.2's autoencoder, not a
transformer. It went through every guard, appeared in the menu, and crashed on
"Cannot copy out of meta tensor": loaded as a transformer, no weight finds its
place and everything stays on 'meta'. "diffusion_pytorch_model" is the name diffusers gives
to the weights of ANY component, hence the confusion (which was mine first:
I had taken it for a transformer shard).

Two traps these tests lock down, both caught on the real files:
  - the all-in-one BUNDLES (transformer + encoder + VAE) carry the same 251 VAE
    keys. The historical transformer counter did not see their keys prefixed
    'model.diffusion_model.double_blocks.*': reused, it would have made them refused.
  - a T5 text encoder also has 'encoder.*' keys. Only a marker specific to the
    VAE (post_quant_conv, decoder.conv_in) tells it apart.

Run:  .venv/Scripts/python tests/test_vae_guard.py

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P


def _t(shape=(4, 4)):
    return {"dtype": "BF16", "shape": list(shape)}


def _refusal(hdr):
    """_safetensors_unsupported on a synthetic header, with no file."""
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
    """gonzalomoKlein and flux2KleinAIO: transformer + encoder + VAE in one file.
    They load perfectly well -- refusing them would be a regression."""
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
    """'encoder.*' keys with no VAE marker at all: this is not a VAE."""
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
