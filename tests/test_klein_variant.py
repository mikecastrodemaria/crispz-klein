"""The 4B and the 9B share the architecture AND the tensor names.

So the architecture guard lets both through, and a 9B loaded in a 4B pipeline
blew up after reading gigabytes, on an unreadable diffusers
message: "expected shape [18432, 3072], but got [24576, 4096]".
Only the HIDDEN DIMENSION separates them -- readable from the header, without loading a
single weight.

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

import cz_pipeline as P

TMP = os.path.join(os.environ.get("TEMP") or "/tmp", "cz_variant")
os.makedirs(TMP, exist_ok=True)

# (the hidden dim, the name) — 3072 = klein-4B, 4096 = klein-9B
DIMS = {"4B": 3072, "9B": 4096}


def _ckpt(name, dim, key="double_stream_modulation_img.lin.weight"):
    """A fake transformer: only the header counts, the weights are empty."""
    p = os.path.join(TMP, name)
    save_file({key: torch.zeros(dim * 6, dim),
               "single_transformer_blocks.0.attn.to_q.weight": torch.zeros(2, 2),
               "x_embedder.weight": torch.zeros(2, 2)}, p)
    return p


def test_hidden_dim_read_from_header():
    for label, dim in DIMS.items():
        p = _ckpt(f"{label}.safetensors", dim)
        assert P._flux2_hidden_dim(p) == dim, (label, P._flux2_hidden_dim(p))
    # the diffusers layout writes '.linear.' instead of '.lin.'
    p = _ckpt("diffusers_layout.safetensors", 3072,
              key="double_stream_modulation_img.linear.weight")
    assert P._flux2_hidden_dim(p) == 3072
    # the ComfyUI prefix
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

    # a 4B base: the 9B is refused, the 4B goes through
    assert _with_base("base-4b", 3072, lambda: P._safetensors_unsupported(p4)) is None
    msg = _with_base("base-4b", 3072, lambda: P._safetensors_unsupported(p9))
    assert msg and "9B" in msg and "4B" in msg, msg
    # the per-file reason stays SHORT: it is repeated as many times as there are
    # files discarded. The instructions go into the summary, once.
    assert len(msg) < 90, f"raison trop longue ({len(msg)} car.): {msg}"
    assert "NON-COMMERCIAL" not in msg

    # a 9B base: the symmetry must hold
    assert _with_base("base-9b", 4096, lambda: P._safetensors_unsupported(p9)) is None
    msg = _with_base("base-9b", 4096, lambda: P._safetensors_unsupported(p4))
    assert msg and "4B" in msg, msg
    print("OK test_mismatch_is_refused_both_ways")


def test_summary_carries_the_instructions_once():
    """The summary carries the key to change AND the 9B's licence - once."""
    line = _with_base("base-4b", 3072, lambda: P._variant_skip_summary(11, 4096))
    assert "11 checkpoint" in line, line
    assert "FLUX.2-klein-9B" in line and "FLUX.2-klein-4B" in line, line
    assert P.CFG_MODEL_KEY in line, f"la clef de config doit etre nommee: {line}"
    assert "zimage_model" not in line, "l ancien nom ne doit plus apparaitre"
    assert "NON-COMMERCIAL" in line, line
    # the other way round: no licence note when discarding a 4B
    line = _with_base("base-9b", 4096, lambda: P._variant_skip_summary(2, 3072))
    assert "NON-COMMERCIAL" not in line, line
    print("OK test_summary_carries_the_instructions_once")


def test_unknown_base_never_discards():
    """A house rule: we NEVER discard a model on a doubt."""
    p9 = _ckpt("u9.safetensors", 4096)
    assert _with_base("mystere", None, lambda: P._safetensors_unsupported(p9)) is None
    print("OK test_unknown_base_never_discards")


def test_bogus_shape_is_not_taken_for_a_hidden_dim():
    """The name alone is not enough: the structural out/in ratio must be 6 (a double
    stream) or 3 (a single stream), otherwise it is not a hidden dimension. Without that,
    a 2x2 fixture carrying the right name made a valid file be refused."""
    p = os.path.join(TMP, "bogus.safetensors")
    save_file({"double_stream_modulation_img.lin.weight": torch.zeros(2, 2)}, p)
    assert P._flux2_hidden_dim(p) is None
    # the right ratio for the single stream (3) -> recognised
    p = os.path.join(TMP, "single.safetensors")
    save_file({"single_stream_modulation.lin.weight": torch.zeros(9216, 3072)}, p)
    assert P._flux2_hidden_dim(p) == 3072
    print("OK test_bogus_shape_is_not_taken_for_a_hidden_dim")


def test_file_without_signature_is_not_filtered():
    """A checkpoint without the modulation keys (another layout) must not be
    discarded for a variant reason: the dimension is simply unknown."""
    p = os.path.join(TMP, "nosig.safetensors")
    save_file({"transformer_blocks.0.attn.to_q.weight": torch.zeros(2, 2),
               "x_embedder.weight": torch.zeros(2, 2)}, p)
    assert P._flux2_hidden_dim(p) is None
    assert _with_base("base-4b", 3072, lambda: P._flux2_variant_mismatch(None)) is None
    print("OK test_file_without_signature_is_not_filtered")


# ---------------------------------------------------------------------------
# On the LoRA side. A LoRA holds no weight of the model -- but its two matrices keep
# a trace of it: lora_A has the shape [rank, in], lora_B [out, rank]. On a
# projection whose input IS the hidden dimension, the shape therefore gives it.
# Without that guard, a 4B LoRA set on a 9B base made peft pour out forty
# lines of "size mismatch ... torch.Size([27648, 128]) ... torch.Size([36864, 128])",
# where nothing says that 27648 = 9 x 3072, so a 4B. Caught for real on an EDIT LoRA,
# which made the whole edit fail without ever naming the cause.
# ---------------------------------------------------------------------------

# The three dialects CAUGHT in a real library of 70 klein LoRAs. The
# first version of the guard knew only the first one -- and so recognised
# only ONE file out of 70. The others are in the original FLUX layout, and one inserts
# peft's adapter name ('default') between the matrix and '.weight'.
_LORA_LAYOUTS = {
    "diffusers": ("transformer.transformer_blocks.0.attn.to_q.lora_{ab}.weight",
                  "transformer.single_transformer_blocks.0.attn.to_out.lora_{ab}.weight"),
    "flux-original": ("diffusion_model.double_blocks.0.img_attn.proj.lora_{ab}.weight",
                      "diffusion_model.single_blocks.0.linear2.lora_{ab}.weight"),
    "peft-adapter-name": ("single_transformer_blocks.0.attn.to_out.lora_{ab}.default.weight",
                          "transformer_blocks.0.attn.to_q.lora_{ab}.default.weight"),
}


def _lora(name, dim, rank=128, layout="diffusers"):
    """A fake FLUX.2 LoRA: only the shapes count. lora_A = [rank, in],
    lora_B = [out, rank]."""
    sd = {}
    for tmpl in _LORA_LAYOUTS[layout]:
        sd[tmpl.format(ab="A")] = torch.zeros(rank, dim)
        sd[tmpl.format(ab="B")] = torch.zeros(dim, rank)
    p = os.path.join(TMP, name)
    save_file(sd, p)
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
        assert "3072" in why and "4096" in why, why      # both numbers, named
        assert "switch the base model" in why, why       # and what to do
        # the control: the right variant goes through
        assert P._lora_unsupported(_lora("lora_9B_on_9B.safetensors", 4096)) is None
    finally:
        P.BASE_REPO = old
    print("OK test_a_4B_lora_on_a_9B_base_is_refused_in_one_sentence")


def test_a_lora_without_a_signature_is_never_filtered():
    """We do not know every LoRA in the world: with no recognised signature, we do NOT
    filter. Discarding a valid LoRA would be worse than the message we replace."""
    p = os.path.join(TMP, "lora_exotic.safetensors")
    save_file({"some.other.arch.lora_A.weight": torch.zeros(8, 999),
               "some.other.arch.lora_B.weight": torch.zeros(999, 8)}, p)
    assert P._flux2_lora_hidden_dim(p) is None
    assert P._lora_unsupported(p) is None
    print("OK test_a_lora_without_a_signature_is_never_filtered")


def test_a_lokr_has_no_lora_signature():
    """A LoKr has neither lora_A nor lora_B: the variant guard must let it through,
    it is the merging (_merge_lokr) that takes care of it."""
    p = os.path.join(TMP, "lokr_no_sig.safetensors")
    save_file({f"diffusion_model.double_blocks.{i}.img_attn.proj.lokr_w{j}":
               torch.zeros(4, 4) for i in range(3) for j in (1, 2)}, p)
    assert P._flux2_lora_hidden_dim(p) is None
    print("OK test_a_lokr_has_no_lora_signature")


def test_every_lora_dialect_is_recognised():
    """The bug that made the guard inert: it only read the diffusers layout,
    whereas the real library is in the original FLUX layout. Recognising the
    matrix by SEGMENT ('lora_A' somewhere in the key) covers all three."""
    for layout in _LORA_LAYOUTS:
        for label, dim in DIMS.items():
            p = _lora(f"dialect_{layout}_{label}.safetensors", dim, layout=layout)
            got = P._flux2_lora_hidden_dim(p)
            assert got == dim, (layout, label, got)
    print("OK test_every_lora_dialect_is_recognised")


def test_derived_widths_do_not_fool_the_detector():
    """A fused qkv is 3x wide, an mlp 4x. Those values are not in
    _FLUX2_VARIANTS and must therefore be ignored, not taken for a dimension."""
    p = os.path.join(TMP, "derived.safetensors")
    save_file({
        # only that pair carries the real dimension (4096)
        "diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight": torch.zeros(16, 4096),
        "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight": torch.zeros(4096, 16),
        # these carry 3x and 4x: to be ignored
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight": torch.zeros(12288, 16),
        "diffusion_model.double_blocks.0.img_mlp.0.lora_B.weight": torch.zeros(16384, 16),
    }, p)
    assert P._flux2_lora_hidden_dim(p) == 4096, P._flux2_lora_hidden_dim(p)
    print("OK test_derived_widths_do_not_fool_the_detector")


if __name__ == "__main__":
    for fn in (test_hidden_dim_read_from_header, test_mismatch_is_refused_both_ways,
               test_summary_carries_the_instructions_once,
               test_unknown_base_never_discards, test_bogus_shape_is_not_taken_for_a_hidden_dim,
               test_file_without_signature_is_not_filtered,
               test_a_lora_declares_its_variant_through_its_shapes,
               test_a_4B_lora_on_a_9B_base_is_refused_in_one_sentence,
               test_a_lora_without_a_signature_is_never_filtered,
               test_a_lokr_has_no_lora_signature,
               test_every_lora_dialect_is_recognised,
               test_derived_widths_do_not_fool_the_detector):
        fn()
    print("All 4B/9B variant tests passed.")
