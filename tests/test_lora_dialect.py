"""Two LoRA traps specific to the klein fork (the edit pipe == the base pipe).

1. THE KEY DIALECT. diffusers/peft expects `.lora_A.weight` / `.lora_B.weight`.
   Some published LoRAs write `.lora.down.weight` / `.lora.up.weight` -- and
   some MIX the two (lrzjason/Consistance_Edit_Lora: 160 PEFT keys +
   40 down/up keys). peft then loads what it recognises and creates a NEW
   adapter for the rest: the LoRA applies partially, WITHOUT an error.

2. THE ADAPTERS' NAMESPACE. Upstream, editing and the base are two distinct
   models. Here it is the same object: the two sets fought over `cz_lora_i`
   ("Adapter name cz_lora_0 already in use") and `set_adapters` silently
   disabled the base LoRAs. So _apply_edit_loras synchronises the UNION.

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
    """The real case: a file that mixes both dialects."""
    p = _write("mixed.safetensors", [
        "transformer.single_transformer_blocks.0.attn.to_out.lora_A.weight",
        "transformer.single_transformer_blocks.0.attn.to_out.lora_B.weight",
        "transformer.transformer_blocks.0.attn.to_k.lora.down.weight",
        "transformer.transformer_blocks.0.attn.to_k.lora.up.weight"])
    assert P._lora_needs_normalizing(p) is True
    sd, n = P._load_lora_normalized(p)
    assert n == 2, n                       # only the 2 down/up keys are renamed
    assert set(sd) == {
        "transformer.single_transformer_blocks.0.attn.to_out.lora_A.weight",
        "transformer.single_transformer_blocks.0.attn.to_out.lora_B.weight",
        "transformer.transformer_blocks.0.attn.to_k.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_k.lora_B.weight"}, sorted(sd)
    assert len(sd) == 4, "no key must be lost or overwritten"
    print("OK test_normalizes_a_mixed_file")


def test_edit_loras_sync_the_union_with_the_base_set():
    """A regression: a base LoRA + an edit preset -> 'Adapter name cz_lora_0
    already in use', editing impossible. _apply_edit_loras must apply the UNION."""
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
# 3. LyCORIS (LoKr / LoHa). The update is factorised there into a Kronecker product
#    (LoKr) or a Hadamard one (LoHa), and diffusers converts neither.
#    The trap was that there was NO error at all: ai-toolkit's keys are called
#    'diffusion_model.<module>.lokr_w1' -- neither '.lora_A/B' nor the 'lora_unet_'
#    prefix -- so the checkpoint guard took them for a model, and the LoRA guard
#    passed them as they were to peft, which applied nothing in silence.
#    The LoKr is now supported by MERGING into the weights (see test_lokr_merge);
#    these tests here only check the ROUTING. The LoHa, for its part, stays refused.
#    Caught on Ashen3/SNOFS (Klein9b, 112 layers x w1/w2/alpha).
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
    """Since 1.27.0 the LoKr is SUPPORTED, by merging into the weights: so it is
    only refused where it does not belong -- the checkpoints folder -- and the refusal says
    where to put it. The merging itself is covered by tests/test_lokr_merge.py."""
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
    """The control: the guard must touch nothing of what worked."""
    p = _write("real_peft.safetensors",
               [f"transformer.blocks.{i}.attn.to_q.lora_{ab}.weight"
                for i in range(6) for ab in ("A", "B")])
    assert P._lora_unsupported(p) is None
    assert P._safetensors_unsupported(p) is not None   # a LoRA stays refused as a checkpoint
    print("OK test_a_real_peft_lora_still_passes")


# ---------------------------------------------------------------------------
# 4. A QUANTISED LoRA. This app's dequant loader only serves the transformer:
#    _safetensors_dequant is only called from _load_transformer. So an fp8 or
#    int8 LoRA went as it was into load_lora_weights, where its
#    'weight_scale' tensors are not known LoRA keys -- so ignored -- and where its
#    weights were cast to bf16 WITHOUT their scale: values several orders of
#    magnitude too small, that is to say a LoRA that does nothing, in silence. The same
#    trap as the FP4 and the LyCORIS, through the same door.
# ---------------------------------------------------------------------------

def _lora_file(name, dtype, scaled=True):
    """A fake FLUX.2 LoRA in the dtype asked for; `scaled` adds the factors."""
    sd = {}
    for i in range(4):
        b = f"transformer.transformer_blocks.{i}.attn.to_q"
        sd[b + ".lora_A.weight"] = torch.zeros(8, 4096, dtype=dtype)
        sd[b + ".lora_B.weight"] = torch.zeros(4096, 8, dtype=dtype)
        if scaled:
            sd[b + ".lora_B.weight_scale"] = torch.ones(4096, 1)
    p = os.path.join(TMP, name)
    save_file(sd, p)
    return p


def _with_9B_base(fn):
    old = P.BASE_REPO
    P.BASE_REPO = "test-only/FLUX.2-klein-9B"
    P._BASE_DIM_CACHE[P.BASE_REPO] = 4096
    try:
        return fn()
    finally:
        P.BASE_REPO = old


def test_a_bf16_lora_still_passes():
    """The control first: the guard must break nothing of what worked."""
    p = _lora_file("q_bf16.safetensors", torch.bfloat16, scaled=False)
    assert _with_9B_base(lambda: P._lora_unsupported(p)) is None
    print("OK test_a_bf16_lora_still_passes")


def test_an_fp8_lora_is_refused_and_points_at_bf16():
    p = _lora_file("q_fp8.safetensors", torch.float8_e4m3fn)
    why = _with_9B_base(lambda: P._lora_unsupported(p))
    assert why and "FP8" in why, why
    assert "bf16" in why, why
    print("OK test_an_fp8_lora_is_refused_and_points_at_bf16")


def test_an_int8_lora_is_refused_too():
    p = _lora_file("q_int8.safetensors", torch.int8)
    why = _with_9B_base(lambda: P._lora_unsupported(p))
    assert why and "INT8" in why, why
    print("OK test_an_int8_lora_is_refused_too")


def test_the_variant_check_still_comes_first():
    """A LoRA that is both 4B AND fp8 must be told about the VARIANT: it is the refusal
    that carries the useful instruction (change base), and the format would change nothing
    there."""
    sd = {}
    for i in range(4):
        b = f"transformer.transformer_blocks.{i}.attn.to_q"
        sd[b + ".lora_A.weight"] = torch.zeros(8, 3072, dtype=torch.float8_e4m3fn)
        sd[b + ".lora_B.weight"] = torch.zeros(3072, 8, dtype=torch.float8_e4m3fn)
        sd[b + ".lora_B.weight_scale"] = torch.ones(3072, 1)
    p = os.path.join(TMP, "q_fp8_4B.safetensors")
    save_file(sd, p)
    why = _with_9B_base(lambda: P._lora_unsupported(p))
    assert why and "4B" in why and "switch the base model" in why, why
    print("OK test_the_variant_check_still_comes_first")


def test_alpha_keys_are_folded_and_dropped():
    """RealSkin (4B/9B), seen on 2026-09-18: diffusers names, lora_down/lora_up matrices and
    one `.alpha` per module. As it is, diffusers refuses the whole file ("all LoRA param names
    contain 'lora'") and nothing is rendered. The alpha is an alpha / rank scale, to be carried
    in B."""
    base = "transformer.single_transformer_blocks.0.attn.to_out"
    A, B = torch.randn(4, 8), torch.randn(6, 4)
    for alpha, scale in ((4.0, 1.0), (2.0, 0.5)):
        p = os.path.join(TMP, f"alpha_{int(alpha)}.safetensors")
        save_file({base + ".lora_down.weight": A, base + ".lora_up.weight": B,
                   base + ".alpha": torch.tensor(alpha)}, p)
        assert P._lora_needs_normalizing(p) is True
        sd, n = P._load_lora_normalized(p)
        assert n == 2 and set(sd) == {base + ".lora_A.weight", base + ".lora_B.weight"}, list(sd)
        assert all("lora" in k for k in sd)
        assert torch.allclose(sd[base + ".lora_A.weight"], A)
        assert torch.allclose(sd[base + ".lora_B.weight"], B * scale), alpha
    # the PEFT dialect + alpha: diffusers refuses it just as much
    p = os.path.join(TMP, "peft_alpha.safetensors")
    save_file({base + ".lora_A.weight": A, base + ".lora_B.weight": B,
               base + ".alpha": torch.tensor(8.0)}, p)
    assert P._lora_needs_normalizing(p) is True
    sd, n = P._load_lora_normalized(p)
    assert n == 0 and set(sd) == {base + ".lora_A.weight", base + ".lora_B.weight"}
    assert torch.allclose(sd[base + ".lora_B.weight"], B * 2.0)
    print("OK test_alpha_keys_are_folded_and_dropped")


def test_kohya_naming_is_left_to_diffusers():
    """The kohya naming (lora_unet_...): diffusers recognises it by its `.lora_down.weight`
    and converts it, the alpha included. Renaming it beforehand hid the format from it."""
    p = _write("kohya.safetensors", [
        "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight",
        "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight",
        "lora_unet_double_blocks_0_img_attn_proj.alpha"])
    assert P._lora_needs_normalizing(p) is False
    print("OK test_kohya_naming_is_left_to_diffusers")


if __name__ == "__main__":
    test_alpha_keys_are_folded_and_dropped()
    test_kohya_naming_is_left_to_diffusers()
    for fn in (test_detects_the_alternate_dialect, test_normalizes_a_mixed_file,
               test_edit_loras_sync_the_union_with_the_base_set,
               test_union_dedupes_on_path_first_weight_wins,
               test_a_lokr_is_routed_to_the_lora_folder,
               test_a_loha_is_named_as_such,
               test_a_real_peft_lora_still_passes,
               test_a_bf16_lora_still_passes,
               test_an_fp8_lora_is_refused_and_points_at_bf16,
               test_an_int8_lora_is_refused_too,
               test_the_variant_check_still_comes_first):
        fn()
    print("All LoRA dialect / namespace tests passed.")
