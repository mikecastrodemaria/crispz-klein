"""Unit tests for the quantized-checkpoint support: header routing
(_safetensors_unsupported / _safetensors_dequant), the ComfyUI FP8/INT8 dequant
loader (_load_dequant_state_dict) and the GGUF guards (_gguf_arch /
_gguf_layout_unsupported). Synthetic files only (a few KB), no model download.

Run:  .venv/Scripts/python tests/test_quant_formats.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import cz_pipeline  # noqa: E402

TMP = tempfile.mkdtemp(prefix="cz_quant_test_")


def _st(name, tensors):
    p = os.path.join(TMP, name)
    save_file(tensors, p)
    return p


# Cles minimales "FLUX.2" (marqueur single_transformer_blocks) au format ComfyUI.
# Taken from the real FLUX.2-klein-4B transformer (see _QWEN_KEY_MARKERS).
_W = "model.diffusion_model.single_transformer_blocks.0.attn.to_q.weight"
# The SAME key in the diffusers layout: the loader has to strip the ComfyUI prefix, or
# Flux2Transformer2DModel (mapped through an identity function in diffusers) recognises
# no key at all and the model stays on 'meta' ("Cannot copy out of meta tensor").
_WD = _W[len("model.diffusion_model."):]


def test_bf16_passthrough():
    p = _st("bf16.safetensors", {_W: torch.randn(4, 4, dtype=torch.bfloat16)})
    assert cz_pipeline._safetensors_unsupported(p) is None
    assert cz_pipeline._safetensors_dequant(p) is None


def test_fp8_pure_detect_and_dequant():
    w = torch.randn(4, 4).to(torch.float8_e4m3fn)
    p = _st("fp8.safetensors", {_W: w})
    assert cz_pipeline._safetensors_dequant(p) == "FP8"
    sd = cz_pipeline._load_dequant_state_dict(p)
    assert _W not in sd, "the ComfyUI prefix must be stripped"
    assert sd[_WD].dtype == cz_pipeline.DTYPE
    assert torch.allclose(sd[_WD].float(), w.float().to(cz_pipeline.DTYPE).float())


def test_fp8_scaled_dequant_math():
    w = torch.randn(4, 4).to(torch.float8_e4m3fn)
    scale = torch.tensor(2.5, dtype=torch.float32)
    p = _st("fp8s.safetensors", {
        _W: w, _W + "_scale": scale,
        _W.replace(".weight", ".comfy_quant"): torch.zeros(27, dtype=torch.uint8),
    })
    assert cz_pipeline._safetensors_dequant(p) == "FP8 scaled"
    sd = cz_pipeline._load_dequant_state_dict(p)
    want = (w.to(torch.float32) * 2.5).to(cz_pipeline.DTYPE)
    assert torch.allclose(sd[_WD].float(), want.float())
    assert _WD + "_scale" not in sd
    assert _WD.replace(".weight", ".comfy_quant") not in sd


def test_int8_per_row_scale():
    w = torch.randint(-127, 127, (4, 3), dtype=torch.int8)
    scale = torch.rand(4, 1, dtype=torch.float32)
    p = _st("int8.safetensors", {_W: w, _W + "_scale": scale})
    assert cz_pipeline._safetensors_dequant(p) == "INT8 scaled"
    sd = cz_pipeline._load_dequant_state_dict(p)
    want = (w.to(torch.float32) * scale).to(cz_pipeline.DTYPE)
    assert torch.allclose(sd[_WD].float(), want.float())


def test_int8_convrot_roundtrip():
    # comfy-quants' int8_tensorwise + ConvRot format: the weights are ROTATED
    # (Hadamard, H4 base, in groups of 256) before quantisation -> the loader has to undo
    # the rotation, or it is pure noise (seen for real on redzit/studio).
    torch.manual_seed(0)
    W = torch.randn(8, 512)                      # in=512 -> 2 groups of 256
    H = cz_pipeline._hadamard_ortho(256)
    wr = (W.view(8, 2, 256) @ H.T).reshape(8, 512)
    scale = (wr.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=1e-30)
    q = (wr / scale).round().clamp(-128, 127).to(torch.int8)
    blob = torch.tensor(
        list(b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'),
        dtype=torch.uint8)
    p = _st("convrot.safetensors", {
        _W: q, _W + "_scale": scale.float(),
        _W.replace(".weight", ".comfy_quant"): blob,
    })
    sd = cz_pipeline._load_dequant_state_dict(p)
    err = float((sd[_WD].float() - W).abs().max() / W.abs().max())
    assert err < 0.02, f"convrot roundtrip error too high: {err}"


def test_aio_bundle_filtered():
    # bundle: transformer BF16 + "encodeur texte" FP8 -> seul le transformer reste
    p = _st("aio.safetensors", {
        _W: torch.randn(4, 4, dtype=torch.bfloat16),
        "text_encoders.qwen2vl.layers.0.weight": torch.randn(4, 4).to(torch.float8_e4m3fn),
        "vae.decoder.weight": torch.randn(2, 2, dtype=torch.bfloat16),
    })
    assert cz_pipeline._safetensors_dequant(p) == "FP8"
    sd = cz_pipeline._load_dequant_state_dict(p)
    assert list(sd) == [_WD]


def test_comfy_prefix_is_stripped():
    """Regression: the single-file Qwen checkpoints (Comfy-Org, Civitai) name EVERYTHING
    'model.diffusion_model.*'. diffusers 0.39 maps Flux2Transformer2DModel through an
    IDENTITY function -- so it does not strip that prefix itself. Keeping it = 100% unknown
    keys, not one weight loaded, the model left on 'meta', and dispatch_model failing on
    "Cannot copy out of meta tensor; no data!" (seen on
    qwen_image_edit_2509_fp8_e4m3fn.safetensors)."""
    w = torch.randn(4, 4).to(torch.float8_e4m3fn)
    p = _st("prefixed.safetensors", {
        _W: w,
        "model.diffusion_model.x_embedder.weight": torch.randn(4, 4, dtype=torch.bfloat16),
    })
    sd = cz_pipeline._load_dequant_state_dict(p)
    assert sorted(sd) == [_WD, "x_embedder.weight"], sorted(sd)
    assert not any(k.startswith("model.diffusion_model.") for k in sd)


def test_unprefixed_keys_left_alone():
    """A file already in the diffusers layout (no prefix) must be left alone."""
    w = torch.randn(4, 4).to(torch.float8_e4m3fn)
    p = _st("plain.safetensors", {_WD: w})
    sd = cz_pipeline._load_dequant_state_dict(p)
    assert list(sd) == [_WD]


def test_bf16_prefixed_is_detected_and_stripped():
    """The NON-quantised path has exactly the same flaw: a bf16 .safetensors in the
    ComfyUI layout, handed to from_single_file as it is, leaves the model on 'meta'.
    So it has to be detected from the header and unprefixed like the FP8/INT8 ones."""
    p = _st("bf16_prefixed.safetensors", {_W: torch.randn(4, 4, dtype=torch.bfloat16)})
    assert cz_pipeline._safetensors_dequant(p) is None      # nothing to dequantise
    assert cz_pipeline._safetensors_comfy_prefixed(p) is True
    sd = cz_pipeline._load_dequant_state_dict(p)
    assert list(sd) == [_WD]

    plain = _st("bf16_plain.safetensors", {_WD: torch.randn(4, 4, dtype=torch.bfloat16)})
    assert cz_pipeline._safetensors_comfy_prefixed(plain) is False


def test_prefixed_single_file_is_loaded_as_a_state_dict():
    """Routing: a prefixed NON-quantised single file has to reach from_single_file as an
    unprefixed STATE DICT -- handing it the path would send it straight back into the
    meta-tensor bug."""
    import diffusers
    p = _st("route.safetensors", {
        _W: torch.randn(4, 4, dtype=torch.bfloat16),
        "model.diffusion_model.x_embedder.weight": torch.randn(4, 4, dtype=torch.bfloat16),
    })
    seen = {}

    class _Fake:
        @staticmethod
        def from_single_file(src, **kw):
            seen["src"] = src
            return "MODEL"

    old_cls = diffusers.Flux2Transformer2DModel
    old_t = cz_pipeline.ZIMAGE_TRANSFORMER
    diffusers.Flux2Transformer2DModel = _Fake
    cz_pipeline.ZIMAGE_TRANSFORMER = p
    try:
        assert cz_pipeline._load_transformer() == "MODEL"
    finally:
        diffusers.Flux2Transformer2DModel = old_cls
        cz_pipeline.ZIMAGE_TRANSFORMER = old_t
    assert isinstance(seen["src"], dict), "a path was passed instead of the state dict"
    assert sorted(seen["src"]) == [_WD, "x_embedder.weight"], sorted(seen["src"])


def test_convert_device_cpu_is_respected():
    """convert_device=cpu = never touch the GPU (a conversion during a render)."""
    w = torch.randn(4, 4).to(torch.float8_e4m3fn)
    p = _st("cpu_dev.safetensors", {_W: w})
    old = cz_pipeline.CONFIG.get("convert_device")
    cz_pipeline.CONFIG["convert_device"] = "cpu"
    try:
        sd = cz_pipeline._load_dequant_state_dict(p)
    finally:
        if old is None:
            cz_pipeline.CONFIG.pop("convert_device", None)
        else:
            cz_pipeline.CONFIG["convert_device"] = old
    assert sd[_WD].device.type == "cpu"
    assert torch.allclose(sd[_WD].float(), w.float().to(cz_pipeline.DTYPE).float())


def test_dequant_result_always_lands_on_cpu():
    """Even converted on the GPU, the state dict handed back must be in RAM (it goes on
    into from_single_file, and the pipeline handles the placement/offload itself)."""
    p = _st("oncpu.safetensors", {_W: torch.randn(4, 4).to(torch.float8_e4m3fn)})
    sd = cz_pipeline._load_dequant_state_dict(p)
    assert all(t.device.type == "cpu" for t in sd.values())


def test_foreign_arch_rejected():
    w = torch.randn(4, 4).to(torch.float8_e4m3fn)
    p = _st("foreign.safetensors",
            {"model.diffusion_model.layers.0.mlp.gate_proj.weight": w})
    raised = False
    try:
        cz_pipeline._load_dequant_state_dict(p)
    except RuntimeError as e:
        raised = "FLUX.2" in str(e)
    assert raised, "a quantised checkpoint of another architecture must be refused plainly"


def test_lora_and_svdq_still_unsupported():
    lora = {f"lora_unet_a{i}.lora_down.weight": torch.zeros(2, 2) for i in range(4)}
    p = _st("lora.safetensors", lora)
    assert "LoRA" in (cz_pipeline._safetensors_unsupported(p) or "")
    p = _st("svdq.safetensors", {"blocks.0.qweight": torch.zeros(2, 2, dtype=torch.int8)})
    assert "SVDQuant" in (cz_pipeline._safetensors_unsupported(p) or "")


def _gguf(name, arch, *tensor_names):
    import numpy as np
    from gguf import GGUFWriter
    p = os.path.join(TMP, name)
    w = GGUFWriter(p, arch)
    for t in tensor_names:
        w.add_tensor(t, np.zeros((4, 32), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return p


# The complete signature of a FLUX.2 transformer (the 3 positive keys + one block).
# The constant name is kept to keep the conflict surface small on an upstream merge.
_QWEN_SIG = ("x_embedder.weight", "context_embedder.weight",
             "double_stream_modulation_img.linear.weight",
             "single_transformer_blocks.0.attn.to_q.weight")


def test_gguf_arch_and_layout():
    ok = _gguf("klein.gguf", "flux2", *_QWEN_SIG)
    assert cz_pipeline._gguf_arch(ok) == "flux2"
    assert cz_pipeline._gguf_layout(ok) == "flux2"
    assert cz_pipeline._gguf_layout_unsupported(ok) is None
    # FLUX.1 au layout city96: noms de tenseurs completement differents -> foreign
    f1 = _gguf("flux1.gguf", "flux", "double_blocks.0.img_attn.qkv.weight")
    assert cz_pipeline._gguf_arch(f1) == "flux"
    assert cz_pipeline._gguf_layout(f1) == "foreign"
    sdcpp = _gguf("sdcpp.gguf", "flux2", "blocks.0.attn.wq.weight")
    assert cz_pipeline._gguf_layout_unsupported(sdcpp) is not None


def test_gguf_layout_beats_a_mislabelled_architecture():
    """Conversion tools stamp 'general.architecture' any old way: Qwen-Image files have
    been published declared as 'wan'. The TENSOR NAMES are proof, the KV label is not ->
    the layout wins, otherwise a perfectly loadable model gets thrown out."""
    p = _gguf("mislabelled.gguf", "wan", *_QWEN_SIG)
    assert cz_pipeline._gguf_arch(p) == "wan"
    assert cz_pipeline._gguf_layout(p) == "flux2"
    assert cz_pipeline._gguf_layout_unsupported(p) is None


def test_gguf_foreign_layout_still_rejected():
    """The layout stays the real guard rail: an sd.cpp schema is refused even when it
    declares the right architecture."""
    p = _gguf("sdcpp2.gguf", "flux2", "blocks.0.attn.wq.weight", "txtmlp.weight")
    assert cz_pipeline._gguf_layout(p) == "foreign"
    assert cz_pipeline._gguf_layout_unsupported(p) is not None


def test_gguf_qwen_diffusers_layout_is_not_mistaken_for_flux2():
    """Qwen-Image in the diffusers layout shares 'transformer_blocks.', 'norm_out' and
    'proj_out' with FLUX.2: those prefixes are therefore NOT enough to conclude. Without
    the FLUX.2 signature (x_embedder + context_embedder + double_stream_modulation) ->
    the layout is undetermined, the declared architecture decides, and the file is
    dropped."""
    p = _gguf("qwenlike.gguf", "qwen_image", "transformer_blocks.0.attn.to_q.weight",
              "img_in.weight", "txt_in.weight", "norm_out.linear.weight")
    assert cz_pipeline._gguf_layout(p) == "unknown"
    assert cz_pipeline._gguf_layout_unsupported(p) is None      # not 'foreign'
    assert cz_pipeline._gguf_arch(p) == "qwen_image"            # -> dropped on its architecture


def test_list_checkpoints_accepts_the_mislabelled_gguf():
    """End to end: the dropdown must offer the mislabelled FLUX.2 GGUF and refuse the
    Qwen-Image one, in a checkpoints folder of its own."""
    import tempfile
    d = tempfile.mkdtemp(prefix="cz_ckpt_")
    for src, dst in ((_gguf("ck_ok.gguf", "wan", *_QWEN_SIG), "hyphoria_like.gguf"),
                     (_gguf("ck_qwen.gguf", "qwen_image",
                            "transformer_blocks.0.attn.to_q.weight",
                            "img_in.weight"), "qwen_like.gguf"),
                     (_gguf("ck_sd.gguf", "flux2", "blocks.0.attn.wq.weight"),
                      "sdcpp_like.gguf")):
        import shutil
        shutil.copy(src, os.path.join(d, dst))
    old_dir, old_extra = cz_pipeline.CHECKPOINTS_DIR, cz_pipeline.CHECKPOINTS_EXTRA_DIR
    cz_pipeline.CHECKPOINTS_DIR, cz_pipeline.CHECKPOINTS_EXTRA_DIR = d, ""
    try:
        got = cz_pipeline.list_checkpoints()
    finally:
        cz_pipeline.CHECKPOINTS_DIR = old_dir
        cz_pipeline.CHECKPOINTS_EXTRA_DIR = old_extra
    assert got == ["hyphoria_like.gguf"], got


def test_int8_convrot_declared_in_header_metadata():
    """The StableYogi variant: convrot declared CENTRALLY in
    __metadata__._quantization_metadata, with no per-tensor blobs. Ignoring it leaves the
    rotation in place -> pure noise (seen on the Krea 2 INT8 files)."""
    import json as _json
    torch.manual_seed(1)
    W = torch.randn(8, 512)
    H = cz_pipeline._hadamard_ortho(256)
    wr = (W.view(8, 2, 256) @ H.T).reshape(8, 512)
    scale = (wr.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=1e-30)
    q = (wr / scale).round().clamp(-128, 127).to(torch.int8)
    layer = _W[:-len(".weight")]
    if layer.startswith("model.diffusion_model."):
        layer = layer[len("model.diffusion_model."):]
    meta = _json.dumps({"format_version": "1.0", "layers": {
        layer: {"format": "int8_tensorwise", "convrot": True,
                "convrot_groupsize": 256}}})
    p = os.path.join(TMP, "convrot_meta.safetensors")
    save_file({_W: q, _W + "_scale": scale.float()}, p,
              metadata={"_quantization_metadata": meta})
    sd = cz_pipeline._load_dequant_state_dict(p)
    k = [x for x in sd if x.endswith("attn.to_q.weight") or "img_attn.proj" in x][0]
    err = float((sd[k].float() - W).abs().max() / W.abs().max())
    assert err < 0.02, f"metadata-declared convrot not undone: err {err}"


# ---------------------------------------------------------------------------
# FP4 (NVFP4 / MXFP4): no path here. The trap is that an FP4 has NO F8 dtype, so
# _safetensors_dequant hands back None and the file goes down the normal bf16 path
# -- silently. snofs14's Civitai page offers that very model as
# "MXFP8 and NVFP4 quants": the refusal has to be named, not discovered in the image.
# ---------------------------------------------------------------------------

def _hdr_with_formats(fmt, dtype="F8_E4M3"):
    """A synthetic header: ComfyUI/ModelOpt quantisation metadata."""
    return {
        "__metadata__": {"_quantization_metadata": json.dumps({
            "format_version": "1.0",
            "layers": {"double_blocks.0.img_attn.qkv": {
                "format": fmt, "group_size": 32,
                "orig_dtype": "torch.bfloat16", "orig_shape": [12288, 4096]}},
        })},
        "double_blocks.0.img_attn.qkv.weight": {"dtype": dtype, "shape": [12288, 4096]},
        "double_blocks.0.img_attn.qkv.weight_scale": {"dtype": "U8", "shape": [12288, 128]},
        "x_embedder.weight": {"dtype": "BF16", "shape": [4, 4]},
    }


def _with_header(hdr, fn):
    """Runs fn() with the header _safetensors_unsupported reads substituted."""
    real = cz_pipeline._safetensors_header
    cz_pipeline._safetensors_header = lambda _p: hdr
    try:
        return fn()
    finally:
        cz_pipeline._safetensors_header = real


def test_nvfp4_is_refused_by_name():
    hdr = _hdr_with_formats("nvfp4", dtype="U8")   # 4 bits packed -> U8
    why = _with_header(hdr, lambda: cz_pipeline._safetensors_unsupported("fake.safetensors"))
    assert why and "NVFP4" in why, why
    assert "fp8 or bf16" in why, why
    # the exact trap: without this refusal, nothing stops it
    assert _with_header(hdr, lambda: cz_pipeline._safetensors_dequant("fake.safetensors")) is None


def test_a_f4_dtype_is_refused_even_without_metadata():
    hdr = {"double_blocks.0.img_attn.qkv.weight": {"dtype": "F4_E2M1", "shape": [12288, 4096]},
           "x_embedder.weight": {"dtype": "BF16", "shape": [4, 4]}}
    why = _with_header(hdr, lambda: cz_pipeline._safetensors_unsupported("fake.safetensors"))
    assert why and "FP4" in why, why


def test_mxfp8_is_not_caught_by_the_fp4_guard():
    """The control: snofs14 is MXFP8 and must pass, or we break what works."""
    hdr = _hdr_with_formats("mxfp8")
    assert _with_header(hdr, lambda: cz_pipeline._safetensors_unsupported("fake.safetensors")) is None
    assert _with_header(hdr, lambda: cz_pipeline._safetensors_dequant("fake.safetensors")) == "FP8 scaled"



if __name__ == "__main__":
    for fn in (test_bf16_passthrough, test_fp8_pure_detect_and_dequant,
               test_fp8_scaled_dequant_math, test_int8_per_row_scale,
               test_int8_convrot_roundtrip, test_aio_bundle_filtered,
               test_comfy_prefix_is_stripped, test_unprefixed_keys_left_alone,
               test_bf16_prefixed_is_detected_and_stripped,
               test_prefixed_single_file_is_loaded_as_a_state_dict,
               test_convert_device_cpu_is_respected,
               test_dequant_result_always_lands_on_cpu,
               test_foreign_arch_rejected, test_lora_and_svdq_still_unsupported,
               test_gguf_arch_and_layout,
               test_nvfp4_is_refused_by_name,
               test_a_f4_dtype_is_refused_even_without_metadata,
               test_mxfp8_is_not_caught_by_the_fp4_guard,
               test_gguf_layout_beats_a_mislabelled_architecture,
               test_gguf_foreign_layout_still_rejected,
               test_gguf_qwen_diffusers_layout_is_not_mistaken_for_flux2,
               test_list_checkpoints_accepts_the_mislabelled_gguf,
               test_int8_convrot_declared_in_header_metadata):
        fn()
        print(f"OK {fn.__name__}")
    print("All quant-format tests passed.")
