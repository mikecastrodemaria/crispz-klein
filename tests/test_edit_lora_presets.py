"""Edit-LoRA presets of crispz-klein (1.36.5).

Covers:
  - every preset names the model it targets (klein-4B or klein-9B) and an advised weight;
  - Consistence-Edit 9B takes a copy found in a LoRA folder (CivitAI download) as is,
    without fetching anything from Hugging Face;
  - the variant guard refuses a 4B LoRA on the 9B and the reverse, with one sentence;
  - picking a preset in the dropdown applies it at its advised weight and moves the
    slider there (it stayed at 1.0).

No GPU, no model: LoRA files are tiny fakes with the right shapes.

Run:  .venv/Scripts/python tests/test_edit_lora_presets.py
"""
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import cz_edit_loras  # noqa: E402


def _fake_lora(path, hidden, rank=4):
    """LoRA minuscule au layout FLUX d'origine, avec la dimension cachee d'une variante."""
    save_file({
        "diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight": torch.zeros(rank, hidden),
        "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight": torch.zeros(hidden, rank),
    }, path)
    return path


def test_every_preset_names_its_model_and_weight():
    for name in cz_edit_loras.names():
        s = cz_edit_loras.spec(name)
        assert s.get("base") in ("klein-4B", "klein-9B"), (name, s.get("base"))
        assert 0.0 < float(s.get("weight", 0)) <= 1.5, (name, s.get("weight"))
    assert cz_edit_loras.spec("Consistence-Edit")["base"] == "klein-4B"
    s9 = cz_edit_loras.spec("Consistence-Edit 9B")
    assert s9["base"] == "klein-9B" and s9["repo"] == "lrzjason/Consistance_Edit_Lora"
    assert s9["weights"] == "f2k_9B_lcs_consist_20260415.safetensors"
    assert cz_edit_loras.spec("consistence-edit-9b") is s9      # par adapter_name aussi


def test_consistence_9b_uses_a_local_copy_without_downloading():
    tmp = tempfile.mkdtemp()
    main, extra = os.path.join(tmp, "loras"), os.path.join(tmp, "civitai")
    os.makedirs(main)
    os.makedirs(extra)
    copy = os.path.join(extra, "f2k_9B_lcs_consist_20260415.safetensors")
    open(copy, "wb").close()
    real = (cz_edit_loras.lora_dirs, cz_edit_loras.edit_loras_dir, cz_edit_loras._download)

    def no_download(*a, **kw):
        raise AssertionError("a local copy exists: nothing must be downloaded")

    cz_edit_loras.lora_dirs = lambda: [main, extra]
    cz_edit_loras.edit_loras_dir = lambda: os.path.join(main, "_hf-edit")
    cz_edit_loras._download = no_download
    try:
        assert cz_edit_loras.is_downloaded("Consistence-Edit 9B")
        assert os.path.samefile(cz_edit_loras.resolve("Consistence-Edit 9B"), copy)
        assert cz_edit_loras.status_label("Consistence-Edit 9B").endswith("✓")
        assert not cz_edit_loras.is_downloaded("Consistence-Edit")    # le 4B n'y est pas
    finally:
        cz_edit_loras.lora_dirs, cz_edit_loras.edit_loras_dir, cz_edit_loras._download = real
        shutil.rmtree(tmp, ignore_errors=True)


def test_variant_guard_refuses_the_other_model():
    import cz_pipeline
    tmp = tempfile.mkdtemp()
    l4 = _fake_lora(os.path.join(tmp, "four.safetensors"), 3072)
    l9 = _fake_lora(os.path.join(tmp, "nine.safetensors"), 4096)
    key = cz_pipeline.BASE_REPO
    had, saved = key in cz_pipeline._BASE_DIM_CACHE, cz_pipeline._BASE_DIM_CACHE.get(key)
    try:
        cz_pipeline._BASE_DIM_CACHE[key] = 4096                # base 9B
        why = cz_pipeline._lora_unsupported(l4)
        assert why and "4B" in why, why
        assert cz_pipeline._lora_unsupported(l9) is None
        cz_pipeline._BASE_DIM_CACHE[key] = 3072                # base 4B
        why = cz_pipeline._lora_unsupported(l9)
        assert why and "9B" in why, why
        assert cz_pipeline._lora_unsupported(l4) is None
    finally:
        if had:
            cz_pipeline._BASE_DIM_CACHE[key] = saved
        else:
            cz_pipeline._BASE_DIM_CACHE.pop(key, None)
        shutil.rmtree(tmp, ignore_errors=True)


def test_picking_a_preset_sets_its_weight_on_the_slider():
    import cz_ui
    import cz_pipeline
    applied = []
    real = (cz_pipeline.set_edit_loras, cz_edit_loras.is_downloaded, cz_ui._edit_lora_choices)
    cz_pipeline.set_edit_loras = lambda slots: applied.append(list(slots))
    cz_edit_loras.is_downloaded = lambda name, index=None: True
    cz_ui._edit_lora_choices = lambda: ["None", "Consistence-Edit 9B ✓"]
    try:
        dd, note, slider = cz_ui._ui_edit_lora_pick("Consistence-Edit 9B ✓", 1.0,
                                                    progress=lambda *a, **k: None)
        assert applied == [[("Consistence-Edit 9B", 0.6)]], applied
        assert slider.get("value") == 0.6, slider
        assert "0.6" in note, note
        # "None" : rien a poser, le curseur ne bouge pas.
        dd, note, slider = cz_ui._ui_edit_lora_pick("None", 0.8, progress=lambda *a, **k: None)
        assert applied[-1] == [] and "value" not in slider, (applied, slider)
    finally:
        cz_pipeline.set_edit_loras, cz_edit_loras.is_downloaded, cz_ui._edit_lora_choices = real


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} edit-LoRA preset tests passed.")
