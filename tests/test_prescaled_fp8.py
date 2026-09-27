"""FP8/INT8 weights stored ALREADY scaled: the weight_scale supplied does not apply.

Caught in the library on 2026-09-10: kleinFinalcutFP16FP8_comfyQuant rendered coloured
noise for every prompt. The file stores the weights as they are in FP8 and supplies
weight_scale = amax / 448 anyway. The loader multiplied: weights 1,200 to 1,700 times
too small. The criterion adopted, measured on the library's 17 quantised files:
max|stored| / (scale x range) is 1.03 for it, 71 to 1,691 for the others.

Run:  .venv/Scripts/python tests/test_prescaled_fp8.py

"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

import cz_pipeline as P

torch.manual_seed(0)
E4 = torch.float8_e4m3fn


def _pair(n=64):
    w = torch.randn(n, n) * 0.02
    return w, (w.abs().max() / 448.0).reshape(())


def test_the_detector_separates_the_two_layouts():
    w, s = _pair()
    regular = (w / s).to(E4)                 # weights / scale: the normal 'scaled' FP8
    prescaled = w.to(E4)                     # the weights as they are, the scale supplied on top
    assert not P._stored_at_scale(regular.float(), s, E4)
    assert P._stored_at_scale(prescaled.float(), s, E4)
    # an arbitrary scale on small values (synthetic test data):
    # a ratio far below 1, this is NOT the 'already scaled' case
    assert not P._stored_at_scale((w * 50).to(E4).float(), torch.tensor(0.5), E4)
    # INT8 normal
    s8 = (w.abs().max() / 127.0).reshape(())
    q8 = torch.round(w / s8).clamp(-127, 127).to(torch.int8)
    assert not P._stored_at_scale(q8.float(), s8, torch.int8)
    # a full INT8 (+-127) with a scale close to 1: a ratio of ~1 as well, but the
    # range is FILLED -- it is a normal file (the test_quant_formats case)
    full = torch.randint(-127, 128, (4, 3), dtype=torch.int8)
    full[0, 0] = 127
    assert not P._stored_at_scale(full.float(), torch.full((4, 1), 0.9), torch.int8)
    # MX scales (an E8M0 exponent in a uint8): never concerned
    assert not P._stored_at_scale(prescaled.float(), torch.tensor([120], dtype=torch.uint8), E4)
    assert not P._stored_at_scale(prescaled.float(), s, E4, {"format": "mxfp8"})
    print("OK test_the_detector_separates_the_two_layouts")


def _tiny(path, prescaled, both=False):
    """x_embedder already scaled when `prescaled`; context_embedder too when `both`
    (a real file is homogeneous: the cache key only samples one tensor)."""
    w1, s1 = _pair(32)
    w2, s2 = _pair(32)
    sd = {"x_embedder.weight": (w1 if prescaled else w1 / s1).to(E4),
          "x_embedder.weight_scale": s1.float(),
          "context_embedder.weight": (w2 if both else w2 / s2).to(E4),
          "context_embedder.weight_scale": s2.float()}
    save_file(sd, path)
    return w1, w2


def test_the_loader_reads_a_file_mixing_both_layouts():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "mixed.safetensors")
    w1, w2 = _tiny(p, prescaled=True)
    out = P._load_dequant_state_dict(p)
    for k, w in (("x_embedder.weight", w1), ("context_embedder.weight", w2)):
        rel = ((out[k].float() - w).norm() / w.norm()).item()
        assert rel < 0.1, (k, rel)      # without the fix: ~1.0 on x_embedder
    print("OK test_the_loader_reads_a_file_mixing_both_layouts")


def test_the_cache_key_changes_only_for_prescaled_files():
    d = tempfile.mkdtemp()
    cache = tempfile.mkdtemp()
    pre, reg = os.path.join(d, "pre.safetensors"), os.path.join(d, "reg.safetensors")
    _tiny(pre, prescaled=True, both=True)
    _tiny(reg, prescaled=False)
    old = P._DQ_CACHE_CFG
    try:
        P._DQ_CACHE_CFG = cache
        assert P._dequant_cache_path(pre) != P._dequant_cache_path(pre, legacy=True)
        assert P._dequant_cache_path(reg) == P._dequant_cache_path(reg, legacy=True), \
            "un fichier normal perdrait son cache pour rien"
        # the new cache written, the old (wrong) one of the same file disappears
        stale = P._dequant_cache_path(pre, legacy=True)
        with open(stale, "wb") as f:
            f.write(b"x")
        P._dequant_cache_store(pre, {"x": torch.zeros(1)})
        assert os.path.isfile(P._dequant_cache_path(pre))
        assert not os.path.exists(stale), "cache faux laisse sur le disque"
    finally:
        P._DQ_CACHE_CFG = old
    print("OK test_the_cache_key_changes_only_for_prescaled_files")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All prescaled-FP8 tests passed.")
