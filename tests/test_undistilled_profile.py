"""The official UNDISTILLED base gets the Undistilled preset, not 4 steps with no CFG.

Caught on the 2026-09-10 bench: flux-2-klein-base-4b-fp8, an official file with no
CivitAI sidecar, fell on the name profile ("klein" -> 4 steps, guidance 1.0) and
rendered an unfinished image, in the bench as in the app.

Run:  .venv/Scripts/python tests/test_undistilled_profile.py

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_civitai
import cz_ui as U


def _profile(path):
    old = cz_civitai.load_civitai_sidecar
    cz_civitai.load_civitai_sidecar = lambda p: {}        # an official file: no sidecar
    try:
        return U._profile_for_checkpoint(path)
    finally:
        cz_civitai.load_civitai_sidecar = old


def test_the_official_undistilled_base_gets_the_undistilled_preset():
    preset, pst, pg = U._undistilled_profile()
    assert preset, "no real-CFG preset in PERFORMANCE"
    for name in ("flux-2-klein-base-4b-fp8.safetensors", "FLUX.2-klein-base-9B.safetensors",
                 "flux2_klein_base_4b.safetensors"):
        st, g, why = _profile(os.path.join("F:\\m", name))
        assert (st, g) == (pst, pg), (name, st, g)
        assert "undistilled" in why, why           # the bench classifies the source by that word
    print("OK test_the_official_undistilled_base_gets_the_undistilled_preset")


def test_the_distilled_files_keep_their_profile():
    for name in ("flux-2-klein-4b-fp8.safetensors", "flux2Klein9bFp8_fp8.safetensors",
                 "baseballKlein_v1.safetensors"):
        st, g, why = _profile(os.path.join("F:\\m", name))
        assert g <= 1.0 and "klein-base" not in why, (name, st, g, why)
    print("OK test_the_distilled_files_keep_their_profile")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All undistilled-profile tests passed.")
