"""The latent -> RGB projection of the live preview (FLUX.2, packed latents).

FLUX.2 hands the callback its latents PACKED: (batch, h*w, 128), one token per 16x16
block of pixels. Getting the grid wrong does not raise -- it shows a scrambled image, or
a correct-looking one at the wrong aspect ratio -- so the shape rules are pinned here.

Neither GPU nor model: the latents are zeros, only the geometry is checked.

Run:  .venv/Scripts/python tests/test_latent_preview.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import cz_pipeline as P  # noqa: E402


def _lat(w, h, ch=128):
    return torch.zeros(1, (h // 16) * (w // 16), ch)


def test_the_grid_comes_from_the_asked_resolution():
    img = P.latent_preview_image(_lat(1024, 768), 1024, 768)
    assert img is not None
    assert img.width > img.height, img.size              # landscape stays landscape
    assert abs(img.width / img.height - 1024 / 768) < 0.01, img.size


def test_a_portrait_render_stays_portrait():
    img = P.latent_preview_image(_lat(768, 1152), 768, 1152)
    assert abs(img.width / img.height - 768 / 1152) < 0.01, img.size


def test_the_preview_is_scaled_up_to_the_configured_side():
    img = P.latent_preview_image(_lat(768, 768), 768, 768)
    assert max(img.size) == P.LIVE_PREVIEW_MAX_SIDE, img.size


def test_an_unknown_resolution_falls_back_to_a_square():
    """The grid cannot be guessed from the token count alone; a square is the only safe
    reading, and it is better than no preview at all."""
    img = P.latent_preview_image(_lat(768, 768), 0, 0)
    assert img is not None and img.width == img.height, img


def test_a_token_count_that_fits_nothing_gives_no_preview():
    """Better nothing than a scrambled image: 3000 tokens is neither 1024x768's grid nor
    a square."""
    assert P.latent_preview_image(torch.zeros(1, 3000, 128), 1024, 768) is None


def test_unpacked_latents_are_refused():
    """A 4D tensor is another pipeline's layout, not FLUX.2's: no guessing."""
    assert P.latent_preview_image(torch.zeros(1, 32, 96, 96), 768, 768) is None


def test_the_matrix_matches_the_packed_channel_count():
    assert len(P._LATENT_RGB) == 128, len(P._LATENT_RGB)
    assert all(len(row) == 3 for row in P._LATENT_RGB)
    assert len(P._LATENT_RGB_BIAS) == 3


def test_a_preview_failure_never_reaches_the_render():
    """_store_preview swallows everything: it is a courtesy, not a result."""
    P.preview_begin()
    try:
        P._store_preview(object(), 1, 8, 768, 768)       # not a tensor at all
        assert P.preview_snapshot()["img"] is None
        P._store_preview(_lat(768, 768), 2, 8, 768, 768)
        snap = P.preview_snapshot()
        assert snap["img"] is not None and snap["step"] == 2 and snap["seq"] == 1, snap
    finally:
        P.preview_end()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} latent preview tests passed.")
