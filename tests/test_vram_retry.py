"""Out-of-VRAM recovery (crispz-klein 1.36.4).

A Reference (Omni) batch with Upscale after generate and the detailer ran out of VRAM
at the fourth image, then every later render failed until the app was restarted.
Covers:
  - is_oom recognizes both forms (torch allocator, direct CUDA call);
  - retry_on_oom frees the VRAM and retries ONCE, frees it again when the retry fails
    too, and leaves the other errors alone;
  - release_vram puts the base pipe's weights back on the CPU only in 'model' offload;
  - the detailer retries a pass, then skips the remaining regions if still out of VRAM;
  - the UI Omni loop retries, and reports what to lower when the retry fails.

No torch run, no GPU: the pipeline calls are stubbed.

Run:  .venv/Scripts/python tests/test_vram_retry.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image  # noqa: E402

OOM = "CUDA error: out of memory\nCUDA kernel errors might be asynchronously reported"


class _Releases:
    """Remplace cz_pipeline.release_vram le temps d'un test et compte les appels."""

    def __init__(self):
        import cz_pipeline
        self.mod, self.calls = cz_pipeline, []

    def __enter__(self):
        self.real = self.mod.release_vram
        self.mod.release_vram = lambda offload=False, why="": self.calls.append(offload)
        return self

    def __exit__(self, *exc):
        self.mod.release_vram = self.real


def test_is_oom_matches_both_forms():
    import cz_pipeline
    assert cz_pipeline.is_oom(RuntimeError(OOM))
    assert cz_pipeline.is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert cz_pipeline.is_oom(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"))
    assert not cz_pipeline.is_oom(RuntimeError("CUDA error: an illegal memory access"))
    assert not cz_pipeline.is_oom(ValueError("Edit needs at least one input image."))


def test_retry_on_oom_frees_the_vram_and_retries_once():
    import cz_pipeline
    tries = []

    def flaky(a, b=0):
        tries.append((a, b))
        if len(tries) == 1:
            raise RuntimeError(OOM)
        return a + b

    with _Releases() as rel:
        assert cz_pipeline.retry_on_oom("test", flaky, 2, b=3) == 5
    assert tries == [(2, 3), (2, 3)], tries
    assert rel.calls == [True], rel.calls          # weights back on the CPU too


def test_retry_on_oom_frees_again_when_the_retry_fails():
    import cz_pipeline
    tries = []

    def always(*a):
        tries.append(a)
        raise RuntimeError(OOM)

    with _Releases() as rel:
        try:
            cz_pipeline.retry_on_oom("test", always, 1)
        except RuntimeError as e:
            assert cz_pipeline.is_oom(e), e
        else:
            raise AssertionError("the second failure must be raised")
    assert len(tries) == 2, tries                  # one retry, not a loop
    assert rel.calls == [True, True], rel.calls    # freed before the error goes up


def test_retry_on_oom_leaves_other_errors_alone():
    import cz_pipeline
    tries = []

    def broken():
        tries.append(1)
        raise ValueError("bad input")

    with _Releases() as rel:
        try:
            cz_pipeline.retry_on_oom("test", broken)
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError must go through")
    assert tries == [1] and rel.calls == [], (tries, rel.calls)


def test_release_vram_offloads_only_a_hooked_base_pipe():
    import cz_pipeline

    class FakePipe:
        def __init__(self, hooks):
            self._all_hooks, self.freed = hooks, 0

        def maybe_free_model_hooks(self):
            self.freed += 1

    real = cz_pipeline._BASE_PIPE
    try:
        hooked, resident = FakePipe(["hook"]), FakePipe([])
        cz_pipeline._BASE_PIPE = hooked
        cz_pipeline.release_vram()
        assert hooked.freed == 0                   # a plain release keeps the weights
        cz_pipeline.release_vram(offload=True)
        assert hooked.freed == 1
        cz_pipeline._BASE_PIPE = resident          # offload 'none': nothing to put back
        cz_pipeline.release_vram(offload=True)
        assert resident.freed == 0
        cz_pipeline._BASE_PIPE = None
        cz_pipeline.release_vram(offload=True)     # no pipe loaded: no error
    finally:
        cz_pipeline._BASE_PIPE = real


def _detail(refine):
    """cz_detailer._detail_regions sur deux zones, avec une passe de refine simulee."""
    import cz_pipeline
    import cz_detailer
    real = (cz_pipeline.get_pipe, cz_pipeline._refine_whole)
    cz_pipeline.get_pipe = lambda kind="img2img": object()
    cz_pipeline._refine_whole = refine
    try:
        with _Releases() as rel:
            img, done = cz_detailer._detail_regions(
                Image.new("RGB", (512, 512)), [(40, 40, 140, 140), (300, 300, 400, 400)],
                "a face", 7, 4, 0.3, "face", min_size=10)
    finally:
        cz_pipeline.get_pipe, cz_pipeline._refine_whole = real
    return img, done, rel.calls


def test_detailer_retries_a_pass_that_ran_out_of_vram():
    tries = []

    def refine(pipe, work, denoise, steps, prompt, seed):
        tries.append(seed)
        if len(tries) == 1:
            raise RuntimeError(OOM)
        return work

    img, done, releases = _detail(refine)
    assert done == 2 and len(tries) == 3, (done, tries)   # zone 1 twice, zone 2 once
    assert releases == [True], releases
    assert img.size == (512, 512)


def test_detailer_skips_the_other_regions_when_still_out_of_vram():
    tries = []

    def refine(pipe, work, denoise, steps, prompt, seed):
        tries.append(seed)
        raise RuntimeError(OOM)

    img, done, releases = _detail(refine)
    # Zone 1: first try + retry. Zone 2 is not attempted: it would fail the same way.
    assert done == 0 and len(tries) == 2, (done, tries)
    assert releases == [True, True], releases
    assert img.size == (512, 512)


def _omni(fake_generate_omni):
    import cz_ui
    from test_variants_wiring import _ui_call, _wildcards
    real = cz_ui.generate_omni
    cz_ui.generate_omni = fake_generate_omni
    try:
        with _Releases() as rel, _wildcards({}):
            gal, rep, _h, _h2 = _ui_call("a car", "", 1, 7, use_input=True,
                                         input_mode="Reference (Omni)",
                                         ref1=Image.new("RGB", (32, 32)))
    finally:
        cz_ui.generate_omni = real
    return gal, rep, rel.calls


def test_ui_omni_retries_after_running_out_of_vram():
    import cz_pipeline
    if not (cz_pipeline.OMNI_MODEL or "").strip():
        print("SKIP test_ui_omni_retries_after_running_out_of_vram (no omni model)")
        return
    tries = []

    def fake_generate_omni(refs, prompt, negative, width, height, steps, seed, **kw):
        tries.append(seed)
        if len(tries) == 1:
            raise RuntimeError(OOM)
        return Image.new("RGB", (32, 32))

    gal, rep, releases = _omni(fake_generate_omni)
    assert tries == [7, 7] and len(gal) == 1, (tries, gal)
    assert "omni x1" in rep and "VRAM" not in rep, rep
    # The retry's release (weights back on the CPU), then the one between two images.
    assert releases == [True, False], releases


def test_ui_omni_reports_what_to_lower_when_the_retry_fails():
    import cz_pipeline
    if not (cz_pipeline.OMNI_MODEL or "").strip():
        print("SKIP test_ui_omni_reports_what_to_lower_when_the_retry_fails (no omni model)")
        return

    def fake_generate_omni(refs, prompt, negative, width, height, steps, seed, **kw):
        raise RuntimeError(OOM)

    gal, rep, releases = _omni(fake_generate_omni)
    assert not gal, gal
    assert "Omni error" in rep and "VRAM saturee" in rep and "redemarre" in rep, rep
    assert releases == [True, True], releases


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} VRAM retry tests passed.")
