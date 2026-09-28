"""An "undistilled" checkpoint rendered a blurry mush, without a word.

klein-4B/9B are distilled: the guidance is INERT on them (measured bit-for-bit identical
from 1.0 to 8.0, see test_klein_guidance.py), so _qwen_call forces it to 1.0. But that
`is_distilled` flag describes the BASE REPO. With a single-file override it no longer
says anything about the model that computes -- and there exist community checkpoints
explicitly NOT distilled ("undistilled - use with Turbo Lora") that require a
real CFG. Forcing them to 1.0 gave a smoothed, unreadable image.

Run:  .venv/Scripts/python tests/test_undistilled_guidance.py

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P


class Pipe:
    config = type("c", (), {"is_distilled": True})()
    _execution_device = "cpu"
    text_encoder = None

    def __call__(self, **kw):
        self.kw = kw
        return type("o", (), {"images": ["I"]})()


def _guidance(cfg, transformer):
    old = (P.GUIDANCE, P.ZIMAGE_TRANSFORMER)
    P.GUIDANCE, P.ZIMAGE_TRANSFORMER = cfg, transformer
    try:
        p = Pipe()
        P._qwen_call(p, prompt="x")
        return p.kw["guidance_scale"]
    finally:
        P.GUIDANCE, P.ZIMAGE_TRANSFORMER = old


CKPT = os.path.join("F:", "x", "kleinUndistilled_v1.safetensors")


def test_the_base_repo_stays_at_one():
    """On the base repo, the guidance is inert: proven, so we do not pass it on."""
    assert _guidance(1.0, None) == 1.0
    assert _guidance(4.0, None) == 1.0, "le repo de base ignore la CFG (mesure)"
    print("OK test_the_base_repo_stays_at_one")


def test_an_override_gets_the_slider():
    """On a third-party checkpoint, we do NOT know whether it is distilled: the user decides."""
    assert _guidance(4.0, CKPT) == 4.0
    assert _guidance(1.0, CKPT) == 1.0, "slider at 1.0: nothing changes"
    print("OK test_an_override_gets_the_slider")


def test_an_explicit_guidance_is_never_overridden():
    p = Pipe()
    P._qwen_call(p, prompt="x", guidance_scale=7.5)
    assert p.kw["guidance_scale"] == 7.5
    print("OK test_an_explicit_guidance_is_never_overridden")



def test_the_ignored_guidance_is_announced_once():
    """A house rule: a value thrown away is said. But once, not on every image."""
    P._CFG_IGNORED_SAID.clear()
    said = []
    real, P._log = P._log, said.append
    try:
        _guidance(4.0, None)
        _guidance(4.0, None)
        _guidance(4.0, None)
    finally:
        P._log = real
    # Filtered, NOT counted raw: _log carries everything, and an unrelated line (the
    # offload restore message this machine's config produces) would fail a test about
    # guidance. The point is that the SAME ignored value is announced once, not that
    # nothing else is ever logged.
    said = [m for m in said if "guidance" in m]
    assert len(said) == 1, f"{len(said)} guidance lines for the same value"
    assert "4" in said[0] and "distilled" in said[0], said[0]
    P._CFG_IGNORED_SAID.clear()
    print("OK test_the_ignored_guidance_is_announced_once")


def test_a_preset_exists_for_undistilled_checkpoints():
    """With no preset, the only path was editing config.txt by hand."""
    import cz_ui
    hits = [(n, v) for n, v in cz_ui.PERFORMANCE.items() if float(v[1]) > 1.0]
    assert hits, f"no preset with a real CFG: {list(cz_ui.PERFORMANCE)}"
    name, (steps, cfg) = hits[0]
    assert steps >= 20 and cfg >= 2.0, (name, steps, cfg)
    # and the radio must be able to light up on it again from the sliders
    assert cz_ui._performance_label_for(steps, cfg) == name
    # the radio's fallback must never name a preset that does not exist
    assert cz_ui._valid_performance(None) in cz_ui.PERFORMANCE
    assert cz_ui._valid_performance("Turbo (8 steps)") in cz_ui.PERFORMANCE
    print("OK test_a_preset_exists_for_undistilled_checkpoints")



def test_an_undistilled_checkpoint_gets_the_right_profile():
    """Choosing the model set 4 steps / CFG 1.0 from the FILE NAME, which
    overwrote the Performance preset the user had just chosen -- in silence.
    The file name does not say that a build is undistilled; the CivitAI sidecar does."""
    import json
    import tempfile
    import cz_ui as U

    tmp = tempfile.mkdtemp(prefix="cz_undist_")
    ck = os.path.join(tmp, "kleinSomething_v1.safetensors")
    open(ck, "wb").write(bytes(16))
    side = os.path.join(tmp, "kleinSomething_v1.civitai.json")

    def profile(model_name):
        with open(side, "w", encoding="utf-8") as f:
            json.dump({"modelName": model_name}, f)
        return U._profile_for_checkpoint(ck)

    st, g, why = profile("Klein Something (undistilled - use with Turbo Lora)")
    assert g > 1.0 and st >= 20, (st, g)
    assert "undistilled" in why, why

    st, g, why = profile("Klein Something Turbo")
    assert (st, g) == (4, 1.0), (st, g)     # the file-name profile, as before
    assert why == ""
    print("OK test_an_undistilled_checkpoint_gets_the_right_profile")


def test_the_civitai_consensus_wins_over_the_filename():
    """The substring profile knows nothing about THIS model: it imposed 4 steps on
    everything called 'klein', whereas the consensus already downloaded asks for 10
    for one and 8 for the other. The data was on the disk, unused."""
    import json
    import tempfile
    import cz_ui as U

    tmp = tempfile.mkdtemp(prefix="cz_reco_")
    ck = os.path.join(tmp, "kleinSomething_v1.safetensors")
    open(ck, "wb").write(bytes(16))
    with open(os.path.join(tmp, "kleinSomething_v1.civitai.json"), "w",
              encoding="utf-8") as f:
        json.dump({"modelName": "Klein Something",
                   "recommended": {"n": 10, "steps": 10, "guidance": 1.0}}, f)
    st, g, why = U._profile_for_checkpoint(ck)
    assert (st, g) == (10, 1.0), (st, g)
    assert "consensus" in why and "10 community" in why, why

    # and it also beats the undistilled flag: more specific than the category
    with open(os.path.join(tmp, "kleinSomething_v1.civitai.json"), "w",
              encoding="utf-8") as f:
        json.dump({"modelName": "Klein Something (undistilled - use with Turbo Lora)",
                   "recommended": {"n": 6, "steps": 20, "guidance": 2.5}}, f)
    st, g, _why = U._profile_for_checkpoint(ck)
    assert (st, g) == (20, 2.5), (st, g)
    print("OK test_the_civitai_consensus_wins_over_the_filename")


def test_the_undistilled_profile_comes_from_the_preset():
    """A single source of truth: no 28/3.5 hardcoded on top of config.txt."""
    import cz_ui as U
    name, st, g = U._undistilled_profile()
    assert name in U.PERFORMANCE, name
    assert U.PERFORMANCE[name] == (st, g) or list(U.PERFORMANCE[name]) == [st, g]
    print("OK test_the_undistilled_profile_comes_from_the_preset")


if __name__ == "__main__":
    test_the_base_repo_stays_at_one()
    test_an_override_gets_the_slider()
    test_an_explicit_guidance_is_never_overridden()
    test_the_ignored_guidance_is_announced_once()
    test_a_preset_exists_for_undistilled_checkpoints()
    test_an_undistilled_checkpoint_gets_the_right_profile()
    test_the_civitai_consensus_wins_over_the_filename()
    test_the_undistilled_profile_comes_from_the_preset()
    print("ALL OK")
