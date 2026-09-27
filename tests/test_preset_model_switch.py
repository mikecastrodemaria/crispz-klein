"""Loading a preset changed everything EXCEPT the model, in silence.

The presets are auto-created per local model. So a klein-9B library on a 4B
install leaves a pile of them naming checkpoints that list_checkpoints
now discards. At Load time, cz_ui pushed that name into the dropdown: the Gradio
frontend refuses a value outside 'choices', the .then chain that applies the model
then re-reads the OLD value, and the next run ran on the previous model.
From the user's side: "I change model, it is still the default model".

No model is loaded here: everything happens on safetensors headers.

Run:  .venv/Scripts/python tests/test_preset_model_switch.py

"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

import cz_core
import cz_pipeline as P
import cz_ui as U

TMP = os.path.join(os.environ.get("TEMP") or "/tmp", "cz_preset_switch")
os.makedirs(TMP, exist_ok=True)


def _stub_dims(fn):
    """Run fn with the 4B/9B hidden dims stubbed, so no network and no gated-repo
    token are needed.

    Seeding _BASE_DIM_CACHE is not enough: set_zimage_model purges the entry of the
    repo it switches TO, on purpose (choosing a repo counts as "retry" once its
    licence is accepted). The dim was therefore re-read from transformer/config.json
    on the Hub, and the gated 9B answers 401 to anyone who has not accepted it - the
    filter then turns OFF and the test failed on the CI runner."""
    real = P._base_hidden_dim
    dims = {U.KLEIN_BASE_4B: 3072, U.KLEIN_BASE_9B: 4096}

    def fake(base=None):
        return dims.get((base or P.BASE_REPO or "").strip())
    P._base_hidden_dim = fake
    try:
        return fn()
    finally:
        P._base_hidden_dim = real


def _stub_dims(fn):
    """Run fn with the 4B/9B hidden dims stubbed, so no network and no gated-repo
    token are needed.

    Seeding _BASE_DIM_CACHE is not enough: set_zimage_model purges the entry of the
    repo it switches TO, on purpose (choosing a repo counts as "retry" once its
    licence is accepted). The dim was therefore re-read from transformer/config.json
    on the Hub, and the gated 9B answers 401 to anyone who has not accepted it - the
    filter then turns OFF and the test failed on the CI runner."""
    real = P._base_hidden_dim
    dims = {U.KLEIN_BASE_4B: 3072, U.KLEIN_BASE_9B: 4096}

    def fake(base=None):
        return dims.get((base or P.BASE_REPO or "").strip())
    P._base_hidden_dim = fake
    try:
        return fn()
    finally:
        P._base_hidden_dim = real


def _ckpt(name, dim):
    """A fake FLUX.2 transformer: only the header is read, the weights are empty."""
    p = os.path.join(TMP, name)
    save_file({"double_stream_modulation_img.lin.weight": torch.zeros(dim * 6, dim),
               "x_embedder.weight": torch.zeros(2, 2)}, p)
    return p


def _with_lib(fn, base_dim=3072):
    """Runs fn with TMP as the checkpoints AND presets folder, on a 4B base
    repo. The install's presets/ folder must see nothing of these fixtures:
    _apply_checkpoint and _refresh_checkpoints create presets per local model."""
    old = (P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR, P.BASE_REPO,
           P.ZIMAGE_TRANSFORMER, dict(P._BASE_DIM_CACHE), U._PRESETS_DIR)
    P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR = TMP, ""
    P.BASE_REPO = "base-4b"
    P._BASE_DIM_CACHE["base-4b"] = base_dim
    U._PRESETS_DIR = os.path.join(TMP, "presets")
    # Starting fresh: TMP survives from one run to the next, and _ensure_model_presets
    # NEVER touches an existing preset -- a file left by a previous run
    # would make a test pass that no longer tests anything.
    shutil.rmtree(U._PRESETS_DIR, ignore_errors=True)
    os.makedirs(U._PRESETS_DIR, exist_ok=True)
    try:
        return fn()
    finally:
        (P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR, P.BASE_REPO,
         P.ZIMAGE_TRANSFORMER, cache, U._PRESETS_DIR) = old
        P._BASE_DIM_CACHE.clear()
        P._BASE_DIM_CACHE.update(cache)


def test_refusal_names_the_reason_and_the_fix():
    _ckpt("good4b.safetensors", 3072)
    _ckpt("big9b.safetensors", 4096)

    def check():
        assert P.checkpoint_refusal("good4b.safetensors") is None
        why = P.checkpoint_refusal("big9b.safetensors")
        assert why and "9B" in why and "4B" in why, why
        # a refusal named file by file must carry the COMPLETE instructions:
        # unlike the listing, there is no summary behind it to say them.
        assert P.CFG_MODEL_KEY in why, why
        assert "NON-COMMERCIAL" in why, why
        # the refusal must name the EXACT repo to choose and the place to choose it:
        # pointing at "the matching repo" and a config file, when a
        # dropdown does the job, is pointing at the wrong place.
        assert "black-forest-labs/FLUX.2-klein-9B" in why, why
        assert "Klein checkpoint" in why, why
        assert "GATED" in why and "licence" in why, why
        # an HF repo / diffusers folder does not go through that filter
        assert P.checkpoint_refusal("black-forest-labs/FLUX.2-klein-4B") is None
        # a name that no longer exists on the disk must be said, not kept quiet
        gone = P.checkpoint_refusal("moved-away.safetensors")
        assert gone and "no such file" in gone, gone
    _with_lib(check)
    print("OK test_refusal_names_the_reason_and_the_fix")


def test_apply_checkpoint_refuses_the_wrong_variant():
    """Without the guard, a 9B became the current transformer and died on the next run,
    after reading gigabytes, on an unreadable diffusers message."""
    _ckpt("good4b.safetensors", 3072)
    _ckpt("big9b.safetensors", 4096)

    def check():
        P.ZIMAGE_TRANSFORMER = None
        msg = U._apply_checkpoint("good4b.safetensors")[0]
        assert P.ZIMAGE_TRANSFORMER and "good4b" in P.ZIMAGE_TRANSFORMER, msg
        assert "Klein transformer" in msg, msg
        assert "Qwen" not in msg, f"nom du fork amont: {msg}"

        kept = P.ZIMAGE_TRANSFORMER
        msg = U._apply_checkpoint("big9b.safetensors")[0]
        assert P.ZIMAGE_TRANSFORMER == kept, "un 9B a ete applique sur un install 4B"
        assert "NOT applied" in msg and "9B" in msg, msg
        assert "good4b" in msg, f"le message doit dire ce qui reste en place: {msg}"
    _with_lib(check)
    print("OK test_apply_checkpoint_refuses_the_wrong_variant")


def _preset(name, ckpt, steps=7, base=None):
    os.makedirs(U._PRESETS_DIR, exist_ok=True)
    p = os.path.join(U._PRESETS_DIR, name + ".json")
    d = {"prompt": "a lighthouse", "steps": steps, "checkpoint": ckpt,
         "transformer": "", "loras": []}
    if base:
        d["base_repo"] = base
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f)
    return p


def _ckpt_update(outs):
    """The checkpoint dropdown's gr.update in _ui_preset_load's output."""
    return outs[U._PRESET_KEYS.index("checkpoint")]


def test_preset_never_pushes_a_checkpoint_the_dropdown_refuses():
    _ckpt("good4b.safetensors", 3072)
    _ckpt("big9b.safetensors", 4096)

    def check():
        _preset("zz-test-ok", "good4b.safetensors")
        _preset("zz-test-bad", "big9b.safetensors")
        P.ZIMAGE_TRANSFORMER = P.resolve_checkpoint("good4b.safetensors")

        outs = U._ui_preset_load("zz-test-ok")
        assert _ckpt_update(outs).get("value") == "good4b.safetensors"
        assert "NOT switched" not in outs[-1], outs[-1]
        # the rest of the preset applies in both cases
        assert outs[U._PRESET_KEYS.index("steps")].get("value") == 7

        outs = U._ui_preset_load("zz-test-bad")
        # nothing must be pushed: a value outside 'choices' is rejected by the
        # frontend, and it is that silent rejection that left the model in place.
        assert "value" not in _ckpt_update(outs), _ckpt_update(outs)
        status = outs[-1]
        assert "NOT switched" in status and "big9b.safetensors" in status, status
        assert "9B" in status, status
        assert "good4b" in status, f"doit dire quel modele reste actif: {status}"
        assert outs[U._PRESET_KEYS.index("steps")].get("value") == 7, \
            "le reste du preset doit s'appliquer quand meme"
    _with_lib(check)
    print("OK test_preset_never_pushes_a_checkpoint_the_dropdown_refuses")



def test_a_preset_switches_its_own_base_repo():
    """A single-file only swaps the transformer: it only makes sense under the base
    that supplies the VAE/encoder/config. A preset that knows its base restores it."""
    _ckpt("good4b.safetensors", 3072)
    _ckpt("big9b.safetensors", 4096)
    saved = {}

    def check():
        real_save, U._save_prefs_keys = U._save_prefs_keys, saved.update
        P._BASE_DIM_CACHE[U.KLEIN_BASE_4B] = 3072
        P._BASE_DIM_CACHE[U.KLEIN_BASE_9B] = 4096
        try:
            _preset("zz-9b", "big9b.safetensors", base=U.KLEIN_BASE_9B)
            P.BASE_REPO = U.KLEIN_BASE_4B
            P.ZIMAGE_TRANSFORMER = P.resolve_checkpoint("good4b.safetensors")

            outs = U._ui_preset_load("zz-9b")
            assert P.BASE_REPO == U.KLEIN_BASE_9B, P.BASE_REPO
            upd, status = _ckpt_update(outs), outs[-1]
            # the base has changed -> the 9B checkpoint is now valid AND offered
            assert upd.get("value") == "big9b.safetensors", upd
            assert "big9b.safetensors" in upd.get("choices"), upd
            assert "NOT switched" not in status, status
            assert "Base model switched" in status, status
            # the cost must be announced, not suffered
            assert "next **Generate**" in status and "released" in status, status
            assert "Non-Commercial" in status, "la note du 9B doit suivre le swap"
            assert saved.get(P.CFG_MODEL_KEY) == U.KLEIN_BASE_9B, saved
        finally:
            U._save_prefs_keys = real_save
    _with_lib(lambda: _stub_dims(check))
    print("OK test_a_preset_switches_its_own_base_repo")


def test_the_refusal_leads_with_the_action():
    """Someone who has just clicked Load is looking for the action, not the diagnosis."""
    _ckpt("good4b.safetensors", 3072)
    _ckpt("big9b.safetensors", 4096)

    def check():
        P._BASE_DIM_CACHE[U.KLEIN_BASE_4B] = 3072
        P.BASE_REPO = U.KLEIN_BASE_4B
        P.ZIMAGE_TRANSFORMER = None
        # a preset WITHOUT base_repo (written before the base was tracked): we cannot
        # guess its base, but we know which variant that file wants.
        _preset("zz-old", "big9b.safetensors")
        status = U._ui_preset_load("zz-old")[-1]
        head = status.split("Why:")[0]
        assert "Klein checkpoint" in head, f"l'action doit venir en premier: {head}"
        assert U.KLEIN_BASE_9B in head, head
        assert "Why:" in status, "le diagnostic doit suivre, pas preceder"
        # and the lasting repair of the old preset
        assert "Update selected" in status, status
    _with_lib(check)
    print("OK test_the_refusal_leads_with_the_action")


def test_saving_records_the_base_repo():
    """Without that, every preset created today reproduces tomorrow's bug."""
    _ckpt("good4b.safetensors", 3072)

    def check():
        P._BASE_DIM_CACHE[U.KLEIN_BASE_4B] = 3072
        P.BASE_REPO = U.KLEIN_BASE_4B
        vals = [""] * len(U._PRESET_KEYS)
        vals[U._PRESET_KEYS.index("checkpoint")] = "good4b.safetensors"
        U._ui_preset_save("zz-saved", *(vals + ["None"] * U.MAX_LORA_SLOTS
                                        + [1.0] * U.MAX_LORA_SLOTS))
        assert U._load_preset_file("zz-saved").get("base_repo") == U.KLEIN_BASE_4B
        # and an auto-created preset as well
        U._ensure_model_presets(["good4b.safetensors"])
        assert U._load_preset_file("good4b").get("base_repo") == U.KLEIN_BASE_4B
    _with_lib(check)
    print("OK test_saving_records_the_base_repo")


def test_both_base_repos_are_selectable_and_the_9b_is_announced():
    """The 4B and the 9B are chosen from the dropdown. The 4B stays the default; the 9B must
    say what it costs BEFORE the run: a non-commercial licence, a gated repo, the VRAM."""
    assert U.KLEIN_BASE_4B in U.ZIMAGE_BASE_REPOS
    assert U.KLEIN_BASE_9B in U.ZIMAGE_BASE_REPOS
    assert P.DEFAULT_BASE_REPO == U.KLEIN_BASE_4B, "le defaut doit rester le 4B"
    note = U.BASE_REPO_NOTES[U.KLEIN_BASE_9B]
    assert "Non-Commercial" in note and "gated" in note and "VRAM" in note, note
    print("OK test_both_base_repos_are_selectable_and_the_9b_is_announced")


def test_base_swap_refreshes_the_checkpoint_list():
    """The 4B and the 9B do not accept the same files: choosing a base repo must
    rebuild the list, otherwise the dropdown offers impossible models."""
    _ckpt("good4b.safetensors", 3072)
    _ckpt("big9b.safetensors", 4096)
    saved = {}

    def check():
        # no writing into the real preferences.json during a test
        real_save, U._save_prefs_keys = U._save_prefs_keys, saved.update
        # both repos' dimension is pre-seeded: no network access here
        P._BASE_DIM_CACHE[U.KLEIN_BASE_4B] = 3072
        P._BASE_DIM_CACHE[U.KLEIN_BASE_9B] = 4096
        try:
            P.BASE_REPO = U.KLEIN_BASE_4B
            P.ZIMAGE_TRANSFORMER = P.resolve_checkpoint("good4b.safetensors")
            out = U._apply_checkpoint(U.KLEIN_BASE_9B)
            assert P.BASE_REPO == U.KLEIN_BASE_9B, P.BASE_REPO
            assert not P.ZIMAGE_TRANSFORMER, \
                "un repo de base complet doit effacer l'override single-file"
            status, choices = out[0], out[4].get("choices")
            assert "Non-Commercial" in status and "gated" in status, status
            assert choices and "big9b.safetensors" in choices, choices
            assert "good4b.safetensors" not in choices, \
                f"un checkpoint 4B ne charge pas dans un pipeline 9B: {choices}"
            assert saved.get(P.CFG_MODEL_KEY) == U.KLEIN_BASE_9B, saved

            # and back to the 4B: the list must flip round
            out = U._apply_checkpoint(U.KLEIN_BASE_4B)
            choices = out[4].get("choices")
            assert "good4b.safetensors" in choices and "big9b.safetensors" not in choices, choices
            assert "Non-Commercial" not in out[0], out[0]
        finally:
            U._save_prefs_keys = real_save
    _with_lib(lambda: _stub_dims(check))
    print("OK test_base_swap_refreshes_the_checkpoint_list")


def test_choosing_a_base_repo_retries_its_dimension():
    """A repo's dimension is cached, failures included. A gated repo refused before
    the licence is accepted left a None stuck there for the whole session: the
    4B/9B filter off even once the licence was accepted. Choosing it counts as "retry"."""
    def check():
        P._BASE_DIM_CACHE[U.KLEIN_BASE_9B] = None     # a previous failure (403 gated)
        P.BASE_REPO = U.KLEIN_BASE_4B
        P.set_zimage_model(U.KLEIN_BASE_9B)
        assert U.KLEIN_BASE_9B not in P._BASE_DIM_CACHE,             "l'echec cache doit etre purge quand on rechoisit le repo"
    _with_lib(check)
    print("OK test_choosing_a_base_repo_retries_its_dimension")



def test_offload_is_forced_when_the_base_cannot_fit():
    """The 9B set whole on a 32 GB card loaded, then died on the first diffusion
    step on 'CUDA error: unknown error' -- which does not even name the VRAM.
    A config that CANNOT work is corrected beforehand, not after 5 minutes."""
    real_dev, real_total = P.DEVICE, P._total_vram_gb
    real_off = P.OFFLOAD_MODE
    try:
        P.DEVICE = "cuda"
        P.OFFLOAD_MODE = "none"
        P.ZIMAGE_TRANSFORMER = None
        P._BASE_DIM_CACHE["fits"] = 3072       # 4B -> 15 Go
        P._BASE_DIM_CACHE["huge"] = 4096       # 9B -> 35 Go
        P._total_vram_gb = lambda: 31.8        # RTX 5090

        P.BASE_REPO = "fits"
        assert P._effective_offload() == "none", "le 4B tient: on ne touche a rien"
        P.BASE_REPO = "huge"
        assert P._effective_offload() == "model", "le 9B ne tient pas: offload force"

        # A single-file override does NOT make the model smaller: an FP8/INT8 is
        # dequantised to bf16 and weighs as much as the original transformer. The
        # first version of the guard skipped that case -> a crash on the first diffusion
        # step, after five minutes of dequantisation.
        P.BASE_REPO = "huge"
        P.ZIMAGE_TRANSFORMER = _ckpt("fp8_9b.safetensors", 4096)
        assert P._effective_offload() == "model", "un FP8 9B ne tient pas plus qu'un bf16"
        P.ZIMAGE_TRANSFORMER = None

        # a card big enough -> no correction at all
        P._total_vram_gb = lambda: 80.0
        assert P._effective_offload() == "none"
        # an unknown variant -> we meddle with nothing (the house rule: no acting on a doubt)
        P._total_vram_gb = lambda: 31.8
        P._BASE_DIM_CACHE["mystere"] = None
        P.BASE_REPO = "mystere"
        assert P._effective_offload() == "none"
    finally:
        P.DEVICE, P._total_vram_gb, P.OFFLOAD_MODE = real_dev, real_total, real_off
        for k in ("fits", "huge", "mystere"):
            P._BASE_DIM_CACHE.pop(k, None)
    print("OK test_offload_is_forced_when_the_base_cannot_fit")


def test_gated_repo_error_says_what_to_do():
    """A 401/403 from the Hub on the 9B must become an instruction, not a trace."""
    hint = P._hf_access_hint(U.KLEIN_BASE_9B,
                             RuntimeError("401 Client Error: Access to model ... is restricted"))
    assert hint and U.KLEIN_BASE_9B in hint, hint
    assert "accept its licence" in hint and "token" in hint, hint
    # a token set by 'huggingface-cli login' counts: telling someone who has one
    # that there is "no token" sends them looking for the wrong cause.
    real, cz_core.hf_token_is_set = cz_core.hf_token_is_set, lambda: True
    try:
        with_tok = P._hf_access_hint(U.KLEIN_BASE_9B, RuntimeError("403 gated"))
    finally:
        cz_core.hf_token_is_set = real
    assert "licence itself" in with_tok, with_tok
    # an ordinary failure must NOT be dressed up as a licence problem
    assert P._hf_access_hint(U.KLEIN_BASE_9B, OSError("disk full")) is None
    print("OK test_gated_repo_error_says_what_to_do")


if __name__ == "__main__":
    test_refusal_names_the_reason_and_the_fix()
    test_apply_checkpoint_refuses_the_wrong_variant()
    test_preset_never_pushes_a_checkpoint_the_dropdown_refuses()
    test_a_preset_switches_its_own_base_repo()
    test_the_refusal_leads_with_the_action()
    test_saving_records_the_base_repo()
    test_both_base_repos_are_selectable_and_the_9b_is_announced()
    test_base_swap_refreshes_the_checkpoint_list()
    test_choosing_a_base_repo_retries_its_dimension()
    test_offload_is_forced_when_the_base_cannot_fit()
    test_gated_repo_error_says_what_to_do()
    print("ALL OK")
