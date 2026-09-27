"""A replacement text encoder (Models > Checkpoints > Text encoder).

FLUX.2 reads three INTERMEDIATE hidden states of the Qwen3 encoder in a
context_embedder 3 x hidden wide: another encoder only plugs in when it has the
same family, the same width and the same number of layers. A Qwen3-4B "abliterated"
suits the 4B; the same one on the 9B (4096 wide) cannot work, and the refusal must
say so BEFORE reading 8 GB.

These tests also lock down what would make the option silently dangerous:
  - a change of encoder empties the embeddings cache (otherwise the old encodings
    stay served) and the encoder is part of the cache KEY -- id(enc) alone is not
    enough, CPython recycles the ids of freed objects;
  - the metadata names the encoder that REALLY ran, by its folder name
    and never by its path (which would end up in the shared PNGs);
  - the queue keeps the job's encoder.

Run:  .venv/Scripts/python tests/test_text_encoder.py

"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import cz_imageio
import cz_pipeline as P

QWEN4B = {"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36,
          "architectures": ["Qwen3ForCausalLM"]}
QWEN8B = {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36,
          "architectures": ["Qwen3ForCausalLM"]}


def _folder(cfg, sub=None, name="enc"):
    root = tempfile.mkdtemp(prefix="te_")
    d = os.path.join(root, name)
    p = os.path.join(d, sub) if sub else d
    os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return d


class _Base:
    """Replaces the encoder config of the base repo (no network, no HF)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __enter__(self):
        self.old = P._base_text_encoder_config
        P._base_text_encoder_config = lambda base=None: self.cfg

    def __exit__(self, *a):
        P._base_text_encoder_config = self.old


def test_same_architecture_is_accepted():
    with _Base(QWEN4B):
        assert P._text_encoder_problem(_folder(QWEN4B)) is None
        # weights in a text_encoder/ subfolder (a copy of a diffusers repo)
        assert P._text_encoder_problem(_folder(QWEN4B, "text_encoder")) is None
    print("OK test_same_architecture_is_accepted")


def test_a_4b_encoder_is_refused_on_the_9b_by_name():
    with _Base(QWEN8B):
        why = P._text_encoder_problem(_folder(QWEN4B))
    assert why and "2560" in why and "4096" in why and "4B" in why, why
    print("OK test_a_4b_encoder_is_refused_on_the_9b_by_name")


def test_other_family_and_layer_count_are_refused():
    with _Base(QWEN4B):
        why = P._text_encoder_problem(_folder({"model_type": "t5", "d_model": 2560,
                                               "num_layers": 36}))
        assert why and "t5" in why, why
        why = P._text_encoder_problem(_folder({**QWEN4B, "num_hidden_layers": 28}))
        assert why and "28" in why and "36" in why, why
    print("OK test_other_family_and_layer_count_are_refused")


def test_gguf_single_file_and_empty_folder_are_refused_with_the_reason():
    with _Base(QWEN4B):
        assert "GGUF" in P._text_encoder_problem(r"F:\x\qwen3-4b-q8_0.gguf")
        assert "FOLDER" in P._text_encoder_problem(r"F:\x\qwen3_4b.safetensors")
        assert "config.json" in P._text_encoder_problem(tempfile.mkdtemp())
    print("OK test_gguf_single_file_and_empty_folder_are_refused_with_the_reason")


def test_hf_ids_may_carry_a_subfolder():
    assert P._split_hf_src("owner/repo") == ("owner/repo", None)
    assert P._split_hf_src("ponpoke/flux2-klein-4b-uncensored-text-encoder/"
                           "flux2-klein-4b-uncensored-text-encoder") == (
        "ponpoke/flux2-klein-4b-uncensored-text-encoder",
        "flux2-klein-4b-uncensored-text-encoder")
    print("OK test_hf_ids_may_carry_a_subfolder")


def test_the_class_comes_from_the_base_repo_model_index():
    base = tempfile.mkdtemp(prefix="base_")
    with open(os.path.join(base, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump({"text_encoder": ["transformers", "Qwen3ForCausalLM"]}, f)
    cls = P._encoder_class(base)
    assert cls.__name__ == "Qwen3ForCausalLM", cls
    print("OK test_the_class_comes_from_the_base_repo_model_index")


def test_changing_the_encoder_frees_the_pipe_and_the_cache():
    old = (P.TEXT_ENCODER, P._BASE_PIPE)
    try:
        P.TEXT_ENCODER = ""
        P._BASE_PIPE = object()
        P._EMBED_CACHE[("k",)] = ("v",)
        P.set_text_encoder(r"D:\enc\qwen3-abl")
        assert P.TEXT_ENCODER == r"D:\enc\qwen3-abl"
        assert P._BASE_PIPE is None, "le pipeline doit etre libere"
        assert not P._EMBED_CACHE, "les anciens encodages resteraient servis"
        # the same value: nothing moves, no pointless reload
        sentinel = P._BASE_PIPE = object()
        P.set_text_encoder(r"D:\enc\qwen3-abl")
        assert P._BASE_PIPE is sentinel
    finally:
        P.TEXT_ENCODER, P._BASE_PIPE = old
        P._EMBED_CACHE.clear()
    print("OK test_changing_the_encoder_frees_the_pipe_and_the_cache")


class FakePipe:
    def __init__(self):
        self.text_encoder = object()
        self._execution_device = "cpu"
        self.n = 0

    def encode_prompt(self, prompt, device=None):
        self.n += 1
        return (torch.zeros(1, 2, 4),)


def test_the_embed_key_carries_the_encoder():
    """The same prompt, the same pipe object, two encoders: two encodings."""
    P._embed_cache_clear()
    old = P._TEXT_ENCODER_ACTIVE
    try:
        pipe = FakePipe()
        P._TEXT_ENCODER_ACTIVE = ""
        P._cached_prompt_embeds(pipe, "p", {})
        P._cached_prompt_embeds(pipe, "p", {})
        assert pipe.n == 1, pipe.n
        P._TEXT_ENCODER_ACTIVE = r"D:\enc\qwen3-abl"
        P._cached_prompt_embeds(pipe, "p", {})
        assert pipe.n == 2, "un encodage de l'autre encodeur a ete resservi"
    finally:
        P._TEXT_ENCODER_ACTIVE = old
        P._embed_cache_clear()
    print("OK test_the_embed_key_carries_the_encoder")


def test_metadata_names_the_encoder_that_ran_and_never_its_path():
    old = (P.TEXT_ENCODER, P._TEXT_ENCODER_ACTIVE)
    path = r"C:\Users\someone\models\text_encoders\qwen3-4b-abliterated"
    try:
        P.TEXT_ENCODER = P._TEXT_ENCODER_ACTIVE = path
        m = P._gen_meta("txt2img", "p")
        assert m["text_encoder"] == "qwen3-4b-abliterated", m
        assert "someone" not in json.dumps(m), "chemin local dans les metadonnees"
        # asked for but discarded at load time: named apart
        P._TEXT_ENCODER_ACTIVE = ""
        m = P._gen_meta("txt2img", "p")
        assert "text_encoder" not in m and m["text_encoder_not_applied"] == "qwen3-4b-abliterated", m
        P.TEXT_ENCODER = ""
        m = P._gen_meta("txt2img", "p")
        assert "text_encoder" not in m and "text_encoder_not_applied" not in m, m
    finally:
        P.TEXT_ENCODER, P._TEXT_ENCODER_ACTIVE = old
    assert P._encoder_label(r"D:\m\ponpoke-uncensored\text_encoder") == "ponpoke-uncensored"
    assert P._encoder_label("owner/repo/sub") == "owner/repo/sub"
    line = cz_imageio._a1111_parameters({"prompt": "p", "text_encoder": "qwen3-4b-abliterated"})
    assert "Text encoder: qwen3-4b-abliterated" in line, line
    print("OK test_metadata_names_the_encoder_that_ran_and_never_its_path")


def test_the_list_finds_encoder_folders():
    d = _folder(QWEN4B, name="qwen3-4b-abliterated")
    root = os.path.dirname(d)
    os.makedirs(os.path.join(root, "empty"))
    old = P.TEXT_ENCODERS_DIR
    try:
        P.TEXT_ENCODERS_DIR = root
        found = P.list_text_encoders()
    finally:
        P.TEXT_ENCODERS_DIR = old
    assert d in found, found
    assert not any(f.endswith("empty") for f in found), found
    print("OK test_the_list_finds_encoder_folders")


def test_the_queue_keeps_the_encoder():
    import cz_ui as U
    calls = []
    old = (P.TEXT_ENCODER, P.set_text_encoder)
    try:
        P.TEXT_ENCODER = r"D:\enc\qwen3-abl"
        ms = U._q_model_state()
        assert ms["text_encoder"] == r"D:\enc\qwen3-abl", ms
        P.set_text_encoder = lambda s: calls.append(s)
        U._q_restore_model_state(ms)
        assert calls == [r"D:\enc\qwen3-abl"], calls
        # a snapshot from before the option: we do not touch the current encoder
        calls.clear()
        U._q_restore_model_state({k: v for k, v in ms.items() if k != "text_encoder"})
        assert calls == [], calls
    finally:
        P.TEXT_ENCODER, P.set_text_encoder = old
    print("OK test_the_queue_keeps_the_encoder")


def test_default_picked_in_the_ui_survives_a_restart():
    """Choosing "Default" writes "" into the preferences: on a restart, a value from
    config.txt must not come back over it. The environment always wins."""
    cfg = {"text_encoder": r"D:\enc\from-config"}
    assert P._resolve_text_encoder({}, {}, cfg) == r"D:\enc\from-config"
    assert P._resolve_text_encoder({}, {"text_encoder": ""}, cfg) == ""
    assert P._resolve_text_encoder({}, {"text_encoder": r"D:\enc\ui"}, cfg) == r"D:\enc\ui"
    assert P._resolve_text_encoder({"KLEIN_TEXT_ENCODER": r"D:\enc\env"},
                                   {"text_encoder": ""}, cfg) == r"D:\enc\env"
    print("OK test_default_picked_in_the_ui_survives_a_restart")


def test_compatible_encoders_in_the_hf_cache_are_listed():
    """An encoder downloaded from HF lives in the HF cache: the list must show it.
    Not a diffusers pipeline, not another width, not a config with no weights."""
    root = tempfile.mkdtemp(prefix="hfcache_")

    def snap(repo, sub=None, cfg=QWEN4B, weights=True, pipeline=False):
        d = os.path.join(root, "models--" + repo.replace("/", "--"), "snapshots", "r1")
        p = os.path.join(d, sub) if sub else d
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        if weights:
            open(os.path.join(p, "model.safetensors"), "wb").close()
        if pipeline:
            with open(os.path.join(d, "model_index.json"), "w", encoding="utf-8") as f:
                f.write("{}")

    snap("huihui-ai/Huihui-Qwen3-4B-abliterated-v2")
    snap("ponpoke/flux2-klein-4b-uncensored-text-encoder", sub="flux2-klein-4b-uncensored-text-encoder")
    snap("Qwen/Qwen3-8B", cfg=QWEN8B)                                    # another width
    snap("Tongyi-MAI/Z-Image-Turbo", sub="text_encoder", pipeline=True)  # pipeline diffusers
    snap("someone/config-only", weights=False)                           # the weights are absent
    old = (P._hf_cache_dir, P._base_text_encoder_config)
    try:
        P._hf_cache_dir = lambda: root
        P._base_text_encoder_config = lambda base=None: QWEN4B
        got = [v for _lab, v in P.list_cached_text_encoders()]
    finally:
        P._hf_cache_dir, P._base_text_encoder_config = old
    assert got == ["huihui-ai/Huihui-Qwen3-4B-abliterated-v2",
                   "ponpoke/flux2-klein-4b-uncensored-text-encoder/"
                   "flux2-klein-4b-uncensored-text-encoder"], got
    # with a 9B base repo: both Qwen3-4B leave the list and are NAMED next to it
    old = (P._hf_cache_dir, P._base_text_encoder_config)
    try:
        P._hf_cache_dir = lambda: root
        P._base_text_encoder_config = lambda base=None: QWEN8B
        # the fake cache's Qwen3-8B suits the 9B: it is the one, and the only one, offered
        assert [v for _l, v in P.list_cached_text_encoders()] == ["Qwen/Qwen3-8B"]
        other, width = P.cached_text_encoder_mismatches()
        import cz_ui as U
        hint = U._te_hint()
    finally:
        P._hf_cache_dir, P._base_text_encoder_config = old
    assert width == 4096 and sorted(h for h, _w in other) == sorted(got), other
    assert "2560" in hint and "4096" in hint and "huihui" in hint, hint
    print("OK test_compatible_encoders_in_the_hf_cache_are_listed")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All text-encoder tests passed.")
