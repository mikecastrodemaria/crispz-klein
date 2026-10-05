"""Unit tests for the job queue pure helpers (no Gradio event needed).

Run:  .venv/Scripts/python tests/test_queue.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_ui  # noqa: E402


def _stub_vals(prompt="a cat", use_input=False, w=1024, h=768, steps=8, n=2, seed=42):
    """36-slot stand-in for _gen_inputs values, with the indexed slots filled."""
    vals = [None] * 36
    vals[cz_ui._Q_IDX["prompt"]] = prompt
    vals[cz_ui._Q_IDX["use_input"]] = use_input
    vals[cz_ui._Q_IDX["width"]] = w
    vals[cz_ui._Q_IDX["height"]] = h
    vals[cz_ui._Q_IDX["gen_steps"]] = steps
    vals[cz_ui._Q_IDX["image_number"]] = n
    vals[cz_ui._Q_IDX["seed"]] = seed
    return vals


def test_label():
    ms = {"base_repo": "black-forest-labs/FLUX.2-klein-4B", "transformer": None}
    lbl = cz_ui._q_label(_stub_vals(), ms)
    assert "txt2img" in lbl and "FLUX.2-klein-4B" in lbl and "1024x768" in lbl
    assert "8 steps" in lbl and "seed 42" in lbl and "x2" in lbl and "a cat" in lbl
    # transformer wins over base repo; img2img mode; long prompt truncated
    ms2 = {"base_repo": "x", "transformer": "D:/models/juggernaut_z.safetensors"}
    lbl2 = cz_ui._q_label(_stub_vals(prompt="p" * 80, use_input=True), ms2)
    assert "img2img" in lbl2 and "juggernaut_z.safetensors" in lbl2 and "…" in lbl2


def test_move():
    # _q_move mutates the list IN PLACE (on purpose): _ui_queue_run holds a reference
    # to that state object, so a reordering has to be visible to it. A fresh list per
    # case rather than expecting a pure function.
    def fresh():
        return [{"label": "a"}, {"label": "b"}, {"label": "c"}]

    items = fresh()
    out, sel = cz_ui._q_move(items, 2, -1)
    assert [i["label"] for i in out] == ["a", "c", "b"] and sel == 1
    assert out is items, "it must mutate the shared object, not hand back a copy"

    items = fresh()
    out, sel = cz_ui._q_move(items, 0, -1)          # top edge: unchanged
    assert [i["label"] for i in out] == ["a", "b", "c"] and sel == 0

    items = fresh()
    out, sel = cz_ui._q_move(items, None, 1)         # no selection
    assert sel is None and len(out) == 3
    assert [i["label"] for i in out] == ["a", "b", "c"]


def test_remove():
    items = [{"label": "a"}, {"label": "b"}, {"label": "c"}]
    out, sel = cz_ui._q_remove(items, 1)
    assert [i["label"] for i in out] == ["a", "c"] and sel == 1
    out, sel = cz_ui._q_remove(out, 1)
    assert [i["label"] for i in out] == ["a"] and sel == 0
    out, sel = cz_ui._q_remove(out, 0)
    assert out == [] and sel is None
    out, sel = cz_ui._q_remove([], None)
    assert out == [] and sel is None


def test_render():
    upd, md, btn = cz_ui._q_render([])
    assert "empty" in md and btn["value"] == "+ Queue (0)"
    items = [{"label": "j1"}, {"label": "j2"}]
    upd, md, btn = cz_ui._q_render(items, 1)
    assert "1. j1" in md and "2. j2" in md
    assert btn["value"] == "+ Queue (2)" and upd["value"] == 1
    upd, _, _ = cz_ui._q_render(items, 99)           # selection out of bounds -> None
    assert upd["value"] is None


def test_model_state_roundtrip_keys():
    """The EDIT set is part of the snapshot since the 'Edit LoRA weight' axis.
    Without it, an edit job replayed by the queue picked up the interface's CURRENT edit
    set instead of its own: reproducible in appearance only."""
    ms = cz_ui._q_model_state()
    # 'text_encoder' since 1.34.0: a job replayed by the queue keeps its own encoder.
    assert set(ms) == {"base_repo", "transformer", "loras", "edit_loras",
                       "edit_loras_enabled", "sampler", "schedule", "text_encoder"}


def test_restore_tolerates_a_snapshot_without_the_edit_set():
    """A queue persisted BEFORE that axis has no 'edit_loras' key. Restoring it must
    touch nothing, and above all not empty the current edit set."""
    import cz_pipeline
    seen = []
    old_set, old_en = cz_pipeline.set_edit_loras, cz_pipeline.set_edit_loras_enabled
    old_zm, old_zt = cz_ui.set_zimage_model, cz_ui.set_zimage_transformer
    old_l, old_sa, old_sc = cz_ui.set_loras, cz_ui.set_sampler, cz_ui.set_schedule
    cz_pipeline.set_edit_loras = lambda v: seen.append(v)
    cz_pipeline.set_edit_loras_enabled = lambda v: seen.append(v)
    cz_ui.set_zimage_model = cz_ui.set_zimage_transformer = lambda *_a: None
    cz_ui.set_loras = cz_ui.set_sampler = cz_ui.set_schedule = lambda *_a: None
    try:
        cz_ui._q_restore_model_state({"base_repo": "", "transformer": None,
                                      "loras": [], "sampler": "euler",
                                      "schedule": "sgm_uniform"})
        assert seen == [], seen                       # an old snapshot -> left alone
        cz_ui._q_restore_model_state({"edit_loras": [("/x.safetensors", 0.6)],
                                      "edit_loras_enabled": True})
        assert seen == [[("/x.safetensors", 0.6)], True], seen
    finally:
        cz_pipeline.set_edit_loras, cz_pipeline.set_edit_loras_enabled = old_set, old_en
        cz_ui.set_zimage_model, cz_ui.set_zimage_transformer = old_zm, old_zt
        cz_ui.set_loras, cz_ui.set_sampler, cz_ui.set_schedule = old_l, old_sa, old_sc


# ---------------------------------------------------- pause / stop semantics ---

def _fake_jobs(n):
    return [{"label": f"j{i + 1}", "ms": {}, "vals": _stub_vals()} for i in range(n)]


def _run_with(stub_generate):
    """Runs _ui_queue_run with a stubbed _ui_generate, touching neither the model
    NOR the queue.json on disk (the user's own instance uses it)."""
    import cz_pipeline
    saved = (cz_ui._ui_generate, cz_ui._q_restore_model_state, cz_ui._q_persist,
             cz_pipeline._STOP, cz_ui._QUEUE_PAUSE)
    ran = []
    try:
        cz_ui._ui_generate = stub_generate
        cz_ui._q_restore_model_state = lambda ms: None
        cz_ui._q_persist = lambda items: None
        cz_pipeline._STOP = False
        items = _fake_jobs(3)
        out = cz_ui._ui_queue_run(items, [])
        return items, out
    finally:
        (cz_ui._ui_generate, cz_ui._q_restore_model_state, cz_ui._q_persist,
         cz_pipeline._STOP, cz_ui._QUEUE_PAUSE) = saved


def test_pause_finishes_current_job_then_halts():
    calls = []

    def gen(*vals, progress=None):
        calls.append(1)
        if len(calls) == 1:                       # pause asked for DURING job 1
            cz_ui._QUEUE_PAUSE = True
        return [], "ok", [], []

    items, out = _run_with(gen)
    assert len(calls) == 1, "pause must let the current job FINISH, then halt"
    assert [j["label"] for j in items] == ["j2", "j3"], \
        "the finished job leaves the queue; the rest stays"
    assert "paused" in out[-3].lower()


def test_stop_keeps_the_interrupted_job_queued():
    import cz_pipeline
    calls = []

    def gen(*vals, progress=None):
        calls.append(1)
        if len(calls) == 1:                       # Stop in the middle of job 1
            cz_pipeline._STOP = True
        return [], "interrupted", [], []

    items, out = _run_with(gen)
    assert len(calls) == 1
    assert [j["label"] for j in items] == ["j1", "j2", "j3"], \
        "an interrupted job must STAY at the head of the queue (it did not finish)"
    assert "interrupted job stays queued" in out[-3]


def test_without_pause_or_stop_the_queue_drains():
    def gen(*vals, progress=None):
        return [], "ok", [], []

    items, out = _run_with(gen)
    assert items == [] and "done: 3 job(s)" in out[-3]


def test_request_pause_sets_the_flag_and_reports():
    saved = cz_ui._QUEUE_PAUSE
    try:
        cz_ui._QUEUE_PAUSE = False
        msg = cz_ui._q_request_pause()
        assert cz_ui._QUEUE_PAUSE is True
        assert "Pause requested" in msg
    finally:
        cz_ui._QUEUE_PAUSE = saved



if __name__ == "__main__":
    for fn in (test_label, test_move, test_remove, test_render,
               test_model_state_roundtrip_keys,
               test_restore_tolerates_a_snapshot_without_the_edit_set,
               test_pause_finishes_current_job_then_halts,
               test_stop_keeps_the_interrupted_job_queued,
               test_without_pause_or_stop_the_queue_drains,
               test_request_pause_sets_the_flag_and_reports):
        fn()
        print(f"OK {fn.__name__}")
    print("All queue tests passed.")
